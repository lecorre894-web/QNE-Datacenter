#define main qne_original_main_unused
#include "native.cpp"
#undef main

#include <node_api.h>
#include <barrier>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <algorithm>

// Quatre GF128 dans un vecteur SIMD de 512 bits.
// Pas de banque intermediaire.

__attribute__((target("avx512f,avx512bw,vpclmulqdq")))
static inline void vector4(
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* out,
    bool stream
) {
    __m512i A=_mm512_loadu_si512((const void*)a);
    __m512i Bv=_mm512_loadu_si512((const void*)b);

    __m512i z0=_mm512_clmulepi64_epi128(A,Bv,0);
    __m512i z2=_mm512_clmulepi64_epi128(A,Bv,0x11);

    __m512i z1=_mm512_xor_si512(
        _mm512_clmulepi64_epi128(
            _mm512_xor_si512(
                A,_mm512_bsrli_epi128(A,8)
            ),
            _mm512_xor_si512(
                Bv,_mm512_bsrli_epi128(Bv,8)
            ),
            0
        ),
        _mm512_xor_si512(z0,z2)
    );

    __m512i L=_mm512_xor_si512(
        z0,_mm512_bslli_epi128(z1,8)
    );

    __m512i H=_mm512_xor_si512(
        z2,_mm512_bsrli_epi128(z1,8)
    );

    __m512i hi=_mm512_bsrli_epi128(H,8);

    __m512i spill=_mm512_xor_si512(
        _mm512_srli_epi64(hi,63),
        _mm512_xor_si512(
            _mm512_srli_epi64(hi,62),
            _mm512_srli_epi64(hi,57)
        )
    );

    __m512i R=_mm512_xor_si512(L,H);

    R=_mm512_xor_si512(R,shl128<1>(H));
    R=_mm512_xor_si512(R,shl128<2>(H));
    R=_mm512_xor_si512(R,shl128<7>(H));

    R=_mm512_xor_si512(R,spill);
    R=_mm512_xor_si512(
        R,_mm512_slli_epi64(spill,1)
    );
    R=_mm512_xor_si512(
        R,_mm512_slli_epi64(spill,2)
    );
    R=_mm512_xor_si512(
        R,_mm512_slli_epi64(spill,7)
    );

    if(stream)
        _mm512_stream_si512((__m512i*)out,R);
    else
        _mm512_storeu_si512((void*)out,R);
}

__attribute__((target("avx512f,avx512bw,vpclmulqdq")))
static void runRange(
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* out,
    size_t lo,
    size_t hi,
    bool stream,
    unsigned prefetch
) {
    size_t i=lo;

    // Quatre groupes de quatre elements.
    // Les produits sont independants.

    for(;i+16<=hi;i+=16) {
        if(prefetch && i+prefetch<hi) {
            __builtin_prefetch(
                a+16*(i+prefetch),0,1
            );
            __builtin_prefetch(
                b+16*(i+prefetch),0,1
            );
        }

        vector4(
            a+16*i,b+16*i,
            out+16*i,stream
        );

        vector4(
            a+16*(i+4),b+16*(i+4),
            out+16*(i+4),stream
        );

        vector4(
            a+16*(i+8),b+16*(i+8),
            out+16*(i+8),stream
        );

        vector4(
            a+16*(i+12),b+16*(i+12),
            out+16*(i+12),stream
        );
    }

    for(;i+4<=hi;i+=4)
        vector4(
            a+16*i,b+16*i,
            out+16*i,stream
        );

    for(;i<hi;i++) {
        B aa,bb;

        std::memcpy(&aa,a+16*i,16);
        std::memcpy(&bb,b+16*i,16);

        B rr=pclmul(aa,bb);

        std::memcpy(out+16*i,&rr,16);
    }

    if(stream)
        _mm_sfence();
}

// Pool de threads permanents.

class Pool {
    std::barrier<> ready,done;
    std::vector<std::thread> threads;
    std::function<void(int)> job;
    bool stop=false;
    std::atomic<int> pinned{0};

public:
    Pool(int n,const std::string& mode):
        ready(n+1),done(n+1) {

        cpu_set_t allowed;
        CPU_ZERO(&allowed);

        std::vector<int> cpus;

        if(sched_getaffinity(
            0,sizeof(allowed),&allowed
        )==0) {
            for(int j=0;j<CPU_SETSIZE;j++)
                if(CPU_ISSET(j,&allowed))
                    cpus.push_back(j);
        }

        for(int i=0;i<n;i++) {
            threads.emplace_back(
                [this,i,n,mode,cpus] {

                    if(mode!="none" &&
                       cpus.size()>=(size_t)n) {

                        size_t j=mode=="compact"
                            ?i
                            :(size_t)i*cpus.size()/n;

                        cpu_set_t mask;
                        CPU_ZERO(&mask);
                        CPU_SET(cpus[j],&mask);

                        if(pthread_setaffinity_np(
                            pthread_self(),
                            sizeof(mask),
                            &mask
                        )==0)
                            pinned.fetch_add(1);
                    }

                    for(;;) {
                        ready.arrive_and_wait();

                        if(stop)
                            break;

                        job(i);
                        done.arrive_and_wait();
                    }
                }
            );
        }
    }

    void execute(std::function<void(int)> f) {
        job=std::move(f);
        ready.arrive_and_wait();
        done.arrive_and_wait();
    }

    int pinnedCount() const {
        return pinned.load();
    }

    ~Pool() {
        stop=true;
        ready.arrive_and_wait();

        for(auto& t:threads)
            t.join();
    }
};

static std::unique_ptr<Pool> pool;
static std::mutex poolMutex;
static int poolSize=0;
static std::string poolPolicy;
static int lastPinned=0;
static bool lastStream=false;

static napi_value fail(
    napi_env e,const char* s
) {
    napi_throw_error(e,nullptr,s);
    return nullptr;
}

static bool buffer(
    napi_env e,napi_value v,
    unsigned char*& p,size_t& n
) {
    bool ok=false;
    void* ptr=nullptr;

    if(napi_is_buffer(e,v,&ok)!=napi_ok ||
       !ok ||
       napi_get_buffer_info(e,v,&ptr,&n)!=napi_ok) {
        napi_throw_type_error(
            e,nullptr,"Buffer requis"
        );
        return false;
    }

    p=(unsigned char*)ptr;
    return true;
}

static bool string(
    napi_env e,napi_value v,
    std::string& s
) {
    char tmp[32];
    size_t n=0;

    if(napi_get_value_string_utf8(
        e,v,tmp,sizeof(tmp),&n
    )!=napi_ok)
        return false;

    s.assign(tmp,n);
    return true;
}

static napi_value mul(
    napi_env e,napi_callback_info info
) {
    napi_value v[7];
    size_t argc=7;

    if(napi_get_cb_info(
        e,info,&argc,v,nullptr,nullptr
    )!=napi_ok || argc!=7)
        return fail(
            e,"mulBatch(A,B,O,threads,affinity,store,prefetch)"
        );

    unsigned char *a,*b,*o;
    size_t na,nb,no;
    uint32_t n,pref;
    std::string affinity,store;

    if(!buffer(e,v[0],a,na) ||
       !buffer(e,v[1],b,nb) ||
       !buffer(e,v[2],o,no) ||
       napi_get_value_uint32(
           e,v[3],&n
       )!=napi_ok ||
       !string(e,v[4],affinity) ||
       !string(e,v[5],store) ||
       napi_get_value_uint32(
           e,v[6],&pref
       )!=napi_ok)
        return nullptr;

    if(na==0 || na%16 ||
       na!=nb || na!=no ||
       n<1 || n>32 || pref>256 ||
       (affinity!="none" &&
        affinity!="compact" &&
        affinity!="spread") ||
       (store!="cached" &&
        store!="stream") ||
       a==o || b==o)
        return fail(
            e,"Dimensions ou options invalides"
        );

    __builtin_cpu_init();

    if(!(
        __builtin_cpu_supports("vpclmulqdq") &&
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("pclmul")
    ))
        return fail(
            e,"AVX512 VPCLMUL absent"
        );

    const size_t count=na/16;

    bool nt=(
        store=="stream" &&
        ((uintptr_t)o%64==0)
    );

    auto work=[&](int t) {
        size_t lo=count*(size_t)t/n;
        size_t hi=count*(size_t)(t+1)/n;

        bool localNt=nt && (lo%4==0);

        runRange(
            a,b,o,lo,hi,localNt,pref
        );
    };

    if(n==1) {
        work(0);
        lastPinned=0;
    } else {
        std::lock_guard<std::mutex> guard(
            poolMutex
        );

        if(!pool ||
           poolSize!=(int)n ||
           poolPolicy!=affinity) {

            pool.reset();
            pool=std::make_unique<Pool>(
                n,affinity
            );

            poolSize=n;
            poolPolicy=affinity;
        }

        pool->execute(work);
        lastPinned=pool->pinnedCount();
    }

    lastStream=nt;

    napi_value result;
    napi_get_undefined(e,&result);
    return result;
}

static void freeAligned(
    napi_env,void* p,void*
) {
    std::free(p);
}

static napi_value allocAligned(
    napi_env e,napi_callback_info info
) {
    napi_value args[1];
    size_t argc=1;

    if(napi_get_cb_info(
        e,info,&argc,args,nullptr,nullptr
    )!=napi_ok || argc!=1)
        return fail(
            e,"allocAligned(bytes)"
        );

    double d;

    if(napi_get_value_double(
        e,args[0],&d
    )!=napi_ok ||
       d<64 ||
       d>1073741824.0 ||
       d!=(double)(size_t)d)
        return fail(
            e,"Taille memoire invalide"
        );

    void* p=nullptr;

    if(posix_memalign(
        &p,64,(size_t)d
    )!=0)
        return fail(
            e,"Allocation memoire impossible"
        );

    napi_value result;

    napi_status rc=napi_create_external_buffer(
        e,(size_t)d,(char*)p,
        freeAligned,nullptr,&result
    );

    if(rc!=napi_ok) {
        std::free(p);
        return fail(
            e,"Buffer externe indisponible"
        );
    }

    return result;
}

static napi_value status(
    napi_env e,napi_callback_info
) {
    napi_value obj,v;
    napi_create_object(e,&obj);

    napi_create_int32(
        e,lastPinned,&v
    );

    napi_set_named_property(
        e,obj,"pinned",v
    );

    napi_get_boolean(
        e,lastStream,&v
    );

    napi_set_named_property(
        e,obj,"stream_aligned",v
    );

    return obj;
}

NAPI_MODULE_INIT() {
    napi_value f;

    napi_create_function(
        env,"mulBatch",NAPI_AUTO_LENGTH,
        mul,nullptr,&f
    );
    napi_set_named_property(
        env,exports,"mulBatch",f
    );

    napi_create_function(
        env,"allocAligned",NAPI_AUTO_LENGTH,
        allocAligned,nullptr,&f
    );
    napi_set_named_property(
        env,exports,"allocAligned",f
    );

    napi_create_function(
        env,"status",NAPI_AUTO_LENGTH,
        status,nullptr,&f
    );
    napi_set_named_property(
        env,exports,"status",f
    );

    return exports;
}
