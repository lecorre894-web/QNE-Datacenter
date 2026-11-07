#define main qne_reference_main_unused
#include "native.cpp"
#undef main

#include <node_api.h>
#include <barrier>
#include <functional>
#include <memory>
#include <mutex>
#include <sched.h>
#include <pthread.h>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>

struct Buffer {
    unsigned char* p;
    size_t n;
};

static bool getBuffer(
    napi_env e,napi_value v,Buffer& b
) {
    bool ok=false;
    void* p=nullptr;
    size_t n=0;

    if(napi_is_buffer(e,v,&ok)!=napi_ok ||
       !ok ||
       napi_get_buffer_info(e,v,&p,&n)!=napi_ok) {
        napi_throw_type_error(
            e,nullptr,"Buffer requis");
        return false;
    }

    b={static_cast<unsigned char*>(p),n};
    return true;
}

static bool getU32(
    napi_env e,napi_value v,uint32_t& x
) {
    return napi_get_value_uint32(e,v,&x)==napi_ok;
}

static bool getString(
    napi_env e,napi_value v,std::string& s
) {
    char tmp[32];
    size_t len=0;

    if(napi_get_value_string_utf8(
        e,v,tmp,sizeof(tmp),&len)!=napi_ok)
        return false;

    s.assign(tmp,len);
    return true;
}

static napi_value error(
    napi_env e,const char* msg
) {
    napi_throw_error(e,nullptr,msg);
    return nullptr;
}

static napi_value undefined(napi_env e) {
    napi_value v;
    napi_get_undefined(e,&v);
    return v;
}

static B read128(const unsigned char* p) {
    B b;
    std::memcpy(&b,p,16);
    return b;
}

static void write128(unsigned char* p,B b) {
    std::memcpy(p,&b,16);
}

static inline B xor128(B a,B b) {
    return {a.lo^b.lo,a.hi^b.hi};
}

// GF128 direct, sans banque de 4096 octets.
// VPCLMUL traite 4 elements de 128 bits.
static void gfWork(
    const unsigned char* a,
    const unsigned char* b,
    unsigned char* out,
    size_t lo,size_t hi,
    bool simd,unsigned prefetch
) {
    size_t i=lo;

    if(simd) {
        B x[4],y[4],z[4];

        for(;i+4<=hi;i+=4) {
            if(prefetch && i+prefetch<hi) {
                __builtin_prefetch(
                    a+16*(i+prefetch),0,1);
                __builtin_prefetch(
                    b+16*(i+prefetch),0,1);
            }

            std::memcpy(x,a+16*i,sizeof(x));
            std::memcpy(y,b+16*i,sizeof(y));

            vpclmul4(x,y,z);

            std::memcpy(
                out+16*i,z,sizeof(z));
        }
    }

    for(;i<hi;i++) {
        write128(
            out+16*i,
            pclmul(
                read128(a+16*i),
                read128(b+16*i)
            )
        );
    }
}

// Datacenter : facteurs GF128 reutilises
// dans chaque worker.
static void dcWork(
    const unsigned char* seeds,
    const unsigned char* factors,
    unsigned char* out,
    size_t lo,size_t hi,
    unsigned regions,
    bool simd,unsigned prefetch
) {
    B f[64];

    for(unsigned j=0;j<regions;j++)
        f[j]=read128(factors+16*j);

    size_t i=lo;

    if(simd) {
        B x[4],y[4],z[4],sum[4];

        for(;i+4<=hi;i+=4) {
            if(prefetch && i+prefetch<hi)
                __builtin_prefetch(
                    seeds+16*(i+prefetch)*regions,
                    0,1
                );

            for(B& v:sum)
                v={0,0};

            for(unsigned j=0;j<regions;j++) {
                for(unsigned lane=0;lane<4;lane++) {
                    x[lane]=read128(
                        seeds+
                        16*((i+lane)*regions+j)
                    );

                    y[lane]=f[j];
                }

                vpclmul4(x,y,z);

                for(unsigned lane=0;lane<4;lane++)
                    sum[lane]=xor128(
                        sum[lane],z[lane]);
            }

            std::memcpy(
                out+16*i,sum,sizeof(sum));
        }
    }

    for(;i<hi;i++) {
        B sum{0,0};

        for(unsigned j=0;j<regions;j++) {
            sum=xor128(
                sum,
                pclmul(
                    read128(
                        seeds+16*(i*regions+j)
                    ),
                    f[j]
                )
            );
        }

        write128(out+16*i,sum);
    }
}

// Threads persistants avec synchronisation
// C++20 et affinite pthreads.
class Workers {
    std::barrier<> start_,done_;
    std::vector<std::thread> threads_;
    std::function<void(int)> job_;
    bool stopping_=false;
    std::atomic<int> pinned_{0};

public:
    Workers(int n,const std::string& policy)
      : start_(n+1),done_(n+1) {

        cpu_set_t allowed;
        CPU_ZERO(&allowed);

        std::vector<int> cpus;

        if(sched_getaffinity(
            0,sizeof(allowed),&allowed)==0) {
            for(int i=0;i<CPU_SETSIZE;i++) {
                if(CPU_ISSET(i,&allowed))
                    cpus.push_back(i);
            }
        }

        for(int t=0;t<n;t++) {
            threads_.emplace_back(
                [&,t,policy,cpus]() {
                    if(policy!="none" &&
                       cpus.size()>=(size_t)n) {

                        size_t index=
                            policy=="compact"
                            ?t
                            :size_t(
                                uint64_t(t)*
                                cpus.size()/n
                            );

                        cpu_set_t target;
                        CPU_ZERO(&target);
                        CPU_SET(cpus[index],&target);

                        if(pthread_setaffinity_np(
                            pthread_self(),
                            sizeof(target),
                            &target
                        )==0)
                            pinned_.fetch_add(1);
                    }

                    for(;;) {
                        start_.arrive_and_wait();

                        if(stopping_)
                            break;

                        job_(t);

                        done_.arrive_and_wait();
                    }
                }
            );
        }
    }

    void execute(std::function<void(int)> f) {
        job_=std::move(f);
        start_.arrive_and_wait();
        done_.arrive_and_wait();
    }

    int pinned() const {
        return pinned_.load();
    }

    ~Workers() {
        stopping_=true;
        start_.arrive_and_wait();

        for(auto& t:threads_)
            t.join();
    }
};

static std::unique_ptr<Workers> pool;
static std::mutex poolMutex;
static int poolThreads=0;
static std::string poolPolicy;
static int lastPinned=0;

static void schedule(
    unsigned threads,
    const std::string& policy,
    const std::function<void(int)>& f
) {
    if(threads==1) {
        f(0);
        lastPinned=0;
        return;
    }

    std::lock_guard<std::mutex> guard(
        poolMutex);

    if(!pool ||
       poolThreads!=(int)threads ||
       poolPolicy!=policy) {

        pool.reset();

        pool=std::make_unique<Workers>(
            threads,policy);

        poolThreads=threads;
        poolPolicy=policy;
    }

    pool->execute(f);
    lastPinned=pool->pinned();
}

static bool options(
    napi_env e,
    napi_value tm,
    napi_value pol,
    napi_value pr,
    uint32_t& threads,
    std::string& policy,
    uint32_t& prefetch
) {
    if(!getU32(e,tm,threads) ||
       !getString(e,pol,policy) ||
       !getU32(e,pr,prefetch))
        return false;

    if(threads<1 ||
       threads>32 ||
       prefetch>256 ||
       (policy!="none" &&
        policy!="compact" &&
        policy!="spread")) {

        napi_throw_range_error(
            e,nullptr,
            "Parametres multithread invalides"
        );

        return false;
    }

    __builtin_cpu_init();

    if(!__builtin_cpu_supports("pclmul") ||
       !__builtin_cpu_supports("vpclmulqdq") ||
       !__builtin_cpu_supports("avx512f") ||
       !__builtin_cpu_supports("avx512bw")) {

        napi_throw_error(
            e,nullptr,"VPCLMUL512 absent");
        return false;
    }

    return true;
}

static napi_value mulBatch(
    napi_env e,napi_callback_info info
) {
    napi_value v[6];
    size_t argc=6;

    if(napi_get_cb_info(
        e,info,&argc,v,nullptr,nullptr
    )!=napi_ok || argc!=6)
        return error(e,"mulBatch: 6 arguments");

    Buffer a,b,o;
    uint32_t t,pref;
    std::string policy;

    if(!getBuffer(e,v[0],a) ||
       !getBuffer(e,v[1],b) ||
       !getBuffer(e,v[2],o) ||
       !options(
           e,v[3],v[4],v[5],
           t,policy,pref
       ))
        return nullptr;

    if(!a.n || a.n%16 ||
       a.n!=b.n || a.n!=o.n ||
       o.p==a.p || o.p==b.p)
        return error(
            e,"Dimensions GF128 invalides");

    size_t n=a.n/16;

    schedule(t,policy,[&](int i) {
        gfWork(
            a.p,b.p,o.p,
            n*i/t,n*(i+1)/t,
            true,pref
        );
    });

    return undefined(e);
}

static napi_value dcBatch(
    napi_env e,napi_callback_info info
) {
    napi_value v[8];
    size_t argc=8;

    if(napi_get_cb_info(
        e,info,&argc,v,nullptr,nullptr
    )!=napi_ok || argc!=8)
        return error(e,"dcBatch: 8 arguments");

    Buffer s,f,o;
    uint32_t requests,regions,t,pref;
    std::string policy;

    if(!getBuffer(e,v[0],s) ||
       !getBuffer(e,v[1],f) ||
       !getBuffer(e,v[2],o) ||
       !getU32(e,v[3],requests) ||
       !getU32(e,v[4],regions) ||
       !options(
           e,v[5],v[6],v[7],
           t,policy,pref
       ))
        return nullptr;

    if(!requests ||
       !regions || regions>64 ||
       s.n!=(size_t)requests*regions*16 ||
       f.n!=(size_t)regions*16 ||
       o.n!=(size_t)requests*16 ||
       s.p==o.p || f.p==o.p)
        return error(
            e,"Dimensions Datacenter invalides");

    schedule(t,policy,[&](int i) {
        dcWork(
            s.p,f.p,o.p,
            (size_t)requests*i/t,
            (size_t)requests*(i+1)/t,
            regions,true,pref
        );
    });

    return undefined(e);
}

static napi_value status(
    napi_env e,napi_callback_info
) {
    napi_value o,x;
    napi_create_object(e,&o);

    napi_create_int32(e,lastPinned,&x);

    napi_set_named_property(
        e,o,"pinned_threads",x);

    return o;
}

NAPI_MODULE_INIT() {
    napi_value f;

    napi_create_function(
        env,"mulBatch",NAPI_AUTO_LENGTH,
        mulBatch,nullptr,&f);

    napi_set_named_property(
        env,exports,"mulBatch",f);

    napi_create_function(
        env,"dcBatch",NAPI_AUTO_LENGTH,
        dcBatch,nullptr,&f);

    napi_set_named_property(
        env,exports,"dcBatch",f);

    napi_create_function(
        env,"status",NAPI_AUTO_LENGTH,
        status,nullptr,&f);

    napi_set_named_property(
        env,exports,"status",f);

    return exports;
}
