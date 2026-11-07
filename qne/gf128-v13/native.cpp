#include <immintrin.h>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

using u64 = uint64_t;
using u128 = unsigned __int128;

struct B {
    u64 lo, hi;
};

static u128 to128(B b) {
    return (u128(b.hi) << 64) | b.lo;
}

static B from128(u128 x) {
    return {(u64)x, (u64)(x >> 64)};
}

static bool eq(B a, B b) {
    return a.lo == b.lo && a.hi == b.hi;
}

static B bx(B a, B b) {
    return {a.lo ^ b.lo, a.hi ^ b.hi};
}

// GF(2^128), reduction selon le polynome QNE 0x87
static B reduce(u128 low, u128 high) {
    u128 spill =
        (high >> 127) ^
        (high >> 126) ^
        (high >> 121);

    return from128(
        low ^ high ^
        (high << 1) ^
        (high << 2) ^
        (high << 7) ^
        spill ^
        (spill << 1) ^
        (spill << 2) ^
        (spill << 7)
    );
}

// Reference logicielle
static B scalar(B aa, B bb) {
    u128 a = to128(aa);
    u128 b = to128(bb);
    u128 r = 0;

    for (int i = 0; i < 128; i++) {
        if (b & 1) r ^= a;
        b >>= 1;

        u128 carry = a >> 127;
        a <<= 1;

        if (carry) a ^= 0x87;
    }

    return from128(r);
}

// Multiplication materielle : 3 produits 64 bits
__attribute__((target("pclmul,sse2")))
static B pclmul(B a, B b) {
    __m128i x = _mm_set_epi64x(
        (long long)a.hi, (long long)a.lo
    );

    __m128i y = _mm_set_epi64x(
        (long long)b.hi, (long long)b.lo
    );

    __m128i lo = _mm_clmulepi64_si128(x,y,0x00);
    __m128i hi = _mm_clmulepi64_si128(x,y,0x11);

    __m128i ax = _mm_xor_si128(
        x,_mm_srli_si128(x,8)
    );

    __m128i by = _mm_xor_si128(
        y,_mm_srli_si128(y,8)
    );

    __m128i mid = _mm_xor_si128(
        _mm_clmulepi64_si128(ax,by,0x00),
        _mm_xor_si128(lo,hi)
    );

    u64 l[2],h[2],m[2];

    _mm_storeu_si128((__m128i*)l,lo);
    _mm_storeu_si128((__m128i*)h,hi);
    _mm_storeu_si128((__m128i*)m,mid);

    u128 L =
        u128(l[0]) |
        (u128(l[1]^m[0]) << 64);

    u128 H =
        u128(h[0]^m[1]) |
        (u128(h[1]) << 64);

    return reduce(L,H);
}

// Decalage interne a chaque voie 128 bits
template<int N>
__attribute__((target("avx512f,avx512bw")))
static __m512i shl128(__m512i v) {
    return _mm512_or_si512(
        _mm512_slli_epi64(v,N),
        _mm512_bslli_epi128(
            _mm512_srli_epi64(v,64-N),8
        )
    );
}

// 4 multiplications GF128 paralleles, reduction SIMD
__attribute__((target("avx512f,avx512bw,vpclmulqdq")))
static void vpclmul4(const B* a,const B* b,B* out) {
    __m512i A = _mm512_loadu_si512((const void*)a);
    __m512i C = _mm512_loadu_si512((const void*)b);

    __m512i z0 = _mm512_clmulepi64_epi128(A,C,0x00);
    __m512i z2 = _mm512_clmulepi64_epi128(A,C,0x11);

    __m512i ax = _mm512_xor_si512(
        A,_mm512_bsrli_epi128(A,8)
    );

    __m512i cx = _mm512_xor_si512(
        C,_mm512_bsrli_epi128(C,8)
    );

    __m512i z1 = _mm512_xor_si512(
        _mm512_clmulepi64_epi128(ax,cx,0x00),
        _mm512_xor_si512(z0,z2)
    );

    __m512i L = _mm512_xor_si512(
        z0,_mm512_bslli_epi128(z1,8)
    );

    __m512i H = _mm512_xor_si512(
        z2,_mm512_bsrli_epi128(z1,8)
    );

    __m512i hi64 = _mm512_bsrli_epi128(H,8);

    __m512i sp = _mm512_xor_si512(
        _mm512_srli_epi64(hi64,63),
        _mm512_xor_si512(
            _mm512_srli_epi64(hi64,62),
            _mm512_srli_epi64(hi64,57)
        )
    );

    __m512i R = _mm512_xor_si512(L,H);

    R = _mm512_xor_si512(R,shl128<1>(H));
    R = _mm512_xor_si512(R,shl128<2>(H));
    R = _mm512_xor_si512(R,shl128<7>(H));

    R = _mm512_xor_si512(R,sp);

    R = _mm512_xor_si512(
        R,_mm512_slli_epi64(sp,1)
    );

    R = _mm512_xor_si512(
        R,_mm512_slli_epi64(sp,2)
    );

    R = _mm512_xor_si512(
        R,_mm512_slli_epi64(sp,7)
    );

    _mm512_storeu_si512((void*)out,R);
}

static u64 rand64(u64& s) {
    s += 0x9e3779b97f4a7c15ULL;
    u64 z=s;

    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;

    return z^(z>>31);
}

static B rnd(u64& s) {
    return {rand64(s),rand64(s)};
}

static B parse(std::string s) {
    if(s.size()>32 || s.empty())
        throw std::runtime_error("Hex invalide");

    s=std::string(32-s.size(),'0')+s;

    return {
        std::stoull(s.substr(16),0,16),
        std::stoull(s.substr(0,16),0,16)
    };
}

static std::string hex(B b) {
    std::ostringstream o;

    o<<std::hex<<std::setfill('0')
     <<std::setw(16)<<b.hi
     <<std::setw(16)<<b.lo;

    return o.str();
}

// Reconstruction geometrique exacte
struct Segment {
    B p,s;
};

static Segment join(Segment a,Segment b) {
    return {
        scalar(a.p,b.p),
        bx(a.s,scalar(a.p,b.s))
    };
}

static Segment repeat(Segment a,u64 n) {
    Segment r{{1,0},{0,0}};

    while(n) {
        if(n&1) r=join(r,a);
        n>>=1;
        if(n) a=join(a,a);
    }

    return r;
}

static std::atomic<u64> sink{0};

// Chronometrage median sur 3 passages
static double benchmark(
    const std::string& mode,
    int threads,
    u64 count
) {
    std::vector<B> A(4096),C(4096);

    u64 state=0x9badf00d231141ULL;

    for(int i=0;i<4096;i++) {
        A[i]=rnd(state);
        C[i]=rnd(state);
    }

    std::vector<double> times;

    for(int pass=0;pass<3;pass++) {
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};

        std::vector<std::thread> pool;
        std::vector<u64> checks(threads);

        for(int t=0;t<threads;t++) {
            pool.emplace_back([&,t] {
                u64 start=count*t/threads;
                u64 stop=count*(t+1)/threads;

                u64 h=0x243f6a8885a308d3ULL^u64(t);

                ready.fetch_add(1);

                while(!go.load())
                    std::this_thread::yield();

                if(mode=="vpclmul") {
                    B aa[4],bb[4],out[4];
                    u64 i=start;

                    for(;i+4<=stop;i+=4) {
                        for(int j=0;j<4;j++) {
                            size_t k=(i+j)&4095;
                            aa[j]=A[k];
                            bb[j]=C[k];
                        }

                        vpclmul4(aa,bb,out);

                        for(int j=0;j<4;j++)
                            h+=out[j].lo ^
                               (out[j].hi<<1) ^
                               (i+j);
                    }

                    for(;i<stop;i++) {
                        size_t k=i&4095;
                        B x=pclmul(A[k],C[k]);

                        h+=x.lo^(x.hi<<1)^i;
                    }
                }
                else {
                    for(u64 i=start;i<stop;i++) {
                        size_t k=i&4095;

                        B x=(mode=="scalar")
                            ? scalar(A[k],C[k])
                            : pclmul(A[k],C[k]);

                        h+=x.lo^(x.hi<<1)^i;
                    }
                }

                checks[t]=h;
            });
        }

        while(ready.load()<threads)
            std::this_thread::yield();

        auto begin=std::chrono::steady_clock::now();

        go.store(true);

        for(auto& th:pool)
            th.join();

        auto end=std::chrono::steady_clock::now();

        for(u64 h:checks)
            sink.fetch_xor(h);

        times.push_back(
            std::chrono::duration<double>(
                end-begin
            ).count()
        );
    }

    std::sort(times.begin(),times.end());

    return double(count)/times[1];
}

int main(int argc,char**argv) {
    std::string file;

    if(argc>1)
        file=argv[1];

    __builtin_cpu_init();

    bool pcl=__builtin_cpu_supports("pclmul");

    bool vp=
        __builtin_cpu_supports("vpclmulqdq") &&
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512bw");

    std::cout<<"PCLMUL="<<pcl
             <<" VPCLMUL512="<<vp<<"\n";

    u64 state=0xabcd123457890ffeULL;

    for(int i=0;i<4096;i++) {
        B a=rnd(state);
        B b=rnd(state);
        B want=scalar(a,b);

        if(pcl && !eq(want,pclmul(a,b))) {
            std::cerr<<"FAIL PCLMUL\n";
            return 3;
        }

        if(vp) {
            B aa[4],bb[4],r[4];

            for(int j=0;j<4;j++) {
                aa[j]=rnd(state);
                bb[j]=rnd(state);
            }

            vpclmul4(aa,bb,r);

            for(int j=0;j<4;j++) {
                if(!eq(r[j],scalar(aa[j],bb[j]))) {
                    std::cerr<<"FAIL VPCLMUL\n";
                    return 4;
                }
            }
        }
    }

    std::cout<<"NATIVE_GF128_VECTORS=4096 PASS\n";

    if(!file.empty()) {
        std::ifstream f(file);
        std::string a,b,c;
        int n=0,wrong=0;

        if(!f) return 5;

        while(f>>a>>b>>c) {
            B got=scalar(parse(a),parse(b));
            B expected=parse(c);

            n++;

            if(!eq(got,expected))
                wrong++;
        }

        std::cout<<"QNE_CANONICAL="<<n
                 <<" status="
                 <<(wrong?"FAIL":"PASS")<<"\n";

        if(wrong || n<100)
            return 6;
    }

    B alpha{
        0x89abcdef01234567ULL,
        0x123456789abcdef0ULL
    };

    Segment g{alpha,{1,0}};

    Segment a=repeat(g,4096);
    Segment b=repeat(a,8092);
    Segment direct=repeat(g,4096ULL*8092ULL);

    if(!eq(b.s,direct.s) ||
       !eq(b.p,direct.p)) {
        std::cerr<<"FAIL RECONSTRUCTION\n";
        return 7;
    }

    Segment slow{{1,0},{0,0}};

    for(int i=0;i<1024;i++)
        slow=join(slow,g);

    if(!eq(slow.s,repeat(g,1024).s)) {
        std::cerr<<"FAIL BRUTE CHECK\n";
        return 8;
    }

    std::cout
      <<"LOGICAL_4096x8092=33144832"
      <<" RECONSTRUCTION=PASS"
      <<" RESULT="<<hex(b.s)<<"\n";

    if(!pcl) {
        std::cout<<"NO_PCLMUL BENCH_SKIPPED\n";
        return 0;
    }

    const u64 TOTAL=4096ULL*8092ULL;

    unsigned hw=std::thread::hardware_concurrency();

    int maximum=hw>=32?32:(hw>=16?16:int(hw));

    std::cout<<std::fixed<<std::setprecision(2);

    for(int threads : {1,16,32}) {
        if(threads>maximum)
            continue;

        // Comparaison sur la meme taille de charge
        const u64 SMALL=1048576ULL;

        std::cout<<"BENCH scalar"
                 <<" threads="<<threads
                 <<" ops="<<SMALL
                 <<" mul_s="
                 <<benchmark("scalar",threads,SMALL)
                 <<"\n";

        std::cout<<"BENCH pclmul"
                 <<" threads="<<threads
                 <<" ops="<<SMALL
                 <<" mul_s="
                 <<benchmark("pclmul",threads,SMALL)
                 <<"\n";

        if(vp)
            std::cout<<"BENCH vpclmul"
                     <<" threads="<<threads
                     <<" ops="<<SMALL
                     <<" mul_s="
                     <<benchmark("vpclmul",threads,SMALL)
                     <<"\n";

        // Amplification physique 4096 x 8092
        std::cout<<"FULL pclmul"
                 <<" threads="<<threads
                 <<" ops="<<TOTAL
                 <<" mul_s="
                 <<benchmark("pclmul",threads,TOTAL)
                 <<"\n";

        if(vp)
            std::cout<<"FULL vpclmul"
                     <<" threads="<<threads
                     <<" ops="<<TOTAL
                     <<" mul_s="
                     <<benchmark("vpclmul",threads,TOTAL)
                     <<"\n";
    }

    std::cout<<"COMPLETE=PASS"
             <<" checksum="<<sink.load()<<"\n";

    return 0;
}
