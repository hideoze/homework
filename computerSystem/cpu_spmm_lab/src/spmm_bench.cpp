#include "kernels.h"
#include "pmu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>
#include <sys/resource.h>
#include <time.h>
#ifdef __linux__
#include <sched.h>
#endif

using Clock = std::chrono::steady_clock;
static double now_ms() { return std::chrono::duration<double,std::milli>(Clock::now().time_since_epoch()).count(); }
static double thread_ms() { timespec t{}; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static std::string js(const std::string& s) {
    std::string o="\"";
    for(unsigned char c:s) { if(c=='"'||c=='\\') { o+='\\';o+=c; }
      else if(c=='\n')o+="\\n"; else if(c=='\r')o+="\\r"; else if(c=='\t')o+="\\t";
      else if(c<32)o+='?'; else o+=c; }
    return o+'"';
}
struct Config {
    int m=4096,k=4096,n=128,degree=16,threads=1,chunk_rows=0,samples=7,profile_iters=3;
    unsigned seed=42;
    double target_ms=35;
    std::string matrix="uniform",file,row_kernel="simd",partition="rows",action="bench",events="none";
};
static Config parse(int argc,char**argv) {
    Config c;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--help") { std::cout<<"CPU CSR SpMM: --action bench|profile --matrix uniform|skew|banded --matrix-file path.mtx "
          "--m M --k K --n N --degree D --threads P --row-kernel simd|scalar|gather "
          "--partition rows|nnz --chunk-rows R --samples S --target-ms T --profile-iters R "
          "--events none|core|cache|branch --seed S\n"; std::exit(0); }
        if(i+1>=argc) throw std::runtime_error("missing value for "+a);
        std::string v=argv[++i];
        if(a=="--m")c.m=std::stoi(v); else if(a=="--k")c.k=std::stoi(v);
        else if(a=="--n")c.n=std::stoi(v); else if(a=="--degree")c.degree=std::stoi(v);
        else if(a=="--threads")c.threads=std::stoi(v); else if(a=="--chunk-rows")c.chunk_rows=std::stoi(v);
        else if(a=="--samples")c.samples=std::stoi(v); else if(a=="--profile-iters")c.profile_iters=std::stoi(v);
        else if(a=="--seed")c.seed=static_cast<unsigned>(std::stoul(v));
        else if(a=="--target-ms")c.target_ms=std::stod(v); else if(a=="--matrix")c.matrix=v;
        else if(a=="--matrix-file")c.file=v; else if(a=="--row-kernel")c.row_kernel=v;
        else if(a=="--partition")c.partition=v; else if(a=="--action")c.action=v;
        else if(a=="--events")c.events=v; else throw std::runtime_error("unknown option "+a);
    }
    if(c.m<=0||c.k<=0||c.n<=0||c.degree<=0||c.threads<1||c.threads>256||c.samples<3||c.samples>101||
       c.profile_iters<1||c.profile_iters>10000||c.chunk_rows<0||!std::isfinite(c.target_ms)||c.target_ms<=0||c.target_ms>5000)
        throw std::runtime_error("invalid dimensions, thread count, repeats, or timing budget");
    if(c.row_kernel!="simd"&&c.row_kernel!="scalar"&&c.row_kernel!="gather")throw std::runtime_error("unknown row kernel");
    if(c.partition!="rows"&&c.partition!="nnz")throw std::runtime_error("unknown partition");
    if(c.action!="bench"&&c.action!="profile")throw std::runtime_error("unknown action");
    if(c.events!="none"&&c.events!="core"&&c.events!="cache"&&c.events!="branch")throw std::runtime_error("unknown events");
    if(c.action=="bench"&&c.events!="none")throw std::runtime_error("PMU must use a separate profile pass");
    return c;
}
struct Matrix { int m=0,k=0; std::vector<int64_t> rp; std::vector<int> ci; std::vector<float> av; };
static Matrix make_matrix(Config& c) {
    Matrix a; a.m=c.m; a.k=c.k;
    std::mt19937 gen(c.seed); std::uniform_real_distribution<float> val(-1,1);
    if(!c.file.empty()) {
        std::ifstream f(c.file); if(!f)throw std::runtime_error("cannot open Matrix Market file");
        std::string line,banner,object,format,field,sym;
        std::getline(f,line);std::istringstream header(line);header>>banner>>object>>format>>field>>sym;
        if(banner!="%%MatrixMarket"||object!="matrix"||format!="coordinate"||
           (field!="real"&&field!="integer"&&field!="pattern")||(sym!="general"&&sym!="symmetric"))
            throw std::runtime_error("supports coordinate real/integer/pattern, general/symmetric Matrix Market only");
        do { if(!std::getline(f,line))throw std::runtime_error("missing Matrix Market shape"); } while(line.empty()||line[0]=='%');
        long long count=0; std::istringstream dims(line); dims>>a.m>>a.k>>count;
        if(!dims||a.m<=0||a.k<=0||count<0||count>100000000||(sym=="symmetric"&&a.m!=a.k))
            throw std::runtime_error("invalid or excessive Matrix Market shape");
        std::vector<std::tuple<int,int,float>> entries;entries.reserve(static_cast<size_t>(count));
        for(long long z=0;z<count;++z) {
            int r,q;double v=1;
            if(!(f>>r>>q))throw std::runtime_error("truncated Matrix Market entries");
            if(field!="pattern"&&!(f>>v))throw std::runtime_error("missing Matrix Market value");
            if(r<1||r>a.m||q<1||q>a.k||!std::isfinite(v)||std::abs(v)>std::numeric_limits<float>::max())
                throw std::runtime_error("invalid Matrix Market entry");
            entries.emplace_back(r-1,q-1,static_cast<float>(v));
            if(sym=="symmetric"&&r!=q)entries.emplace_back(q-1,r-1,static_cast<float>(v));
        }
        std::sort(entries.begin(),entries.end());
        a.rp.assign(static_cast<size_t>(a.m)+1,0);
        for(size_t i=0;i<entries.size();) {
            int r=std::get<0>(entries[i]),q=std::get<1>(entries[i]);double v=0;
            do {v+=std::get<2>(entries[i++]);}while(i<entries.size()&&std::get<0>(entries[i])==r&&std::get<1>(entries[i])==q);
            if(!std::isfinite(v)||std::abs(v)>std::numeric_limits<float>::max())throw std::runtime_error("coalesced value overflows FP32");
            if(v!=0) {a.ci.push_back(q);a.av.push_back(static_cast<float>(v));++a.rp[r+1];}
        }
        std::partial_sum(a.rp.begin(),a.rp.end(),a.rp.begin());c.m=a.m;c.k=a.k;
    } else {
        if(c.matrix!="uniform"&&c.matrix!="skew"&&c.matrix!="banded")throw std::runtime_error("unknown generator");
        a.rp.push_back(0);
        for(int r=0;r<a.m;++r) {
            int d=c.degree;
            if(c.matrix=="skew") d=(r<std::max(1,a.m/8))?c.degree*64:c.degree;
            d=std::min(d,a.k);
            std::vector<int> cols;
            if(c.matrix=="banded") {for(int j=0;j<d;++j)cols.push_back((r+j)%a.k);std::sort(cols.begin(),cols.end());}
            else {
                std::uniform_int_distribution<int> col(0,a.k-1);
                while(static_cast<int>(cols.size())<d) {
                    int missing=d-static_cast<int>(cols.size());for(int j=0;j<missing;++j)cols.push_back(col(gen));
                    std::sort(cols.begin(),cols.end());cols.erase(std::unique(cols.begin(),cols.end()),cols.end());
                }
            }
            for(int q:cols){a.ci.push_back(q);a.av.push_back(val(gen));}
            a.rp.push_back(static_cast<int64_t>(a.ci.size()));
        }
    }
    return a;
}
static uint64_t hash_bytes(uint64_t h,const void* ptr,size_t n) {
    const auto* p=static_cast<const unsigned char*>(ptr);for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;
}
struct Chunk {std::vector<int> bounds;};
static std::vector<Chunk> make_plan(const Config& c,const Matrix&a) {
    std::vector<Chunk> plan;int chunk=c.chunk_rows?c.chunk_rows:a.m;
    for(int start=0;start<a.m;start+=chunk) {
        int end=static_cast<int>(std::min<int64_t>(a.m,static_cast<int64_t>(start)+chunk));Chunk cp;cp.bounds.push_back(start);
        for(int t=1;t<c.threads;++t) {
            int split;
            if(c.partition=="rows")split=start+static_cast<int>((int64_t(end-start)*t)/c.threads);
            else {
                int64_t target=a.rp[start]+(a.rp[end]-a.rp[start])*t/c.threads;
                split=static_cast<int>(std::lower_bound(a.rp.begin()+start,a.rp.begin()+end+1,target)-a.rp.begin());
            }
            cp.bounds.push_back(std::clamp(split,start,end));
        }
        cp.bounds.push_back(end);plan.push_back(std::move(cp));
        if(end==a.m)break;
    }
    return plan;
}
struct alignas(128) Worker {
    int64_t rows=0,nnz=0,regions=0;double wall_ms=0,cpu_ms=0;
    int first_cpu=-1,last_cpu=-1;PmuResult pmu;
};
static void merge_pmu(PmuResult& a,const PmuResult& b) {
    if(b.status=="not_requested")return;
    if(a.status=="not_requested") {a=b;return;}
    if(b.status!="ok"&&b.status!="low_coverage") {a.status=b.status;a.reason=b.reason;return;}
    if(a.status!="ok"&&a.status!="low_coverage")return;
    for(int i=0;i<2;++i){a.raw[i]+=b.raw[i];a.scaled[i]+=b.scaled[i];}
    a.enabled_ns+=b.enabled_ns;a.running_ns+=b.running_ns;
    if(b.status=="low_coverage")a.status=b.status;
}
static void execute(const Config& c,const Matrix&a,const std::vector<float>& b,std::vector<float>& out,
                    const std::vector<Chunk>& plan,std::vector<Worker>* stats=nullptr) {
    RowKernel kernel=c.row_kernel=="simd"?row_simd:c.row_kernel=="scalar"?row_scalar:row_gather;
    bool wrong_team=false;
    for(const auto& chunk:plan) {
        #pragma omp parallel num_threads(c.threads) shared(wrong_team)
        {
            int tid=omp_get_thread_num();
            if(omp_get_num_threads()!=c.threads) {
                #pragma omp atomic write
                wrong_team=true;
            } else {
                int lo=chunk.bounds[tid],hi=chunk.bounds[tid+1];
                if(!stats) {
                    for(int r=lo;r<hi;++r)kernel(r,c.n,a.rp.data(),a.ci.data(),a.av.data(),b.data(),out.data());
                } else {
                    Pmu pmu(c.events);pmu.start();
                    auto& w=(*stats)[tid];
                    #ifdef __linux__
                    int cpu=sched_getcpu();if(w.first_cpu<0)w.first_cpu=cpu;w.last_cpu=cpu;
                    #endif
                    double ct=thread_ms(),wt=now_ms();
                    for(int r=lo;r<hi;++r)kernel(r,c.n,a.rp.data(),a.ci.data(),a.av.data(),b.data(),out.data());
                    w.wall_ms+=now_ms()-wt;w.cpu_ms+=thread_ms()-ct;
                    merge_pmu(w.pmu,pmu.stop());
                    w.rows+=hi-lo;w.nnz+=a.rp[hi]-a.rp[lo];++w.regions;
                }
            }
        }
    }
    if(wrong_team)throw std::runtime_error("OpenMP supplied fewer workers than requested; refusing invalid partition");
}
struct Validation {bool passed=true;double max_abs=0,max_scaled=0,checksum=0;};
static Validation validate(const Config&c,const Matrix&a,const std::vector<float>&b,const std::vector<float>&out) {
    Validation v;
    for(int r=0;r<a.m;++r)for(int f=0;f<c.n;++f) {
        double ref=0;for(int64_t z=a.rp[r];z<a.rp[r+1];++z)ref+=double(a.av[z])*double(b[static_cast<size_t>(a.ci[z])*c.n+f]);
        double x=out[static_cast<size_t>(r)*c.n+f],e=std::abs(x-ref),scaled=e/(1e-4+3e-4*std::abs(ref));
        if(!std::isfinite(x)||!std::isfinite(ref)||scaled>1)v.passed=false;
        v.max_abs=std::max(v.max_abs,e);v.max_scaled=std::max(v.max_scaled,scaled);v.checksum+=x;
    }
    return v;
}
static double median(std::vector<double> x) {std::sort(x.begin(),x.end());size_t n=x.size();return n%2?x[n/2]:(x[n/2-1]+x[n/2])/2;}
static void array_json(const std::vector<double>& v) {std::cout<<'[';for(size_t i=0;i<v.size();++i){if(i)std::cout<<',';std::cout<<v[i];}std::cout<<']';}
static double ru_ms(const rusage&r) {return 1000.0*(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+(r.ru_utime.tv_usec+r.ru_stime.tv_usec)/1000.0;}
int main(int argc,char**argv) {
    try {
        Config c=parse(argc,argv);omp_set_dynamic(0);omp_set_num_threads(c.threads);
        double start=now_ms();Matrix a=make_matrix(c);
        uint64_t dense_count=static_cast<uint64_t>(a.k)*c.n+static_cast<uint64_t>(a.m)*c.n;
        if(dense_count>250000000)throw std::runtime_error("dense B+C exceeds the built-in 1 GB safety cap");
        std::vector<float>b(static_cast<size_t>(a.k)*c.n),out(static_cast<size_t>(a.m)*c.n);
        std::mt19937 gen(c.seed+7919);std::uniform_real_distribution<float> val(-1,1);for(float&x:b)x=val(gen);
        double setup_ms=now_ms()-start;start=now_ms();auto plan=make_plan(c,a);double plan_ms=now_ms()-start;
        uint64_t h=1469598103934665603ULL;
        h=hash_bytes(h,a.rp.data(),a.rp.size()*sizeof(int64_t));h=hash_bytes(h,a.ci.data(),a.ci.size()*sizeof(int));
        h=hash_bytes(h,a.av.data(),a.av.size()*sizeof(float));h=hash_bytes(h,b.data(),b.size()*sizeof(float));
        execute(c,a,b,out,plan);Validation check=validate(c,a,b,out);
        if(!check.passed)throw std::runtime_error("full FP64 reference check failed");
        for(int i=0;i<3;++i)execute(c,a,b,out,plan);
        std::vector<double> times;std::vector<Worker> workers(c.threads);int batch=1;
        rusage before{},after{};double measured_ms=0;
        if(c.action=="bench") {
            start=now_ms();execute(c,a,b,out,plan);double pilot=std::max(0.001,now_ms()-start);
            batch=std::clamp(static_cast<int>(std::min(4096.0,std::ceil(c.target_ms/pilot))),1,4096);
            getrusage(RUSAGE_SELF,&before);double all=now_ms();
            for(int s=0;s<c.samples;++s) {
                start=now_ms();for(int r=0;r<batch;++r)execute(c,a,b,out,plan);
                times.push_back((now_ms()-start)/batch);
            }
            measured_ms=now_ms()-all;getrusage(RUSAGE_SELF,&after);
        } else {
            getrusage(RUSAGE_SELF,&before);start=now_ms();
            for(int r=0;r<c.profile_iters;++r)execute(c,a,b,out,plan,&workers);
            measured_ms=now_ms()-start;getrusage(RUSAGE_SELF,&after);
        }
        // Validate final output too: correctness is checked outside BOTH timing boundaries.
        check=validate(c,a,b,out);if(!check.passed)throw std::runtime_error("post-run reference check failed");
        double mean=double(a.ci.size())/a.m,variance=0,max_row=0;
        for(int r=0;r<a.m;++r){double d=a.rp[r+1]-a.rp[r];variance+=(d-mean)*(d-mean);max_row=std::max(max_row,d);}
        uint64_t footprint=a.rp.size()*sizeof(int64_t)+a.ci.size()*sizeof(int)+a.av.size()*sizeof(float)+dense_count*sizeof(float);
        std::cout<<std::setprecision(12)<<"{\"schema_version\":1,\"action\":"<<js(c.action)<<",\"config\":{";
        std::cout<<"\"m\":"<<a.m<<",\"k\":"<<a.k<<",\"n\":"<<c.n<<",\"threads\":"<<c.threads
          <<",\"row_kernel\":"<<js(c.row_kernel)<<",\"partition\":"<<js(c.partition)<<",\"chunk_rows\":"<<c.chunk_rows<<",\"seed\":"<<c.seed<<"},";
        std::cout<<"\"matrix\":{\"nnz\":"<<a.ci.size()<<",\"row_nnz_mean\":"<<mean<<",\"row_nnz_cv\":"<<(mean?std::sqrt(variance/a.m)/mean:0)
          <<",\"row_nnz_max\":"<<max_row<<",\"storage_footprint_bytes\":"<<footprint<<",\"input_hash\":"<<js(std::to_string(h))<<"},";
        std::cout<<"\"work\":{\"algorithmic_flops_per_spmm\":"<<2.0*a.ci.size()*c.n
          <<",\"parallel_regions_per_spmm\":"<<plan.size()<<",\"zero_C_included\":true,\"setup_included\":false},";
        std::cout<<"\"setup_ms\":"<<setup_ms<<",\"plan_ms\":"<<plan_ms<<",\"validation\":{\"passed\":"<<(check.passed?"true":"false")
          <<",\"scope\":\"all output elements, independent FP64 accumulation\",\"max_abs_error\":"<<check.max_abs
          <<",\"max_scaled_error\":"<<check.max_scaled<<",\"checksum\":"<<check.checksum<<"},";
        std::cout<<"\"timing\":{\"clock\":\"std::chrono::steady_clock\",\"regime\":\"warmed repeated execution\",\"batch\":"<<batch
          <<",\"samples_ms\":";array_json(times);
        if(!times.empty()) {
            double med=median(times);std::vector<double>dev;for(double x:times)dev.push_back(std::abs(x-med));
            std::cout<<",\"median_ms\":"<<med<<",\"relative_mad\":"<<(med?median(dev)/med:0)
              <<",\"gflops\":"<<(med?2.0*a.ci.size()*c.n/(med*1e6):0);
        }
        std::cout<<",\"measured_region_ms\":"<<measured_ms<<"},";
        std::cout<<"\"os\":{\"process_cpu_ms\":"<<ru_ms(after)-ru_ms(before)<<",\"voluntary_context_switches\":"<<after.ru_nvcsw-before.ru_nvcsw
          <<",\"involuntary_context_switches\":"<<after.ru_nivcsw-before.ru_nivcsw<<",\"minor_faults\":"<<after.ru_minflt-before.ru_minflt
          <<",\"major_faults\":"<<after.ru_majflt-before.ru_majflt<<"},";
        std::cout<<"\"profile_iters\":"<<(c.action=="profile"?c.profile_iters:0)<<",\"event_group\":"<<js(c.events)<<",\"workers\":[";
        if(c.action=="profile")for(int t=0;t<c.threads;++t) {
            if(t)std::cout<<',';const auto&w=workers[t];bool valid=w.pmu.status=="ok"||w.pmu.status=="low_coverage";
            std::cout<<"{\"id\":"<<t<<",\"rows\":"<<w.rows<<",\"nnz\":"<<w.nnz<<",\"active_wall_ms\":"<<w.wall_ms
              <<",\"active_cpu_ms\":"<<w.cpu_ms<<",\"regions\":"<<w.regions<<",\"cpu_first\":"<<w.first_cpu<<",\"cpu_last\":"<<w.last_cpu
              <<",\"pmu\":{\"status\":"<<js(w.pmu.status)<<",\"reason\":"<<js(w.pmu.reason)<<",\"enabled_ns\":"<<w.pmu.enabled_ns
              <<",\"running_ns\":"<<w.pmu.running_ns<<",\"raw\":";
            if(valid)std::cout<<'['<<w.pmu.raw[0]<<','<<w.pmu.raw[1]<<']';else std::cout<<"null";
            std::cout<<",\"scaled\":";if(valid)std::cout<<'['<<w.pmu.scaled[0]<<','<<w.pmu.scaled[1]<<']';else std::cout<<"null";
            std::cout<<"}}";
        }
        std::cout<<"]}\n";return 0;
    } catch(const std::exception&e) {std::cerr<<"spmm_bench: "<<e.what()<<'\n';return 2;}
}
