#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <string>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Two events per pass, opened on each OpenMP worker (pid=0,cpu=-1).
// Only row computation is enabled, not allocation/initialization/validation.
// This is counting, NOT sampled call-stack or source-line profiling.
struct PmuResult {
    std::string status = "not_requested", reason;
    std::array<double, 2> raw{0,0}, scaled{0,0};
    uint64_t enabled_ns = 0, running_ns = 0;
};
class Pmu {
    int leader_ = -1, member_ = -1;
    PmuResult result_;
public:
    explicit Pmu(const std::string& kind) {
        if (kind == "none") return;
#ifdef __linux__
        std::array<uint64_t,2> events;
        if (kind == "core") events = {PERF_COUNT_HW_CPU_CYCLES,PERF_COUNT_HW_INSTRUCTIONS};
        else if (kind == "cache") events = {PERF_COUNT_HW_CACHE_REFERENCES,PERF_COUNT_HW_CACHE_MISSES};
        else if (kind == "branch") events = {PERF_COUNT_HW_BRANCH_INSTRUCTIONS,PERF_COUNT_HW_BRANCH_MISSES};
        else { result_.status="error"; result_.reason="unknown event group"; return; }
        for (int i=0;i<2;++i) {
            perf_event_attr a{};
            a.type=PERF_TYPE_HARDWARE; a.size=sizeof(a); a.config=events[i];
            a.disabled=(i==0); a.exclude_kernel=1; a.exclude_hv=1;
            a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
            int fd=static_cast<int>(syscall(SYS_perf_event_open,&a,0,-1,i==0?-1:leader_,PERF_FLAG_FD_CLOEXEC));
            if(fd<0) {
                result_.status="unavailable"; result_.reason=std::string("perf_event_open: ")+std::strerror(errno);
                return;
            }
            if(i==0) leader_=fd; else member_=fd;
        }
        result_.status="ready";
#else
        result_.status="unavailable"; result_.reason="Linux perf_event_open required";
#endif
    }
    Pmu(const Pmu&)=delete;
    Pmu& operator=(const Pmu&)=delete;
    ~Pmu() {
#ifdef __linux__
        if(member_>=0) close(member_);
        if(leader_>=0) close(leader_);
#endif
    }
    void start() {
#ifdef __linux__
        if(result_.status!="ready") return;
        if(ioctl(leader_,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)<0 ||
           ioctl(leader_,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)<0) {
            result_.status="error"; result_.reason=std::strerror(errno);
        }
#endif
    }
    PmuResult stop() {
#ifdef __linux__
        if(result_.status!="ready") return result_;
        if(ioctl(leader_,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)<0) {
            result_.status="error"; result_.reason=std::strerror(errno); return result_;
        }
        struct Data { uint64_t nr,enabled,running,values[2]; } d{};
        if(read(leader_,&d,sizeof(d))!=static_cast<ssize_t>(sizeof(d)) || d.nr!=2) {
            result_.status="error"; result_.reason="short or invalid PMU group read"; return result_;
        }
        result_.enabled_ns=d.enabled; result_.running_ns=d.running;
        if(!d.running || !d.enabled) {
            result_.status="not_counted"; result_.reason="PMU group was never scheduled"; return result_;
        }
        for(int i=0;i<2;++i) {
            result_.raw[i]=static_cast<double>(d.values[i]);
            result_.scaled[i]=result_.raw[i]*static_cast<double>(d.enabled)/d.running;
        }
        result_.status=(static_cast<double>(d.running)/d.enabled>=0.90)?"ok":"low_coverage";
#endif
        return result_;
    }
};
