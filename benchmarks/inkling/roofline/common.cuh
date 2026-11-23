// Inkling-2-Mini roofline kit for one RTX 5090 (sm_120a). Shared helpers.
//
// Every benchmark prints ONE JSON object per line on stdout (machine-readable; perf/bounds.py reads them) and a
// human-readable line on stderr. Timings: CUDA events around a calibrated number of repetitions, several trials,
// best and median reported. Each benchmark checks its own result (checksums / dependency chains) and prints
// "ok": true|false.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define RF_CHECK(expr)                                                                             \
    do {                                                                                           \
        const cudaError_t rf_err__ = (expr);                                                       \
        if (rf_err__ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,            \
                         cudaGetErrorString(rf_err__), #expr);                                     \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace rf {

struct Options {
    double trial_seconds = 0.2; // target length of one timed trial
    int trials           = 5;
    double dram_gib      = 0.0; // 0: min(4 GiB, 35 % of free)
    double host_gib      = 1.0; // largest host copy
    bool quick           = false;
};

inline Options& options() {
    static Options o;
    return o;
}

// One JSON line. Fields are passed preformatted as "key":value pairs (callers build them with kv()).
inline std::string kv(const char* key, double value) {
    char buf[128];
    if (std::isfinite(value)) {
        std::snprintf(buf, sizeof(buf), "\"%s\":%.6g", key, value);
    } else {
        std::snprintf(buf, sizeof(buf), "\"%s\":null", key);
    }
    return buf;
}
inline std::string kv(const char* key, long long value) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "\"%s\":%lld", key, value);
    return buf;
}
inline std::string kv(const char* key, int value) { return kv(key, static_cast<long long>(value)); }
inline std::string kv(const char* key, std::size_t value) {
    return kv(key, static_cast<long long>(value));
}
inline std::string kv(const char* key, bool value) {
    return std::string("\"") + key + "\":" + (value ? "true" : "false");
}
inline std::string kv(const char* key, const char* value) {
    return std::string("\"") + key + "\":\"" + value + "\"";
}
inline std::string kv(const char* key, const std::string& value) { return kv(key, value.c_str()); }

inline void emit(const char* bench, std::initializer_list<std::string> fields) {
    std::string line = std::string("{\"bench\":\"") + bench + "\"";
    for (const std::string& f : fields) { line += "," + f; }
    line += "}";
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

inline void note(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

class Event {
public:
    Event() { RF_CHECK(cudaEventCreate(&e_)); }
    ~Event() { cudaEventDestroy(e_); }
    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const { return e_; }

private:
    cudaEvent_t e_{};
};

template <typename Launch> double elapsed_ms(cudaStream_t stream, int reps, Launch&& launch) {
    Event a, b;
    RF_CHECK(cudaEventRecord(a.get(), stream));
    for (int i = 0; i < reps; ++i) { launch(); }
    RF_CHECK(cudaEventRecord(b.get(), stream));
    RF_CHECK(cudaEventSynchronize(b.get()));
    float ms = 0.f;
    RF_CHECK(cudaEventElapsedTime(&ms, a.get(), b.get()));
    return static_cast<double>(ms);
}

struct Timing {
    double best_ns   = 0; // per launch() call
    double median_ns = 0;
    int reps         = 0;
};

// Warm up, calibrate reps to ~trial_seconds, then time `trials` trials. Returns ns per launch() call.
template <typename Launch>
Timing time_it(cudaStream_t stream, Launch&& launch, int warmup = 3, double trial_seconds = -1,
               int trials = -1) {
    if (trial_seconds <= 0) trial_seconds = options().trial_seconds;
    if (trials <= 0) trials = options().trials;
    for (int i = 0; i < warmup; ++i) launch();
    RF_CHECK(cudaStreamSynchronize(stream));
    double cal = elapsed_ms(stream, 1, launch);
    if (cal < 0.05) cal = elapsed_ms(stream, 16, launch) / 16.0;
    int reps = std::max(1, static_cast<int>(std::ceil(trial_seconds * 1e3 / std::max(cal, 1e-4))));
    reps     = std::min(reps, 1 << 20);
    std::vector<double> ns;
    for (int t = 0; t < trials; ++t) ns.push_back(elapsed_ms(stream, reps, launch) * 1e6 / reps);
    std::sort(ns.begin(), ns.end());
    RF_CHECK(cudaGetLastError());
    return Timing{ns.front(), ns[ns.size() / 2], reps};
}

inline cudaDeviceProp device_props() {
    int dev = 0;
    RF_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p{};
    RF_CHECK(cudaGetDeviceProperties(&p, dev));
    return p;
}

// 32-bit hash for deterministic fills (device and host give the same values).
__host__ __device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

void run_dram();
void run_ctas();
void run_mma();
void run_host();
void run_gap();

} // namespace rf
