// ENGINE-INKLING lane h: day-1 roofline kit (RTX 5090, sm_120a). One binary, four parts:
//   dram  (1) DRAM streaming read / write / copy + one-kernel-per-chunk weight streaming (stream / graph / graph+PDL)
//   mma   (2) dense MMA peaks: mma.sync instruction ceilings (BF16, FP8 e4m3, NVFP4) + cuBLASLt GEMMs
//   host  (3) pinned H2D / D2H / bidirectional, pageable H2D, P2P if two devices are visible
//   gap   (4) per-kernel gap of a 400-kernel dependent chain: stream vs CUDA graph, without / with PDL
// Usage: inkling_roofline [--only dram,mma,host,gap] [--quick] [--trial-seconds S] [--trials N] [--dram-gib G]
//                         [--host-gib G]
// stdout: JSON lines (perf/bounds.py --measured <file> reads them); stderr: human-readable progress.
#include "common.cuh"

#include <cstring>
#include <set>
#include <sstream>

int main(int argc, char** argv) {
    std::set<std::string> parts = {"dram", "mma", "host", "gap"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value after %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--only") {
            parts.clear();
            std::stringstream ss(next());
            std::string item;
            while (std::getline(ss, item, ',')) parts.insert(item);
        } else if (a == "--quick") {
            rf::options().quick = true;
        } else if (a == "--trial-seconds") {
            rf::options().trial_seconds = std::atof(next());
        } else if (a == "--trials") {
            rf::options().trials = std::atoi(next());
        } else if (a == "--dram-gib") {
            rf::options().dram_gib = std::atof(next());
        } else if (a == "--host-gib") {
            rf::options().host_gib = std::atof(next());
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    const cudaDeviceProp p = rf::device_props();
    int drv = 0, rt = 0;
    RF_CHECK(cudaDriverGetVersion(&drv));
    RF_CHECK(cudaRuntimeGetVersion(&rt));
    std::size_t free_b = 0, total_b = 0;
    RF_CHECK(cudaMemGetInfo(&free_b, &total_b));
    int clk_khz = 0, mem_khz = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
    rf::emit("device", {rf::kv("name", p.name), rf::kv("cc", std::to_string(p.major) + "." + std::to_string(p.minor)),
                        rf::kv("sms", p.multiProcessorCount), rf::kv("l2_bytes", p.l2CacheSize),
                        rf::kv("bus_bits", p.memoryBusWidth), rf::kv("max_sm_clock_khz", clk_khz),
                        rf::kv("mem_clock_khz", mem_khz), rf::kv("smem_optin_bytes", p.sharedMemPerBlockOptin),
                        rf::kv("total_bytes", total_b), rf::kv("free_bytes_at_start", free_b),
                        rf::kv("driver", drv), rf::kv("runtime", rt), rf::kv("ok", true)});
    rf::note("%s cc %d.%d, %d SMs, L2 %.0f MiB, %.2f / %.2f GiB free, driver %d runtime %d\n", p.name, p.major,
             p.minor, p.multiProcessorCount, p.l2CacheSize / 1048576.0, free_b / 1073741824.0,
             total_b / 1073741824.0, drv, rt);
    if (parts.count("gap")) rf::run_gap();
    if (parts.count("dram")) rf::run_dram();
    if (parts.count("host")) rf::run_host();
    if (parts.count("mma")) rf::run_mma();
    rf::emit("done", {rf::kv("ok", true)});
    return 0;
}
