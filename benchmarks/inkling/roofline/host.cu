// (3) Host <-> device bandwidth: pinned H2D / D2H at 64 KiB .. host_gib, both directions at once, pageable H2D for
// reference (host-tier restores, adapter swaps), and P2P between the first two visible devices when there are two.
// Correctness: a pinned round trip H2D then D2H of the largest size reproduces the source bytes.
#include "common.cuh"

#include <cstring>

namespace rf {
namespace {

void host_case(const char* dir, std::size_t bytes, void* h, void* h2, void* d, void* d2, cudaStream_t s,
               cudaStream_t s2) {
    Timing t{};
    if (std::strcmp(dir, "h2d") == 0) {
        t = time_it(s, [&] { RF_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s)); });
    } else if (std::strcmp(dir, "d2h") == 0) {
        t = time_it(s, [&] { RF_CHECK(cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s)); });
    } else { // bidir: H2D on s and D2H on s2, joined; bytes counted per direction
        cudaEvent_t join{};
        RF_CHECK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
        t = time_it(s, [&] {
            RF_CHECK(cudaEventRecord(join, s));
            RF_CHECK(cudaStreamWaitEvent(s2, join, 0));
            RF_CHECK(cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s));
            RF_CHECK(cudaMemcpyAsync(h2, d2, bytes, cudaMemcpyDeviceToHost, s2));
            RF_CHECK(cudaEventRecord(join, s2));
            RF_CHECK(cudaStreamWaitEvent(s, join, 0));
        });
        RF_CHECK(cudaEventDestroy(join));
    }
    emit("host_bw", {kv("dir", dir), kv("memory", "pinned"), kv("bytes", bytes), kv("gbps", bytes / t.best_ns),
                     kv("gbps_median", bytes / t.median_ns), kv("us", t.best_ns / 1e3), kv("ok", true)});
    note("  host_bw %-6s pinned %10.3f MB  %7.2f GB/s  (%9.1f us)\n", dir, bytes / 1e6, bytes / t.best_ns,
         t.best_ns / 1e3);
}

} // namespace

void run_host() {
    const std::size_t max_bytes = static_cast<std::size_t>(options().host_gib * (1ull << 30)) / 4096 * 4096;
    cudaStream_t s{}, s2{};
    RF_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    RF_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    void *h = nullptr, *h2 = nullptr, *d = nullptr, *d2 = nullptr;
    RF_CHECK(cudaHostAlloc(&h, max_bytes, cudaHostAllocPortable));
    RF_CHECK(cudaHostAlloc(&h2, max_bytes, cudaHostAllocPortable));
    RF_CHECK(cudaMalloc(&d, max_bytes));
    RF_CHECK(cudaMalloc(&d2, max_bytes));
    for (std::size_t i = 0; i < max_bytes / 4; ++i) static_cast<uint32_t*>(h)[i] = hash32(static_cast<uint32_t>(i));
    std::memset(h2, 0, max_bytes);

    // correctness: round trip of the largest size
    RF_CHECK(cudaMemcpyAsync(d, h, max_bytes, cudaMemcpyHostToDevice, s));
    RF_CHECK(cudaMemcpyAsync(h2, d, max_bytes, cudaMemcpyDeviceToHost, s));
    RF_CHECK(cudaStreamSynchronize(s));
    const bool ok = std::memcmp(h, h2, max_bytes) == 0;
    emit("host_check", {kv("bytes", max_bytes), kv("ok", ok)});
    note("  host round trip %.0f MB: %s\n", max_bytes / 1e6, ok ? "ok" : "MISMATCH");

    std::vector<std::size_t> sizes = {64ull << 10, 1ull << 20, 4ull << 20, 16ull << 20, 64ull << 20, 256ull << 20};
    if (max_bytes > (256ull << 20)) sizes.push_back(max_bytes);
    for (const char* dir : {"h2d", "d2h", "bidir"})
        for (std::size_t b : sizes) host_case(dir, b, h, h2, d, d2, s, s2);

    // pageable reference (adapter files read into ordinary memory)
    {
        const std::size_t b = std::min<std::size_t>(max_bytes, 256ull << 20);
        std::vector<uint8_t> pageable(b, 1);
        const Timing t = time_it(s, [&] { RF_CHECK(cudaMemcpyAsync(d, pageable.data(), b, cudaMemcpyHostToDevice, s)); },
                                 1, options().trial_seconds, 3);
        emit("host_bw", {kv("dir", "h2d"), kv("memory", "pageable"), kv("bytes", b), kv("gbps", b / t.best_ns),
                         kv("gbps_median", b / t.median_ns), kv("us", t.best_ns / 1e3), kv("ok", true)});
        note("  host_bw h2d    pageable %8.3f MB  %7.2f GB/s\n", b / 1e6, b / t.best_ns);
    }

    // P2P (only when a second device is visible)
    int n_dev = 0;
    RF_CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) {
        emit("p2p", {kv("devices", n_dev), kv("measured", false), kv("reason", "single visible device"),
                     kv("ok", true)});
        note("  p2p: n/a (%d visible device)\n", n_dev);
    } else {
        int can01 = 0, can10 = 0;
        RF_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
        RF_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
        const std::size_t b = std::min<std::size_t>(max_bytes, 256ull << 20);
        void* d1            = nullptr;
        RF_CHECK(cudaSetDevice(1));
        RF_CHECK(cudaMalloc(&d1, b));
        if (can10) cudaDeviceEnablePeerAccess(0, 0);
        RF_CHECK(cudaSetDevice(0));
        if (can01) cudaDeviceEnablePeerAccess(1, 0);
        cudaGetLastError();
        const Timing t = time_it(s, [&] { RF_CHECK(cudaMemcpyPeerAsync(d1, 1, d, 0, b, s)); }, 1,
                                 options().trial_seconds, 3);
        emit("p2p", {kv("devices", n_dev), kv("measured", true), kv("peer_access", can01 && can10), kv("bytes", b),
                     kv("gbps", b / t.best_ns), kv("ok", true)});
        note("  p2p 0->1 (peer access %d/%d; else staged by the driver) %.2f GB/s\n", can01, can10, b / t.best_ns);
        RF_CHECK(cudaSetDevice(1));
        RF_CHECK(cudaFree(d1));
        RF_CHECK(cudaSetDevice(0));
    }

    RF_CHECK(cudaFreeHost(h));
    RF_CHECK(cudaFreeHost(h2));
    RF_CHECK(cudaFree(d));
    RF_CHECK(cudaFree(d2));
    RF_CHECK(cudaStreamDestroy(s));
    RF_CHECK(cudaStreamDestroy(s2));
}

} // namespace rf
