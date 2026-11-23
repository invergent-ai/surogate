// (1) DRAM bandwidth: streaming read / write / copy over a large buffer, and "weight streaming": a sweep over a
// 0.5-4 GiB buffer in fixed chunks, ONE kernel per chunk (as one GEMV / MoE kernel reads its weights), launched
// (a) on a stream, (b) as a CUDA graph, (c) as a CUDA graph with programmatic dependent launch (each kernel triggers
// its dependents at entry, reads its chunk, then waits for its predecessor before its single dependent store: the
// "weights do not depend on the previous kernel" prefetch pattern the attention / MoE kernels plan).
//
// Correctness: the buffer holds word[i] = i * 2654435761 (mod 2^32); every read kernel adds its words into one
// uint32 counter; the host compares with the closed form.
#include "common.cuh"

namespace rf {
namespace {

constexpr int kThreads      = 256;
constexpr uint32_t kMult    = 2654435761u;
constexpr int kMaxSweepKern = 2048;

__global__ void fill_kernel(uint32_t* __restrict__ p, std::size_t n_words) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words;
         i += stride) {
        p[i] = static_cast<uint32_t>(i) * kMult;
    }
}

__device__ __forceinline__ uint32_t sum4(const uint4 v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ uint32_t block_sum(uint32_t v) {
    __shared__ uint32_t warp_sums[kThreads / 32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    uint32_t s = 0;
    if (threadIdx.x == 0) {
        for (int w = 0; w < kThreads / 32; ++w) s += warp_sums[w];
    }
    return s;
}

// Reads n_vec uint4 starting at src with U independent 16-byte loads in flight per thread per iteration.
// PDL: trigger dependents at entry; wait for the predecessor only before the dependent store.
template <int U, bool PDL>
__global__ void __launch_bounds__(kThreads)
    read_chunk_kernel(const uint4* __restrict__ src, std::size_t n_vec, uint32_t* __restrict__ acc) {
#if __CUDA_ARCH__ >= 900
    if (PDL) cudaTriggerProgrammaticLaunchCompletion();
#endif
    const std::size_t tid    = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    uint32_t s               = 0;
    std::size_t i            = tid;
    for (; i + (U - 1) * stride < n_vec; i += U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) v[u] = __ldcs(src + i + u * stride);
#pragma unroll
        for (int u = 0; u < U; ++u) s += sum4(v[u]);
    }
    for (; i < n_vec; i += stride) s += sum4(__ldcs(src + i));
    s = block_sum(s);
#if __CUDA_ARCH__ >= 900
    if (PDL) cudaGridDependencySynchronize();
#endif
    if (threadIdx.x == 0) atomicAdd(acc, s);
}

__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, std::size_t n_vec) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t i            = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n_vec; i += 4 * stride) {
        const uint4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
        dst[i] = a, dst[i + stride] = b, dst[i + 2 * stride] = c, dst[i + 3 * stride] = d;
    }
    for (; i < n_vec; i += stride) dst[i] = src[i];
}

__global__ void write_kernel(uint4* __restrict__ dst, std::size_t n_vec) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const uint4 v            = make_uint4(1u, 2u, 3u, 4u);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
         i += stride)
        dst[i] = v;
}

// Closed-form sum of word[i] = i * kMult over words [w0, w1), mod 2^32.
uint32_t expected_sum(std::size_t w0, std::size_t w1) {
    const unsigned __int128 a = w0, b = w1;
    const unsigned __int128 s = (a + b - 1) * (b - a) / 2; // sum of i
    return static_cast<uint32_t>(static_cast<uint64_t>(s)) * kMult;
}

template <typename K> int resident_blocks(K kernel, int sms) {
    int per_sm = 0;
    RF_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    return std::max(1, per_sm) * sms;
}

template <int U, bool PDL>
void launch_read(const uint4* src, std::size_t n_vec, uint32_t* acc, int grid, cudaStream_t s) {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim          = dim3(grid);
    cfg.blockDim         = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream           = s;
    cudaLaunchAttribute attr{};
    attr.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    if (PDL) {
        cfg.attrs    = &attr;
        cfg.numAttrs = 1;
    }
    RF_CHECK(cudaLaunchKernelEx(&cfg, read_chunk_kernel<U, PDL>, src, n_vec, acc));
}

uint32_t read_acc(uint32_t* d_acc, cudaStream_t s) {
    uint32_t h = 0;
    RF_CHECK(cudaMemcpyAsync(&h, d_acc, 4, cudaMemcpyDeviceToHost, s));
    RF_CHECK(cudaStreamSynchronize(s));
    return h;
}

template <int U, bool PDL>
void chunk_case(const char* mode, const uint4* buf, std::size_t buf_bytes, std::size_t chunk_bytes,
                uint32_t* d_acc, int sms, cudaStream_t s, const char* label) {
    const std::size_t chunk_vec = chunk_bytes / 16;
    std::size_t n_chunks        = std::min<std::size_t>(buf_bytes / chunk_bytes, kMaxSweepKern);
    const int max_grid          = resident_blocks(read_chunk_kernel<U, PDL>, sms);
    const int grid              = static_cast<int>(std::min<std::size_t>(
        std::max<std::size_t>(1, (chunk_vec + kThreads * U - 1) / (kThreads * U)), max_grid));
    const bool use_graph        = std::string(mode) != "stream";

    auto sweep = [&] {
        for (std::size_t k = 0; k < n_chunks; ++k)
            launch_read<U, PDL>(buf + k * chunk_vec, chunk_vec, d_acc, grid, s);
    };
    cudaGraphExec_t exec = nullptr;
    if (use_graph) {
        cudaGraph_t graph = nullptr;
        RF_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        sweep();
        RF_CHECK(cudaStreamEndCapture(s, &graph));
        RF_CHECK(cudaGraphInstantiate(&exec, graph, 0));
        RF_CHECK(cudaGraphDestroy(graph));
    }
    auto run = [&] {
        if (use_graph) {
            RF_CHECK(cudaGraphLaunch(exec, s));
        } else {
            sweep();
        }
    };
    // correctness: one sweep from a zero counter
    RF_CHECK(cudaMemsetAsync(d_acc, 0, 4, s));
    run();
    const uint32_t got  = read_acc(d_acc, s);
    const uint32_t want = expected_sum(0, n_chunks * chunk_bytes / 4);
    const Timing t      = time_it(s, run, 2, options().trial_seconds * 0.5, 3);
    const double per_k  = t.best_ns / static_cast<double>(n_chunks);
    const double gbps   = static_cast<double>(chunk_bytes) / per_k;
    const double med    = static_cast<double>(chunk_bytes) / (t.median_ns / n_chunks);
    emit("dram_chunk", {kv("mode", mode), kv("label", label), kv("chunk_bytes", chunk_bytes),
                        kv("chunk_mb", chunk_bytes / 1e6), kv("loads_in_flight", U), kv("grid", grid),
                        kv("kernels", n_chunks), kv("ns_per_kernel", per_k), kv("gbps", gbps),
                        kv("gbps_median", med), kv("ok", got == want)});
    note("  dram_chunk %-11s %-16s %8.3f MB U=%d grid=%5d  %8.2f us/kernel  %7.1f GB/s  %s\n", mode, label,
         chunk_bytes / 1e6, U, grid, per_k / 1e3, gbps, got == want ? "ok" : "CHECKSUM MISMATCH");
    if (exec) RF_CHECK(cudaGraphExecDestroy(exec));
}

} // namespace

// Bandwidth reachable by a FIXED number of CTAs (one per SM up to the SM count), each with U x 16 B x 256 threads
// of loads in flight, streaming a 2 GiB buffer: what a decode kernel with few CTAs (e.g. a fixed split size at short
// context, one CTA per (stream, KV head) without split-KV) can pull.
void run_ctas() {
    const cudaDeviceProp p   = device_props();
    const std::size_t bytes  = 2ull << 30;
    const std::size_t n_vec  = bytes / 16;
    uint4* src               = nullptr;
    uint32_t* d_acc          = nullptr;
    RF_CHECK(cudaMalloc(&src, bytes));
    RF_CHECK(cudaMalloc(&d_acc, 64));
    cudaStream_t s{};
    RF_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    fill_kernel<<<p.multiProcessorCount * 8, kThreads, 0, s>>>(reinterpret_cast<uint32_t*>(src), bytes / 4);
    for (int i = 0; i < 16; ++i)
        read_chunk_kernel<8, false><<<p.multiProcessorCount * 4, kThreads, 0, s>>>(src, n_vec, d_acc);
    RF_CHECK(cudaStreamSynchronize(s));
    const int counts[] = {8, 16, 24, 32, 48, 64, 96, 128, 170, 256, 340, 680};
    auto one = [&](auto kernel, int U, int n) {
        auto run = [&] { kernel<<<n, kThreads, 0, s>>>(src, n_vec, d_acc); };
        RF_CHECK(cudaMemsetAsync(d_acc, 0, 4, s));
        run();
        const bool ok  = read_acc(d_acc, s) == expected_sum(0, bytes / 4);
        const Timing t = time_it(s, run, 1, options().trial_seconds * 0.5, 3);
        const double g = bytes / t.best_ns;
        emit("dram_ctas", {kv("ctas", n), kv("loads_in_flight", U), kv("bytes_in_flight_per_cta", U * 16 * kThreads),
                           kv("gbps", g), kv("gbps_per_cta", g / n), kv("ok", ok)});
        note("  dram_ctas %4d CTAs U=%2d (%3d KB in flight/CTA)  %8.1f GB/s  %6.2f GB/s per CTA  %s\n", n, U,
             U * 16 * kThreads / 1024, g, g / n, ok ? "ok" : "CHECKSUM MISMATCH");
    };
    for (int n : counts) {
        one(read_chunk_kernel<8, false>, 8, n);
        one(read_chunk_kernel<16, false>, 16, n);
    }
    RF_CHECK(cudaStreamDestroy(s));
    RF_CHECK(cudaFree(src));
    RF_CHECK(cudaFree(d_acc));
}

void run_dram() {
    const cudaDeviceProp p = device_props();
    std::size_t free_b = 0, total_b = 0;
    RF_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::size_t bytes =
        options().dram_gib > 0 ? static_cast<std::size_t>(options().dram_gib * (1ull << 30))
                               : std::min<std::size_t>(4ull << 30, static_cast<std::size_t>(free_b * 0.35));
    bytes = bytes / (64ull << 20) * (64ull << 20);
    if (bytes < 8ull * p.l2CacheSize || 2 * bytes > free_b) {
        note("dram: not enough free memory (free %.2f GiB, want 2 x %.2f GiB >= 8 x L2)\n", free_b / 1073741824.0,
             bytes / 1073741824.0);
        emit("dram_skip", {kv("free_bytes", free_b), kv("ok", false)});
        return;
    }
    const std::size_t n_vec = bytes / 16;
    uint4 *src = nullptr, *dst = nullptr;
    uint32_t* d_acc = nullptr;
    RF_CHECK(cudaMalloc(&src, bytes));
    RF_CHECK(cudaMalloc(&dst, bytes));
    RF_CHECK(cudaMalloc(&d_acc, 64));
    cudaStream_t s{};
    RF_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    fill_kernel<<<p.multiProcessorCount * 8, kThreads, 0, s>>>(reinterpret_cast<uint32_t*>(src), bytes / 4);
    RF_CHECK(cudaGetLastError());
    // precondition clocks: ~1 s of copies
    for (int i = 0; i < 64; ++i) copy_kernel<<<p.multiProcessorCount * 4, kThreads, 0, s>>>(src, dst, n_vec);
    RF_CHECK(cudaStreamSynchronize(s));

    emit("dram_info", {kv("buffer_bytes", bytes), kv("l2_bytes", p.l2CacheSize), kv("sms", p.multiProcessorCount),
                       kv("bus_bits", p.memoryBusWidth), kv("free_bytes", free_b), kv("total_bytes", total_b)});

    // ---- streaming, whole buffer ----
    auto stream_read = [&](auto kernel, const char* name, int U) {
        const int grid = resident_blocks(kernel, p.multiProcessorCount);
        auto run       = [&] { kernel<<<grid, kThreads, 0, s>>>(src, n_vec, d_acc); };
        RF_CHECK(cudaMemsetAsync(d_acc, 0, 4, s));
        run();
        const bool ok  = read_acc(d_acc, s) == expected_sum(0, bytes / 4);
        const Timing t = time_it(s, run);
        emit("dram_stream", {kv("op", "read"), kv("variant", name), kv("loads_in_flight", U), kv("grid", grid),
                             kv("bytes", bytes), kv("gbps", bytes / t.best_ns),
                             kv("gbps_median", bytes / t.median_ns), kv("ok", ok)});
        note("  dram_stream read  %-12s grid=%5d %8.1f GB/s (median %8.1f)  %s\n", name, grid, bytes / t.best_ns,
             bytes / t.median_ns, ok ? "ok" : "CHECKSUM MISMATCH");
    };
    stream_read(read_chunk_kernel<1, false>, "u1", 1);
    stream_read(read_chunk_kernel<4, false>, "u4", 4);
    stream_read(read_chunk_kernel<8, false>, "u8", 8);
    {
        const int grid = p.multiProcessorCount * 4;
        auto run       = [&] { write_kernel<<<grid, kThreads, 0, s>>>(dst, n_vec); };
        const Timing t = time_it(s, run);
        emit("dram_stream", {kv("op", "write"), kv("variant", "kernel"), kv("bytes", bytes),
                             kv("gbps", bytes / t.best_ns), kv("gbps_median", bytes / t.median_ns),
                             kv("ok", true)});
        note("  dram_stream write kernel        %8.1f GB/s\n", bytes / t.best_ns);
    }
    {
        const int grid = p.multiProcessorCount * 4;
        auto run       = [&] { copy_kernel<<<grid, kThreads, 0, s>>>(src, dst, n_vec); };
        const Timing t = time_it(s, run);
        emit("dram_stream", {kv("op", "copy"), kv("variant", "kernel_x4"), kv("bytes", bytes),
                             kv("gbps", 2.0 * bytes / t.best_ns), kv("gbps_median", 2.0 * bytes / t.median_ns),
                             kv("ok", true)});
        note("  dram_stream copy  kernel (bus)  %8.1f GB/s\n", 2.0 * bytes / t.best_ns);
    }
    {
        auto run = [&] { RF_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s)); };
        const Timing t = time_it(s, run);
        emit("dram_stream", {kv("op", "copy"), kv("variant", "cudaMemcpyD2D"), kv("bytes", bytes),
                             kv("gbps", 2.0 * bytes / t.best_ns), kv("gbps_median", 2.0 * bytes / t.median_ns),
                             kv("ok", true)});
        note("  dram_stream copy  memcpy (bus)  %8.1f GB/s\n", 2.0 * bytes / t.best_ns);
    }

    // ---- weight streaming: one kernel per chunk ----
    struct Chunk {
        double mb;
        const char* label;
    };
    const Chunk chunks[] = {
        {0.25, "generic"},        {0.59, "router_visit"},   {1.0, "generic"},        {2.0, "generic"},
        {2.10, "local_kv_visit"}, {4.0, "generic"},         {4.48, "nvfp4_expert"},  {7.08, "wo_fp8"},
        {8.21, "mxfp8_expert"},   {10.0, "generic"},        {19.81, "qkvr_fp8"},     {36.4, "moe8_nvfp4"},
        {53.8, "attn_bf16_visit"}, {66.3, "moe8_mxfp8"},    {134.0, "global_kv_64k"},
    };
    const bool quick = options().quick;
    for (const Chunk& c : chunks) {
        if (quick && std::string(c.label) == "generic" && c.mb != 1.0 && c.mb != 4.0) continue;
        std::size_t cb = static_cast<std::size_t>(c.mb * 1e6);
        cb             = (cb + 4095) / 4096 * 4096; // whole uint4 x 256 threads
        if (cb * 4 > bytes) continue;
        chunk_case<1, false>("stream", src, bytes, cb, d_acc, p.multiProcessorCount, s, c.label);
        chunk_case<1, false>("graph", src, bytes, cb, d_acc, p.multiProcessorCount, s, c.label);
        chunk_case<4, false>("graph", src, bytes, cb, d_acc, p.multiProcessorCount, s, c.label);
        chunk_case<1, true>("graph_pdl", src, bytes, cb, d_acc, p.multiProcessorCount, s, c.label);
        chunk_case<4, true>("graph_pdl", src, bytes, cb, d_acc, p.multiProcessorCount, s, c.label);
    }

    RF_CHECK(cudaStreamDestroy(s));
    RF_CHECK(cudaFree(src));
    RF_CHECK(cudaFree(dst));
    RF_CHECK(cudaFree(d_acc));
}

} // namespace rf
