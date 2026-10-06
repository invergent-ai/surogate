// Block-FP8 routed experts on Hopper's grouped GEMM (fp8_moe_sm90), at Qwen3.6-35B-A3B's shape.
//
// Every tile family (and the GEMV path, on rounds narrow enough for it) runs uniform, skewed (a
// few experts take every row, most are empty) and near-empty (one token) routings, with the gate/up scratch aliasing the output as the prefill
// family passes it, against an FP32 reference that quantises its activations with the same rule
// (E4M3 per row per 128, scale amax / 448) but its own serial code. One round is also captured
// in a CUDA graph and replayed at a GEMV width and a GEMM width, which must reproduce the direct
// launch bit for bit.
//
// SUROGATE_FP8_MOE_BENCH=1 adds a timing sweep over round widths and tile families: the numbers
// the automatic tile choice in fp8_moe_sm90.cu was set from.

#include "ops/sparse_moe/fp8_sm90/fp8_moe_sm90.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fp8 = sinfer::ops::detail::fp8_moe_sm90;

#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        const cudaError_t status_ = (call);                                                        \
        if (status_ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call,                      \
                         cudaGetErrorString(status_));                                             \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int kHidden  = 2048;
constexpr int kExperts = 256;
constexpr int kTopK    = 8;
constexpr int kInter   = 512;
constexpr int kBlock   = 128;

const fp8::Geometry kGeometry{kHidden, kExperts, kTopK, kInter, fp8::Activation::Swiglu};

template <class T>
struct DeviceBuffer {
    T* data           = nullptr;
    std::size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) : count(n) {
        CHECK_CUDA(cudaMalloc(&data, std::max<std::size_t>(n, 1) * sizeof(T)));
    }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : data(o.data), count(o.count) { o.data = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        std::swap(data, o.data);
        std::swap(count, o.count);
        return *this;
    }
    ~DeviceBuffer() {
        if (data != nullptr) { cudaFree(data); }
    }
    void upload(const std::vector<T>& host) {
        count = host.size();
        CHECK_CUDA(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(std::size_t n) const {
        std::vector<T> host(n);
        CHECK_CUDA(cudaMemcpy(host.data(), data, n * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
};

__device__ __forceinline__ std::uint32_t mix(std::uint64_t i, std::uint32_t seed) {
    std::uint64_t z = i * 0x9E3779B97F4A7C15ull + seed;
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return static_cast<std::uint32_t>(z ^ (z >> 31));
}

// Codes of uniform values in [-2, 2): every finite E4M3 magnitude below 2, never a NaN pattern.
__global__ void fill_codes_kernel(std::uint8_t* codes, std::size_t n, std::uint32_t seed) {
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float u = static_cast<float>(mix(i, seed) >> 8) * (1.0f / 16777216.0f);
        codes[i]      = __nv_cvt_float_to_fp8(4.0f * u - 2.0f, __NV_SATFINITE, __NV_E4M3);
    }
}

__device__ __forceinline__ float decode(std::uint8_t code) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw(code, __NV_E4M3);
    return __half2float(__half(h));
}

// What the reference does to an activation plane before a GEMM: the GEMM tiles' E4M3 rule, or
// the GEMV path's (the gathered input as it is, the SwiGLU output rounded through BF16).
enum class Plane { Quantize, Copy, RoundBf16 };

// The reference quantiser: one thread per (row, block), serial over the block. `source` row r is
// `rows_of[r]` when given (the gather), else r.
__global__ void ref_quantize_kernel(const float* source, int width, const int* rows_of, int rows,
                                    float* dequantized, Plane plane) {
    const int item    = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int kblocks = width / kBlock;
    if (item >= rows * kblocks) { return; }
    const int row   = item / kblocks;
    const int block = item % kblocks;
    const float* in =
        source + static_cast<std::int64_t>(rows_of ? rows_of[row] : row) * width + block * kBlock;
    if (plane != Plane::Quantize) {
        float* out = dequantized + static_cast<std::int64_t>(row) * width + block * kBlock;
        for (int i = 0; i < kBlock; ++i) {
            out[i] = plane == Plane::Copy ? in[i] : __bfloat162float(__float2bfloat16(in[i]));
        }
        return;
    }
    float amax = 0.0f;
    for (int i = 0; i < kBlock; ++i) { amax = fmaxf(amax, fabsf(in[i])); }
    const float scale   = amax > 0.0f ? amax / 448.0f : 1.0f;
    const float inverse = 1.0f / scale;
    float* out          = dequantized + static_cast<std::int64_t>(row) * width + block * kBlock;
    for (int i = 0; i < kBlock; ++i) {
        out[i] = decode(__nv_cvt_float_to_fp8(in[i] * inverse, __NV_SATFINITE, __NV_E4M3)) * scale;
    }
}

// out[c][j] = sum_k a[c][k] * w[e][j][k] * s[e][j / 128][k / 128], e = column_expert[c]; FP32.
// With `bf16_round` the result is rounded through BF16, as the module's GEMM writes it.
__global__ void ref_gemm_kernel(const float* a, const int* column_expert, int columns, int n, int k,
                                const std::uint8_t* codes, const float* scales, float* out,
                                bool bf16_round) {
    const int c = static_cast<int>(blockIdx.x);
    const int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (c >= columns || j >= n) { return; }
    const int e           = column_expert[c];
    const int kblocks     = k / kBlock;
    const float* row      = a + static_cast<std::int64_t>(c) * k;
    const std::uint8_t* w = codes + (static_cast<std::int64_t>(e) * n + j) * k;
    const float* s        = scales + (static_cast<std::int64_t>(e) * (n / kBlock) + j / kBlock) * kblocks;
    float acc             = 0.0f;
    for (int kb = 0; kb < kblocks; ++kb) {
        float partial = 0.0f;
        for (int i = 0; i < kBlock; ++i) { partial += row[kb * kBlock + i] * decode(w[kb * kBlock + i]); }
        acc += partial * s[kb];
    }
    out[static_cast<std::int64_t>(c) * n + j] =
        bf16_round ? __bfloat162float(__float2bfloat16(acc)) : acc;
}

__global__ void ref_swiglu_kernel(const float* gate_up, int inter, int rows, float* out) {
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::int64_t>(rows) * inter) { return; }
    const std::int64_t row = i / inter;
    const int col          = static_cast<int>(i % inter);
    const float g          = gate_up[row * 2 * inter + col];
    const float u          = gate_up[row * 2 * inter + inter + col];
    out[i]                 = g / (1.0f + expf(-g)) * u;
}

__global__ void bf16_to_float_kernel(const __nv_bfloat16* in, std::size_t n, float* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) { out[i] = __bfloat162float(in[i]); }
}

struct Experts {
    DeviceBuffer<std::uint8_t> gate_up_codes;
    DeviceBuffer<float> gate_up_scales;
    DeviceBuffer<std::uint8_t> down_codes;
    DeviceBuffer<float> down_scales;

    fp8::Fp8RoutedExperts view() const {
        return {gate_up_codes.data, gate_up_scales.data, down_codes.data, down_scales.data};
    }
};

Experts make_experts(std::mt19937& rng) {
    Experts w;
    const std::size_t gate_up = static_cast<std::size_t>(kExperts) * 2 * kInter * kHidden;
    const std::size_t down    = static_cast<std::size_t>(kExperts) * kHidden * kInter;
    w.gate_up_codes           = DeviceBuffer<std::uint8_t>(gate_up);
    w.down_codes              = DeviceBuffer<std::uint8_t>(down);
    fill_codes_kernel<<<4096, 256>>>(w.gate_up_codes.data, gate_up, 17u);
    fill_codes_kernel<<<4096, 256>>>(w.down_codes.data, down, 29u);
    CHECK_CUDA(cudaGetLastError());
    std::uniform_real_distribution<float> scale(0.5e-2f, 1.5e-2f);
    std::vector<float> gs(static_cast<std::size_t>(kExperts) * (2 * kInter / kBlock) * (kHidden / kBlock));
    std::vector<float> ds(static_cast<std::size_t>(kExperts) * (kHidden / kBlock) * (kInter / kBlock));
    for (float& v : gs) { v = scale(rng); }
    for (float& v : ds) { v = 4.0f * scale(rng); }
    w.gate_up_scales = DeviceBuffer<float>(gs.size());
    w.gate_up_scales.upload(gs);
    w.down_scales = DeviceBuffer<float>(ds.size());
    w.down_scales.upload(ds);
    return w;
}

enum class Mode { Uniform, Skewed };

struct Round {
    int tokens = 0;
    std::vector<int> column_token;
    std::vector<int> column_expert;
    std::vector<int> offsets;
    std::vector<__nv_bfloat16> x;
};

Round make_round(int tokens, Mode mode, std::mt19937& rng) {
    Round r;
    r.tokens = tokens;
    std::vector<int> choice(static_cast<std::size_t>(tokens) * kTopK);
    // Skewed rounds draw from the first 12 experts: a handful of experts take every row, in
    // several tiles each, and the other 244 are empty.
    const int pool = mode == Mode::Skewed ? 12 : kExperts;
    std::vector<int> ids(pool);
    for (int t = 0; t < tokens; ++t) {
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int k = 0; k < kTopK; ++k) { choice[static_cast<std::size_t>(t) * kTopK + k] = ids[k]; }
    }
    std::vector<int> counts(kExperts, 0);
    for (int e : choice) { ++counts[e]; }
    r.offsets.assign(kExperts + 1, 0);
    for (int e = 0; e < kExperts; ++e) { r.offsets[e + 1] = r.offsets[e] + counts[e]; }
    std::vector<int> cursor(r.offsets.begin(), r.offsets.end() - 1);
    r.column_token.assign(choice.size(), -1);
    r.column_expert.assign(choice.size(), -1);
    for (std::size_t a = 0; a < choice.size(); ++a) {
        const int column        = cursor[choice[a]]++;
        r.column_token[column]  = static_cast<int>(a / kTopK);
        r.column_expert[column] = choice[a];
    }
    std::normal_distribution<float> normal(0.0f, 1.0f);
    r.x.resize(static_cast<std::size_t>(tokens) * kHidden);
    for (auto& v : r.x) { v = __float2bfloat16(normal(rng)); }
    return r;
}

struct DeviceRound {
    DeviceBuffer<int> column_token;
    DeviceBuffer<int> column_expert;
    DeviceBuffer<int> offsets;
    DeviceBuffer<__nv_bfloat16> x;
    explicit DeviceRound(const Round& r)
        : column_token(r.column_token.size()), column_expert(r.column_expert.size()),
          offsets(r.offsets.size()), x(r.x.size()) {
        column_token.upload(r.column_token);
        column_expert.upload(r.column_expert);
        offsets.upload(r.offsets);
        x.upload(r.x);
    }
};

/// The FP32 reference for the GEMM tiles (`gemv` false) or for the GEMV path (`gemv` true).
std::vector<float> reference(const Round& r, const DeviceRound& d, const Experts& w, bool gemv) {
    const int columns = static_cast<int>(r.column_token.size());
    DeviceBuffer<float> xf(r.x.size());
    bf16_to_float_kernel<<<static_cast<unsigned>((r.x.size() + 255) / 256), 256>>>(d.x.data, r.x.size(), xf.data);
    DeviceBuffer<float> xq(static_cast<std::size_t>(columns) * kHidden);
    ref_quantize_kernel<<<(columns * (kHidden / kBlock) + 127) / 128, 128>>>(
        xf.data, kHidden, d.column_token.data, columns, xq.data, gemv ? Plane::Copy : Plane::Quantize);
    DeviceBuffer<float> gate_up(static_cast<std::size_t>(columns) * 2 * kInter);
    ref_gemm_kernel<<<dim3(columns, 2 * kInter / 256), 256>>>(
        xq.data, d.column_expert.data, columns, 2 * kInter, kHidden, w.gate_up_codes.data,
        w.gate_up_scales.data, gate_up.data, !gemv);
    DeviceBuffer<float> mid(static_cast<std::size_t>(columns) * kInter);
    ref_swiglu_kernel<<<(columns * kInter + 255) / 256, 256>>>(gate_up.data, kInter, columns, mid.data);
    DeviceBuffer<float> midq(static_cast<std::size_t>(columns) * kInter);
    ref_quantize_kernel<<<(columns * (kInter / kBlock) + 127) / 128, 128>>>(
        mid.data, kInter, nullptr, columns, midq.data, gemv ? Plane::RoundBf16 : Plane::Quantize);
    DeviceBuffer<float> out(static_cast<std::size_t>(columns) * kHidden);
    ref_gemm_kernel<<<dim3(columns, kHidden / 256), 256>>>(midq.data, d.column_expert.data, columns,
                                                           kHidden, kInter, w.down_codes.data,
                                                           w.down_scales.data, out.data, false);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return out.download(static_cast<std::size_t>(columns) * kHidden);
}

struct Workspace {
    DeviceBuffer<char> scratch;
    std::size_t bytes = 0;
    explicit Workspace(int max_tokens) : bytes(fp8::workspace_bytes(kGeometry, max_tokens)) {
        scratch = DeviceBuffer<char>(bytes);
    }
};

// The prefill family's [assignments][hidden] block, aliased as the gate/up scratch.
DeviceBuffer<__nv_bfloat16> output_block(int tokens) {
    const std::size_t rows = static_cast<std::size_t>(tokens) * kTopK;
    return DeviceBuffer<__nv_bfloat16>(rows * std::max(kHidden, 2 * kInter));
}

const char* tile_name(fp8::Tile tile) {
    switch (tile) {
    case fp8::Tile::Auto: return "auto";
    case fp8::Tile::Swap16: return "swap16";
    case fp8::Tile::Swap32: return "swap32";
    case fp8::Tile::Swap64: return "swap64";
    case fp8::Tile::Wide128: return "wide128";
    case fp8::Tile::Gemv: return "gemv";
    }
    return "?";
}

bool compare(const char* label, const std::vector<float>& ref, const std::vector<__nv_bfloat16>& got) {
    double err2 = 0.0, ref2 = 0.0, max_err = 0.0, max_ref = 0.0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = __bfloat162float(got[i]);
        if (!std::isfinite(g)) {
            ++bad;
            continue;
        }
        const double d = g - ref[i];
        err2 += d * d;
        ref2 += static_cast<double>(ref[i]) * ref[i];
        max_err = std::max(max_err, std::fabs(d));
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
    }
    const double rel = ref2 > 0.0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    const bool ok    = bad == 0 && rel < 1e-2 && max_err <= 3e-2 * max_ref;
    std::printf("%-34s rel_l2 %.2e  max_err/max_ref %.2e  non-finite %zu  %s\n", label, rel,
                max_ref > 0 ? max_err / max_ref : max_err, bad, ok ? "ok" : "FAIL");
    return ok;
}

bool check_case(const char* name, int tokens, Mode mode, const Experts& w, Workspace& ws,
                std::mt19937& rng) {
    const Round r = make_round(tokens, mode, rng);
    const DeviceRound d(r);
    const std::vector<float> ref      = reference(r, d, w, false);
    const std::vector<float> ref_gemv = reference(r, d, w, true);
    const std::size_t n               = static_cast<std::size_t>(tokens) * kTopK * kHidden;
    bool ok                           = true;
    for (fp8::Tile tile : {fp8::Tile::Swap16, fp8::Tile::Swap32, fp8::Tile::Swap64,
                           fp8::Tile::Wide128, fp8::Tile::Gemv, fp8::Tile::Auto}) {
        if (tile == fp8::Tile::Gemv && static_cast<std::int64_t>(tokens) * kTopK > fp8::kGemvMaxAssignments) {
            continue;
        }
        DeviceBuffer<__nv_bfloat16> out = output_block(tokens);
        CHECK_CUDA(cudaMemset(out.data, 0xFF, out.count * sizeof(__nv_bfloat16)));
        fp8::run(kGeometry, d.x.data, tokens, d.column_token.data, d.offsets.data, w.view(),
                 ws.scratch.data, ws.bytes, out.data, out.data, nullptr, tile);
        CHECK_CUDA(cudaDeviceSynchronize());
        const std::string label = std::string(name) + " T=" + std::to_string(tokens) + " " + tile_name(tile);
        const fp8::Tile ran = tile == fp8::Tile::Auto ? fp8::resolved_tile(kGeometry, tokens) : tile;
        ok = compare(label.c_str(), ran == fp8::Tile::Gemv ? ref_gemv : ref, out.download(n)) && ok;
    }
    return ok;
}

bool check_graph(int tokens, const Experts& w, Workspace& ws, std::mt19937& rng) {
    const Round r = make_round(tokens, Mode::Uniform, rng);
    const DeviceRound d(r);
    const std::size_t n = static_cast<std::size_t>(tokens) * kTopK * kHidden;
    DeviceBuffer<__nv_bfloat16> direct = output_block(tokens);
    DeviceBuffer<__nv_bfloat16> replay = output_block(tokens);
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    fp8::run(kGeometry, d.x.data, tokens, d.column_token.data, d.offsets.data, w.view(),
             ws.scratch.data, ws.bytes, direct.data, direct.data, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    fp8::run(kGeometry, d.x.data, tokens, d.column_token.data, d.offsets.data, w.view(),
             ws.scratch.data, ws.bytes, replay.data, replay.data, stream);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    CHECK_CUDA(cudaMemsetAsync(replay.data, 0, replay.count * sizeof(__nv_bfloat16), stream));
    CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto a  = direct.download(n);
    const auto b  = replay.download(n);
    const bool ok = std::memcmp(a.data(), b.data(), n * sizeof(__nv_bfloat16)) == 0;
    std::printf("graph capture and replay, T=%d: %s\n", tokens, ok ? "bit-identical" : "DIFFERS");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return ok;
}

/// A comma list from the environment, or `fallback` when the variable is unset.
std::vector<std::string> env_list(const char* name, std::vector<std::string> fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    std::vector<std::string> out;
    std::string item;
    for (const char* p = raw;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!item.empty()) { out.push_back(item); }
            item.clear();
            if (*p == '\0') { break; }
        } else {
            item.push_back(*p);
        }
    }
    return out;
}

// SUROGATE_FP8_MOE_BENCH_T and SUROGATE_FP8_MOE_BENCH_TILES narrow the sweep (comma lists of
// widths and tile names), for a profiler run.
void bench(const Experts& w, std::mt19937& rng) {
    std::vector<int> widths;
    for (const std::string& t : env_list("SUROGATE_FP8_MOE_BENCH_T",
                                         {"1", "2", "4", "8", "16", "32", "64", "128", "256", "512",
                                          "1024", "2048", "4096"})) {
        widths.push_back(std::atoi(t.c_str()));
    }
    const std::vector<std::string> tiles =
        env_list("SUROGATE_FP8_MOE_BENCH_TILES",
                 {"swap16", "swap32", "swap64", "wide128", "gemv", "auto"});
    Workspace ws(4096);
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    std::printf("\n%6s %8s %10s %10s %8s\n", "tokens", "tile", "us/round", "us/token", "TFLOP/s");
    for (int tokens : widths) {
        const Round r = make_round(tokens, Mode::Uniform, rng);
        const DeviceRound d(r);
        DeviceBuffer<__nv_bfloat16> out = output_block(tokens);
        const double flops = 2.0 * tokens * kTopK * (2.0 * kInter * kHidden + 1.0 * kHidden * kInter);
        for (fp8::Tile tile : {fp8::Tile::Swap16, fp8::Tile::Swap32, fp8::Tile::Swap64,
                               fp8::Tile::Wide128, fp8::Tile::Gemv, fp8::Tile::Auto}) {
            if (std::find(tiles.begin(), tiles.end(), tile_name(tile)) == tiles.end()) { continue; }
            if (tile == fp8::Tile::Gemv &&
                static_cast<std::int64_t>(tokens) * kTopK > fp8::kGemvMaxAssignments) {
                continue;
            }
            const auto launch = [&] {
                fp8::run(kGeometry, d.x.data, tokens, d.column_token.data, d.offsets.data, w.view(),
                         ws.scratch.data, ws.bytes, out.data, out.data, nullptr, tile);
            };
            for (int i = 0; i < 3; ++i) { launch(); }
            const int iters = tokens >= 1024 ? 10 : 30;
            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < iters; ++i) { launch(); }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            const double us = 1e3 * ms / iters;
            std::printf("%6d %8s %10.1f %10.3f %8.1f\n", tokens, tile_name(tile), us, us / tokens,
                        flops / (us * 1e-6) / 1e12);
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    if (!fp8::available()) {
        std::printf("SKIP: FP8 routed experts need an sm_90 device and a 90a build\n");
        return 77;
    }
    std::mt19937 rng(20261005u);
    const Experts w = make_experts(rng);
    Workspace ws(256);

    bool ok = true;
    ok      = check_case("one token", 1, Mode::Uniform, w, ws, rng) && ok;
    ok      = check_case("uniform", 3, Mode::Uniform, w, ws, rng) && ok;
    ok      = check_case("uniform", 37, Mode::Uniform, w, ws, rng) && ok;
    ok      = check_case("uniform", 256, Mode::Uniform, w, ws, rng) && ok;
    ok      = check_case("skewed (12 experts)", 256, Mode::Skewed, w, ws, rng) && ok;
    ok      = check_case("skewed (12 experts)", 16, Mode::Skewed, w, ws, rng) && ok;
    ok      = check_graph(4, w, ws, rng) && ok;
    ok      = check_graph(37, w, ws, rng) && ok;

    const char* bench_env = std::getenv("SUROGATE_FP8_MOE_BENCH");
    if (bench_env != nullptr && std::string(bench_env) == "1") { bench(w, rng); }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
