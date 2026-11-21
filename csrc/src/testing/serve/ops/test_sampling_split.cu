// S1 (ENGINE-INKLING lane c, agent-EN GO 15:3x): the split full-vocabulary draw
// (sampling_full_vocab_partial_kernel + finalize) against the one-CTA-per-row kernel it replaces, through the whole
// ops::sample() op, on every row kind the engine serves: greedy, temperature 1 with per-(request, position) seeds,
// min_p, penalties with token counts, top-p, top-k, other temperatures, logit bias / barred tokens / bitmask, and
// forced ties (quantised logits, several +inf logits in different chunks, fully barred rows), at Inkling's shape
// (FP32 logits, 201,024 tokens) and W4g's (BF16 logits, 151,669 of 151,936 physical rows). Every row's token and every
// row's token counts must be identical. `bench` times sample() on both routes.

#include "api/ops/sampling.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ops/launcher/sampling.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace sinfer;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Logits of row b: mode 0 smooth (sum of uniforms, a few peaks), 1 quantised to steps of 0.5 in [-4, 4] (many exact
// ties), 2 = mode 1 plus +inf at three ids in different chunks (ties at +inf), 3 = mode 0 (rows barred by config).
template <typename T>
__global__ void gen_logits(T* out, int rows, int physical, int domain, unsigned long long seed, const int* modes) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * physical) { return; }
    const int b = static_cast<int>(i / physical), v = static_cast<int>(i % physical);
    const unsigned long long h = mix64(seed ^ (static_cast<unsigned long long>(b) << 40) ^ static_cast<unsigned long long>(v));
    const float u0 = static_cast<float>(h & 0xffffff) / 16777216.0f, u1 = static_cast<float>((h >> 24) & 0xffffff) / 16777216.0f;
    float x;
    const int mode = modes[b];
    if (mode == 1 || mode == 2) {
        x = rintf((u0 - 0.5f) * 16.0f) * 0.5f;
        if (mode == 2 && (v == 7 * 4096 + 11 || v == 23 * 4096 + 5 || v == 31 * 4096 + 99)) { x = INFINITY; }
    } else {
        x = (u0 + u1 - 1.0f) * 6.0f + (((h >> 48) & 0x3ff) == 0 ? 8.0f : 0.0f);
    }
    if (v >= domain) { x = 1000.0f; } // padding rows past the domain must never be drawn
    if constexpr (sizeof(T) == 2) {
        out[i] = __float2bfloat16_rn(x);
    } else {
        out[i] = x;
    }
}

struct Shape {
    const char* name;
    bool fp32;
    int domain, physical;
};

struct Class {
    std::string name;
    int kind;
};

constexpr int kB     = 64;
constexpr int kCalls = 160; // 10,240 rows a class

struct Run {
    std::vector<int> tokens;
    std::vector<int> counts; // every row's token-count array, concatenated
};

struct Harness {
    Shape shape;
    void* logits = nullptr;
    int* d_out   = nullptr;
    int* d_pos   = nullptr;
    int* d_modes = nullptr;
    ops::SamplingConfig* d_cfg = nullptr;
    int* d_counts = nullptr;         // [kB][domain]
    float* d_bias = nullptr;         // [domain]
    std::int32_t* d_mask = nullptr;  // [ceil(domain / 32)]
    std::int32_t* d_none = nullptr;  // an all-zero mask
    std::vector<int> counts0;        // initial counts
    WorkspaceArena* work = nullptr;

    explicit Harness(Shape s) : shape(s) {
        const std::size_t n = static_cast<std::size_t>(kB) * s.physical;
        ck(cudaMalloc(&logits, n * (s.fp32 ? 4 : 2)), "logits");
        ck(cudaMalloc(&d_out, kB * 4), "out");
        ck(cudaMalloc(&d_pos, kB * 4), "pos");
        ck(cudaMalloc(&d_modes, kB * 4), "modes");
        ck(cudaMalloc(&d_cfg, kB * sizeof(ops::SamplingConfig)), "cfg");
        ck(cudaMalloc(&d_counts, static_cast<std::size_t>(kB) * s.domain * 4), "counts");
        ck(cudaMalloc(&d_bias, static_cast<std::size_t>(s.domain) * 4), "bias");
        const int words = (s.domain + 31) / 32;
        ck(cudaMalloc(&d_mask, static_cast<std::size_t>(words) * 4), "mask");
        ck(cudaMalloc(&d_none, static_cast<std::size_t>(words) * 4), "none");
        std::mt19937 g(77);
        std::vector<float> bias(s.domain);
        for (auto& b : bias) { b = std::uniform_real_distribution<float>(-1.0f, 1.0f)(g) * ((g() % 5) == 0 ? 3.0f : 0.0f); }
        ck(cudaMemcpy(d_bias, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), "bias up");
        std::vector<std::int32_t> mask(words), none(words, 0);
        for (auto& w : mask) { w = static_cast<std::int32_t>(g() | g()); } // ~3/4 of the ids allowed
        ck(cudaMemcpy(d_mask, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice), "mask up");
        ck(cudaMemcpy(d_none, none.data(), none.size() * 4, cudaMemcpyHostToDevice), "none up");
        counts0.assign(static_cast<std::size_t>(kB) * s.domain, 0);
        for (int b = 0; b < kB; ++b) {
            for (int k = 0; k < 300; ++k) { counts0[static_cast<std::size_t>(b) * s.domain + g() % s.domain] += 1 + g() % 3; }
        }
        work = new WorkspaceArena(ops::sampling_workspace_capacity_bytes(s.domain, 1, kB) + 4096);
    }

    // One call: B rows of class `kind` (call index c), on the current route.
    Run call(int kind, int c, int B, bool split) {
        std::mt19937_64 g(1000003ull * static_cast<unsigned long long>(kind) + static_cast<unsigned long long>(c));
        std::vector<ops::SamplingConfig> cfg(B);
        std::vector<int> pos(B), modes(B, 0);
        for (int b = 0; b < B; ++b) {
            ops::SamplingConfig& x = cfg[b];
            x.seed        = g();
            pos[b]        = static_cast<int>(g() % 262144);
            x.temperature = 1.0f;
            int k         = kind == 9 ? static_cast<int>(g() % 9) : kind; // 9 = a mixed call
            switch (k) {
            case 0: // greedy, some with bias / mask / barred ids
                x.temperature = 0.0f;
                if (g() % 2) { x.logit_bias = d_bias; }
                if (g() % 3 == 0) { x.token_bitmask = d_mask; }
                if (g() % 3 == 0) { x.suppressed[0] = static_cast<int>(g() % shape.domain); x.suppressed_count = 1; }
                break;
            case 1: break; // temperature 1, untruncated
            case 2: x.min_p = (g() % 3 == 0) ? 0.02f : (g() % 2 ? 0.05f : 0.1f); break;
            case 3:
                x.presence_penalty   = 0.5f;
                x.frequency_penalty  = 0.3f;
                x.repetition_penalty = (g() % 2) ? 1.1f : 1.0f;
                x.token_counts       = d_counts + static_cast<std::size_t>(b) * shape.domain;
                if (g() % 2) { x.min_p = 0.05f; }
                break;
            case 4: x.top_p = (g() % 2) ? 0.9f : 0.95f; break;
            case 5: x.top_k = (g() % 3 == 0) ? 20 : (g() % 2 ? 50 : 1000); break;
            case 6: {
                const float t[4] = {0.3f, 0.7f, 1.5f, 2.0f};
                x.temperature = t[g() % 4];
                break;
            }
            case 7:
                x.logit_bias = d_bias;
                if (g() % 2) { x.token_bitmask = d_mask; }
                x.suppressed[0] = static_cast<int>(g() % shape.domain);
                x.suppressed[1] = 0;
                x.suppressed_count = 2;
                x.token_counts = d_counts + static_cast<std::size_t>(b) * shape.domain; // counts only (no penalty)
                break;
            case 8: // ties: quantised logits, +inf at three ids in different chunks, or every id barred
                modes[b] = 1 + static_cast<int>(g() % 2);
                if (g() % 4 == 0) { x.token_bitmask = d_none; modes[b] = 0; }
                if (g() % 3 == 0) { x.temperature = 0.0f; }
                if (g() % 4 == 0) { x.min_p = 0.05f; }
                break;
            }
            if (g() % 8 == 0) { x.continue_decode_key = 1; }
        }
        ck(cudaMemcpy(d_cfg, cfg.data(), B * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice), "cfg up");
        ck(cudaMemcpy(d_pos, pos.data(), B * 4, cudaMemcpyHostToDevice), "pos up");
        ck(cudaMemcpy(d_modes, modes.data(), B * 4, cudaMemcpyHostToDevice), "modes up");
        ck(cudaMemcpy(d_counts, counts0.data(), counts0.size() * 4, cudaMemcpyHostToDevice), "counts up");
        const long long n = static_cast<long long>(B) * shape.physical;
        const unsigned long long seed = 0xC0FFEEull + 7919ull * static_cast<unsigned long long>(kind * 100000 + c);
        if (shape.fp32) {
            gen_logits<float><<<static_cast<unsigned>((n + 255) / 256), 256>>>(static_cast<float*>(logits), B, shape.physical,
                                                                               shape.domain, seed, d_modes);
        } else {
            gen_logits<__nv_bfloat16><<<static_cast<unsigned>((n + 255) / 256), 256>>>(
                static_cast<__nv_bfloat16*>(logits), B, shape.physical, shape.domain, seed, d_modes);
        }
        ck(cudaGetLastError(), "gen");
        ops::detail::sampling_full_vocab_split() = split;
        Tensor lt(logits, shape.fp32 ? DType::FP32 : DType::BF16, {shape.physical, B});
        Tensor ot(d_out, DType::I32, {B});
        Tensor pt(d_pos, DType::I32, {B});
        const int purpose = (c % 5 == 0) ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode;
        ops::sample(lt, ot, shape.domain, d_cfg, pt, purpose, *work, nullptr);
        ck(cudaDeviceSynchronize(), "sample");
        Run r;
        r.tokens.resize(B);
        ck(cudaMemcpy(r.tokens.data(), d_out, B * 4, cudaMemcpyDeviceToHost), "out down");
        r.counts.resize(static_cast<std::size_t>(B) * shape.domain);
        ck(cudaMemcpy(r.counts.data(), d_counts, r.counts.size() * 4, cudaMemcpyDeviceToHost), "counts down");
        return r;
    }
};

int run_tests() {
    const Shape shapes[2] = {{"Inkling fp32 201024", true, 201024, 201024}, {"W4g bf16 151669/151936", false, 151669, 151936}};
    const Class classes[10] = {{"greedy (bias / mask / barred)", 0}, {"t1 untruncated", 1}, {"t1 min_p", 2},
                               {"penalties + counts (+min_p)", 3}, {"top_p", 4}, {"top_k", 5},
                               {"temperatures 0.3-2", 6}, {"bias + barred + mask + counts", 7},
                               {"ties (quantised, +inf, all barred)", 8}, {"mixed calls", 9}};
    int failures = 0;
    std::printf("| shape | row class | rows | tokens identical | count arrays identical | tokens that differ from row 0's draw |\n");
    std::printf("|---|---|---|---|---|---|\n");
    for (const Shape& s : shapes) {
        Harness h(s);
        for (const Class& cl : classes) {
            long rows = 0, same_tok = 0, same_cnt = 0, varied = 0;
            for (int c = 0; c < kCalls; ++c) {
                const Run a = h.call(cl.kind, c, kB, false);
                const Run b = h.call(cl.kind, c, kB, true);
                for (int r = 0; r < kB; ++r) {
                    ++rows;
                    same_tok += a.tokens[r] == b.tokens[r];
                    varied += a.tokens[r] != a.tokens[0];
                    same_cnt += std::memcmp(&a.counts[static_cast<std::size_t>(r) * s.domain],
                                            &b.counts[static_cast<std::size_t>(r) * s.domain],
                                            static_cast<std::size_t>(s.domain) * 4) == 0;
                }
            }
            std::printf("| %s | %s | %ld | %ld | %ld | %ld |\n", s.name, cl.name.c_str(), rows, same_tok, same_cnt, varied);
            std::fflush(stdout);
            if (same_tok != rows || same_cnt != rows) { ++failures; }
        }
    }
    ops::detail::sampling_full_vocab_split() = true;
    std::printf(failures == 0 ? "SAMPLING SPLIT ALL IDENTICAL\n" : "SAMPLING SPLIT DIFFERENCES in %d classes\n", failures);
    return failures == 0 ? 0 : 1;
}

void run_bench() {
    const Shape shapes[2] = {{"Inkling fp32 201024", true, 201024, 201024}, {"W4g bf16 151669/151936", false, 151669, 151936}};
    for (const Shape& s : shapes) {
        Harness h(s);
        for (const int B : {1, 8, 64}) {
            for (const bool split : {false, true}) {
                h.call(1, 0, B, split); // t1 untruncated rows, warm
                Tensor lt(h.logits, s.fp32 ? DType::FP32 : DType::BF16, {s.physical, B});
                Tensor ot(h.d_out, DType::I32, {B});
                Tensor pt(h.d_pos, DType::I32, {B});
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0);
                for (int i = 0; i < 50; ++i) { ops::sample(lt, ot, s.domain, h.d_cfg, pt, ops::kSamplePurposeDecode, *h.work, nullptr); }
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, e0, e1);
                std::printf("bench sample() %-26s B=%-2d t1 untruncated  %-10s %8.1f us / call (eager, whole op)\n", s.name, B,
                            split ? "split" : "one-CTA", 1000.0f * ms / 50);
            }
        }
    }
    ops::detail::sampling_full_vocab_split() = true;
}

} // namespace

int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    if (argc > 1 && std::string(argv[1]) == "bench") {
        run_bench();
        return 0;
    }
    return run_tests();
}
