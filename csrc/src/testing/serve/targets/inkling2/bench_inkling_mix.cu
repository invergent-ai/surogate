// Lane mx: critical-path cost of the inkling2 mix ops inside a decode-like PDL chain (CUDA graph), reference kernels
// vs the latency-cut ones (inkling_mix_use_reference_kernels). One "step" = 48 visits; per visit, as the engine runs it:
//   non-looped (even visits): attn_res (attention input, norm) -> A (attention stand-in) -> short_conv (attention
//     output: residual + addend) -> attn_res (MLP input, + addend, norm) -> M (MoE stand-in) -> short_conv (MLP output)
//   looped (odd visits): blend_inputs (S = 3) -> down partials -> controller_tail -> A -> short_conv ->
//     blend_inputs (S = 1, + addend) -> down partials -> controller_tail -> M -> short_conv
// The stand-ins wait, trigger, stream `mb` MB of a large buffer (the weight reads that evict the residual stack from
// L2 in the engine) and write the conv input. A graph is captured per configuration:
//   ref / fast: every mix op on the reference / fast kernels;
//   -K: op kind K left out (the chain without it), +K: only K on the fast kernels.
// The exclusive (critical-path) time of K per call = (t(ref) - t(-K)) / calls, fast: (t(+K) - t(-K)) / calls.
// Usage: sinfer_inkling_mix_bench [T=1] [mb=40] [iterations=20] [repeats=3]
//        sinfer_inkling_mix_bench eager <T> <fast 0|1>   (3 eager steps, no stand-in reads: for ncu)

#include "api/ops/inkling_mix.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sinfer;

namespace {

#define CK(x)                                                                                              \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

constexpr int kD = 2304, kSlots = 7, kR = 64, kVisits = 48;

/// Stand-in for attention / MoE: reads `n_vec` uint4 of `big` (grid-stride) and writes `T` columns of `out`.
__global__ void stand_in_kernel(const uint4* big, std::size_t n_vec, __nv_bfloat16* out, int T, int salt) {
    pdl::wait_for_dependencies();
    pdl::trigger_dependents();
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = big[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * kD; i += gridDim.x * blockDim.x) {
        const float v = static_cast<float>(((i * 37 + salt) % 101) - 50) * 0.01F + (acc == 0x12345678U ? 1.0F : 0.0F);
        out[i]        = __float2bfloat16_rn(v);
    }
}

void* dalloc(std::size_t bytes, std::uint32_t seed, int kind) { // kind 0 bf16 random, 1 fp32 random (small), 2 zero
    void* p = nullptr;
    CK(cudaMalloc(&p, bytes));
    if (kind == 2) {
        CK(cudaMemset(p, 0, bytes));
        return p;
    }
    std::uint32_t h = seed * 2654435761U + 1;
    const auto next = [&] {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        return static_cast<float>(h & 0xFFFFFF) / 16777216.0F * 2.0F - 1.0F;
    };
    if (kind == 0) {
        std::vector<__nv_bfloat16> v(bytes / 2);
        for (auto& x : v) { x = __float2bfloat16_rn(next() * 0.5F); }
        CK(cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<float> v(bytes / 4);
        for (auto& x : v) { x = next() / 48.0F; }
        CK(cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice));
    }
    return p;
}

struct Bufs {
    int T = 1;
    void *x, *attn_part, *moe_out, *attn_in, *mlp_in, *dest, *msg, *pair, *partials, *blended, *addend_buf;
    void *q1, *q3, *msgq, *nw3, *nw1, *convw, *down, *gfw, *state;
    int *pos, *slots;
    uint4* big;
    std::size_t big_vec, partials_bytes;
};

enum Kind { kAttnRes = 0, kBlend = 1, kTail = 2, kConv = 3, kKinds = 4 };
const char* kKindName[kKinds] = {"attn_res", "blend_inputs", "controller_tail", "short_conv"};

struct Config {
    bool fast[kKinds];
    bool skip[kKinds];
};

ops::InklingPlane slot_plane(const Bufs& b, int slot) {
    return {static_cast<__nv_bfloat16*>(b.x) + slot * kD, DType::BF16, kSlots * kD};
}

void record_step(const Bufs& b, const Config& cfg, cudaStream_t s, std::size_t vec_per_stand_in) {
    const int T = b.T;
    const auto mode = [&](Kind k) { ops::inkling_mix_use_reference_kernels(!cfg.fast[k]); };
    // conv rows: B = T rows of W = 1 (decode)
    ops::InklingConvRows rows{1, T, Tensor(b.pos, DType::I32, {1, T}), Tensor{}, Tensor(b.slots, DType::I32, {T})};
    const ops::InklingConvStateView view{b.state, static_cast<std::int64_t>(ops::InklingMixGeometry::kConvChannels) * 3,
                                         ops::InklingMixGeometry::kConvChannels};
    Tensor convw(b.convw, DType::BF16, {4, kD});
    int salt = 0;
    const auto stand_in = [&](void* out) {
        ++salt;
        const uint4* region = b.big + static_cast<std::size_t>(salt % 4) * vec_per_stand_in;
        CK(pdl::launch_dependent({dim3(340), dim3(256), 0, s}, stand_in_kernel, region, vec_per_stand_in,
                                 static_cast<__nv_bfloat16*>(out), T, salt));
    };
    const auto conv = [&](void* in, int slot, bool residual) {
        if (cfg.skip[kConv]) { return; }
        mode(kConv);
        const ops::InklingPlane xp{in, DType::BF16, kD};
        const ops::InklingPlane ap{b.addend_buf, DType::BF16, kD};
        const ops::InklingPlane out = slot_plane(b, slot);
        ops::inkling_short_conv(xp, kD, convw, rows, view, ops::InklingMixGeometry::kStreamAttn, residual ? &out : nullptr,
                                &ap, out, s);
    };
    for (int v = 0; v < kVisits; ++v) {
        const bool looped = (v % 2) == 1;
        const int N       = 2 + (v / 2) % 5;     // 2..6 sources
        const bool global = (v % 6) == 5;
        ops::InklingAttnResSources src;
        src.stack_count = N - 1;
        for (int i = 0; i < N - 1; ++i) { src.stack[i] = slot_plane(b, i); }
        src.last = slot_plane(b, 6);
        // attention input
        if (!looped) {
            if (!cfg.skip[kAttnRes]) {
                mode(kAttnRes);
                const int Q = global ? 3 : 1;
                Tensor q(global ? b.q3 : b.q1, DType::FP32, {kD, Q});
                Tensor w(b.nw3, DType::BF16, {kD, Q});
                Tensor out(b.attn_in, DType::BF16, {kD, T, Q});
                ops::inkling_attn_res(src, T, q, &w, 1e-6F, out, s);
            }
        } else {
            const int dq = global ? 3 : 1;
            { // attention inputs: S = 3 streams
                ops::InklingAttnResSources bank;
                bank.stack_count = 1;
                bank.stack[0]    = slot_plane(b, 5);
                bank.last        = slot_plane(b, 4);
                Tensor dq_t(global ? b.q3 : b.q1, DType::FP32, {kD, dq});
                Tensor mq_t(b.msgq, DType::FP32, {kD, 3});
                Tensor dest(b.dest, DType::FP32, {kD, T, dq}), msg(b.msg, DType::FP32, {kD, T, 3}),
                    pair(b.pair, DType::BF16, {2 * kD, T, 3});
                if (!cfg.skip[kBlend]) {
                    mode(kBlend);
                    ops::inkling_blend_inputs(src, dq_t, bank, mq_t, T, dest, msg, pair, s);
                }
                const ops::InklingGemmBatch batch{3, static_cast<std::int64_t>(kR) * 2 * kD,
                                                  static_cast<std::int64_t>(T) * 2 * kD, static_cast<std::int64_t>(T) * kR};
                ops::inkling_gemm_inv_partials(b.down, kR, 2 * kD, b.pair, T, batch, static_cast<float*>(b.partials),
                                               b.partials_bytes, s);
                if (!cfg.skip[kTail]) {
                    mode(kTail);
                    Tensor nw(b.nw3, DType::BF16, {kD, 3}), out(b.attn_in, DType::BF16, {kD, T, 3});
                    ops::inkling_controller_tail(static_cast<const float*>(b.partials), b.gfw, dest, msg, nw, 1e-6F, out, s);
                }
            }
        }
        stand_in(b.attn_part);
        conv(b.attn_part, 6, false);
        // MLP input (+ addend = the attention output)
        src.addend = ops::InklingPlane{b.addend_buf, DType::BF16, kD};
        if (!looped) {
            if (!cfg.skip[kAttnRes]) {
                mode(kAttnRes);
                Tensor q(b.q1, DType::FP32, {kD, 1}), w(b.nw1, DType::BF16, {kD, 1}), out(b.mlp_in, DType::BF16, {kD, T, 1});
                ops::inkling_attn_res(src, T, q, &w, 1e-6F, out, s);
            }
        } else {
            ops::InklingAttnResSources bank;
            bank.stack_count = 1;
            bank.stack[0]    = slot_plane(b, 5);
            bank.last        = slot_plane(b, 4);
            Tensor dq_t(b.q1, DType::FP32, {kD, 1});
            Tensor mq_t(static_cast<float*>(b.msgq) + 3 * kD, DType::FP32, {kD, 1});
            Tensor dest(b.dest, DType::FP32, {kD, T, 1}), msg(b.msg, DType::FP32, {kD, T, 1}),
                pair(b.pair, DType::BF16, {2 * kD, T, 1});
            if (!cfg.skip[kBlend]) {
                mode(kBlend);
                ops::inkling_blend_inputs(src, dq_t, bank, mq_t, T, dest, msg, pair, s);
            }
            const ops::InklingGemmBatch batch{1, static_cast<std::int64_t>(kR) * 2 * kD,
                                              static_cast<std::int64_t>(T) * 2 * kD, static_cast<std::int64_t>(T) * kR};
            ops::inkling_gemm_inv_partials(b.down, kR, 2 * kD, b.pair, T, batch, static_cast<float*>(b.partials),
                                           b.partials_bytes, s);
            if (!cfg.skip[kTail]) {
                mode(kTail);
                Tensor nw(b.nw1, DType::BF16, {kD, 1}), out(b.mlp_in, DType::BF16, {kD, T, 1});
                ops::inkling_controller_tail(static_cast<const float*>(b.partials), b.gfw, dest, msg, nw, 1e-6F, out, s);
            }
        }
        stand_in(b.moe_out);
        conv(b.moe_out, 6, true);
    }
    ops::inkling_mix_use_reference_kernels(false);
}

/// Calls of each kind per step (for the per-call division).
void count_calls(int (&calls)[kKinds]) {
    for (int& c : calls) { c = 0; }
    for (int v = 0; v < kVisits; ++v) {
        if ((v % 2) == 1) {
            calls[kBlend] += 2;
            calls[kTail] += 2;
        } else {
            calls[kAttnRes] += 2;
        }
        calls[kConv] += 2;
    }
}

double time_graph(const Bufs& b, const Config& cfg, cudaStream_t s, std::size_t vec, int iterations) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    record_step(b, cfg, s, vec);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 3; ++i) { CK(cudaGraphLaunch(ge, s)); }
    CK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iterations; ++i) { CK(cudaGraphLaunch(ge, s)); }
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0F;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return 1000.0 * ms / iterations; // us per step
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    // `eager <T> <fast 0|1>`: three eager steps (no graph, no stand-in reads), for ncu's kernel filter.
    const bool eager     = argc > 1 && std::string(argv[1]) == "eager";
    if (eager) {
        argv += 1;
        argc -= 1;
    }
    const int T          = argc > 1 ? std::atoi(argv[1]) : 1;
    const int mb         = argc > 2 ? std::atoi(argv[2]) : 40;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 20;
    const int repeats    = argc > 4 ? std::atoi(argv[4]) : 3;
    Bufs b;
    b.T             = T;
    b.x             = dalloc(static_cast<std::size_t>(T) * kSlots * kD * 2, 1, 0);
    b.attn_part     = dalloc(static_cast<std::size_t>(T) * kD * 2, 2, 0);
    b.moe_out       = dalloc(static_cast<std::size_t>(T) * kD * 2, 3, 0);
    b.addend_buf    = dalloc(static_cast<std::size_t>(T) * kD * 2, 4, 0);
    b.attn_in       = dalloc(static_cast<std::size_t>(T) * kD * 3 * 2, 5, 2);
    b.mlp_in        = dalloc(static_cast<std::size_t>(T) * kD * 2, 6, 2);
    b.dest          = dalloc(static_cast<std::size_t>(T) * kD * 3 * 4, 7, 2);
    b.msg           = dalloc(static_cast<std::size_t>(T) * kD * 3 * 4, 8, 2);
    b.pair          = dalloc(static_cast<std::size_t>(T) * kD * 2 * 3 * 2, 9, 2);
    b.partials_bytes = ops::inkling_gemm_inv_workspace_bytes(kR, 2 * kD, T, 3);
    b.partials      = dalloc(b.partials_bytes, 10, 2);
    b.blended       = nullptr;
    b.q1            = dalloc(static_cast<std::size_t>(kD) * 4, 11, 1);
    b.q3            = dalloc(static_cast<std::size_t>(kD) * 3 * 4, 12, 1);
    b.msgq          = dalloc(static_cast<std::size_t>(kD) * 4 * 4, 13, 1);
    b.nw3           = dalloc(static_cast<std::size_t>(kD) * 3 * 2, 14, 0);
    b.nw1           = dalloc(static_cast<std::size_t>(kD) * 2, 15, 0);
    b.convw         = dalloc(static_cast<std::size_t>(kD) * 4 * 2, 16, 0);
    b.down          = dalloc(static_cast<std::size_t>(3) * kR * 2 * kD * 2, 17, 0);
    b.gfw           = dalloc(static_cast<std::size_t>(3) * 2 * kD * kR * 2, 18, 0);
    b.state         = dalloc(static_cast<std::size_t>(T) * ops::InklingMixGeometry::kConvChannels * 3 * 2, 19, 0);
    std::vector<int> pos(T), slots(T);
    for (int i = 0; i < T; ++i) {
        pos[i]   = 5000 + 17 * i;
        slots[i] = i;
    }
    CK(cudaMalloc(&b.pos, T * 4));
    CK(cudaMalloc(&b.slots, T * 4));
    CK(cudaMemcpy(b.pos, pos.data(), T * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.slots, slots.data(), T * 4, cudaMemcpyHostToDevice));
    const std::size_t big_bytes = static_cast<std::size_t>(std::max(mb, 1)) * 4 * 1024 * 1024; // 4 rotating regions
    b.big     = static_cast<uint4*>(dalloc(big_bytes, 20, 2));
    b.big_vec = big_bytes / 16;
    const std::size_t vec = static_cast<std::size_t>(mb) * 1024 * 1024 / 16;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    if (eager) {
        Config cfg{};
        const bool fast_all = argc > 2 && std::atoi(argv[2]) != 0;
        for (int k = 0; k < kKinds; ++k) { cfg.fast[k] = fast_all; }
        for (int i = 0; i < 3; ++i) { record_step(b, cfg, s, 0); }
        CK(cudaStreamSynchronize(s));
        std::printf("eager T=%d fast=%d DONE\n", T, fast_all ? 1 : 0);
        return 0;
    }
    int calls[kKinds];
    count_calls(calls);
    std::printf("T=%d mb=%d iterations=%d (calls per step: attn_res %d, blend_inputs %d, controller_tail %d, short_conv %d)\n",
                T, mb, iterations, calls[0], calls[1], calls[2], calls[3]);
    Config ref{}, fast{};
    for (int k = 0; k < kKinds; ++k) { fast.fast[k] = true; }
    for (int r = 0; r < repeats; ++r) {
        const double t_ref = time_graph(b, ref, s, vec, iterations);
        const double t_fast = time_graph(b, fast, s, vec, iterations);
        std::printf("rep %d: step ref %.1f us, fast %.1f us, saved %.1f us\n", r, t_ref, t_fast, t_ref - t_fast);
        for (int k = 0; k < kKinds; ++k) {
            Config minus = ref, plus = ref;
            minus.skip[k] = true;
            plus.fast[k]  = true;
            const double t_minus = time_graph(b, minus, s, vec, iterations);
            const double t_plus  = time_graph(b, plus, s, vec, iterations);
            std::printf("  %-16s exclusive per call: ref %.2f us, fast %.2f us (step: ref %.1f, -K %.1f, +K fast %.1f)\n",
                        kKindName[k], (t_ref - t_minus) / calls[k], (t_plus - t_minus) / calls[k], t_ref, t_minus, t_plus);
        }
    }
    std::printf("DONE\n");
    return 0;
}
