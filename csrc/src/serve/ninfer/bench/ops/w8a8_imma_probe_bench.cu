// surogate vendor patch (PATCHES.md #17 WIP): W8A8-int IMMA prefill probe.
//
// The W8 A16 MMA family dequantizes int8 codes to BF16 in-kernel and tops
// out at ~90-100 TF/s on the target GEMMs (K=1024/2048) — the measured
// ceiling behind vLLM's remaining long-prefill lead. sm_120's int8 tensor
// cores run at 2x the BF16/FP16 MMA rate, W8 codes are ALREADY int8
// (weights stay bit-exact), and mma.m16n8k32 consumes exactly one 32-value
// quantization group per instruction, so the per-group weight scale (and a
// per-token activation scale) applies on the int32 group result before the
// FP32 accumulate. Activations quantize to int8 per token; that pre-pass is
// O(T*K) and amortizes at prefill token counts.
//
// This bench measures a deliberately simple pipeline (2-stage cp.async,
// BM64 x BN64 x BK64, 8 warps, no smem swizzle sophistication) against the
// same shapes as q08_route_sweep_bench, plus a CPU spot-check of the
// numerics. If even the naive pipeline clears the BF16 ceiling, the
// productized kernel (marlin-class staging) has real headroom.

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/common/memory.cuh"
#include "ops/linear/w8a8/w8a8_imma_gemm.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;

namespace {

constexpr int BM      = 64;
constexpr int BN      = 128;
constexpr int BK      = 64;  // two 32-value groups per k-tile
constexpr int WARPS_M = 2;   // warp tile 32 x 16
constexpr int WARPS_N = 8;
constexpr int THREADS = WARPS_M * WARPS_N * 32;

__device__ __forceinline__ void imma_16n8k32(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                             unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                             unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// W: [rows, k] int8 codes row-major + fp16 scales [rows, k/32].
// X: [k, tokens] column-major int8 (token-major staging) + fp32 scale/token.
// OUT: [rows, tokens] bf16 (column-major like the W8 family outputs).
template <int STAGES>
__global__ __launch_bounds__(THREADS, 2) void w8a8_imma_kernel(
    const std::int8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ out, int rows, int k, int tokens) {
    __shared__ std::int8_t Ws[STAGES][BM * BK];
    __shared__ std::int8_t Xs[STAGES][BN * BK];
    __shared__ __half Ss[STAGES][BM * 2];  // per-row scales for the groups in flight

    const int m0   = static_cast<int>(blockIdx.x) * BM;
    const int n0   = static_cast<int>(blockIdx.y) * BN;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = (warp % WARPS_M) * 32;  // warp row offset in tile
    const int wn   = (warp / WARPS_M) * 16;  // warp col offset in tile
    const int kg   = k / 32;

    const auto stage = [&](int kt, int slot) {
        const int kbase = kt * BK;
        // Weights: BM rows x BK bytes, 16B per cp.async lane.
        for (int i = tid; i < BM * (BK / 16); i += THREADS) {
            const int row = i / (BK / 16);
            const int col = (i % (BK / 16)) * 16;
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Ws[slot][row * BK + col],
                &w_codes[static_cast<std::int64_t>(m0 + row) * k + kbase + col]);
        }
        // Activations: BN tokens x BK bytes.
        for (int i = tid; i < BN * (BK / 16); i += THREADS) {
            const int token = i / (BK / 16);
            const int col   = (i % (BK / 16)) * 16;
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Xs[slot][token * BK + col],
                &x_codes[static_cast<std::int64_t>(n0 + token) * k + kbase + col]);
        }
        // Group scales for this k-tile (two groups).
        for (int i = tid; i < BM * 2; i += THREADS) {
            const int row   = i >> 1;
            const int group = i & 1;
            Ss[slot][i]     = w_scales[static_cast<std::int64_t>(m0 + row) * kg + kt * 2 + group];
        }
    };

    float acc[2][2][4];  // [m16 frag][n8 frag][quad]
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mi][ni][q] = 0.0f;

    const int nkt = k / BK;
    for (int pre = 0; pre < STAGES - 1 && pre < nkt; ++pre) {
        stage(pre, pre);
        ninfer::ops::cp_commit();
    }

    for (int kt = 0; kt < nkt; ++kt) {
        const int slot = kt % STAGES;
        ninfer::ops::cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < nkt) {
            stage(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
            ninfer::ops::cp_commit();
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int kb = group * 32;
            // A fragment: 16 rows x 32 k, int8. Row = wm + mi*16 + (lane/4 [+8]),
            // per-lane 4B at k = (lane%4)*4 + {0,16} within the group.
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int arow0 = wm + mi * 16;
                unsigned a[4];
#pragma unroll
                for (int half_k = 0; half_k < 2; ++half_k) {
#pragma unroll
                    for (int half_m = 0; half_m < 2; ++half_m) {
                        const int row = arow0 + (lane >> 2) + half_m * 8;
                        const int col = kb + (lane & 3) * 4 + half_k * 16;
                        a[half_k * 2 + half_m] =
                            *reinterpret_cast<const unsigned*>(&Ws[slot][row * BK + col]);
                    }
                }
                int d[2][4];
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int token0 = wn + ni * 8;
                    unsigned b[2];
#pragma unroll
                    for (int half_k = 0; half_k < 2; ++half_k) {
                        const int token = token0 + (lane >> 2);
                        const int col   = kb + (lane & 3) * 4 + half_k * 16;
                        b[half_k] =
                            *reinterpret_cast<const unsigned*>(&Xs[slot][token * BK + col]);
                    }
                    d[ni][0] = d[ni][1] = d[ni][2] = d[ni][3] = 0;
                    imma_16n8k32(d[ni][0], d[ni][1], d[ni][2], d[ni][3], a[0], a[1], a[2], a[3],
                                 b[0], b[1]);
                }
                // Rescale the int32 group result: weight scale per row, token
                // scale applied at the epilogue (per-token constant).
                const float ws0 =
                    __half2float(Ss[slot][(arow0 + (lane >> 2)) * 2 + group]);
                const float ws8 =
                    __half2float(Ss[slot][(arow0 + (lane >> 2) + 8) * 2 + group]);
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    acc[mi][ni][0] += static_cast<float>(d[ni][0]) * ws0;
                    acc[mi][ni][1] += static_cast<float>(d[ni][1]) * ws0;
                    acc[mi][ni][2] += static_cast<float>(d[ni][2]) * ws8;
                    acc[mi][ni][3] += static_cast<float>(d[ni][3]) * ws8;
                }
            }
        }
        __syncthreads();
    }

    // Epilogue: d quad layout of m16n8: rows (lane>>2)+{0,8}, cols (lane%4)*2+{0,1}.
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row   = m0 + wm + mi * 16 + (lane >> 2) + (q >= 2 ? 8 : 0);
                const int token = n0 + wn + ni * 8 + (lane & 3) * 2 + (q & 1);
                if (row < rows && token < tokens) {
                    out[static_cast<std::int64_t>(token) * rows + row] =
                        __float2bfloat16_rn(acc[mi][ni][q] * x_scales[token]);
                }
            }
        }
    }
}

// Per-token symmetric int8 activation quantization: bf16 x[token][k]
// (contiguous per token, matching the engine's {hidden, tokens} layout) ->
// int8 codes + fp32 scale per token. One CTA per token; absmax reduction
// then quantize — the O(T*K) pre-pass whose cost this bench reports.
__global__ void act_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                 std::int8_t* __restrict__ codes,
                                 float* __restrict__ scales, int k) {
    __shared__ float red[256];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * k;

    float local = 0.0f;
    for (int i = tid; i < k; i += 256) {
        local = fmaxf(local, fabsf(__bfloat162float(row[i])));
    }
    red[tid] = local;
    __syncthreads();
    for (int step = 128; step > 0; step >>= 1) {
        if (tid < step) red[tid] = fmaxf(red[tid], red[tid + step]);
        __syncthreads();
    }
    const float absmax = red[0];
    const float scale  = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    const float inv    = absmax > 0.0f ? 127.0f / absmax : 0.0f;
    if (tid == 0) scales[token] = scale;
    std::int8_t* out = codes + static_cast<std::int64_t>(token) * k;
    for (int i = tid; i < k; i += 256) {
        const float v = __bfloat162float(row[i]) * inv;
        out[i] = static_cast<std::int8_t>(__float2int_rn(fminf(fmaxf(v, -127.0f), 127.0f)));
    }
}

// ldmatrix variant: fragments load via ldmatrix.m8n8 (x4 for A, x2 for B)
// from 80-byte-stride staging — 80/4 = 20 banks per row step makes the
// 8-row address groups conflict-free without XOR swizzles, and replaces the
// six manual 32-bit gathers per mma with two ldmatrix issues.
constexpr int BK_PAD = 80;

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, const void* smem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__global__ __launch_bounds__(THREADS, 2) void w8a8_imma_ldmatrix_kernel(
    const std::int8_t* __restrict__ w_codes, const __half* __restrict__ w_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ out, int rows, int k, int tokens) {
    __shared__ std::int8_t Ws[2][BM * BK_PAD];
    __shared__ std::int8_t Xs[2][BN * BK_PAD];
    __shared__ __half Ss[2][BM * 2];

    const int m0   = static_cast<int>(blockIdx.x) * BM;
    const int n0   = static_cast<int>(blockIdx.y) * BN;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = (warp % WARPS_M) * 32;
    const int wn   = (warp / WARPS_M) * 16;
    const int kg   = k / 32;

    const auto stage = [&](int kt, int slot) {
        const int kbase = kt * BK;
        for (int i = tid; i < BM * (BK / 16); i += THREADS) {
            const int row = i / (BK / 16);
            const int col = (i % (BK / 16)) * 16;
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Ws[slot][row * BK_PAD + col],
                &w_codes[static_cast<std::int64_t>(m0 + row) * k + kbase + col]);
        }
        for (int i = tid; i < BN * (BK / 16); i += THREADS) {
            const int token = i / (BK / 16);
            const int col   = (i % (BK / 16)) * 16;
            ninfer::ops::cp_async<16, ninfer::ops::Cache::cg>(
                &Xs[slot][token * BK_PAD + col],
                &x_codes[static_cast<std::int64_t>(n0 + token) * k + kbase + col]);
        }
        for (int i = tid; i < BM * 2; i += THREADS) {
            const int row   = i >> 1;
            const int group = i & 1;
            Ss[slot][i]     = w_scales[static_cast<std::int64_t>(m0 + row) * kg + kt * 2 + group];
        }
    };

    float acc[2][2][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[mi][ni][q] = 0.0f;

    const int nkt = k / BK;
    stage(0, 0);
    ninfer::ops::cp_commit();

    for (int kt = 0; kt < nkt; ++kt) {
        const int slot = kt & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if (kt + 1 < nkt) {
            stage(kt + 1, slot ^ 1);
            ninfer::ops::cp_commit();
        }

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int kb = group * 32;
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int arow0 = wm + mi * 16;
                // A: lanes in groups of 8 address rows {0-7, 8-15} x k-halves
                // {kb, kb+16}; ldmatrix x4 returns the mma A fragment order.
                unsigned a[4];
                {
                    const int row  = arow0 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int cofs = kb + (lane >> 4) * 16;
                    ldmatrix_x4(a[0], a[1], a[2], a[3], &Ws[slot][row * BK_PAD + cofs]);
                }
                int d[2][4];
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    const int token0 = wn + ni * 8;
                    unsigned b[2];
                    {
                        // B: lanes 0-7 address tokens at kb, lanes 8-15 at kb+16.
                        const int token = token0 + (lane & 7);
                        const int cofs  = kb + ((lane >> 3) & 1) * 16;
                        ldmatrix_x2(b[0], b[1], &Xs[slot][token * BK_PAD + cofs]);
                    }
                    d[ni][0] = d[ni][1] = d[ni][2] = d[ni][3] = 0;
                    imma_16n8k32(d[ni][0], d[ni][1], d[ni][2], d[ni][3], a[0], a[1], a[2], a[3],
                                 b[0], b[1]);
                }
                const float ws0 = __half2float(Ss[slot][(arow0 + (lane >> 2)) * 2 + group]);
                const float ws8 = __half2float(Ss[slot][(arow0 + (lane >> 2) + 8) * 2 + group]);
#pragma unroll
                for (int ni = 0; ni < 2; ++ni) {
                    acc[mi][ni][0] += static_cast<float>(d[ni][0]) * ws0;
                    acc[mi][ni][1] += static_cast<float>(d[ni][1]) * ws0;
                    acc[mi][ni][2] += static_cast<float>(d[ni][2]) * ws8;
                    acc[mi][ni][3] += static_cast<float>(d[ni][3]) * ws8;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row   = m0 + wm + mi * 16 + (lane >> 2) + (q >= 2 ? 8 : 0);
                const int token = n0 + wn + ni * 8 + (lane & 3) * 2 + (q & 1);
                if (row < rows && token < tokens) {
                    out[static_cast<std::int64_t>(token) * rows + row] =
                        __float2bfloat16_rn(acc[mi][ni][q] * x_scales[token]);
                }
            }
        }
    }
}

struct ProbeCfg128 {
    static constexpr int BM      = 128;
    static constexpr int BN      = 128;
    static constexpr int BK      = 64;
    static constexpr int BK_PAD  = 80;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 8;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
};

struct ProbeStore {
    __nv_bfloat16* out;
    int rows;
    __device__ __forceinline__ void operator()(int row, int token, float value) const {
        out[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

struct Case {
    const char* name;
    int rows;
    int k;
};

constexpr Case kCases[] = {
    {"gate_up_12288x2048", 12288, 2048},
    {"qkvz_8192x2048", 8192, 2048},
    {"gate_up_7168x1024", 7168, 1024},
    {"qkvz_8192x1024", 8192, 1024},
};

} // namespace

int main(int argc, char** argv) {
    int warmup = 15, repeat = 60;
    if (argc > 1) warmup = std::atoi(argv[1]);
    if (argc > 2) repeat = std::atoi(argv[2]);
    cudaStream_t stream = nullptr;

    std::printf("%-22s %6s %12s %14s %9s %9s\n", "case", "tokens", "imma us/TFs", "quant+imma", "max_rel", "rms_rel");
    for (const Case& c : kCases) {
        bench::PackedQuantizedWeight weight =
            bench::make_row_split_weight(QType::W8G32_F16S, c.rows, c.k, c.k);
        // The bench fixture packs planes; for the probe, build plain row-major
        // codes + scales host-side and upload (prototype-only layout).
        std::vector<std::int8_t> host_codes(static_cast<std::size_t>(c.rows) * c.k);
        std::vector<__half> host_scales(static_cast<std::size_t>(c.rows) * (c.k / 32));
        std::srand(1234);
        for (auto& v : host_codes) v = static_cast<std::int8_t>((std::rand() % 255) - 127);
        for (auto& v : host_scales) v = __float2half(0.002f + 0.00001f * (std::rand() % 100));

        const int max_tokens = 1912;
        std::vector<std::int8_t> host_x(static_cast<std::size_t>(max_tokens) * c.k);
        std::vector<float> host_xs(max_tokens);
        for (auto& v : host_x) v = static_cast<std::int8_t>((std::rand() % 255) - 127);
        for (auto& v : host_xs) v = 0.01f;

        DeviceBuffer d_codes(host_codes.size());
        DeviceBuffer d_scales(host_scales.size() * 2);
        DeviceBuffer d_x(host_x.size());
        DeviceBuffer d_xs(host_xs.size() * 4);
        DeviceBuffer d_out(static_cast<std::size_t>(c.rows) * max_tokens * 2);
        cudaMemcpy(d_codes.p, host_codes.data(), host_codes.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales.p, host_scales.data(), host_scales.size() * 2,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_x.p, host_x.data(), host_x.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_xs.p, host_xs.data(), host_xs.size() * 4, cudaMemcpyHostToDevice);

        for (const int tokens : {232, 472, 960, 1912}) {
            const dim3 grid(c.rows / BM, (tokens + BN - 1) / BN);
            const auto launchP = [&](cudaStream_t s) {
                const dim3 pgrid(c.rows / ops::detail::W8A8ImmaConfig::BM,
                                 (tokens + ops::detail::W8A8ImmaConfig::BN - 1) /
                                     ops::detail::W8A8ImmaConfig::BN);
                ops::detail::w8a8_imma_gemm_kernel<ops::detail::W8A8IdentityRowMap, ProbeStore>
                    <<<pgrid, ops::detail::W8A8ImmaConfig::THREADS, 0, s>>>(
                        static_cast<const std::int8_t*>(d_codes.p),
                        static_cast<const std::uint8_t*>(d_scales.p),
                        static_cast<const std::int8_t*>(d_x.p),
                        static_cast<const float*>(d_xs.p), c.rows, c.k, tokens,
                        ops::detail::W8A8IdentityRowMap{},
                        ProbeStore{static_cast<__nv_bfloat16*>(d_out.p), c.rows});
            };
            const auto launchL = [&](cudaStream_t s) {
                w8a8_imma_ldmatrix_kernel<<<grid, THREADS, 0, s>>>(
                    static_cast<const std::int8_t*>(d_codes.p),
                    static_cast<const __half*>(d_scales.p),
                    static_cast<const std::int8_t*>(d_x.p), static_cast<const float*>(d_xs.p),
                    static_cast<__nv_bfloat16*>(d_out.p), c.rows, c.k, tokens);
            };
            const auto launch4 = [&](cudaStream_t s) {
                w8a8_imma_kernel<4><<<grid, THREADS, 0, s>>>(
                    static_cast<const std::int8_t*>(d_codes.p),
                    static_cast<const __half*>(d_scales.p),
                    static_cast<const std::int8_t*>(d_x.p), static_cast<const float*>(d_xs.p),
                    static_cast<__nv_bfloat16*>(d_out.p), c.rows, c.k, tokens);
            };
            const auto launch = [&](cudaStream_t s) {
                const dim3 g128((c.rows + 127) / 128,
                                (tokens + ProbeCfg128::BN - 1) / ProbeCfg128::BN);
                ops::detail::w8a8_imma_gemm_kernel<ops::detail::W8A8IdentityRowMap, ProbeStore,
                                                   ProbeCfg128>
                    <<<g128, ProbeCfg128::THREADS, 0, s>>>(
                        static_cast<const std::int8_t*>(d_codes.p),
                        static_cast<const std::uint8_t*>(d_scales.p),
                        static_cast<const std::int8_t*>(d_x.p),
                        static_cast<const float*>(d_xs.p), c.rows, c.k, tokens,
                        ops::detail::W8A8IdentityRowMap{},
                        ProbeStore{static_cast<__nv_bfloat16*>(d_out.p), c.rows});
            };
            const auto launch_unused = [&](cudaStream_t s) {
                w8a8_imma_kernel<2><<<grid, THREADS, 0, s>>>(
                    static_cast<const std::int8_t*>(d_codes.p),
                    static_cast<const __half*>(d_scales.p),
                    static_cast<const std::int8_t*>(d_x.p), static_cast<const float*>(d_xs.p),
                    static_cast<__nv_bfloat16*>(d_out.p), c.rows, c.k, tokens);
            };
            launchL(stream);  // correctness check runs on the ldmatrix variant
            const cudaError_t status = cudaDeviceSynchronize();
            if (status != cudaSuccess) {
                std::printf("%-22s %6d LAUNCH FAILED: %s\n", c.name, tokens,
                            cudaGetErrorString(status));
                continue;
            }

            // Spot-check 8 outputs against a CPU int32 oracle.
            std::vector<__nv_bfloat16> host_out(static_cast<std::size_t>(c.rows) * tokens);
            cudaMemcpy(host_out.data(), d_out.p, host_out.size() * 2, cudaMemcpyDeviceToHost);
            double max_rel = 0.0, rms = 0.0;
            int checked = 0;
            for (int probe = 0; probe < 8; ++probe) {
                const int row   = (probe * 977) % c.rows;
                const int token = (probe * 331) % tokens;
                double ref      = 0.0;
                for (int g = 0; g < c.k / 32; ++g) {
                    long long gi = 0;
                    for (int j = 0; j < 32; ++j) {
                        gi += static_cast<long long>(
                                  host_codes[static_cast<std::size_t>(row) * c.k + g * 32 + j]) *
                              host_x[static_cast<std::size_t>(token) * c.k + g * 32 + j];
                    }
                    ref += static_cast<double>(gi) *
                           __half2float(host_scales[static_cast<std::size_t>(row) * (c.k / 32) + g]);
                }
                ref *= host_xs[token];
                const double got =
                    __bfloat162float(host_out[static_cast<std::size_t>(token) * c.rows + row]);
                const double rel = std::abs(got - ref) / std::max(std::abs(ref), 1e-6);
                max_rel          = std::max(max_rel, rel);
                rms += rel * rel;
                ++checked;
            }
            rms = std::sqrt(rms / checked);

            const auto timing  = bench::measure_launch(launch, stream, warmup, repeat);
            const auto timing4 = bench::measure_launch(launchP, stream, warmup, repeat);
            const double gflop = 2.0 * c.rows * c.k * static_cast<double>(tokens) / 1e9;

            // Combined: per-token act-quant (from bf16) + the IMMA GEMM — the
            // integration-honest cost. bf16 source generated on device.
            DeviceBuffer d_xbf = bench::make_bf16(static_cast<std::size_t>(tokens) * c.k);
            const auto launch_combined = [&](cudaStream_t s) {
                act_quant_kernel<<<tokens, 256, 0, s>>>(
                    static_cast<const __nv_bfloat16*>(d_xbf.p),
                    static_cast<std::int8_t*>(d_x.p), static_cast<float*>(d_xs.p), c.k);
                launch(s);
            };
            const auto combined =
                bench::measure_launch(launch_combined, stream, warmup, repeat);
            // act_quant overwrote d_x/d_xs from random bf16; restore the
            // oracle-known planes for the next token count's spot check.
            cudaMemcpy(d_x.p, host_x.data(), host_x.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_xs.p, host_xs.data(), host_xs.size() * 4, cudaMemcpyHostToDevice);
            std::printf("%-22s %6d c128 %5.1f/%3.0f  pr %7.1f/%3.0f  +q %7.1f/%3.0f %9.2e\n",
                        c.name, tokens, timing.median_us,
                        gflop / (timing.median_us * 1e-6) / 1e3, timing4.median_us,
                        gflop / (timing4.median_us * 1e-6) / 1e3, combined.median_us,
                        gflop / (combined.median_us * 1e-6) / 1e3, max_rel);
            std::fflush(stdout);
        }
        (void)weight;
    }
    return 0;
}
