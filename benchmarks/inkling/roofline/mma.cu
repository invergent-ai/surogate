// (2) Dense MMA peaks per format, fp32 accumulate, two ways:
//  (a) INSTRUCTION ceiling: a register-only loop of the exact mma.sync forms our engine kernels issue on sm_120a
//      (copied from csrc/src/serve/ops/common/mma.cuh: mma_bf16 m16n8k16, mma_fp8_e4m3 kind::f8f6f4 m16n8k32,
//      mma_nvfp4_e4m3 kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64 with ue4m3 scales). Operand bits are
//      pseudo-random per thread; with toggle=1 one B register flips a mantissa bit every iteration (so the tensor
//      datapath toggles, as with real data; constant operands under-state power and over-state the clock).
//  (b) LIBRARY GEMM: cuBLASLt on large random matrices (TN, fp32 compute, bf16 D): BF16, FP8 e4m3 (scalar scale),
//      FP8 with 1x128 / 128x128 fp32 block scales (DESIGN 10.1 candidates), MXFP8 (e4m3 + ue8m0 / 32), NVFP4
//      (e2m1 + ue4m3 / 16). An unsupported combination prints "supported": false (no heuristic / launch error).
#include "common.cuh"

#include <cublasLt.h>

namespace rf {
namespace {

enum Fmt { kBF16 = 0, kFP8 = 1, kNVFP4 = 2 };

__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_fp8(float* c, const unsigned* a, const unsigned* b) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_nvfp4(float* c, const unsigned* a, const unsigned* b, unsigned sfa,
                                          unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId), "h"(kScaleThreadId));
}

// Operand bits that are finite in every format: bf16 = +-[0.5, 1) (exponent 126), e4m3 bytes with bit 6 clear
// (exponent <= 7: never the NaN pattern), e2m1 any nibble. The toggle flips bit 0 of every byte / half: still finite.
template <int FMT> __device__ __forceinline__ unsigned operand_bits(uint32_t h) {
    if (FMT == kBF16) return (h & 0x807F807Fu) | 0x3F003F00u;
    if (FMT == kFP8) return h & 0xB7B7B7B7u;
    return h;
}

template <int FMT, int ILP>
__global__ void __launch_bounds__(128) mma_loop_kernel(float* __restrict__ out, int iters, int toggle) {
    unsigned a[ILP][4], b[ILP][2];
    float c[ILP][4];
    const uint32_t seed = (blockIdx.x * 128u + threadIdx.x) * 977u;
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
#pragma unroll
        for (int r = 0; r < 4; ++r) a[j][r] = operand_bits<FMT>(hash32(seed + j * 8 + r));
#pragma unroll
        for (int r = 0; r < 2; ++r) b[j][r] = operand_bits<FMT>(hash32(seed + j * 8 + 4 + r));
#pragma unroll
        for (int r = 0; r < 4; ++r) c[j][r] = 0.f;
    }
    const unsigned sfa = 0x38383838u; // ue4m3 1.0 x 4
    const unsigned sfb = 0x38383838u;
    const unsigned flip = FMT == kBF16 ? 0x00010001u : 0x01010101u;
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < ILP; ++j) {
            if (FMT == kBF16) mma_bf16(c[j], a[j], b[j]);
            if (FMT == kFP8) mma_fp8(c[j], a[j], b[j]);
            if (FMT == kNVFP4) mma_nvfp4(c[j], a[j], b[j], sfa, sfb);
        }
        const unsigned m = toggle ? flip : 0u;
#pragma unroll
        for (int j = 0; j < ILP; ++j) b[j][0] ^= m;
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < ILP; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[blockIdx.x * 128 + threadIdx.x] = s;
}

constexpr double flops_per_mma(int fmt) { return fmt == kBF16 ? 4096.0 : fmt == kFP8 ? 8192.0 : 16384.0; }
const char* fmt_name(int fmt) { return fmt == kBF16 ? "bf16" : fmt == kFP8 ? "fp8_e4m3" : "nvfp4_e2m1_ue4m3"; }

template <int FMT, int ILP> void mma_loop_case(int sms, int blocks_per_sm, int toggle, cudaStream_t s) {
    const int grid  = sms * blocks_per_sm;
    float* out      = nullptr;
    RF_CHECK(cudaMalloc(&out, static_cast<std::size_t>(grid) * 128 * sizeof(float)));
    const int iters = 4096;
    auto run        = [&] { mma_loop_kernel<FMT, ILP><<<grid, 128, 0, s>>>(out, iters, toggle); };
    run();
    RF_CHECK(cudaStreamSynchronize(s));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        emit("mma_loop", {kv("format", fmt_name(FMT)), kv("supported", false), kv("error", cudaGetErrorString(e)),
                          kv("ok", false)});
        note("  mma_loop %-18s launch error: %s\n", fmt_name(FMT), cudaGetErrorString(e));
        cudaFree(out);
        return;
    }
    // the sum must be finite (operands finite, accumulators bounded for these iteration counts)
    std::vector<float> h(16);
    RF_CHECK(cudaMemcpy(h.data(), out, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (float v : h) ok = ok && std::isfinite(v);
    const Timing t       = time_it(s, run, 2, options().trial_seconds * 2.5, options().trials);
    const double flops   = flops_per_mma(FMT) * ILP * static_cast<double>(iters) * grid * 4; // 4 warps per CTA
    const double tflops  = flops / t.best_ns / 1e3;
    const double tflopsm = flops / t.median_ns / 1e3;
    emit("mma_loop", {kv("format", fmt_name(FMT)), kv("supported", true), kv("ilp", ILP),
                      kv("warps_per_sm", 4 * blocks_per_sm), kv("toggle", toggle), kv("tflops", tflops),
                      kv("tflops_median", tflopsm), kv("kernel_ms", t.best_ns / 1e6), kv("ok", ok)});
    note("  mma_loop %-18s ilp=%d warps/SM=%2d toggle=%d  %8.1f TFLOP/s (median %8.1f)  %s\n", fmt_name(FMT), ILP,
         4 * blocks_per_sm, toggle, tflops, tflopsm, ok ? "ok" : "NON-FINITE");
    RF_CHECK(cudaFree(out));
}

// ------------------------------------------------------------------------------------------------------------
// cuBLASLt
#define LT_TRY(expr)                                                                                               \
    do {                                                                                                           \
        const cublasStatus_t st__ = (expr);                                                                        \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                                       \
            status = st__;                                                                                         \
            where  = #expr;                                                                                        \
            goto done;                                                                                             \
        }                                                                                                          \
    } while (false)

__global__ void fill_bytes_kernel(uint8_t* p, std::size_t n, uint32_t seed, uint32_t and_mask, uint32_t or_mask) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n / 4; i += stride) {
        reinterpret_cast<uint32_t*>(p)[i] = (hash32(static_cast<uint32_t>(i) ^ seed) & and_mask) | or_mask;
    }
}
__global__ void fill_f32_kernel(float* p, std::size_t n, float v) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = v;
}

enum class Lt { BF16, FP8, FP8_BLK, MXFP8, NVFP4 };
const char* lt_name(Lt k) {
    switch (k) {
    case Lt::BF16: return "bf16";
    case Lt::FP8: return "fp8_e4m3_tensor";
    case Lt::FP8_BLK: return "fp8_e4m3_blk128x128_a_vec128_b";
    case Lt::MXFP8: return "mxfp8_e4m3_ue8m0_32";
    case Lt::NVFP4: return "nvfp4_e2m1_ue4m3_16";
    }
    return "?";
}

std::size_t roundup(std::size_t x, std::size_t m) { return (x + m - 1) / m * m; }

void lt_case(cublasLtHandle_t lt, Lt kind, int M, int N, int K, const char* shape, cudaStream_t s, void* ws,
             std::size_t ws_bytes) {
    // A = weights, row-major [M][K] (= column-major K x M, op T); B = activations row-major [N][K] (column-major
    // K x N, op N); D = bf16 column-major M x N.
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    const char* where     = "";
    cublasLtMatmulDesc_t desc         = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
    cublasLtMatmulPreference_t pref   = nullptr;
    cublasLtMatmulHeuristicResult_t heur[8]{};
    int n_heur = 0;
    void *A = nullptr, *B = nullptr, *D = nullptr, *sa = nullptr, *sb = nullptr;
    float alpha = 1.f, beta = 0.f;
    double best_ns = 1e30, best_median = 1e30;
    int best_algo  = -1;
    bool ran       = false;

    const bool fp4         = kind == Lt::NVFP4;
    const cudaDataType_t t = kind == Lt::BF16 ? CUDA_R_16BF : fp4 ? CUDA_R_4F_E2M1 : CUDA_R_8F_E4M3;
    const double elem      = kind == Lt::BF16 ? 2.0 : fp4 ? 0.5 : 1.0;
    const std::size_t a_bytes = static_cast<std::size_t>(M * elem * K);
    const std::size_t b_bytes = static_cast<std::size_t>(N * elem * K);
    const std::size_t d_bytes = static_cast<std::size_t>(M) * N * 2;
    std::size_t sa_bytes = 0, sb_bytes = 0;
    int32_t mode_a = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    if (kind == Lt::NVFP4) {
        mode_a = mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        sa_bytes        = roundup(M, 128) * roundup(K / 16, 4);
        sb_bytes        = roundup(N, 128) * roundup(K / 16, 4);
    } else if (kind == Lt::MXFP8) {
        mode_a = mode_b = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        sa_bytes        = roundup(M, 128) * roundup(K / 32, 4);
        sb_bytes        = roundup(N, 128) * roundup(K / 32, 4);
    } else if (kind == Lt::FP8_BLK) {
        mode_a   = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
        mode_b   = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
        sa_bytes = roundup(M, 128) / 128 * roundup(K, 128) / 128 * 4 * 4; // generous
        sb_bytes = roundup(N, 128) * roundup(K, 128) / 128 * 4 * 4;
    } else if (kind == Lt::FP8) {
        sa_bytes = sb_bytes = 4; // fp32 scalar 1.0
    }

    RF_CHECK(cudaMalloc(&A, a_bytes));
    RF_CHECK(cudaMalloc(&B, b_bytes));
    RF_CHECK(cudaMalloc(&D, d_bytes));
    {
        // bf16 values +-[0.5, 1), e4m3 bytes without the NaN pattern, e2m1 any.
        const uint32_t am = kind == Lt::BF16 ? 0x807F807Fu : fp4 ? 0xFFFFFFFFu : 0xB7B7B7B7u;
        const uint32_t om = kind == Lt::BF16 ? 0x3F003F00u : 0u;
        fill_bytes_kernel<<<1024, 256, 0, s>>>(static_cast<uint8_t*>(A), a_bytes, 17u, am, om);
        fill_bytes_kernel<<<1024, 256, 0, s>>>(static_cast<uint8_t*>(B), b_bytes, 91u, am, om);
    }
    if (sa_bytes) {
        RF_CHECK(cudaMalloc(&sa, sa_bytes));
        RF_CHECK(cudaMalloc(&sb, sb_bytes));
        if (kind == Lt::NVFP4) { // ue4m3 0.5..0.9 (finite, no NaN)
            fill_bytes_kernel<<<256, 256, 0, s>>>(static_cast<uint8_t*>(sa), sa_bytes, 5u, 0x07070707u, 0x30303030u);
            fill_bytes_kernel<<<256, 256, 0, s>>>(static_cast<uint8_t*>(sb), sb_bytes, 6u, 0x07070707u, 0x30303030u);
        } else if (kind == Lt::MXFP8) { // ue8m0 2^-1 .. 2^1
            fill_bytes_kernel<<<256, 256, 0, s>>>(static_cast<uint8_t*>(sa), sa_bytes, 7u, 0x01010101u, 0x7E7E7E7Eu);
            fill_bytes_kernel<<<256, 256, 0, s>>>(static_cast<uint8_t*>(sb), sb_bytes, 8u, 0x01010101u, 0x7E7E7E7Eu);
        } else {
            fill_f32_kernel<<<256, 256, 0, s>>>(static_cast<float*>(sa), sa_bytes / 4, 1.f);
            fill_f32_kernel<<<256, 256, 0, s>>>(static_cast<float*>(sb), sb_bytes / 4, 1.f);
        }
    }
    RF_CHECK(cudaStreamSynchronize(s));
    RF_CHECK(cudaGetLastError());

    {
        LT_TRY(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
        LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        if (sa_bytes) {
            LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode_a, sizeof(mode_a)));
            LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode_b, sizeof(mode_b)));
            LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
            LT_TRY(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
        }
        LT_TRY(cublasLtMatrixLayoutCreate(&la, t, K, M, K));
        LT_TRY(cublasLtMatrixLayoutCreate(&lb, t, K, N, K));
        LT_TRY(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, M, N, M));
        LT_TRY(cublasLtMatmulPreferenceCreate(&pref));
        LT_TRY(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                    sizeof(ws_bytes)));
        LT_TRY(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, ld, ld, pref, 8, heur, &n_heur));
        if (n_heur == 0) {
            status = CUBLAS_STATUS_NOT_SUPPORTED;
            where  = "no heuristic result";
            goto done;
        }
        for (int i = 0; i < std::min(n_heur, 4); ++i) {
            cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
            auto run          = [&] {
                st = cublasLtMatmul(lt, desc, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &heur[i].algo, ws, ws_bytes,
                                    s);
            };
            run();
            if (st != CUBLAS_STATUS_SUCCESS || cudaStreamSynchronize(s) != cudaSuccess) {
                cudaGetLastError();
                continue;
            }
            const Timing tm = time_it(s, run, 2, options().trial_seconds, 3);
            if (st != CUBLAS_STATUS_SUCCESS) continue;
            ran = true;
            if (tm.best_ns < best_ns) {
                best_ns     = tm.best_ns;
                best_median = tm.median_ns;
                best_algo   = i;
            }
        }
        if (!ran) {
            status = CUBLAS_STATUS_EXECUTION_FAILED;
            where  = "every heuristic algo failed to run";
        }
    }
done:
    {
        const double flops = 2.0 * M * N * static_cast<double>(K);
        if (ran) {
            // sanity: D must be finite somewhere (bf16): copy a few values
            uint16_t h[8]{};
            RF_CHECK(cudaMemcpy(h, D, sizeof(h), cudaMemcpyDeviceToHost));
            bool ok = true;
            for (uint16_t v : h) ok = ok && ((v & 0x7F80u) != 0x7F80u); // not inf / nan
            emit("mma_cublaslt", {kv("format", lt_name(kind)), kv("shape", shape), kv("M", M), kv("N", N),
                                  kv("K", K), kv("supported", true), kv("algo_rank", best_algo),
                                  kv("tflops", flops / best_ns / 1e3), kv("tflops_median", flops / best_median / 1e3),
                                  kv("ms", best_ns / 1e6), kv("ok", ok)});
            note("  cublasLt %-32s %-10s M=%5d N=%5d K=%5d  %8.1f TFLOP/s  %s\n", lt_name(kind), shape, M, N, K,
                 flops / best_ns / 1e3, ok ? "ok" : "NON-FINITE");
        } else {
            emit("mma_cublaslt", {kv("format", lt_name(kind)), kv("shape", shape), kv("M", M), kv("N", N), kv("K", K),
                                  kv("supported", false), kv("status", static_cast<int>(status)), kv("where", where),
                                  kv("ok", true)});
            note("  cublasLt %-32s %-10s NOT SUPPORTED (status %d at %s)\n", lt_name(kind), shape,
                 static_cast<int>(status), where);
        }
    }
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (ld) cublasLtMatrixLayoutDestroy(ld);
    if (desc) cublasLtMatmulDescDestroy(desc);
    cudaFree(A);
    cudaFree(B);
    cudaFree(D);
    if (sa) cudaFree(sa);
    if (sb) cudaFree(sb);
    cudaGetLastError();
}

} // namespace

void run_mma() {
    const cudaDeviceProp p = device_props();
    cudaStream_t s{};
    RF_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int sms = p.multiProcessorCount;

    // (a) instruction ceilings
    for (int toggle = 0; toggle <= 1; ++toggle) {
        mma_loop_case<kBF16, 4>(sms, 4, toggle, s);
        mma_loop_case<kBF16, 8>(sms, 2, toggle, s);
        mma_loop_case<kFP8, 4>(sms, 4, toggle, s);
        mma_loop_case<kFP8, 8>(sms, 2, toggle, s);
        mma_loop_case<kNVFP4, 4>(sms, 4, toggle, s);
        mma_loop_case<kNVFP4, 8>(sms, 2, toggle, s);
    }

    // (b) cuBLASLt
    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        emit("mma_cublaslt", {kv("supported", false), kv("where", "cublasLtCreate"), kv("ok", false)});
        return;
    }
    std::size_t ws_bytes = 64ull << 20;
    void* ws             = nullptr;
    RF_CHECK(cudaMalloc(&ws, ws_bytes));
    struct Shape {
        int M, N, K;
        const char* name;
    };
    // square; the Inkling qkvr projection (8600 x 2304) and wo (2304 x 3072) at an 8,192-token prefill chunk;
    // one MoE expert w13 (2304 x 2304) at 512 rows.
    const Shape shapes[] = {{8192, 8192, 8192, "square8k"},
                            {8600, 8192, 2304, "qkvr_8k"},
                            {2304, 8192, 3072, "wo_8k"},
                            {2304, 512, 2304, "expert512"}};
    const Lt kinds[]     = {Lt::BF16, Lt::FP8, Lt::FP8_BLK, Lt::MXFP8, Lt::NVFP4};
    for (const Lt k : kinds)
        for (const Shape& sh : shapes) {
            if (options().quick && std::string(sh.name) != "square8k") continue;
            lt_case(lt, k, sh.M, sh.N, sh.K, sh.name, s, ws, ws_bytes);
        }
    cublasLtDestroy(lt);
    RF_CHECK(cudaFree(ws));
    RF_CHECK(cudaStreamDestroy(s));
}

} // namespace rf
