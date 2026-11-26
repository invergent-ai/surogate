// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// The routed experts' backward for one group of experts (moe_backward.h). GEMMs: cuBLASLt, bf16 x bf16 -> fp32 for the
// decoded expert units (fp32 accumulation), fp32 for the LoRA paths. Everything else: the kernels below.

#include "kernels/inkling/moe_backward.h"

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace inkling {
namespace {

using bf16 = nv_bfloat16;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling moe_bwd: ") + what + ": " + cudaGetErrorString(e));
}

void check_blas(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string("inkling moe_bwd: cuBLASLt ") + what + ": status " + std::to_string(int(s)));
}

cublasLtHandle_t lt_handle() {
    static thread_local cublasLtHandle_t h = nullptr;
    if (h == nullptr) check_blas(cublasLtCreate(&h), "create");
    return h;
}

// Column-major C[m, n] = alpha * op(A) * op(B) + beta * C. Row-major callers pass their arrays as the transposes.
struct Mat {
    const void* p;
    cudaDataType_t type;
    int rows, cols, ld;
    cublasOperation_t op;
};

void gemm(const Mat& A, const Mat& B, void* C, cudaDataType_t ctype, int m, int n, int ldc, float alpha, float beta,
          void* ws, std::size_t ws_bytes, cudaStream_t stream) {
    if (m == 0 || n == 0) return;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    check_blas(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    cublasStatus_t st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &A.op, sizeof(A.op));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &B.op, sizeof(B.op));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&la, A.type, A.rows, A.cols, A.ld);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lb, B.type, B.rows, B.cols, B.ld);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lc, ctype, m, n, ldc);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmul(lt_handle(), desc, &alpha, A.p, la, B.p, lb, &beta, C, lc, C, lc, nullptr, ws, ws_bytes, stream);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    check_blas(st, "matmul");
}

// Row-major helpers. X [n, K], W [M, K]:  Y [n, M] = alpha * X @ W^T + beta * Y.
void xwT(const void* X, cudaDataType_t xt, const void* W, cudaDataType_t wt, void* Y, cudaDataType_t yt, int n, int K, int M,
         float alpha, float beta, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({W, wt, K, M, K, CUBLAS_OP_T}, {X, xt, K, n, K, CUBLAS_OP_N}, Y, yt, M, n, M, alpha, beta, ws, wsb, s);
}
// X [n, K], W [K, M]:  Y [n, M] = alpha * X @ W + beta * Y.
void xw(const void* X, cudaDataType_t xt, const void* W, cudaDataType_t wt, void* Y, cudaDataType_t yt, int n, int K, int M,
        float alpha, float beta, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({W, wt, M, K, M, CUBLAS_OP_N}, {X, xt, K, n, K, CUBLAS_OP_N}, Y, yt, M, n, M, alpha, beta, ws, wsb, s);
}
// G [n, M], L [n, r]:  dB [M, r] += alpha * G^T @ L.
void gTl(const float* G, const float* L, float* dB, int n, int M, int r, float alpha, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({L, CUDA_R_32F, r, n, r, CUBLAS_OP_N}, {G, CUDA_R_32F, M, n, M, CUBLAS_OP_T}, dB, CUDA_R_32F, r, M, r, alpha, 1.0f, ws, wsb, s);
}
// L [n, r], X [n, K]:  dA [r, K] += alpha * L^T @ X.
void lTx(const float* L, const float* X, float* dA, int n, int r, int K, float alpha, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({X, CUDA_R_32F, K, n, K, CUBLAS_OP_N}, {L, CUDA_R_32F, r, n, r, CUBLAS_OP_T}, dA, CUDA_R_32F, K, r, K, alpha, 1.0f, ws, wsb, s);
}

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// xb[i] = x[tok[i]] (bf16), xf = the same values in fp32.
__global__ void k_gather_x(bf16* xb, float* xf, const bf16* x, const int* tok, int n, int D) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(n) * D) return;
    const int i = static_cast<int>(idx / D), j = static_cast<int>(idx % D);
    const bf16 v = x[static_cast<long>(tok[i]) * D + j];
    xb[idx] = v;
    xf[idx] = __bfloat162float(v);
}

// gf[i] = g[tok[i]] (fp32), gb = bf16(gf * f).
__global__ void k_gather_g(float* gf, bf16* gb, const float* g, const int* tok, int n, int D, float f) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(n) * D) return;
    const int i = static_cast<int>(idx / D), j = static_cast<int>(idx % D);
    const float v = g[static_cast<long>(tok[i]) * D + j];
    gf[idx] = v;
    gb[idx] = __float2bfloat16_rn(v * f);
}

__global__ void k_scale(float* v, long n, float f) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) v[idx] *= f;
}

// gu (fp32, [n, 2I] interleaved, before its bf16 rounding) -> y = bf16(((4 tanh(g / 4) * sig(g)) * 25 tanh(u / 25)) * gam).
__global__ void k_swiglu_fwd(bf16* yb, float* yf, const float* gu, const float* gam, int n, int I) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(n) * I) return;
    const int i = static_cast<int>(idx / I);
    const float g = bfr(gu[2 * idx]), u = bfr(gu[2 * idx + 1]);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float gc = 4.0f * tanhf(g * 0.25f);
    const float uc = 25.0f * tanhf(u / 25.0f);
    const bf16 y = __float2bfloat16_rn(((gc * sig) * uc) * gam[i]);
    yb[idx] = y;
    yf[idx] = __bfloat162float(y);
}

// dy [n, I] -> dgu [n, 2I] (fp32), dgub = bf16(dgu * f), part [n, I] = dy * ((gc * sig) * uc) (summed per row below).
__global__ void k_swiglu_bwd(float* dgu, bf16* dgub, float* part, const float* dy, const float* gu, const float* gam, int n,
                             int I, float f) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(n) * I) return;
    const int i = static_cast<int>(idx / I);
    const float g = bfr(gu[2 * idx]), u = bfr(gu[2 * idx + 1]);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float tg = tanhf(g * 0.25f), tu = tanhf(u / 25.0f);
    const float gc = 4.0f * tg, uc = 25.0f * tu;
    const float d = dy[idx];
    const float w = d * gam[i];
    part[idx] = d * ((gc * sig) * uc);
    // d (gc * sig) / dg = (1 - tg^2) * sig + gc * sig * (1 - sig);  d uc / du = 1 - tu^2
    const float dg = (w * uc) * ((1.0f - tg * tg) * sig + gc * (sig * (1.0f - sig)));
    const float du = (w * (gc * sig)) * (1.0f - tu * tu);
    dgu[2 * idx] = dg;
    dgu[2 * idx + 1] = du;
    dgub[2 * idx] = __float2bfloat16_rn(dg * f);
    dgub[2 * idx + 1] = __float2bfloat16_rn(du * f);
}

// out[i] = sum_j part[i, j], in index order (one thread per row: a fixed order).
__global__ void k_row_sum(float* out, const float* part, int n, int I) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = part + static_cast<long>(i) * I;
    float s = 0.0f;
    for (int j = 0; j < I; ++j) s += p[j];
    out[i] = s;
}

// dx[tok[i]] += v[i]. The tokens of one expert's rows are distinct: no two threads write one element.
__global__ void k_scatter_add(float* dx, const float* v, const int* tok, int n, int D) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(n) * D) return;
    const int i = static_cast<int>(idx / D), j = static_cast<int>(idx % D);
    dx[static_cast<long>(tok[i]) * D + j] += v[idx];
}

inline unsigned blocks(long n) { return static_cast<unsigned>((n + 255) / 256); }

constexpr std::size_t kBlasWs = std::size_t(32) << 20;

struct Layout {
    std::size_t xb, xf, gu, la, yb, yf, gf, gb, lb, dy, part, dgu, dgub, lc, dxr, blas, total;
};

Layout layout(int rows, int D, int I, int r) {
    Layout L{};
    std::size_t o = 0;
    auto take = [&o](std::size_t bytes) {
        const std::size_t at = o;
        o += (bytes + 255) & ~std::size_t(255);
        return at;
    };
    const std::size_t n = static_cast<std::size_t>(rows);
    L.xb = take(n * D * 2);
    L.xf = take(n * D * 4);
    L.gu = take(n * 2 * I * 4);
    L.la = take(n * std::max(r, 1) * 4);
    L.yb = take(n * I * 2);
    L.yf = take(n * I * 4);
    L.gf = take(n * D * 4);
    L.gb = take(n * D * 2);
    L.lb = take(n * std::max(r, 1) * 4);
    L.dy = take(n * I * 4);
    L.part = take(n * I * 4);
    L.dgu = take(n * 2 * I * 4);
    L.dgub = take(n * 2 * I * 2);
    L.lc = take(n * std::max(r, 1) * 4);
    L.dxr = take(n * D * 4);
    L.blas = take(kBlasWs);
    L.total = o;
    return L;
}

}  // namespace

std::size_t moe_group_bwd_workspace_bytes(int rows, int D, int I, int rank) { return layout(rows, D, I, rank).total; }

void moe_group_bwd(const MoeGroupBwdArgs& a, cudaStream_t stream) {
    const int D = a.D, I = a.I, r = a.rank;
    const bool lora = a.A13 != nullptr;
    if (lora && (a.B13 == nullptr || a.A2 == nullptr || a.B2 == nullptr || r <= 0))
        throw std::runtime_error("inkling moe_bwd: an incomplete LoRA (A13, B13, A2, B2 and rank go together)");
    const int cap = a.max_rows > 0 ? a.max_rows : 8192;
    const Layout L = layout(cap, D, I, r);
    if (a.workspace == nullptr || a.workspace_bytes < L.total)
        throw std::runtime_error("inkling moe_bwd: workspace of " + std::to_string(a.workspace_bytes) + " bytes < " +
                                 std::to_string(L.total));
    auto* base = static_cast<unsigned char*>(a.workspace);
    auto* xb = reinterpret_cast<bf16*>(base + L.xb);
    auto* xf = reinterpret_cast<float*>(base + L.xf);
    auto* gu = reinterpret_cast<float*>(base + L.gu);
    auto* la = reinterpret_cast<float*>(base + L.la);
    auto* yb = reinterpret_cast<bf16*>(base + L.yb);
    auto* yf = reinterpret_cast<float*>(base + L.yf);
    auto* gf = reinterpret_cast<float*>(base + L.gf);
    auto* gb = reinterpret_cast<bf16*>(base + L.gb);
    auto* lb = reinterpret_cast<float*>(base + L.lb);
    auto* dy = reinterpret_cast<float*>(base + L.dy);
    auto* part = reinterpret_cast<float*>(base + L.part);
    auto* dgu = reinterpret_cast<float*>(base + L.dgu);
    auto* dgub = reinterpret_cast<bf16*>(base + L.dgub);
    auto* lc = reinterpret_cast<float*>(base + L.lc);
    auto* dxr = reinterpret_cast<float*>(base + L.dxr);
    void* bws = base + L.blas;
    const float s = a.scale;

    for (int e = 0; e < a.ng; ++e) {
        const bf16* U13 = a.u13 + static_cast<std::size_t>(e) * 2 * I * D;  // [2I, D]
        const bf16* U2 = a.u2 + static_cast<std::size_t>(e) * D * I;        // [D, I]
        const float f13 = a.f13[e], f2 = a.f2[e];
        const float* A13 = lora ? a.A13 + static_cast<std::size_t>(e) * r * D : nullptr;      // [r, D]
        const float* B13 = lora ? a.B13 + static_cast<std::size_t>(e) * 2 * I * r : nullptr;  // [2I, r]
        const float* A2 = lora ? a.A2 + static_cast<std::size_t>(e) * r * I : nullptr;        // [r, I]
        const float* B2 = lora ? a.B2 + static_cast<std::size_t>(e) * D * r : nullptr;        // [D, r]
        for (int r0 = a.seg[e]; r0 < a.seg[e + 1]; r0 += cap) {
            const int n = std::min(cap, a.seg[e + 1] - r0);
            const int* tok = a.tok + r0;
            const float* gam = a.gam + r0;
            // forward recompute: gu, y
            k_gather_x<<<blocks(static_cast<long>(n) * D), 256, 0, stream>>>(xb, xf, a.x, tok, n, D);
            xwT(xb, CUDA_R_16BF, U13, CUDA_R_16BF, gu, CUDA_R_32F, n, D, 2 * I, 1.0f, 0.0f, bws, kBlasWs, stream);
            if (f13 != 1.0f) k_scale<<<blocks(static_cast<long>(n) * 2 * I), 256, 0, stream>>>(gu, static_cast<long>(n) * 2 * I, f13);
            if (lora) {
                xwT(xf, CUDA_R_32F, A13, CUDA_R_32F, la, CUDA_R_32F, n, D, r, 1.0f, 0.0f, bws, kBlasWs, stream);       // la = x @ A13^T
                xwT(la, CUDA_R_32F, B13, CUDA_R_32F, gu, CUDA_R_32F, n, r, 2 * I, s, 1.0f, bws, kBlasWs, stream);     // gu += s la @ B13^T
            }
            k_swiglu_fwd<<<blocks(static_cast<long>(n) * I), 256, 0, stream>>>(yb, yf, gu, gam, n, I);
            // d dn = g[token] (straight through the output sum and the rounding)
            k_gather_g<<<blocks(static_cast<long>(n) * D), 256, 0, stream>>>(gf, gb, a.g, tok, n, D, f2);
            // dy = bf16(g f2) @ U2 + s (g @ B2) @ A2
            xw(gb, CUDA_R_16BF, U2, CUDA_R_16BF, dy, CUDA_R_32F, n, D, I, 1.0f, 0.0f, bws, kBlasWs, stream);
            if (lora) {
                xw(gf, CUDA_R_32F, B2, CUDA_R_32F, lb, CUDA_R_32F, n, D, r, 1.0f, 0.0f, bws, kBlasWs, stream);         // lb = g @ B2
                gemm({A2, CUDA_R_32F, I, r, I, CUBLAS_OP_N}, {lb, CUDA_R_32F, r, n, r, CUBLAS_OP_N}, dy, CUDA_R_32F, I, n, I, s,
                     1.0f, bws, kBlasWs, stream);                                                                      // dy += s lb @ A2
                if (a.dB2 != nullptr || a.dA2 != nullptr) {
                    if (a.dA2 != nullptr)
                        lTx(lb, yf, a.dA2 + static_cast<std::size_t>(e) * r * I, n, r, I, s, bws, kBlasWs, stream);     // dA2 += s lb^T @ y
                    if (a.dB2 != nullptr) {
                        xwT(yf, CUDA_R_32F, A2, CUDA_R_32F, lc, CUDA_R_32F, n, I, r, 1.0f, 0.0f, bws, kBlasWs, stream); // lc = y @ A2^T
                        gTl(gf, lc, a.dB2 + static_cast<std::size_t>(e) * D * r, n, D, r, s, bws, kBlasWs, stream);     // dB2 += s g^T @ lc
                    }
                }
            }
            // through the activation: dgu, d gamma
            k_swiglu_bwd<<<blocks(static_cast<long>(n) * I), 256, 0, stream>>>(dgu, dgub, part, dy, gu, gam, n, I, f13);
            k_row_sum<<<blocks(n), 256, 0, stream>>>(a.dgam + r0, part, n, I);
            // dx = bf16(dgu f13) @ U13 + s (dgu @ B13) @ A13
            xw(dgub, CUDA_R_16BF, U13, CUDA_R_16BF, dxr, CUDA_R_32F, n, 2 * I, D, 1.0f, 0.0f, bws, kBlasWs, stream);
            if (lora) {
                xw(dgu, CUDA_R_32F, B13, CUDA_R_32F, lc, CUDA_R_32F, n, 2 * I, r, 1.0f, 0.0f, bws, kBlasWs, stream);   // lc = dgu @ B13
                gemm({A13, CUDA_R_32F, D, r, D, CUBLAS_OP_N}, {lc, CUDA_R_32F, r, n, r, CUBLAS_OP_N}, dxr, CUDA_R_32F, D, n, D, s,
                     1.0f, bws, kBlasWs, stream);                                                                      // dx += s lc @ A13
                if (a.dA13 != nullptr)
                    lTx(lc, xf, a.dA13 + static_cast<std::size_t>(e) * r * D, n, r, D, s, bws, kBlasWs, stream);        // dA13 += s lc^T @ x
                if (a.dB13 != nullptr)
                    gTl(dgu, la, a.dB13 + static_cast<std::size_t>(e) * 2 * I * r, n, 2 * I, r, s, bws, kBlasWs, stream);  // dB13 += s dgu^T @ la
            }
            k_scatter_add<<<blocks(static_cast<long>(n) * D), 256, 0, stream>>>(a.dx, dxr, tok, n, D);
        }
    }
    check(cudaGetLastError(), "kernel launch");
}

}  // namespace inkling
