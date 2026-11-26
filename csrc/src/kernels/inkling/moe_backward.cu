// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// The routed experts' backward for one group of experts (moe_backward.h). GEMMs: cuBLASLt, bf16 x bf16 -> fp32 for the
// decoded expert units (fp32 accumulation), fp32 for the LoRA paths. Everything else: the kernels below.

#include "kernels/inkling/moe_backward.h"

#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// Column-major C[m, n] = alpha * op(A) * op(B) + beta * C, for `batch` problems at constant strides (elements) of A, B
// and C. Row-major callers pass their arrays as the transposes.
struct Mat {
    const void* p;
    cudaDataType_t type;
    int rows, cols, ld;
    cublasOperation_t op;
    long stride;  // elements between batch items
};

void gemm(const Mat& A, const Mat& B, void* C, cudaDataType_t ctype, int m, int n, int ldc, long cstride, int batch, float alpha,
          float beta, void* ws, std::size_t ws_bytes, cudaStream_t stream) {
    if (m == 0 || n == 0 || batch == 0) return;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    check_blas(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
    cublasStatus_t st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &A.op, sizeof(A.op));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &B.op, sizeof(B.op));
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&la, A.type, A.rows, A.cols, A.ld);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lb, B.type, B.rows, B.cols, B.ld);
    if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lc, ctype, m, n, ldc);
    if (st == CUBLAS_STATUS_SUCCESS && batch > 1) {
        const std::int32_t bc = batch;
        const std::int64_t sa = A.stride, sb = B.stride, sc = cstride;
        cublasLtMatrixLayout_t ls[3] = {la, lb, lc};
        const std::int64_t ss[3] = {sa, sb, sc};
        for (int k = 0; k < 3 && st == CUBLAS_STATUS_SUCCESS; ++k) {
            st = cublasLtMatrixLayoutSetAttribute(ls[k], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
            if (st == CUBLAS_STATUS_SUCCESS)
                st = cublasLtMatrixLayoutSetAttribute(ls[k], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ss[k], sizeof(ss[k]));
        }
    }
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmul(lt_handle(), desc, &alpha, A.p, la, B.p, lb, &beta, C, lc, C, lc, nullptr, ws, ws_bytes, stream);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    check_blas(st, "matmul");
}

// Batched row-major helpers: `nb` experts, `n` (padded) rows each. Row buffers are [nb, n, .] (stride n * width); weight
// tensors are [nb, ., .] at stride `ws_` elements.
// X [n, K], W [M, K]:  Y [n, M] = alpha * X @ W^T + beta * Y.
void xwT(const void* X, cudaDataType_t xt, const void* W, cudaDataType_t wt, void* Y, cudaDataType_t yt, int nb, int n, int K, int M,
         float alpha, float beta, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({W, wt, K, M, K, CUBLAS_OP_T, long(M) * K}, {X, xt, K, n, K, CUBLAS_OP_N, long(n) * K}, Y, yt, M, n, M, long(n) * M, nb, alpha,
         beta, ws, wsb, s);
}
// X [n, K], W [K, M]:  Y [n, M] = alpha * X @ W + beta * Y.
void xw(const void* X, cudaDataType_t xt, const void* W, cudaDataType_t wt, void* Y, cudaDataType_t yt, int nb, int n, int K, int M,
        float alpha, float beta, void* ws, std::size_t wsb, cudaStream_t s) {
    gemm({W, wt, M, K, M, CUBLAS_OP_N, long(K) * M}, {X, xt, K, n, K, CUBLAS_OP_N, long(n) * K}, Y, yt, M, n, M, long(n) * M, nb, alpha,
         beta, ws, wsb, s);
}
// G [n, M], L [n, r]:  dB [M, r] += alpha * G^T @ L.
void gTl(const float* G, const float* L, float* dB, int nb, int n, int M, int r, float alpha, void* ws, std::size_t wsb,
         cudaStream_t s) {
    gemm({L, CUDA_R_32F, r, n, r, CUBLAS_OP_N, long(n) * r}, {G, CUDA_R_32F, M, n, M, CUBLAS_OP_T, long(n) * M}, dB, CUDA_R_32F, r, M, r,
         long(M) * r, nb, alpha, 1.0f, ws, wsb, s);
}
// L [n, r], X [n, K]:  dA [r, K] += alpha * L^T @ X.
void lTx(const float* L, const float* X, float* dA, int nb, int n, int r, int K, float alpha, void* ws, std::size_t wsb,
         cudaStream_t s) {
    gemm({X, CUDA_R_32F, K, n, K, CUBLAS_OP_N, long(n) * K}, {L, CUDA_R_32F, r, n, r, CUBLAS_OP_T, long(n) * r}, dA, CUDA_R_32F, K, r, K,
         long(r) * K, nb, alpha, 1.0f, ws, wsb, s);
}

__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// One block = nb experts (<= kMaxBlock) with `cap` padded rows each: padded row p = b * cap + i is the expert's row
// off[b] + i of the caller's sorted rows when i < cnt[b], else padding (zero input, no output).
constexpr int kMaxBlock = 16;
struct Block {
    int nb, cap;
    int off[kMaxBlock];
    int cnt[kMaxBlock];
    float f13[kMaxBlock];
    float f2[kMaxBlock];
};

// xb[p] = x[tok[row(p)]] (bf16; zero on padding), xf = the same values in fp32.
__global__ void k_gather_x(bf16* xb, float* xf, const bf16* x, const int* tok, Block B, int D) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(B.nb) * B.cap * D) return;
    const int p = static_cast<int>(idx / D), j = static_cast<int>(idx % D);
    const int b = p / B.cap, i = p % B.cap;
    float v = 0.0f;
    bf16 vb = __float2bfloat16_rn(0.0f);
    if (i < B.cnt[b]) {
        vb = x[static_cast<long>(tok[B.off[b] + i]) * D + j];
        v = __bfloat162float(vb);
    }
    xb[idx] = vb;
    xf[idx] = v;
}

// gf[p] = g[tok[row(p)]] (fp32; zero on padding), gb = bf16(gf * f2[expert]).
__global__ void k_gather_g(float* gf, bf16* gb, const float* g, const int* tok, Block B, int D) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(B.nb) * B.cap * D) return;
    const int p = static_cast<int>(idx / D), j = static_cast<int>(idx % D);
    const int b = p / B.cap, i = p % B.cap;
    const float v = i < B.cnt[b] ? g[static_cast<long>(tok[B.off[b] + i]) * D + j] : 0.0f;
    gf[idx] = v;
    gb[idx] = __float2bfloat16_rn(v * B.f2[b]);
}

// v[p, :] *= f13[expert]
__global__ void k_scale(float* v, Block B, int W) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(B.nb) * B.cap * W) return;
    v[idx] *= B.f13[static_cast<int>(idx / W) / B.cap];
}

// gu (fp32, [P, 2I] interleaved, before its bf16 rounding) -> y = bf16(((4 tanh(g / 4) * sig(g)) * 25 tanh(u / 25)) * gam).
__global__ void k_swiglu_fwd(bf16* yb, float* yf, const float* gu, const float* gam, Block B, int I) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(B.nb) * B.cap * I) return;
    const int p = static_cast<int>(idx / I);
    const int b = p / B.cap, i = p % B.cap;
    bf16 y = __float2bfloat16_rn(0.0f);
    if (i < B.cnt[b]) {
        const float g = bfr(gu[2 * idx]), u = bfr(gu[2 * idx + 1]);
        const float sig = 1.0f / (1.0f + expf(-g));
        const float gc = 4.0f * tanhf(g * 0.25f);
        const float uc = 25.0f * tanhf(u / 25.0f);
        y = __float2bfloat16_rn(((gc * sig) * uc) * gam[B.off[b] + i]);
    }
    yb[idx] = y;
    yf[idx] = __bfloat162float(y);
}

// dy [P, I] -> dgu [P, 2I] (fp32), dgub = bf16(dgu * f13[expert]), part [P, I] = dy * ((gc * sig) * uc) (summed per row below).
__global__ void k_swiglu_bwd(float* dgu, bf16* dgub, float* part, const float* dy, const float* gu, const float* gam, Block B,
                             int I) {
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long>(B.nb) * B.cap * I) return;
    const int p = static_cast<int>(idx / I);
    const int b = p / B.cap, i = p % B.cap;
    float dg = 0.0f, du = 0.0f, pt = 0.0f;
    if (i < B.cnt[b]) {
        const float g = bfr(gu[2 * idx]), u = bfr(gu[2 * idx + 1]);
        const float sig = 1.0f / (1.0f + expf(-g));
        const float tg = tanhf(g * 0.25f), tu = tanhf(u / 25.0f);
        const float gc = 4.0f * tg, uc = 25.0f * tu;
        const float d = dy[idx];
        const float w = d * gam[B.off[b] + i];
        pt = d * ((gc * sig) * uc);
        // d (gc * sig) / dg = (1 - tg^2) * sig + gc * sig * (1 - sig);  d uc / du = 1 - tu^2
        dg = (w * uc) * ((1.0f - tg * tg) * sig + gc * (sig * (1.0f - sig)));
        du = (w * (gc * sig)) * (1.0f - tu * tu);
    }
    part[idx] = pt;
    dgu[2 * idx] = dg;
    dgu[2 * idx + 1] = du;
    dgub[2 * idx] = __float2bfloat16_rn(dg * B.f13[b]);
    dgub[2 * idx + 1] = __float2bfloat16_rn(du * B.f13[b]);
}

// dgam[row(p)] = sum_j part[p, j], in index order (one thread per row: a fixed order).
__global__ void k_row_sum(float* dgam, const float* part, Block B, int I) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= B.nb * B.cap) return;
    const int b = p / B.cap, i = p % B.cap;
    if (i >= B.cnt[b]) return;
    const float* q = part + static_cast<long>(p) * I;
    float s = 0.0f;
    for (int j = 0; j < I; ++j) s += q[j];
    dgam[B.off[b] + i] = s;
}

// dx[tok[off + i]] += v[i] for ONE expert's rows (its tokens are distinct: no two threads write one element; experts are
// scattered one after the other because a token can sit in several experts of a block).
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
    const int cap_rows = a.max_rows > 0 ? a.max_rows : 8192;
    const Layout L = layout(cap_rows, D, I, r);
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
    const cudaDataType_t BF = CUDA_R_16BF, FP = CUDA_R_32F;

    // One block: experts e0 .. e0 + B.nb - 1 (consecutive, so their weights sit at constant strides), B.cap padded rows each.
    auto run_block = [&](int e0, const Block& B) {
        const int nb = B.nb, n = B.cap;
        const long P = static_cast<long>(nb) * n;
        const bf16* U13 = a.u13 + static_cast<std::size_t>(e0) * 2 * I * D;  // [nb, 2I, D]
        const bf16* U2 = a.u2 + static_cast<std::size_t>(e0) * D * I;        // [nb, D, I]
        const float* A13 = lora ? a.A13 + static_cast<std::size_t>(e0) * r * D : nullptr;      // [nb, r, D]
        const float* B13 = lora ? a.B13 + static_cast<std::size_t>(e0) * 2 * I * r : nullptr;  // [nb, 2I, r]
        const float* A2 = lora ? a.A2 + static_cast<std::size_t>(e0) * r * I : nullptr;        // [nb, r, I]
        const float* B2 = lora ? a.B2 + static_cast<std::size_t>(e0) * D * r : nullptr;        // [nb, D, r]
        bool scale13 = false;
        for (int b = 0; b < nb; ++b) scale13 = scale13 || B.f13[b] != 1.0f;
        // forward recompute: gu, y
        k_gather_x<<<blocks(P * D), 256, 0, stream>>>(xb, xf, a.x, a.tok, B, D);
        xwT(xb, BF, U13, BF, gu, FP, nb, n, D, 2 * I, 1.0f, 0.0f, bws, kBlasWs, stream);
        if (scale13) k_scale<<<blocks(P * 2 * I), 256, 0, stream>>>(gu, B, 2 * I);
        if (lora) {
            xwT(xf, FP, A13, FP, la, FP, nb, n, D, r, 1.0f, 0.0f, bws, kBlasWs, stream);       // la = x @ A13^T
            xwT(la, FP, B13, FP, gu, FP, nb, n, r, 2 * I, s, 1.0f, bws, kBlasWs, stream);      // gu += s la @ B13^T
        }
        k_swiglu_fwd<<<blocks(P * I), 256, 0, stream>>>(yb, yf, gu, a.gam, B, I);
        // d dn = g[token] (straight through the output sum and the rounding)
        k_gather_g<<<blocks(P * D), 256, 0, stream>>>(gf, gb, a.g, a.tok, B, D);
        // dy = bf16(g f2) @ U2 + s (g @ B2) @ A2
        xw(gb, BF, U2, BF, dy, FP, nb, n, D, I, 1.0f, 0.0f, bws, kBlasWs, stream);
        if (lora) {
            xw(gf, FP, B2, FP, lb, FP, nb, n, D, r, 1.0f, 0.0f, bws, kBlasWs, stream);         // lb = g @ B2
            xw(lb, FP, A2, FP, dy, FP, nb, n, r, I, s, 1.0f, bws, kBlasWs, stream);            // dy += s lb @ A2
            if (a.dA2 != nullptr)
                lTx(lb, yf, a.dA2 + static_cast<std::size_t>(e0) * r * I, nb, n, r, I, s, bws, kBlasWs, stream);   // dA2 += s lb^T @ y
            if (a.dB2 != nullptr) {
                xwT(yf, FP, A2, FP, lc, FP, nb, n, I, r, 1.0f, 0.0f, bws, kBlasWs, stream);                        // lc = y @ A2^T
                gTl(gf, lc, a.dB2 + static_cast<std::size_t>(e0) * D * r, nb, n, D, r, s, bws, kBlasWs, stream);   // dB2 += s g^T @ lc
            }
        }
        // through the activation: dgu, d gamma
        k_swiglu_bwd<<<blocks(P * I), 256, 0, stream>>>(dgu, dgub, part, dy, gu, a.gam, B, I);
        k_row_sum<<<blocks(P), 256, 0, stream>>>(a.dgam, part, B, I);
        // dx = bf16(dgu f13) @ U13 + s (dgu @ B13) @ A13
        xw(dgub, BF, U13, BF, dxr, FP, nb, n, 2 * I, D, 1.0f, 0.0f, bws, kBlasWs, stream);
        if (lora) {
            xw(dgu, FP, B13, FP, lc, FP, nb, n, 2 * I, r, 1.0f, 0.0f, bws, kBlasWs, stream);   // lc = dgu @ B13
            xw(lc, FP, A13, FP, dxr, FP, nb, n, r, D, s, 1.0f, bws, kBlasWs, stream);          // dx += s lc @ A13
            if (a.dA13 != nullptr)
                lTx(lc, xf, a.dA13 + static_cast<std::size_t>(e0) * r * D, nb, n, r, D, s, bws, kBlasWs, stream);        // dA13 += s lc^T @ x
            if (a.dB13 != nullptr)
                gTl(dgu, la, a.dB13 + static_cast<std::size_t>(e0) * 2 * I * r, nb, n, 2 * I, r, s, bws, kBlasWs, stream);  // dB13 += s dgu^T @ la
        }
        for (int b = 0; b < nb; ++b)
            if (B.cnt[b] > 0)
                k_scatter_add<<<blocks(static_cast<long>(B.cnt[b]) * D), 256, 0, stream>>>(
                    a.dx, dxr + static_cast<long>(b) * n * D, a.tok + B.off[b], B.cnt[b], D);
    };

    // Experts with few rows run together (one batched GEMM per product, rows padded to the block's longest expert): the
    // per-call cost of a product is about that of 40 rows, and most experts of a short sequence hold fewer. An expert with
    // more than `small` rows runs alone, in row chunks.
    const int small = std::max(1, cap_rows / std::max(a.ng, 1));
    int e = 0;
    while (e < a.ng) {
        const int cnt = a.seg[e + 1] - a.seg[e];
        if (cnt > small) {
            for (int r0 = a.seg[e]; r0 < a.seg[e + 1]; r0 += cap_rows) {
                Block B{};
                B.nb = 1;
                B.cap = std::min(cap_rows, a.seg[e + 1] - r0);
                B.off[0] = r0;
                B.cnt[0] = B.cap;
                B.f13[0] = a.f13[e];
                B.f2[0] = a.f2[e];
                run_block(e, B);
            }
            ++e;
            continue;
        }
        Block B{};
        const int e0 = e;
        int rows = 0;
        while (e < a.ng && B.nb < kMaxBlock && a.seg[e + 1] - a.seg[e] <= small) {
            const int c = a.seg[e + 1] - a.seg[e];
            B.off[B.nb] = a.seg[e];
            B.cnt[B.nb] = c;
            B.f13[B.nb] = a.f13[e];
            B.f2[B.nb] = a.f2[e];
            B.cap = std::max(B.cap, c);
            rows += c;
            ++B.nb;
            ++e;
        }
        if (rows > 0) run_block(e0, B);
    }
    check(cudaGetLastError(), "kernel launch");
}

}  // namespace inkling
