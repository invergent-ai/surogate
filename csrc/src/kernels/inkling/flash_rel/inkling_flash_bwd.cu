// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Backward launchers of the Inkling attention parts on the FlashAttention-2 fork (namespace inkling_flash). FA2's
// seqk-parallel backward with: the GIVEN lse and delta (union or per-part form, see inkling_attention.h), the
// relative bias re-added on every recomputed tile and d_rel = dS written per pair (near). One CTA per (key block,
// segment, KV head) loops the query heads of its KV head (make_fork.py B5/B11): dK / dV summed in registers in head order
// and ADDED once into the op's fp32 [total_k, Hkv, 128] accumulators -- no per-query-head buffer (it would be 1.6 GB
// per tensor at a 131k key prefix). dQ: ordered fp32 accumulation (B9, the port of cmake/patch_flash_attn.sh: key
// block n adds after key block n - 1, the chain starting at the first key block that reaches the query block) into
// dq_accum, converted and ADDED into the op's fp32 dq. Every reduction has a fixed order: bitwise reproducible.
// Tiles: FA2's hdim-128 choice for <= 99 KB smem (sm_120): 64 x 64, 8 warps, V in registers (run_mha_bwd_hdim128).

#define FLASH_NAMESPACE inkling_flash

#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"  // dot_do_o (referenced by the Is_first branch)
#include "flash_bwd_kernel.h"
#include "kernel_traits.h"
#include "kernels/inkling/flash_rel/inkling_attention.h"

// Named, not anonymous: nvcc host stubs of kernels in an anonymous namespace become ambiguous next to the
// global `using namespace cute;` of the FA2 headers.
namespace inkling_flash_bwd_detail {

using BTraits = inkling_flash::Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2, 2, true, false, cutlass::bfloat16_t>;
constexpr int kD = 128;

template <bool Is_causal, bool Is_local, bool Has_rel>
__global__ void __launch_bounds__(BTraits::kNThreads) inkling_bwd_kernel(__grid_constant__ const inkling_flash::Flash_bwd_params params) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    inkling_flash::compute_dq_dk_dv_seqk_parallel<BTraits, /*Is_dropout=*/false, Is_causal, Is_local, Has_rel,
                                                  /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false>(params);
#endif
}

__global__ void seqused_kernel(int* seqused, const int* cu_k, int num_segs, int window) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_segs) seqused[s] = max(cu_k[s + 1] - cu_k[s] - window, 0);
}

// delta [H, total_q] -> FA2's padded dsoftmax_sum [H, total_q + 128 * segs] (segment s shifted by 128 * s).
__global__ void delta_pad_kernel(float* dpsum, const float* delta, const int* cu_q, int num_segs, int total_q, int H) {
    const int s = blockIdx.y, h = blockIdx.z;
    const int q0 = cu_q[s], n = cu_q[s + 1] - q0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dpsum[static_cast<long>(h) * (total_q + 128L * num_segs) + q0 + 128L * s + i] = delta[static_cast<long>(h) * total_q + q0 + i];
}

// dq[t, h, :] += scale * dQ for the rows t of segment s. FA2's dq_accum is NOT row-major: the backward adds each
// thread's MMA fragment acc_dq(i) at the AtomicAdd copy partition's element i, and FA2's convert_dQ reads it back
// through the SAME AtomicAdd partition into the same fragment index (whose coordinate is the MMA C partition's): this
// kernel is convert_dQ with an fp32 += epilogue instead of the bf16 store. CTA per (64-row query block, segment, head).
__global__ void __launch_bounds__(BTraits::kNThreads) dq_convert_add_kernel(const inkling_flash::Flash_bwd_params params,
                                                                            float* dq, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int kBlockM = BTraits::kBlockM;
    constexpr int kHeadDim = BTraits::kHeadDim;
    const int m_block = blockIdx.x, bidb = blockIdx.y, bidh = blockIdx.z;
    const int tidx = threadIdx.x;
    const inkling_flash::BlockInfo</*Varlen=*/true> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;
    const int64_t row_offset_dq_accum =
        binfo.q_offset(params.seqlen_q_rounded * params.h * params.d_rounded, params.h * params.d_rounded, bidb) +
        (m_block * kBlockM + 128ll * bidb) * params.h * params.d_rounded + bidh * params.d_rounded;
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_stride(params.h * params.d_rounded, _1{}));
    typename BTraits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;  // the layout the atomics wrote (convert_dQ)
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);
    typename BTraits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccdQc = thr_mma_dq.partition_C(cdQ);  // coordinates of acc_dq(i)
    CUTE_STATIC_ASSERT_V(size(taccdQc) == size(tdQgdQaccum));
    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);  // rows past the segment: its 128-row padding
    const int rows_left = binfo.actual_seqlen_q - m_block * kBlockM;
    const int64_t ld = static_cast<int64_t>(params.h) * kHeadDim;
    float* dq_blk = dq + (binfo.q_offset(int64_t(0), int64_t(1), bidb) + int64_t(m_block) * kBlockM) * ld + bidh * kHeadDim;
#pragma unroll
    for (int i = 0; i < size(tdQrdQaccum); ++i) {
        const int r = get<0>(taccdQc(i)), c = get<1>(taccdQc(i));
        if (r < rows_left) dq_blk[int64_t(r) * ld + c] += scale * tdQrdQaccum(i);
    }
#endif
}

struct Ws {
    float *dq_accum, *dpsum;
    int *sem, *seqused;
    size_t sem_count;
};
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
int m_blocks(int max_q) { return (max_q + BTraits::kBlockM - 1) / BTraits::kBlockM; }
// Layout: dq_accum fp32 [(total_q + 128 segs), Hq, 128] | dpsum fp32 [Hq, total_q + 128 segs] |
// dq semaphores int32 [segs, Hq, m_blocks(max_q)] | seqused int32 [segs].
Ws carve(void* base, int total_q, int max_q, int num_segs, int Hq) {
    char* p = static_cast<char*>(base);
    Ws w{};
    const size_t qacc = align256(sizeof(float) * (static_cast<size_t>(total_q) + 128 * num_segs) * Hq * kD);
    const size_t dps = align256(sizeof(float) * (static_cast<size_t>(total_q) + 128 * num_segs) * Hq);
    w.sem_count = static_cast<size_t>(num_segs) * Hq * m_blocks(max_q);
    const size_t sem = align256(sizeof(int) * w.sem_count);
    w.dq_accum = reinterpret_cast<float*>(p);
    w.dpsum = reinterpret_cast<float*>(p + qacc);
    w.sem = reinterpret_cast<int*>(p + qacc + dps);
    w.seqused = reinterpret_cast<int*>(p + qacc + dps + sem);
    return w;
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling attention bwd ") + what + ": " + cudaGetErrorString(e));
}

template <bool Is_causal, bool Is_local, bool Has_rel>
void run(const inkling::AttnPartBwdArgs& a, cudaStream_t stream) {
    const auto& f = a.fwd;
    if (!a.dout || !a.lse || !a.delta || !a.dq || !a.dk || !a.dv || !a.workspace) throw std::invalid_argument("inkling attention bwd: null");
    if (f.Hkv <= 0 || f.Hq % f.Hkv) throw std::invalid_argument("inkling attention bwd: Hq % Hkv");
    if (f.total_q == 0) return;
    Ws w = carve(a.workspace, f.total_q, f.max_q, f.num_segs, f.Hq);
    const long rows_q = static_cast<long>(f.total_q) + 128L * f.num_segs;
    check(cudaMemsetAsync(w.dq_accum, 0, sizeof(float) * rows_q * f.Hq * kD, stream), "memset dq_accum");
    check(cudaMemsetAsync(w.sem, 0, sizeof(int) * w.sem_count, stream), "memset dq semaphores");
    if (Has_rel) check(cudaMemsetAsync(a.d_rel, 0, sizeof(float) * static_cast<size_t>(f.total_q) * f.Hq * f.rel_extent, stream), "memset d_rel");
    delta_pad_kernel<<<dim3(16, f.num_segs, f.Hq), 256, 0, stream>>>(w.dpsum, a.delta, f.cu_q, f.num_segs, f.total_q, f.Hq);
    if (Is_causal) seqused_kernel<<<(f.num_segs + 127) / 128, 128, 0, stream>>>(w.seqused, f.cu_k, f.num_segs, f.window);

    inkling_flash::Flash_bwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.q_ptr = const_cast<nv_bfloat16*>(f.q);
    p.k_ptr = const_cast<nv_bfloat16*>(f.k);
    p.v_ptr = const_cast<nv_bfloat16*>(f.v);
    p.o_ptr = const_cast<nv_bfloat16*>(a.dout);  // not read by the seqk-parallel backward
    p.q_row_stride = f.q_row_stride ? f.q_row_stride : static_cast<long>(f.Hq) * kD;
    p.k_row_stride = f.k_row_stride ? f.k_row_stride : static_cast<long>(f.Hkv) * kD;
    p.v_row_stride = f.v_row_stride ? f.v_row_stride : static_cast<long>(f.Hkv) * kD;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = kD;
    p.o_row_stride = static_cast<long>(f.Hq) * kD;
    p.o_head_stride = kD;
    p.h = f.Hq;
    p.h_k = f.Hkv;
    p.h_h_k_ratio = f.Hq / f.Hkv;
    p.b = f.num_segs;
    p.seqlen_q = f.max_q;
    p.seqlen_k = f.max_k;
    p.d = kD;
    p.d_rounded = kD;
    p.seqlen_q_rounded = (f.max_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (f.max_k + 127) / 128 * 128;
    p.total_q = f.total_q;
    p.scale_softmax = f.scale;
    p.scale_softmax_log2 = f.scale * static_cast<float>(M_LOG2E);
    p.cu_seqlens_q = const_cast<int*>(f.cu_q);
    p.cu_seqlens_k = const_cast<int*>(f.cu_k);
    p.is_seqlens_k_cumulative = true;
    p.seqused_k = Is_causal ? w.seqused : nullptr;
    p.softmax_lse_ptr = const_cast<float*>(a.lse);
    p.is_bf16 = true;
    p.is_causal = Is_causal;
    p.p_dropout = 1.0f;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = f.scale;
    p.p_dropout_in_uint8_t = 255;
    p.window_size_left = Is_local ? f.window - 1 : -1;
    p.window_size_right = 0;
    p.unpadded_lse = true;
    p.rel_ptr = Has_rel ? const_cast<nv_bfloat16*>(f.rel) : nullptr;
    p.rel_row_stride = static_cast<long>(f.Hq) * f.rel_extent;
    p.rel_extent = f.rel_extent;
    p.d_rel_ptr = Has_rel ? a.d_rel : nullptr;
    p.do_ptr = const_cast<nv_bfloat16*>(a.dout);
    p.do_row_stride = static_cast<long>(f.Hq) * kD;
    p.do_head_stride = kD;
    p.dq_ptr = w.dq_accum;  // not written by the seqk-parallel kernel (dq_convert_add below)
    p.dq_row_stride = static_cast<long>(f.Hq) * kD;
    p.dq_head_stride = kD;
    p.dk_ptr = a.dk;  // fp32 [total_k, Hkv, 128], ADDED into (one CTA per key block and KV head)
    p.dv_ptr = a.dv;
    p.dk_row_stride = p.dv_row_stride = static_cast<long>(f.Hkv) * kD;
    p.dk_head_stride = p.dv_head_stride = kD;
    p.dq_accum_ptr = w.dq_accum;
    p.dsoftmax_sum = w.dpsum;
    p.deterministic = false;
    p.dq_semaphore = w.sem;  // ordered dQ (B9)
    p.dq_semaphore_m_stride = m_blocks(f.max_q);
    // FA2's backward constructs its Dropout object from rng_state even without dropout: give it readable memory.
    p.rng_state = reinterpret_cast<uint64_t*>(w.dpsum);

    auto kernel = &inkling_bwd_kernel<Is_causal, Is_local, Has_rel>;
    constexpr int smem = BTraits::kSmemSize1colblock;
    if (smem >= 48 * 1024) check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem");
    // One CTA per (key block, segment, KV head); the CTA loops the Hq / Hkv query heads (B11). Every key block of a
    // segment gets its own CTA (grid x = all key blocks): the ordered dQ chain relies on it (CTA n waits for CTA n - 1,
    // issued before it).
    const dim3 grid((f.max_k + BTraits::kBlockN - 1) / BTraits::kBlockN, f.num_segs, f.Hkv);
    kernel<<<grid, BTraits::kNThreads, smem, stream>>>(p);
    check(cudaGetLastError(), "kernel");
    dq_convert_add_kernel<<<dim3((f.max_q + BTraits::kBlockM - 1) / BTraits::kBlockM, f.num_segs, f.Hq), BTraits::kNThreads, 0,
                            stream>>>(p, a.dq, f.scale);
    check(cudaGetLastError(), "epilogue kernels");
}

}  // namespace inkling_flash_bwd_detail

namespace inkling {

using namespace inkling_flash_bwd_detail;

size_t attention_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq) {
    return align256(sizeof(float) * (static_cast<size_t>(total_q) + 128 * num_segs) * Hq * kD) +
           align256(sizeof(float) * (static_cast<size_t>(total_q) + 128 * num_segs) * Hq) +
           align256(sizeof(int) * static_cast<size_t>(num_segs) * Hq * m_blocks(max_q)) + align256(sizeof(int) * (num_segs + 1));
}

void attention_near_bwd(const AttnPartBwdArgs& a, cudaStream_t stream) {
    if (!a.fwd.rel || !a.d_rel) throw std::invalid_argument("inkling attention near bwd: rel / d_rel");
    run</*Is_causal=*/false, /*Is_local=*/true, /*Has_rel=*/true>(a, stream);
}

void attention_far_bwd(const AttnPartBwdArgs& a, cudaStream_t stream) {
    run</*Is_causal=*/true, /*Is_local=*/false, /*Has_rel=*/false>(a, stream);
}

}  // namespace inkling
