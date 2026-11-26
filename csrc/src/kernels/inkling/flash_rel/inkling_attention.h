// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer attention parts on the FlashAttention-2 fork (`inkling_flash`, make_fork.py), hdim 128, bf16.
// NEAR: keys at distance 0 .. window-1 (FA2 local window, left = window-1, right = 0) + the relative bias rel[t, h, d];
//       the whole attention of a local visit. FAR (global visits): keys at distance >= window, no bias = FA2 causal
//       with seqused_k[s] = max(k_len_s - window, 0) under FA2's bottom-right alignment.
// Geometry = the executor's chunk segment form (backend_kvprefix): segment s has queries cu_q[s] .. cu_q[s+1] (rows of
// q) and keys cu_k[s] .. cu_k[s+1] (rows of k / v); query i of a segment sits at key index k_len - q_len + i.
// Outputs: out fp32 [total_q, Hq, 128] normalised over the part's keys, lse fp32 [Hq, total_q] (natural log; -inf =
// no keys). References: lanes/g2/ref/inkling_kernels_ref.py attn_part_fwd / attn_part_bwd(_std).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inkling {

struct AttnPartArgs {
    const nv_bfloat16* q = nullptr;  // [total_q, Hq, 128]
    const nv_bfloat16* k = nullptr;  // [total_k, Hkv, 128]
    const nv_bfloat16* v = nullptr;
    long q_row_stride = 0, k_row_stride = 0, v_row_stride = 0;  // elements; 0 = dense (Hq*128 / Hkv*128)
    const int* cu_q = nullptr;  // [num_segs + 1] device
    const int* cu_k = nullptr;  // [num_segs + 1] device
    int* seqused_k = nullptr;   // FAR: [num_segs] device scratch (filled by the call)
    int num_segs = 1, max_q = 0, max_k = 0, total_q = 0;
    int Hq = 24, Hkv = 8;
    float scale = 0.08838834764831845f;  // 128^-0.5
    int window = 1024;
    const nv_bfloat16* rel = nullptr;  // NEAR: [total_q, Hq, rel_extent]
    int rel_extent = 1024;
    float* out = nullptr;  // [total_q, Hq, 128] fp32
    float* lse = nullptr;  // [Hq, total_q] fp32
};

void attention_near_fwd(const AttnPartArgs& a, cudaStream_t stream);
void attention_far_fwd(const AttnPartArgs& a, cudaStream_t stream);

/// Backward of one part, union or per-part form (the kernel takes the dout operand, lse and delta):
///   union:    dout = bf16(d_result * sig(gate)), lse = lse_s, delta = rowsum(dout * out_s)   (attn_merge_bwd_union)
///   per-part: dout = bf16(d_out_p),               lse = lse_p, delta = rowsum(dout * out_p) - d_lse_p (attn_part_bwd_prep)
/// dq [total_q, Hq, 128], dk / dv [total_k, Hkv, 128] fp32 are ACCUMULATED (+=); NEAR also writes d_rel
/// [total_q, Hq, rel_extent] fp32 (zeroed first, then dS at each in-band pair). FAR: no d_rel.
/// Deterministic: dK / dV summed over the query heads of a KV head in registers (head order), dQ summed over key
/// blocks in key-block order (ordered semaphores); bitwise reproducible for identical inputs.
struct AttnPartBwdArgs {
    AttnPartArgs fwd;               // geometry and q / k / v / rel (out unused; lse unused -- see below)
    int total_k = 0;                // rows of k / v (informational)
    const nv_bfloat16* dout = nullptr;  // [total_q, Hq, 128]
    const float* lse = nullptr;     // [Hq, total_q]
    const float* delta = nullptr;   // [Hq, total_q]
    float* dq = nullptr;
    float* dk = nullptr;
    float* dv = nullptr;
    float* d_rel = nullptr;         // NEAR only
    void* workspace = nullptr;      // attention_bwd_workspace_bytes()
};
size_t attention_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq);
void attention_near_bwd(const AttnPartBwdArgs& a, cudaStream_t stream);
void attention_far_bwd(const AttnPartBwdArgs& a, cudaStream_t stream);

}  // namespace inkling
