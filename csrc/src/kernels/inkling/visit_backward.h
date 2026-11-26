// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer: BACKWARD-ONLY entry points (the trainer's forward is the engine's own ops). The coarse op
// `ink_attention` (engine: prologue + relative attention + gate / sink) gets its backward from two calls:
//   attention_visit_bwd  -- from the engine's saved o (before sink and gate) and l (natural lse, no sink): the
//                           union-form backward of near (+ far) + merge + sink + gate -> dq, dk, dv, d_r, d_sink, d_gate;
//   attention_prologue_bwd -- from those gradients and the op's INPUTS (qside q|r|gate|sink, k_raw, v_raw, conv
//                           weights / carries / dynamic taps, positions, scales, tau): the scale + clip (+ tau) of q / r,
//                           the K / V convs + clip / RMS (e4m3 KV store straight-through) -> d_qside, d_k_raw, d_v_raw.
// Plus the glue the IR needs: a gather of the engine's paged e4m3 KV plane into contiguous bf16 keys. Every buffer is
// device memory; nothing here allocates (workspace sizes are queried); every reduction has a fixed order.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "kernels/inkling/flash_rel/inkling_attention.h"

namespace inkling {

/// Keys [pos0, pos0 + rows) of one row of the engine's paged e4m3 plane -> contiguous bf16 [rows, Hkv, 128] (exact).
/// Element offset of (head, pos, d): 128 * 64 * (head + Hkv * page) + 128 * (pos & 63) + d, page = block_table[pos >> 6]
/// (`paged_kv_element_offset<128, 8>`).
void kv_plane_gather_bf16(nv_bfloat16* dst, const std::uint8_t* plane, const int* block_table, long pos0, long rows,
                          int Hkv, cudaStream_t stream);

/// rel = bf16(r @ rel_proj): r bf16 [rowsH, 16], rel_proj bf16 [16, E] -> rel bf16 [rowsH, E] (fp32 sum, k in order).
void rel_bias_fwd(nv_bfloat16* rel, const nv_bfloat16* r, const nv_bfloat16* rel_proj, long rowsH, int E,
                  cudaStream_t stream);
/// d_r = d_rel @ rel_proj^T: d_rel fp32 [rowsH, E] -> d_r fp32 [rowsH, 16] (fixed-order sums).
void rel_bias_bwd(float* d_r, const float* d_rel, const nv_bfloat16* rel_proj, long rowsH, int E, cudaStream_t stream);

struct AttnVisitBwdArgs {
    // Geometry and operands of the parts, as for the forward (AttnPartArgs: q [T, Hq, 128] = the prologue's q,
    // k / v contiguous bf16 key buffers, cu_q / cu_k / num_segs / max_q / max_k / total_q, Hq, Hkv, scale, window).
    // rel / out / lse / seqused_k of `geo` are ignored (rel is rebuilt from r inside).
    AttnPartArgs geo;
    int total_k = 0;
    bool is_global = false;             // far part (distance >= window) present
    const nv_bfloat16* r = nullptr;     // [T, Hq, 16] the prologue's r (clipped, x tau on global visits)
    const nv_bfloat16* rel_proj = nullptr;  // [16, E]
    const float* o = nullptr;           // [T, Hq, 128] fp32: attention output over all keys, BEFORE sink and gate
    const float* l = nullptr;           // [Hq, T] natural-log lse over the keys, WITHOUT the sink (-inf: no keys)
    const nv_bfloat16* gate = nullptr;  // [T, gate_ld] view, Hq * 128 columns
    long gate_ld = 0;                   // 0 = dense
    const nv_bfloat16* sink = nullptr;  // [T, sink_ld] view, Hq columns
    long sink_ld = 0;
    const void* d_result = nullptr;     // [T, Hq * 128]
    bool d_result_fp32 = false;
    float* dq = nullptr;                // [T, Hq, 128] written
    float* dk = nullptr;                // [total_k, Hkv, 128] ADDED (+=)
    float* dv = nullptr;
    float* d_r = nullptr;               // [T, Hq, 16] written
    float* d_sink = nullptr;            // [T, Hq] written
    float* d_gate = nullptr;            // [T, Hq * 128] written
    void* workspace = nullptr;
};
size_t attention_visit_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq, int E);
void attention_visit_bwd(const AttnVisitBwdArgs& a, cudaStream_t stream);

struct AttnPrologueBwdArgs {
    int T = 0, Hq = 24, Hkv = 8, d_rel = 16;
    bool is_global = false;
    const int* positions = nullptr;      // [T] position within the document
    // forward inputs
    const nv_bfloat16* qside = nullptr;  // [T, qside_ld]: q_raw [Hq*128] | r_raw [Hq*16] | gate | sink
    long qside_ld = 0;
    const nv_bfloat16* q_scale = nullptr;  // [Hq, 128]
    const nv_bfloat16* r_scale = nullptr;  // [Hq, 16]
    const float* tau = nullptr;            // [T] (global visits) or null
    float q_clip = 22.627416997969522f, r_clip = 8.0f, k_clip = 22.627416997969522f, eps = 1e-6f;
    const nv_bfloat16* k_raw = nullptr;    // [T, Hkv*128]
    const nv_bfloat16* v_raw = nullptr;
    // local visits: static short convs w [Hkv*128, 4] (tap 3 = current); global: static [4, Hkv*128] + dyn [T, Hkv*4]
    const nv_bfloat16* k_conv_w = nullptr;
    const nv_bfloat16* v_conv_w = nullptr;
    const nv_bfloat16* k_dyn = nullptr;
    const nv_bfloat16* v_dyn = nullptr;
    const nv_bfloat16* k_carry = nullptr;  // [3, Hkv*128] previous chunk's last inputs (oldest first) or null
    const nv_bfloat16* v_carry = nullptr;
    const float* d_k_carry_next = nullptr;  // [3, Hkv*128] the next chunk's gradient on our last 3 inputs, or null
    const float* d_v_carry_next = nullptr;
    // incoming gradients (fp32, dense; from attention_visit_bwd)
    const float* dq = nullptr;   // [T, Hq, 128]
    const float* d_r = nullptr;  // [T, Hq, 16]
    const float* dk = nullptr;   // [T, Hkv, 128] (this chunk's key rows)
    const float* dv = nullptr;
    const float* d_gate = nullptr;  // [T, Hq * 128]
    const float* d_sink = nullptr;  // [T, Hq]
    // outgoing gradients (fp32)
    float* d_qside = nullptr;  // [T, d_qside_ld]: q | r | gate | sink columns (written)
    long d_qside_ld = 0;
    float* d_k_raw = nullptr;  // [T, Hkv*128] written
    float* d_v_raw = nullptr;
    float* d_k_dyn = nullptr;  // [T, Hkv*4] written (global)
    float* d_v_dyn = nullptr;
    float* d_k_carry = nullptr;  // [3, Hkv*128] written (gradient on the carry), or null
    float* d_v_carry = nullptr;
    void* workspace = nullptr;
};
size_t attention_prologue_bwd_workspace_bytes(int T, int Hq, int Hkv, int d_rel);
void attention_prologue_bwd(const AttnPrologueBwdArgs& a, cudaStream_t stream);

}  // namespace inkling
