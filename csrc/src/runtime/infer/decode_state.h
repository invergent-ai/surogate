// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DecodeState: shared struct for KV-cache state during autoregressive generation.
// Used by both CompiledExecutor (for dispatch) and GraphExecutor (for orchestration).

#ifndef SUROGATE_SRC_RUNTIME_INFER_DECODE_STATE_H
#define SUROGATE_SRC_RUNTIME_INFER_DECODE_STATE_H

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace infer {

/// State passed to the compiled executor during decode steps.
/// Contains pointers to the KV-cache and per-sequence tracking buffers.
struct DecodeState {
    void* k_data = nullptr;             // K cache: [num_layers, B, max_seq_len, Hkv, Hs]
    void* v_data = nullptr;             // V cache: [num_layers, B, max_seq_len, Hkv, Hs]
    std::size_t per_buffer_bytes = 0;   // Total bytes of one K or V buffer
    float* k_scales = nullptr;          // FP8 K scales (nullptr for BF16)
    float* v_scales = nullptr;          // FP8 V scales (nullptr for BF16)
    int* seq_lens_gpu = nullptr;        // Per-sequence current lengths [batch_size]
    int32_t* cu_seqlens_q_gpu = nullptr; // Cumulative Q lengths [batch_size + 1]
    int32_t* cu_seqlens_k_gpu = nullptr; // Cumulative K lengths [batch_size + 1]
    int max_seq_len = 0;                // Maximum sequence length capacity
    int num_kv_heads = 0;               // Number of KV heads
    int head_dim = 0;                   // Head dimension
    int max_seqlen_k = 0;              // Max K length in current batch (for attention)
    bool fp8 = false;                   // FP8 E4M3 KV-cache

    // Logits output for generation (Phase 2).
    // When non-null, dispatch_fused_lm_head_loss writes raw logits [B, V] here
    // and skips loss computation. Used for sampling during generation.
    float* logits_out_gpu = nullptr;
    int vocab_size = 0;                 // Needed for logits buffer sizing

    // Paged KV-cache mode (Phase 3).
    // When paged=true, dispatch_rope uses kv_cache_append_paged_bf16 and
    // dispatch_flash_attention uses attention_decode_flash_paged.
    // EOS masking: GPU-side per-sequence finished flags [batch_size].
    // When finished_gpu[i] != 0, the sequence is done — dispatch_rope skips
    // KV-cache append, and the token is replaced with 0 (pad) before embedding.
    int* finished_gpu = nullptr;

    // Recurrent state for delta-rule / SSM layers (Qwen3.5 hybrid).
    // Maps layer_idx → GPU buffer holding the recurrent state [B, H, K, V].
    // During prefill: dispatch_chunk_gated_delta_rule saves final_state here.
    // During decode: dispatch_chunk_gated_delta_rule reads initial_state from here
    //                and updates it with the new final_state.
    std::unordered_map<int, void*>* recurrent_states = nullptr;

    // Per-layer causal-conv state for Mamba-style conv1d in Qwen3.5 linear blocks.
    // Maps layer_idx -> GPU buffer [B, conv_dim, kernel-1] with the rolling tail.
    // Used by dispatch_mamba_conv1d in decode mode (non-paged path).
    std::unordered_map<int, void*>* conv_states = nullptr;

    // Prefill mode: when true, dispatch_rope writes K/V to the KV-cache at ALL
    // positions (not just position seq_lens[i]) AND lets self-attention proceed
    // normally (dispatch_flash_attention uses standard training path, NOT decode path).
    // This populates the KV-cache in O(1) model forwards instead of O(T).
    bool prefill_mode = false;

    // FP8 E4M3 KV-cache (SM89+).
    // When fp8=true, KV data is stored as FP8 with per-head scales.
    // Contiguous: k_scales_fp8/v_scales_fp8 layout [layers, B, max_seq, Hkv]
    // Paged: scales stored alongside page pool [layers, total_pages, page_block_size, Hkv]
    float* k_scales_fp8 = nullptr;      // Per-head K scales (contiguous mode)
    float* v_scales_fp8 = nullptr;      // Per-head V scales (contiguous mode)
    float* k_scales_paged_fp8 = nullptr; // Per-head K scales (paged mode, all layers)
    float* v_scales_paged_fp8 = nullptr; // Per-head V scales (paged mode, all layers)

    bool paged = false;
    void* k_pages = nullptr;            // K page pool base (all layers)
    void* v_pages = nullptr;            // V page pool base (all layers)
    std::size_t per_pool_bytes = 0;     // Bytes for one K or V page pool
    int* block_table_gpu = nullptr;     // [batch_size, max_pages_per_seq] on GPU
    int block_table_stride = 0;         // = max_pages_per_seq
    int page_block_size = 0;            // Tokens per page
    int total_pages = 0;                // Total physical pages in the pool
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_DECODE_STATE_H
