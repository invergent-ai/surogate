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
    // Optional recurrent state stride metadata.
    // Maps layer_idx → bytes for one sequence state slice [1, H, K, V].
    std::unordered_map<int, std::size_t>* recurrent_state_bytes = nullptr;

    // Per-layer causal-conv state for Mamba-style conv1d in Qwen3.5 linear blocks.
    // Maps layer_idx -> GPU buffer [B, conv_dim, kernel-1] with the rolling tail.
    // Used by dispatch_mamba_conv1d in decode mode (non-paged path).
    std::unordered_map<int, void*>* conv_states = nullptr;
    // Optional per-sequence conv-state size metadata.
    // Maps layer_idx -> bytes for one sequence state slice [1, conv_dim, kernel-1].
    std::unordered_map<int, std::size_t>* conv_state_bytes = nullptr;
    // When true, decode recurrent/conv state buffers must already be fully
    // allocated and shape-compatible. Dispatch paths must not allocate/free or
    // mutate state-map topology during decode (required for stable FULL-step
    // CUDA graph capture/replay).
    bool strict_state_buffers = false;

    // Prefill mode: when true, dispatch_rope writes K/V to the KV-cache at ALL
    // positions (not just position seq_lens[i]) AND lets self-attention proceed
    // normally (dispatch_flash_attention uses standard training path, NOT decode path).
    // This populates the KV-cache in O(1) model forwards instead of O(T).
    bool prefill_mode = false;
    // Absolute token offset for chunked prefill writes.
    // During prompt prefill with chunking, kernels receive T=chunk_len and must
    // write at [prefill_pos_offset, prefill_pos_offset + T) in KV-cache.
    int prefill_pos_offset = 0;

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

    // ========================================================================
    // Flat-token mode: mixed prefill+decode in a single forward pass.
    // When flat_token_mode=true:
    //   - mB=1, mT=total_tokens (all tokens flattened)
    //   - dispatch_rope writes KV via kv_cache_store_flat_paged_bf16
    //     using token_to_req_gpu and kv_write_pos_gpu
    //   - dispatch_flash_attention uses attention_flat_paged_flashinfer
    //     with q_indptr_gpu for per-request Q boundaries
    //   - dispatch_fused_lm_head_loss writes logits for all total_tokens
    //     positions; the caller samples from logits_indices_gpu
    // ========================================================================
    bool flat_token_mode = false;
    int flat_batch_size = 0;            // Number of requests in the flat batch
    int flat_total_tokens = 0;          // Total Q tokens across all requests

    // Per-token → request mapping [flat_total_tokens]
    int32_t* token_to_req_gpu = nullptr;
    // Per-token KV write position [flat_total_tokens]
    int32_t* kv_write_pos_gpu = nullptr;
    // Cumulative Q token counts [flat_batch_size + 1]
    int32_t* q_indptr_gpu = nullptr;
    // Per-request total KV length (after this step) [flat_batch_size]
    int32_t* seq_lens_k_gpu = nullptr;
    // Total Q tiles for FlashInfer grid dispatch (computed on CPU from q_indptr).
    int flat_padded_batch_size = 0;

    // LM head optimization: in flat-token mode, we gather last-token hidden
    // states per request and run a [batch_size, C] matmul instead of
    // [total_tokens, C]. These indices mark the position in xF_flat to gather.
    // If non-null, the LM head gathers from xF_flat at these indices first.
    int32_t* flat_last_token_indices_gpu = nullptr;  // [flat_batch_size]

    // Pre-computed FlashInfer PrefillPlan (set once per flat_step, reused across layers).
    int32_t* flat_page_indptr_gpu = nullptr;
    int32_t* flat_page_indices_gpu = nullptr;
    int32_t* flat_last_page_len_gpu = nullptr;
    void* flat_plan_int_ws_gpu = nullptr;
    void* flat_plan_float_ws_gpu = nullptr;
    // Opaque storage for flashinfer::PrefillPlanInfo (avoid pulling FlashInfer
    // headers into every TU that includes decode_state.h).
    static constexpr std::size_t kPlanInfoSize = 256;
    alignas(8) char flat_plan_info_storage[kPlanInfoSize] = {};
};

}  // namespace infer

#endif  // SUROGATE_SRC_RUNTIME_INFER_DECODE_STATE_H
