// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Attention kernel declarations. Grouped alongside the attention backend
// abstraction (see attention_backend.h). Implementations live in
// ``runtime/attention/`` next to this header:
//   - attention_custom.cu         : in-tree flash/SWA kernel (FP32/BF16)
//   - attention_cudnn.cpp         : cuDNN SDPA
//   - flash_attn_varlen.cpp       : DAO-AiLab flash-attention varlen wrapper
//   - flash_attn_scatter.cu       : dQ/dKV scatter + GQA reduce helpers

#ifndef SUROGATE_SRC_RUNTIME_ATTENTION_ATTENTION_KERNELS_H
#define SUROGATE_SRC_RUNTIME_ATTENTION_ATTENTION_KERNELS_H

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasContext* cublasHandle_t;

struct Tensor;

void attention_forward_cudnn(nv_bfloat16* out,        // output: (B, T, Nq, HS)
                             float* stats,            // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace,
                             cudnnHandle_t handle,
                             int B,
                             int T,
                             int Hq,
                             int Hkv,
                             int HS,
                             cudaStream_t stream);

void attention_forward_cudnn(float* out,        // output: (B, T, Nq, HS)
                             float* stats,      // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace,
                             cudnnHandle_t handle,
                             int B,
                             int T,
                             int Hq,
                             int Hkv,
                             int HS,
                             cudaStream_t stream);

void attention_forward_cudnn(Tensor& out,        // output: (B, T, Nq, HS)
                             Tensor& stats,      // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             Tensor& workspace,
                             cudnnHandle_t handle,
                             int B,
                             int T,
                             int Hq,
                             int Hkv,
                             int HS,
                             cudaStream_t stream);

// Custom (non-cuDNN) attention forward using the in-tree kernel (supports FP32/BF16).
// `scale` overrides the softmax scale. 0.0f → 1/sqrt(HS) (default SDPA).
// Non-zero → used verbatim (e.g., Gemma4 passes 1.0 because Q/K-norm already
// produces unit-RMS Q and K).
void attention_forward_custom(Tensor& out,        // output: (B, T, Nq, HS)
                              Tensor& stats,      // output for backward pass: (B, Hq, T)
                              const Tensor& inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              int window_size,
                              cudaStream_t stream,
                              float scale = 0.0f);
// Custom (non-cuDNN) attention backward using the in-tree kernel (supports FP32/BF16).
void attention_backward_custom(Tensor& dqkv,
                               const Tensor& stats,
                               const Tensor& out,
                               const Tensor& dout,
                               const Tensor& qkv,
                               int B,
                               int T,
                               int Hq,
                               int Hkv,
                               int HS,
                               int window_size,
                               cudaStream_t stream,
                               float scale = 0.0f);

// cuBLAS matmul-based attention (SDPA math equivalent). No head_dim limit.
// Computes: scores = Q @ K^T * scale, causal mask, softmax, out = attn @ V.
// Uses temporary memory for the (B, Hq, T, T) attention matrix.
// `scale` overrides the softmax scale. 0.0f (default) → 1/sqrt(HS) (standard
// SDPA). Non-zero → used verbatim (e.g., Gemma4 passes 1.0 because Q/K-norm
// already produces unit-RMS Q and K).
void attention_forward_matmul(Tensor& out,
                              Tensor& stats,
                              const Tensor& qkv,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cublasHandle_t cublas,
                              cudaStream_t stream,
                              float scale = 0.0f,
                              int window_size = 0);

// cuBLAS matmul-based attention backward. Matches forward_matmul precision.
void attention_backward_matmul(Tensor& d_qkv,
                               const Tensor& lse,
                               const Tensor& out,
                               const Tensor& d_out,
                               const Tensor& qkv,
                               int B,
                               int T,
                               int Hq,
                               int Hkv,
                               int HS,
                               cublasHandle_t cublas,
                               cudaStream_t stream,
                               float scale = 0.0f,
                               int window_size = 0);

std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle);
void attention_backward_cudnn(nv_bfloat16* dqkv,
                              const float* stats,
                              const nv_bfloat16* out,
                              const nv_bfloat16* dout,
                              const nv_bfloat16* qkv,
                              std::byte* workspace,
                              cudnnHandle_t handle,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cudaStream_t stream);
void attention_backward_cudnn(Tensor& dqkv,
                              const Tensor& stats,
                              const Tensor& out,
                              const Tensor& dout,
                              const Tensor& qkv,
                              Tensor& workspace,
                              cudnnHandle_t handle,
                              int B,
                              int T,
                              int Hq,
                              int Hkv,
                              int HS,
                              cudaStream_t stream);

// Flash Attention varlen forward (for document-level attention masking in packed sequences).
// Q/K/V read from interleaved qkv buffer via strides; output written to out (total_q, Hq, HS).
// LSE written in unpadded (Hq, total_q) format.
// `scale` overrides the softmax scale. 0.0f → 1/sqrt(HS) (default); non-zero → used verbatim.
// `window_size > 0` enables sliding-window (local) attention: each token
// attends to its `window_size` most recent tokens (including itself),
// matching HF's sliding_window convention. 0 → full causal (default).
void attention_forward_flash_varlen(nv_bfloat16* out,
                                    float* lse,
                                    const nv_bfloat16* qkv,
                                    const int32_t* cu_seqlens_gpu,
                                    int B_ragged,
                                    int max_seqlen,
                                    int total_q,
                                    int Hq,
                                    int Hkv,
                                    int HS,
                                    cudaStream_t stream,
                                    float scale = 0.0f,
                                    int window_size = 0);

// Flash Attention varlen backward.
// dq_accum: (total_q + 128*B_ragged, Hq, HS_rounded) FP32 temp.
// dsoftmax_sum: (Hq, total_q + 128*B_ragged) FP32 temp.
// dk_expanded/dv_expanded: (total_q, Hq, HS) BF16 temps for GQA (Hq != Hkv).
//   When Hq != Hkv, these must be non-null. The backward kernel writes dK/dV with
//   Hq head indices; these buffers are then reduced to Hkv heads and scattered
//   into the K/V sections of interleaved dqkv.
//   When Hq == Hkv (MHA), pass nullptr — dK/dV are written directly to dqkv.
// `scale` must match the forward scale used for this tensor (0.0f → 1/sqrt(HS) default).
void attention_backward_flash_varlen(nv_bfloat16* dqkv,
                                     const float* lse,
                                     const nv_bfloat16* out,
                                     const nv_bfloat16* dout,
                                     const nv_bfloat16* qkv,
                                     const int32_t* cu_seqlens_gpu,
                                     float* dq_accum,
                                     float* dsoftmax_sum,
                                     nv_bfloat16* dk_expanded,
                                     nv_bfloat16* dv_expanded,
                                     int B_ragged,
                                     int max_seqlen,
                                     int total_q,
                                     int Hq,
                                     int Hkv,
                                     int HS,
                                     bool deterministic,
                                     cudaStream_t stream,
                                     float scale = 0.0f,
                                     int window_size = 0);

// ---------------------------------------------------------------------------
// Chunked-sequence (KV-prefix) attention. q = current chunk (T rows of the
// interleaved qkv buffer), k/v = contiguous caches holding the whole prefix
// INCLUDING this chunk (kv_len rows, (kv_len, Hkv, HS) each). Causal
// alignment is bottom-right; sliding windows count from the aligned diagonal.
// ---------------------------------------------------------------------------

// Copy the chunk's (post-rope) K/V heads from interleaved qkv into the
// contiguous caches at row offset `pos`.
void append_kv_to_cache(nv_bfloat16* k_cache,
                        nv_bfloat16* v_cache,
                        const nv_bfloat16* qkv,
                        int pos,
                        int T,
                        int Hq,
                        int Hkv,
                        int HS,
                        cudaStream_t stream);

void attention_forward_flash_kvprefix(nv_bfloat16* out,
                                      float* lse,
                                      const nv_bfloat16* qkv,
                                      const nv_bfloat16* k_cache,
                                      const nv_bfloat16* v_cache,
                                      const int32_t* cu_seqlens_q_gpu,
                                      const int32_t* cu_seqlens_k_gpu,
                                      int T,
                                      int kv_len,
                                      int Hq,
                                      int Hkv,
                                      int HS,
                                      cudaStream_t stream,
                                      float scale = 0.0f,
                                      int window_size = 0,
                                      int num_segs = 1,
                                      int max_seqlen_q = 0,
                                      int max_seqlen_k = 0);

// Backward for one chunk against its full prefix. Writes dQ into the chunk's
// interleaved dqkv; reduces the prefix-wide expanded dK/dV (kv_len, Hq, HS)
// into the FP32 accumulators (kv_len, Hkv, HS, caller-zeroed at step start),
// then emits accumulator rows [chunk_pos, chunk_pos+T) — complete once
// chunks run last-to-first — into dqkv's K/V sections.
// dq_accum/dsoftmax_sum must be zeroed by the caller (as with the dense
// varlen backward); dk/dv_expanded are (kv_len, Hq, HS) BF16 temps.
void attention_backward_flash_kvprefix(nv_bfloat16* dqkv,
                                       float* dk_accum,
                                       float* dv_accum,
                                       const float* lse,
                                       const nv_bfloat16* out,
                                       const nv_bfloat16* dout,
                                       const nv_bfloat16* qkv,
                                       const nv_bfloat16* k_cache,
                                       const nv_bfloat16* v_cache,
                                       const int32_t* cu_seqlens_q_gpu,
                                       const int32_t* cu_seqlens_k_gpu,
                                       float* dq_accum,
                                       float* dsoftmax_sum,
                                       nv_bfloat16* dk_expanded,
                                       nv_bfloat16* dv_expanded,
                                       int chunk_pos,
                                       int T,
                                       int kv_len,
                                       int Hq,
                                       int Hkv,
                                       int HS,
                                       bool deterministic,
                                       cudaStream_t stream,
                                       float scale = 0.0f,
                                       int window_size = 0,
                                       int num_segs = 1,
                                       int max_seqlen_q = 0,
                                       int max_seqlen_k = 0);

// Reduce Hq-expanded dK/dV over the whole prefix into FP32 Hkv accumulators.
void accum_add_dkv(float* dk_accum,
                   float* dv_accum,
                   const nv_bfloat16* dk_expanded,
                   const nv_bfloat16* dv_expanded,
                   int kv_len,
                   int Hq,
                   int Hkv,
                   int HS,
                   cudaStream_t stream);

// Emit completed accumulator rows [pos, pos+T) into interleaved dqkv (BF16).
void emit_chunk_dkv(nv_bfloat16* dqkv,
                    const float* dk_accum,
                    const float* dv_accum,
                    int pos,
                    int T,
                    int Hq,
                    int Hkv,
                    int HS,
                    cudaStream_t stream);

// Reduce dk_expanded/dv_expanded (Hq heads) to Hkv KV heads and scatter
// into the K/V sections of interleaved dqkv buffer (total_q, Hq+2*Hkv, HS).
void reduce_scatter_dkv(nv_bfloat16* dqkv,
                        const nv_bfloat16* dk_expanded,
                        const nv_bfloat16* dv_expanded,
                        int total_q,
                        int Hq,
                        int Hkv,
                        int HS,
                        cudaStream_t stream);

// Apply attention sinks: scale output and update LSE to include sink logits.
void attention_apply_sinks(nv_bfloat16* out,
                           float* lse,
                           const nv_bfloat16* sinks,
                           int B,
                           int T,
                           int Hq,
                           int Hs,
                           cudaStream_t stream);
void attention_apply_sinks(float* out,
                           float* lse,
                           const float* sinks,
                           int B,
                           int T,
                           int Hq,
                           int Hs,
                           cudaStream_t stream);
void attention_sinks_backward(float* d_sinks,
                              const nv_bfloat16* out,
                              const nv_bfloat16* dout,
                              const float* lse,
                              const nv_bfloat16* sinks,
                              int B,
                              int T,
                              int Hq,
                              int Hs,
                              cudaStream_t stream);
void attention_sinks_backward(float* d_sinks,
                              const float* out,
                              const float* dout,
                              const float* lse,
                              const float* sinks,
                              int B,
                              int T,
                              int Hq,
                              int Hs,
                              cudaStream_t stream);

#endif  // SUROGATE_SRC_RUNTIME_ATTENTION_ATTENTION_KERNELS_H
