// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_KERNELS_KERNELS_H
#define SUROGATE_SRC_KERNELS_KERNELS_H

#include <cstdint>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

struct cudaDeviceProp;
typedef struct cudnnContext* cudnnHandle_t;
typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct cublasContext* cublasHandle_t;

struct Tensor;
enum class ETensorDType: int;

enum class EMMTranspose { TT, TN, NT, NN };

void encoder_forward(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C, int V, cudaStream_t stream);
void encoder_forward(nv_bfloat16* out, const int* inp, const nv_bfloat16* wte, const nv_bfloat16* wpe, int B, int T, int C, int V, cudaStream_t stream);
void encoder_forward(Tensor& out, const Tensor& inp, const Tensor& wte, std::optional<Tensor> wpe, int B, int T, int C, int V, cudaStream_t stream);

void encoder_backward(float* dwte, int* scratch,
                      int* workload_indices, int4* bucket_info,
                      const float* dout, const int* inp, const int* inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);
void encoder_backward(nv_bfloat16* dwte, int* scratch,
                      int* workload_indices, int4* bucket_info,
                      const nv_bfloat16* dout, const int* inp, const int* inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);

// The kernel runs on `stream`, but the bucket info that gets generated on CPU to enable efficient determinism
// can be copied using `copy_stream`, so the kernel launch does not have to wait.
void encoder_backward(Tensor& dwte, Tensor& scratch,
                      Tensor& workload_indices, Tensor& bucket_info,
                      const Tensor& dout, const Tensor& inp, const Tensor& inputs_cpu,
                      int B, int T, int C, unsigned int seed, cudaStream_t stream, cudaEvent_t sync_event, cudaStream_t copy_stream);

void rmsnorm_forward(float* out, float* rms, const float* inp, const float* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);
void rmsnorm_forward(nv_bfloat16* out, float* rms, const nv_bfloat16* inp, const nv_bfloat16* weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);
void rmsnorm_forward(Tensor& out, Tensor& rms, const Tensor& inp, const Tensor& weight, float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);

void rmsnorm_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr, float* rms, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);
void rmsnorm_forward_quant(Tensor& out, float* scale_ptr, Tensor& rms, const Tensor& inp, const Tensor& weight, const float* abs_max_ptr, float epsilon, int B, int T, int C, cudaStream_t stream);

int get_rmsnorm_backward_scratch_size(int C, const cudaDeviceProp& dp);
void rmsnorm_backward(float* dinp, float* dweight, std::byte* scratch, const float* dresidual, const float* dout, const float* inp, const float* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad = false);
void rmsnorm_backward(nv_bfloat16* dinp, nv_bfloat16* dweight, std::byte* scratch, const nv_bfloat16* dresidual, const nv_bfloat16* dout, const nv_bfloat16* inp, const nv_bfloat16* weight, const float* rstd, float* abs_max_ptr,
                      int B, int T, int C, const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad = false);
void rmsnorm_backward(Tensor& dinp, Tensor& dweight, Tensor& scratch, const Tensor& dresidual, const Tensor& dout, const Tensor& inp, const Tensor& weight, const Tensor& rstd, float* abs_max_ptr,
                      int B, int T, int C,  const cudaDeviceProp& dp, cudaStream_t stream, bool skip_weight_grad = false);

void fused_residual_rmsnorm_forward(float* residual, float* normed, float* rrms, const float* inp1, const float* inp2, const float* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);
void fused_residual_rmsnorm_forward(nv_bfloat16* residual, nv_bfloat16* normed, float* rrms, const nv_bfloat16* inp1, const nv_bfloat16* inp2, const nv_bfloat16* weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);
void fused_residual_rmsnorm_forward(Tensor& residual, Tensor& normed, Tensor& rrms, const Tensor& inp1, const Tensor& inp2, const Tensor& weight, float* abs_max_ptr,
                                    float epsilon, int N, int C, cudaStream_t stream);

// Head-wise RMSNorm over packed QKV buffers (used for Qwen3-style Q/K norm).
// Operates in-place on @p qkv, normalizing vectors of length @p head_size for each (token, head)
// starting at @p channel_offset within the packed last dimension.
void qkv_head_rmsnorm_forward(Tensor& qkv, Tensor& rstd, const Tensor& weight,
                              float epsilon, int B, int T, int qkv_channels,
                              int num_heads, int head_size, int channel_offset,
                              cudaStream_t stream);
void qkv_head_rmsnorm_backward_dx(Tensor& d_qkv, const Tensor& qkv_out, const Tensor& weight, const Tensor& rstd,
                                  int B, int T, int qkv_channels,
                                  int num_heads, int head_size, int channel_offset,
                                  cudaStream_t stream);
void qkv_head_rmsnorm_backward_dweight(Tensor& d_weight, const Tensor& d_qkv, const Tensor& qkv_out, const Tensor& weight,
                                       int B, int T, int qkv_channels,
                                       int num_heads, int head_size, int channel_offset,
                                       bool accumulate, cudaStream_t stream);

// Fused QK RMSNorm + RoPE for Qwen3-style Q/K norm.
// Forward: computes per-head RMSNorm (with weight) on Q and K heads and applies RoPE rotation in-place on @p qkv.
// Backward: avoids separate RoPE backward by applying inverse RoPE inside the RMSNorm backward kernels.
void qkv_qk_norm_rope_forward(Tensor& qkv,
                              Tensor& q_rstd, Tensor& k_rstd,
                              const Tensor& q_weight, const Tensor& k_weight,
                              const Tensor& freqs_cis, const int* position_ids,
                              float epsilon, int B, int T, int Hq, int Hkv, int head_size,
                              cudaStream_t stream);

void qkv_head_rmsnorm_rope_backward_dx(Tensor& d_qkv, const Tensor& qkv_rope, const Tensor& weight, const Tensor& rstd,
                                       const Tensor& freqs_cis, const int* position_ids,
                                       int B, int T, int qkv_channels,
                                       int num_heads, int head_size, int channel_offset,
                                       cudaStream_t stream);

void qkv_head_rmsnorm_rope_backward_dweight(Tensor& d_weight, const Tensor& d_qkv, const Tensor& qkv_rope, const Tensor& weight,
                                            const Tensor& freqs_cis, const int* position_ids,
                                            int B, int T, int qkv_channels,
                                            int num_heads, int head_size, int channel_offset,
                                            bool accumulate, cudaStream_t stream);

void matmul(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul_strided_c(float* c, const float* a, const float* b, const float* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream);

void matmul(float* c, const nv_bfloat16* a, const nv_bfloat16* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const float* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(float* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul_strided_c(nv_bfloat16* c, const nv_bfloat16* a, const nv_bfloat16* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream);

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e4m3* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(nv_bfloat16* c, const __nv_fp8_e4m3* a, const __nv_fp8_e5m2* b, const nv_bfloat16* bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, std::byte* workspace, std::size_t workspace_size,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

void matmul(Tensor& c, const Tensor& a, const Tensor& b, std::optional<Tensor> bias, const float* scale_a, const float* scale_b,
            cublasLtHandle_t handle, Tensor& workspace,
            int M, int N, int K, EMMTranspose mode, bool accumulate, cudaStream_t stream);

// Like `matmul`, but allows overriding the output leading dimension (stride between columns)
// for the destination matrix in column-major cuBLASLt terms. Useful for writing into slices
// of a larger fused output (e.g. QKV, gate+up).
void matmul_strided_c(Tensor& c, const Tensor& a, const Tensor& b, std::optional<Tensor> bias, const float* scale_a, const float* scale_b,
                      cublasLtHandle_t handle, Tensor& workspace,
                      int M, int N, int K, EMMTranspose mode, bool accumulate, int ldc, cudaStream_t stream);

void add_bias(float* out, const float* bias, int B, int T, int OC, cudaStream_t stream);
void add_bias(nv_bfloat16* out, const nv_bfloat16* bias, int B, int T, int OC, cudaStream_t stream);
int get_bias_backward_scratch_size(ETensorDType dtype, int OC, const cudaDeviceProp& dp);
void backward_bias(float* dbias, const float* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const nv_bfloat16* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e4m3* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(nv_bfloat16* dbias, const __nv_fp8_e5m2* dout, const float* scale_a, const float* scale_b, float* dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);
void backward_bias(Tensor& dbias, const Tensor& dout, const float* scale_a, const float* scale_b, Tensor& dbias_buffer, int B, int T, int OC, const cudaDeviceProp& dp, cudaStream_t stream);


void precompute_freqs_cis(float *freqs_cis, int dim, int end, float theta);
void precompute_freqs_cis(nv_bfloat16 *freqs_cis, int dim, int end, float theta);
// Full RoPE (rotary_dim == head_dim)
void rope_forward(float* out, const float* in, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(float* dinp, const float* dout, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);

// Partial RoPE (rotary_dim < head_dim, for GLM4 style partial_rotary_factor)
// Only dimensions [0, rotary_dim) are rotated; [rotary_dim, head_dim) pass through unchanged
void rope_forward(float* out, const float* in, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);
void rope_forward(nv_bfloat16* out, const nv_bfloat16* in, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);
void rope_forward(Tensor& out, const Tensor& in, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);
void rope_backward(float* dinp, const float* dout, const float *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);
void rope_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16 *freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);
void rope_backward(Tensor& dinp, const Tensor& dout, const Tensor& freqs_cis, const int* position_ids, float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, int rotary_dim, cudaStream_t stream);

void rope_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* inp, const nv_bfloat16* freqs_cis, const int* position_ids, const float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_forward_quant(Tensor& out, float* scale_ptr, const Tensor& inp, const Tensor& freqs_cis, const int* position_ids, const float* abs_max_ptr, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);

// Fused RoPE kernels with shared memory cos/sin caching (TransformerEngine-style optimization).
// These compute cos/sin on-the-fly via sincosf() and cache in shared memory, eliminating the
// need for precomputed freqs_cis tensors and reducing global memory bandwidth.
void rope_fused_forward(float* out, const float* inp, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_fused_forward(nv_bfloat16* out, const nv_bfloat16* inp, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_fused_forward(Tensor& out, const Tensor& inp, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_fused_backward(float* dinp, const float* dout, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_fused_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);
void rope_fused_backward(Tensor& dinp, const Tensor& dout, const int* position_ids, float* abs_max_ptr, float theta, int B, int T, int Nq, int Nkv, int head_dim, cudaStream_t stream);

// swiglu assumes that input is the concatenation of up and gate projection.
void swiglu_forward(nv_bfloat16* out, const nv_bfloat16* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward(float* out, const float* inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward(Tensor& out, const Tensor& inp, float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);

void swiglu_forward_quant(__nv_fp8_e4m3* out, float* scale_ptr,const nv_bfloat16* inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);
void swiglu_forward_quant(Tensor& out, float* scale_ptr, const Tensor& inp, const float* abs_max_ptr, int B, int T, int C, cudaStream_t stream);

void swiglu_backward(nv_bfloat16* dinp, const nv_bfloat16* dout, const nv_bfloat16* inp, float* abs_max, int B, int T, int C, cudaStream_t stream);
void swiglu_backward(float* dinp, const float* dout, const float* inp, float* abs_max, int B, int T, int C, cudaStream_t stream);
void swiglu_backward(Tensor& dinp, const Tensor& dout, const Tensor& inp, float* abs_max, int B, int T, int C, cudaStream_t stream);

// ============================================================================
// Fast LoRA SiLU kernels for MoE expert optimization
// ============================================================================

/// @brief Compute h = silu(e) * g from separate gate and up outputs.
/// Unlike swiglu_forward which takes concatenated [up, gate], this takes separate tensors.
/// @param h Output tensor (N, D).
/// @param e Gate projection output (N, D).
/// @param g Up projection output (N, D).
/// @param N Number of tokens.
/// @param D Intermediate dimension.
/// @param stream CUDA stream.
void silu_mul_forward(nv_bfloat16* h, const nv_bfloat16* e, const nv_bfloat16* g, int N, int D, cudaStream_t stream);
void silu_mul_forward(float* h, const float* e, const float* g, int N, int D, cudaStream_t stream);
void silu_mul_forward(Tensor& h, const Tensor& e, const Tensor& g, int N, int D, cudaStream_t stream);

/// @brief In-place backward through SiLU multiplication for fast LoRA.
/// Computes de, dg from dh and overwrites e->de, g->dg IN-PLACE.
/// Optionally outputs reconstructed h for down LoRA gradient computation.
/// @param e Gate output (N, D) - modified in-place to de.
/// @param g Up output (N, D) - modified in-place to dg.
/// @param dh Upstream gradient (N, D).
/// @param h_out Optional output for reconstructed h (can be nullptr).
/// @param N Number of tokens.
/// @param D Intermediate dimension.
/// @param stream CUDA stream.
void silu_mul_backward_inplace(nv_bfloat16* e, nv_bfloat16* g, const nv_bfloat16* dh, nv_bfloat16* h_out, int N, int D, cudaStream_t stream);
void silu_mul_backward_inplace(float* e, float* g, const float* dh, float* h_out, int N, int D, cudaStream_t stream);
void silu_mul_backward_inplace(Tensor& e, Tensor& g, const Tensor& dh, Tensor* h_out, int N, int D, cudaStream_t stream);

/// @brief Split (N, 2D) gate_up tensor into separate (N, D) up and gate tensors.
/// Layout: gate_up = [up | gate], up is columns [0,D), gate is columns [D,2D).
/// @param gate_up Input tensor (N, 2D).
/// @param up Output up projection (N, D).
/// @param gate Output gate projection (N, D).
/// @param N Number of tokens.
/// @param D Intermediate dimension.
/// @param stream CUDA stream.
void split_gate_up(const nv_bfloat16* gate_up, nv_bfloat16* up, nv_bfloat16* gate, int N, int D, cudaStream_t stream);
void split_gate_up(const float* gate_up, float* up, float* gate, int N, int D, cudaStream_t stream);
void split_gate_up(const Tensor& gate_up, Tensor& up, Tensor& gate, int N, int D, cudaStream_t stream);

/// @brief Concatenate (N, D) dg and de tensors into (N, 2D) d_gate_up.
/// Layout: d_gate_up = [dg | de], dg is columns [0,D), de is columns [D,2D).
/// @param dg Gradient w.r.t. up output (N, D).
/// @param de Gradient w.r.t. gate output (N, D).
/// @param d_gate_up Output tensor (N, 2D).
/// @param N Number of tokens.
/// @param D Intermediate dimension.
/// @param stream CUDA stream.
void concat_d_gate_up(const nv_bfloat16* dg, const nv_bfloat16* de, nv_bfloat16* d_gate_up, int N, int D, cudaStream_t stream);
void concat_d_gate_up(const float* dg, const float* de, float* d_gate_up, int N, int D, cudaStream_t stream);
void concat_d_gate_up(const Tensor& dg, const Tensor& de, Tensor& d_gate_up, int N, int D, cudaStream_t stream);

void attention_forward_cudnn(nv_bfloat16* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_forward_cudnn(float* out,  // output: (B, T, Nq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const float* inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void attention_forward_cudnn(Tensor& out,  // output: (B, T, Nq, HS)
                             Tensor& stats, // output for backward pass: (B, Hq, T)
                             const Tensor& inp,  // input: (B, T, Hq + 2Hkv, HS) QKV
                             Tensor& workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle);
void attention_backward_cudnn(nv_bfloat16* dqkv, const float* stats,
                              const nv_bfloat16* out, const nv_bfloat16* dout, const nv_bfloat16* qkv,
                              std::byte* workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);
void attention_backward_cudnn(Tensor& dqkv, const Tensor& stats,
                              const Tensor& out, const Tensor& dout, const Tensor& qkv,
                              Tensor& workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream);

void fused_classifier(float* logits, float* losses,
                      float dloss, const int* targets, int* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(float* logits, float* losses,
                      float dloss, const int* targets, int* valid_token_count,
                      int* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(nv_bfloat16* logits, float* losses,
                      float dloss, const int* targets, int* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(nv_bfloat16* logits, float* losses,
                      float dloss, const int* targets, int* valid_token_count,
                      int* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(Tensor& logits, Tensor& losses,
                      float dloss, const Tensor& targets, Tensor* valid_token_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);
void fused_classifier(Tensor& logits, Tensor& losses,
                      float dloss, const Tensor& targets, Tensor* valid_token_count,
                      Tensor* correct_count,
                      int BT, int V, int P, bool write_dlogits, cudaStream_t stream);

int get_max_num_block_sums(const cudaDeviceProp& dp);
void global_norm_squared(float* out, const float* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
void global_norm_squared(float* out, const nv_bfloat16* values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
void global_norm_squared(Tensor& out, const Tensor& values, size_t count, const cudaDeviceProp& dp, cudaStream_t stream);
/// puts norm squared in out[0], norm in out_cpu, and grad scale factor in out[1]
// Computes the final gradient norm and produces a single scalar multiplier that will be applied to gradients:
// - rescales for masked-token normalization using `valid_token_count` + `total_tokens` when provided
// - applies gradient clipping against the *scaled* norm
//
// Writes:
// - out[1] := total multiplier to apply to gradients (may be > 1 when many tokens are masked)
// - out_cpu := scaled gradient norm (norm * token_scale), for logging
void global_norm_sqrt(float* out, float* out_cpu, float grad_clip,
                      const int* valid_token_count, float total_tokens,
                      const cudaDeviceProp& dp, cudaStream_t stream);

void deterministic_sum(float* out, const float* values, std::size_t count, cudaStream_t stream);
void deterministic_sum(float* out, const nv_bfloat16* values, std::size_t count, cudaStream_t stream);


// 8-bit AdamW optimizer - functions moved to modules/optimizers/adamw_8bit.h
// Bring them into global namespace for backward compatibility
#include "modules/optimizers/adamw_8bit.h"

using optimizers::adamw_update_8bit;
using optimizers::adamw_update_8bit_multi_tensor;
using optimizers::init_adamw8bit_state;
using optimizers::create_adamw8bit_quantiles1;
using optimizers::create_adamw8bit_quantiles2;

// quantization
void abs_max(float* scale, const float* in, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void abs_max(float* scale, const Tensor& in, long N, const cudaDeviceProp& dp, cudaStream_t stream);

void quantize_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const float* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, long N, const cudaDeviceProp& dp, cudaStream_t stream);

void quantize_and_transpose_with_abs_max(nv_bfloat16* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(std::int8_t* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e4m3* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const float* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(__nv_fp8_e5m2* out, float* scale_ptr, const nv_bfloat16* in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_and_transpose_with_abs_max(Tensor& out, float* scale_ptr, const Tensor& in, const float* abs_max, int rows, int cols, const cudaDeviceProp& dp, cudaStream_t stream);

// Delayed scaling quantization (TransformerEngine-style)
// Quantizes using a pre-computed (delayed) scale from previous iteration, while recording
// the current abs_max for future scale computation. Used in FP8 HYBRID mode.
void quantize_with_delayed_scale(__nv_fp8_e4m3* out, float* recorded_amax, float* inv_scale_ptr, const nv_bfloat16* in, const float* delayed_scale, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_delayed_scale(__nv_fp8_e5m2* out, float* recorded_amax, float* inv_scale_ptr, const nv_bfloat16* in, const float* delayed_scale, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_delayed_scale(__nv_fp8_e4m3* out, float* recorded_amax, float* inv_scale_ptr, const float* in, const float* delayed_scale, long N, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_with_delayed_scale(__nv_fp8_e5m2* out, float* recorded_amax, float* inv_scale_ptr, const float* in, const float* delayed_scale, long N, const cudaDeviceProp& dp, cudaStream_t stream);

void transpose(float* dst, const float* src, int rows, int cols, cudaStream_t stream);
void transpose(__nv_fp8_e4m3* dst, const __nv_fp8_e4m3* src, int rows, int cols, cudaStream_t stream);
void transpose(__nv_fp8_e5m2* dst, const __nv_fp8_e5m2* src, int rows, int cols, cudaStream_t stream);
void transpose(nv_bfloat16* dst, const nv_bfloat16* src, int rows, int cols, cudaStream_t stream);
void transpose(Tensor& dst, const Tensor& src, int rows, int cols, cudaStream_t stream);

void vector_add_sr(float* dest, const float* left, const float* right, float scale, long nelem, unsigned seed, cudaStream_t stream);
void vector_add_sr(nv_bfloat16* dest, const nv_bfloat16* left, const nv_bfloat16* right, float scale, long nelem, unsigned seed, cudaStream_t stream);
void vector_add_sr(Tensor& dest, const Tensor& left, const Tensor& right, float scale, long nelem, unsigned seed, cudaStream_t stream);

// Add a packed (rows x src_cols) matrix into a strided destination matrix (rows x dst_cols),
// offsetting the destination columns by dst_col_offset.
void add_2d_slice(float* dst, const float* src, long rows, long dst_cols, long src_cols, long dst_col_offset, cudaStream_t stream);
void add_2d_slice(nv_bfloat16* dst, const nv_bfloat16* src, long rows, long dst_cols, long src_cols, long dst_col_offset, cudaStream_t stream);
void add_2d_slice(Tensor& dst, const Tensor& src, long rows, long dst_cols, long src_cols, long dst_col_offset, cudaStream_t stream);

//! \fn void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
//! \brief Reduce shards of tensor using stochastic rounding
//! \details Interprets `src` as a tensor of `n_shard` shards of size `nelem` each. The shards are summed together, and the result is either written to (`accumulate = false`)
//! or added into (`accumulate = true`) `dest`, after being scaled by `scale`. All intermediate calculations are done in float precision, and stochastic rounding using the
//! provided `seed` is applied before writing to `dest`. The `skip` parameter allows to skip one of the shards. Set to -1 to disable skipping.
void vector_reduce_sr(float* dest, const float* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
void vector_reduce_sr(nv_bfloat16* dest, const nv_bfloat16* src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);
void vector_reduce_sr(Tensor& dest, const Tensor& src, float scale, int n_shards, int skip, long nelem, bool accumulate, unsigned seed, cudaStream_t stream);

void fill_normal(float* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);
void fill_normal(nv_bfloat16* dst, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);
void fill_normal(Tensor& dest, std::size_t count, float mean, float std, unsigned long long seed, unsigned long long subsequence, cudaStream_t stream);

void fill_constant(float* dst, float value, std::size_t count, cudaStream_t stream);
void fill_constant(nv_bfloat16* dst, nv_bfloat16 value, std::size_t count, cudaStream_t stream);
void fill_constant(Tensor& dest, float value, std::size_t count, cudaStream_t stream);

void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size);
void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size);
void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size);
void convert_dtype(float* target, const nv_bfloat16* source, std::size_t size, cudaStream_t stream);
void convert_dtype(nv_bfloat16* target, const float* source, std::size_t size, cudaStream_t stream);
void convert_dtype(nv_bfloat16* target, const half* source, std::size_t size, cudaStream_t stream);

// Per-block quantization (QLoRA-style)
// Quantizes BF16 weights to FP8 E4M3 with per-block scales for memory-efficient QLoRA training.
// Each (block_size x block_size) tile gets its own scale factor.
void quantize_per_block(__nv_fp8_e4m3* out, float* block_scales, const nv_bfloat16* in,
                        int M, int K, int block_size, const cudaDeviceProp& dp, cudaStream_t stream);
void quantize_per_block(Tensor& out, Tensor& block_scales, const Tensor& in,
                        int block_size, const cudaDeviceProp& dp, cudaStream_t stream);

// Per-block dequantization (QLoRA-style)
// Dequantizes FP8 E4M3 weights to BF16 using per-block scales for on-the-fly reconstruction.
void dequantize_per_block(nv_bfloat16* out, const __nv_fp8_e4m3* in, const float* block_scales,
                          int M, int K, int block_size, const cudaDeviceProp& dp, cudaStream_t stream);
void dequantize_per_block(Tensor& out, const Tensor& in, const Tensor& block_scales,
                          int block_size, const cudaDeviceProp& dp, cudaStream_t stream);

// Fused per-block to per-tensor FP8 conversion (QLoRA optimization for FP8 forward mode).
// Converts per-block quantized FP8 to per-tensor FP8 in a single operation, eliminating
// the intermediate BF16 dequantization step. This is used when --qlora-fp8 is combined
// with --recipe=fp8-hybrid to avoid wasteful FP8→BF16→FP8 conversion.
void fused_dequant_requant_per_block_to_tensor(__nv_fp8_e4m3* out, float* out_scale,
                                                const __nv_fp8_e4m3* in, const float* block_scales,
                                                int M, int K, int block_size,
                                                const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// FP4 E2M1 Quantization (NVFP4 format)
// ============================================================================
// Two-level block scaling:
// - Level 1: FP8 E4M3 scale per 16 consecutive values
// - Level 2: FP32 global per-tensor scale
// Block scale shape: (round_to_multiple(M, 128), round_to_multiple(ceil(K/16), 4))
// with F8_128x4 tensor reordering for cuBLAS compatibility.
// Requires: CUDA 12.8+, Blackwell GPU (SM100+) for native FP4 PTX instructions.

/// @brief FP4 block quantization with explicit scales.
/// @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
/// @param[out] block_scales Output FP8 E4M3 block scales (swizzled for cuBLAS).
/// @param[out] global_amax Output per-tensor absolute maximum.
/// @param[in] in Input BF16 data (M, K).
/// @param global_encode_scale Pre-computed global encode scale.
/// @param global_decode_scale Pre-computed global decode scale.
void quantize_fp4_block(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                        const nv_bfloat16* in, int M, int K,
                        float global_encode_scale, float global_decode_scale,
                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 block quantization with auto scale computation.
void quantize_fp4_block_auto_scale(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                    const nv_bfloat16* in, int M, int K,
                                    const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 block quantization with Four Over Six (4/6) adaptive block scaling.
/// For each block, evaluates both max=4 and max=6 scaling and selects the one with lower error.
/// Uses tensor scale 1536 (vs 2688 for standard) for correct dequantization.
/// @param metric Error metric for 4/6 selection (0=MSE, 1=L1, 2=AbsMax)
void quantize_fp4_block_4o6_auto_scale(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                        const nv_bfloat16* in, int M, int K,
                                        int metric,
                                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 weight quantization with column-major scale layout for cuDNN.
/// Produces scales in (K/16, N) layout with F8_128x4 swizzle, as expected by cuDNN B operand.
void quantize_fp4_weight(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                         const nv_bfloat16* in, int N, int K,
                         float global_encode_scale, float global_decode_scale,
                         const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 weight quantization with auto scale computation and column-major scale layout.
void quantize_fp4_weight_auto_scale(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                    const nv_bfloat16* in, int N, int K,
                                    const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 block dequantization.
void dequantize_fp4_block(nv_bfloat16* out, const uint8_t* in_fp4,
                          const __nv_fp8_e4m3* block_scales, float global_decode_scale,
                          int M, int K, const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 block quantization with stochastic rounding (for gradients in backward pass).
void quantize_fp4_block_stochastic(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                    const nv_bfloat16* in, int M, int K,
                                    float global_encode_scale, float global_decode_scale,
                                    unsigned int seed,
                                    const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 block quantization with stochastic rounding and auto scale computation (for gradients).
void quantize_fp4_block_stochastic_auto_scale(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                              const nv_bfloat16* in, int M, int K,
                                              unsigned int seed,
                                              const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 weight quantization with 16x16 block scaling (TransformerEngine NVFP4 recipe).
/// Produces scales in (K/16, N) layout with F8_128x4 swizzle, as expected by cuDNN B operand.
void quantize_fp4_weight_2d(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                            const nv_bfloat16* in, int N, int K,
                            float global_encode_scale, float global_decode_scale,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 weight quantization with 16x16 block scaling and auto scale computation.
void quantize_fp4_weight_2d_auto_scale(uint8_t* out_fp4, __nv_fp8_e4m3* block_scales, float* global_amax,
                                       const nv_bfloat16* in, int N, int K,
                                       const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based FP4 block quantization.
void quantize_fp4_block(Tensor& out_fp4, Tensor& block_scales, Tensor& global_amax,
                        const Tensor& in, float global_encode_scale, float global_decode_scale,
                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based FP4 block dequantization.
void dequantize_fp4_block(Tensor& out, const Tensor& in_fp4, const Tensor& block_scales,
                          float global_decode_scale,
                          const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// FP4 Alpha Scaling (post-matmul correction)
// ============================================================================
// After FP4 matmul with block scale dequantization, the output needs to be
// scaled by alpha = (global_amax_a * global_amax_b) / (FP4_MAX^2 * FP8_MAX^2)
// to get the correct result. This is because:
// - Block scales store: block_amax / FP4_MAX * global_encode_scale
// - global_encode_scale = FP8_MAX * FP4_MAX / global_amax
// - The global amax factor is "divided out" during block scale computation
//   and must be multiplied back after matmul.

/// @brief FP4 alpha scaling with BF16 output.
void fp4_alpha_scale(nv_bfloat16* out, const float* global_amax_a, const float* global_amax_b,
                     long N, const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief FP4 alpha scaling with FP32 output.
void fp4_alpha_scale(float* out, const float* global_amax_a, const float* global_amax_b,
                     long N, const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based FP4 alpha scaling.
void fp4_alpha_scale(Tensor& out, const Tensor& global_amax_a, const Tensor& global_amax_b,
                     const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Compute FP4 alpha factor from two global amax values.
///
/// Computes alpha = (amax_a * amax_b) / (6^2 * 448^2) for use with matmul_cutlass_fp4_alpha().
/// This allows fusing alpha scaling into CUTLASS epilogue instead of a separate kernel.
///
/// @param alpha_out Output alpha value (device pointer to single float)
/// @param global_amax_a Global amax of tensor A (device pointer)
/// @param global_amax_b Global amax of tensor B (device pointer)
/// @param stream CUDA stream
void compute_fp4_alpha(float* alpha_out, const float* global_amax_a, const float* global_amax_b,
                       cudaStream_t stream);

/// @brief Compute FP4 alpha for Four Over Six (4/6) quantization.
///
/// For 4/6 quantization, the tensor scale factor is 1536 (= 384 * 4) instead of
/// 2688 (= 448 * 6). This gives: alpha = (amax_a * amax_b) / (1536 * 1536)
///
/// @param alpha_out Output alpha value (device pointer to single float)
/// @param global_amax_a Global amax of tensor A (device pointer)
/// @param global_amax_b Global amax of tensor B (device pointer)
/// @param stream CUDA stream
void compute_fp4_alpha_4o6(float* alpha_out, const float* global_amax_a, const float* global_amax_b,
                           cudaStream_t stream);

/// @brief Fused FP4 alpha scaling + FP32→BF16 conversion.
///
/// Combines alpha scaling and type conversion into a single kernel, eliminating
/// intermediate FP32 storage and reducing memory traffic. Optimized for datacenter
/// GPUs (B200/H100) where kernel launch overhead is significant.
///
/// @param out_bf16 Output BF16 tensor
/// @param in_f32 Input FP32 tensor (from FP4 matmul)
/// @param global_amax_a Global amax of tensor A (device pointer)
/// @param global_amax_b Global amax of tensor B (device pointer)
/// @param N Number of elements
/// @param dp CUDA device properties
/// @param stream CUDA stream
void fp4_alpha_scale_convert(nv_bfloat16* out_bf16, const float* in_f32,
                             const float* global_amax_a, const float* global_amax_b,
                             long N, const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// Random Hadamard Transform (RHT)
// ============================================================================
// Smooths tensor value distributions before FP4 quantization, reducing the
// impact of outliers. Uses 16x16 block-wise Hadamard transform with random
// diagonal sign matrix for decorrelation.
//
// Algorithm: y = (1/4) * H16 @ D @ x.reshape(-1, 16).T
// Inverse:   x = D @ H16 @ y * (1/4)  (since H16 is symmetric and D^T = D)

/// @brief Random Hadamard Transform forward (BF16).
/// @param[out] out Output BF16 tensor (same shape as input).
/// @param[in] in Input BF16 tensor.
/// @param[out] amax_out Optional output for tracking absolute maximum (can be nullptr).
/// @param M Number of rows.
/// @param K Number of columns (must be multiple of 16).
/// @param seed Random seed for sign matrix generation.
/// @param stream CUDA stream.
void hadamard_transform_forward(nv_bfloat16* out, const nv_bfloat16* in,
                                float* amax_out, int M, int K,
                                unsigned int seed, cudaStream_t stream);

/// @brief Random Hadamard Transform inverse (BF16).
/// @param[out] out Output BF16 tensor.
/// @param[in] in Input BF16 tensor (post-transform).
/// @param M Number of rows.
/// @param K Number of columns (must be multiple of 16).
/// @param seed Random seed (must match forward transform).
/// @param stream CUDA stream.
void hadamard_transform_inverse(nv_bfloat16* out, const nv_bfloat16* in,
                                int M, int K, unsigned int seed, cudaStream_t stream);

/// @brief Random Hadamard Transform forward applied along the row dimension (BF16).
/// Applies independent 16-point RHTs down the M dimension for each column.
/// @param M Number of rows (must be multiple of 16).
/// @param K Number of columns.
void hadamard_transform_forward_rows(nv_bfloat16* out, const nv_bfloat16* in,
                                     float* amax_out, int M, int K,
                                     unsigned int seed, cudaStream_t stream);

/// @brief Random Hadamard Transform inverse applied along the row dimension (BF16).
void hadamard_transform_inverse_rows(nv_bfloat16* out, const nv_bfloat16* in,
                                     int M, int K, unsigned int seed, cudaStream_t stream);

/// @brief Random Hadamard Transform forward (FP32).
void hadamard_transform_forward(float* out, const float* in,
                                float* amax_out, int M, int K,
                                unsigned int seed, cudaStream_t stream);

/// @brief Random Hadamard Transform inverse (FP32).
void hadamard_transform_inverse(float* out, const float* in,
                                int M, int K, unsigned int seed, cudaStream_t stream);

/// @brief Tensor-based Random Hadamard Transform forward.
void hadamard_transform_forward(Tensor& out, const Tensor& in,
                                float* amax_out, int M, int K,
                                unsigned int seed, cudaStream_t stream);

/// @brief Tensor-based Random Hadamard Transform inverse.
void hadamard_transform_inverse(Tensor& out, const Tensor& in,
                                int M, int K, unsigned int seed, cudaStream_t stream);

// ============================================================================
// MX FP8 Block Quantization (SM120 Block-Scaled GEMM)
// ============================================================================
// Per-block quantization compatible with SM120 (Blackwell GeForce) CUTLASS
// block-scaled GEMM operations. Uses MX format with:
// - FP8 E4M3 or E5M2 data values
// - UE8M0 scale factors (8-bit unsigned exponent-only: value = 2^(e-127))
// - 32 elements per scaling group (MX standard)
// Produces scale layout matching CUTLASS Sm1xxBlkScaledConfig for direct use
// with matmul_cutlass_fp8_sm120().

/// @brief Compute the size of MX scale tensor for given matrix dimensions.
/// @param rows Number of rows in the data matrix.
/// @param cols Number of columns in the data matrix.
/// @return Number of UE8M0 scale elements needed.
size_t compute_mx_fp8_scale_size(int rows, int cols);

/// @brief MX FP8 E4M3 quantization with UE8M0 block scales.
/// @param[out] out_fp8 Output FP8 E4M3 data (M, K).
/// @param[out] mx_scales Output UE8M0 block scales in swizzled layout.
/// @param[in] in Input BF16 data (M, K).
/// @param M Number of rows.
/// @param K Number of columns.
void quantize_mx_fp8_e4m3(__nv_fp8_e4m3* out_fp8, uint8_t* mx_scales,
                          const nv_bfloat16* in, int M, int K,
                          const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief MX FP8 E5M2 quantization with UE8M0 block scales.
void quantize_mx_fp8_e5m2(__nv_fp8_e5m2* out_fp8, uint8_t* mx_scales,
                          const nv_bfloat16* in, int M, int K,
                          const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief MX FP8 E4M3 dequantization.
void dequantize_mx_fp8_e4m3(nv_bfloat16* out, const __nv_fp8_e4m3* in_fp8,
                            const uint8_t* mx_scales, int M, int K,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief MX FP8 E5M2 dequantization.
void dequantize_mx_fp8_e5m2(nv_bfloat16* out, const __nv_fp8_e5m2* in_fp8,
                            const uint8_t* mx_scales, int M, int K,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief MX FP8 weight quantization for GEMM B operand.
/// Quantizes weight matrix with scale layout for use as GEMM B operand.
void quantize_mx_fp8_weight(__nv_fp8_e4m3* out_fp8, uint8_t* mx_scales,
                            const nv_bfloat16* in, int N, int K,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based MX FP8 E4M3 quantization.
void quantize_mx_fp8_e4m3(Tensor& out_fp8, Tensor& mx_scales, const Tensor& in,
                          const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based MX FP8 E5M2 quantization.
void quantize_mx_fp8_e5m2(Tensor& out_fp8, Tensor& mx_scales, const Tensor& in,
                          const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based MX FP8 E4M3 dequantization.
void dequantize_mx_fp8_e4m3(Tensor& out, const Tensor& in_fp8, const Tensor& mx_scales,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based MX FP8 E5M2 dequantization.
void dequantize_mx_fp8_e5m2(Tensor& out, const Tensor& in_fp8, const Tensor& mx_scales,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based MX FP8 weight quantization.
void quantize_mx_fp8_weight(Tensor& out_fp8, Tensor& mx_scales, const Tensor& in,
                            const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// FP4 Matmul (cuDNN frontend)
// ============================================================================
// Block-scaled FP4 E2M1 matmul using cuDNN graph API.
// Requires: cuDNN 9.7.0+, Blackwell GPU (SM100+)
//
// Memory layout:
// - A: FP4 packed row-major (M, K), stored as (M, K/2) bytes
// - B: FP4 packed column-major (K, N), stored as (K/2, N) bytes
// - scale_a: FP8 E4M3, F8_128x4 swizzled
// - scale_b: FP8 E4M3, F8_128x4 swizzled

/// @brief Check if device supports FP4 operations (Blackwell SM100+).
bool device_supports_fp4();

/// @brief Get required workspace size for FP4 matmul.
std::size_t fp4_matmul_get_workspace_size(int M, int N, int K, int block_size,
                                           cudnnHandle_t cudnn_handle);

/// @brief FP4 block-scaled matmul with BF16 output.
void fp4_matmul(nv_bfloat16* d, const uint8_t* a, const uint8_t* b,
                const __nv_fp8_e4m3* scale_a, const __nv_fp8_e4m3* scale_b,
                float global_scale_a, float global_scale_b,
                std::byte* workspace, std::size_t workspace_size,
                int M, int N, int K, int block_size,
                cudnnHandle_t cudnn_handle, cudaStream_t stream);

/// @brief FP4 block-scaled matmul with FP32 output.
void fp4_matmul_f32(float* d, const uint8_t* a, const uint8_t* b,
                    const __nv_fp8_e4m3* scale_a, const __nv_fp8_e4m3* scale_b,
                    float global_scale_a, float global_scale_b,
                    std::byte* workspace, std::size_t workspace_size,
                    int M, int N, int K, int block_size,
                    cudnnHandle_t cudnn_handle, cudaStream_t stream);

/// @brief Tensor-based FP4 block-scaled matmul.
void fp4_matmul(Tensor& d, const Tensor& a, const Tensor& b,
                const Tensor& scale_a, const Tensor& scale_b,
                float global_scale_a, float global_scale_b,
                Tensor& workspace, int M, int N, int K, int block_size,
                cudnnHandle_t cudnn_handle, cudaStream_t stream);

// ============================================================================
// NVFP4 Block Quantization (CUTLASS-compatible layout)
// ============================================================================
// Per-block FP4 E2M1 quantization with UE4M3 scales in CUTLASS Sm1xxBlkScaledConfig
// interleaved layout. Compatible with matmul_cutlass_fp4().
//
// Memory layout:
// - Data: FP4 E2M1 packed row-major (M, K), stored as (M, K/2) bytes
// - Scales: UE4M3 (unsigned E4M3) in CUTLASS interleaved layout
// - Block size: 16 elements per scale (NVFP4 standard)

/// @brief Compute size of NVFP4 scale tensor for CUTLASS layout.
/// @param rows Number of rows in the data matrix.
/// @param cols Number of columns in the data matrix.
/// @return Number of UE4M3 scale elements needed.
size_t compute_nvfp4_cutlass_scale_size(int rows, int cols);

/// @brief NVFP4 activation quantization with CUTLASS-compatible scale layout.
/// @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
/// @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
/// @param[in] in Input BF16 data (M, K).
/// @param M Number of rows.
/// @param K Number of columns.
void quantize_nvfp4_cutlass(uint8_t* out_fp4, uint8_t* block_scales,
                             const nv_bfloat16* in, int M, int K,
                             const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 weight quantization with CUTLASS-compatible scale layout.
/// @param[out] out_fp4 Output packed FP4 data (N, K/2 bytes).
/// @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
/// @param[in] in Input BF16 weight data (N, K) row-major.
/// @param N Number of rows (out_channels).
/// @param K Number of columns (in_channels).
void quantize_nvfp4_weight_cutlass(uint8_t* out_fp4, uint8_t* block_scales,
                                    const nv_bfloat16* in, int N, int K,
                                    const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 weight quantization producing a transposed (K, N) output layout.
/// Input is BF16 weight in row-major (N, K); output is FP4 packed row-major (K, N/2 bytes)
/// with CUTLASS-compatible scale layout. Used to avoid an explicit BF16 transpose in backward dgrad.
void quantize_nvfp4_weight_cutlass_transpose_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                                         float* global_amax, const nv_bfloat16* in,
                                                         int N, int K,
                                                         const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 quantization with stochastic rounding (for gradients).
void quantize_nvfp4_stochastic_cutlass(uint8_t* out_fp4, uint8_t* block_scales,
                                        const nv_bfloat16* in, int M, int K,
                                        unsigned int seed,
                                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 quantization with two-level scaling (global + block).
/// Computes global amax first, then bakes it into block scales for proper scaling.
/// The global_amax output is needed for alpha correction after GEMM.
void quantize_nvfp4_cutlass_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                        float* global_amax, const nv_bfloat16* in,
                                        int M, int K,
                                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 quantization using a pre-computed global amax (skips abs_max reduction).
/// Useful on fast GPUs (e.g., B200) where separate amax kernels dominate runtime.
void quantize_nvfp4_cutlass_from_amax(uint8_t* out_fp4, uint8_t* block_scales,
                                       const float* global_amax, const nv_bfloat16* in,
                                       int M, int K,
                                       const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 stochastic quantization with two-level scaling (global + block).
void quantize_nvfp4_stochastic_cutlass_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                                   float* global_amax, const nv_bfloat16* in,
                                                   int M, int K, unsigned int seed,
                                                   const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 stochastic quantization using a pre-computed global amax (skips abs_max reduction).
void quantize_nvfp4_stochastic_cutlass_from_amax(uint8_t* out_fp4, uint8_t* block_scales,
                                                  const float* global_amax, const nv_bfloat16* in,
                                                  int M, int K, unsigned int seed,
                                                  const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 weight quantization with two-level scaling (global + block).
void quantize_nvfp4_weight_cutlass_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                               float* global_amax, const nv_bfloat16* in,
                                               int N, int K,
                                               const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 weight quantization using a pre-computed global amax (skips abs_max reduction).
void quantize_nvfp4_weight_cutlass_from_amax(uint8_t* out_fp4, uint8_t* block_scales,
                                              const float* global_amax, const nv_bfloat16* in,
                                              int N, int K,
                                              const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NVFP4 dequantization with CUTLASS-compatible scale layout.
/// Converts FP4 E2M1 with UE4M3 block scales back to BF16 (fake quantization).
/// @param[out] out Output BF16 data (M, K).
/// @param[in] in_fp4 Input packed FP4 data (M, K/2 bytes).
/// @param[in] block_scales Input UE4M3 block scales in CUTLASS layout.
/// @param M Number of rows.
/// @param K Number of columns (must be even).
void dequantize_nvfp4_cutlass(nv_bfloat16* out, const uint8_t* in_fp4,
                               const uint8_t* block_scales, int M, int K,
                               const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based NVFP4 quantization with CUTLASS layout.
void quantize_nvfp4_cutlass(Tensor& out_fp4, Tensor& block_scales, const Tensor& in,
                             const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based NVFP4 weight quantization with CUTLASS layout.
void quantize_nvfp4_weight_cutlass(Tensor& out_fp4, Tensor& block_scales, const Tensor& in,
                                    const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based NVFP4 stochastic quantization with CUTLASS layout.
void quantize_nvfp4_stochastic_cutlass(Tensor& out_fp4, Tensor& block_scales, const Tensor& in,
                                        unsigned int seed,
                                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based NVFP4 dequantization with CUTLASS layout.
void dequantize_nvfp4_cutlass(Tensor& out, const Tensor& in_fp4,
                               const Tensor& block_scales,
                               const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// Four Over Six (4/6) NVFP4 Quantization (CUTLASS layout)
// ============================================================================
//
// Implements adaptive block scaling from arXiv:2512.02010:
// "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
//
// For each block, evaluates both max=6.0 and max=4.0 scaling, selecting the
// option with lower quantization error. This improves representation of
// near-maximal values where FP4's coarse quantization step (4→6) causes high error.

// Forward declaration of error metric enum (defined in nvfp4_recipe.h)
namespace recipes { enum class FourOverSixErrorMetric; }

/// @brief Four Over Six NVFP4 quantization with CUTLASS-compatible scale layout.
/// @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
/// @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
/// @param[out] global_amax Output per-tensor absolute maximum.
/// @param[in] in Input BF16 data (M, K).
/// @param M Number of rows.
/// @param K Number of columns.
/// @param error_metric Error metric for 4/6 block selection (MSE, L1, AbsMax).
void quantize_nvfp4_4o6_cutlass_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                            float* global_amax, const nv_bfloat16* in,
                                            int M, int K,
                                            recipes::FourOverSixErrorMetric error_metric,
                                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Four Over Six NVFP4 stochastic quantization with CUTLASS layout (for gradients).
void quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(uint8_t* out_fp4, uint8_t* block_scales,
                                                       float* global_amax, const nv_bfloat16* in,
                                                       int M, int K,
                                                       recipes::FourOverSixErrorMetric error_metric,
                                                       unsigned int seed,
                                                       const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based Four Over Six NVFP4 quantization with CUTLASS layout.
void quantize_nvfp4_4o6_cutlass(Tensor& out_fp4, Tensor& block_scales, Tensor& global_amax,
                                 const Tensor& in,
                                 recipes::FourOverSixErrorMetric error_metric,
                                 const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based Four Over Six NVFP4 stochastic quantization with CUTLASS layout.
void quantize_nvfp4_4o6_stochastic_cutlass(Tensor& out_fp4, Tensor& block_scales, Tensor& global_amax,
                                            const Tensor& in,
                                            recipes::FourOverSixErrorMetric error_metric,
                                            unsigned int seed,
                                            const cudaDeviceProp& dp, cudaStream_t stream);

// ============================================================================
// CUTLASS FP4 GEMM Operations (Blackwell-only: SM100+)
// ============================================================================

/// @brief Check if CUTLASS FP4 is supported on the current GPU.
/// FP4 tensor core operations require Blackwell architecture (SM100+).
/// @return true if FP4 CUTLASS is available (SM100+), false otherwise
bool cutlass_supports_fp4();

/// @brief FP4 E2M1 block-scaled matrix multiplication using CUTLASS with BF16 output.
/// Computes: C = A @ B where A and B are FP4 E2M1 with UE4M3 block scales
/// @param c Output matrix (BF16), size M x N
/// @param a Input matrix A (packed FP4 E2M1, 2 values per byte), size M x K/2 bytes
/// @param b Input matrix B (packed FP4 E2M1, 2 values per byte), size K x N/2 bytes
/// @param scale_a Block scale factors for A (UE4M3), interleaved layout
/// @param scale_b Block scale factors for B (UE4M3), interleaved layout
/// @param workspace Temporary workspace memory
/// @param workspace_size Size of workspace in bytes
/// @param M Number of rows in C
/// @param N Number of columns in C
/// @param K Inner dimension (number of elements, not bytes)
/// @param stream CUDA stream for execution
void matmul_cutlass_fp4(
    nv_bfloat16* c,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace,
    std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream
);

/// @brief FP4 E2M1 block-scaled matrix multiplication using CUTLASS with FP32 output.
/// Same as matmul_cutlass_fp4 but outputs FP32 for alpha scaling before BF16 conversion.
void matmul_cutlass_fp4_f32(
    float* c,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace,
    std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream
);

/// @brief FP4 E2M1 block-scaled matrix multiplication using CUTLASS with alpha in epilogue.
///
/// Fuses alpha scaling into GEMM epilogue: D = alpha * (A @ B)
/// This eliminates the need for FP32 intermediate tensor and separate alpha kernel.
/// More efficient than matmul_cutlass_fp4_f32 + fp4_alpha_scale + convert_dtype.
///
/// @param c Output BF16 tensor (M, N)
/// @param a FP4 packed input A (M, K/2)
/// @param b FP4 packed input B (N, K/2)
/// @param scale_a UE4M3 block scales for A in CUTLASS layout
/// @param scale_b UE4M3 block scales for B in CUTLASS layout
/// @param alpha_ptr Device pointer to alpha = (amax_a * amax_b) / (FP4_MAX^2 * FP8_MAX^2)
/// @param workspace CUTLASS workspace buffer
/// @param workspace_size Size of workspace in bytes
/// @param M Number of rows in A/C
/// @param N Number of columns in B/C
/// @param K Inner dimension
/// @param stream CUDA stream
void matmul_cutlass_fp4_alpha(
    nv_bfloat16* c,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace,
    std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream
);

/// @brief Get workspace size required for FP4 CUTLASS GEMM
/// @param M Number of rows in output
/// @param N Number of columns in output
/// @param K Inner dimension
/// @return Required workspace size in bytes
std::size_t cutlass_fp4_workspace_size(int M, int N, int K);

// ============================================================================
// BitsAndBytes NF4 Quantization (QLoRA-style blockwise quantization)
// ============================================================================
// 4-bit Normal Float (NF4) quantization with per-block absmax scaling.
// Compatible with any CUDA GPU (no SM89+ or SM100+ requirement).
//
// NF4 uses 16 asymmetric bins derived from a standard normal distribution,
// which better represents neural network weight distributions compared to
// uniform quantization.
//
// Double quantization is supported to further reduce memory overhead by
// quantizing the absmax scaling factors themselves.

/// @brief BitsAndBytes-style NF4 blockwise quantization.
/// Quantizes BF16 weights to packed 4-bit NF4 with per-block absmax scaling.
/// @param[out] out Output packed 4-bit data (M*K/2 bytes).
/// @param[out] absmax Output per-block absmax scales (M*K/block_size floats).
/// @param[in] in Input BF16 data (M*K elements).
/// @param M Number of rows.
/// @param K Number of columns.
/// @param block_size Quantization block size (64, 128, 256, or 512).
/// @param dp CUDA device properties.
/// @param stream CUDA stream.
void quantize_bnb_nf4(unsigned char* out, float* absmax, const nv_bfloat16* in,
                      int M, int K, int block_size,
                      const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based BnB NF4 quantization.
void quantize_bnb_nf4(Tensor& out, Tensor& absmax, const Tensor& in,
                      int block_size,
                      const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief BitsAndBytes-style NF4 blockwise dequantization.
/// Dequantizes packed 4-bit NF4 data back to BF16 using per-block absmax scales.
/// @param[out] out Output BF16 data (M*K elements).
/// @param[in] in Input packed 4-bit data (M*K/2 bytes).
/// @param[in] absmax Per-block absmax scales.
/// @param M Number of rows.
/// @param K Number of columns.
/// @param block_size Quantization block size.
/// @param dp CUDA device properties.
/// @param stream CUDA stream.
void dequantize_bnb_nf4(nv_bfloat16* out, const unsigned char* in, const float* absmax,
                        int M, int K, int block_size,
                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Tensor-based BnB NF4 dequantization.
void dequantize_bnb_nf4(Tensor& out, const Tensor& in, const Tensor& absmax,
                        int block_size,
                        const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Double quantization: quantize absmax values to INT8.
/// Reduces memory overhead by quantizing the absmax scaling factors themselves.
/// @param[out] out_quant Output INT8 quantized absmax.
/// @param[out] out_scale Per-group dequantization scale (FP32).
/// @param[out] out_offset Per-group offset (FP32).
/// @param[in] absmax Input FP32 absmax values.
/// @param num_absmax Number of absmax values.
/// @param group_size Values per quantization group (default 256).
/// @param dp CUDA device properties.
/// @param stream CUDA stream.
void quantize_absmax_double(unsigned char* out_quant, float* out_scale, float* out_offset,
                            const float* absmax, int num_absmax, int group_size,
                            const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Dequantize INT8 absmax values back to FP32.
void dequantize_absmax_double(float* out_absmax,
                              const unsigned char* in_quant, const float* in_scale, const float* in_offset,
                              int num_absmax, int group_size,
                              const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief NF4 dequantization with inline absmax dequantization (double quant).
/// Handles both absmax dequantization and NF4 dequantization in one kernel.
/// @param[out] out Output BF16 data (M*K elements).
/// @param[in] in Input packed 4-bit NF4 data (M*K/2 bytes).
/// @param[in] absmax_quant Quantized INT8 absmax values.
/// @param[in] absmax_scale Per-group FP32 scale for absmax.
/// @param[in] absmax_offset Per-group FP32 offset for absmax.
/// @param M Number of rows.
/// @param K Number of columns.
/// @param block_size Quantization block size.
/// @param absmax_group_size Group size for double quantization (typically 256).
/// @param dp CUDA device properties.
/// @param stream CUDA stream.
void dequantize_bnb_nf4_double(nv_bfloat16* out, const unsigned char* in,
                               const unsigned char* absmax_quant,
                               const float* absmax_scale, const float* absmax_offset,
                               int M, int K, int block_size, int absmax_group_size,
                               const cudaDeviceProp& dp, cudaStream_t stream);

/// @brief Get the NF4 codebook values (host-side).
/// Returns pointer to the 16-element NF4 lookup table for debugging.
/// @return Pointer to static array of 16 floats.
const float* get_nf4_codebook();

// ============================================================================
// Mixture of Experts (MoE) Kernels
// ============================================================================
// Routing, expert selection, and token dispatch operations for MoE layers.

/// @brief Row-wise softmax for MoE routing logits.
/// @param out Output routing probabilities (num_tokens, num_experts).
/// @param inp Input routing logits (num_tokens, num_experts).
/// @param num_tokens Number of tokens (batch_size * seq_len).
/// @param num_experts Number of experts.
/// @param stream CUDA stream.
void moe_softmax_forward(float* out, const float* inp, int num_tokens, int num_experts, cudaStream_t stream);
void moe_softmax_forward(nv_bfloat16* out, const nv_bfloat16* inp, int num_tokens, int num_experts, cudaStream_t stream);

/// @brief Element-wise sigmoid for DeepSeek-style MoE routing.
/// @param out Output routing scores (num_elements).
/// @param inp Input routing logits (num_elements).
/// @param num_elements Total number of elements.
/// @param stream CUDA stream.
void moe_sigmoid_forward(float* out, const float* inp, int num_elements, cudaStream_t stream);
void moe_sigmoid_forward(nv_bfloat16* out, const nv_bfloat16* inp, int num_elements, cudaStream_t stream);

/// @brief Top-K expert selection per token.
/// @param expert_indices Output expert indices (num_tokens, top_k).
/// @param routing_weights Output routing weights (num_tokens, top_k).
/// @param scores Input routing scores (num_tokens, num_experts).
/// @param num_tokens Number of tokens.
/// @param num_experts Number of experts.
/// @param top_k Number of experts per token.
/// @param normalize_weights Whether to normalize weights to sum to 1.
/// @param stream CUDA stream.
void moe_topk_forward(int* expert_indices, float* routing_weights, const float* scores,
                      int num_tokens, int num_experts, int top_k, bool normalize_weights, cudaStream_t stream);
void moe_topk_forward(int* expert_indices, nv_bfloat16* routing_weights, const nv_bfloat16* scores,
                      int num_tokens, int num_experts, int top_k, bool normalize_weights, cudaStream_t stream);

/// @brief Backward through top-k selection (with optional post-selection normalization).
/// Given gradients wrt the selected weights (num_tokens, top_k), scatter them back to a dense
/// gradient wrt the full probability matrix (num_tokens, num_experts).
/// When @p normalize_weights is true, assumes forward did: w_k = p_k / sum_j p_j over selected experts.
/// Inputs/outputs are FP32 because routing is computed in FP32 in the modular model path.
void moe_topk_backward(float* d_probs, const float* d_routing_weights, const float* probs, const int* expert_indices,
                       int num_tokens, int num_experts, int top_k, bool normalize_weights, cudaStream_t stream);

/// @brief Compute histogram of tokens assigned to each expert.
/// @param expert_counts Output counts per expert (num_experts).
/// @param expert_indices Input expert indices (num_tokens, top_k).
/// @param num_tokens Number of tokens.
/// @param top_k Number of experts per token.
/// @param num_experts Number of experts.
/// @param stream CUDA stream.
void moe_compute_expert_counts(int* expert_counts, const int* expert_indices,
                               int num_tokens, int top_k, int num_experts, cudaStream_t stream);

/// @brief Compute expert offsets from counts (exclusive prefix sum).
/// expert_offsets[i] = sum(expert_counts[0:i])
/// expert_offsets[num_experts] = total_tokens (num_tokens * top_k)
/// @param expert_offsets Output offsets (num_experts + 1).
/// @param expert_counts Input counts per expert (num_experts).
/// @param num_experts Number of experts.
/// @param stream CUDA stream.
void moe_compute_expert_offsets(int* expert_offsets, const int* expert_counts,
                                int num_experts, cudaStream_t stream);

/// @brief Build gather/scatter indices for token permutation.
/// gather_indices[permuted_pos] = original token assignment index
/// scatter_indices[assignment_idx] = permuted position
/// @param gather_indices Output gather indices (total_tokens).
/// @param scatter_indices Output scatter indices (total_tokens).
/// @param expert_indices Expert assignments (num_tokens, top_k).
/// @param expert_offsets Expert offsets from compute_expert_offsets (num_experts + 1).
/// @param expert_positions Temporary storage for atomic counters (num_experts). Zeroed by caller.
/// @param num_tokens Number of original tokens.
/// @param top_k Number of experts per token.
/// @param num_experts Number of experts.
/// @param stream CUDA stream.
void moe_build_indices(int* gather_indices, int* scatter_indices,
                       const int* expert_indices, const int* expert_offsets,
                       int* expert_positions, int num_tokens, int top_k,
                       int num_experts, cudaStream_t stream);

/// @brief Remap expert indices from global to compact indices for selective dequantization.
/// Transforms expert_indices from global range [0, num_total_experts) to compact range
/// [0, num_active_experts) using the provided mapping table.
/// @param remapped_indices Output remapped indices (num_tokens, top_k).
/// @param expert_indices Input global expert indices (num_tokens, top_k).
/// @param expert_to_compact Mapping table: global_idx -> compact_idx, -1 if not active (num_total_experts).
/// @param num_tokens Number of tokens (B*T).
/// @param top_k Number of experts per token.
/// @param stream CUDA stream.
void moe_remap_expert_indices(int* remapped_indices, const int* expert_indices,
                              const int* expert_to_compact, int num_tokens, int top_k,
                              cudaStream_t stream);

/// @brief Permute tokens from natural order to expert-grouped order.
/// @param out Output permuted hidden states (total_tokens, hidden_size).
/// @param inp Input hidden states (num_tokens, hidden_size).
/// @param gather_indices Gather indices from compute_gather_indices (total_tokens).
/// @param total_tokens Total token-expert assignments (num_tokens * top_k).
/// @param num_tokens Number of original tokens.
/// @param hidden_size Hidden dimension.
/// @param top_k Number of experts per token.
/// @param stream CUDA stream.
void moe_permute_tokens(float* out, const float* inp, const int* gather_indices,
                        int total_tokens, int num_tokens, int hidden_size, int top_k, cudaStream_t stream);
void moe_permute_tokens(nv_bfloat16* out, const nv_bfloat16* inp, const int* gather_indices,
                        int total_tokens, int num_tokens, int hidden_size, int top_k, cudaStream_t stream);

/// @brief Unpermute and weight-combine expert outputs back to token order.
/// @param out Output combined hidden states (num_tokens, hidden_size).
/// @param expert_out Expert outputs in permuted order (total_tokens, hidden_size).
/// @param routing_weights Routing weights (num_tokens, top_k).
/// @param scatter_indices Scatter indices (inverse of gather_indices).
/// @param num_tokens Number of original tokens.
/// @param total_tokens Total token-expert assignments.
/// @param hidden_size Hidden dimension.
/// @param top_k Number of experts per token.
/// @param stream CUDA stream.
void moe_unpermute_and_combine(float* out, const float* expert_out, const float* routing_weights,
                               const int* scatter_indices, int num_tokens, int total_tokens,
                               int hidden_size, int top_k, cudaStream_t stream);
void moe_unpermute_and_combine(nv_bfloat16* out, const nv_bfloat16* expert_out, const nv_bfloat16* routing_weights,
                               const int* scatter_indices, int num_tokens, int total_tokens,
                               int hidden_size, int top_k, cudaStream_t stream);

/// @brief Compute auxiliary load-balancing loss for MoE training.
/// aux_loss = coef * num_experts * sum_e(fraction_e * prob_e)
/// @param aux_loss Output scalar loss value (device pointer).
/// @param routing_probs Routing probabilities post-softmax (num_tokens, num_experts).
/// @param expert_indices Expert indices (num_tokens, top_k).
/// @param num_tokens Number of tokens.
/// @param num_experts Number of experts.
/// @param top_k Number of experts per token.
/// @param aux_loss_coef Loss coefficient (typically 0.01).
/// @param stream CUDA stream.
void moe_compute_aux_loss(float* aux_loss, const float* routing_probs, const int* expert_indices,
                          int num_tokens, int num_experts, int top_k, float aux_loss_coef, cudaStream_t stream);
void moe_compute_aux_loss(float* aux_loss, const nv_bfloat16* routing_probs, const int* expert_indices,
                          int num_tokens, int num_experts, int top_k, float aux_loss_coef, cudaStream_t stream);

// MoE Backward Kernels

/// @brief Backward pass for softmax routing.
/// d_logits = softmax_probs * (d_probs - sum_j(d_probs_j * softmax_probs_j))
/// @param d_logits Output gradient w.r.t. logits (num_tokens, num_experts).
/// @param d_probs Input gradient w.r.t. softmax output (num_tokens, num_experts).
/// @param softmax_probs Saved softmax probabilities from forward (num_tokens, num_experts).
/// @param num_tokens Number of tokens.
/// @param num_experts Number of experts.
/// @param stream CUDA stream.
void moe_softmax_backward(float* d_logits, const float* d_probs, const float* softmax_probs,
                          int num_tokens, int num_experts, cudaStream_t stream);
void moe_softmax_backward(nv_bfloat16* d_logits, const nv_bfloat16* d_probs, const nv_bfloat16* softmax_probs,
                          int num_tokens, int num_experts, cudaStream_t stream);

/// @brief Backward pass for unpermute+combine operation.
/// Computes gradients for expert outputs and optionally routing weights.
/// @param d_expert_out Output gradient for expert outputs (total_tokens, hidden_size).
/// @param d_routing_weights Optional output gradient for routing weights (num_tokens, top_k). Can be NULL.
/// @param d_output Input gradient from downstream (num_tokens, hidden_size).
/// @param expert_out Expert outputs from forward pass (for weight gradient computation).
/// @param routing_weights Routing weights from forward pass.
/// @param scatter_indices Scatter indices from forward pass.
/// @param num_tokens Number of original tokens.
/// @param total_tokens Total token-expert assignments.
/// @param hidden_size Hidden dimension.
/// @param top_k Number of experts per token.
/// @param stream CUDA stream.
void moe_combine_backward(float* d_expert_out, float* d_routing_weights,
                          const float* d_output, const float* expert_out, const float* routing_weights,
                          const int* scatter_indices, int num_tokens, int total_tokens,
                          int hidden_size, int top_k, cudaStream_t stream);
void moe_combine_backward(nv_bfloat16* d_expert_out, nv_bfloat16* d_routing_weights,
                          const nv_bfloat16* d_output, const nv_bfloat16* expert_out, const nv_bfloat16* routing_weights,
                          const int* scatter_indices, int num_tokens, int total_tokens,
                          int hidden_size, int top_k, cudaStream_t stream);

/// @brief Backward pass for token permutation.
/// Gathers gradients from permuted order back to token order.
/// @param d_input Output gradient in token order (num_tokens, hidden_size).
/// @param d_permuted Input gradient in permuted order (total_tokens, hidden_size).
/// @param gather_indices Gather indices from forward pass.
/// @param total_tokens Total token-expert assignments.
/// @param num_tokens Number of original tokens.
/// @param hidden_size Hidden dimension.
/// @param top_k Number of experts per token.
/// @param stream CUDA stream.
void moe_permute_backward(float* d_input, const float* d_permuted, const int* gather_indices,
                          int total_tokens, int num_tokens, int hidden_size, int top_k, cudaStream_t stream);
void moe_permute_backward(nv_bfloat16* d_input, const nv_bfloat16* d_permuted, const int* gather_indices,
                          int total_tokens, int num_tokens, int hidden_size, int top_k, cudaStream_t stream);

/// @brief Compute router z-loss for MoE training regularization.
/// z_loss = coef * (1/num_tokens) * sum_t(logsumexp(logits_t))^2
/// Encourages smaller router logits to prevent instability.
/// @param z_loss Output scalar loss value (device pointer, accumulated via atomicAdd).
/// @param router_logits Pre-softmax router logits (num_tokens, num_experts).
/// @param num_tokens Number of tokens.
/// @param num_experts Number of experts.
/// @param z_loss_coef Loss coefficient (typically 0.001-0.01).
/// @param stream CUDA stream.
void moe_router_z_loss_forward(float* z_loss, const float* router_logits,
                               int num_tokens, int num_experts, float z_loss_coef, cudaStream_t stream);
void moe_router_z_loss_forward(float* z_loss, const nv_bfloat16* router_logits,
                               int num_tokens, int num_experts, float z_loss_coef, cudaStream_t stream);

/// @brief Backward pass for router z-loss.
/// d_logits = coef * (2 * logsumexp / num_tokens) * softmax(logits)
/// @param d_logits Output gradient w.r.t. router logits (num_tokens, num_experts).
/// @param router_logits Pre-softmax router logits from forward pass.
/// @param num_tokens Number of tokens.
/// @param num_experts Number of experts.
/// @param z_loss_coef Loss coefficient (same as forward).
/// @param stream CUDA stream.
void moe_router_z_loss_backward(float* d_logits, const float* router_logits,
                                int num_tokens, int num_experts, float z_loss_coef, cudaStream_t stream);
void moe_router_z_loss_backward(nv_bfloat16* d_logits, const nv_bfloat16* router_logits,
                                int num_tokens, int num_experts, float z_loss_coef, cudaStream_t stream);

/// @brief Generic Grouped GEMM for MoE across all experts.
/// @param output Output tensor (total_tokens, M).
/// @param input Input tensor in permuted order (total_tokens, K).
/// @param weights Batched expert weights (num_experts, M, K).
/// @param expert_offsets Token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param M Output dimension.
/// @param K Input dimension.
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
/// @param host_offsets Optional: pre-cached host offsets.
/// @param active_expert_indices Optional host pointer (CPU) to a list of active global expert indices.
/// @param weight_is_compact If true, `weights` is indexed by compact active-expert order; otherwise by global expert id.
/// @param num_active_experts Number of entries in @p active_expert_indices (<= num_experts). If <= 0, uses num_experts.
void moe_grouped_gemm(float* output, const float* input, const float* weights,
                      const int* expert_offsets, int num_experts,
                      int M, int K,
                      cublasHandle_t cublas_handle, cudaStream_t stream,
                      const int* host_offsets = nullptr,
                      float alpha = 1.0f, float beta = 0.0f,
                      EMMTranspose mode = EMMTranspose::TN,
                      const int* active_expert_indices = nullptr,
                      bool weight_is_compact = true,
                      int num_active_experts = -1);
void moe_grouped_gemm(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* weights,
                      const int* expert_offsets, int num_experts,
                      int M, int K,
                      cublasHandle_t cublas_handle, cudaStream_t stream,
                      const int* host_offsets = nullptr,
                      float alpha = 1.0f, float beta = 0.0f,
                      EMMTranspose mode = EMMTranspose::TN,
                      const int* active_expert_indices = nullptr,
                      bool weight_is_compact = true,
                      int num_active_experts = -1);

/// @brief Computes weight gradients across all experts: dW = grad_output^T @ input
/// @param d_weight Output gradient tensor (num_experts, M, N).
/// @param grad_output Input gradient from downstream (total_tokens, M).
/// @param input Input activations (total_tokens, N).
/// @param expert_offsets Token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param M Output rows of dW.
/// @param N Output cols of dW.
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
/// @param host_offsets Optional: pre-cached host offsets.
/// @param alpha Scaling factor.
/// @param beta Accumulation factor.
/// @param active_expert_indices Optional host pointer (CPU) to a list of active global expert indices.
/// @param weight_is_compact If true, `d_weight` is indexed by compact active-expert order; otherwise by global expert id.
/// @param num_active_experts Number of entries in @p active_expert_indices (<= num_experts). If <= 0, uses num_experts.
void moe_grouped_gemm_weight_grad(float* d_weight, const float* grad_output, const float* input,
                                  const int* expert_offsets, int num_experts,
                                  int M, int N,
                                  cublasHandle_t cublas_handle, cudaStream_t stream,
                                  const int* host_offsets = nullptr,
                                  float alpha = 1.0f, float beta = 0.0f,
                                  const int* active_expert_indices = nullptr,
                                  bool weight_is_compact = true,
                                  int num_active_experts = -1);
void moe_grouped_gemm_weight_grad(nv_bfloat16* d_weight, const nv_bfloat16* grad_output, const nv_bfloat16* input,
                                  const int* expert_offsets, int num_experts,
                                  int M, int N,
                                  cublasHandle_t cublas_handle, cudaStream_t stream,
                                  const int* host_offsets = nullptr,
                                  float alpha = 1.0f, float beta = 0.0f,
                                  const int* active_expert_indices = nullptr,
                                  bool weight_is_compact = true,
                                  int num_active_experts = -1);

/// @brief Grouped GEMM for MoE gate+up projection across all experts.
/// Runs all expert GEMMs in parallel instead of sequentially.
/// @param output Output tensor (total_tokens, 2*intermediate_size).
/// @param input Input tensor in permuted order (total_tokens, hidden_size).
/// @param weights Batched expert weights (num_experts, 2*intermediate_size, hidden_size).
/// @param expert_offsets Token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param hidden_size Input hidden dimension (C).
/// @param intermediate_size Expert intermediate dimension (D, output is 2*D).
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
/// @param active_expert_indices Optional host pointer (CPU) to a list of active global expert indices.
/// @param weight_is_compact If true, `weights` is indexed by compact active-expert order; otherwise by global expert id.
/// @param num_active_experts Number of entries in @p active_expert_indices (<= num_experts). If <= 0, uses num_experts.
void moe_grouped_gemm_gate_up(float* output, const float* input, const float* weights,
                              const int* expert_offsets, int num_experts,
                              int hidden_size, int intermediate_size,
                              cublasHandle_t cublas_handle, cudaStream_t stream,
                              const int* host_offsets = nullptr,
                              const int* active_expert_indices = nullptr,
                              bool weight_is_compact = true,
                              int num_active_experts = -1);
void moe_grouped_gemm_gate_up(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* weights,
                              const int* expert_offsets, int num_experts,
                              int hidden_size, int intermediate_size,
                              cublasHandle_t cublas_handle, cudaStream_t stream,
                              const int* host_offsets = nullptr,
                              const int* active_expert_indices = nullptr,
                              bool weight_is_compact = true,
                              int num_active_experts = -1);

/// @brief Grouped GEMM for MoE down projection across all experts.
/// Runs all expert GEMMs in parallel instead of sequentially.
/// @param output Output tensor (total_tokens, hidden_size).
/// @param input Input tensor (SwiGLU output) in permuted order (total_tokens, intermediate_size).
/// @param weights Batched expert weights (num_experts, hidden_size, intermediate_size).
/// @param expert_offsets Token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param hidden_size Output hidden dimension (C).
/// @param intermediate_size Expert intermediate dimension (D).
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
/// @param active_expert_indices Optional host pointer (CPU) to a list of active global expert indices.
/// @param weight_is_compact If true, `weights` is indexed by compact active-expert order; otherwise by global expert id.
/// @param num_active_experts Number of entries in @p active_expert_indices (<= num_experts). If <= 0, uses num_experts.
void moe_grouped_gemm_down(float* output, const float* input, const float* weights,
                           const int* expert_offsets, int num_experts,
                           int hidden_size, int intermediate_size,
                           cublasHandle_t cublas_handle, cudaStream_t stream,
                           const int* host_offsets = nullptr,
                           const int* active_expert_indices = nullptr,
                           bool weight_is_compact = true,
                           int num_active_experts = -1);
void moe_grouped_gemm_down(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* weights,
                           const int* expert_offsets, int num_experts,
                           int hidden_size, int intermediate_size,
                           cublasHandle_t cublas_handle, cudaStream_t stream,
                           const int* host_offsets = nullptr,
                           const int* active_expert_indices = nullptr,
                           bool weight_is_compact = true,
                           int num_active_experts = -1);

/// @brief Grouped GEMM backward through down projection for all MoE experts.
/// Computes d_swiglu = d_output @ down_proj (no transpose on weight).
/// @param d_input Output gradient w.r.t. SwiGLU output (total_tokens, D).
/// @param d_output Input gradient from downstream (total_tokens, C).
/// @param weights Down projection weights in batched layout (num_experts, C, D).
/// @param expert_offsets Cumulative token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param hidden_size Hidden dimension (C).
/// @param intermediate_size Expert intermediate dimension (D).
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
void moe_grouped_gemm_down_backward(float* d_input, const float* d_output, const float* weights,
                                     const int* expert_offsets, int num_experts,
                                     int hidden_size, int intermediate_size,
                                     cublasHandle_t cublas_handle, cudaStream_t stream,
                                     const int* host_offsets = nullptr,
                                     const int* active_expert_indices = nullptr,
                                     bool weight_is_compact = true,
                                     int num_active_experts = -1);
void moe_grouped_gemm_down_backward(nv_bfloat16* d_input, const nv_bfloat16* d_output, const nv_bfloat16* weights,
                                     const int* expert_offsets, int num_experts,
                                     int hidden_size, int intermediate_size,
                                     cublasHandle_t cublas_handle, cudaStream_t stream,
                                     const int* host_offsets = nullptr,
                                     const int* active_expert_indices = nullptr,
                                     bool weight_is_compact = true,
                                     int num_active_experts = -1);

/// @brief Grouped GEMM backward through gate+up projection for all MoE experts.
/// Computes d_input = d_gate_up @ gate_up_proj (no transpose on weight).
/// @param d_input Output gradient w.r.t. original input (total_tokens, C).
/// @param d_gate_up Input gradient from SwiGLU backward (total_tokens, 2*D).
/// @param weights Gate+up projection weights in batched layout (num_experts, 2*D, C).
/// @param expert_offsets Cumulative token offsets per expert (num_experts + 1).
/// @param num_experts Number of experts.
/// @param hidden_size Hidden dimension (C).
/// @param intermediate_size Expert intermediate dimension (D, d_gate_up is 2*D).
/// @param cublas_handle cuBLAS handle.
/// @param stream CUDA stream.
void moe_grouped_gemm_gate_up_backward(float* d_input, const float* d_gate_up, const float* weights,
                                        const int* expert_offsets, int num_experts,
                                        int hidden_size, int intermediate_size,
                                        cublasHandle_t cublas_handle, cudaStream_t stream,
                                        const int* host_offsets = nullptr,
                                        const int* active_expert_indices = nullptr,
                                        bool weight_is_compact = true,
                                        int num_active_experts = -1);
void moe_grouped_gemm_gate_up_backward(nv_bfloat16* d_input, const nv_bfloat16* d_gate_up, const nv_bfloat16* weights,
                                        const int* expert_offsets, int num_experts,
                                        int hidden_size, int intermediate_size,
                                        cublasHandle_t cublas_handle, cudaStream_t stream,
                                        const int* host_offsets = nullptr,
                                        const int* active_expert_indices = nullptr,
                                        bool weight_is_compact = true,
                                        int num_active_experts = -1);

#endif //SUROGATE_SRC_KERNELS_KERNELS_H
