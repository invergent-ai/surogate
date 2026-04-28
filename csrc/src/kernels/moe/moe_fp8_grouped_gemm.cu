// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_fp8_grouped_gemm.cu.

// ============================================================================
// FP8 MoE Grouped GEMM Implementations
// ============================================================================
//
// These implementations dispatch FP8 E4M3 × E4M3 (forward) or E4M3 × E5M2
// (backward) GEMMs for MoE layers. Forward/dgrad still use cuBLASLt per expert
// for scale-pointer support; wgrad uses native cuBLAS grouped GEMM when the
// installed CUDA exposes no cuBLASLt grouped matmul entry point.

/// @brief FP8 MoE grouped GEMM: E4M3 input × E4M3 weights → BF16 output
void moe_grouped_gemm(nv_bfloat16* output,
                      const __nv_fp8_e4m3* input,
                      const __nv_fp8_e4m3* weights,
                      const float* scale_input,
                      const float* scale_weights,
                      const int* expert_offsets,
                      int num_experts,
                      int M,
                      int K,
                      cublasLtHandle_t cublas_handle,
                      cudaStream_t stream,
                      const int* host_offsets,
                      float alpha,
                      float beta,
                      EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts) {
    // Loop over each expert and dispatch FP8 matmul
    const int num_active = (num_active_experts > 0) ? num_active_experts : num_experts;

    for (int idx = 0; idx < num_active; ++idx) {
        const int expert_id = active_expert_indices ? active_expert_indices[idx] : idx;
        const int weight_idx = weight_is_compact ? idx : expert_id;

        // Get token range for this expert
        const int start = host_offsets[expert_id];
        const int end = host_offsets[expert_id + 1];
        const int num_tokens = end - start;

        if (num_tokens <= 0) continue;

        // Compute pointers for this expert's slice
        const __nv_fp8_e4m3* input_slice = input + static_cast<long>(start) * K;
        const __nv_fp8_e4m3* weight_slice = weights + static_cast<long>(weight_idx) * M * K;
        nv_bfloat16* output_slice = output + static_cast<long>(start) * M;
        const float* weight_scale_slice = scale_weights ? (scale_weights + weight_idx) : nullptr;

        // Dispatch FP8 matmul: output = input @ weight.T
        // input: (num_tokens, K) E4M3
        // weight: (M, K) E4M3
        // output: (num_tokens, M) BF16
        matmul(output_slice,
               weight_slice,
               input_slice,
               static_cast<nv_bfloat16*>(nullptr),  // no bias
               weight_scale_slice,
               scale_input,
               cublas_handle,
               nullptr,
               0,  // no workspace needed
               M,
               num_tokens,
               K,
               mode,  // typically TN
               /*accumulate=*/false,
               stream);
    }
}

/// @brief FP8 MoE grouped GEMM backward: E4M3 weights × E5M2 gradients → BF16 dinp
void moe_grouped_gemm_up_backward(nv_bfloat16* d_input,
                                  const __nv_fp8_e5m2* d_output,
                                  const __nv_fp8_e4m3* weights,
                                  const float* scale_dout,
                                  const float* scale_weights,
                                  const int* expert_offsets,
                                  int num_experts,
                                  int hidden_size,
                                  int intermediate_size,
                                  cublasLtHandle_t cublas_handle,
                                  cudaStream_t stream,
                                  const int* host_offsets,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    // Compute dinp = weights^T @ dout
    // weights: (num_experts, M, K) E4M3
    // dout: (total_tokens, M) E5M2
    // dinp: (total_tokens, K) BF16

    const int M = intermediate_size;
    const int K = hidden_size;
    const int num_active = (num_active_experts > 0) ? num_active_experts : num_experts;

    for (int idx = 0; idx < num_active; ++idx) {
        const int expert_id = active_expert_indices ? active_expert_indices[idx] : idx;
        const int weight_idx = weight_is_compact ? idx : expert_id;

        const int start = host_offsets[expert_id];
        const int end = host_offsets[expert_id + 1];
        const int num_tokens = end - start;

        if (num_tokens <= 0) continue;

        const __nv_fp8_e5m2* dout_slice = d_output + static_cast<long>(start) * M;
        const __nv_fp8_e4m3* weight_slice = weights + static_cast<long>(weight_idx) * M * K;
        nv_bfloat16* dinp_slice = d_input + static_cast<long>(start) * K;
        const float* weight_scale_slice = scale_weights ? (scale_weights + weight_idx) : nullptr;

        // dinp = W^T @ dout => (K, M) @ (num_tokens, M)^T = (K, num_tokens)^T = (num_tokens, K)
        // Using NN mode: dinp = weight @ dout where weight is (M, K) -> need (K, M)
        matmul(dinp_slice,
               weight_slice,
               dout_slice,
               static_cast<nv_bfloat16*>(nullptr),  // no bias
               weight_scale_slice,
               scale_dout,
               cublas_handle,
               nullptr,
               0,
               K,
               num_tokens,
               M,
               EMMTranspose::NN,  // weight needs to be transposed
               /*accumulate=*/false,
               stream);
    }
}
