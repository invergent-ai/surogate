// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_grouped_gemm_expert.cu.

template <typename T>
void moe_grouped_gemm_gate_up_impl(
    T* output,                  // (total_tokens, 2*D) - gate+up output
    const T* input,             // (total_tokens, C) - permuted tokens
    const T* weights,           // (num_experts, 2*D, C) - batched weights (ignored when weight_ptrs != null)
    const int* expert_offsets,  // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,        // C
    int intermediate_size,  // D (output is 2*D for gate+up)
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,  // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts,
    const void* const* weight_ptrs  // Optional: per-expert weight pointers (LLEP)
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    // Get host-side offsets - either use cached or copy from device
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        // Use pre-cached host offsets (no sync needed)
        h_offsets = host_offsets;
    } else {
        // Copy from device (requires sync - slower path)
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(),
                                   expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int out_dim = 2 * intermediate_size;

    // One GEMM per expert with tokens, through moe_expert_gemms (moe_common.cuh).
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        const int weight_idx = weight_is_compact ? e : global_idx;
        const T* A_ptr = weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                     : weights + static_cast<std::size_t>(weight_idx) * out_dim * hidden_size;
        const T* B_ptr = input + static_cast<std::size_t>(h_offsets[global_idx]) * hidden_size;
        T* C_ptr = output + static_cast<std::size_t>(h_offsets[global_idx]) * out_dim;

        m_vec.push_back(out_dim);
        n_vec.push_back(tokens_e);
        k_vec.push_back(hidden_size);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(hidden_size);
        ldc_vec.push_back(out_dim);

        A_vec.push_back(A_ptr);
        B_vec.push_back(B_ptr);
        C_vec.push_back(C_ptr);
    }

    if (m_vec.empty()) return;

    moe_expert_gemms<T>(cublas_handle,
                        stream,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        m_vec,
                        n_vec,
                        k_vec,
                        alpha,
                        A_vec,
                        lda_vec,
                        B_vec,
                        ldb_vec,
                        beta,
                        C_vec,
                        ldc_vec);
}

template <typename T>
void moe_grouped_gemm_down_impl(
    T* output,                  // (total_tokens, C) - down proj output
    const T* input,             // (total_tokens, D) - SwiGLU output
    const T* weights,           // (num_experts, C, D) - batched weights (ignored when weight_ptrs != null)
    const int* expert_offsets,  // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,        // C
    int intermediate_size,  // D
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,  // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts,
    const void* const* weight_ptrs  // Optional: per-expert weight pointers (LLEP)
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
    // Get host-side offsets - either use cached or copy from device
    std::vector<int> local_offsets;
    const int* h_offsets;

    if (host_offsets) {
        h_offsets = host_offsets;
    } else {
        local_offsets.resize(num_experts + 1);
        CUDA_CHECK(cudaMemcpyAsync(local_offsets.data(),
                                   expert_offsets,
                                   (num_experts + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        h_offsets = local_offsets.data();
    }

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // One GEMM per expert with tokens, through moe_expert_gemms (moe_common.cuh).
    std::vector<int> m_vec, n_vec, k_vec;
    std::vector<int> lda_vec, ldb_vec, ldc_vec;
    std::vector<const T*> A_vec, B_vec;
    std::vector<T*> C_vec;

    m_vec.reserve(n_active);
    n_vec.reserve(n_active);
    k_vec.reserve(n_active);
    lda_vec.reserve(n_active);
    ldb_vec.reserve(n_active);
    ldc_vec.reserve(n_active);
    A_vec.reserve(n_active);
    B_vec.reserve(n_active);
    C_vec.reserve(n_active);

    for (int e = 0; e < n_active; ++e) {
        int global_idx = active_expert_indices ? active_expert_indices[e] : e;
        int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
        if (tokens_e == 0) continue;

        const int weight_idx = weight_is_compact ? e : global_idx;

        m_vec.push_back(hidden_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(intermediate_size);

        lda_vec.push_back(intermediate_size);
        ldb_vec.push_back(intermediate_size);
        ldc_vec.push_back(hidden_size);

        A_vec.push_back(weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                    : weights + static_cast<std::size_t>(weight_idx) * hidden_size * intermediate_size);
        B_vec.push_back(input + static_cast<std::size_t>(h_offsets[global_idx]) * intermediate_size);
        C_vec.push_back(output + static_cast<std::size_t>(h_offsets[global_idx]) * hidden_size);
    }

    if (m_vec.empty()) return;

    moe_expert_gemms<T>(cublas_handle,
                        stream,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        m_vec,
                        n_vec,
                        k_vec,
                        alpha,
                        A_vec,
                        lda_vec,
                        B_vec,
                        ldb_vec,
                        beta,
                        C_vec,
                        ldc_vec);
}

void moe_grouped_gemm_gate_up(nv_bfloat16* output,
                              const nv_bfloat16* input,
                              const nv_bfloat16* weights,
                              const int* expert_offsets,
                              int num_experts,
                              int hidden_size,
                              int intermediate_size,
                              cublasHandle_t cublas_handle,
                              cudaStream_t stream,
                              const int* host_offsets,
                              const int* active_expert_indices,
                              bool weight_is_compact,
                              int num_active_experts,
                              const void* const* weight_ptrs) {
    moe_grouped_gemm_gate_up_impl(output,
                                  input,
                                  weights,
                                  expert_offsets,
                                  num_experts,
                                  hidden_size,
                                  intermediate_size,
                                  cublas_handle,
                                  stream,
                                  host_offsets,
                                  active_expert_indices,
                                  weight_is_compact,
                                  num_active_experts,
                                  weight_ptrs);
}

void moe_grouped_gemm_gate_up(float* output,
                              const float* input,
                              const float* weights,
                              const int* expert_offsets,
                              int num_experts,
                              int hidden_size,
                              int intermediate_size,
                              cublasHandle_t cublas_handle,
                              cudaStream_t stream,
                              const int* host_offsets,
                              const int* active_expert_indices,
                              bool weight_is_compact,
                              int num_active_experts,
                              const void* const* weight_ptrs) {
    moe_grouped_gemm_gate_up_impl(output,
                                  input,
                                  weights,
                                  expert_offsets,
                                  num_experts,
                                  hidden_size,
                                  intermediate_size,
                                  cublas_handle,
                                  stream,
                                  host_offsets,
                                  active_expert_indices,
                                  weight_is_compact,
                                  num_active_experts,
                                  weight_ptrs);
}

void moe_grouped_gemm_down(nv_bfloat16* output,
                           const nv_bfloat16* input,
                           const nv_bfloat16* weights,
                           const int* expert_offsets,
                           int num_experts,
                           int hidden_size,
                           int intermediate_size,
                           cublasHandle_t cublas_handle,
                           cudaStream_t stream,
                           const int* host_offsets,
                           const int* active_expert_indices,
                           bool weight_is_compact,
                           int num_active_experts,
                           const void* const* weight_ptrs) {
    moe_grouped_gemm_down_impl(output,
                               input,
                               weights,
                               expert_offsets,
                               num_experts,
                               hidden_size,
                               intermediate_size,
                               cublas_handle,
                               stream,
                               host_offsets,
                               active_expert_indices,
                               weight_is_compact,
                               num_active_experts,
                               weight_ptrs);
}

void moe_grouped_gemm_down(float* output,
                           const float* input,
                           const float* weights,
                           const int* expert_offsets,
                           int num_experts,
                           int hidden_size,
                           int intermediate_size,
                           cublasHandle_t cublas_handle,
                           cudaStream_t stream,
                           const int* host_offsets,
                           const int* active_expert_indices,
                           bool weight_is_compact,
                           int num_active_experts,
                           const void* const* weight_ptrs) {
    moe_grouped_gemm_down_impl(output,
                               input,
                               weights,
                               expert_offsets,
                               num_experts,
                               hidden_size,
                               intermediate_size,
                               cublas_handle,
                               stream,
                               host_offsets,
                               active_expert_indices,
                               weight_is_compact,
                               num_active_experts,
                               weight_ptrs);
}
