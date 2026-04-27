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

    // Optional debug: force per-expert GEMM loop to bypass grouped GEMM.
    static int force_loop = -1;
    if (force_loop < 0) {
        force_loop = (std::getenv("SUROGATE_MOE_GEMM_LOOP") != nullptr) ? 1 : 0;
    }
    if (force_loop) {
        static int force_default_algo = -1;
        if (force_default_algo < 0) {
            force_default_algo = (std::getenv("SUROGATE_MOE_GEMM_DEFAULT") != nullptr) ? 1 : 0;
        }
        const cublasGemmAlgo_t algo = force_default_algo ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for (int e = 0; e < n_active; ++e) {
            int global_idx = active_expert_indices ? active_expert_indices[e] : e;
            int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
            if (tokens_e == 0) continue;
            const int weight_idx = weight_is_compact ? e : global_idx;
            const T* A_ptr = weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                         : weights + weight_idx * out_dim * hidden_size;
            const T* B_ptr = input + h_offsets[global_idx] * hidden_size;
            T* C_ptr = output + h_offsets[global_idx] * out_dim;

            CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      out_dim,
                                      tokens_e,
                                      hidden_size,
                                      &alpha,
                                      A_ptr,
                                      cublas_dtype<T>(),
                                      hidden_size,
                                      B_ptr,
                                      cublas_dtype<T>(),
                                      hidden_size,
                                      &beta,
                                      C_ptr,
                                      cublas_dtype<T>(),
                                      out_dim,
                                      CUBLAS_COMPUTE_32F,
                                      algo));
        }
        return;
    }

    // Use Grouped GEMM to submit all expert computations in a single call.
    // This significantly reduces CPU overhead and kernel launch latency compared to a loop.
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
        const T* A_ptr =
            weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx]) : weights + weight_idx * out_dim * hidden_size;
        const T* B_ptr = input + h_offsets[global_idx] * hidden_size;
        T* C_ptr = output + h_offsets[global_idx] * out_dim;

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

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(cublas_handle,
                                            transa_vec.data(),
                                            transb_vec.data(),
                                            m_vec.data(),
                                            n_vec.data(),
                                            k_vec.data(),
                                            alpha_vec.data(),
                                            reinterpret_cast<const void**>(d_A_array),
                                            cublas_dtype<T>(),
                                            lda_vec.data(),
                                            reinterpret_cast<const void**>(d_B_array),
                                            cublas_dtype<T>(),
                                            ldb_vec.data(),
                                            beta_vec.data(),
                                            reinterpret_cast<void**>(d_C_array),
                                            cublas_dtype<T>(),
                                            ldc_vec.data(),
                                            gemm_count,
                                            group_size_vec.data(),
                                            CUBLAS_COMPUTE_32F));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
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

    // Optional debug: force per-expert GEMM loop to bypass grouped GEMM.
    static int force_loop = -1;
    if (force_loop < 0) {
        force_loop = (std::getenv("SUROGATE_MOE_GEMM_LOOP") != nullptr) ? 1 : 0;
    }
    if (force_loop) {
        static int force_default_algo = -1;
        if (force_default_algo < 0) {
            force_default_algo = (std::getenv("SUROGATE_MOE_GEMM_DEFAULT") != nullptr) ? 1 : 0;
        }
        const cublasGemmAlgo_t algo = force_default_algo ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        for (int e = 0; e < n_active; ++e) {
            int global_idx = active_expert_indices ? active_expert_indices[e] : e;
            int tokens_e = h_offsets[global_idx + 1] - h_offsets[global_idx];
            if (tokens_e == 0) continue;
            const int weight_idx = weight_is_compact ? e : global_idx;
            const T* A_ptr = weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                         : weights + weight_idx * hidden_size * intermediate_size;
            const T* B_ptr = input + h_offsets[global_idx] * intermediate_size;
            T* C_ptr = output + h_offsets[global_idx] * hidden_size;

            CUBLAS_CHECK(cublasGemmEx(cublas_handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      hidden_size,
                                      tokens_e,
                                      intermediate_size,
                                      &alpha,
                                      A_ptr,
                                      cublas_dtype<T>(),
                                      intermediate_size,
                                      B_ptr,
                                      cublas_dtype<T>(),
                                      intermediate_size,
                                      &beta,
                                      C_ptr,
                                      cublas_dtype<T>(),
                                      hidden_size,
                                      CUBLAS_COMPUTE_32F,
                                      algo));
        }
        return;
    }

    // Use Grouped GEMM to submit all expert computations in a single call.
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
                                    : weights + weight_idx * hidden_size * intermediate_size);
        B_vec.push_back(input + h_offsets[global_idx] * intermediate_size);
        C_vec.push_back(output + h_offsets[global_idx] * hidden_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    // cublasGemmGroupedBatchedEx requires pointer arrays to be in device memory
    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_T);
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_N);
    std::vector<int> group_size_vec(gemm_count, 1);
    std::vector<float> alpha_vec(gemm_count, alpha);
    std::vector<float> beta_vec(gemm_count, beta);

    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(cublas_handle,
                                            transa_vec.data(),
                                            transb_vec.data(),
                                            m_vec.data(),
                                            n_vec.data(),
                                            k_vec.data(),
                                            alpha_vec.data(),
                                            reinterpret_cast<const void**>(d_A_array),
                                            cublas_dtype<T>(),
                                            lda_vec.data(),
                                            reinterpret_cast<const void**>(d_B_array),
                                            cublas_dtype<T>(),
                                            ldb_vec.data(),
                                            beta_vec.data(),
                                            reinterpret_cast<void**>(d_C_array),
                                            cublas_dtype<T>(),
                                            ldc_vec.data(),
                                            gemm_count,
                                            group_size_vec.data(),
                                            CUBLAS_COMPUTE_32F));

    // Free device pointer arrays
    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
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
