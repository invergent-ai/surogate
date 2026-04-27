// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_grouped_gemm_backward.cu.

// ============================================================================
// Grouped GEMM Backward for MoE Expert Computation
// ============================================================================
// These compute the backward pass through expert projections:
// - down_backward: d_swiglu = d_output @ down_proj (no transpose on weight)
// - gate_up_backward: d_input = d_gate_up @ gate_up_proj (no transpose on weight)

// Kernel to build pointer arrays for down backward on device
template <typename T>
__global__ void build_gemm_pointers_down_backward_kernel(const T** A_ptrs,  // output: d_output pointers
                                                         const T** B_ptrs,  // output: weight pointers
                                                         T** C_ptrs,        // output: d_input pointers
                                                         int* lda_arr,
                                                         int* ldb_arr,
                                                         int* ldc_arr,
                                                         int* m_arr,
                                                         int* n_arr,
                                                         int* k_arr,
                                                         const T* d_output,
                                                         const T* weights,
                                                         T* d_input,
                                                         const int* expert_offsets,
                                                         int num_experts,
                                                         int hidden_size,       // C
                                                         int intermediate_size  // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // d_output: (tokens_e, C) at offset start * C
    A_ptrs[e] = d_output + start * hidden_size;
    // Weight: (C, D) for expert e
    B_ptrs[e] = weights + e * hidden_size * intermediate_size;
    // d_input: (tokens_e, D) at offset start * D
    C_ptrs[e] = d_input + start * intermediate_size;

    // For backward: d_input = d_output @ W (no transpose on W)
    // Row-major: d_input[t][d] = sum_c d_output[t][c] * W[c][d]
    //
    // In column-major:
    // - d_output is (C, tokens) col-major
    // - W is (D, C) col-major (because row-major (C, D))
    // - d_input is (D, tokens) col-major
    //
    // So: d_input(D, tokens) = W(D, C) @ d_output(C, tokens)
    // With CUBLAS_OP_N on both: M = D, N = tokens, K = C

    m_arr[e] = intermediate_size;    // M = D
    n_arr[e] = tokens_e;             // N = tokens
    k_arr[e] = hidden_size;          // K = C
    lda_arr[e] = intermediate_size;  // lda = D (leading dim of W in col-major)
    ldb_arr[e] = hidden_size;        // ldb = C (leading dim of d_output in col-major)
    ldc_arr[e] = intermediate_size;  // ldc = D (leading dim of d_input in col-major)
}

// Kernel to build pointer arrays for gate_up backward on device
template <typename T>
__global__ void build_gemm_pointers_gate_up_backward_kernel(const T** A_ptrs,  // output: d_gate_up pointers
                                                            const T** B_ptrs,  // output: weight pointers
                                                            T** C_ptrs,        // output: d_input pointers
                                                            int* lda_arr,
                                                            int* ldb_arr,
                                                            int* ldc_arr,
                                                            int* m_arr,
                                                            int* n_arr,
                                                            int* k_arr,
                                                            const T* d_gate_up,
                                                            const T* weights,
                                                            T* d_input,
                                                            const int* expert_offsets,
                                                            int num_experts,
                                                            int hidden_size,       // C
                                                            int intermediate_size  // D
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;

    int start = expert_offsets[e];
    int end = expert_offsets[e + 1];
    int tokens_e = end - start;

    // d_gate_up: (tokens_e, 2*D) at offset start * 2*D
    A_ptrs[e] = d_gate_up + start * (2 * intermediate_size);
    // Weight: (2*D, C) for expert e
    B_ptrs[e] = weights + e * (2 * intermediate_size) * hidden_size;
    // d_input: (tokens_e, C) at offset start * C
    C_ptrs[e] = d_input + start * hidden_size;

    // For backward: d_input = d_gate_up @ W (no transpose on W)
    // Row-major: d_input[t][c] = sum_d d_gate_up[t][d] * W[d][c]
    //
    // In column-major:
    // - d_gate_up is (2*D, tokens) col-major
    // - W is (C, 2*D) col-major (because row-major (2*D, C))
    // - d_input is (C, tokens) col-major
    //
    // So: d_input(C, tokens) = W(C, 2*D) @ d_gate_up(2*D, tokens)
    // With CUBLAS_OP_N on both: M = C, N = tokens, K = 2*D

    m_arr[e] = hidden_size;              // M = C
    n_arr[e] = tokens_e;                 // N = tokens
    k_arr[e] = 2 * intermediate_size;    // K = 2*D
    lda_arr[e] = hidden_size;            // lda = C (leading dim of W in col-major)
    ldb_arr[e] = 2 * intermediate_size;  // ldb = 2*D (leading dim of d_gate_up in col-major)
    ldc_arr[e] = hidden_size;            // ldc = C (leading dim of d_input in col-major)
}

template <typename T>
void moe_grouped_gemm_down_backward_impl(
    T* d_input,                 // (total_tokens, D) - gradient w.r.t. SwiGLU output
    const T* d_output,          // (total_tokens, C) - gradient from downstream
    const T* weights,           // (num_experts, C, D) - down_proj weights (ignored when weight_ptrs != null)
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

        m_vec.push_back(intermediate_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(hidden_size);

        lda_vec.push_back(intermediate_size);
        ldb_vec.push_back(hidden_size);
        ldc_vec.push_back(intermediate_size);

        const int weight_idx = weight_is_compact ? e : global_idx;
        A_vec.push_back(weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                    : weights + weight_idx * hidden_size * intermediate_size);
        B_vec.push_back(d_output + h_offsets[global_idx] * hidden_size);
        C_vec.push_back(d_input + h_offsets[global_idx] * intermediate_size);
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

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
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
void moe_grouped_gemm_gate_up_backward_impl(
    T* d_input,                 // (total_tokens, C) - gradient w.r.t. input
    const T* d_gate_up,         // (total_tokens, 2*D) - gradient from SwiGLU backward
    const T* weights,           // (num_experts, 2*D, C) - gate_up_proj weights (ignored when weight_ptrs != null)
    const int* expert_offsets,  // (num_experts + 1) - token offsets per expert (device)
    int num_experts,
    int hidden_size,        // C
    int intermediate_size,  // D (d_gate_up is 2*D)
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const int* host_offsets,  // Optional: pre-cached host offsets to avoid D2H sync
    const int* active_expert_indices,
    bool weight_is_compact,
    int num_active_experts,
    const void* const* weight_ptrs  // Optional: per-expert weight pointers (LLEP)
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
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
    const int gate_up_dim = 2 * intermediate_size;

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

        m_vec.push_back(hidden_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(gate_up_dim);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(gate_up_dim);
        ldc_vec.push_back(hidden_size);

        const int weight_idx = weight_is_compact ? e : global_idx;
        A_vec.push_back(weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                    : weights + weight_idx * gate_up_dim * hidden_size);
        B_vec.push_back(d_gate_up + h_offsets[global_idx] * gate_up_dim);
        C_vec.push_back(d_input + h_offsets[global_idx] * hidden_size);
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

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
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
void moe_grouped_gemm_up_backward_impl(
    T* d_input,                 // (total_tokens, C) - gradient w.r.t. input
    const T* d_up,              // (total_tokens, D) - gradient from activation backward
    const T* weights,           // (num_experts, D, C) - up projection weights
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
    const void* const* weight_ptrs = nullptr  // Optional: per-expert weight pointers (LLEP)
) {
    int n_active = (num_active_experts <= 0) ? num_experts : num_active_experts;
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
    const int up_dim = intermediate_size;

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

        m_vec.push_back(hidden_size);
        n_vec.push_back(tokens_e);
        k_vec.push_back(up_dim);

        lda_vec.push_back(hidden_size);
        ldb_vec.push_back(up_dim);
        ldc_vec.push_back(hidden_size);

        const int weight_idx = weight_is_compact ? e : global_idx;
        A_vec.push_back(weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx])
                                    : weights + weight_idx * up_dim * hidden_size);
        B_vec.push_back(d_up + h_offsets[global_idx] * up_dim);
        C_vec.push_back(d_input + h_offsets[global_idx] * hidden_size);
    }

    if (m_vec.empty()) return;

    const int gemm_count = static_cast<int>(m_vec.size());

    const T** d_A_array = nullptr;
    const T** d_B_array = nullptr;
    T** d_C_array = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_A_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_B_array), sizeof(T*) * gemm_count, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_C_array), sizeof(T*) * gemm_count, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, A_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, B_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, C_vec.data(), sizeof(T*) * gemm_count, cudaMemcpyHostToDevice, stream));

    std::vector<cublasOperation_t> transa_vec(gemm_count, CUBLAS_OP_N);
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

    CUDA_CHECK(cudaFreeAsync(d_A_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_B_array, stream));
    CUDA_CHECK(cudaFreeAsync(d_C_array, stream));
}

// Host wrappers for grouped GEMM backward
void moe_grouped_gemm_down_backward(nv_bfloat16* d_input,
                                    const nv_bfloat16* d_output,
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
    moe_grouped_gemm_down_backward_impl(d_input,
                                        d_output,
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

void moe_grouped_gemm_down_backward(float* d_input,
                                    const float* d_output,
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
    moe_grouped_gemm_down_backward_impl(d_input,
                                        d_output,
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

void moe_grouped_gemm_gate_up_backward(nv_bfloat16* d_input,
                                       const nv_bfloat16* d_gate_up,
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
    moe_grouped_gemm_gate_up_backward_impl(d_input,
                                           d_gate_up,
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

void moe_grouped_gemm_gate_up_backward(float* d_input,
                                       const float* d_gate_up,
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
    moe_grouped_gemm_gate_up_backward_impl(d_input,
                                           d_gate_up,
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

void moe_grouped_gemm_up_backward(nv_bfloat16* d_input,
                                  const nv_bfloat16* d_up,
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
    moe_grouped_gemm_up_backward_impl(d_input,
                                      d_up,
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

void moe_grouped_gemm_up_backward(float* d_input,
                                  const float* d_up,
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
    moe_grouped_gemm_up_backward_impl(d_input,
                                      d_up,
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
