// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "kernels/moe/moe_common.cuh"

// Split from src/kernels/moe_kernels.cu: moe_grouped_gemm_core.cu.

// ============================================================================
// Grouped GEMM for MoE Expert Computation
// ============================================================================
// Uses cuBLAS grouped batched GEMM to run all experts in parallel.
// This reduces kernel launch overhead from O(num_experts) to O(1).
//
// The expert weights are stored in a batched layout:
//   gate_up_proj: (num_experts, 2*D, C)
//   down_proj:    (num_experts, C, D)
//
// Input tokens are permuted to expert-grouped order, with expert_offsets[e]
// pointing to where expert e's tokens start.

// Kernel to build pointer arrays on device (avoids host-device sync)
template <typename T>
__global__ void build_gemm_pointers_gate_up_kernel(const T** A_ptrs,  // output: input pointers
                                                   const T** B_ptrs,  // output: weight pointers
                                                   T** C_ptrs,        // output: output pointers
                                                   int* lda_arr,
                                                   int* ldb_arr,
                                                   int* ldc_arr,
                                                   int* m_arr,
                                                   int* n_arr,
                                                   int* k_arr,
                                                   const T* input,
                                                   const T* weights,
                                                   T* output,
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

    // Input: (tokens_e, C) at offset start * C
    A_ptrs[e] = input + start * hidden_size;
    // Weight: (2*D, C) for expert e
    B_ptrs[e] = weights + e * (2 * intermediate_size) * hidden_size;
    // Output: (tokens_e, 2*D) at offset start * 2*D
    C_ptrs[e] = output + start * (2 * intermediate_size);

    // Row-major: output(tokens, 2*D) = input(tokens, C) @ weight^T(C, 2*D)
    // In column-major (treating row-major as col-major):
    // - input becomes (C, tokens) col-major
    // - weight becomes (C, 2*D) col-major
    // - output becomes (2*D, tokens) col-major
    //
    // So: output(2*D, tokens) = weight^T(2*D, C) @ input(C, tokens)
    // cuBLAS: C = op(A) @ op(B)
    // A = weight with CUBLAS_OP_T: op(A) = (2*D, C)
    // B = input with CUBLAS_OP_N: op(B) = (C, tokens)
    // M = 2*D, N = tokens, K = C

    m_arr[e] = 2 * intermediate_size;    // M = 2*D
    n_arr[e] = tokens_e;                 // N = tokens
    k_arr[e] = hidden_size;              // K = C
    lda_arr[e] = hidden_size;            // lda = C (leading dim of weight in col-major)
    ldb_arr[e] = hidden_size;            // ldb = C (leading dim of input in col-major)
    ldc_arr[e] = 2 * intermediate_size;  // ldc = 2*D (leading dim of output in col-major)
}

template <typename T>
__global__ void build_gemm_pointers_down_kernel(const T** A_ptrs,
                                                const T** B_ptrs,
                                                T** C_ptrs,
                                                int* lda_arr,
                                                int* ldb_arr,
                                                int* ldc_arr,
                                                int* m_arr,
                                                int* n_arr,
                                                int* k_arr,
                                                const T* input,
                                                const T* weights,
                                                T* output,
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

    // Input: (tokens_e, D) at offset start * D
    A_ptrs[e] = input + start * intermediate_size;
    // Weight: (C, D) for expert e
    B_ptrs[e] = weights + e * hidden_size * intermediate_size;
    // Output: (tokens_e, C) at offset start * C
    C_ptrs[e] = output + start * hidden_size;

    // Row-major: output(tokens, C) = input(tokens, D) @ weight^T(D, C)
    // Col-major: output(C, tokens) = weight^T(C, D) @ input(D, tokens)
    // A = weight with CUBLAS_OP_T: op(A) = (C, D)
    // B = input with CUBLAS_OP_N: op(B) = (D, tokens)
    // M = C, N = tokens, K = D

    m_arr[e] = hidden_size;          // M = C
    n_arr[e] = tokens_e;             // N = tokens
    k_arr[e] = intermediate_size;    // K = D
    lda_arr[e] = intermediate_size;  // lda = D
    ldb_arr[e] = intermediate_size;  // ldb = D
    ldc_arr[e] = hidden_size;        // ldc = C
}

template <typename T>
void moe_grouped_gemm_impl(T* output,
                           const T* input,
                           const T* weights,
                           const int* expert_offsets,
                           int num_experts,
                           int M,
                           int K,
                           cublasHandle_t cublas_handle,
                           cudaStream_t stream,
                           const int* host_offsets,
                           float alpha,
                           float beta,
                           EMMTranspose mode,
                           const int* active_expert_indices,
                           bool weight_is_compact,
                           int num_active_experts,
                           const void* const* weight_ptrs = nullptr) {
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

    cublasOperation_t transa = (mode == EMMTranspose::TN || mode == EMMTranspose::TT) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (mode == EMMTranspose::NT || mode == EMMTranspose::TT) ? CUBLAS_OP_T : CUBLAS_OP_N;

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

        m_vec.push_back(M);
        n_vec.push_back(tokens_e);
        k_vec.push_back(K);

        // Row-major A(M, K) @ B(K, N) = C(M, N)
        // In column-major: C(M, N) = A(M, K) @ B(K, N)
        // transa on A, transb on B
        lda_vec.push_back((transa == CUBLAS_OP_N) ? M : K);
        ldb_vec.push_back((transb == CUBLAS_OP_N) ? K : tokens_e);
        ldc_vec.push_back(M);

        const int weight_idx = weight_is_compact ? e : global_idx;
        A_vec.push_back(weight_ptrs ? static_cast<const T*>(weight_ptrs[weight_idx]) : weights + weight_idx * M * K);
        B_vec.push_back(input + h_offsets[global_idx] * K);
        C_vec.push_back(output + h_offsets[global_idx] * M);
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

    std::vector<cublasOperation_t> transa_vec(gemm_count, transa);
    std::vector<cublasOperation_t> transb_vec(gemm_count, transb);
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

void moe_grouped_gemm(float* output,
                      const float* input,
                      const float* weights,
                      const int* expert_offsets,
                      int num_experts,
                      int M,
                      int K,
                      cublasHandle_t cublas_handle,
                      cudaStream_t stream,
                      const int* host_offsets,
                      float alpha,
                      float beta,
                      EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts,
                      const void* const* weight_ptrs) {
    moe_grouped_gemm_impl(output,
                          input,
                          weights,
                          expert_offsets,
                          num_experts,
                          M,
                          K,
                          cublas_handle,
                          stream,
                          host_offsets,
                          alpha,
                          beta,
                          mode,
                          active_expert_indices,
                          weight_is_compact,
                          num_active_experts,
                          weight_ptrs);
}

void moe_grouped_gemm(nv_bfloat16* output,
                      const nv_bfloat16* input,
                      const nv_bfloat16* weights,
                      const int* expert_offsets,
                      int num_experts,
                      int M,
                      int K,
                      cublasHandle_t cublas_handle,
                      cudaStream_t stream,
                      const int* host_offsets,
                      float alpha,
                      float beta,
                      EMMTranspose mode,
                      const int* active_expert_indices,
                      bool weight_is_compact,
                      int num_active_experts,
                      const void* const* weight_ptrs) {
    moe_grouped_gemm_impl(output,
                          input,
                          weights,
                          expert_offsets,
                          num_experts,
                          M,
                          K,
                          cublas_handle,
                          stream,
                          host_offsets,
                          alpha,
                          beta,
                          mode,
                          active_expert_indices,
                          weight_is_compact,
                          num_active_experts,
                          weight_ptrs);
}

template <typename T>
void moe_grouped_gemm_weight_grad_impl(T* d_weight,
                                       const T* grad_output,
                                       const T* input,
                                       const int* expert_offsets,
                                       int num_experts,
                                       int M,
                                       int N,
                                       cublasHandle_t cublas_handle,
                                       cudaStream_t stream,
                                       const int* host_offsets,
                                       float alpha,
                                       float beta,
                                       const int* active_expert_indices,
                                       bool weight_is_compact,
                                       int num_active_experts) {
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

    // dW(M, N) = grad_output^T(M, K) @ input(K, N)  where K = tokens_e
    // In column-major: dW(M, N) = A @ B
    // A is grad_output treated as (K, M) col-major => A^T is (M, K)
    // B is input treated as (K, N) col-major => B is (K, N)

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

        m_vec.push_back(M);
        n_vec.push_back(N);
        k_vec.push_back(tokens_e);

        lda_vec.push_back(M);
        ldb_vec.push_back(N);
        ldc_vec.push_back(M);

        // Row-major grad_output is (tokens, M). Treated as col-major it's (M, tokens).
        // Transpose A (CUBLAS_OP_T) gives (M, tokens)? NO.
        // If row-major (tokens, M) is treated as col-major (M, tokens),
        // we want result (M, N).
        // C(M, N) = A(M, K) @ B(K, N)
        // A is grad_output(M, K) col-major. OP_N.
        // B is input(K, N) col-major. OP_T?
        // Row-major input is (K, N). Treated as col-major it's (N, K).
        // OP_T on B gives (K, N).
        // So: C(M, N) = A(M, K) @ B^T(K, N)

        A_vec.push_back(grad_output + h_offsets[global_idx] * M);
        B_vec.push_back(input + h_offsets[global_idx] * N);
        C_vec.push_back(d_weight + (weight_is_compact ? e : global_idx) * M * N);
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
    std::vector<cublasOperation_t> transb_vec(gemm_count, CUBLAS_OP_T);
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

void moe_grouped_gemm_weight_grad(float* d_weight,
                                  const float* grad_output,
                                  const float* input,
                                  const int* expert_offsets,
                                  int num_experts,
                                  int M,
                                  int N,
                                  cublasHandle_t cublas_handle,
                                  cudaStream_t stream,
                                  const int* host_offsets,
                                  float alpha,
                                  float beta,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    moe_grouped_gemm_weight_grad_impl(d_weight,
                                      grad_output,
                                      input,
                                      expert_offsets,
                                      num_experts,
                                      M,
                                      N,
                                      cublas_handle,
                                      stream,
                                      host_offsets,
                                      alpha,
                                      beta,
                                      active_expert_indices,
                                      weight_is_compact,
                                      num_active_experts);
}

void moe_grouped_gemm_weight_grad(nv_bfloat16* d_weight,
                                  const nv_bfloat16* grad_output,
                                  const nv_bfloat16* input,
                                  const int* expert_offsets,
                                  int num_experts,
                                  int M,
                                  int N,
                                  cublasHandle_t cublas_handle,
                                  cudaStream_t stream,
                                  const int* host_offsets,
                                  float alpha,
                                  float beta,
                                  const int* active_expert_indices,
                                  bool weight_is_compact,
                                  int num_active_experts) {
    moe_grouped_gemm_weight_grad_impl(d_weight,
                                      grad_output,
                                      input,
                                      expert_offsets,
                                      num_experts,
                                      M,
                                      N,
                                      cublas_handle,
                                      stream,
                                      host_offsets,
                                      alpha,
                                      beta,
                                      active_expert_indices,
                                      weight_is_compact,
                                      num_active_experts);
}

bool fp8_wgrad_grouped_shape_supported(int M, int N) {
    // FP8 grouped GEMM paths route to tensor-core kernels on Ada/Hopper/Blackwell.
    // Keep the rollout conservative and let callers use the BF16 fallback for
    // odd projection dimensions.
    return M > 0 && N > 0 && (M % 16) == 0 && (N % 16) == 0;
}

bool aligned_16(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % 16) == 0;
}

void moe_grouped_gemm_weight_grad_fp8(nv_bfloat16* d_weight,
                                      const __nv_fp8_e5m2* grad_output,
                                      const __nv_fp8_e4m3* input,
                                      const float* grad_output_scale,
                                      const float* input_scale,
                                      const int* expert_offsets,
                                      int num_experts,
                                      int M,
                                      int N,
                                      cublasHandle_t cublas_handle,
                                      cudaStream_t stream,
                                      const int* host_offsets,
                                      float alpha,
                                      float beta,
                                      const int* active_expert_indices,
                                      bool weight_is_compact,
                                      int num_active_experts) {
    (void)cublas_handle;
    if (!host_offsets) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 requires host_offsets");
    }
    if (!d_weight || !grad_output || !input || !grad_output_scale || !input_scale) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 received null required pointer");
    }
    if (!fp8_wgrad_grouped_shape_supported(M, N)) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 unsupported projection alignment");
    }
    if (!aligned_16(d_weight) || !aligned_16(grad_output) || !aligned_16(input)) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 requires 16-byte aligned tensors");
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error("moe_grouped_gemm_weight_grad_fp8 grouped path is disabled during CUDA graph capture");
    }

    float h_scales[2] = {1.0f, 1.0f};
    CUDA_CHECK(cudaMemcpyAsync(&h_scales[0], grad_output_scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_scales[1], input_scale, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaDeviceProp props{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major < 9) {
        throw std::runtime_error(
            "moe_grouped_gemm_weight_grad_fp8 CUTLASS path requires Hopper/Blackwell tensor cores");
    }

    moe_grouped_gemm_weight_grad_fp8_cutlass(d_weight,
                                             grad_output,
                                             input,
                                             num_experts,
                                             M,
                                             N,
                                             stream,
                                             host_offsets,
                                             alpha * h_scales[0] * h_scales[1],
                                             beta,
                                             active_expert_indices,
                                             weight_is_compact,
                                             num_active_experts);

    static int debug_log_count = 0;
    if (std::getenv("SUROGATE_DEBUG_FP8_MOE_WGRAD") && debug_log_count++ < 8) {
        std::fprintf(stderr,
                     "[FP8 MoE wgrad] CUTLASS tensor-core path: sm=%d%d M=%d N=%d\n",
                     props.major,
                     props.minor,
                     M,
                     N);
    }
}
