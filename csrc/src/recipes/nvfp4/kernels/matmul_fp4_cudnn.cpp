// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP4 (NVFP4/E2M1) matmul using cuDNN frontend graph API.
// Requires: cuDNN 9.7.0+, Blackwell GPU (SM100+)

#include <cudnn_frontend.h>

#include "utilities/tensor.h"
#include "utilities/utils.h"

#include <cuda_fp8.h>

namespace fe = cudnn_frontend;

/**
 * @brief Checks a cuDNN status code and throws on error.
 */
static void cuDNNCheck_(cudnnStatus_t error, const char* file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cuDNNCheck(err) (cuDNNCheck_(err, __FILE__, __LINE__))

/**
 * @brief Checks a cuDNN frontend error object and throws on error.
 */
static void checkCudnnFE(const fe::error_object& e, const char* file, int line) {
    if (!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

/**
 * @brief Unique identifiers for FP4 matmul graph tensors.
 */
enum FP4UIDs {
    FP4_A_UID,               ///< Input tensor A (FP4 packed data)
    FP4_B_UID,               ///< Input tensor B (FP4 packed data)
    FP4_SCALE_A_UID,         ///< Block descale tensor for A (FP8 E4M3)
    FP4_SCALE_B_UID,         ///< Block descale tensor for B (FP8 E4M3)
    FP4_GLOBAL_SCALE_A_UID,  ///< Global scale for A (scalar FP32)
    FP4_GLOBAL_SCALE_B_UID,  ///< Global scale for B (scalar FP32)
    FP4_D_UID,               ///< Output tensor D (BF16)
};

/**
 * @brief Cache key for FP4 matmul graphs.
 * (M, N, K, block_size, output_dtype)
 */
using fp4_matmul_cache_key = std::tuple<int64_t, int64_t, int64_t, int32_t, int>;

/**
 * @brief Cache for compiled FP4 matmul graphs.
 */
using fp4_matmul_cache_type = std::map<fp4_matmul_cache_key, std::shared_ptr<fe::graph::Graph>>;

/**
 * @brief Round up to multiple helper.
 */
static inline int64_t round_up_to_multiple(int64_t x, int64_t multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}

/**
 * @brief Check if device supports FP4 operations (Blackwell SM100+).
 */
bool device_supports_fp4() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Blackwell is SM100+
    return prop.major >= 10;
}

/**
 * @brief Get cuDNN frontend data type from output dtype enum.
 */
static fe::DataType_t get_output_dtype(int dtype_id) {
    switch (dtype_id) {
        case 0: return fe::DataType_t::FLOAT;
        case 1: return fe::DataType_t::BFLOAT16;
        case 2: return fe::DataType_t::HALF;
        default: return fe::DataType_t::BFLOAT16;
    }
}

/**
 * @brief Builds and caches an FP4 matmul graph for the given dimensions.
 *
 * Creates a cuDNN frontend graph for FP4 block-scaled matmul:
 * D = dequant(A, scale_A) @ dequant(B, scale_B)
 *
 * For linear layer y = x @ W^T:
 * - A = input x: {M=BT, K=C} row-major
 * - B = weight W: stored as {N=OC, K=C} row-major, passed as {K=C, N=OC} column-major
 * - D = output y: {M=BT, N=OC} row-major
 *
 * Uses two-level block scaling:
 * - FP8 E4M3 block scales (per 16 consecutive values)
 * - FP32 global scale (per tensor)
 *
 * @param M Number of rows in A (batch*seq_len)
 * @param N Number of columns in output D (out_channels)
 * @param K Inner dimension (in_channels)
 * @param block_size Block size for quantization (typically 16 for FP4)
 * @param output_dtype_id Output data type (0=FP32, 1=BF16, 2=FP16)
 * @param cudnn_handle cuDNN handle for graph building
 * @return Shared pointer to the cached or newly built cuDNN graph
 */
static auto lookup_cache_or_build_fp4_matmul_graph(int64_t M,
                                                   int64_t N,
                                                   int64_t K,
                                                   int32_t block_size,
                                                   int output_dtype_id,
                                                   cudnnHandle_t cudnn_handle) {
    thread_local fp4_matmul_cache_type fp4_matmul_cache;

    auto key = std::make_tuple(M, N, K, block_size, output_dtype_id);
    auto it = fp4_matmul_cache.find(key);
    if (it != fp4_matmul_cache.end()) {
        return it->second;
    }

    // F8_128x4 block alignment constants
    constexpr int64_t BLOCK_128 = 128;
    constexpr int64_t BLOCK_4 = 4;

    // Compute scale tensor dimensions with F8_128x4 alignment
    // A (input): {M, K} -> scale: {ceil(M/128)*128, ceil(K/16/4)*4}
    // B (weight): {N, K} stored row-major -> scale: {ceil(N/128)*128, ceil(K/16/4)*4}
    int64_t rounded_m = round_up_to_multiple(M, BLOCK_128);
    int64_t rounded_n = round_up_to_multiple(N, BLOCK_128);
    int64_t rounded_block_scale_k = round_up_to_multiple(K / block_size, BLOCK_4);

    auto output_dtype = get_output_dtype(output_dtype_id);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    // Tensor A: input {M, K} row-major
    // FP4 is packed 2 values per byte, but cuDNN handles this internally
    auto tensor_a = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("tensor_a")
                                      .set_data_type(fe::DataType_t::FP4_E2M1)
                                      .set_dim({1, M, K})
                                      .set_stride({M * K, K, 1})
                                      .set_uid(FP4_A_UID));

    // Tensor B: weight {N, K} stored row-major, interpreted as {K, N} column-major
    // cuDNN matmul: D[M,N] = A[M,K] @ B[K,N]
    // We have weight stored as (OC, C) = (N, K) row-major
    // For column-major {K, N}, strides should be {1, K}
    // But since cuDNN expects dims {K, N} with column-major strides:
    // dims = {K, N}, strides = {1, K} means stride-1 on K dimension
    auto tensor_b = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("tensor_b")
                                      .set_data_type(fe::DataType_t::FP4_E2M1)
                                      .set_dim({1, K, N})
                                      .set_stride({K * N, 1, K})
                                      .set_uid(FP4_B_UID));

    // Block descale tensor for A (input): {rounded_m, rounded_block_scale_k}
    auto block_descale_a = graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("block_descale_a")
                                             .set_data_type(fe::DataType_t::FP8_E4M3)
                                             .set_dim({1, rounded_m, rounded_block_scale_k})
                                             .set_stride({rounded_m * rounded_block_scale_k, rounded_block_scale_k, 1})
                                             .set_reordering_type(fe::TensorReordering_t::F8_128x4)
                                             .set_uid(FP4_SCALE_A_UID));

    // Block descale tensor for B (weight): K-major layout expected by cuDNN for FP4 matmul.
    // (K/16, N) with F8_128x4 reordering (specialized for B operand).
    auto block_descale_b = graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("block_descale_b")
                                             .set_data_type(fe::DataType_t::FP8_E4M3)
                                             .set_dim({1, rounded_block_scale_k, rounded_n})
                                             .set_stride({rounded_n * rounded_block_scale_k, 1, rounded_block_scale_k})
                                             .set_reordering_type(fe::TensorReordering_t::F8_128x4)
                                             .set_uid(FP4_SCALE_B_UID));

    // Block scale dequantization for both operands
    auto dequant_attr = fe::graph::Block_scale_dequantize_attributes().set_block_size(block_size);

    auto dequant_a = graph->block_scale_dequantize(tensor_a, block_descale_a, dequant_attr);
    auto dequant_b = graph->block_scale_dequantize(tensor_b, block_descale_b, dequant_attr);

    // Matrix multiplication: D = A @ B
    auto matmul_attr = fe::graph::Matmul_attributes().set_name("FP4_GEMM").set_compute_data_type(fe::DataType_t::FLOAT);

    auto tensor_d = graph->matmul(dequant_a, dequant_b, matmul_attr);

    // Configure output tensor: {M, N} row-major
    tensor_d->set_output(true)
        .set_data_type(output_dtype)
        .set_dim({1, M, N})
        .set_stride({M * N, N, 1})
        .set_uid(FP4_D_UID);

    checkCudnnFE(graph->validate());

    // Build operation graph (expensive operation - cached)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    checkCudnnFE(graph->create_execution_plans({fe::HeurMode_t::A}));
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    fp4_matmul_cache.insert({key, graph});
    return graph;
}

/**
 * @brief Get required workspace size for FP4 matmul.
 *
 * @param M Number of rows in A and output D
 * @param N Number of columns in B and output D
 * @param K Number of columns in A / rows in B
 * @param block_size Block size for quantization (typically 16)
 * @param cudnn_handle cuDNN handle
 * @return Required workspace size in bytes
 */
std::size_t fp4_matmul_get_workspace_size(int M, int N, int K, int block_size, cudnnHandle_t cudnn_handle) {
    auto graph = lookup_cache_or_build_fp4_matmul_graph(M, N, K, block_size, 1, cudnn_handle);
    return graph->get_workspace_size();
}

/**
 * @brief Execute FP4 block-scaled matmul for linear layer: D = A @ B^T
 *
 * Implements y = x @ W^T (standard linear layer) with FP4 quantized operands.
 * Both operands use two-level block scaling (FP8 block scales + FP32 global scale).
 *
 * Memory layout (matches linear layer convention):
 * - A: input x, FP4 packed row-major (M, K) = (batch*seq, in_channels)
 * - B: weight W, FP4 packed row-major (N, K) = (out_channels, in_channels)
 *      cuDNN interprets B as column-major (K, N), effectively computing A @ B^T
 * - scale_a: FP8 E4M3, F8_128x4 swizzled, shape (ceil(M/128)*128, ceil(K/16/4)*4)
 * - scale_b: FP8 E4M3, F8_128x4 swizzled, shape (ceil(N/128)*128, ceil(K/16/4)*4)
 * - D: output y, BF16 row-major (M, N) = (batch*seq, out_channels)
 *
 * @param[out] d Output tensor (M, N) in BF16
 * @param[in] a Input tensor (M, K) in FP4 packed format, row-major
 * @param[in] b Weight tensor (N, K) in FP4 packed format, row-major
 * @param[in] scale_a Block descale tensor for A in FP8 E4M3, F8_128x4 swizzled
 * @param[in] scale_b Block descale tensor for B in FP8 E4M3, F8_128x4 swizzled
 * @param global_scale_a Global descale factor for A (unused, baked into block scales)
 * @param global_scale_b Global descale factor for B (unused, baked into block scales)
 * @param workspace Pre-allocated GPU workspace
 * @param workspace_size Size of workspace in bytes
 * @param M Batch dimension (batch * seq_len)
 * @param N Output channels
 * @param K Input channels
 * @param block_size Block size for quantization (typically 16 for FP4)
 * @param cudnn_handle cuDNN handle
 * @param stream CUDA stream for execution
 */
void fp4_matmul(nv_bfloat16* d,
                const uint8_t* a,
                const uint8_t* b,
                const __nv_fp8_e4m3* scale_a,
                const __nv_fp8_e4m3* scale_b,
                [[maybe_unused]] float global_scale_a,
                [[maybe_unused]] float global_scale_b,
                std::byte* workspace,
                [[maybe_unused]] std::size_t workspace_size,
                int M,
                int N,
                int K,
                int block_size,
                cudnnHandle_t cudnn_handle,
                cudaStream_t stream) {
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get or build the cached graph
    auto graph = lookup_cache_or_build_fp4_matmul_graph(M, N, K, block_size, 1, cudnn_handle);

    // Build variant pack mapping UIDs to device pointers
    // Note: Global scales are baked into block scales during quantization,
    // so we don't need to pass them separately to the graph
    std::unordered_map<int64_t, void*> variant_pack = {
        {FP4_A_UID, (void*)a},
        {FP4_B_UID, (void*)b},
        {FP4_SCALE_A_UID, (void*)scale_a},
        {FP4_SCALE_B_UID, (void*)scale_b},
        {FP4_D_UID, (void*)d},
    };

    // Execute the graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Execute FP4 block-scaled matmul with FP32 output.
 */
void fp4_matmul_f32(float* d,
                    const uint8_t* a,
                    const uint8_t* b,
                    const __nv_fp8_e4m3* scale_a,
                    const __nv_fp8_e4m3* scale_b,
                    [[maybe_unused]] float global_scale_a,
                    [[maybe_unused]] float global_scale_b,
                    std::byte* workspace,
                    [[maybe_unused]] std::size_t workspace_size,
                    int M,
                    int N,
                    int K,
                    int block_size,
                    cudnnHandle_t cudnn_handle,
                    cudaStream_t stream) {
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get or build the cached graph with FP32 output
    auto graph = lookup_cache_or_build_fp4_matmul_graph(M, N, K, block_size, 0, cudnn_handle);

    std::unordered_map<int64_t, void*> variant_pack = {
        {FP4_A_UID, (void*)a},
        {FP4_B_UID, (void*)b},
        {FP4_SCALE_A_UID, (void*)scale_a},
        {FP4_SCALE_B_UID, (void*)scale_b},
        {FP4_D_UID, (void*)d},
    };

    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Tensor-based wrapper for FP4 matmul.
 */
void fp4_matmul(Tensor& d,
                const Tensor& a,
                const Tensor& b,
                const Tensor& scale_a,
                const Tensor& scale_b,
                float global_scale_a,
                float global_scale_b,
                Tensor& workspace,
                int M,
                int N,
                int K,
                int block_size,
                cudnnHandle_t cudnn_handle,
                cudaStream_t stream) {
    if (d.DType == ETensorDType::BF16) {
        fp4_matmul(d.get<nv_bfloat16>(),
                   a.get<uint8_t>(),
                   b.get<uint8_t>(),
                   scale_a.get<__nv_fp8_e4m3>(),
                   scale_b.get<__nv_fp8_e4m3>(),
                   global_scale_a,
                   global_scale_b,
                   workspace.get<std::byte>(),
                   workspace.bytes(),
                   M,
                   N,
                   K,
                   block_size,
                   cudnn_handle,
                   stream);
    } else if (d.DType == ETensorDType::FP32) {
        fp4_matmul_f32(d.get<float>(),
                       a.get<uint8_t>(),
                       b.get<uint8_t>(),
                       scale_a.get<__nv_fp8_e4m3>(),
                       scale_b.get<__nv_fp8_e4m3>(),
                       global_scale_a,
                       global_scale_b,
                       workspace.get<std::byte>(),
                       workspace.bytes(),
                       M,
                       N,
                       K,
                       block_size,
                       cudnn_handle,
                       stream);
    } else {
        throw std::runtime_error("fp4_matmul: unsupported output dtype (must be BF16 or FP32)");
    }
}
