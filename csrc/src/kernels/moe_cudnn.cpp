// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// cuDNN Frontend MoE grouped matmul.
// Requires cuDNN runtime >= 9.15.0 (moe_grouped_matmul operation).

#include <cudnn_frontend.h>

#include "utilities/utils.h"

namespace fe = cudnn_frontend;

static void cuDNNCheck_(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cuDNNCheck(err) (cuDNNCheck_(err, __FILE__, __LINE__))

static void checkCudnnFE_(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN FE ERROR] at %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE_(err, __FILE__, __LINE__)

// UIDs for BF16 MoE grouped matmul graph tensors
enum MoeUIDs {
    MOE_TOKEN_UID = 100,
    MOE_WEIGHT_UID,
    MOE_FIRST_TOKEN_OFFSET_UID,
    MOE_OUTPUT_UID,
};

// Cache key: (num_experts, total_tokens, K, N)
using moe_gemm_cache_key = std::tuple<int64_t, int64_t, int64_t, int64_t>;
using moe_gemm_cache_type = std::map<moe_gemm_cache_key, std::shared_ptr<fe::graph::Graph>>;

// Minimum cuDNN runtime version for moe_grouped_matmul support
static constexpr size_t CUDNN_MOE_MIN_VERSION = 91500;

static void check_cudnn_moe_support() {
    static bool checked = false;
    if (checked) return;
    checked = true;
    size_t runtime_ver = cudnnGetVersion();
    if (runtime_ver < CUDNN_MOE_MIN_VERSION) {
        printf("[moe_cudnn] FATAL: cuDNN runtime %zu.%zu.%zu is too old for MoE grouped matmul "
               "(requires >= 9.15.0, compiled against %d.%d.%d).\n"
               "Update nvidia-cudnn-cu12: pip install nvidia-cudnn-cu12>=9.15.0\n",
               runtime_ver / 10000, (runtime_ver % 10000) / 100, runtime_ver % 100,
               CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
        exit(EXIT_FAILURE);
    }
}

/// Build (or retrieve from cache) a cuDNN FE graph for BF16 MoE grouped GEMM.
///
/// Computes: output[t, n] = sum_k token[t, k] * weight[e, k, n]
/// for each expert e, over the token range defined by first_token_offset.
///
/// Weight layout: Surogate stores (E, N, K) row-major. cuDNN FE expects (E, K, N).
/// We set dim=(E, K, N) stride=(K*N, 1, K) which maps to the same memory layout.
static auto lookup_cache_or_build_moe_gemm_graph(
    int64_t num_experts, int64_t total_tokens, int64_t K, int64_t N,
    cudnnHandle_t cudnn_handle)
{
    thread_local moe_gemm_cache_type cache;

    auto key = std::make_tuple(num_experts, total_tokens, K, N);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    check_cudnn_moe_support();

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Token tensor: (1, total_tokens, K) row-major
    auto token = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("token")
        .set_dim({1, total_tokens, K})
        .set_stride({total_tokens * K, K, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_uid(MOE_TOKEN_UID));

    // Weight tensor: logical (E, K, N), physical memory is (E, N, K) row-major
    // stride {K*N, 1, K} maps logical [e,k,n] -> offset e*K*N + k + n*K = e*N*K + n*K + k
    auto weight = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("weight")
        .set_dim({num_experts, K, N})
        .set_stride({K * N, 1, K})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_uid(MOE_WEIGHT_UID));

    // First token offset: (num_experts, 1, 1) INT32
    auto first_token_offset = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("first_token_offset")
        .set_dim({num_experts, 1, 1})
        .set_stride({1, 1, 1})
        .set_data_type(fe::DataType_t::INT32)
        .set_uid(MOE_FIRST_TOKEN_OFFSET_UID));

    // MoE grouped matmul: NONE mode (tokens already permuted)
    auto moe_attr = fe::graph::Moe_grouped_matmul_attributes()
        .set_name("moe_gemm")
        .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto output = graph->moe_grouped_matmul(
        token, weight, first_token_offset,
        nullptr, nullptr, moe_attr);

    output->set_output(true)
           .set_data_type(fe::DataType_t::BFLOAT16)
           .set_dim({1, total_tokens, N})
           .set_stride({total_tokens * N, N, 1})
           .set_uid(MOE_OUTPUT_UID);

    checkCudnnFE(graph->validate());
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    checkCudnnFE(graph->create_execution_plans({fe::HeurMode_t::A}));
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

    cache.insert({key, graph});
    return graph;
}

// =============================================================================
// FP8 Weight-Only Quantization (WoQ) MoE GEMM via cuDNN Frontend
// =============================================================================
// Uses block_scale_dequantize(FP8_weight) → moe_grouped_matmul(BF16_token, dequant_weight).
// Requires cuDNN >= 9.18.0 and pre-quantized FP8 E4M3 expert weights with block scales.
// Block scale layout follows the cuDNN convention: block_size={block_size, 1},
// meaning 1D blocking along the K (reduction) dimension.

#if (CUDNN_VERSION >= 91800)

// UIDs for FP8 WoQ MoE graph tensors (separate range from BF16 UIDs)
enum MoeFP8UIDs {
    MOE_FP8_TOKEN_UID = 200,
    MOE_FP8_WEIGHT_UID,
    MOE_FP8_BLOCK_SCALE_UID,
    MOE_FP8_FIRST_TOKEN_OFFSET_UID,
    MOE_FP8_OUTPUT_UID,
};

// Cache key: (num_experts, total_tokens, K, N, block_size)
using moe_fp8_cache_key = std::tuple<int64_t, int64_t, int64_t, int64_t, int32_t>;
using moe_fp8_cache_type = std::map<moe_fp8_cache_key, std::shared_ptr<fe::graph::Graph>>;

// Minimum cuDNN runtime version for FP8 WoQ MoE support
static constexpr size_t CUDNN_MOE_FP8_MIN_VERSION = 91800;

/// Build (or retrieve from cache) a cuDNN FE graph for FP8 WoQ MoE grouped GEMM.
///
/// Graph: block_scale_dequantize(FP8_weight, block_scales) → moe_grouped_matmul.
/// Token stays BF16; only the weight is quantized to FP8 E4M3 with per-block scales.
///
/// Block scale convention: block_size along K dimension, 1 along N.
/// Scale tensor shape: (E, ceil(K/bs), N) with transposed strides matching weight layout.
static auto lookup_cache_or_build_moe_fp8_graph(
    int64_t num_experts, int64_t total_tokens, int64_t K, int64_t N,
    int32_t block_size, cudnnHandle_t cudnn_handle)
{
    thread_local moe_fp8_cache_type cache;

    auto key = std::make_tuple(num_experts, total_tokens, K, N, block_size);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return std::make_pair(it->second, true);
    }

    check_cudnn_moe_support();

    // Additional runtime check for FP8 WoQ support
    size_t runtime_ver = cudnnGetVersion();
    if (runtime_ver < CUDNN_MOE_FP8_MIN_VERSION) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Token tensor: BF16 (1, total_tokens, K) row-major
    auto token = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("token")
        .set_dim({1, total_tokens, K})
        .set_stride({total_tokens * K, K, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_uid(MOE_FP8_TOKEN_UID));

    // Weight tensor: FP8 E4M3, logical (E, K, N), physical (E, N, K) row-major
    auto weight_fp8 = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("weight_fp8")
        .set_dim({num_experts, K, N})
        .set_stride({K * N, 1, K})
        .set_data_type(fe::DataType_t::FP8_E4M3)
        .set_uid(MOE_FP8_WEIGHT_UID));

    // Block scale tensor: FP32, shape (E, ceil(K/bs), N)
    // Physical layout matches weight: (E, N, ceil(K/bs)) row-major
    // stride = {ceil(K/bs)*N, 1, ceil(K/bs)}  →  scale[e][k_blk][n] = e*Kb*N + k_blk + n*Kb
    const int64_t K_blocks = (K + block_size - 1) / block_size;
    auto block_scale = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("block_scale")
        .set_dim({num_experts, K_blocks, N})
        .set_stride({K_blocks * N, 1, K_blocks})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(MOE_FP8_BLOCK_SCALE_UID));

    // Block scale dequantize: FP8 weight → virtual BF16 weight (fused with matmul)
    auto dequant_attr = fe::graph::Block_scale_dequantize_attributes()
        .set_block_size({block_size, 1})
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto weight_dequant = graph->block_scale_dequantize(weight_fp8, block_scale, dequant_attr);
    weight_dequant->set_data_type(fe::DataType_t::BFLOAT16);

    // First token offset: INT32 (E, 1, 1)
    auto first_token_offset = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("first_token_offset")
        .set_dim({num_experts, 1, 1})
        .set_stride({1, 1, 1})
        .set_data_type(fe::DataType_t::INT32)
        .set_uid(MOE_FP8_FIRST_TOKEN_OFFSET_UID));

    // MoE grouped matmul with dequantized weight
    auto moe_attr = fe::graph::Moe_grouped_matmul_attributes()
        .set_name("moe_gemm_fp8_woq")
        .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto output = graph->moe_grouped_matmul(
        token, weight_dequant, first_token_offset,
        nullptr, nullptr, moe_attr);

    output->set_output(true)
           .set_data_type(fe::DataType_t::BFLOAT16)
           .set_dim({1, total_tokens, N})
           .set_stride({total_tokens * N, N, 1})
           .set_uid(MOE_FP8_OUTPUT_UID);

    auto status = graph->validate();
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->build_operation_graph(cudnn_handle);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->check_support(cudnn_handle);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->build_plans(cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }

    cache.insert({key, graph});
    return std::make_pair(graph, true);
}

/// Execute FP8 WoQ MoE grouped GEMM via cuDNN FE.
/// Returns true on success, false if FP8 WoQ is not supported (caller should fall back to BF16).
bool moe_cudnn_grouped_gemm_fp8(
    nv_bfloat16* output,
    const nv_bfloat16* input,
    const void* weights_fp8,
    const float* block_scales,
    const int* expert_offsets,
    int num_experts,
    int N, int K,
    int total_tokens,
    int block_size,
    cudnnHandle_t cudnn_handle,
    std::byte* workspace,
    [[maybe_unused]] std::size_t workspace_size,
    cudaStream_t stream)
{
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    auto [graph, supported] = lookup_cache_or_build_moe_fp8_graph(
        num_experts, total_tokens, K, N, block_size, cudnn_handle);

    if (!supported || !graph) {
        return false;
    }

    std::unordered_map<int64_t, void*> variant_pack = {
        {MOE_FP8_TOKEN_UID, (void*)input},
        {MOE_FP8_WEIGHT_UID, (void*)weights_fp8},
        {MOE_FP8_BLOCK_SCALE_UID, (void*)block_scales},
        {MOE_FP8_FIRST_TOKEN_OFFSET_UID, (void*)expert_offsets},
        {MOE_FP8_OUTPUT_UID, (void*)output},
    };

    auto status = graph->execute(cudnn_handle, variant_pack, workspace);
    if (!status.is_good()) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// =============================================================================
// FP4 Weight-Only Quantization (WoQ) MoE GEMM via cuDNN Frontend
// =============================================================================
// Uses block_scale_dequantize(FP4_E2M1_weight) → moe_grouped_matmul(BF16_token, dequant_weight).
// Requires pre-quantized FP4 E2M1 expert weights with FP32 block scales.
// The caller must pre-combine NVFP4's two-level scales (FP8 block × FP32 global) into FP32.

// UIDs for FP4 WoQ MoE graph tensors (separate range from BF16 and FP8 UIDs)
enum MoeFP4UIDs {
    MOE_FP4_TOKEN_UID = 300,
    MOE_FP4_WEIGHT_UID,
    MOE_FP4_BLOCK_SCALE_UID,
    MOE_FP4_FIRST_TOKEN_OFFSET_UID,
    MOE_FP4_OUTPUT_UID,
};

// Cache key: (num_experts, total_tokens, K, N, block_size)
using moe_fp4_cache_key = std::tuple<int64_t, int64_t, int64_t, int64_t, int32_t>;
using moe_fp4_cache_type = std::map<moe_fp4_cache_key, std::shared_ptr<fe::graph::Graph>>;

/// Build (or retrieve from cache) a cuDNN FE graph for FP4 WoQ MoE grouped GEMM.
///
/// Graph: block_scale_dequantize(FP4_weight, block_scales) → moe_grouped_matmul.
/// Token stays BF16; only the weight is quantized to FP4 E2M1 with per-block FP32 scales.
///
/// Block scale convention: block_size along K dimension, 1 along N.
/// Scale tensor shape: (E, ceil(K/bs), N) with transposed strides matching weight layout.
static auto lookup_cache_or_build_moe_fp4_graph(
    int64_t num_experts, int64_t total_tokens, int64_t K, int64_t N,
    int32_t block_size, cudnnHandle_t cudnn_handle)
{
    thread_local moe_fp4_cache_type cache;

    auto key = std::make_tuple(num_experts, total_tokens, K, N, block_size);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return std::make_pair(it->second, true);
    }

    check_cudnn_moe_support();

    // FP4 WoQ requires same cuDNN version as FP8 WoQ
    size_t runtime_ver = cudnnGetVersion();
    if (runtime_ver < CUDNN_MOE_FP8_MIN_VERSION) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // Token tensor: BF16 (1, total_tokens, K) row-major
    auto token = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("token")
        .set_dim({1, total_tokens, K})
        .set_stride({total_tokens * K, K, 1})
        .set_data_type(fe::DataType_t::BFLOAT16)
        .set_uid(MOE_FP4_TOKEN_UID));

    // Weight tensor: FP4 E2M1, logical (E, K, N), physical (E, N, K) row-major
    // cuDNN handles 4-bit packing internally — dims/strides are in logical elements.
    auto weight_fp4 = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("weight_fp4")
        .set_dim({num_experts, K, N})
        .set_stride({K * N, 1, K})
        .set_data_type(fe::DataType_t::FP4_E2M1)
        .set_uid(MOE_FP4_WEIGHT_UID));

    // Block scale tensor: FP32, shape (E, ceil(K/bs), N)
    // Physical layout matches weight: (E, N, ceil(K/bs)) row-major
    const int64_t K_blocks = (K + block_size - 1) / block_size;
    auto block_scale = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("block_scale")
        .set_dim({num_experts, K_blocks, N})
        .set_stride({K_blocks * N, 1, K_blocks})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(MOE_FP4_BLOCK_SCALE_UID));

    // Block scale dequantize: FP4 weight → virtual BF16 weight (fused with matmul)
    auto dequant_attr = fe::graph::Block_scale_dequantize_attributes()
        .set_block_size({block_size, 1})
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto weight_dequant = graph->block_scale_dequantize(weight_fp4, block_scale, dequant_attr);
    weight_dequant->set_data_type(fe::DataType_t::BFLOAT16);

    // First token offset: INT32 (E, 1, 1)
    auto first_token_offset = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("first_token_offset")
        .set_dim({num_experts, 1, 1})
        .set_stride({1, 1, 1})
        .set_data_type(fe::DataType_t::INT32)
        .set_uid(MOE_FP4_FIRST_TOKEN_OFFSET_UID));

    // MoE grouped matmul with dequantized weight
    auto moe_attr = fe::graph::Moe_grouped_matmul_attributes()
        .set_name("moe_gemm_fp4_woq")
        .set_mode(fe::MoeGroupedMatmulMode_t::NONE)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto output = graph->moe_grouped_matmul(
        token, weight_dequant, first_token_offset,
        nullptr, nullptr, moe_attr);

    output->set_output(true)
           .set_data_type(fe::DataType_t::BFLOAT16)
           .set_dim({1, total_tokens, N})
           .set_stride({total_tokens * N, N, 1})
           .set_uid(MOE_FP4_OUTPUT_UID);

    auto status = graph->validate();
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->build_operation_graph(cudnn_handle);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->create_execution_plans({fe::HeurMode_t::A});
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->check_support(cudnn_handle);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }
    status = graph->build_plans(cudnn_handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
        return std::make_pair(std::shared_ptr<fe::graph::Graph>(nullptr), false);
    }

    cache.insert({key, graph});
    return std::make_pair(graph, true);
}

/// Execute FP4 WoQ MoE grouped GEMM via cuDNN FE.
/// Returns true on success, false if FP4 WoQ is not supported (caller should fall back to BF16).
bool moe_cudnn_grouped_gemm_fp4(
    nv_bfloat16* output,
    const nv_bfloat16* input,
    const void* weights_fp4,
    const float* block_scales,
    const int* expert_offsets,
    int num_experts,
    int N, int K,
    int total_tokens,
    int block_size,
    cudnnHandle_t cudnn_handle,
    std::byte* workspace,
    [[maybe_unused]] std::size_t workspace_size,
    cudaStream_t stream)
{
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    auto [graph, supported] = lookup_cache_or_build_moe_fp4_graph(
        num_experts, total_tokens, K, N, block_size, cudnn_handle);

    if (!supported || !graph) {
        return false;
    }

    std::unordered_map<int64_t, void*> variant_pack = {
        {MOE_FP4_TOKEN_UID, (void*)input},
        {MOE_FP4_WEIGHT_UID, (void*)weights_fp4},
        {MOE_FP4_BLOCK_SCALE_UID, (void*)block_scales},
        {MOE_FP4_FIRST_TOKEN_OFFSET_UID, (void*)expert_offsets},
        {MOE_FP4_OUTPUT_UID, (void*)output},
    };

    auto status = graph->execute(cudnn_handle, variant_pack, workspace);
    if (!status.is_good()) {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else  // CUDNN_VERSION < 91800

bool moe_cudnn_grouped_gemm_fp8(
    nv_bfloat16*, const nv_bfloat16*, const void*, const float*,
    const int*, int, int, int, int, int,
    cudnnHandle_t, std::byte*, std::size_t, cudaStream_t)
{
    return false;  // Not supported with cuDNN < 9.18.0
}

bool moe_cudnn_grouped_gemm_fp4(
    nv_bfloat16*, const nv_bfloat16*, const void*, const float*,
    const int*, int, int, int, int, int,
    cudnnHandle_t, std::byte*, std::size_t, cudaStream_t)
{
    return false;  // Not supported with cuDNN < 9.18.0
}

#endif  // CUDNN_VERSION >= 91800

// =============================================================================
// BF16 MoE GEMM utility functions
// =============================================================================

/// Get workspace size for BF16 MoE grouped GEMM.
std::size_t moe_cudnn_grouped_gemm_workspace_size(
    int num_experts, int total_tokens, int N, int K,
    cudnnHandle_t cudnn_handle)
{
    auto graph = lookup_cache_or_build_moe_gemm_graph(
        num_experts, total_tokens, K, N, cudnn_handle);
    return graph->get_workspace_size();
}

/// Execute BF16 MoE grouped GEMM via cuDNN FE.
void moe_cudnn_grouped_gemm(
    nv_bfloat16* output,
    const nv_bfloat16* input,
    const nv_bfloat16* weights,
    const int* expert_offsets,
    int num_experts,
    int N, int K,
    int total_tokens,
    cudnnHandle_t cudnn_handle,
    std::byte* workspace,
    [[maybe_unused]] std::size_t workspace_size,
    cudaStream_t stream)
{
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    auto graph = lookup_cache_or_build_moe_gemm_graph(
        num_experts, total_tokens, K, N, cudnn_handle);

    std::unordered_map<int64_t, void*> variant_pack = {
        {MOE_TOKEN_UID, (void*)input},
        {MOE_WEIGHT_UID, (void*)weights},
        {MOE_FIRST_TOKEN_OFFSET_UID, (void*)expert_offsets},
        {MOE_OUTPUT_UID, (void*)output},
    };

    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}
