// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//
// Based on llm.c https://github.com/karpathy/llm.c

#include <cudnn_frontend.h>

#include "utilities/utils.h"

namespace fe = cudnn_frontend;

// Forward declaration for workspace size helper (defined later in file)
std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle);

/**
 * @brief Checks a cuDNN status code and exits on error.
 * @param error The cuDNN status code to check.
 * @param file Source file name for error reporting.
 * @param line Source line number for error reporting.
 */
static void cuDNNCheck_(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck_(err, __FILE__, __LINE__))

/**
 * @brief Checks a cuDNN frontend error object and exits on error.
 * @param e The cuDNN frontend error object to check.
 * @param file Source file name for error reporting.
 * @param line Source line number for error reporting.
 */
static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

/**
 * @brief Unique identifiers for cuDNN graph tensors.
 *
 * These UIDs are used to map tensor pointers to their corresponding
 * positions in the cuDNN execution graph via the variant_pack.
 */
enum UIDs {
    Q_UID,          ///< Query tensor
    K_UID,          ///< Key tensor
    V_UID,          ///< Value tensor
    Attn_scale_UID, ///< Attention scale factor (1/sqrt(HS))
    O_UID,          ///< Output tensor
    Stats_UID,      ///< Statistics tensor for backward pass
    dO_UID,         ///< Gradient of output tensor
    dQ_UID,         ///< Gradient of query tensor
    dK_UID,         ///< Gradient of key tensor
    dV_UID          ///< Gradient of value tensor
};

/// @brief Cache type for forward attention graphs, keyed by (B, Hq, Hkv, T, HS, is_inference_only).
/// Graph building is expensive, so we cache built graphs to avoid rebuilding.
using cache_type_fwd = std::map<std::tuple<int,int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

/// @brief Cache type for backward attention graphs, keyed by (B, Hq, Hkv, T, HS).
using cache_type_bwd = std::map<std::tuple<int,int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

/**
 * @brief Retrieves a cached forward attention graph or builds a new one.
 *
 * Constructs a cuDNN frontend SDPA (Scaled Dot-Product Attention) graph for the
 * forward pass with causal masking. The graph is cached in a thread-local map
 * keyed by (B, Hq, Hkv, T, HS, is_inference_only) to avoid expensive rebuilds.
 *
 * Input tensor layout: QKV is (B, T, Hq + 2*Hkv, HS) where Q, K, V are interleaved.
 * Output tensor layout: O is (B, T, Hq, HS).
 *
 * @param B Batch size.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads (supports grouped-query attention when Hkv < Hq).
 * @param T Sequence length.
 * @param HS Head size (hidden dimension per head).
 * @param is_inference_only If true, skips computing stats needed for backward pass.
 * @param cudnn_handle cuDNN handle for graph building.
 * @return Shared pointer to the cached or newly built cuDNN graph.
 */
auto lookup_cache_or_build_graph_fwd(int B, int Hq, int Hkv, int T, int HS, int is_inference_only, cudnnHandle_t cudnn_handle) {

    thread_local cache_type_fwd user_maintained_cache_fwd;

    auto key = std::make_tuple(B, Hq, Hkv, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    // for (B, N, (NH + 2*(NH/replicate_factor)) * HS)
    // (B, T, Hq + 2Hkv, HS)
    int H = Hq + 2 * Hkv;
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, Hq, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({H * HS * T,  HS, H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, Hkv, T, HS})
                               .set_uid(K_UID)
                               .set_stride({H * HS * T, HS, H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, Hkv, T, HS})
                               .set_uid(V_UID)
                               .set_stride({H * HS * T, HS, H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, Hq, HS) BF16/FP16 and stats for backward pass is (B, Hq, T) FP32
    O->set_output(true).set_dim({B, Hq, T, HS}).set_stride({Hq * HS * T, HS, Hq * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, Hq, T, 1})
                               .set_stride({Hq * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

/**
 * @brief Retrieves a cached backward attention graph or builds a new one.
 *
 * Constructs a cuDNN frontend SDPA backward graph for computing gradients
 * with respect to Q, K, and V tensors. Uses causal masking and deterministic
 * algorithm (cuDNN frontend 1.5+). The graph is cached in a thread-local map
 * keyed by (B, Hq, Hkv, T, HS) to avoid expensive rebuilds.
 *
 * @param B Batch size.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads (supports grouped-query attention when Hkv < Hq).
 * @param T Sequence length.
 * @param HS Head size (hidden dimension per head).
 * @param cudnn_handle cuDNN handle for graph building.
 * @return Shared pointer to the cached or newly built cuDNN backward graph.
 */
auto lookup_cache_or_build_graph_bwd(int B, int Hq, int Hkv, int T, int HS, cudnnHandle_t cudnn_handle) {
    thread_local cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, Hq, Hkv, T, HS);

    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    int H = Hq + 2*Hkv;
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, Hkv, T, HS})
                            .set_uid(K_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, Hkv, T, HS})
                            .set_uid(V_UID)
                            .set_stride({H * HS * T, HS, H * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(O_UID)
                            .set_stride({Hq * HS * T, HS, Hq * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, Hq, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({Hq * HS * T, HS, Hq * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, Hq, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({Hq * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    // Deterministic algorithm requires cuDNN 9.18.0+ on Blackwell (SM100+)
    // Check at runtime whether we can enable it
    bool use_deterministic = false;
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        int sm_version = props.major * 10 + props.minor;
        size_t cudnn_version = cudnnGetVersion();
        // Blackwell (SM100+) requires cuDNN 9.18.0 (91800) for deterministic backward
        if (sm_version < 100 || cudnn_version >= 91800) {
            use_deterministic = true;
        }
    }
#endif
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(use_deterministic)
#endif
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, Hq, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, Hkv, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, Hkv, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

/**
 * @brief Executes the forward pass of scaled dot-product attention using cuDNN.
 *
 * Performs FlashAttention-style forward pass with causal masking. Attention scale
 * is computed as 1/sqrt(HS). Supports grouped-query attention when Hkv < Hq.
 *
 * @param[out] out Output tensor of shape (B, T, Hq, HS) in BF16.
 * @param[out] stats Statistics tensor of shape (B, Hq, T) in FP32, required for backward
 *                   pass. Pass nullptr for inference-only mode.
 * @param[in] inp Input QKV tensor of shape (B, T, Hq + 2*Hkv, HS) in BF16. Q, K, V are
 *                stored interleaved along the head dimension.
 * @param workspace Pre-allocated GPU workspace buffer.
 * @param handle cuDNN handle (will be bound to the provided stream).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size (hidden dimension per head).
 * @param stream CUDA stream for asynchronous execution.
 */
void attention_forward_cudnn(nv_bfloat16* out,  // output: (B, T, Hq, HS)
                             float* stats, // output for backward pass: (B, Hq, T)
                             const nv_bfloat16* inp,  // input: (B, T, Hq + Hk + Hv, HS) QKV
                             std::byte* workspace, cudnnHandle_t handle,
                             int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    bool is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, Hq, Hkv, T, HS, is_inference_only, handle);

    // Prepare all the tensor pointers for executing the graph
    const void* devPtrQ = inp;
    const void* devPtrK = (inp + Hq * HS);
    const void* devPtrV = (inp + (Hq + Hkv) * HS);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, (void*)devPtrQ}, {K_UID, (void*)devPtrK}, {V_UID, (void*)devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Executes the backward pass of scaled dot-product attention using cuDNN.
 *
 * Computes gradients dQ, dK, dV for the attention backward pass. Uses the stats
 * tensor computed during forward pass for numerical stability.
 *
 * @param[out] dqkvr Output gradient tensor of shape (B, T, Hq + 2*Hkv, HS) in BF16.
 *                   Contains dQ, dK, dV interleaved along the head dimension.
 * @param[in] stats Statistics tensor from forward pass, shape (B, Hq, T) in FP32.
 * @param[in] dout Upstream gradient of output, shape (B, T, Hq, HS) in BF16.
 * @param[in] qkvr Input QKV tensor from forward pass, shape (B, T, Hq + 2*Hkv, HS) in BF16.
 * @param[in] o Output tensor from forward pass, shape (B, T, Hq, HS) in BF16.
 * @param workspace Pre-allocated GPU workspace buffer.
 * @param handle cuDNN handle (will be bound to the provided stream).
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size (hidden dimension per head).
 * @param stream CUDA stream for asynchronous execution.
 */
void attention_backward_cudnn(nv_bfloat16* dqkvr,                                       // output
                              const float* stats,
                              const nv_bfloat16* o, const nv_bfloat16* dout, const nv_bfloat16* qkvr, // inputs: out, dout, qkv (matches header)
                              std::byte* workspace, cudnnHandle_t handle,
                              int B, int T, int Hq, int Hkv, int HS, cudaStream_t stream) {
    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, Hq, Hkv, T, HS, handle);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = (void*)qkvr;
    void* devPtrK = (void*)(qkvr + Hq * HS);
    void* devPtrV = (void*)(qkvr + (Hq + Hkv) * HS);
    void* devPtrO = (void*)o;
    void* devPtrdO = (void*)dout;
    void* devPtrStats = (void*)stats;
    float attn_scale_cpu = 1.f / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + Hq * HS);
    void* devPtrdV = (dqkvr + (Hq + Hkv) * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(handle, stream));
    checkCudnnFE(graph->execute(handle, variant_pack, workspace));
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Gets the required GPU workspace size for attention operations.
 *
 * Returns the workspace size needed for the backward pass graph, which is
 * typically larger than or equal to the forward pass workspace requirement.
 * The workspace must be pre-allocated before calling attention_forward_cudnn
 * or attention_backward_cudnn.
 *
 * @param B Batch size.
 * @param T Sequence length.
 * @param Hq Number of query heads.
 * @param Hkv Number of key/value heads.
 * @param HS Head size (hidden dimension per head).
 * @param handle cuDNN handle for graph building.
 * @return Required workspace size in bytes.
 */
std::size_t cudnn_get_workspace_size(int B, int T, int Hq, int Hkv, int HS, cudnnHandle_t handle)
{
    auto graph = lookup_cache_or_build_graph_bwd(B, Hq, Hkv, T, HS, handle);
    return graph->get_workspace_size();
}

/**
 * @brief Creates a new cuDNN handle.
 *
 * Allocates and initializes a cuDNN library handle. The handle should be
 * destroyed with destroy_cudnn_handle() when no longer needed.
 *
 * @return Newly created cuDNN handle.
 */
cudnnHandle_t create_cudnn_handle() {
    cudnnHandle_t handle;
    cuDNNCheck(cudnnCreate(&handle));
    return handle;
}

/**
 * @brief Destroys a cuDNN handle and releases associated resources.
 *
 * Safely destroys the cuDNN handle. Does nothing if handle is null.
 * This function is noexcept to allow safe use in destructors.
 *
 * @param handle cuDNN handle to destroy (may be null).
 */
void destroy_cudnn_handle(cudnnHandle_t handle) noexcept {
    if (handle) {
        (void)cudnnDestroy(handle);
    }
}
