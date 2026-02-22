// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/moe/moe_types.h"
#include "recipes/recipe.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

// MoE compact weight information (moved out of anonymous namespace for split files)
MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream,
                                      int layer_idx,
                                      const char* tag) {
    MoeCompactInfo info;
    if (!expert_offsets_dev || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.host_offsets.resize(num_experts + 1, 0);
    CUDA_CHECK(cudaMemcpyAsync(info.host_offsets.data(),
                               expert_offsets_dev,
                               static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (info.host_offsets[e + 1] > info.host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}

MoeCompactInfo build_moe_compact_info_from_host(const int* host_offsets,
                                                int num_experts,
                                                int weight_experts,
                                                int layer_idx,
                                                const char* tag) {
    MoeCompactInfo info;
    if (!host_offsets || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}
bool build_selective_info_from_offsets(const int* host_offsets,
                                       int num_experts,
                                       modules::SelectiveExpertInfo& selection) {
    if (!host_offsets || num_experts <= 0) {
        selection.reset();
        return false;
    }
    selection.reset();
    selection.enabled = true;
    selection.num_total = num_experts;
    selection.expert_to_compact.assign(num_experts, -1);
    selection.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            selection.expert_to_compact[e] = static_cast<int>(selection.active_experts.size());
            selection.active_experts.push_back(e);
        }
    }
    selection.num_active = static_cast<int>(selection.active_experts.size());
    if (selection.num_active == 0) {
        selection.enabled = false;
        return false;
    }
    return true;
}

bool refresh_moe_experts_if_needed(int layer_idx,
                                   const int* host_offsets,
                                   int num_experts,
                                   DslParamStore& weights,
                                   cudaStream_t stream) {
    if (layer_idx < 0) {
        return false;
    }
    auto* provider = weights.qlora_provider();
    if (!provider || !provider->supports_selective_moe()) {
        return false;
    }
    modules::SelectiveExpertInfo selection;
    if (!build_selective_info_from_offsets(host_offsets, num_experts, selection)) {
        return false;
    }
    const bool refreshed = provider->refresh_moe_experts(layer_idx, selection, stream);
    return refreshed;
}

const int* CompiledExecutor::get_or_sync_moe_host_offsets(int layer_idx,
                                                           const int* device_offsets,
                                                           int num_experts) {
    if (layer_idx < 0 || num_experts <= 0 || !device_offsets) {
        return nullptr;
    }
    auto it = mMoEHostOffsetsCache.find(layer_idx);
    if (it != mMoEHostOffsetsCache.end()) {
        return it->second.data();
    }
    // Cache miss: sync from device (at most once per layer per pass)
    auto& cached = mMoEHostOffsetsCache[layer_idx];
    cached.resize(static_cast<std::size_t>(num_experts + 1));
    CUDA_CHECK(cudaMemcpyAsync(cached.data(),
                               device_offsets,
                               static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
    return cached.data();
}

// Global state for QKV gradient tracking (shared across split op files)
std::vector<std::byte*> g_qkv_dA_ptr_by_layer;
std::vector<int> g_qkv_dA_micro_by_layer;

float env_float(const char* name, float fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    float out = std::strtof(value, &end);
    if (end == value) {
        return fallback;
    }
    return out;
}

int env_int(const char* name, int fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    long out = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(out);
}

// ============================================================================
// Operation type conversion
// ============================================================================

const char* op_type_to_string(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::Embedding: return "embedding";
        case CompiledOpType::Zeros: return "zeros";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::LayerNorm: return "layernorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Add: return "add";
        case CompiledOpType::Matmul: return "matmul";
        case CompiledOpType::MatmulBias: return "matmul_bias";
        case CompiledOpType::BiasAdd: return "bias_add";
        case CompiledOpType::SwiGLU: return "swiglu";
        case CompiledOpType::GptOssMoeAct: return "gpt_oss_moe_act";
        case CompiledOpType::Silu: return "silu";
        case CompiledOpType::Gelu: return "gelu";
        case CompiledOpType::Relu2: return "relu2";
        case CompiledOpType::Mul: return "mul";
        case CompiledOpType::MaskScatter: return "mask_scatter";
        case CompiledOpType::DeepstackInject: return "deepstack_inject";
        case CompiledOpType::MatmulSwiGLU: return "matmul_swiglu";
        case CompiledOpType::QKVQKNorm: return "qkv_qk_norm";
        case CompiledOpType::QKVQKNormRoPE: return "qkv_qk_norm_rope";
        case CompiledOpType::MRoPE: return "mrope";
        case CompiledOpType::RoPE: return "rope";
        case CompiledOpType::FlashAttention: return "flash_attention";
        case CompiledOpType::CrossEntropyLoss: return "cross_entropy_loss";
        case CompiledOpType::FusedLMHeadLoss: return "fused_lm_head_loss";
        // MoE forward
        case CompiledOpType::MoESoftmax: return "moe_softmax";
        case CompiledOpType::MoESigmoid: return "moe_sigmoid";
        case CompiledOpType::MoETopK: return "moe_topk";
        case CompiledOpType::MoEPermute: return "moe_permute";
        case CompiledOpType::MoEGroupedGemm: return "moe_grouped_gemm";
        case CompiledOpType::MoEGroupedGemmGateUp: return "moe_grouped_gemm_gate_up";
        case CompiledOpType::MoEGroupedGemmDown: return "moe_grouped_gemm_down";
        case CompiledOpType::MoEUnpermute: return "moe_unpermute";
        case CompiledOpType::MoEExpertBiasAdd: return "moe_expert_bias_add";
        // Expert Parallelism forward
        case CompiledOpType::EpDispatch: return "ep_dispatch";
        case CompiledOpType::EpCombine: return "ep_combine";
        // Backward
        case CompiledOpType::ViewBackward: return "view_backward";
        case CompiledOpType::AddBackward: return "add_backward";
        case CompiledOpType::MatmulBackward: return "matmul_backward";
        case CompiledOpType::BiasAddBackward: return "bias_add_backward";
        case CompiledOpType::SwiGLUBackward: return "swiglu_backward";
        case CompiledOpType::GptOssMoeActBackward: return "gpt_oss_moe_act_backward";
        case CompiledOpType::SiluBackward: return "silu_backward";
        case CompiledOpType::GeluBackward: return "gelu_backward";
        case CompiledOpType::Relu2Backward: return "relu2_backward";
        case CompiledOpType::MulBackward: return "mul_backward";
        case CompiledOpType::MaskScatterBackward: return "mask_scatter_backward";
        case CompiledOpType::DeepstackInjectBackward: return "deepstack_inject_backward";
        case CompiledOpType::MatmulSwiGLUBackward: return "matmul_swiglu_backward";
        case CompiledOpType::QKVQKNormBackward: return "qkv_qk_norm_backward";
        case CompiledOpType::RoPEBackward: return "rope_backward";
        case CompiledOpType::QKVQKNormRoPEBackward: return "qkv_qk_norm_rope_backward";
        case CompiledOpType::MRoPEBackward: return "mrope_backward";
        case CompiledOpType::FlashAttentionBackward: return "flash_attention_backward";
        case CompiledOpType::ZerosBackward: return "zeros_backward";
        case CompiledOpType::FusedResidualRMSNormBackward: return "fused_residual_rmsnorm_backward";
        case CompiledOpType::LayerNormBackward: return "layernorm_backward";
        case CompiledOpType::EmbeddingBackward: return "embedding_backward";
        case CompiledOpType::CrossEntropyLossBackward: return "cross_entropy_backward";
        case CompiledOpType::FusedLMHeadLossBackward: return "fused_lm_head_loss_backward";
        // MoE backward
        case CompiledOpType::MoESoftmaxBackward: return "moe_softmax_backward";
        case CompiledOpType::MoESigmoidBackward: return "moe_sigmoid_backward";
        case CompiledOpType::MoETopKBackward: return "moe_topk_backward";
        case CompiledOpType::MoEPermuteBackward: return "moe_permute_backward";
        case CompiledOpType::MoEGroupedGemmBackward: return "moe_grouped_gemm_backward";
        case CompiledOpType::MoEGroupedGemmGateUpBackward: return "moe_grouped_gemm_gate_up_backward";
        case CompiledOpType::MoEGroupedGemmDownBackward: return "moe_grouped_gemm_down_backward";
        case CompiledOpType::MoEUnpermuteBackward: return "moe_unpermute_backward";
        case CompiledOpType::MoEExpertBiasAddBackward: return "moe_expert_bias_add_backward";
        // Expert Parallelism backward
        case CompiledOpType::EpDispatchBackward: return "ep_dispatch_backward";
        case CompiledOpType::EpCombineBackward: return "ep_combine_backward";
        // Mamba/SSM forward
        case CompiledOpType::MambaSplitProj: return "mamba_split_proj";
        case CompiledOpType::MambaConv1d: return "mamba_conv1d";
        case CompiledOpType::MambaSplitConvOut: return "mamba_split_conv_out";
        case CompiledOpType::MambaSsmScan: return "mamba_ssm_scan";
        case CompiledOpType::MambaGatedRMSNorm: return "mamba_gated_rmsnorm";
        case CompiledOpType::MambaOutProj: return "mamba_out_proj";
        // Mamba/SSM backward
        case CompiledOpType::MambaSplitProjBackward: return "mamba_split_proj_backward";
        case CompiledOpType::MambaConv1dBackward: return "mamba_conv1d_backward";
        case CompiledOpType::MambaSplitConvOutBackward: return "mamba_split_conv_out_backward";
        case CompiledOpType::MambaSsmScanBackward: return "mamba_ssm_scan_backward";
        case CompiledOpType::MambaGatedRMSNormBackward: return "mamba_gated_rmsnorm_backward";
        case CompiledOpType::MambaOutProjBackward: return "mamba_out_proj_backward";
        case CompiledOpType::Unknown: return "unknown";
    }
    return "unknown";
}


// ============================================================================
// CompiledExecutor implementation
// ============================================================================

CompiledExecutor::CompiledExecutor(DslRunState& run_state,
                                   DslParamStore& weights,
                                   DslGradStore& grads,
                                   const modules::ModelConfig& config,
                                   const RuntimeOptions& options)
    : mRunState(run_state)
    , mWeights(weights)
    , mGrads(grads)
    , mConfig(config)
    , mOptions(options)
{}

CompiledExecutor::~CompiledExecutor() {
    // Free persistent GPU buffers
    if (mMoEExpertOffsetsGPU) {
        cudaFree(mMoEExpertOffsetsGPU);
        mMoEExpertOffsetsGPU = nullptr;
        mMoEExpertOffsetsGPUSize = 0;
    }

    // Free persistent MoE saved tensor buffers
    for (auto& [name, buffer] : mMoeSavedBuffers) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    mMoeSavedBuffers.clear();
    mMoeSavedSizes.clear();

    // Free EP per-layer persistent buffers
    for (auto& [layer, state] : mEpStates) {
        if (state.send_order_gpu) cudaFree(state.send_order_gpu);
        if (state.recv_reorder_gpu) cudaFree(state.recv_reorder_gpu);
        if (state.llep_send_reorder_gpu) cudaFree(state.llep_send_reorder_gpu);
        if (state.local_scatter_gpu) cudaFree(state.local_scatter_gpu);
    }
    mEpStates.clear();

    // Free shared EP buffers (shared across layers)
    if (mSharedEpCombinedGpu) cudaFree(mSharedEpCombinedGpu);
    if (mSharedEpSortedRecvGpu) cudaFree(mSharedEpSortedRecvGpu);
    if (mSharedEpLlepCombineGpu) cudaFree(mSharedEpLlepCombineGpu);

    // Free retired shared EP buffers (old allocations kept alive during forward/backward)
    for (auto& e : mEpRetiredBufs) {
        if (e.ptr) cudaFree(e.ptr);
    }
    mEpRetiredBufs.clear();

    // Free EP buffer pool
    for (auto& e : mEpBufPool) {
        if (e.ptr) cudaFree(e.ptr);
    }
    mEpBufPool.clear();

    // Destroy weight transfer stream
    if (mWeightTransferStream) {
        cudaStreamDestroy(mWeightTransferStream);
        mWeightTransferStream = nullptr;
    }
}

void* CompiledExecutor::ep_buf_acquire(size_t need) {
    if (need == 0) return nullptr;
    // Find smallest buffer that fits (best-fit to reduce waste)
    size_t best_size = SIZE_MAX;
    int best_idx = -1;
    for (int i = 0; i < static_cast<int>(mEpBufPool.size()); ++i) {
        if (mEpBufPool[i].bytes >= need && mEpBufPool[i].bytes < best_size) {
            best_size = mEpBufPool[i].bytes;
            best_idx = i;
        }
    }
    if (best_idx >= 0) {
        void* ptr = mEpBufPool[best_idx].ptr;
        mEpBufPool.erase(mEpBufPool.begin() + best_idx);
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, need));
    return ptr;
}

void CompiledExecutor::ep_buf_release(void* ptr, size_t bytes) {
    if (ptr && bytes > 0) {
        mEpBufPool.push_back({ptr, bytes});
    }
}

void CompiledExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                      modules::ModularLoRAWeightsManager* weights,
                                      modules::ModularLoRAGradsManager* grads,
                                      modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
}

void CompiledExecutor::set_weight_manager(DslWeightManager* weight_manager) {
    mWeightManager = weight_manager;
}

void CompiledExecutor::set_recipe(const recipes::Recipe* recipe) {
    mRecipe = recipe;
}

void CompiledExecutor::set_hook_context(void* context) {
    mHookContext = context;
}

void CompiledExecutor::set_recompute_fn(std::function<void(int, long, long, bool)> fn) {
    mRecomputeFn = std::move(fn);
}

void CompiledExecutor::set_recompute_enabled(bool enabled) {
    mRecomputeEnabled = enabled;
    mLastRecomputeLayer = -1;
}

void CompiledExecutor::set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache) {
    mFP8Cache = cache;
}

void CompiledExecutor::set_fp8_cache_transposed(std::unordered_map<std::string, FP8WeightCacheEntry>* cache_t) {
    mFP8CacheT = cache_t;
}

void CompiledExecutor::set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                                     std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t) {
    mFP4Cache = cache;
    mFP4CacheT = cache_t;
}

void CompiledExecutor::set_saved_tensors(std::unordered_map<std::string, Tensor>* saved) {
    mSaved = saved;
}

void CompiledExecutor::set_save_list(const std::vector<std::string>* save_list) {
    mSaveList = save_list;
    mSaveSet.clear();
    if (save_list) {
        mSaveSet.insert(save_list->begin(), save_list->end());
    }
}

void CompiledExecutor::set_last_inputs_cpu(const Tensor* inputs_cpu) {
    mLastInputsCpu = inputs_cpu;
}

void CompiledExecutor::set_rng_seed_fn(std::function<unsigned int()> fn) {
    mRngSeedFn = std::move(fn);
}

const Tensor* CompiledExecutor::try_get_tensor(const std::string& name) const {
    // Fast path: check flat tensor vector using compile-time ID
    if (mCurrentGraph) {
        int tid = mCurrentGraph->find_tensor_id(name);
        if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
            return &mTensors[tid];
        }
    }
    return nullptr;
}

const Tensor* CompiledExecutor::try_get_tensor_fuzzy(const std::string& name) {
    if (const Tensor* direct = try_get_tensor(name)) {
        return direct;
    }
    // Try SSA-suffixed entries via pre-computed ssa_base_to_id (O(1) vs O(N) scan).
    if (mCurrentGraph) {
        auto ssa_it = mCurrentGraph->ssa_base_to_id.find(name);
        if (ssa_it != mCurrentGraph->ssa_base_to_id.end()) {
            int sid = ssa_it->second;
            if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) {
                return &mTensors[sid];
            }
        }
    }

    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return nullptr;
    }
    const std::string base_field = strip_ssa_suffix(field);
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) {
        return nullptr;
    }
    auto& acts = mRunState.simplified_acts(layer_idx);
    if (base_field == "ln1_rstd" || base_field == "ln_rstd") return &acts.ln1_rstd;
    if (base_field == "ln2_rstd") return &acts.ln2_rstd;
    if (base_field == "q_rstd") return &acts.q_rstd;
    if (base_field == "k_rstd") return &acts.k_rstd;
    if (base_field == "lse") return &acts.lse;
    if (base_field == "ln1" || base_field == "ln1_flat" ||
        base_field == "ln" || base_field == "ln_flat") return &acts.ln1;
    if (base_field == "ln2" || base_field == "ln2_flat") return &acts.ln2;
    if (base_field == "qkv" || base_field == "qkv_norm") return &acts.qkv;
    if (base_field == "qkv_rope") {
        Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        return &src;
    }
    if (base_field == "att" || base_field == "att_flat") return &acts.att;
    if (base_field == "att_out" || base_field == "att_out_flat") return &acts.att_out;
    if (base_field == "mlp_up" || base_field == "mlp_up_flat") return &acts.mlp_up;
    if (base_field == "swiglu" || base_field == "swiglu_flat") return &acts.swiglu;
    if (base_field == "mlp_down" || base_field == "mlp_down_flat") return &acts.mlp_down;
    if (base_field == "res_att" || base_field == "residual_att") return &acts.residual_att;
    if (base_field == "res_ffn" || base_field == "residual_ffn" || base_field == "res_in") {
        Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
        return &res;
    }
    return nullptr;
}

void CompiledExecutor::save_moe_layer_tensors(int layer_idx) {
    // Copy MoE tensors from this layer to persistent storage before stack restore.
    // This allows stack memory to be reclaimed while preserving tensors for backward.
    if (mCapturing) {
        return;
    }
    if (mConfig.NumExperts == 0) {
        return;
    }

    if (!mCurrentGraph) return;

    // Build layer prefix pattern (e.g., "blocks[5].")
    std::string layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";

    // Iterate through compile-time tensor name map looking for MoE tensors from this layer
    for (const auto& [name, tid] : mCurrentGraph->tensor_name_to_id) {
        // Skip global MoE tensors - these are scratch space reused each layer
        // and are NOT needed for backward (backward uses mMoEExpertOffsetsGPU).
        if (name == "moe_expert_offsets" || name == "moe_gather_indices") {
            continue;
        }

        // Check if tensor belongs to this layer
        if (name.find(layer_prefix) != 0) {
            continue;
        }

        // Check if this is an MoE-related tensor that needs persistent storage
        bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                              name.find("ep_") != std::string::npos ||
                              name.find("scatter_indices") != std::string::npos ||
                              name.find("routing_weights") != std::string::npos ||
                              name.find("routing_indices") != std::string::npos ||
                              name.find("router_") != std::string::npos ||
                              name.find("permuted") != std::string::npos ||
                              name.find("expert_") != std::string::npos);

        if (!is_moe_tensor) continue;
        if (tid < 0 || static_cast<std::size_t>(tid) >= mTensors.size()) continue;
        auto& tensor = mTensors[static_cast<std::size_t>(tid)];
        if (!tensor.Data) continue;

        const size_t bytes = tensor.bytes();
        if (bytes == 0) {
            continue;
        }

        // Allocate or resize persistent buffer if needed
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            // Free old buffer if exists
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            // Allocate new buffer
            void* new_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
            mMoeSavedBuffers[name] = new_buffer;
            mMoeSavedSizes[name] = bytes;
        }

        // Copy data to persistent buffer
        void* dst_buffer = mMoeSavedBuffers[name];
        CUDA_CHECK(cudaMemcpyAsync(dst_buffer, tensor.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Update tensor to point to persistent buffer (so backward finds it)
        tensor.Data = static_cast<std::byte*>(dst_buffer);
    }
}

void CompiledExecutor::prepare_saved_buffers_for_capture(const std::vector<std::string>& save_list) {
    // Only needed when recompute is enabled or MoE tensors require persistence.
    if (!mSaved) {
        return;
    }

    const bool recompute_enabled = mOptions.recompute_enabled();

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return mSlotRegistry->is_shared(strip_ssa_suffix(field));
        }
        return mSlotRegistry->is_shared(strip_ssa_suffix(name));
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int lid = -1;
        std::string fld;
        if (parse_block_param(name, lid, fld)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(fld))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist) -> bool {
        if (force_persist) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        // Unknown tensors (not in slot registry) are treated as needing persistence.
        return true;
    };

    auto ensure_buffer = [&](const std::string& name, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            void* new_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
            mMoeSavedBuffers[name] = new_buffer;
            mMoeSavedSizes[name] = bytes;
        }
    };

    auto resolve_source = [&](const std::string& name) -> std::optional<Tensor> {
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                return mTensors[tid];
            }
        }
        if (name == "token_ids") {
            return mRunState.Inputs;
        }
        if (name == "position_ids") {
            return mRunState.PositionIDs;
        }

        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
                return std::nullopt;
            }
            const std::string base_field = strip_ssa_suffix(field);
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (base_field == "ln1_rstd" || base_field == "ln_rstd") return acts.ln1_rstd;
            if (base_field == "ln2_rstd") return acts.ln2_rstd;
            if (base_field == "q_rstd") return acts.q_rstd;
            if (base_field == "k_rstd") return acts.k_rstd;
            if (base_field == "lse") return acts.lse;
            if (base_field == "ln1" || base_field == "ln1_flat" ||
                base_field == "ln" || base_field == "ln_flat") return acts.ln1;
            if (base_field == "ln2" || base_field == "ln2_flat") return acts.ln2;
            if (base_field == "qkv" || base_field == "qkv_norm") return acts.qkv;
            if (base_field == "qkv_rope") {
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                return src;
            }
            if (base_field == "qkv_flat") {
                Tensor qkv = acts.qkv;
                return view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
            }
            if (base_field == "att" || base_field == "att_flat") return acts.att;
            if (base_field == "att_out" || base_field == "att_out_flat") return acts.att_out;
            if (base_field == "mlp_up" || base_field == "mlp_up_flat") return acts.mlp_up;
            if (base_field == "swiglu") return acts.swiglu;
            if (base_field == "swiglu_flat") {
                Tensor swiglu = acts.swiglu;
                return view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
            }
            if (base_field == "mlp_down" || base_field == "mlp_down_flat") return acts.mlp_down;
            if (base_field == "res_att" || base_field == "residual_att") return acts.residual_att;
            if (base_field == "res_ffn" || base_field == "residual_ffn" || base_field == "res_in") {
                Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
                return res;
            }
            if (base_field == "router_logits") return acts.router_logits;
            if (base_field == "router_probs") return acts.router_probs;
            if (base_field == "routing_weights") return acts.routing_weights;
            if (base_field == "routing_indices") return acts.routing_indices;
            if (base_field == "permuted_input") return acts.permuted_input;
            if (base_field == "scatter_indices") return acts.scatter_indices;
            if (base_field == "expert_gate_up") return acts.expert_gate_up;
            if (base_field == "expert_act") return acts.expert_act;
            if (base_field == "expert_down") return acts.expert_down;
            if (base_field == "moe_out" || base_field == "moe_out_flat") return acts.moe_out;
        } else if (name == "ln_final" || name == "xF") {
            return mRunState.non_block_activations().ln_final;
        } else if (name == "final_residual" || name == "residual_final") {
            return mRunState.get_final_residual();
        } else if (name == "xF_flat") {
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            return view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
        } else if (name == "ln_final_rstd") {
            return mRunState.non_block_activations().ln_final_rstd;
        } else if (name == "encoded" || name == "x0") {
            return mRunState.non_block_activations().encoded;
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            return mRunState.non_block_activations().freq_cis;
        }

        return std::nullopt;
    };

    auto infer_block_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) {
            return false;
        }
        const std::string base_field = strip_ssa_suffix(field);
        const long B = (mB > 0) ? mB : mRunState.B;
        const long T = (mT > 0) ? mT : mRunState.T;
        if (B <= 0 || T <= 0) {
            return false;
        }
        const long C = mConfig.HiddenSize;
        const long D = mConfig.IntermediateSize;
        const long Hq = mConfig.NumQueryHeads;
        const long Hkv = mConfig.NumKeyValHeads;
        const long Hs = mConfig.head_size();
        const long QKV = Hs * (Hq + 2 * Hkv);
        const long AttnDim = Hq * Hs;
        const long MUp = mConfig.mlp_up_rows();

        std::vector<long> shape;
        if (base_field == "qkv_flat" || base_field == "qkv_biased") {
            shape = {B * T, QKV};
        } else if (base_field == "ln1_flat" || base_field == "ln2_flat" ||
                   base_field == "ln_flat") {
            shape = {B * T, C};
        } else if (base_field == "att_out_flat") {
            shape = {B * T, C};
        } else if (base_field == "att_flat") {
            shape = {B * T, AttnDim};
        } else if (base_field == "mlp_up_flat") {
            shape = {B * T, MUp};
        } else if (base_field == "mlp_down_flat") {
            shape = {B * T, C};
        } else if (base_field == "swiglu_flat") {
            shape = {B * T, D};
        } else if (base_field == "ln1" || base_field == "ln2" || base_field == "ln" ||
                   base_field == "res_att" || base_field == "residual_att" ||
                   base_field == "res_ffn" || base_field == "residual_ffn" ||
                   base_field == "res_in" || base_field == "att_out" ||
                   base_field == "mlp_down") {
            shape = {B, T, C};
        } else if (base_field == "ln1_rstd" || base_field == "ln2_rstd" ||
                   base_field == "ln_rstd") {
            shape = {B, T};
        } else if (base_field == "mlp_up") {
            shape = {B, T, MUp};
        } else if (base_field == "swiglu") {
            shape = {B, T, D};
        } else if (base_field == "qkv" || base_field == "qkv_rope" || base_field == "qkv_norm") {
            shape = {B, T, QKV};
        } else if (base_field == "att") {
            shape = {B, T, AttnDim};
        } else if (base_field == "q_rstd") {
            shape = {B, T, Hq};
        } else if (base_field == "k_rstd") {
            shape = {B, T, Hkv};
        } else if (base_field == "lse") {
            shape = {B, Hq, T};
        } else {
            // Fallback: use conservative upper bound for any unrecognized block tensor.
            // This ensures persist_saved_layer_tensors won't cudaMalloc during capture.
            const long max_dim = std::max({C, QKV, MUp, D});
            shape = {B, T, max_dim};
        }

        ETensorDType dtype = ETensorDType::BF16;
        if (mConfig.NumLayers > 0) {
            dtype = mRunState.simplified_acts(0).ln1.DType;
        }
        if (base_field == "ln1_rstd" || base_field == "ln2_rstd" || base_field == "ln_rstd" ||
            base_field == "q_rstd" || base_field == "k_rstd" || base_field == "lse") {
            dtype = ETensorDType::FP32;
        }
        const std::size_t nelem = shape_nelem(shape);
        out_bytes = nelem * static_cast<std::size_t>(get_dtype_size(dtype));
        return out_bytes > 0;
    };

    for (const auto& name : save_list) {
        if (mWeights.has(name)) {
            continue;
        }

        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" ||
             name == "ln_final_rstd" || name == "residual_final" || name == "final_residual");
        const bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                                    name.find("scatter_indices") != std::string::npos ||
                                    name.find("routing_weights") != std::string::npos ||
                                    name.find("routing_indices") != std::string::npos ||
                                    name.find("router_probs") != std::string::npos ||
                                    name.find("router_logits") != std::string::npos ||
                                    name.find("permuted_input") != std::string::npos ||
                                    name.find("expert_") != std::string::npos);
        const bool prefer_live = prefer_live_tensor(name);
        const bool force_persist = is_moe_tensor && mConfig.NumExperts > 0;
        const bool need_persist = should_persist(name, prefer_live, force_persist || force_persist_name);

        // Block-level tensors may be stack-backed and persisted by
        // persist_saved_layer_tensors() during execute_forward.  Pre-allocate
        // buffers for those too, since cudaMalloc is not allowed during CUDA
        // graph capture.  However, skip tensors that will be recomputed in
        // backward — they only need metadata-only saves, not persistent buffers.
        int layer_idx = -1;
        std::string field;
        const bool is_block_tensor = parse_block_param(name, layer_idx, field);
        if (!need_persist && !(is_block_tensor && !prefer_live)) {
            continue;
        }

        // For tensors that need_persist from save_tensors, always pre-allocate.
        // For block tensors that don't need_persist but might be stack-backed:
        // only pre-allocate if the resolved source has null Data (indicating it
        // will be stack-allocated at runtime). Per-layer allocator buffers
        // (non-null Data) don't need persistent copies since they survive the
        // entire training step.
        auto src_opt = resolve_source(name);
        if (!need_persist && src_opt.has_value() && src_opt->Data) {
            // Per-layer allocator buffer — persist_saved_layer_tensors will
            // skip it (Stack.owns returns false), so no pre-allocation needed.
            continue;
        }
        if (src_opt.has_value() && src_opt->Data) {
            ensure_buffer(name, src_opt->bytes());
            continue;
        }
        std::size_t inferred_bytes = 0;
        if (infer_block_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        // Infer sizes for non-block tensors (xF_flat, ln_final, etc.)
        {
            const long B = (mB > 0) ? mB : mRunState.B;
            const long T = (mT > 0) ? mT : mRunState.T;
            const long C = mConfig.HiddenSize;
            if (B > 0 && T > 0 && C > 0) {
                const std::size_t elem_size = static_cast<std::size_t>(get_dtype_size(ETensorDType::BF16));
                std::size_t nbytes = 0;
                if (name == "xF_flat") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "ln_final" || name == "xF") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "ln_final_rstd") {
                    nbytes = static_cast<std::size_t>(B * T) * sizeof(float);
                } else if (name == "encoded" || name == "x0") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "final_residual" || name == "residual_final") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                }
                if (nbytes > 0) {
                    ensure_buffer(name, nbytes);
                }
            }
        }
    }
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list, bool force_persist) {
    if (!mSaved) {
        return;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
         capture_status != cudaStreamCaptureStatusNone);
    const bool capturing = mCapturing || in_capture;

    // Recompute is only active when enabled in runtime options.
    // Do NOT use mRecomputeFn here: it's always set when the graph is compiled,
    // even for no-recompute runs, and would cause metadata-only saves.
    const bool recompute_enabled = mOptions.recompute_enabled();

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (force_persist) {
            return false;
        }
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Use will_recompute which checks the recompute_policy
        // In FFT mode (!lora_only), tensors with lora_only policy should NOT prefer live
        // because they won't be recomputed and the live buffer may have stale data
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        const std::string base_name = strip_ssa_suffix(tensor_name);
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // Helper to copy tensor to persistent buffer when needed in recompute mode.
    // Returns true if tensor was copied to persistent storage, false if metadata-only save.
    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return mSlotRegistry->is_shared(strip_ssa_suffix(field));
        }
        return mSlotRegistry->is_shared(strip_ssa_suffix(name));
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist_name) -> bool {
        if (force_persist || force_persist_name) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            // Slot resolves to Mapped: no persistent buffer, so saved data must be copied.
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        return true;
    };

    auto save_tensor_with_policy = [&](const std::string& name, const Tensor& src,
                                       bool prefer_live, bool force_persist_name) -> void {
        if (force_persist) {
            prefer_live = false;
            force_persist_name = true;
        }
        if (prefer_live) {
            // Save metadata only - will resolve from live buffer or recompute
            Tensor meta = src;
            meta.Data = nullptr;
            (*mSaved)[name] = meta;
            return;
        }

        const bool need_persist = should_persist(name, prefer_live, force_persist_name) && src.Data != nullptr;
        if (need_persist && src.Data != nullptr) {
            const size_t bytes = src.bytes();
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (capturing) {
                    throw std::runtime_error("CompiledExecutor: missing preallocated save buffer for '" + name +
                                             "' during CUDA graph capture");
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoeSavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor;
            saved_tensor.DType = src.DType;
            saved_tensor.Rank = src.Rank;
            for (int i = 0; i < src.Rank; ++i) {
                saved_tensor.Sizes[i] = src.Sizes[i];
            }
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            return;
        }

        // Non-recompute mode: just store reference
        (*mSaved)[name] = src;
    };

    for (const auto& name : save_list) {
        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" ||
             name == "ln_final_rstd" || name == "residual_final" || name == "final_residual");
        // Skip tensors already saved directly by dispatch (e.g. mamba_gated_rmsnorm saves rstd/normed),
        // unless we are forcing a persistent copy to replace metadata-only entries.
        auto saved_it = mSaved->find(name);
        if (saved_it != mSaved->end()) {
            if (!force_persist || saved_it->second.Data != nullptr) {
                continue;
            }
            // Allow refresh when force_persist=true and existing entry is metadata-only.
            mSaved->erase(saved_it);
        }

        // First check intermediate tensors via flat vector (fast path) or mirror map
        Tensor* found_tensor = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                found_tensor = &mTensors[tid];
            }
        }
        if (found_tensor) {
            const bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                                        name.find("ep_") != std::string::npos ||
                                        name.find("scatter_indices") != std::string::npos ||
                                        name.find("routing_weights") != std::string::npos ||
                                        name.find("routing_indices") != std::string::npos ||
                                        name.find("router_probs") != std::string::npos ||
                                        name.find("router_logits") != std::string::npos ||
                                        name.find("permuted_input") != std::string::npos ||
                                        name.find("expert_") != std::string::npos);
            const bool prefer_live = prefer_live_tensor(name);
            const bool force_persist = is_moe_tensor && mConfig.NumExperts > 0;
            save_tensor_with_policy(name, *found_tensor, prefer_live, force_persist || force_persist_name);
            continue;
        }

        // Check special tensors
        if (name == "token_ids") {
            save_tensor_with_policy(name, mRunState.Inputs, prefer_live_tensor(name), force_persist_name);
            continue;
        }
        if (name == "position_ids") {
            save_tensor_with_policy(name, mRunState.PositionIDs, prefer_live_tensor(name), force_persist_name);
            continue;
        }

        // Try to look up as a pre-allocated activation by creating a TensorRef
        // This handles tensors like "blocks[0].ln1_rstd" that map to slots
        TensorRef ref;
        ref.name = name;
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            // Map common saved fields
            const bool prefer_live = prefer_live_tensor(name);
            if (field == "ln1_rstd" || field == "ln_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1_rstd, prefer_live, force_persist_name);
            } else if (field == "ln2_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2_rstd, prefer_live, force_persist_name);
            } else if (field == "q_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).q_rstd, prefer_live, force_persist_name);
            } else if (field == "k_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).k_rstd, prefer_live, force_persist_name);
            } else if (field == "lse") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).lse, prefer_live, force_persist_name);
            } else if (field == "ln1" || field == "ln1_flat" || field == "ln" || field == "ln_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1, prefer_live, force_persist_name);
            } else if (field == "ln2" || field == "ln2_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2, prefer_live, force_persist_name);
            } else if (field == "qkv" || field == "qkv_norm") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).qkv, prefer_live, force_persist_name);
            } else if (field == "qkv_rope") {
                // qkv_rope has RoPE applied - save it if available, otherwise fall back to qkv
                auto& acts = mRunState.simplified_acts(layer_idx);
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                save_tensor_with_policy(name, src, prefer_live, force_persist_name);
            } else if (field == "qkv_flat") {
                // Save the flattened version for matmul backward shape resolution
                Tensor qkv = mRunState.simplified_acts(layer_idx).qkv;
                Tensor flat = view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
            } else if (field == "att" || field == "att_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att, prefer_live, force_persist_name);
            } else if (field == "att_out" || field == "att_out_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att_out, prefer_live, force_persist_name);
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_up, prefer_live, force_persist_name);
            } else if (field == "swiglu") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).swiglu, prefer_live, force_persist_name);
            } else if (field == "swiglu_flat") {
                Tensor swiglu = mRunState.simplified_acts(layer_idx).swiglu;
                Tensor flat = view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_down, prefer_live, force_persist_name);
            } else if (field == "res_att" || field == "residual_att") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).residual_att, prefer_live, force_persist_name);
            } else if (field == "res_ffn" || field == "residual_ffn" || field == "res_in") {
                // res_ffn is the residual stream after FFN block (residual_att + mlp_down)
                Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
                save_tensor_with_policy(name, res, prefer_live, force_persist_name);
            } else if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
            } else {
                throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
            }
        } else if (name == "ln_final" || name == "xF") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final, prefer_live_tensor(name), force_persist_name);
        } else if (name == "final_residual" || name == "residual_final") {
            save_tensor_with_policy(name, mRunState.get_final_residual(), prefer_live_tensor(name), force_persist_name);
        } else if (name == "xF_flat") {
            // Save the flattened version for matmul backward
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            Tensor flat = view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
            save_tensor_with_policy(name, flat, prefer_live_tensor(name), force_persist_name);
        } else if (name == "ln_final_rstd") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final_rstd, prefer_live_tensor(name), force_persist_name);
        } else if (name == "encoded" || name == "x0") {
            save_tensor_with_policy(name, mRunState.non_block_activations().encoded, prefer_live_tensor(name), force_persist_name);
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            save_tensor_with_policy(name, mRunState.non_block_activations().freq_cis, prefer_live_tensor(name), force_persist_name);
        } else if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
        } else {
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }
    }

    // For MoE models, copy expert_offsets data to persistent storage for backward pass
    // The original tensor is stack-allocated and will be freed before backward runs
    if (mConfig.NumExperts > 0) {
        Tensor* moe_offsets = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id("moe_expert_offsets");
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                moe_offsets = &mTensors[tid];
            }
        }
        if (moe_offsets) {
            const Tensor& src = *moe_offsets;
            const int num_elements = static_cast<int>(src.nelem());
            mMoEExpertOffsetsData.resize(num_elements);
            CUDA_CHECK(cudaMemcpy(mMoEExpertOffsetsData.data(), src.Data,
                                  num_elements * sizeof(int), cudaMemcpyDeviceToHost));
            // Store metadata for reconstruction in backward
            mMoEExpertOffsets = src;  // Copy the tensor metadata (shape, dtype, etc.)
            mMoEExpertOffsets.Data = nullptr;  // Data will be restored from CPU storage
        }
    }
}

Tensor* CompiledExecutor::try_resolve_saved_live(const std::string& name, const Tensor& saved) {
    std::vector<long> shape;
    shape.reserve(static_cast<std::size_t>(saved.Rank));
    for (int i = 0; i < saved.Rank; ++i) {
        shape.push_back(saved.Sizes[i]);
    }

    auto map_view = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            return nullptr;
        }
        if (shape.empty() || tensor_shape_matches(base, shape)) {
            return &base;
        }
        if (shape_nelem(shape) != base.nelem()) {
            return nullptr;
        }
        Tensor view = view_tensor(base, shape);
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                mTensors[tid] = view;
                return &mTensors[tid];
            }
        }
        return &base;  // Can't cache view without tensor_id, return base
    };

    if (name == "token_ids") {
        return map_view(mRunState.Inputs);
    }
    if (name == "position_ids") {
        return map_view(mRunState.PositionIDs);
    }
    if (name == "encoded" || name == "x0") {
        return map_view(mRunState.non_block_activations().encoded);
    }
    if (name == "ln_final" || name == "xF" || name == "xF_flat") {
        return map_view(mRunState.non_block_activations().ln_final);
    }
    if (name == "ln_final_rstd") {
        return map_view(mRunState.non_block_activations().ln_final_rstd);
    }
    if (name == "final_residual" || name == "residual_final") {
        return map_view(mRunState.get_final_residual());
    }
    if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        return map_view(mRunState.non_block_activations().freq_cis);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return nullptr;
        }
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (field == "ln1" || field == "ln1_flat") return map_view(acts.ln1);
        if (field == "ln1_rstd") return map_view(acts.ln1_rstd);
        if (field == "ln2" || field == "ln2_flat") return map_view(acts.ln2);
        if (field == "ln2_rstd") return map_view(acts.ln2_rstd);
        if (field == "q_rstd") return map_view(acts.q_rstd);
        if (field == "k_rstd") return map_view(acts.k_rstd);
        if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased" || field == "qkv_norm") return map_view(acts.qkv);
        if (field == "qkv_rope") {
            Tensor* base = acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
            return map_view(*base);
        }
        if (field == "lse") return map_view(acts.lse);
        if (field == "att" || field == "att_flat") return map_view(acts.att);
        if (field == "att_out" || field == "att_out_flat") return map_view(acts.att_out);
        if (field == "res_att" || field == "residual_att") return map_view(acts.residual_att);
        if (field == "mlp_up" || field == "mlp_up_flat") return map_view(acts.mlp_up);
        if (field == "swiglu" || field == "swiglu_flat") return map_view(acts.swiglu);
        if (field == "mlp_down" || field == "mlp_down_flat") return map_view(acts.mlp_down);
        if (field == "router_logits") return map_view(acts.router_logits);
        if (field == "router_probs") return map_view(acts.router_probs);
        if (field == "routing_weights") return map_view(acts.routing_weights);
        if (field == "routing_indices") return map_view(acts.routing_indices);
        if (field == "permuted_input") return map_view(acts.permuted_input);
        if (field == "scatter_indices") return map_view(acts.scatter_indices);
        if (field == "expert_gate_up") return map_view(acts.expert_gate_up);
        if (field == "expert_act") return map_view(acts.expert_act);
        if (field == "expert_down") return map_view(acts.expert_down);
        if (field == "moe_out" || field == "moe_out_flat") return map_view(acts.moe_out);
        if (field == "res_ffn" || field == "residual_ffn" || field == "res_in") {
            Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
            return map_view(res);
        }
        if (field == "rope_freqs" || field == "freq_cis") {
            return map_view(mRunState.non_block_activations().freq_cis);
        }
    }

    return nullptr;
}

Tensor& CompiledExecutor::resolve_tensor(const TensorRef& ref) {
    auto& rs = mRunState;
    const int tid = ref.tensor_id;
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    auto log_tensor = [&](const Tensor& t, const char* tag) {
        if (debug_name) {
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] resolve_tensor %s %s dtype=%s\n",
                    ref.name.c_str(), tag, dtype_to_str(t.DType));
        }
    };

    // Only check gradient resolution for tensors flagged as gradients at compile time.
    // This avoids calling base_param_from_grad() (string substr + find) for the ~80%
    // of tensors that are activations/weights, not gradients.
    if (ref.is_gradient && !ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = ref.shape.empty() ? *grad : view_tensor(*grad, ref.shape);
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Check if we already have a tensor cached (e.g., from MoE temp allocation)
        if (tid >= 0) {
            auto& cached = mTensors[static_cast<std::size_t>(tid)];
            if (cached.Data) {
                return cached;
            }
        }
        // Need to create a view - get the base tensor and create view
        Tensor* base = nullptr;
        switch (ref.slot) {
            case TensorSlot::TokenIDs: base = &rs.Inputs; break;
            case TensorSlot::PositionIDs: base = &rs.PositionIDs; break;
            case TensorSlot::Targets: base = &rs.Targets; break;
            case TensorSlot::Losses: base = &rs.Losses; break;
            case TensorSlot::DLoss: base = &rs.scratch().cross_entropy_dloss; break;
            case TensorSlot::BlockDLN1: base = &rs.simplified_grads(ref.layer_idx).d_ln1; break;
            case TensorSlot::BlockDQKV: base = &rs.simplified_grads(ref.layer_idx).d_qkv; break;
            case TensorSlot::BlockDAtt: base = &rs.simplified_grads(ref.layer_idx).d_att; break;
            case TensorSlot::BlockDSwiGLU: base = &rs.simplified_grads(ref.layer_idx).d_swiglu; break;
            case TensorSlot::BlockDMLPUp: base = &rs.simplified_grads(ref.layer_idx).d_mlp_up; break;
            case TensorSlot::BlockDMLPDown: base = &rs.simplified_grads(ref.layer_idx).d_mlp_down; break;
            case TensorSlot::BlockDLN2: base = &rs.simplified_grads(ref.layer_idx).d_ln2; break;
            case TensorSlot::BlockDResAtt: base = &rs.simplified_grads(ref.layer_idx).d_res_att; break;
            case TensorSlot::BlockDAttOut: base = &rs.simplified_grads(ref.layer_idx).d_att_out; break;
            case TensorSlot::BlockDResFFN: base = &rs.simplified_grads(ref.layer_idx).d_res_ffn; break;
            case TensorSlot::BlockLN1: base = &rs.simplified_acts(ref.layer_idx).ln1; break;
            case TensorSlot::BlockLN2: base = &rs.simplified_acts(ref.layer_idx).ln2; break;
            case TensorSlot::BlockQKV: base = &rs.simplified_acts(ref.layer_idx).qkv; break;
            case TensorSlot::BlockAtt: base = &rs.simplified_acts(ref.layer_idx).att; break;
            case TensorSlot::BlockAttOut: base = &rs.simplified_acts(ref.layer_idx).att_out; break;
            case TensorSlot::BlockMLPUp: base = &rs.simplified_acts(ref.layer_idx).mlp_up; break;
            case TensorSlot::BlockSwiGLU: base = &rs.simplified_acts(ref.layer_idx).swiglu; break;
            case TensorSlot::BlockMLPDown: base = &rs.simplified_acts(ref.layer_idx).mlp_down; break;
            default: break;
        }
        if (base && base->Data) {
            Tensor view = view_tensor(*base, ref.shape);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return *base;  // tid < 0 should not happen, return base as fallback
        }
    }

    // Check flat tensor vector first for cached/aliased tensors (e.g., view_backward aliases).
    // This is critical because view_backward stores aliases, and subsequent ops
    // (like rmsnorm_backward) must use that aliased tensor, not the pre-allocated simplified_grads buffer.
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        log_tensor(mTensors[static_cast<std::size_t>(tid)], "cached");
        return mTensors[static_cast<std::size_t>(tid)];
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs:
            return rs.Inputs;
        case TensorSlot::PositionIDs:
            return rs.PositionIDs;
        case TensorSlot::Targets:
            return rs.Targets;
        case TensorSlot::Losses:
            return rs.Losses;
        case TensorSlot::DLoss:
            return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded:
            return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal:
            return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD:
            return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual:
            return rs.get_final_residual();
        case TensorSlot::FreqCis:
            return rs.non_block_activations().freq_cis;
        case TensorSlot::BlockLN1:
            return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD:
            return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2:
            return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD:
            return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD:
            return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD:
            return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV:
            return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE:
            return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt:
            return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut:
            return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt:
            return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp:
            return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU:
            return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown:
            return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockResidualFFN:
            return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockDLN1:
            return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV:
            return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt:
            return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU:
            return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp:
            return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown:
            return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDLN2:
            return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt:
            return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDAttOut:
            return rs.simplified_grads(ref.layer_idx).d_att_out;
        case TensorSlot::BlockDResFFN:
            return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter:
            return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    // If the saved tensor has actual data, use it directly.
                    // Only resolve from live buffers when Data == nullptr (metadata-only mode).
                    if (it->second.Data != nullptr) {
                        return it->second;
                    }
                    // Metadata-only: try to resolve from live buffer via flat vector
                    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            // Already checked mTensors[tid] above; if we get here, tensor was not found
            if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
                return mTensors[static_cast<std::size_t>(tid)];
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary:
            throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    throw std::runtime_error("CompiledExecutor: invalid tensor slot");
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    const int tid = ref.tensor_id;
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    if (debug_name) {
        fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] ensure_output_tensor enter %s slot=%d ref=%s\n",
                ref.name.c_str(), static_cast<int>(ref.slot), dtype_to_str(ref.dtype));
    }

    // Fast path: pre-allocated block slots with existing data bypass string parsing.
    // This covers most activation and gradient outputs during forward/backward.
    // Only Mapped/Temporary/Parameter/Saved slots need the string-heavy resolution below.
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Saved) {
        Tensor& t = resolve_tensor(ref);
        if (t.Data) {
            if (!ref.shape.empty()) {
                Tensor view = view_tensor(t, ref.shape);
                if (tid >= 0) {
                    mTensors[static_cast<std::size_t>(tid)] = view;
                    return mTensors[static_cast<std::size_t>(tid)];
                }
                return t;
            }
            return t;
        }
        // No data — fall through to slow path for alias resolution or temp allocation
    }

    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = ref.shape.empty() ? *grad : view_tensor(*grad, ref.shape);
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // DSL-driven aliasing: allow gradients to reuse existing activation buffers.
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout() && !ref.name.empty()) {
        std::string base_name = strip_ssa_suffix(ref.name);
        const bool is_grad_name = starts_with(base_name, "d_");
        std::string parse_name = is_grad_name ? base_name.substr(2) : base_name;
        int layer_idx = -1;
        std::string field;
        std::string lookup_name = base_name;
        if (parse_block_param(parse_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            lookup_name = is_grad_name ? ("d_" + base_field) : base_field;
        }
        if (auto slot_entry = mSlotRegistry->lookup(lookup_name)) {
            if (!slot_entry->alias_of.empty()) {
                const std::string alias_field = slot_entry->alias_of;
                std::string alias_name = alias_field;
                if (layer_idx >= 0) {
                    alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                }
                if (auto alias_entry = mSlotRegistry->lookup(alias_field)) {
                    TensorRef alias_ref;
                    alias_ref.name = alias_name;
                    alias_ref.layer_idx = layer_idx;
                    alias_ref.slot = alias_entry->slot;
                    alias_ref.shape = ref.shape;
                    alias_ref.dtype = ref.dtype;
                    // Resolve alias tensor_id from current graph
                    if (mCurrentGraph) {
                        alias_ref.tensor_id = mCurrentGraph->find_tensor_id(alias_name);
                    }
                    if (mSaveSet.find(alias_name) != mSaveSet.end()) {
                        alias_ref.slot = TensorSlot::Saved;
                    }
                    try {
                        Tensor& base = resolve_tensor(alias_ref);
                        Tensor view = ref.shape.empty() ? base : view_tensor(base, ref.shape);
                        if (tid >= 0) {
                            mTensors[static_cast<std::size_t>(tid)] = view;
                            return mTensors[static_cast<std::size_t>(tid)];
                        }
                        return base;
                    } catch (const std::exception&) {
                        // Fall through to normal allocation if alias resolution fails.
                    }
                }
            }
        }
    }

    // For pre-allocated slots, just return the tensor
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary) {
        Tensor& t = resolve_tensor(ref);
        if (!t.Data) {
            mRunState.temp_acquire(t);
            mTemps.push_back(t);
        }
        if (!ref.shape.empty()) {
            Tensor view = view_tensor(t, ref.shape);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return t;  // tid < 0 should not happen for pre-allocated slots
        }
        return t;
    }

    // For mapped/temporary tensors, check flat vector first
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        if (debug_name) {
            const auto& t = mTensors[static_cast<std::size_t>(tid)];
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] reuse tid=%d dtype=%s\n",
                    tid, dtype_to_str(t.DType));
        }
        return mTensors[static_cast<std::size_t>(tid)];
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape);

    // Zero gradient tensors to prevent stale values from accumulating.
    if (ref.is_gradient) {
        fill_zero(t, mRunState.MainStream);
    }

    mTemps.push_back(t);
    if (tid >= 0) {
        mTensors[static_cast<std::size_t>(tid)] = t;
        if (debug_name) {
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] alloc tid=%d dtype=%s\n",
                    tid, dtype_to_str(t.DType));
        }
        return mTensors[static_cast<std::size_t>(tid)];
    }
    throw std::runtime_error("CompiledExecutor: ensure_output_tensor requires valid tensor_id for: " + ref.name);
}

void CompiledExecutor::handle_layer_start(int layer_idx) {
    if (mWeightManager && mWeightManager->needs_block_gather() && !mCapturing) {
        mWeightManager->wait_for_gather(layer_idx, mRunState.MainStream);
    }

    // Prefetch next layer in the current traversal direction
    const int next_layer = layer_idx + mPrefetchDirection;
    if (next_layer >= 0 && next_layer < static_cast<int>(mConfig.NumLayers) && !mCapturing) {
        if (mWeightManager && mWeightManager->needs_block_gather()) {
            if (mComm) {
                mWeightManager->gather_block(next_layer, *mComm, mRunState.side_stream());
            }
        }
        // QLoRA offload: prefetch quantized weights for the next layer
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                provider->prefetch_for_layer(next_layer, mRunState.side_stream());
            }
        }
    }

    mCurrentLayer = layer_idx;
}

void CompiledExecutor::handle_layer_end(int layer_idx) {
    // Release previous layer's weights
    if (mWeightManager && mWeightManager->needs_block_gather() && !mCapturing) {
        mWeightManager->release_block(layer_idx, mRunState.MainStream);
    }

    // Offload residual if enabled
    if (mRunState.has_residual_offloading() && !mCapturing) {
        mRunState.mark_residual_ready(layer_idx, mRunState.MainStream);
        mRunState.put_residual(layer_idx, mRunState.side_stream());
    }
}


void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    mComm = &comm;
    mCurrentGraph = &graph;
    mTemps.clear();
    mMoEHostOffsetsCache.clear();
    // cudaFree and cudaMemPoolTrimTo are prohibited during CUDA stream capture —
    // they invalidate the capture. Skip all cleanup when capturing; it will run
    // on the next eager (non-captured) step instead.
    // Check both the inner capture flag and the actual stream status (for outer
    // whole-step captures like train_step_graphed that don't set mCapturing).
    cudaStreamCaptureStatus cleanup_capture_status = cudaStreamCaptureStatusNone;
    const bool cleanup_capturing =
        mCapturing ||
        (cudaStreamIsCapturing(mRunState.MainStream, &cleanup_capture_status) == cudaSuccess &&
         cleanup_capture_status != cudaStreamCaptureStatusNone);
    if (!cleanup_capturing) {
        // Free retired shared EP buffers from previous steps (accumulated during reallocation).
        // Previous step is fully complete, so these are no longer referenced.
        for (auto& e : mEpRetiredBufs) {
            if (e.ptr) cudaFree(e.ptr);
        }
        mEpRetiredBufs.clear();
        // Free EP buffer pool — temporary buffers with short lifetimes (acquired/released
        // within a single dispatch call). As routing imbalance changes during training,
        // buffer sizes drift and stale entries become unreusable zombies. Clearing per-step
        // prevents this accumulation; cudaMalloc overhead is negligible vs A2A/GEMM costs.
        for (auto& e : mEpBufPool) {
            if (e.ptr) cudaFree(e.ptr);
        }
        mEpBufPool.clear();
        // Trim CUDA stream-ordered memory pool to release cached allocations.
        // cuBLAS cublasGemmGroupedBatchedEx internally uses cudaMallocAsync;
        // trimming reclaims unused cached blocks from previous steps.
        {
            int device;
            cudaGetDevice(&device);
            cudaMemPool_t pool;
            if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                cudaMemPoolTrimTo(pool, 0);
            }
        }
    }
    // Initialize flat tensor vector indexed by compile-time tensor IDs
    mTensors.assign(static_cast<std::size_t>(graph.num_tensors), Tensor{});
    mCurrentLayer = -1;

    // Match GraphExecutor behavior: initialize loss/counter buffers for full forward runs.
    // This avoids stale accumulation when tests call CompiledExecutor directly.
    if (full) {
        bool has_loss_op = false;
        for (const auto& op : graph.ops) {
            if (op.type == CompiledOpType::CrossEntropyLoss ||
                op.type == CompiledOpType::FusedLMHeadLoss) {
                has_loss_op = true;
                break;
            }
        }
        if (has_loss_op) {
            fill_zero(mRunState.Losses, mRunState.MainStream);
            fill_zero(mRunState.ValidTokenCount, mRunState.MainStream);
            fill_zero(mRunState.CorrectCount, mRunState.MainStream);
        }
    }
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    // Reuse member vectors to avoid per-forward heap allocations.
    auto& layer_checkpoints = mLayerCheckpoints;
    auto& layer_temp_marks = mLayerTempMarks;
    auto& layer_active = mLayerActive;
    if (num_layers > 0) {
        layer_checkpoints.resize(static_cast<std::size_t>(num_layers));
        layer_temp_marks.resize(static_cast<std::size_t>(num_layers));
        layer_active.assign(static_cast<std::size_t>(num_layers), 0);
    }

    // Build save mask for fast per-ID lookup during pruning
    mSaveMask.assign(static_cast<std::size_t>(graph.num_tensors), false);
    if (mSaveList) {
        for (const auto& name : *mSaveList) {
            int sid = graph.find_tensor_id(name);
            if (sid >= 0) {
                mSaveMask[static_cast<std::size_t>(sid)] = true;
            }
        }
    }

    auto prune_stack_tensors = [&]() {
        // Prune flat tensor vector using pre-computed metadata (no string parsing)
        for (int id = 0; id < graph.num_tensors; ++id) {
            auto& t = mTensors[static_cast<std::size_t>(id)];
            if (!t.Data) continue;
            // Skip tensors needed for backward (in save list)
            if (mSaveMask[static_cast<std::size_t>(id)]) continue;
            // Skip cross-layer connector tensors (layerN.out, layerN.res_in, etc.)
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
            if (meta.is_cross_layer()) continue;
            if (mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                t = Tensor{};
            }
        }
    };
    // Detect if the stream is being captured (either by internal graphs via mCapturing,
    // or by an outer full-step graph from train_step_graphed in py_train.cpp).
    cudaStreamCaptureStatus fwd_capture_status = cudaStreamCaptureStatusNone;
    const bool fwd_stream_capturing =
        mCapturing ||
        (cudaStreamIsCapturing(mRunState.MainStream, &fwd_capture_status) == cudaSuccess &&
         fwd_capture_status != cudaStreamCaptureStatusNone);

    // Check if a tensor will be recomputed in backward (same logic as save_tensors).
    const bool recompute_enabled_flag = mOptions.recompute_enabled();
    auto will_recompute_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled_flag || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int lyr = -1;
        std::string fld;
        if (parse_block_param(tensor_name, lyr, fld)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(fld), lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    auto persist_saved_layer_tensors = [&](int layer_idx) {
        if (!mSaved || !mSaveList) {
            return;
        }
        const std::string prefix = "blocks[" + std::to_string(layer_idx) + "].";
        auto resolve_saved_source = [&](const std::string& name) -> std::optional<Tensor> {
            // Prefer exact match from the flat tensor vector (O(1) lookup).
            if (mCurrentGraph) {
                int tid = mCurrentGraph->find_tensor_id(name);
                if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                    return mTensors[tid];
                }
                // Fall back to SSA-suffixed entries via pre-computed ssa_base_to_id (O(1) vs O(N) scan).
                auto ssa_it = mCurrentGraph->ssa_base_to_id.find(name);
                if (ssa_it != mCurrentGraph->ssa_base_to_id.end()) {
                    int sid = ssa_it->second;
                    if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) {
                        return mTensors[sid];
                    }
                }
            }

            int resolved_layer = -1;
            std::string field;
            if (!parse_block_param(name, resolved_layer, field)) {
                return std::nullopt;
            }
            const std::string base_field = strip_ssa_suffix(field);
            auto& acts = mRunState.simplified_acts(resolved_layer);
            if (base_field == "ln1_rstd" || base_field == "ln_rstd") return acts.ln1_rstd;
            if (base_field == "ln2_rstd") return acts.ln2_rstd;
            if (base_field == "q_rstd") return acts.q_rstd;
            if (base_field == "k_rstd") return acts.k_rstd;
            if (base_field == "lse") return acts.lse;
            if (base_field == "ln1" || base_field == "ln1_flat" ||
                base_field == "ln" || base_field == "ln_flat") return acts.ln1;
            if (base_field == "ln2" || base_field == "ln2_flat") return acts.ln2;
            if (base_field == "qkv" || base_field == "qkv_norm") return acts.qkv;
            if (base_field == "qkv_rope") {
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                return src;
            }
            if (base_field == "qkv_flat") {
                Tensor qkv = acts.qkv;
                return view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
            }
            if (base_field == "att" || base_field == "att_flat") return acts.att;
            if (base_field == "att_out" || base_field == "att_out_flat") return acts.att_out;
            if (base_field == "mlp_up" || base_field == "mlp_up_flat") return acts.mlp_up;
            if (base_field == "swiglu") return acts.swiglu;
            if (base_field == "swiglu_flat") {
                Tensor swiglu = acts.swiglu;
                return view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
            }
            if (base_field == "mlp_down" || base_field == "mlp_down_flat") return acts.mlp_down;
            if (base_field == "res_att" || base_field == "residual_att") return acts.residual_att;
            if (base_field == "res_ffn" || base_field == "residual_ffn" || base_field == "res_in") {
                Tensor& res = mRunState.get_residual(resolved_layer, mRunState.MainStream);
                return res;
            }
            return std::nullopt;
        };
        
        int saved_count = 0;
        int recompute_count = 0;
        for (const auto& name : *mSaveList) {
            if (name.rfind(prefix, 0) != 0) {
                continue;
            }
            if (mSaved->find(name) != mSaved->end()) {
                continue;
            }
            // Skip tensors that will be recomputed in backward — save metadata only.
            // This avoids allocating persistent cudaMalloc buffers for tensors like
            // mlp_up and swiglu that are stack-backed but fully recomputable.
            if (will_recompute_tensor(name)) {
                auto src_opt = resolve_saved_source(name);
                if (src_opt.has_value() && src_opt->Data) {
                    Tensor meta = *src_opt;
                    meta.Data = nullptr;  // Metadata only — no data pointer
                    (*mSaved)[name] = meta;
                    recompute_count++;
                }
                continue;
            }
            auto src_opt = resolve_saved_source(name);
            if (!src_opt.has_value()) {
                continue;
            }
            const Tensor& src = *src_opt;
            if (!src.Data || !mRunState.Stack.owns(src.Data)) {
                continue;
            }
            const size_t bytes = src.bytes();
            if (bytes == 0) {
                continue;
            }
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (fwd_stream_capturing) {
                    // Cannot cudaMalloc during any CUDA graph capture (internal or outer).
                    // Skip this tensor — the outer capture warmup or
                    // prepare_saved_buffers_for_capture should have pre-allocated it.
                    continue;
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoeSavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor = src;
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            bind_tensor(name, saved_tensor);
            saved_count++;
        }
    };

    // Bind known inputs (into both flat vector and write-through mirror)
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    if (!mRunState.DeepstackVisualEmbeds.empty()) {
        for (std::size_t i = 0; i < mRunState.DeepstackVisualEmbeds.size(); ++i) {
            if (!mRunState.DeepstackVisualEmbeds[i].Data) {
                continue;
            }
            bind_tensor("deepstack_visual_embeds_" + std::to_string(i), mRunState.DeepstackVisualEmbeds[i]);
        }
    }
    bind_tensor("x0", mRunState.non_block_activations().encoded);

    // Ensure non-block weights are gathered if streaming/offload is enabled
    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_embeddings(comm, mRunState.MainStream);
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
    }

    // Prefetch layer 0 before loop
    mPrefetchDirection = 1;  // Forward traversal
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->needs_block_gather()) {
            mWeightManager->gather_block(0, comm, mRunState.side_stream());
        }
        // QLoRA offload: prefetch first layer's quantized weights
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                provider->prefetch_for_layer(0, mRunState.side_stream());
            }
        }
    }

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        const auto& op = graph.ops[idx];

        if (op_trace) {
            std::cerr << "[OP " << idx << "] " << op_type_to_string(op.type)
                      << " id=" << op.op_id << std::endl;
        }

        // Handle layer boundaries
        if (op.layer_start >= 0) {
            if (op.layer_start < num_layers &&
                !layer_active[static_cast<std::size_t>(op.layer_start)]) {
                layer_checkpoints[static_cast<std::size_t>(op.layer_start)] = mRunState.Stack.checkpoint();
                layer_temp_marks[static_cast<std::size_t>(op.layer_start)] = mTemps.size();
                layer_active[static_cast<std::size_t>(op.layer_start)] = 1;
            }
            handle_layer_start(op.layer_start);
        }

        try {
            // Direct dispatch via switch (branch predictor friendly, no string compare)
            switch (op.type) {
                case CompiledOpType::Embedding:
                    dispatch_embedding(op);
                    break;
                case CompiledOpType::Zeros:
                    dispatch_zeros(op);
                    break;
                case CompiledOpType::FusedResidualRMSNorm:
                    dispatch_fused_residual_rmsnorm(op);
                    break;
                case CompiledOpType::LayerNorm:
                    dispatch_layernorm(op);
                    break;
                case CompiledOpType::View:
                    dispatch_view(op);
                    break;
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                case CompiledOpType::Matmul:
                case CompiledOpType::MatmulBias:
                    dispatch_matmul(op, hook);
                    break;
                case CompiledOpType::BiasAdd:
                    dispatch_bias_add(op);
                    break;
                case CompiledOpType::SwiGLU:
                    dispatch_swiglu(op);
                    break;
                case CompiledOpType::GptOssMoeAct:
                    dispatch_gpt_oss_moe_act(op);
                    break;
                case CompiledOpType::Silu:
                    dispatch_silu(op);
                    break;
                case CompiledOpType::Gelu:
                    dispatch_gelu(op);
                    break;
                case CompiledOpType::Relu2:
                    dispatch_relu2(op);
                    break;
                case CompiledOpType::Mul:
                    dispatch_mul(op);
                    break;
                case CompiledOpType::MaskScatter:
                    dispatch_mask_scatter(op);
                    break;
                case CompiledOpType::DeepstackInject:
                    dispatch_deepstack_inject(op);
                    break;
                case CompiledOpType::MatmulSwiGLU:
                    dispatch_matmul_swiglu(op, hook);
                    break;
                case CompiledOpType::QKVQKNorm:
                    dispatch_qkv_qk_norm(op);
                    break;
                case CompiledOpType::QKVQKNormRoPE:
                    dispatch_qkv_qk_norm_rope(op);
                    break;
                case CompiledOpType::MRoPE:
                    dispatch_mrope(op);
                    break;
                case CompiledOpType::RoPE:
                    dispatch_rope(op);
                    break;
                case CompiledOpType::FlashAttention:
                    dispatch_flash_attention(op);
                    break;
                case CompiledOpType::CrossEntropyLoss:
                    dispatch_cross_entropy_loss(op);
                    break;
                case CompiledOpType::FusedLMHeadLoss:
                    dispatch_fused_lm_head_loss(op);
                    break;
                // MoE operations
                case CompiledOpType::MoESoftmax:
                    dispatch_moe_softmax(op);
                    break;
                case CompiledOpType::MoESigmoid:
                    dispatch_moe_sigmoid(op);
                    break;
                case CompiledOpType::MoETopK:
                    dispatch_moe_topk(op);
                    break;
                case CompiledOpType::MoEPermute:
                    dispatch_moe_permute(op);
                    break;
                case CompiledOpType::MoEGroupedGemm:
                    dispatch_moe_grouped_gemm(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUp:
                    dispatch_moe_grouped_gemm_gate_up(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDown:
                    dispatch_moe_grouped_gemm_down(op);
                    break;
                case CompiledOpType::MoEUnpermute:
                    dispatch_moe_unpermute(op);
                    break;
                case CompiledOpType::MoEExpertBiasAdd:
                    dispatch_moe_expert_bias_add(op);
                    break;
                // Expert Parallelism forward operations
                case CompiledOpType::EpDispatch:
                    dispatch_ep_dispatch(op);
                    break;
                case CompiledOpType::EpCombine:
                    dispatch_ep_combine(op);
                    break;
                // Mamba/SSM forward operations
                case CompiledOpType::MambaSplitProj:
                    dispatch_mamba_split_proj(op);
                    break;
                case CompiledOpType::MambaConv1d:
                    dispatch_mamba_conv1d(op);
                    break;
                case CompiledOpType::MambaSplitConvOut:
                    dispatch_mamba_split_conv_out(op);
                    break;
                case CompiledOpType::MambaSsmScan:
                    dispatch_mamba_ssm_scan(op);
                    break;
                case CompiledOpType::MambaGatedRMSNorm:
                    dispatch_mamba_gated_rmsnorm(op);
                    break;
                case CompiledOpType::MambaOutProj:
                    dispatch_mamba_out_proj(op, hook);
                    break;
                default:
                    throw std::runtime_error("CompiledExecutor: unsupported forward op type");
            }
            // After each op, check for sticky CUDA errors (debug mode)
            if (op_trace) {
                auto post_err = cudaGetLastError();
                if (post_err != cudaSuccess) {
                    std::ostringstream oss2;
                    oss2 << "CompiledExecutor forward op " << idx
                         << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id
                         << "): left sticky CUDA error: " << cudaGetErrorString(post_err);
                    throw std::runtime_error(oss2.str());
                }
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            throw std::runtime_error(oss.str());
        }

        // Handle layer end
        if (op.layer_end >= 0) {
            if (op.layer_end < num_layers &&
                layer_active[static_cast<std::size_t>(op.layer_end)]) {
                // Debug dump per-layer tensors before shared buffers are overwritten.
                if (mDebugDumpLayerFn) {
                    mDebugDumpLayerFn(op.layer_end);
                }
                if (mConfig.NumExperts > 0) {
                    save_moe_layer_tensors(op.layer_end);
                }
                // Persist stack-backed saved tensors for this layer before the stack is restored.
                persist_saved_layer_tensors(op.layer_end);
                mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(op.layer_end)]);
                if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(op.layer_end)]) {
                    mTemps.resize(layer_temp_marks[static_cast<std::size_t>(op.layer_end)]);
                }
                prune_stack_tensors();
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                // Note: cudnn_workspace is persistently allocated, don't clear
                layer_active[static_cast<std::size_t>(op.layer_end)] = 0;
            }
            handle_layer_end(op.layer_end);
        }
    }

    // Free temporaries
    for (auto it = mTemps.rbegin(); it != mTemps.rend(); ++it) {
        mRunState.temp_free(*it);
    }
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_embeddings(mRunState.MainStream);
        mWeightManager->release_final_norm(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook,
                                        bool skip_zeroing) {
    mComm = &comm;
    mCurrentGraph = &graph;
    mRunState.reset_simplified_gradients();
    mTemps.clear();
    // For EP models, keep forward-cached host offsets (populated by ep_dispatch).
    // During gradient checkpointing recompute, ep_dispatch is skipped (it's a
    // communication op), so the GPU persistent buffers may be stale. The forward
    // cache has the correct merged expert offsets for each layer.
    if (mConfig.EPSize <= 1) {
        mMoEHostOffsetsCache.clear();
    }
    mTensors.assign(static_cast<std::size_t>(graph.num_tensors), Tensor{});
    mAccumulateTensors.clear();
    mCurrentLayer = -1;
    mLastRecomputeLayer = -1;
    mMicroStep = micro_step;

    // Clear activation/non-block gradients for each micro-step.
    // When called from GraphExecutor::backward_with_hook(), the caller already zeroes these
    // buffers, so skip_zeroing=true avoids redundant GPU work.
    if (!skip_zeroing) {
        fill_zero(mRunState.non_block_gradients().d_ln_final, mRunState.MainStream);
        if (mRunState.non_block_gradients().d_embeddings.Data && !mRunState.is_lora_only_mode()) {
            fill_zero(mRunState.non_block_gradients().d_embeddings, mRunState.MainStream);
        }
        if (mConfig.NumLayers > 0) {
            fill_zero(mRunState.simplified_grads(static_cast<int>(mConfig.NumLayers) - 1).d_res_ffn,
                      mRunState.MainStream);
        }
        mRunState.zero_activation_gradients(mRunState.MainStream);
    }

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->gather_lm_head(comm, mRunState.MainStream);
        }
    }

    // Prefetch last layer before backward loop (layers processed in reverse)
    mPrefetchDirection = -1;  // Backward traversal
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->needs_block_gather()) {
            const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
            mWeightManager->gather_block(last_layer, comm, mRunState.side_stream());
        }
        // QLoRA offload: prefetch last layer's quantized weights for backward
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
                provider->prefetch_for_layer(last_layer, mRunState.side_stream());
            }
        }
    }

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
    auto clear_shared_grads = [&](int layer_idx) {
        // No-op: d_ln2, d_att, d_ln1 are fully overwritten by their respective
        // backward ops before being read. Gradient data flows through mTensors[],
        // not through the pre-allocated simplified_grads buffers. The matmul backward
        // only temporarily remaps these pointers during LoRA hooks, then restores them.
        (void)layer_idx;
    };
    auto prune_stack_tensors = [&](int current_layer) {
        // Prune flat tensor vector using pre-computed metadata (no string parsing)
        for (int id = 0; id < graph.num_tensors; ++id) {
            auto& t = mTensors[static_cast<std::size_t>(id)];
            if (!t.Data) continue;
            const auto& meta = graph.tensor_meta[static_cast<std::size_t>(id)];
            // Skip MoE expert_offsets
            if (meta.is_moe_offsets()) continue;
            // Skip cross-layer gradients for earlier layers
            if (current_layer >= 0 && meta.is_d_blocks() &&
                meta.block_layer_idx >= 0 && meta.block_layer_idx < current_layer) continue;
            // Skip saved tensors for earlier layers
            if (current_layer >= 0 && meta.is_blocks() &&
                meta.block_layer_idx >= 0 && meta.block_layer_idx < current_layer) continue;
            // Skip tensors with unparseable layer index (be safe)
            if ((meta.is_d_blocks() || meta.is_blocks()) && meta.block_layer_idx < 0) continue;
            if (mRunState.Stack.owns(t.Data) && !mRunState.Stack.is_live(t.Data)) {
                t = Tensor{};
            }
        }
    };

    // Bind initial gradient tensors (from loss computation)
    // d_logits is stored in the output buffer after loss backward (only when lmhead_chunks == 1)
    auto& output = mRunState.non_block_activations().output;
    if (!output.Data) {
        throw std::runtime_error("CompiledExecutor: output tensor has no data (B=" +
                                std::to_string(mB) + ", T=" + std::to_string(mT) + ")");
    }

    if (mOptions.LMHeadChunks <= 1) {
        Tensor logits_view = view_tensor(output, {mB, mT, static_cast<long>(mConfig.VocabSize)});
        bind_tensor("d_logits", logits_view);
        // Also provide flattened version for matmul backward ops
        Tensor logits_flat = view_tensor(output, {mB * mT, static_cast<long>(mConfig.VocabSize)});
        if (logits_flat.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat has wrong rank=" +
                                    std::to_string(logits_flat.Rank) + " expected 2");
        }
        bind_tensor("d_logits_flat", logits_flat);
    }

    // Bind gradient output buffers for final layer norm backward
    // DSL-driven: use slot registry to derive all mappings from gradient_of relationships
    Tensor& d_ln_final_buf = mRunState.non_block_gradients().d_ln_final;
    Tensor& d_embeddings_buf = mRunState.non_block_gradients().d_embeddings;

    Tensor d_ln_final_flat = view_tensor(d_ln_final_buf,
                                         {mB * mT, static_cast<long>(mConfig.HiddenSize)});

    // Helper to determine target buffer based on gradient_of field
    auto get_target_buffer = [&](const std::string& grad_of) -> Tensor* {
        // Final norm gradients (xF, ln_final, residual_final)
        if (grad_of == "xF" || grad_of == "ln_final" || grad_of == "xF_flat" ||
            grad_of == "residual_final" || grad_of == "final_residual") {
            return &d_ln_final_buf;
        }
        // Embedding output gradients (x0, encoded) — always bind to persistent buffer
        if (grad_of == "x0" || grad_of == "encoded" || grad_of == "embeddings") {
            return &d_embeddings_buf;
        }
        // Note: d_xN, d_residualN don't map to persistent buffers - they're computed on-the-fly
        return nullptr;
    };

    // Bind global gradient tensors - these are always needed regardless of DSL layout
    // The DSL gradient slots declare shape/dtype but the actual buffers come from RunState
    bind_tensor("d_xF_flat", d_ln_final_flat);
    bind_tensor("d_xF", d_ln_final_buf);
    bind_tensor("d_ln_final", d_ln_final_buf);
    bind_tensor("d_ln_final_flat", d_ln_final_flat);

    // Always bind embedding gradients to the persistent d_embeddings buffer, even in
    // LoRA-only mode. This prevents ensure_output_tensor from stack-allocating them,
    // which would block can_restore_stack for the entire backward pass.
    bind_tensor("d_encoded", d_embeddings_buf);
    bind_tensor("d_x0", d_embeddings_buf);

    // DSL-driven binding for any additional gradient slots declared in the Python model
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout()) {
        mSlotRegistry->for_each([&](const std::string& slot_name,
                                    const TensorSlotRegistry::SlotEntry& entry) {
            if (entry.scope != ActivationScope::GlobalGradient) return;
            // Skip if already bound above
            if (mCurrentGraph) {
                int sid = mCurrentGraph->find_tensor_id(slot_name);
                if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) return;
            }

            Tensor* target_buf = get_target_buffer(entry.gradient_of);
            if (target_buf && target_buf->Data) {
                bind_tensor(slot_name, *target_buf);
            }
        });
    }

    // Ensure global block outputs (xN/residualN) map to the last block's gradients.
    // These gradients must survive layer-boundary stack restores in recompute mode.
    if (mConfig.NumLayers > 0) {
        const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
        auto& last_grads = mRunState.simplified_grads(last_layer);
        if (last_grads.d_mlp_down.Data) {
            bind_tensor("d_xN", last_grads.d_mlp_down);
        }
        if (last_grads.d_res_att.Data) {
            bind_tensor("d_residualN", last_grads.d_res_att);
        }

        // Heuristic aliasing for non-inlined StackedBlocks outputs (e.g., "StackedBlocks_4").
        if (mSaved) {
            std::vector<std::pair<int, std::string>> stacked;
            stacked.reserve(2);
            for (const auto& kv : *mSaved) {
                const std::string& name = kv.first;
                if (name.rfind("StackedBlocks_", 0) != 0) {
                    continue;
                }
                int idx = -1;
                const char* s = name.c_str() + std::strlen("StackedBlocks_");
                if (*s) {
                    char* end = nullptr;
                    long parsed = std::strtol(s, &end, 10);
                    if (end != s) {
                        idx = static_cast<int>(parsed);
                    }
                }
                if (idx >= 0) {
                    stacked.emplace_back(idx, name);
                }
            }
            if (!stacked.empty()) {
                std::sort(stacked.begin(), stacked.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                if (stacked.size() == 1) {
                    if (last_grads.d_res_att.Data) {
                        bind_tensor("d_" + stacked[0].second, last_grads.d_res_att);
                    }
                } else {
                    if (last_grads.d_mlp_down.Data) {
                        bind_tensor("d_" + stacked[0].second, last_grads.d_mlp_down);
                    }
                    if (last_grads.d_res_att.Data) {
                        bind_tensor("d_" + stacked[1].second, last_grads.d_res_att);
                    }
                }
            }
        }
    }

    // Bind autodiff-generated gradient names (d_embed_1, etc.) from forward embedding outputs.
    // Always bind even in LoRA-only mode to prevent stack-allocation (see d_embeddings comment above).
    for (const auto& emb_out : mEmbeddingOutputs) {
        std::string grad_name = "d_" + emb_out;
        bind_tensor(grad_name, d_embeddings_buf);
    }

    // Restore MoE expert_offsets from persistent CPU storage
    // This is needed by grouped GEMM backward ops for proper token routing
    if (mConfig.NumExperts > 0 && !mMoEExpertOffsetsData.empty()) {
        // Allocate PERSISTENT GPU buffer for expert_offsets (not stack-allocated)
        // This ensures the memory won't be invalidated by stack restores or temp_free calls
        const int num_elements = static_cast<int>(mMoEExpertOffsetsData.size());
        const size_t needed_bytes = num_elements * sizeof(int);

        // Allocate or resize GPU buffer if needed
        if (mMoEExpertOffsetsGPU == nullptr || mMoEExpertOffsetsGPUSize < needed_bytes) {
            if (mMoEExpertOffsetsGPU) {
                CUDA_CHECK(cudaFree(mMoEExpertOffsetsGPU));
            }
            CUDA_CHECK(cudaMalloc(&mMoEExpertOffsetsGPU, needed_bytes));
            mMoEExpertOffsetsGPUSize = needed_bytes;
        }

        // Copy data from CPU to GPU
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsGPU, mMoEExpertOffsetsData.data(),
                                   needed_bytes, cudaMemcpyHostToDevice, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

        // Create tensor wrapper pointing to persistent buffer
        Tensor expert_offsets;
        expert_offsets.DType = ETensorDType::INT32;
        expert_offsets.Rank = 1;
        expert_offsets.Sizes[0] = num_elements;
        expert_offsets.Data = static_cast<std::byte*>(mMoEExpertOffsetsGPU);

        bind_tensor("moe_expert_offsets", expert_offsets);
        // Note: NOT adding to mTemps since this is persistent memory managed separately
    }

    // Also bind standard inputs that backward ops may reference
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    if (!mRunState.DeepstackVisualEmbeds.empty()) {
        for (std::size_t i = 0; i < mRunState.DeepstackVisualEmbeds.size(); ++i) {
            if (!mRunState.DeepstackVisualEmbeds[i].Data) {
                continue;
            }
            bind_tensor("deepstack_visual_embeds_" + std::to_string(i), mRunState.DeepstackVisualEmbeds[i]);
        }
    }

    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors so they're used instead of temporaries.
    // This mirrors the logic in graph_executor_backward.cpp (bind_param_grad).
    for (const auto& param_name : mGrads.param_names()) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            continue;
        }
        bool accumulate = false;
        Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
        if (grad_tensor && grad_tensor->Data) {
            std::string grad_name = "d_" + param_name;
            bind_tensor(grad_name, *grad_tensor);
            if (accumulate) {
                mAccumulateTensors.insert(grad_name);
            }
        }
    }

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss:
                return true;
            default:
                return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto op_layer_idx = [&](const CompiledOp& op) -> int {
        int detected_non_grad = -1;
        for (const auto& ref : op.inputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        for (const auto& ref : op.outputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        return (detected_non_grad >= 0) ? detected_non_grad : -1;
    };

    auto op_layer_idx_any = [&](const CompiledOp& op) -> int {
        int detected_any = -1;
        for (const auto& ref : op.inputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_any = std::max(detected_any, op.attrs.layer_idx);
        }
        return detected_any;
    };

    const bool skip_logits_grad = (mOptions.LMHeadChunks > 1);
    auto is_logits_grad_name = [](const std::string& name) {
        return name == "d_logits" || name == "d_logits_flat";
    };
    auto is_logits_grad_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        return false;
    };

    // Use pre-computed last-use data from graph compilation (avoids rebuilding every backward).
    const auto& last_use_names = graph.last_use_names;
    const auto& last_use = graph.last_use_index;
    auto prune_by_last_use = [&](std::size_t idx) {
        if (idx >= last_use_names.size()) {
            return;
        }
        for (const auto& name : last_use_names[idx]) {
            if (mCurrentGraph) {
                int tid = mCurrentGraph->find_tensor_id(name);
                if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                    mTensors[tid] = Tensor{};
                }
            }
        }
    };
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";

    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);
        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);
            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                    clear_shared_grads(layer_idx);
                    layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                }
                mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                mLastRecomputeLayer = layer_idx;
            }
        }
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            const int layer_idx_any = op_layer_idx_any(op);
            // Always recompute when switching layers. This is critical because:
            // - Shared buffers (ln1, ln2, qkv, mlp_up, swiglu) contain only ONE layer's data
            // - If the backward graph interleaves ops from different layers, we MUST
            //   recompute to ensure the correct layer's data is in the shared buffers
            // - The old check (missing_start || op_before_start) would skip recomputation
            //   for layer N's late ops if we had already visited layer N earlier, causing
            //   those ops to read stale data from whatever layer was recomputed last
            // Use layer_idx_any as fallback when layer_idx is -1
            const int effective_layer_idx = (layer_idx >= 0) ? layer_idx : layer_idx_any;
            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                if (effective_layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(effective_layer_idx)]) {
                    clear_shared_grads(effective_layer_idx);
                    layer_seen_any[static_cast<std::size_t>(effective_layer_idx)] = true;
                }
                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                mLastRecomputeLayer = effective_layer_idx;
            }
        }

        try {
            switch (op.type) {
                // Explicit backward ops
                case CompiledOpType::ViewBackward:
                    dispatch_view_backward(op);
                    break;
                case CompiledOpType::AddBackward:
                    dispatch_add_backward(op);
                    break;
                case CompiledOpType::CrossEntropyLossBackward:
                    dispatch_cross_entropy_loss_backward(op);
                    break;
                case CompiledOpType::FusedLMHeadLossBackward:
                    dispatch_fused_lm_head_loss_backward(op);
                    break;
                case CompiledOpType::MatmulBackward:
                    dispatch_matmul_backward(op, hook);
                    // After the first matmul_backward (LM-head backward), free the output tensor
                    // to reclaim ~1.2GB of stack memory. The d_logits data has been consumed.
                    if (idx == 1) {
                        mRunState.temp_free(mRunState.non_block_activations().output);
                        mTemps.clear();
                        // Update initial_checkpoint to reflect the freed output tensor
                        // This prevents subsequent checkpoint restores from re-allocating it
                        initial_checkpoint = mRunState.Stack.checkpoint();
                    }
                    break;
                case CompiledOpType::BiasAddBackward:
                    dispatch_bias_add_backward(op);
                    break;
                case CompiledOpType::SwiGLUBackward:
                    dispatch_swiglu_backward(op);
                    break;
                case CompiledOpType::GptOssMoeActBackward:
                    dispatch_gpt_oss_moe_act_backward(op);
                    break;
                case CompiledOpType::SiluBackward:
                    dispatch_silu_backward(op);
                    break;
                case CompiledOpType::GeluBackward:
                    dispatch_gelu_backward(op);
                    break;
                case CompiledOpType::Relu2Backward:
                    dispatch_relu2_backward(op);
                    break;
                case CompiledOpType::MulBackward:
                    dispatch_mul_backward(op);
                    break;
                case CompiledOpType::MaskScatterBackward:
                    dispatch_mask_scatter_backward(op);
                    break;
                case CompiledOpType::DeepstackInjectBackward:
                    dispatch_deepstack_inject_backward(op);
                    break;
                case CompiledOpType::MatmulSwiGLUBackward:
                    dispatch_matmul_swiglu_backward(op, hook);
                    break;
                case CompiledOpType::QKVQKNormBackward:
                    dispatch_qkv_qk_norm_backward(op);
                    break;
                case CompiledOpType::RoPEBackward:
                    dispatch_rope_backward(op);
                    break;
                case CompiledOpType::QKVQKNormRoPEBackward:
                    dispatch_qkv_qk_norm_rope_backward(op);
                    break;
                case CompiledOpType::MRoPEBackward:
                    dispatch_mrope_backward(op);
                    break;
                case CompiledOpType::FlashAttentionBackward:
                    dispatch_flash_attention_backward(op);
                    break;
                case CompiledOpType::ZerosBackward:
                    dispatch_zeros_backward(op);
                    break;
                case CompiledOpType::FusedResidualRMSNormBackward:
                    dispatch_fused_residual_rmsnorm_backward(op);
                    break;
                case CompiledOpType::LayerNormBackward:
                    dispatch_layernorm_backward(op);
                    break;
                case CompiledOpType::EmbeddingBackward:
                    dispatch_embedding_backward(op);
                    break;

                // Forward ops that appear in backward graph (autodiff generates these)
                // View/reshape is the same operation in forward and backward - just reshapes gradient
                case CompiledOpType::View:
                    dispatch_view_backward(op);
                    break;
                // "add" ops in the backward graph are gradient-accumulation nodes,
                // so we must execute them as forward add (sum inputs), not add-backward.
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                // Zeros in backward is a no-op
                case CompiledOpType::Zeros:
                    dispatch_zeros_backward(op);
                    break;

                // MoE backward operations
                case CompiledOpType::MoESoftmaxBackward:
                    dispatch_moe_softmax_backward(op);
                    break;
                case CompiledOpType::MoESigmoidBackward:
                    dispatch_moe_sigmoid_backward(op);
                    break;
                case CompiledOpType::MoETopKBackward:
                    dispatch_moe_topk_backward(op);
                    break;
                case CompiledOpType::MoEPermuteBackward:
                    dispatch_moe_permute_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmBackward:
                    dispatch_moe_grouped_gemm_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUpBackward:
                    dispatch_moe_grouped_gemm_gate_up_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDownBackward:
                    dispatch_moe_grouped_gemm_down_backward(op);
                    break;
                case CompiledOpType::MoEUnpermuteBackward:
                    dispatch_moe_unpermute_backward(op);
                    break;
                case CompiledOpType::MoEExpertBiasAddBackward:
                    dispatch_moe_expert_bias_add_backward(op);
                    break;

                // Expert Parallelism backward operations
                case CompiledOpType::EpDispatchBackward:
                    dispatch_ep_dispatch_backward(op);
                    break;
                case CompiledOpType::EpCombineBackward:
                    dispatch_ep_combine_backward(op);
                    break;

                // MoE forward ops that may appear in backward graph
                case CompiledOpType::MoESoftmax:
                case CompiledOpType::MoESigmoid:
                case CompiledOpType::MoETopK:
                case CompiledOpType::MoEPermute:
                case CompiledOpType::MoEGroupedGemm:
                case CompiledOpType::MoEGroupedGemmGateUp:
                case CompiledOpType::MoEGroupedGemmDown:
                case CompiledOpType::MoEUnpermute:
                case CompiledOpType::EpDispatch:
                case CompiledOpType::EpCombine:
                case CompiledOpType::Silu:
                case CompiledOpType::Relu2:
                case CompiledOpType::Mul:
                    // These forward MoE/EP ops may appear in backward graph due to autodiff
                    throw std::runtime_error("CompiledExecutor: MoE/EP forward op in backward graph not yet supported");

                // Mamba/SSM backward operations
                case CompiledOpType::MambaSplitProjBackward:
                    dispatch_mamba_split_proj_backward(op);
                    break;
                case CompiledOpType::MambaConv1dBackward:
                    dispatch_mamba_conv1d_backward(op);
                    break;
                case CompiledOpType::MambaSplitConvOutBackward:
                    dispatch_mamba_split_conv_out_backward(op);
                    break;
                case CompiledOpType::MambaSsmScanBackward:
                    dispatch_mamba_ssm_scan_backward(op);
                    break;
                case CompiledOpType::MambaGatedRMSNormBackward:
                    dispatch_mamba_gated_rmsnorm_backward(op);
                    break;
                case CompiledOpType::MambaOutProjBackward:
                    dispatch_mamba_out_proj_backward(op, hook);
                    break;

                // Mamba forward ops that may appear in backward graph
                case CompiledOpType::MambaSplitProj:
                case CompiledOpType::MambaConv1d:
                case CompiledOpType::MambaSplitConvOut:
                case CompiledOpType::MambaSsmScan:
                case CompiledOpType::MambaGatedRMSNorm:
                case CompiledOpType::MambaOutProj:
                    // These forward Mamba ops may appear in backward graph due to autodiff
                    throw std::runtime_error("CompiledExecutor: Mamba forward op in backward graph not yet supported");

                default: {
                    std::ostringstream oss;
                    oss << "CompiledExecutor: unsupported backward op type at idx " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            // Per-op CUDA error check in trace mode: detects illegal memory accesses from launched kernels
            if (op_trace) {
                auto op_err = cudaDeviceSynchronize();
                if (op_err != cudaSuccess) {
                    std::ostringstream oss;
                    oss << "CompiledExecutor backward op " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id
                        << "): CUDA error after execution: " << cudaGetErrorString(op_err);
                    throw std::runtime_error(oss.str());
                }
            }

            // Memory management - prune tensors after last use, then restore stack at layer boundaries.
            // If live cross-layer tensors exist on the stack, persist them to allocated memory first.
            prune_by_last_use(idx);
            if (op.layer_end >= 0 &&
                op.layer_end != last_layer_restored) {
                // Persist any cross-layer tensors that still live on the stack.
                // These tensors (e.g., d_blocks[N].router_logits for MoE aux loss) have
                // last_use beyond the current layer, so they must survive stack restore.
                for (const auto& [name, use_idx] : last_use) {
                    if (use_idx <= idx) continue;
                    int tid = graph.find_tensor_id(name);
                    if (tid < 0 || static_cast<std::size_t>(tid) >= mTensors.size()) continue;
                    auto& tensor = mTensors[static_cast<std::size_t>(tid)];
                    if (tensor.Data && mRunState.Stack.owns(tensor.Data)) {
                        // Copy to persistent GPU memory
                        const std::size_t nbytes = tensor.bytes();
                        std::byte* persistent = nullptr;
                        CUDA_CHECK(cudaMalloc(&persistent, nbytes));
                        CUDA_CHECK(cudaMemcpyAsync(persistent, tensor.Data, nbytes,
                                                    cudaMemcpyDeviceToDevice, mRunState.MainStream));
                        tensor.Data = persistent;
                        // Track for cleanup at end of backward
                        mPersistedBackwardTensors.push_back(persistent);
                    }
                }

                // Release this layer's offloaded weights (if applicable)
                handle_layer_end(op.layer_end);

                // Trigger async gradient reduction for this layer on side_stream.
                // This overlaps communication with the next layer's backward compute on MainStream.
                if (mComm && mComm->world_size() > 1) {
                    // Record event on MainStream to ensure layer gradients are ready
                    CUDA_CHECK(cudaEventRecord(mRunState.side_stream_event(), mRunState.MainStream));
                    // Wait for gradients on side_stream before starting reduction
                    CUDA_CHECK(cudaStreamWaitEvent(mRunState.side_stream(), mRunState.side_stream_event(), 0));
                    // Start async reduction on side_stream (overlaps with next layer backward on MainStream)
                    mGrads.notify_block(op.layer_end, mRunState.side_stream(), *mComm);
                }
                // Restore stack and clear temps
                mRunState.Stack.restore(initial_checkpoint);
                mTemps.clear();
                prune_stack_tensors(op.layer_end);
                // Note: cudnn_workspace is persistently allocated, no need to clear
                // Clear stack-allocated tensor pointers in simplified_acts/grads for this layer.
                // These pointers become stale after checkpoint restore.
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                if (mRunState.large_bwd_temps_on_stack()) {
                    auto& grads_to_clear = mRunState.simplified_grads(op.layer_end);
                    grads_to_clear.d_qkv.Data = nullptr;
                    grads_to_clear.d_mlp_up.Data = nullptr;
                    grads_to_clear.d_swiglu.Data = nullptr;
                }
                last_layer_restored = op.layer_end;
            }
            // Every N ops as fallback (catches non-annotated layers)
            // NOTE: When recompute is disabled, we cannot aggressively prune tensors because
            // the backward graph may reference intermediate tensors (like d_blocks[N].view_K)
            // that were produced earlier but are still needed. The stack restore + prune
            // would remove these tensors from mTensors, causing "tensor not found" errors.
            // For now, skip periodic cleanup when recompute is disabled to preserve correctness.
            // Memory usage will be higher but the backward pass will complete successfully.

            // After each backward op, check for CUDA errors (lightweight, non-blocking)
            {
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::ostringstream oss2;
                    oss2 << "CompiledExecutor backward op " << idx
                         << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id
                         << "): CUDA error: " << cudaGetErrorString(err);
                    throw std::runtime_error(oss2.str());
                }
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            // Add inputs/outputs for debugging
            oss << "\n  inputs: [";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.inputs[i].name << "(slot=" << static_cast<int>(op.inputs[i].slot) << ")";
            }
            oss << "]";
            oss << "\n  outputs: [";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.outputs[i].name << "(slot=" << static_cast<int>(op.outputs[i].slot) << ")";
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }

    }

    // Final cleanup - pass -1 to allow full pruning (backward complete)
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors(-1);
    mTemps.clear();

    // Free persisted cross-layer backward tensors
    // Sync main stream first to ensure all consumers have completed
    if (!mPersistedBackwardTensors.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        for (auto* ptr : mPersistedBackwardTensors) {
            cudaFree(ptr);
        }
        mPersistedBackwardTensors.clear();
    }

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_final_norm(mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
    }
}

}  // namespace dsl
