// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
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

namespace {

}  // namespace

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
        case CompiledOpType::Ones: return "ones";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::RMSNorm: return "rmsnorm";
        case CompiledOpType::LayerNorm: return "layernorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Transpose: return "transpose";
        case CompiledOpType::Split: return "split";
        case CompiledOpType::Narrow: return "narrow";
        case CompiledOpType::Concat: return "concat";
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
        case CompiledOpType::Scale: return "scale";
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
        case CompiledOpType::ScaleBackward: return "scale_backward";
        case CompiledOpType::NarrowBackward: return "narrow_backward";
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
        case CompiledOpType::RMSNormBackward: return "rmsnorm_backward";
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
        case CompiledOpType::ChunkGatedDeltaRule: return "chunk_gated_delta_rule";
        case CompiledOpType::Qwen3_5Decay: return "qwen3_5_decay";
        case CompiledOpType::RepeatInterleaveHeads: return "repeat_interleave_heads";
        case CompiledOpType::ChunkGatedDeltaRuleBackward: return "chunk_gated_delta_rule_backward";
        case CompiledOpType::Qwen3_5DecayBackward: return "qwen3_5_decay_backward";
        case CompiledOpType::RepeatInterleaveHeadsBackward: return "repeat_interleave_heads_backward";
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
{
    // Load JIT-compiled Triton kernels for gated delta rule (if manifests available)
    if (!options.JitKernelManifests.empty()) {
        mGdrKernels.load(options.JitKernelManifests);
    }
}

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
        if (state.sorted_recv_gpu) cudaFree(state.sorted_recv_gpu);
        if (state.combined_gpu) cudaFree(state.combined_gpu);
        if (state.llep_combined_gpu) cudaFree(state.llep_combined_gpu);
        if (state.dispatch_bwd_send_gpu) cudaFree(state.dispatch_bwd_send_gpu);
        if (state.dispatch_bwd_out_gpu) cudaFree(state.dispatch_bwd_out_gpu);
        if (state.combine_bwd_sorted_gpu) cudaFree(state.combine_bwd_sorted_gpu);
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

void CompiledExecutor::clear_replay_copied_refs() {
    if (mReplayCopiedBuffers.empty()) {
        return;
    }

    auto points_to_replay_copy = [&](const std::byte* data) -> bool {
        if (!data) {
            return false;
        }
        for (void* ptr : mReplayCopiedBuffers) {
            if (data == static_cast<const std::byte*>(ptr)) {
                return true;
            }
        }
        return false;
    };

    for (auto& tensor : mTensors) {
        if (points_to_replay_copy(tensor.Data)) {
            tensor = Tensor{};
        }
    }

    for (auto it = mNamedTensors.begin(); it != mNamedTensors.end();) {
        if (points_to_replay_copy(it->second.Data)) {
            it = mNamedTensors.erase(it);
        } else {
            ++it;
        }
    }

    if (mSaved) {
        for (auto& [name, tensor] : *mSaved) {
            if (points_to_replay_copy(tensor.Data)) {
                tensor.Data = nullptr;
            }
        }
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



// ============================================================================
// Split-attention CUDA graph support
// ============================================================================

void CompiledExecutor::dispatch_forward_op(const CompiledOp& op,
                                            const modules::ForwardHook* hook) {
    switch (op.type) {
        case CompiledOpType::Embedding:          dispatch_embedding(op); break;
        case CompiledOpType::Zeros:              dispatch_zeros(op); break;
        case CompiledOpType::Ones:               dispatch_ones(op); break;
        case CompiledOpType::FusedResidualRMSNorm: dispatch_fused_residual_rmsnorm(op); break;
        case CompiledOpType::RMSNorm:            dispatch_rmsnorm(op); break;
        case CompiledOpType::LayerNorm:          dispatch_layernorm(op); break;
        case CompiledOpType::View:               dispatch_view(op); break;
        case CompiledOpType::Transpose:          dispatch_transpose(op); break;
        case CompiledOpType::Split:              dispatch_split(op); break;
        case CompiledOpType::Narrow:             dispatch_narrow(op); break;
        case CompiledOpType::Concat:             dispatch_concat(op); break;
        case CompiledOpType::Add:                dispatch_add(op); break;
        case CompiledOpType::Matmul:
        case CompiledOpType::MatmulBias:         dispatch_matmul(op, hook); break;
        case CompiledOpType::BiasAdd:            dispatch_bias_add(op); break;
        case CompiledOpType::SwiGLU:             dispatch_swiglu(op); break;
        case CompiledOpType::GptOssMoeAct:       dispatch_gpt_oss_moe_act(op); break;
        case CompiledOpType::Silu:               dispatch_silu(op); break;
        case CompiledOpType::Gelu:               dispatch_gelu(op); break;
        case CompiledOpType::Relu2:              dispatch_relu2(op); break;
        case CompiledOpType::Mul:                dispatch_mul(op); break;
        case CompiledOpType::Scale:              dispatch_scale(op); break;
        case CompiledOpType::MaskScatter:        dispatch_mask_scatter(op); break;
        case CompiledOpType::DeepstackInject:    dispatch_deepstack_inject(op); break;
        case CompiledOpType::MatmulSwiGLU:       dispatch_matmul_swiglu(op, hook); break;
        case CompiledOpType::QKVQKNorm:          dispatch_qkv_qk_norm(op); break;
        case CompiledOpType::QKVQKNormRoPE:      dispatch_qkv_qk_norm_rope(op); break;
        case CompiledOpType::MRoPE:              dispatch_mrope(op); break;
        case CompiledOpType::RoPE:               dispatch_rope(op); break;
        case CompiledOpType::FlashAttention:     dispatch_flash_attention(op); break;
        case CompiledOpType::CrossEntropyLoss:   dispatch_cross_entropy_loss(op); break;
        case CompiledOpType::FusedLMHeadLoss:    dispatch_fused_lm_head_loss(op); break;
        case CompiledOpType::MoESoftmax:         dispatch_moe_softmax(op); break;
        case CompiledOpType::MoESigmoid:         dispatch_moe_sigmoid(op); break;
        case CompiledOpType::MoETopK:            dispatch_moe_topk(op); break;
        case CompiledOpType::MoEPermute:         dispatch_moe_permute(op); break;
        case CompiledOpType::MoEGroupedGemm:     dispatch_moe_grouped_gemm(op); break;
        case CompiledOpType::MoEGroupedGemmGateUp: dispatch_moe_grouped_gemm_gate_up(op); break;
        case CompiledOpType::MoEGroupedGemmDown: dispatch_moe_grouped_gemm_down(op); break;
        case CompiledOpType::MoEUnpermute:       dispatch_moe_unpermute(op); break;
        case CompiledOpType::MoEExpertBiasAdd:   dispatch_moe_expert_bias_add(op); break;
        case CompiledOpType::EpDispatch:         dispatch_ep_dispatch(op); break;
        case CompiledOpType::EpCombine:          dispatch_ep_combine(op); break;
        case CompiledOpType::MambaSplitProj:     dispatch_mamba_split_proj(op); break;
        case CompiledOpType::MambaConv1d:        dispatch_mamba_conv1d(op); break;
        case CompiledOpType::MambaSplitConvOut:  dispatch_mamba_split_conv_out(op); break;
        case CompiledOpType::MambaSsmScan:       dispatch_mamba_ssm_scan(op); break;
        case CompiledOpType::MambaGatedRMSNorm:  dispatch_mamba_gated_rmsnorm(op); break;
        case CompiledOpType::MambaOutProj:       dispatch_mamba_out_proj(op, hook); break;
        case CompiledOpType::ChunkGatedDeltaRule: dispatch_chunk_gated_delta_rule(op); break;
        case CompiledOpType::Qwen3_5Decay:       dispatch_qwen3_5_decay(op); break;
        case CompiledOpType::RepeatInterleaveHeads: dispatch_repeat_interleave_heads(op); break;
        default:
            throw std::runtime_error("dispatch_forward_op: unsupported op type");
    }
}

void CompiledExecutor::dispatch_backward_op(const CompiledOp& op,
                                             const modules::BackwardHook* hook) {
    switch (op.type) {
        // Explicit backward ops
        case CompiledOpType::ViewBackward:         dispatch_view_backward(op); break;
        case CompiledOpType::AddBackward:          dispatch_add_backward(op); break;
        case CompiledOpType::MatmulBackward:       dispatch_matmul_backward(op, hook); break;
        case CompiledOpType::BiasAddBackward:      dispatch_bias_add_backward(op); break;
        case CompiledOpType::SwiGLUBackward:       dispatch_swiglu_backward(op); break;
        case CompiledOpType::GptOssMoeActBackward: dispatch_gpt_oss_moe_act_backward(op); break;
        case CompiledOpType::SiluBackward:         dispatch_silu_backward(op); break;
        case CompiledOpType::GeluBackward:         dispatch_gelu_backward(op); break;
        case CompiledOpType::Relu2Backward:        dispatch_relu2_backward(op); break;
        case CompiledOpType::MulBackward:          dispatch_mul_backward(op); break;
        case CompiledOpType::ScaleBackward:        dispatch_scale_backward(op); break;
        case CompiledOpType::NarrowBackward:       dispatch_narrow_backward(op); break;
        case CompiledOpType::MaskScatterBackward:  dispatch_mask_scatter_backward(op); break;
        case CompiledOpType::DeepstackInjectBackward: dispatch_deepstack_inject_backward(op); break;
        case CompiledOpType::MatmulSwiGLUBackward: dispatch_matmul_swiglu_backward(op, hook); break;
        case CompiledOpType::QKVQKNormBackward:    dispatch_qkv_qk_norm_backward(op); break;
        case CompiledOpType::RoPEBackward:         dispatch_rope_backward(op); break;
        case CompiledOpType::QKVQKNormRoPEBackward: dispatch_qkv_qk_norm_rope_backward(op); break;
        case CompiledOpType::MRoPEBackward:        dispatch_mrope_backward(op); break;
        case CompiledOpType::FlashAttentionBackward: dispatch_flash_attention_backward(op); break;
        case CompiledOpType::ZerosBackward:        dispatch_zeros_backward(op); break;
        case CompiledOpType::FusedResidualRMSNormBackward: dispatch_fused_residual_rmsnorm_backward(op); break;
        case CompiledOpType::RMSNormBackward:      dispatch_rmsnorm_backward(op); break;
        case CompiledOpType::LayerNormBackward:    dispatch_layernorm_backward(op); break;
        case CompiledOpType::EmbeddingBackward:    dispatch_embedding_backward(op); break;
        case CompiledOpType::CrossEntropyLossBackward: dispatch_cross_entropy_loss_backward(op); break;
        case CompiledOpType::FusedLMHeadLossBackward: dispatch_fused_lm_head_loss_backward(op); break;
        // Forward ops that appear in backward graph (autodiff generates these)
        case CompiledOpType::View:                 dispatch_view_backward(op); break;
        case CompiledOpType::Transpose:            dispatch_transpose(op); break;
        case CompiledOpType::Split:                dispatch_split(op); break;
        case CompiledOpType::Narrow:               dispatch_narrow(op); break;
        case CompiledOpType::Concat:               dispatch_concat(op); break;
        case CompiledOpType::Add:                  dispatch_add(op); break;
        case CompiledOpType::Zeros:                dispatch_zeros_backward(op); break;
        case CompiledOpType::Ones:                 dispatch_zeros_backward(op); break;
        // MoE backward operations
        case CompiledOpType::MoESoftmaxBackward:   dispatch_moe_softmax_backward(op); break;
        case CompiledOpType::MoESigmoidBackward:   dispatch_moe_sigmoid_backward(op); break;
        case CompiledOpType::MoETopKBackward:      dispatch_moe_topk_backward(op); break;
        case CompiledOpType::MoEPermuteBackward:   dispatch_moe_permute_backward(op); break;
        case CompiledOpType::MoEGroupedGemmBackward: dispatch_moe_grouped_gemm_backward(op); break;
        case CompiledOpType::MoEGroupedGemmGateUpBackward: dispatch_moe_grouped_gemm_gate_up_backward(op); break;
        case CompiledOpType::MoEGroupedGemmDownBackward: dispatch_moe_grouped_gemm_down_backward(op); break;
        case CompiledOpType::MoEUnpermuteBackward: dispatch_moe_unpermute_backward(op); break;
        case CompiledOpType::MoEExpertBiasAddBackward: dispatch_moe_expert_bias_add_backward(op); break;
        // Expert Parallelism backward operations
        case CompiledOpType::EpDispatchBackward:   dispatch_ep_dispatch_backward(op); break;
        case CompiledOpType::EpCombineBackward:    dispatch_ep_combine_backward(op); break;
        // Mamba/SSM backward operations
        case CompiledOpType::MambaSplitProjBackward: dispatch_mamba_split_proj_backward(op); break;
        case CompiledOpType::MambaConv1dBackward:  dispatch_mamba_conv1d_backward(op); break;
        case CompiledOpType::MambaSplitConvOutBackward: dispatch_mamba_split_conv_out_backward(op); break;
        case CompiledOpType::MambaSsmScanBackward: dispatch_mamba_ssm_scan_backward(op); break;
        case CompiledOpType::MambaGatedRMSNormBackward: dispatch_mamba_gated_rmsnorm_backward(op); break;
        case CompiledOpType::MambaOutProjBackward: dispatch_mamba_out_proj_backward(op, hook); break;
        // Qwen3.5 gated delta rule backward
        case CompiledOpType::ChunkGatedDeltaRuleBackward: dispatch_chunk_gated_delta_rule_backward(op); break;
        case CompiledOpType::Qwen3_5DecayBackward: dispatch_qwen3_5_decay_backward(op); break;
        case CompiledOpType::RepeatInterleaveHeadsBackward: dispatch_repeat_interleave_heads_backward(op); break;
        default: {
            std::ostringstream oss;
            oss << "dispatch_backward_op: unsupported op type "
                << op_type_to_string(op.type) << " (id=" << op.op_id << ")";
            throw std::runtime_error(oss.str());
        }
    }
}

void CompiledExecutor::reset_segment_graphs() {
    auto destroy = [](std::vector<std::vector<SegmentGraphExec>>& layers) {
        for (auto& segs : layers) {
            for (auto& sg : segs) {
                if (sg.exec) {
                    cudaGraphExecDestroy(sg.exec);
                    sg.exec = nullptr;
                }
                sg.checkpoint = {};
            }
        }
    };
    destroy(mFwdSegGraphs);
    destroy(mBwdSegGraphs[0]);
    destroy(mBwdSegGraphs[1]);
    mSegGraphB = 0;
    mSegGraphT = 0;
}

void CompiledExecutor::resize_segment_graphs(const CompiledGraph& fwd_graph,
                                              const CompiledGraph& bwd_graph) {
    auto resize = [](std::vector<std::vector<SegmentGraphExec>>& layers,
                     const std::vector<std::vector<GraphSegment>>& segs) {
        layers.resize(segs.size());
        for (std::size_t L = 0; L < segs.size(); ++L) {
            layers[L].resize(segs[L].size());
        }
    };
    resize(mFwdSegGraphs, fwd_graph.layer_segments);
    resize(mBwdSegGraphs[0], bwd_graph.layer_segments);
    resize(mBwdSegGraphs[1], bwd_graph.layer_segments);
}

}  // namespace dsl
