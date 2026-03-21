// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <array>
#include <cctype>
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
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool is_capturing =
        (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
         capture_status != cudaStreamCaptureStatusNone);
    if (is_capturing) {
        // Host-side offset sync is not capture-safe; callers will fall back to
        // device-offset paths for compact-expert handling.
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
        case CompiledOpType::Ones: return "ones";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::LayerNorm: return "layernorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Transpose: return "transpose";
        case CompiledOpType::Split: return "split";
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

    // Build layer prefix pattern (e.g., "blocks[5]." or "layer5.")
    // Check the first block tensor name to determine naming convention
    std::string layer_prefix;
    for (const auto& [n, _] : mCurrentGraph->tensor_name_to_id) {
        if (n.rfind("layer", 0) == 0 && n.size() > 5 && std::isdigit(n[5])) {
            layer_prefix = "layer" + std::to_string(layer_idx) + ".";
            break;
        }
        if (n.rfind("blocks[", 0) == 0) {
            layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
            break;
        }
    }
    if (layer_prefix.empty()) {
        layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
    }

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

void CompiledExecutor::prepare_saved_buffers_for_capture(
    const std::vector<std::string>& save_list,
    const CompiledGraph* capture_graph) {
    // Only needed when recompute is enabled or MoE tensors require persistence.
    if (!mSaved) {
        return;
    }

    const bool recompute_enabled = mRecomputeEnabled;
    // When forward replay is active, ALL block tensors will be regenerated
    // by replay_layer_forward during backward — no persistent buffers needed.
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Forward replay: all block tensors are replayed, no persistent save needed
        if (forward_replay_active) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(tensor_name, layer_idx, field)) {
                return true;
            }
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
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
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
        // Unknown tensors default to non-persistent to avoid over-allocating
        // during graph-capture preallocation.
        return false;
    };

    const bool debug_save_buffers = (std::getenv("SUROGATE_DEBUG_SAVE_BUFFERS") != nullptr);
    auto ensure_buffer = [&](const std::string& name, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            if (debug_save_buffers) {
                std::cerr << "[SAVE-BUF] alloc name=" << name
                          << " bytes=" << bytes
                          << " old_bytes=" << (buf_it == mMoeSavedBuffers.end() ? 0 : mMoeSavedSizes[name])
                          << std::endl;
            }
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            void* new_buffer = nullptr;
            cudaError_t alloc_err = cudaMalloc(&new_buffer, bytes);
            if (alloc_err != cudaSuccess) {
                std::size_t free_bytes = 0;
                std::size_t total_bytes = 0;
                (void)cudaMemGetInfo(&free_bytes, &total_bytes);
                std::ostringstream oss;
                oss << "CompiledExecutor::prepare_saved_buffers_for_capture: cudaMalloc failed for saved tensor '"
                    << name << "' (" << bytes << " bytes, "
                    << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB)"
                    << ", free=" << static_cast<double>(free_bytes) / (1024.0 * 1024.0) << " MiB"
                    << ", total=" << static_cast<double>(total_bytes) / (1024.0 * 1024.0) << " MiB"
                    << ", error=" << cudaGetErrorString(alloc_err);
                throw std::runtime_error(oss.str());
            }
            mMoeSavedBuffers[name] = new_buffer;
            mMoeSavedSizes[name] = bytes;
        }
    };

    const CompiledGraph* graph_for_lookup = capture_graph ? capture_graph : mCurrentGraph;
    auto resolve_source = [&](const std::string& name) -> std::optional<Tensor> {
        if (graph_for_lookup) {
            int tid = graph_for_lookup->find_tensor_id(name);
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
            return false;
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

    auto infer_graph_ref_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
        if (!graph) {
            return false;
        }
        auto ref_bytes = [&](const TensorRef& ref, std::size_t& bytes_out) -> bool {
            if (ref.name != name || ref.shape.empty()) {
                return false;
            }
            const std::size_t nelem = shape_nelem(ref.shape);
            if (nelem == 0) {
                return false;
            }
            bytes_out = nelem * static_cast<std::size_t>(get_dtype_size(ref.dtype));
            return bytes_out > 0;
        };
        // Prefer producer shape (outputs), then fallback to consumer refs (inputs).
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.outputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.inputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        return false;
    };

    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const std::string base = strip_ssa_suffix(field);
        return (base == "swiglu" || base == "swiglu_flat" ||
                base == "att" || base == "att_flat");
    };

    for (const auto& name : save_list) {
        if (mWeights.has(name)) {
            continue;
        }

        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" ||
             name == "ln_final_rstd" || name == "residual_final" || name == "final_residual" ||
             force_lora_hook);
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
        // persist_saved_layer_tensors() during execute_forward. Pre-allocate
        // buffers for those only when they can actually require persistence.
        int layer_idx = -1;
        std::string field;
        const bool is_block_tensor = parse_block_param(name, layer_idx, field);
        if (!need_persist && (!is_block_tensor || prefer_live)) {
            continue;
        }

        auto src_opt = resolve_source(name);
        // For capture safety, preallocate block save tensors even when they
        // appear non-stack-backed at prep time. Runtime dispatch may route
        // them through stack-backed storage depending on recompute/layout.
        if (src_opt.has_value() && src_opt->Data) {
            ensure_buffer(name, src_opt->bytes());
            continue;
        }
        std::size_t inferred_bytes = 0;
        if (infer_block_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        if (infer_graph_ref_bytes(name, inferred_bytes)) {
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

    // Some ops persist internal saved tensors with op-scoped names that are not
    // in forward.save (e.g., mamba_gated_rmsnorm saves "<op_id>.rstd"/".normed").
    // Pre-allocate those buffers before capture to avoid cudaMalloc in dispatch.
    const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
    if (graph) {
        for (const auto& op : graph->ops) {
            if (op.type != CompiledOpType::MambaGatedRMSNorm || op.op_id.empty()) {
                continue;
            }
            if (op.inputs.empty() || op.inputs[0].shape.empty()) {
                continue;
            }
            const auto& x_ref = op.inputs[0];
            const auto x_elems = shape_nelem(x_ref.shape);
            if (x_elems == 0) {
                continue;
            }

            const std::size_t normed_bytes =
                x_elems * static_cast<std::size_t>(get_dtype_size(x_ref.dtype));
            ensure_buffer(op.op_id + ".normed", normed_bytes);

            const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
            long rows = x_ref.shape[0];
            if (x_ref.shape.size() >= 3) {
                rows *= x_ref.shape[1];
            }
            if (rows > 0 && groups > 0) {
                const std::size_t rstd_bytes =
                    static_cast<std::size_t>(rows) *
                    static_cast<std::size_t>(groups) * sizeof(float);
                ensure_buffer(op.op_id + ".rstd", rstd_bytes);
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

    // Recompute is only active when explicitly enabled for this execution.
    // This gate is set by GraphExecutor after validating runtime options + plan.
    const bool recompute_enabled = mRecomputeEnabled;
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (force_persist) {
            return false;
        }
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            // When forward replay is active, ALL block tensors will be regenerated
            if (forward_replay_active) {
                return true;
            }
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
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
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
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
        // Unknown tensors default to non-persistent unless explicitly forced.
        return false;
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

        const bool src_stack_backed = src.Data && mRunState.Stack.owns(src.Data);
        const bool need_persist =
            (should_persist(name, prefer_live, force_persist_name) || src_stack_backed) &&
            src.Data != nullptr;
        if (need_persist && src.Data != nullptr) {
            const size_t bytes = src.bytes();
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (capturing) {
                    // During CUDA graph capture we cannot allocate new buffers.
                    // Fall back to metadata-only save so backward can resolve
                    // from live/recomputed tensors instead of aborting capture.
                    Tensor meta = src;
                    meta.Data = nullptr;
                    (*mSaved)[name] = meta;
                    return;
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

    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const std::string base = strip_ssa_suffix(field);
        return (base == "swiglu" || base == "swiglu_flat" ||
                base == "att" || base == "att_flat");
    };

    for (const auto& name : save_list) {
        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" ||
             name == "ln_final_rstd" || name == "residual_final" || name == "final_residual" ||
             force_lora_hook);
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

    if (!ref.name.empty()) {
        auto name_it = mNamedTensors.find(ref.name);
        if (name_it != mNamedTensors.end() && name_it->second.Data) {
            log_tensor(name_it->second, "named");
            return name_it->second;
        }
    }

    // Only check gradient resolution for tensors flagged as gradients at compile time.
    // This avoids calling base_param_from_grad() (string substr + find) for the ~80%
    // of tensors that are activations/weights, not gradients.
    if (ref.is_gradient && !ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() &&
                        shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
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
                    // Metadata-only: resolve from current live tensors first.
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    // As a last resort, reuse cached tensor-id only if it points to
                    // persistent (non-stack) memory; stack pointers can become stale
                    // across layer checkpoint restores under recompute/capture.
                    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data &&
                        !mRunState.Stack.owns(mTensors[static_cast<std::size_t>(tid)].Data)) {
                        return mTensors[static_cast<std::size_t>(tid)];
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
            // Materialize effective tensors on demand (e.g. *_weight_eff = *_weight + 1).
            if (!ref.name.empty() && ref.name.size() > 4 &&
                ref.name.compare(ref.name.size() - 4, 4, "_eff") == 0) {
                const std::string base_name = ref.name.substr(0, ref.name.size() - 4);
                Tensor* base_ptr = nullptr;
                if (mWeights.has(base_name)) {
                    base_ptr = &mWeights.get(base_name);
                } else if (mSaved) {
                    auto it = mSaved->find(base_name);
                    if (it != mSaved->end() && it->second.Data) {
                        base_ptr = &it->second;
                    }
                }
                if (!base_ptr && mCurrentGraph) {
                    const int base_tid = mCurrentGraph->find_tensor_id(base_name);
                    if (base_tid >= 0 && static_cast<std::size_t>(base_tid) < mTensors.size() &&
                        mTensors[static_cast<std::size_t>(base_tid)].Data) {
                        base_ptr = &mTensors[static_cast<std::size_t>(base_tid)];
                    }
                }
                if (base_ptr && base_ptr->Data) {
                    std::vector<long> shape(base_ptr->Sizes.begin(),
                                            base_ptr->Sizes.begin() + base_ptr->Rank);
                    Tensor ones = mRunState.temp_alloc(base_ptr->DType, shape, "ones");
                    Tensor eff = mRunState.temp_alloc(base_ptr->DType, shape, "eff");
                    mTemps.push_back(ones);
                    mTemps.push_back(eff);
                    fill_constant(ones, 1.0f, static_cast<std::size_t>(ones.nelem()), mRunState.MainStream);
                    vector_add_sr(eff, *base_ptr, ones, 1.0f,
                                  static_cast<long>(base_ptr->nelem()), 0, mRunState.MainStream);
                    if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                        mTensors[static_cast<std::size_t>(tid)] = eff;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    throw std::runtime_error(
                        "CompiledExecutor: mapped effective tensor requires valid tensor_id: " + ref.name);
                }
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
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() &&
                        shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
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
                    // Reconstruct using the same naming convention as the source tensor
                    if (parse_name.rfind("layer", 0) == 0) {
                        alias_name = "layer" + std::to_string(layer_idx) + "." + alias_field;
                    } else {
                        alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                    }
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

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape, "t");

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


// ---------------------------------------------------------------------------
// replay_layer_forward — torch-style gradient checkpointing
//
// Re-execute a single layer's compiled forward ops during backward to
// regenerate activations. The data lives on the stack; the caller (backward)
// must restore the stack checkpoint after consuming the data.
// ---------------------------------------------------------------------------
void CompiledExecutor::replay_layer_forward(int layer_idx, long B, long T,
                                            const CompiledGraph& fwd_graph,
                                            const modules::ForwardHook* hook) {
    static const bool debug_replay = std::getenv("SUROGATE_DEBUG_REPLAY") != nullptr;
    if (debug_replay) {
        fprintf(stderr, "[REPLAY] replay_layer_forward layer=%d B=%ld T=%ld\n", layer_idx, B, T);
    }
    // Restore any previous deferred checkpoint before starting a new replay
    if (mHasDeferredReplayCheckpoint) {
        mRunState.Stack.restore(mDeferredReplayCheckpoint);
        if (mTemps.size() > mDeferredReplayTempMark) {
            mTemps.resize(mDeferredReplayTempMark);
        }
        mHasDeferredReplayCheckpoint = false;
    }
    // Free persistent copies from previous replay (backward has consumed them)
    for (void* ptr : mReplayCopiedBuffers) {
        cudaFreeAsync(ptr, mRunState.MainStream);
    }
    mReplayCopiedBuffers.clear();

    // Save current execution state
    const CompiledGraph* saved_graph = mCurrentGraph;
    std::vector<Tensor> saved_tensors;
    std::unordered_map<std::string, Tensor> saved_named_tensors;
    saved_tensors.swap(mTensors);
    saved_named_tensors.swap(mNamedTensors);

    // Set replay mode
    mInReplay = true;
    mReplayLayerIdx = layer_idx;

    // Initialize fresh tensor storage for the forward graph
    mCurrentGraph = &fwd_graph;
    mTensors.assign(static_cast<std::size_t>(fwd_graph.num_tensors), Tensor{});
    mNamedTensors.clear();

    // Bind known inputs
    bind_tensor("token_ids", mRunState.Inputs);
    bind_tensor("position_ids", mRunState.PositionIDs);
    if (mRunState.VisualPosMasks.Data) {
        bind_tensor("visual_pos_masks", mRunState.VisualPosMasks);
    }
    if (mRunState.VisualEmbeds.Data) {
        bind_tensor("visual_embeds", mRunState.VisualEmbeds);
    }
    bind_tensor("x0", mRunState.non_block_activations().encoded);

    // Take stack checkpoint — backward will restore this after consuming replay data
    auto replay_checkpoint = mRunState.Stack.checkpoint();
    auto replay_temp_mark = mTemps.size();

    // Find the op range for this layer
    if (layer_idx < 0 ||
        static_cast<std::size_t>(layer_idx) >= fwd_graph.layer_start_indices.size() ||
        fwd_graph.layer_start_indices[static_cast<std::size_t>(layer_idx)] == SIZE_MAX) {
        // Layer not found in forward graph — restore state and return
        mTensors.swap(saved_tensors);
        mNamedTensors.swap(saved_named_tensors);
        mCurrentGraph = saved_graph;
        mInReplay = false;
        mReplayLayerIdx = -1;
        return;
    }

    const std::size_t start = fwd_graph.layer_start_indices[static_cast<std::size_t>(layer_idx)];
    const std::size_t end = (static_cast<std::size_t>(layer_idx) < fwd_graph.layer_end_indices.size())
                                ? fwd_graph.layer_end_indices[static_cast<std::size_t>(layer_idx)]
                                : fwd_graph.ops.size();

    // Collect tensor IDs produced within this layer's op range
    std::unordered_set<int> produced_ids;
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
        for (const auto& out : fwd_graph.ops[idx].outputs) {
            if (out.tensor_id >= 0) {
                produced_ids.insert(out.tensor_id);
            }
        }
    }

    // Pre-bind external inputs: tensors consumed by this layer but produced before it.
    // These include the layer's input residual, previous block outputs, RoPE freqs, etc.
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
        for (const auto& inp : fwd_graph.ops[idx].inputs) {
            if (inp.tensor_id < 0) continue;
            if (produced_ids.count(inp.tensor_id)) continue;
            // Already bound?
            if (static_cast<std::size_t>(inp.tensor_id) < mTensors.size() && mTensors[inp.tensor_id].Data) continue;

            // Try to resolve from known sources
            Tensor resolved{};

            // Check slot type first
            switch (inp.slot) {
                case TensorSlot::FreqCis:
                    resolved = mRunState.non_block_activations().freq_cis;
                    break;
                case TensorSlot::Encoded:
                    resolved = mRunState.non_block_activations().encoded;
                    break;
                case TensorSlot::TokenIDs:
                    resolved = mRunState.Inputs;
                    break;
                case TensorSlot::PositionIDs:
                    resolved = mRunState.PositionIDs;
                    break;
                default:
                    break;
            }

            // If not resolved by slot, try by name
            if (!resolved.Data && !inp.name.empty()) {
                int lyr = -1;
                std::string field;
                if (parse_block_param(inp.name, lyr, field)) {
                    const std::string base = strip_ssa_suffix(field);
                    if (base == "res_ffn" || base == "residual_ffn" || base == "res_in") {
                        resolved = mRunState.get_residual(lyr, mRunState.MainStream);
                    } else {
                        auto& acts = mRunState.simplified_acts(lyr);
                        if (base == "mlp_down" || base == "mlp_down_flat") resolved = acts.mlp_down;
                        else if (base == "res_att" || base == "residual_att") resolved = acts.residual_att;
                        else if (base == "att_out" || base == "att_out_flat") resolved = acts.att_out;
                    }
                } else {
                    // Global tensors by name
                    if (inp.name.find("freq_cis") != std::string::npos ||
                        inp.name.find("rope_freqs") != std::string::npos) {
                        resolved = mRunState.non_block_activations().freq_cis;
                    }
                }

                // Cross-layer connector tensors: "layerN.field" (used by HybridStackedBlocks)
                // These are inter-block connectors with a neutral naming prefix to avoid
                // the block-activation resolver. Resolve them the same way as blocks[N].field.
                if (!resolved.Data && inp.name.rfind("layer", 0) == 0) {
                    auto dot = inp.name.find('.');
                    if (dot != std::string::npos) {
                        try {
                            int cross_lyr = std::stoi(inp.name.substr(5, dot - 5));
                            std::string cross_field = strip_ssa_suffix(inp.name.substr(dot + 1));
                            if (cross_field == "res_ffn" || cross_field == "residual_ffn" || cross_field == "res_in") {
                                resolved = mRunState.get_residual(cross_lyr, mRunState.MainStream);
                            } else {
                                auto& acts = mRunState.simplified_acts(cross_lyr);
                                if (cross_field == "out" || cross_field == "out_flat" ||
                                    cross_field == "mlp_down" || cross_field == "mlp_down_flat") resolved = acts.mlp_down;
                                else if (cross_field == "res_att" || cross_field == "residual_att") resolved = acts.residual_att;
                                else if (cross_field == "att_out" || cross_field == "att_out_flat") resolved = acts.att_out;
                                else if (cross_field == "ln1" || cross_field == "ln1_flat" ||
                                         cross_field == "ln" || cross_field == "ln_flat") resolved = acts.ln1;
                                else if (cross_field == "ln2" || cross_field == "ln2_flat") resolved = acts.ln2;
                                else if (cross_field == "qkv" || cross_field == "qkv_norm") resolved = acts.qkv;
                                else if (cross_field == "qkv_rope") resolved = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                                else if (cross_field == "att" || cross_field == "att_flat") resolved = acts.att;
                                else if (cross_field == "mlp_up" || cross_field == "mlp_up_flat") resolved = acts.mlp_up;
                                else if (cross_field == "swiglu") resolved = acts.swiglu;
                            }
                        } catch (...) {}
                    }
                }
            }

            // For layer 0 input: the "zeros" residual
            if (!resolved.Data && !inp.name.empty() && inp.name.find("zeros") != std::string::npos) {
                // Allocate a zeros tensor on stack for the initial residual
                long C = static_cast<long>(mConfig.HiddenSize);
                resolved = mRunState.temp_alloc(ETensorDType::BF16, {mB, mT, C}, "zeros");
                fill_zero(resolved, mRunState.MainStream);
            }

            // Embedding output (embed_1, embed_0, etc.)
            if (!resolved.Data && !inp.name.empty() && inp.name.find("embed") != std::string::npos) {
                resolved = mRunState.non_block_activations().encoded;
            }

            // Last resort: check mSaved (forward saved tensors)
            if (!resolved.Data && mSaved && !inp.name.empty()) {
                auto it = mSaved->find(inp.name);
                if (it != mSaved->end() && it->second.Data) {
                    resolved = it->second;
                }
            }

            // Very last resort: check backward graph's named tensors
            if (!resolved.Data && !inp.name.empty()) {
                auto it = saved_named_tensors.find(inp.name);
                if (it != saved_named_tensors.end() && it->second.Data) {
                    resolved = it->second;
                }
            }

            if (resolved.Data) {
                store_tensor(inp, resolved);
                if (!inp.name.empty()) {
                    mNamedTensors[inp.name] = resolved;
                }
            }
        }
    }

    // Replay the layer's forward ops.
    // Tile MLP during replay to avoid allocating full-size intermediates (swiglu_flat,
    // mlp_up). The tiled backward will recompute these per-chunk during backward execution.
    std::unordered_map<std::size_t, const MlpTileGroup*> replay_tile_groups;
    for (const auto& tg : fwd_graph.mlp_tile_groups) {
        if (tg.start_op_idx >= start && tg.end_op_idx <= end) {
            replay_tile_groups[tg.start_op_idx] = &tg;
        }
    }
    for (std::size_t idx = start; idx <= end && idx < fwd_graph.ops.size(); ++idx) {
        const auto& op = fwd_graph.ops[idx];

        // Skip loss ops — these should never be replayed
        if (op.type == CompiledOpType::CrossEntropyLoss ||
            op.type == CompiledOpType::FusedLMHeadLoss) {
            continue;
        }

        // Tile MLP during replay — produces correct output without full intermediates.
        // The backward tiled execution will recompute intermediates per-chunk.
        if (!replay_tile_groups.empty()) {
            auto tg_it = replay_tile_groups.find(idx);
            if (tg_it != replay_tile_groups.end()) {
                execute_tiled_mlp(fwd_graph, *tg_it->second, B, T, hook);
                idx = tg_it->second->end_op_idx;
                continue;
            }
        }

        try {
            switch (op.type) {
                case CompiledOpType::Embedding:           dispatch_embedding(op); break;
                case CompiledOpType::Zeros:               dispatch_zeros(op); break;
                case CompiledOpType::Ones:                dispatch_ones(op); break;
                case CompiledOpType::FusedResidualRMSNorm: dispatch_fused_residual_rmsnorm(op); break;
                case CompiledOpType::LayerNorm:           dispatch_layernorm(op); break;
                case CompiledOpType::View:                dispatch_view(op); break;
                case CompiledOpType::Transpose:           dispatch_transpose(op); break;
                case CompiledOpType::Split:               dispatch_split(op); break;
                case CompiledOpType::Concat:              dispatch_concat(op); break;
                case CompiledOpType::Add:                 dispatch_add(op); break;
                case CompiledOpType::Matmul:
                case CompiledOpType::MatmulBias:          dispatch_matmul(op, hook); break;
                case CompiledOpType::BiasAdd:             dispatch_bias_add(op); break;
                case CompiledOpType::SwiGLU:              dispatch_swiglu(op); break;
                case CompiledOpType::GptOssMoeAct:        dispatch_gpt_oss_moe_act(op); break;
                case CompiledOpType::Silu:                dispatch_silu(op); break;
                case CompiledOpType::Gelu:                dispatch_gelu(op); break;
                case CompiledOpType::Relu2:               dispatch_relu2(op); break;
                case CompiledOpType::Mul:                 dispatch_mul(op); break;
                case CompiledOpType::MaskScatter:         dispatch_mask_scatter(op); break;
                case CompiledOpType::DeepstackInject:     dispatch_deepstack_inject(op); break;
                case CompiledOpType::MatmulSwiGLU:        dispatch_matmul_swiglu(op, hook); break;
                case CompiledOpType::QKVQKNorm:           dispatch_qkv_qk_norm(op); break;
                case CompiledOpType::QKVQKNormRoPE:       dispatch_qkv_qk_norm_rope(op); break;
                case CompiledOpType::MRoPE:               dispatch_mrope(op); break;
                case CompiledOpType::RoPE:                dispatch_rope(op); break;
                case CompiledOpType::FlashAttention:       dispatch_flash_attention(op); break;
                // MoE operations
                case CompiledOpType::MoESoftmax:          dispatch_moe_softmax(op); break;
                case CompiledOpType::MoESigmoid:          dispatch_moe_sigmoid(op); break;
                case CompiledOpType::MoETopK:             dispatch_moe_topk(op); break;
                case CompiledOpType::MoEPermute:          dispatch_moe_permute(op); break;
                case CompiledOpType::MoEGroupedGemm:      dispatch_moe_grouped_gemm(op); break;
                case CompiledOpType::MoEGroupedGemmGateUp: dispatch_moe_grouped_gemm_gate_up(op); break;
                case CompiledOpType::MoEGroupedGemmDown:  dispatch_moe_grouped_gemm_down(op); break;
                case CompiledOpType::MoEUnpermute:        dispatch_moe_unpermute(op); break;
                case CompiledOpType::MoEExpertBiasAdd:    dispatch_moe_expert_bias_add(op); break;
                // EP operations
                case CompiledOpType::EpDispatch:          dispatch_ep_dispatch(op); break;
                case CompiledOpType::EpCombine:           dispatch_ep_combine(op); break;
                // Mamba/SSM operations
                case CompiledOpType::MambaSplitProj:      dispatch_mamba_split_proj(op); break;
                case CompiledOpType::MambaConv1d:         dispatch_mamba_conv1d(op); break;
                case CompiledOpType::MambaSplitConvOut:   dispatch_mamba_split_conv_out(op); break;
                case CompiledOpType::MambaSsmScan:        dispatch_mamba_ssm_scan(op); break;
                case CompiledOpType::MambaGatedRMSNorm:   dispatch_mamba_gated_rmsnorm(op); break;
                case CompiledOpType::MambaOutProj:        dispatch_mamba_out_proj(op, hook); break;
                // Qwen3.5 gated delta rule
                case CompiledOpType::ChunkGatedDeltaRule: dispatch_chunk_gated_delta_rule(op); break;
                case CompiledOpType::Qwen3_5Decay:        dispatch_qwen3_5_decay(op); break;
                case CompiledOpType::RepeatInterleaveHeads: dispatch_repeat_interleave_heads(op); break;
                default: break;  // Skip unknown ops
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "replay_layer_forward layer=" << layer_idx
                << " op=" << (idx - start) << " (type=" << op_type_to_string(op.type)
                << "): " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    // Persist replayed tensors into mSaved — save stack pointers directly (no D2D copy).
    // The stack will stay live until backward consumes the data.
    if (mSaved && mSaveList) {
        for (const auto& name : *mSaveList) {
            {
                int lyr_check = -1;
                std::string fld_check;
                if (!parse_block_param(name, lyr_check, fld_check) || lyr_check != layer_idx) continue;
            }

            // Try to find the tensor from the replayed forward graph
            int tid = fwd_graph.find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                (*mSaved)[name] = mTensors[tid];
                continue;
            }
            // Try SSA-stripped lookup
            auto ssa_it = fwd_graph.ssa_base_to_id.find(name);
            if (ssa_it != fwd_graph.ssa_base_to_id.end()) {
                int sid = ssa_it->second;
                if (sid >= 0 && static_cast<std::size_t>(sid) < mTensors.size() && mTensors[sid].Data) {
                    (*mSaved)[name] = mTensors[sid];
                    continue;
                }
            }
            // Fallback: resolve from simplified_acts (for tensors that live in pre-allocated buffers)
            int lyr = -1;
            std::string field;
            if (parse_block_param(name, lyr, field)) {
                const std::string base = strip_ssa_suffix(field);
                auto& acts = mRunState.simplified_acts(lyr);
                Tensor resolved{};
                if (base == "ln1_rstd" || base == "ln_rstd") resolved = acts.ln1_rstd;
                else if (base == "ln2_rstd") resolved = acts.ln2_rstd;
                else if (base == "q_rstd") resolved = acts.q_rstd;
                else if (base == "k_rstd") resolved = acts.k_rstd;
                else if (base == "lse") resolved = acts.lse;
                else if (base == "att" || base == "att_flat") resolved = acts.att;
                else if (base == "ln1" || base == "ln1_flat" || base == "ln" || base == "ln_flat") resolved = acts.ln1;
                else if (base == "ln2" || base == "ln2_flat") resolved = acts.ln2;
                else if (base == "qkv" || base == "qkv_norm") resolved = acts.qkv;
                else if (base == "qkv_rope") resolved = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                else if (base == "att_out" || base == "att_out_flat") resolved = acts.att_out;
                else if (base == "mlp_up" || base == "mlp_up_flat") resolved = acts.mlp_up;
                else if (base == "swiglu") resolved = acts.swiglu;
                else if (base == "mlp_down" || base == "mlp_down_flat") resolved = acts.mlp_down;
                else if (base == "res_att" || base == "residual_att") resolved = acts.residual_att;
                else if (base == "res_ffn" || base == "residual_ffn" || base == "res_in") {
                    resolved = mRunState.get_residual(lyr, mRunState.MainStream);
                }
                if (resolved.Data) {
                    (*mSaved)[name] = resolved;
                }
            }
        }
    }

    // Restore tensor storage — backward graph uses its own namespace
    mTensors.swap(saved_tensors);
    mNamedTensors.swap(saved_named_tensors);
    mCurrentGraph = saved_graph;
    mInReplay = false;
    mReplayLayerIdx = -1;

    // Eagerly restore the stack checkpoint by first copying any stack-resident
    // saved tensors to persistent GPU memory. This frees the replay's stack
    // allocation before backward ops start allocating their temps, preventing
    // stack OOM on memory-hungry backward ops (e.g., gated delta rule).
    if (mSaved) {
        for (auto& [name, tensor] : *mSaved) {
            if (!tensor.Data) continue;
            if (!mRunState.Stack.owns(tensor.Data)) continue;
            // This tensor lives on the stack — copy to persistent memory
            const std::size_t bytes = tensor.bytes();
            if (bytes == 0) continue;
            void* persistent = nullptr;
            CUDA_CHECK(cudaMallocAsync(&persistent, bytes, mRunState.MainStream));
            CUDA_CHECK(cudaMemcpyAsync(persistent, tensor.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            tensor.Data = static_cast<std::byte*>(persistent);
            mReplayCopiedBuffers.push_back(persistent);
        }
    }
    // Now safe to restore — stack-resident data has been copied
    mRunState.Stack.restore(replay_checkpoint);
    if (mTemps.size() > replay_temp_mark) {
        mTemps.resize(replay_temp_mark);
    }
    mHasDeferredReplayCheckpoint = false;

    // Debug: dump saved tensor states after replay
    if (debug_replay && mSaved) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        int null_count = 0, live_count = 0;
        for (const auto& [sname, stensor] : *mSaved) {
            int lyr = -1; std::string fld;
            if (parse_block_param(sname, lyr, fld) && lyr == layer_idx) {
                if (stensor.Data) {
                    live_count++;
                } else {
                    null_count++;
                    if (debug_replay) fprintf(stderr, "[REPLAY] layer=%d saved NULL: %s\n", layer_idx, sname.c_str());
                }
            }
        }
        if (debug_replay) fprintf(stderr, "[REPLAY] layer=%d saved stats: live=%d null=%d\n", layer_idx, live_count, null_count);
    }
}


// ============================================================================
// Tiled MLP execution for long-context mode
// ============================================================================
// Executes an MLP tile group (view→matmul→view→swiglu→view→matmul→view) in
// chunks along the flattened B*T dimension. Reduces peak MLP activation memory
// from O(B*T * intermediate) to O(chunk_size * intermediate).
//
// Operates in flat 2D space to avoid shape mismatches with compiled view ops:
//   chunk_input[chunk, C] → matmul → up[chunk, MUp] → swiglu → act[chunk, M]
//   → matmul → down[chunk, C] → write into full_output
void CompiledExecutor::execute_tiled_mlp(const CompiledGraph& graph,
                                         const MlpTileGroup& group,
                                         long B, long T,
                                         const modules::ForwardHook* hook) {
    // The first op is a view: [B,T,C] → [B*T,C]. Get the 3D input.
    const auto& first_view_op = graph.ops[group.start_op_idx];
    Tensor full_input_3d = resolve_tensor(first_view_op.inputs[0]);
    const long BT = B * T;
    const long C = full_input_3d.Sizes[full_input_3d.Rank - 1];
    Tensor full_input = (full_input_3d.Rank == 3) ? view_tensor(full_input_3d, {BT, C}) : full_input_3d;

    // Find up-weight and down-weight from the matmul ops in the group
    Tensor up_weight{};
    Tensor down_weight{};
    const CompiledOp* up_matmul_op = nullptr;
    const CompiledOp* down_matmul_op = nullptr;
    for (std::size_t idx = group.start_op_idx; idx <= group.end_op_idx; ++idx) {
        const auto& op = graph.ops[idx];
        if (op.type != CompiledOpType::Matmul && op.type != CompiledOpType::MatmulBias) continue;
        for (const auto& inp : op.inputs) {
            if (inp.name.size() >= 13 &&
                inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                up_weight = resolve_tensor(inp);
                up_matmul_op = &op;
            }
            if (inp.name.size() >= 15 &&
                inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                down_weight = resolve_tensor(inp);
                down_matmul_op = &op;
            }
        }
    }
    if (!up_weight.Data || !down_weight.Data || !up_matmul_op || !down_matmul_op) {
        throw std::runtime_error("execute_tiled_mlp: could not find MLP weights in tile group");
    }

    const long MUp = up_weight.Sizes[0];   // [MUp, C] with NT transpose
    const long M = MUp / 2;                // SwiGLU halves the dimension
    const ETensorDType dtype = full_input.DType;

    // Pre-allocate full output [B*T, C]
    Tensor full_output = mRunState.temp_alloc(dtype, {BT, C}, "tiled_mlp_out");

    // Chunk size: min(B*T, C) — Unsloth's "arctic" strategy
    const long chunk_size = std::min(BT, C);
    const long num_chunks = (BT + chunk_size - 1) / chunk_size;

    for (long chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const long offset = chunk_idx * chunk_size;
        const long N = std::min(chunk_size, BT - offset);
        const std::size_t byte_offset = static_cast<std::size_t>(offset) * static_cast<std::size_t>(C) * static_cast<std::size_t>(get_dtype_size(dtype));

        // Narrow input: [N, C]
        Tensor chunk_in = full_input;
        chunk_in.Data = static_cast<std::byte*>(full_input.Data) + byte_offset;
        chunk_in.Sizes[0] = N;

        // Save stack checkpoint for chunk intermediates
        auto ckpt = mRunState.Stack.checkpoint();
        auto temp_mark = mTemps.size();

        // 1) Up-proj matmul: [N, C] × [MUp, C]^T → [N, MUp]
        Tensor up_out = mRunState.temp_alloc(dtype, {N, MUp}, "up_out");
        mTemps.push_back(up_out);
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(chunk_in, up_weight, up_matmul_op->attrs.transpose, mm_M, mm_N, mm_K);
            matmul(up_out, up_weight, chunk_in, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(up_matmul_op->attrs.transpose),
                   false, mRunState.MainStream);
        }

        // 2) SwiGLU: [N, MUp] → [N, M] (operate in 2D, swiglu_forward handles it as B=1, T=N)
        Tensor act_out = mRunState.temp_alloc(dtype, {N, M}, "act_out");
        mTemps.push_back(act_out);
        swiglu_forward(act_out, up_out, nullptr, 1, static_cast<int>(N), static_cast<int>(M),
                       mRunState.MainStream);

        // 3) Down-proj matmul: [N, M] × [C, M]^T → [N, C]
        //    Write directly into the full_output at the correct offset
        Tensor chunk_out = full_output;
        chunk_out.Data = static_cast<std::byte*>(full_output.Data) + byte_offset;
        chunk_out.Sizes[0] = N;
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(act_out, down_weight, down_matmul_op->attrs.transpose, mm_M, mm_N, mm_K);
            matmul(chunk_out, down_weight, act_out, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(down_matmul_op->attrs.transpose),
                   false, mRunState.MainStream);
        }

        // Restore stack — free chunk intermediates (up_out, act_out)
        mRunState.Stack.restore(ckpt);
        if (mTemps.size() > temp_mark) {
            mTemps.resize(temp_mark);
        }
    }

    // Store results under the tensor IDs that subsequent ops expect:
    // - down_matmul output [B*T, C]
    store_tensor(down_matmul_op->outputs[0], full_output);
    mTemps.push_back(full_output);
    // - last view output [B, T, C]
    const auto& last_view_op = graph.ops[group.end_op_idx];
    Tensor full_output_3d = view_tensor(full_output, {B, T, C});
    store_tensor(last_view_op.outputs[0], full_output_3d);
    // - first view output [B*T, C] (for any references to ln_flat)
    store_tensor(first_view_op.outputs[0], full_input);
    // - intermediate tensor IDs (mlp_up, swiglu, etc.) are not needed
    //   since they only live within the tile group
}

// ============================================================================
// Tiled MLP Backward: combined forward recompute + backward per chunk
// ============================================================================
//
// Instead of running the full backward ops (which need full-size intermediates
// from replay), we recompute the forward MLP per-chunk and immediately compute
// gradients, then free intermediates. This reduces memory from O(B*T * intermediate)
// to O(chunk_size * intermediate) during backward.
//
// Backward MLP group structure (reverse order):
//   view_bwd → matmul_bwd(down) → view_bwd → swiglu_bwd → view_bwd → matmul_bwd(up) → view_bwd
//
// We replace all ops in the group with per-chunk:
//   1. Forward recompute: up_out = ln2_chunk @ up_weight^T, act_out = swiglu(up_out)
//   2. Backward down-proj: d_act = d_out_chunk @ down_weight, d_down_weight += act_out^T @ d_out_chunk
//   3. Backward swiglu: d_up = swiglu_backward(d_act, up_out)
//   4. Backward up-proj: d_ln2_chunk = d_up @ up_weight, d_up_weight += ln2_chunk^T @ d_up
void CompiledExecutor::execute_tiled_mlp_backward(const CompiledGraph& bwd_graph,
                                                   const MlpTileGroup& group,
                                                   long B, long T,
                                                   const modules::BackwardHook* hook) {
    const long BT = B * T;

    // Find the matmul_backward ops and extract weight/gradient references
    const CompiledOp* down_bwd_op = nullptr;
    const CompiledOp* up_bwd_op = nullptr;
    for (std::size_t idx = group.start_op_idx; idx <= group.end_op_idx; ++idx) {
        const auto& op = bwd_graph.ops[idx];
        if (op.type != CompiledOpType::MatmulBackward) continue;
        for (const auto& inp : op.inputs) {
            if (inp.name.size() >= 15 &&
                inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                down_bwd_op = &op;
            }
            if (inp.name.size() >= 13 &&
                inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                up_bwd_op = &op;
            }
        }
    }
    if (!down_bwd_op || !up_bwd_op) {
        throw std::runtime_error("execute_tiled_mlp_backward: could not find MLP backward ops");
    }

    // Resolve inputs from backward ops:
    // down_bwd inputs: [d_out (d_mlp_down_flat), activation(swiglu_flat), down_weight]
    // up_bwd inputs: [d_mlp_up_flat, activation(ln2_flat), up_weight]
    //
    // The group may or may not start/end with ViewBackward ops. The actual structure is:
    //   [ViewBackward?] MatmulBackward(down) [View] SwiGLUBackward [View] MatmulBackward(up) [ViewBackward?]
    // We skip all ops in the group and compute everything from raw kernel calls.

    // If the group starts with a ViewBackward, dispatch it to produce the flat gradient
    const auto& first_op = bwd_graph.ops[group.start_op_idx];
    if (first_op.type == CompiledOpType::ViewBackward) {
        dispatch_view_backward(first_op);
    }

    // Get the incoming gradient (d_out for down-proj backward) — already available from prior ops
    Tensor d_mlp_out_flat = resolve_tensor(down_bwd_op->inputs[0]);

    // Get ln2_flat from the up-proj backward's activation input (input[1])
    Tensor ln2_flat = resolve_tensor(up_bwd_op->inputs[1]);
    // Get weights
    Tensor up_weight = resolve_tensor(up_bwd_op->inputs[2]);
    Tensor down_weight = resolve_tensor(down_bwd_op->inputs[2]);

    const long C = ln2_flat.Sizes[ln2_flat.Rank - 1];
    const long MUp = up_weight.Sizes[0];  // [MUp, C] with NT
    const long M = MUp / 2;               // SwiGLU halves
    const ETensorDType dtype = ln2_flat.DType;
    const EMMTranspose fwd_mode = down_bwd_op->attrs.transpose;  // NT for standard matmul
    const int layer_idx = down_bwd_op->attrs.layer_idx;

    // Allocate full d_ln2_flat [B*T, C] output
    Tensor d_ln2_flat = mRunState.temp_alloc(dtype, {BT, C}, "d_ln2_flat");
    fill_zero(d_ln2_flat, mRunState.MainStream);
    mTemps.push_back(d_ln2_flat);

    // Resolve weight gradient tensors (pre-allocated by gradient store)
    // down_bwd outputs: [d_swiglu_flat, d_down_weight]
    // up_bwd outputs: [d_ln2_flat, d_up_weight]
    // Check if weight gradients should be computed (frozen in LoRA)
    auto get_weight_grad = [&](const CompiledOp& op, int out_idx) -> Tensor* {
        if (static_cast<int>(op.outputs.size()) <= out_idx) return nullptr;
        const std::string& dB_name = op.outputs[out_idx].name;
        if (dB_name.empty()) return nullptr;
        std::string weight_name = dB_name;
        if (weight_name.rfind("d_", 0) == 0) weight_name = weight_name.substr(2);
        bool accum = false;
        return mGrads.get_param_grad(weight_name, accum);
    };
    Tensor* d_down_weight_ptr = get_weight_grad(*down_bwd_op, 1);
    Tensor* d_up_weight_ptr = get_weight_grad(*up_bwd_op, 1);

    // Determine accumulation: first chunk uses existing flag, subsequent chunks always accumulate
    const std::string& dB_down_name = down_bwd_op->outputs.size() > 1 ? down_bwd_op->outputs[1].name : "";
    const std::string& dB_up_name = up_bwd_op->outputs.size() > 1 ? up_bwd_op->outputs[1].name : "";
    bool base_accumulate_down = mAccumulateTensors.count(dB_down_name) > 0;
    bool base_accumulate_up = mAccumulateTensors.count(dB_up_name) > 0;

    // Compute backward transpose modes for NT forward
    EMMTranspose mode_dA = EMMTranspose::NN;  // dA = d_out @ B (NN for NT forward)
    EMMTranspose mode_dB_rm = EMMTranspose::TN;  // dB = d_out^T @ A (TN for NT forward)

    // Chunk size: min(B*T, C) — same as forward
    const long chunk_size = std::min(BT, C);
    const long num_chunks = (BT + chunk_size - 1) / chunk_size;

    // Tiled backward invokes LoRA hooks with chunk-sized tensors. The hook logic
    // reads rs.B/rs.T to derive BT, so temporarily switch to (1, chunk_tokens)
    // around hook invocation to avoid over-reading chunk buffers.
    auto invoke_chunk_lora_hook = [&](const std::optional<modules::BackwardHookPoint>& point,
                                      bool accumulate,
                                      long chunk_tokens,
                                      bool force_chunk_ln2_input) {
        if (!(hook && *hook) || !point.has_value() || layer_idx < 0) {
            return;
        }

        const long prev_B = mRunState.B;
        const long prev_T = mRunState.T;
        Tensor prev_recompute_ln{};
        Tensor prev_recompute_rstd{};
        bool patched_recompute_ln = false;

        // Up-proj LoRA hook can recompute LN2 from full residual when recompute is
        // enabled. For tiled execution we must use the chunk LN2 tensor we just set.
        if (force_chunk_ln2_input && mLoRARunState && mLoRARunState->recompute_ln.Data) {
            prev_recompute_ln = mLoRARunState->recompute_ln;
            prev_recompute_rstd = mLoRARunState->recompute_rstd;
            mLoRARunState->recompute_ln.Data = nullptr;
            mLoRARunState->recompute_rstd.Data = nullptr;
            patched_recompute_ln = true;
        }

        mRunState.B = 1;
        mRunState.T = chunk_tokens;

        try {
            (*hook)(layer_idx, accumulate, mRunState.MainStream, *point, mHookContext);
        } catch (...) {
            mRunState.B = prev_B;
            mRunState.T = prev_T;
            if (patched_recompute_ln) {
                mLoRARunState->recompute_ln = prev_recompute_ln;
                mLoRARunState->recompute_rstd = prev_recompute_rstd;
            }
            throw;
        }

        mRunState.B = prev_B;
        mRunState.T = prev_T;
        if (patched_recompute_ln) {
            mLoRARunState->recompute_ln = prev_recompute_ln;
            mLoRARunState->recompute_rstd = prev_recompute_rstd;
        }
    };

    for (long chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const long offset = chunk_idx * chunk_size;
        const long N = std::min(chunk_size, BT - offset);

        auto ckpt = mRunState.Stack.checkpoint();

        // Narrow input tensors to chunk
        auto narrow = [&](const Tensor& full, long cols) -> Tensor {
            Tensor chunk = full;
            const std::size_t byte_off = static_cast<std::size_t>(offset) * static_cast<std::size_t>(cols) *
                                         static_cast<std::size_t>(get_dtype_size(dtype));
            chunk.Data = static_cast<std::byte*>(full.Data) + byte_off;
            chunk.Sizes[0] = N;
            return chunk;
        };

        Tensor ln2_chunk = narrow(ln2_flat, C);
        Tensor d_out_chunk = narrow(d_mlp_out_flat, C);
        Tensor d_ln2_chunk = narrow(d_ln2_flat, C);

        // ---- Forward recompute for this chunk ----
        // 1) Up-proj: up_out = ln2_chunk @ up_weight^T → [N, MUp]
        Tensor up_out = mRunState.temp_alloc(dtype, {N, MUp}, "up_out");
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(ln2_chunk, up_weight, fwd_mode, mm_M, mm_N, mm_K);
            matmul(up_out, up_weight, ln2_chunk, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(fwd_mode), false, mRunState.MainStream);
        }

        // 2) SwiGLU: act_out = swiglu(up_out) → [N, M]
        Tensor act_out = mRunState.temp_alloc(dtype, {N, M}, "act_out");
        swiglu_forward(act_out, up_out, nullptr, 1, static_cast<int>(N), static_cast<int>(M),
                       mRunState.MainStream);

        // ---- Backward for this chunk ----
        // 3) Down-proj backward: dA = d_out_chunk @ down_weight (activation grad)
        Tensor d_act = mRunState.temp_alloc(dtype, {N, M}, "d_act");
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_out_chunk, down_weight, mode_dA, mm_M, mm_N, mm_K);
            matmul(d_act, down_weight, d_out_chunk, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(mode_dA), false, mRunState.MainStream);
        }
        // dB = d_out_chunk^T @ act_out (weight grad, accumulated)
        if (d_down_weight_ptr && d_down_weight_ptr->Data) {
            bool accum = base_accumulate_down || (chunk_idx > 0);
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_out_chunk, act_out, mode_dB_rm, mm_M, mm_N, mm_K);
            matmul(*d_down_weight_ptr, act_out, d_out_chunk, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(mode_dB_rm), accum, mRunState.MainStream);
        }

        // LoRA backward hook for down-proj
        if (hook && *hook && down_bwd_op->attrs.backward_hook_point.has_value() && layer_idx >= 0) {
            bool lora_accum = base_accumulate_down || (chunk_idx > 0);
            auto& grads = mRunState.simplified_grads(layer_idx);
            auto& acts = mRunState.simplified_acts(layer_idx);
            // Save full-size tensor state
            auto prev_d_swiglu = grads.d_swiglu;
            auto prev_d_res_ffn = grads.d_res_ffn;
            auto prev_a_swiglu = acts.swiglu;
            auto prev_a_mlp_up = acts.mlp_up;
            // Set chunk-sized tensors with correct shapes
            // d_swiglu: the activation gradient [N, M]
            grads.d_swiglu = d_act;
            // d_res_ffn: the incoming gradient d_out [N, C]
            grads.d_res_ffn = d_out_chunk;
            // acts.swiglu: the forward activation [N, M]
            acts.swiglu = act_out;
            // acts.mlp_up: the pre-activation [N, MUp] (needed for swiglu recompute)
            acts.mlp_up = up_out;
            invoke_chunk_lora_hook(down_bwd_op->attrs.backward_hook_point,
                                   lora_accum,
                                   N,
                                   false);
            // Restore full-size tensors
            grads.d_swiglu = prev_d_swiglu;
            grads.d_res_ffn = prev_d_res_ffn;
            acts.swiglu = prev_a_swiglu;
            acts.mlp_up = prev_a_mlp_up;
        }

        // 4) SwiGLU backward: d_up = swiglu_backward(d_act, up_out) → [N, MUp]
        Tensor d_up = mRunState.temp_alloc(dtype, {N, MUp}, "d_up");
        swiglu_backward(d_up, d_act, up_out, nullptr,
                        1, static_cast<int>(N), static_cast<int>(M), mRunState.MainStream);

        // 5) Up-proj backward: dA = d_up @ up_weight (activation grad → d_ln2_chunk)
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_up, up_weight, mode_dA, mm_M, mm_N, mm_K);
            matmul(d_ln2_chunk, up_weight, d_up, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(mode_dA), false, mRunState.MainStream);
        }
        // dB = d_up^T @ ln2_chunk (weight grad, accumulated)
        if (d_up_weight_ptr && d_up_weight_ptr->Data) {
            bool accum = base_accumulate_up || (chunk_idx > 0);
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_up, ln2_chunk, mode_dB_rm, mm_M, mm_N, mm_K);
            matmul(*d_up_weight_ptr, ln2_chunk, d_up, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   mm_N, mm_M, mm_K, swap_transpose(mode_dB_rm), accum, mRunState.MainStream);
        }

        // LoRA backward hook for up-proj
        if (hook && *hook && up_bwd_op->attrs.backward_hook_point.has_value() && layer_idx >= 0) {
            bool lora_accum = base_accumulate_up || (chunk_idx > 0);
            auto& grads = mRunState.simplified_grads(layer_idx);
            auto& acts = mRunState.simplified_acts(layer_idx);
            // Save full-size tensor state
            auto prev_d_ln2 = grads.d_ln2;
            auto prev_d_mlp_up = grads.d_mlp_up;
            auto prev_a_ln2 = acts.ln2;
            // Set chunk-sized tensors with correct shapes
            grads.d_ln2 = d_ln2_chunk;
            grads.d_mlp_up = d_up;
            acts.ln2 = ln2_chunk;
            invoke_chunk_lora_hook(up_bwd_op->attrs.backward_hook_point,
                                   lora_accum,
                                   N,
                                   true);
            // Restore full-size tensors
            grads.d_ln2 = prev_d_ln2;
            grads.d_mlp_up = prev_d_mlp_up;
            acts.ln2 = prev_a_ln2;
        }

        // Free chunk intermediates
        mRunState.Stack.restore(ckpt);
    }

    // Store outputs at tensor IDs the subsequent ops expect.
    // If the group ends with a ViewBackward, simulate its reshape.
    // Otherwise store directly at the up-proj backward output.
    const auto& last_op = bwd_graph.ops[group.end_op_idx];
    if (last_op.type == CompiledOpType::ViewBackward) {
        // The ViewBackward reshapes d_ln2_flat [B*T,C] → d_ln2 [B,T,C]
        Tensor d_ln2_3d = view_tensor(d_ln2_flat, {B, T, C});
        store_tensor(last_op.outputs[0], d_ln2_3d);
    }
    // Always store the flat version at the up-proj backward output ID
    store_tensor(up_bwd_op->outputs[0], d_ln2_flat);

    // Store weight gradients at backward op output tensor IDs
    if (d_down_weight_ptr && d_down_weight_ptr->Data && down_bwd_op->outputs.size() > 1) {
        store_tensor(down_bwd_op->outputs[1], *d_down_weight_ptr);
    }
    if (d_up_weight_ptr && d_up_weight_ptr->Data && up_bwd_op->outputs.size() > 1) {
        store_tensor(up_bwd_op->outputs[1], *d_up_weight_ptr);
    }

    // Also store the intermediate gradient tensor IDs that other backward ops might reference
    // d_swiglu_flat (down_bwd output[0]) and d_mlp_up_flat (from swiglu backward) are NOT needed
    // outside the tiled group — they're consumed within the MLP backward sequence.
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
    mNamedTensors.clear();
    mCurrentLayer = -1;
    mSegmentDispatchedUntil = 0;

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
    const bool recompute_enabled_flag = mRecomputeEnabled;
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

    // When forward replay is active, ALL block tensors will be regenerated by
    // replay_layer_forward during backward. Save metadata only — no D2D copies needed.
    const bool forward_replay_active = recompute_enabled_flag && static_cast<bool>(mRecomputeFn);

    auto name_belongs_to_layer = [](const std::string& name, int target_layer) -> bool {
        int lyr = -1;
        std::string fld;
        return parse_block_param(name, lyr, fld) && lyr == target_layer;
    };

    auto persist_saved_layer_tensors = [&](int layer_idx) {
        if (!mSaved || !mSaveList) {
            return;
        }

        // When forward replay is active, save metadata only for ALL block tensors.
        // Replay will regenerate them during backward.
        if (forward_replay_active) {
            for (const auto& name : *mSaveList) {
                if (!name_belongs_to_layer(name, layer_idx)) continue;
                if (mSaved->find(name) != mSaved->end()) continue;
                // Save an empty metadata entry — replay_layer_forward will fill in real data
                Tensor meta{};
                (*mSaved)[name] = meta;
            }
            return;
        }

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
            if (!name_belongs_to_layer(name, layer_idx)) {
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
    const int debug_nonfinite_mode = env_int("SUROGATE_DEBUG_CHECK_NONFINITE", 0);
    const bool debug_nonfinite_forward = (debug_nonfinite_mode & 0x1) != 0;
    const char* watch_tensor_env = std::getenv("SUROGATE_DEBUG_WATCH_TENSOR");
    const std::string watch_tensor_name = watch_tensor_env ? std::string(watch_tensor_env) : std::string();
    const bool watch_tensor_enabled = !watch_tensor_name.empty();
    const float watch_amax_delta = env_float("SUROGATE_DEBUG_WATCH_AMAX_DELTA", 1.0f);
    const float watch_alarm_amax = env_float("SUROGATE_DEBUG_WATCH_ALARM_AMAX", 1e6f);
    const bool watch_abort_on_alarm = env_int("SUROGATE_DEBUG_WATCH_ABORT", 0) != 0;
    int watch_tensor_id = -1;
    if (watch_tensor_enabled && mCurrentGraph) {
        const int tid = mCurrentGraph->find_tensor_id(watch_tensor_name);
        if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
            watch_tensor_id = tid;
        }
    }
    if (watch_tensor_enabled && watch_tensor_id < 0) {
        for (const auto& scan_op : graph.ops) {
            bool found = false;
            for (const auto& ref : scan_op.inputs) {
                if (ref.name == watch_tensor_name && ref.tensor_id >= 0 &&
                    static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
                    watch_tensor_id = ref.tensor_id;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
            for (const auto& ref : scan_op.outputs) {
                if (ref.name == watch_tensor_name && ref.tensor_id >= 0 &&
                    static_cast<std::size_t>(ref.tensor_id) < mTensors.size()) {
                    watch_tensor_id = ref.tensor_id;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
    }
    if (watch_tensor_enabled) {
        int watch_input_refs = 0;
        int watch_output_refs = 0;
        for (const auto& scan_op : graph.ops) {
            for (const auto& ref : scan_op.inputs) {
                if (ref.name == watch_tensor_name) {
                    watch_input_refs++;
                }
            }
            for (const auto& ref : scan_op.outputs) {
                if (ref.name == watch_tensor_name) {
                    watch_output_refs++;
                }
            }
        }
        std::cerr << "[WATCH_META] tensor='" << watch_tensor_name
                  << "' tensor_id=" << watch_tensor_id
                  << " input_refs=" << watch_input_refs
                  << " output_refs=" << watch_output_refs
                  << std::endl;
    }
    auto try_bind_watch_tensor_id_from_ref = [&](const TensorRef& ref) {
        if (watch_tensor_id >= 0) {
            return;
        }
        if (ref.name != watch_tensor_name) {
            return;
        }
        if (ref.tensor_id < 0) {
            return;
        }
        if (static_cast<std::size_t>(ref.tensor_id) >= mTensors.size()) {
            return;
        }
        watch_tensor_id = ref.tensor_id;
    };
    auto try_get_watch_tensor = [&](const CompiledOp* op_ctx = nullptr) -> const Tensor* {
        if (!watch_tensor_enabled) {
            return nullptr;
        }
        if (op_ctx) {
            for (const auto& ref : op_ctx->inputs) {
                try_bind_watch_tensor_id_from_ref(ref);
            }
            for (const auto& ref : op_ctx->outputs) {
                try_bind_watch_tensor_id_from_ref(ref);
            }
        }
        if (watch_tensor_id >= 0 &&
            static_cast<std::size_t>(watch_tensor_id) < mTensors.size() &&
            mTensors[static_cast<std::size_t>(watch_tensor_id)].Data) {
            return &mTensors[static_cast<std::size_t>(watch_tensor_id)];
        }
        if (mWeights.has(watch_tensor_name)) {
            Tensor& w = mWeights.get(watch_tensor_name);
            if (w.Data) {
                return &w;
            }
        }
        if (const Tensor* direct = try_get_tensor(watch_tensor_name)) {
            return direct;
        }
        return try_get_tensor_fuzzy(watch_tensor_name);
    };
    auto watch_non_finite_count = [&](const Tensor& t) -> int {
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) {
            return -1;
        }
        Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
        CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
        count_non_finite(non_finite_count, t, mRunState.MainStream);
        int host_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                   non_finite_count.get<int>(),
                                   sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(non_finite_count);
        return host_count;
    };
    auto watch_absmax = [&](const Tensor& t) -> float {
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) {
            return -1.0f;
        }
        Tensor amax = mRunState.temp_alloc(ETensorDType::FP32, {1}, "amax");
        CUDA_CHECK(cudaMemsetAsync(amax.Data, 0, sizeof(float), mRunState.MainStream));
        global_amax(amax.get<float>(), t, static_cast<std::size_t>(t.nelem()), mRunState.DeviceProp, mRunState.MainStream);
        float host_amax = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&host_amax,
                                   amax.get<float>(),
                                   sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(amax);
        return host_amax;
    };
    auto check_nonfinite_refs = [&](const CompiledOp& op, const std::vector<TensorRef>& refs) {
        if (!debug_nonfinite_forward) {
            return;
        }
        for (const auto& ref : refs) {
            if (ref.name.empty()) {
                continue;
            }
            const Tensor* t = try_get_tensor(ref.name);
            if (!t || !t->Data) {
                continue;
            }
            if (t->DType != ETensorDType::BF16 && t->DType != ETensorDType::FP32) {
                continue;
            }

            Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
            CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
            count_non_finite(non_finite_count, *t, mRunState.MainStream);
            int host_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                       non_finite_count.get<int>(),
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            mRunState.temp_free(non_finite_count);

            if (host_count > 0) {
                std::ostringstream oss;
                oss << "Non-finite detected in forward output tensor '" << ref.name
                    << "' at op id=" << op.op_id
                    << " type=" << op_type_to_string(op.type)
                    << " count=" << host_count
                    << " dtype=" << static_cast<int>(t->DType)
                    << " shape=[";
                for (int d = 0; d < t->Rank; ++d) {
                    if (d > 0) oss << ",";
                    oss << t->Sizes[d];
                }
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }
    };
    // Build tile group lookup for long-context tiled MLP execution
    std::unordered_map<std::size_t, const MlpTileGroup*> tile_group_starts;
    for (const auto& tg : graph.mlp_tile_groups) {
        tile_group_starts[tg.start_op_idx] = &tg;
    }
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        // Check if this op starts a tiled MLP group
        if (!tile_group_starts.empty()) {
            auto tg_it = tile_group_starts.find(idx);
            if (tg_it != tile_group_starts.end()) {
                const auto& tg = *tg_it->second;
                // Handle layer start if the first op has one
                const auto& first_op = graph.ops[tg.start_op_idx];
                if (first_op.layer_start >= 0) {
                    if (first_op.layer_start < num_layers &&
                        !layer_active[static_cast<std::size_t>(first_op.layer_start)]) {
                        layer_checkpoints[static_cast<std::size_t>(first_op.layer_start)] = mRunState.Stack.checkpoint();
                        layer_temp_marks[static_cast<std::size_t>(first_op.layer_start)] = mTemps.size();
                        layer_active[static_cast<std::size_t>(first_op.layer_start)] = 1;
                    }
                    handle_layer_start(first_op.layer_start);
                }
                execute_tiled_mlp(graph, tg, mB, mT, hook);
                // Handle layer end if the last op has one
                const auto& last_op = graph.ops[tg.end_op_idx];
                if (last_op.layer_end >= 0) {
                    if (last_op.layer_end < num_layers &&
                        layer_active[static_cast<std::size_t>(last_op.layer_end)]) {
                        if (mDebugDumpLayerFn) mDebugDumpLayerFn(last_op.layer_end);
                        persist_saved_layer_tensors(last_op.layer_end);
                        mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(last_op.layer_end)]);
                        if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(last_op.layer_end)]) {
                            mTemps.resize(layer_temp_marks[static_cast<std::size_t>(last_op.layer_end)]);
                        }
                        prune_stack_tensors();
                        layer_active[static_cast<std::size_t>(last_op.layer_end)] = 0;
                    }
                    handle_layer_end(last_op.layer_end);
                }
                idx = tg.end_op_idx;  // skip past the tile group (loop will ++idx)
                continue;
            }
        }

        const auto& op = graph.ops[idx];

        bool watch_pre_valid = false;
        int watch_pre_nf = -1;
        float watch_pre_amax = -1.0f;
        if (watch_tensor_enabled) {
            if (const Tensor* wt = try_get_watch_tensor(&op)) {
                if (wt->Data && (wt->DType == ETensorDType::BF16 || wt->DType == ETensorDType::FP32)) {
                    watch_pre_nf = watch_non_finite_count(*wt);
                    watch_pre_amax = watch_absmax(*wt);
                    watch_pre_valid = true;
                }
            }
        }

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

            // Split-attention graph mode: pre-dispatch all layer ops via segments.
            // Non-attention segments are captured/replayed as CUDA graphs;
            // graph-breaking ops run eagerly. The normal loop still iterates
            // through these ops for layer_end handling, tensor persistence, and
            // pruning — but skips the per-op dispatch (already done here).
            if (mSplitAttentionGraphs &&
                !graph.layer_segments.empty() &&
                static_cast<std::size_t>(op.layer_start) < graph.layer_segments.size() &&
                !graph.layer_segments[static_cast<std::size_t>(op.layer_start)].empty()) {
                const int L = op.layer_start;
                const auto& segs = graph.layer_segments[static_cast<std::size_t>(L)];
                for (std::size_t s = 0; s < segs.size(); ++s) {
                    const auto& seg = segs[s];
                    if (seg.eager) {
                        // Check if this eager segment matches an MLP tile group.
                        // compute_layer_segments emits these as single eager segments.
                        const MlpTileGroup* tile_group = nullptr;
                        for (const auto& tg : graph.mlp_tile_groups) {
                            if (tg.start_op_idx == seg.start_op) {
                                tile_group = &tg;
                                break;
                            }
                        }
                        if (tile_group) {
                            execute_tiled_mlp(graph, *tile_group, mB, mT, hook);
                        } else {
                            for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                dispatch_forward_op(graph.ops[i], hook);
                            }
                        }
                    } else {
                        auto& sg = mFwdSegGraphs[static_cast<std::size_t>(L)][s];
                        const bool is_capture = (sg.exec == nullptr);
                        // Track mSaved entries before segment to detect new ones
                        std::size_t saved_before = mSaved ? mSaved->size() : 0;
                        auto run = [&]() {
                            for (std::size_t i = seg.start_op; i < seg.end_op; ++i) {
                                dispatch_forward_op(graph.ops[i], hook);
                            }
                        };
                        trace_or_execute_cuda_graph_with_stack(
                            run, mRunState.MainStream, sg.exec, true,
                            mRunState.Stack, sg.checkpoint);
                        if (is_capture) {
                            // Save post-dispatch stack state. On replay,
                            // trace_or_execute restores to sg.checkpoint (pre-alloc)
                            // before launching the graph. We must then advance
                            // past the graph's stack allocations so the next
                            // segment doesn't overlap.
                            sg.post_checkpoint = mRunState.Stack.checkpoint();

                            // Snapshot all tensor entries after capture. On replay
                            // dispatch doesn't run so mTensors/mNamedTensors/mSaved
                            // wouldn't be populated.
                            sg.tensor_snapshot.clear();
                            sg.named_snapshot.clear();
                            sg.saved_snapshot.clear();
                            for (int tid = 0; tid < static_cast<int>(mTensors.size()); ++tid) {
                                if (mTensors[tid].Data) {
                                    sg.tensor_snapshot.emplace_back(tid, mTensors[tid]);
                                }
                            }
                            for (const auto& [name, t] : mNamedTensors) {
                                if (t.Data) {
                                    sg.named_snapshot.emplace_back(name, t);
                                }
                            }
                            // Snapshot mSaved entries added by dispatch (e.g.
                            // MambaGatedRMSNorm writes rstd directly to mSaved)
                            if (mSaved && mSaved->size() > saved_before) {
                                for (const auto& [name, t] : *mSaved) {
                                    sg.saved_snapshot.emplace_back(name, t);
                                }
                            }
                        } else {
                            // Advance stack past graph's allocations so the next
                            // segment (eager attention) doesn't overlap.
                            mRunState.Stack.restore(sg.post_checkpoint);

                            // Restore tensor/saved entries from capture snapshot
                            for (const auto& [tid, t] : sg.tensor_snapshot) {
                                if (static_cast<std::size_t>(tid) < mTensors.size()) {
                                    mTensors[tid] = t;
                                }
                            }
                            for (const auto& [name, t] : sg.named_snapshot) {
                                mNamedTensors[name] = t;
                            }
                            if (mSaved) {
                                for (const auto& [name, t] : sg.saved_snapshot) {
                                    if (mSaved->find(name) == mSaved->end()) {
                                        (*mSaved)[name] = t;
                                    }
                                }
                            }
                        }
                    }
                }
                // Mark: layer ops already dispatched. The normal loop will still
                // iterate through them for layer_end handling but skip dispatch.
                mSegmentDispatchedUntil = graph.layer_end_indices[static_cast<std::size_t>(L)];
            }
        }

        // Skip dispatch for ops already handled by split-attention segment execution.
        // The normal loop still runs for these ops to handle layer_end boundaries,
        // tensor persistence, pruning, etc.
        if (idx < mSegmentDispatchedUntil) {
            goto skip_dispatch;
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
                case CompiledOpType::Ones:
                    dispatch_ones(op);
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
                case CompiledOpType::Transpose:
                    dispatch_transpose(op);
                    break;
                case CompiledOpType::Split:
                    dispatch_split(op);
                    break;
                case CompiledOpType::Concat:
                    dispatch_concat(op);
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
                // Qwen3.5 gated delta rule forward operations
                case CompiledOpType::ChunkGatedDeltaRule:
                    dispatch_chunk_gated_delta_rule(op);
                    break;
                case CompiledOpType::Qwen3_5Decay:
                    dispatch_qwen3_5_decay(op);
                    break;
                case CompiledOpType::RepeatInterleaveHeads:
                    dispatch_repeat_interleave_heads(op);
                    break;
                default:
                    throw std::runtime_error("CompiledExecutor: unsupported forward op type");
            }
            check_nonfinite_refs(op, op.outputs);
            if (watch_tensor_enabled) {
                bool watch_post_valid = false;
                int watch_post_nf = -1;
                float watch_post_amax = -1.0f;
                if (const Tensor* wt = try_get_watch_tensor(&op)) {
                    if (wt->Data && (wt->DType == ETensorDType::BF16 || wt->DType == ETensorDType::FP32)) {
                        watch_post_nf = watch_non_finite_count(*wt);
                        watch_post_amax = watch_absmax(*wt);
                        watch_post_valid = true;
                    }
                }

                const bool became_invalid = watch_pre_valid && !watch_post_valid;
                const bool became_valid = !watch_pre_valid && watch_post_valid;
                const bool nf_changed = watch_pre_valid && watch_post_valid && watch_pre_nf != watch_post_nf;
                const bool amax_changed = watch_pre_valid && watch_post_valid &&
                                          std::fabs(watch_post_amax - watch_pre_amax) > watch_amax_delta;
                const bool alarm = watch_post_valid &&
                    (watch_post_nf > 0 || !std::isfinite(watch_post_amax) || watch_post_amax >= watch_alarm_amax);

                if (became_invalid || became_valid || nf_changed || amax_changed || alarm) {
                    std::ostringstream in_list;
                    for (std::size_t ri = 0; ri < op.inputs.size(); ++ri) {
                        if (ri > 0) in_list << ",";
                        in_list << op.inputs[ri].name << "#" << op.inputs[ri].tensor_id;
                    }
                    std::ostringstream out_list;
                    for (std::size_t ro = 0; ro < op.outputs.size(); ++ro) {
                        if (ro > 0) out_list << ",";
                        out_list << op.outputs[ro].name << "#" << op.outputs[ro].tensor_id;
                    }
                    std::cerr << "[WATCH] tensor='" << watch_tensor_name
                              << "' op_idx=" << idx
                              << " op_id=" << op.op_id
                              << " type=" << op_type_to_string(op.type)
                              << " pre_valid=" << (watch_pre_valid ? 1 : 0)
                              << " post_valid=" << (watch_post_valid ? 1 : 0)
                              << " pre_nf=" << watch_pre_nf
                              << " post_nf=" << watch_post_nf
                              << " pre_amax=" << watch_pre_amax
                              << " post_amax=" << watch_post_amax
                              << " inputs=[" << in_list.str() << "]"
                              << " outputs=[" << out_list.str() << "]"
                              << std::endl;
                }
                if (alarm && watch_abort_on_alarm) {
                    std::ostringstream oss_watch;
                    oss_watch << "Watch tensor '" << watch_tensor_name
                              << "' alarm after op idx=" << idx
                              << " id=" << op.op_id
                              << " type=" << op_type_to_string(op.type)
                              << " nf=" << watch_post_nf
                              << " amax=" << watch_post_amax;
                    throw std::runtime_error(oss_watch.str());
                }
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

        skip_dispatch:

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

    // Dump requested non-block tensors (e.g. xF/residual_final) after forward.
    // Per-layer block dumps are handled in the layer_end callback above.
    if (mDebugDumpFn) {
        static const char* dump_tensors_env = std::getenv("SUROGATE_DEBUG_DUMP_TENSORS");
        if (dump_tensors_env && *dump_tensors_env) {
            std::vector<std::string> names;
            {
                std::string token;
                std::stringstream ss(dump_tensors_env);
                while (std::getline(ss, token, ',')) {
                    // trim ASCII whitespace
                    std::size_t b = 0;
                    while (b < token.size() &&
                           std::isspace(static_cast<unsigned char>(token[b]))) {
                        ++b;
                    }
                    std::size_t e = token.size();
                    while (e > b &&
                           std::isspace(static_cast<unsigned char>(token[e - 1]))) {
                        --e;
                    }
                    if (e > b) {
                        names.emplace_back(token.substr(b, e - b));
                    }
                }
            }
            std::vector<std::string> global_names;
            global_names.reserve(names.size());
            for (const auto& name : names) {
                if (name.rfind("blocks[", 0) != 0) {
                    global_names.push_back(name);
                }
            }
            if (!global_names.empty()) {
                cudaStreamCaptureStatus dump_capture_status = cudaStreamCaptureStatusNone;
                const bool dump_capturing =
                    (cudaStreamIsCapturing(mRunState.MainStream, &dump_capture_status) == cudaSuccess &&
                     dump_capture_status != cudaStreamCaptureStatusNone);
                if (!dump_capturing) {
                    mDebugDumpFn(global_names, -1);
                }
            }
        }
    }

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
    mNamedTensors.clear();
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
    static const bool debug_replay = std::getenv("SUROGATE_DEBUG_REPLAY") != nullptr;
    const char* op_trace_env = std::getenv("SUROGATE_OP_TRACE");
    const bool op_trace = op_trace_env && std::string(op_trace_env) != "0";
    const char* op_profile_env = std::getenv("SUROGATE_OP_PROFILE");
    const bool op_profile = op_profile_env && std::string(op_profile_env) != "0";
    const int debug_nonfinite_mode = env_int("SUROGATE_DEBUG_CHECK_NONFINITE", 0);
    const bool debug_nonfinite_backward = (debug_nonfinite_mode & 0x2) != 0;
    auto check_nonfinite_refs = [&](const CompiledOp& op, const std::vector<TensorRef>& refs) {
        if (!debug_nonfinite_backward) {
            return;
        }
        for (const auto& ref : refs) {
            if (ref.name.empty()) {
                continue;
            }
            const Tensor* t = try_get_tensor(ref.name);
            if (!t || !t->Data) {
                continue;
            }
            if (t->DType != ETensorDType::BF16 && t->DType != ETensorDType::FP32) {
                continue;
            }

            Tensor non_finite_count = mRunState.temp_alloc(ETensorDType::INT32, {1}, "non_finite_count");
            CUDA_CHECK(cudaMemsetAsync(non_finite_count.Data, 0, sizeof(int), mRunState.MainStream));
            count_non_finite(non_finite_count, *t, mRunState.MainStream);
            int host_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(&host_count,
                                       non_finite_count.get<int>(),
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       mRunState.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            mRunState.temp_free(non_finite_count);

            if (host_count > 0) {
                std::ostringstream oss;
                oss << "Non-finite detected in backward output tensor '" << ref.name
                    << "' at op id=" << op.op_id
                    << " type=" << op_type_to_string(op.type)
                    << " count=" << host_count
                    << " dtype=" << static_cast<int>(t->DType)
                    << " shape=[";
                for (int d = 0; d < t->Rank; ++d) {
                    if (d > 0) oss << ",";
                    oss << t->Sizes[d];
                }
                oss << "]";
                throw std::runtime_error(oss.str());
            }
        }
    };
    std::unordered_map<std::string, double> op_profile_total_ms;
    std::unordered_map<std::string, std::size_t> op_profile_counts;
    cudaEvent_t op_profile_start = nullptr;
    cudaEvent_t op_profile_end = nullptr;
    if (op_profile) {
        CUDA_CHECK(cudaEventCreateWithFlags(&op_profile_start, cudaEventDefault));
        CUDA_CHECK(cudaEventCreateWithFlags(&op_profile_end, cudaEventDefault));
    }
    cudaStreamCaptureStatus bwd_capture_status = cudaStreamCaptureStatusNone;
    const bool bwd_stream_capturing =
        mCapturing ||
        (cudaStreamIsCapturing(mRunState.MainStream, &bwd_capture_status) == cudaSuccess &&
         bwd_capture_status != cudaStreamCaptureStatusNone);

    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    // Build backward tile group lookup for long-context tiled MLP execution.
    // Only include groups that contain MatmulBackward ops (not forward groups).
    std::unordered_map<std::size_t, const MlpTileGroup*> bwd_tile_group_starts;
    for (const auto& tg : graph.mlp_tile_groups) {
        for (std::size_t i = tg.start_op_idx; i <= tg.end_op_idx && i < graph.ops.size(); ++i) {
            if (graph.ops[i].type == CompiledOpType::MatmulBackward) {
                bwd_tile_group_starts[tg.start_op_idx] = &tg;
                break;
            }
        }
    }
    
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);
        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        // Check if this op starts a backward tiled MLP group
        if (!bwd_tile_group_starts.empty()) {
            auto tg_it = bwd_tile_group_starts.find(idx);
            if (tg_it != bwd_tile_group_starts.end()) {
                const auto& tg = *tg_it->second;
                // Handle layer start/recompute if the first op has one
                const auto& first_op = graph.ops[tg.start_op_idx];
                if (first_op.layer_start >= 0) {
                    handle_layer_start(first_op.layer_start);
                    if (mRecomputeEnabled && mRecomputeFn) {
                        const int layer_idx = first_op.layer_start;
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
                execute_tiled_mlp_backward(graph, tg, mB, mT, hook);
                // Prune tensors for all ops in the group
                for (std::size_t gi = tg.start_op_idx; gi <= tg.end_op_idx; ++gi) {
                    prune_by_last_use(gi);
                }
                idx = tg.end_op_idx;
                continue;
            }
        }

        if (op_profile) {
            CUDA_CHECK(cudaEventRecord(op_profile_start, mRunState.MainStream));
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);
            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                if (debug_replay) {
                    fprintf(stderr, "[BWD] layer_start=%d for op %zu type=%s\n",
                            layer_idx, idx, op_type_to_string(op.type));
                }
                if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                    clear_shared_grads(layer_idx);
                    layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                }
                mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                mLastRecomputeLayer = layer_idx;
            }
        }

            // Note: backward always runs through the normal dispatch loop (no segment
            // graph shortcut) because backward tensor lifetime management (cross-layer
            // persistence, prune_by_last_use, deferred recompute checkpoints) is too
            // complex to replicate in the segmented path. The forward segmented path
            // provides the graph capture benefit; backward runs fully eager.
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            const int layer_idx_any = op_layer_idx_any(op);
            const int effective_layer_idx = (layer_idx >= 0) ? layer_idx : layer_idx_any;
            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                if (debug_replay) {
                    fprintf(stderr, "[BWD] op_layer_detect=%d (non_grad=%d any=%d) for op %zu type=%s\n",
                            effective_layer_idx, layer_idx, layer_idx_any, idx, op_type_to_string(op.type));
                }
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
                case CompiledOpType::Transpose:
                    dispatch_transpose(op);
                    break;
                // Split/concat may appear in backward graphs via autodiff rules.
                case CompiledOpType::Split:
                    dispatch_split(op);
                    break;
                case CompiledOpType::Concat:
                    dispatch_concat(op);
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
                // Ones in backward is a no-op
                case CompiledOpType::Ones:
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

                case CompiledOpType::ChunkGatedDeltaRuleBackward:
                    dispatch_chunk_gated_delta_rule_backward(op);
                    break;
                case CompiledOpType::Qwen3_5DecayBackward:
                    dispatch_qwen3_5_decay_backward(op);
                    break;
                case CompiledOpType::RepeatInterleaveHeadsBackward:
                    dispatch_repeat_interleave_heads_backward(op);
                    break;

                // Mamba forward ops that may appear in backward graph
                case CompiledOpType::MambaSplitProj:
                case CompiledOpType::MambaConv1d:
                case CompiledOpType::MambaSplitConvOut:
                case CompiledOpType::MambaSsmScan:
                case CompiledOpType::MambaGatedRMSNorm:
                case CompiledOpType::MambaOutProj:
                case CompiledOpType::ChunkGatedDeltaRule:
                case CompiledOpType::Qwen3_5Decay:
                case CompiledOpType::RepeatInterleaveHeads:
                    // These forward Mamba/GDN ops may appear in backward graph due to autodiff
                    throw std::runtime_error(
                        "CompiledExecutor: Mamba/GatedDelta forward op in backward graph not yet supported");

                default: {
                    std::ostringstream oss;
                    oss << "CompiledExecutor: unsupported backward op type at idx " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                    throw std::runtime_error(oss.str());
                }
            }
            check_nonfinite_refs(op, op.outputs);

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

            if (op_profile) {
                CUDA_CHECK(cudaEventRecord(op_profile_end, mRunState.MainStream));
                CUDA_CHECK(cudaEventSynchronize(op_profile_end));
                float elapsed_ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, op_profile_start, op_profile_end));
                const std::string op_name = op_type_to_string(op.type);
                op_profile_total_ms[op_name] += static_cast<double>(elapsed_ms);
                op_profile_counts[op_name] += 1;
            }

            // Memory management - prune tensors after last use, then restore stack at layer boundaries.
            // If live cross-layer tensors exist on the stack, persist them to allocated memory first.
            prune_by_last_use(idx);
            if (op.layer_end >= 0 && op.layer_end != last_layer_restored) {
                if (!bwd_stream_capturing) {
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
                            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&persistent), nbytes));
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
                }

                // During CUDA graph capture, we must still restore stack at each layer boundary
                // to bound peak memory; only capture-unsafe work above is skipped.
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

    if (op_profile && !op_profile_total_ms.empty()) {
        std::vector<std::pair<std::string, double>> totals(op_profile_total_ms.begin(), op_profile_total_ms.end());
        std::sort(totals.begin(), totals.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cerr << "[OP PROFILE][backward] totals:\n";
        for (const auto& [name, total_ms] : totals) {
            const std::size_t count = op_profile_counts[name];
            const double avg_ms = count > 0 ? (total_ms / static_cast<double>(count)) : 0.0;
            std::cerr << "  " << name
                      << " total=" << total_ms << "ms"
                      << " count=" << count
                      << " avg=" << avg_ms << "ms\n";
        }
    }
    if (op_profile) {
        if (op_profile_start) cudaEventDestroy(op_profile_start);
        if (op_profile_end) cudaEventDestroy(op_profile_end);
    }

    // Restore any deferred replay checkpoint before final cleanup
    if (mHasDeferredReplayCheckpoint) {
        mRunState.Stack.restore(mDeferredReplayCheckpoint);
        if (mTemps.size() > mDeferredReplayTempMark) {
            mTemps.resize(mDeferredReplayTempMark);
        }
        mHasDeferredReplayCheckpoint = false;
    }

    // Free persistent copies from last replay
    for (void* ptr : mReplayCopiedBuffers) {
        cudaFreeAsync(ptr, mRunState.MainStream);
    }
    mReplayCopiedBuffers.clear();

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
        case CompiledOpType::LayerNorm:          dispatch_layernorm(op); break;
        case CompiledOpType::View:               dispatch_view(op); break;
        case CompiledOpType::Transpose:          dispatch_transpose(op); break;
        case CompiledOpType::Split:              dispatch_split(op); break;
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
        case CompiledOpType::LayerNormBackward:    dispatch_layernorm_backward(op); break;
        case CompiledOpType::EmbeddingBackward:    dispatch_embedding_backward(op); break;
        case CompiledOpType::CrossEntropyLossBackward: dispatch_cross_entropy_loss_backward(op); break;
        case CompiledOpType::FusedLMHeadLossBackward: dispatch_fused_lm_head_loss_backward(op); break;
        // Forward ops that appear in backward graph (autodiff generates these)
        case CompiledOpType::View:                 dispatch_view_backward(op); break;
        case CompiledOpType::Transpose:            dispatch_transpose(op); break;
        case CompiledOpType::Split:                dispatch_split(op); break;
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
