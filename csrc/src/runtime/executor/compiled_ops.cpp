// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "runtime/executor/compiled_ops.h"
#include "runtime/dsl/tensor_slot_dispatch.h"

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
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/ep/ep_strategy.h"
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

namespace {}  // namespace

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

const int* CompiledExecutor::get_or_sync_moe_host_offsets(int layer_idx, const int* device_offsets, int num_experts) {
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
        case CompiledOpType::GeluGlu: return "gelu_glu";
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
        case CompiledOpType::GeluGluBackward: return "gelu_glu_backward";
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
    : mRunState(run_state),
      mWeights(weights),
      mGrads(grads),
      mConfig(config),
      mOptions(options),
      mEpStrategy(ep::create_strategy(options)),
      mEpStates(mEpStrategy->ep_states()),
      mLLEPStates(mEpStrategy->llep_states()),
      mEPLayerMeta(mEpStrategy->layer_meta()) {
    if (const char* env = std::getenv("SUROGATE_ENABLE_SCHEMA_HOOK_DISPATCH")) {
        mSchemaHookDispatchEnabled = (std::string_view(env) == "1");
    }
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
    if (mReplayPersistArena) {
        cudaFree(mReplayPersistArena);
        mReplayPersistArena = nullptr;
        mReplayPersistCapacity = 0;
        mReplayPersistOffset = 0;
    }
    if (mMemEffScratchArena) {
        cudaFree(mMemEffScratchArena);
        mMemEffScratchArena = nullptr;
        mMemEffScratchCapacity = 0;
        mMemEffScratchOffset = 0;
    }

    // Free persistent MoE saved tensor buffers. Arena-backed entries are
    // skipped — their storage is owned by mPhaseArenas.
    for (auto& [name, buffer] : mMoeSavedBuffers) {
        if (!buffer) continue;
        auto ab_it = mMoeSavedArenaBacked.find(name);
        const bool arena_backed = ab_it != mMoeSavedArenaBacked.end() && ab_it->second;
        if (!arena_backed) cudaFree(buffer);
    }
    mMoeSavedBuffers.clear();
    mMoeSavedSizes.clear();
    mMoeSavedArenaBacked.clear();

    // EP per-layer state, LLEP state, shared buffers, buffer pool, and
    // the weight-transfer stream are all owned by mEpStrategy and released
    // by its destructor automatically via unique_ptr cleanup.
}

void CompiledExecutor::clear_replay_copied_refs() {
    // Arena-backed replay persists: bump offset reset so the next replay
    // overwrites this layer's persistent copies. The arena pointer stays
    // stable, matching captured-graph expectations. References to those
    // pointers in mTensors / mNamedTensors / mSaved are scrubbed below.
    const std::byte* arena_begin = mReplayPersistArena;
    const std::byte* arena_end = arena_begin + mReplayPersistOffset;
    mReplayPersistOffset = 0;

    auto points_to_replay_copy = [&](const std::byte* data) -> bool {
        if (!data) return false;
        if (arena_begin && data >= arena_begin && data < arena_end) return true;
        for (void* ptr : mReplayCopiedBuffers) {
            if (data == static_cast<const std::byte*>(ptr)) return true;
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

void CompiledExecutor::set_schema_hook_registry(const HookRegistry* registry) {
    mSchemaHookRegistry = registry;
}

int CompiledExecutor::dispatch_schema_hook(HookEventKind event,
                                           int layer_idx,
                                           std::string_view schema_id,
                                           std::string_view slot_name,
                                           void* payload) {
    if (!mSchemaHookDispatchEnabled || !mSchemaHookRegistry || schema_id.empty() || slot_name.empty()) {
        return 0;
    }
    HookContext context;
    context.layer_idx = layer_idx;
    context.target = HookTarget{std::string(schema_id), std::string(slot_name)};
    context.event = event;
    context.stream = mRunState.MainStream;
    context.payload = payload;
    return mSchemaHookRegistry->dispatch(context);
}

void CompiledExecutor::set_recompute_fn(std::function<void(int, long, long, bool)> fn) {
    mRecomputeFn = std::move(fn);
}

void CompiledExecutor::set_recompute_enabled(bool enabled) {
    mRecomputeEnabled = enabled;
    mLastRecomputeLayer = -1;
}

CompiledExecutor::MoeSavedAlloc CompiledExecutor::allocate_moe_saved(std::size_t nbytes) {
    MoeSavedAlloc result;
    if (nbytes == 0) return result;
    if (mPhaseArenas && mPhaseArenas->allocated && mPhaseArenas->moe_saved_ptr &&
        mMoeSavedBumpOffset + nbytes <= mPhaseArenas->moe_saved_bytes) {
        result.ptr = mPhaseArenas->moe_saved_ptr + mMoeSavedBumpOffset;
        result.arena_backed = true;
        mMoeSavedBumpOffset += nbytes;
        return result;
    }
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, nbytes));
    result.ptr = static_cast<std::byte*>(raw);
    result.arena_backed = false;
    return result;
}

void CompiledExecutor::prepare_replay_persist_arena_for_capture() {
    ensure_replay_persist_arena();
}

void CompiledExecutor::ensure_replay_persist_arena() {
    // 256 MiB is a conservative upper bound — one layer's worth of
    // stack-backed replay persistence (typically 9 slots × up to 8 MiB for
    // (B, T, C) bf16 ≈ 40 MiB) plus margin for larger models. Too-large
    // single requests fall back to caller's cudaMallocAsync. Must be called
    // OUTSIDE any CUDA stream capture — cudaMalloc during capture is illegal.
    if (mReplayPersistArena) return;
    constexpr std::size_t kArenaBytes = 256ull * 1024 * 1024;
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, kArenaBytes));
    mReplayPersistArena = static_cast<std::byte*>(raw);
    mReplayPersistCapacity = kArenaBytes;
}

void CompiledExecutor::prepare_mem_eff_scratch_for_capture(std::size_t bytes) {
    if (bytes == 0) return;
    if (mMemEffScratchArena && mMemEffScratchCapacity >= bytes) return;
    if (mMemEffScratchArena) {
        cudaFree(mMemEffScratchArena);
        mMemEffScratchArena = nullptr;
        mMemEffScratchCapacity = 0;
        mMemEffScratchOffset = 0;
    }
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, bytes));
    mMemEffScratchArena = static_cast<std::byte*>(raw);
    mMemEffScratchCapacity = bytes;
    mMemEffScratchOffset = 0;
}

void CompiledExecutor::mem_eff_scratch_reset() {
    mMemEffScratchOffset = 0;
}

std::byte* CompiledExecutor::mem_eff_scratch_alloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    // Lazy allocation for eager / non-capture paths where
    // prepare_mem_eff_scratch_for_capture was not called first. Must
    // NOT fire during stream capture — that would be an illegal
    // cudaMalloc inside the capture.
    if (!mMemEffScratchArena) {
        std::size_t default_bytes = 1024ull * 1024 * 1024;
        if (const char* env = std::getenv("SUROGATE_MEM_EFF_ARENA_MB")) {
            char* end = nullptr;
            const unsigned long mb = std::strtoul(env, &end, 10);
            if (end != env && mb > 0) {
                default_bytes = static_cast<std::size_t>(mb) * 1024ull * 1024ull;
            }
        }
        prepare_mem_eff_scratch_for_capture(default_bytes);
    }
    constexpr std::size_t kAlign = 128;  // align for 128-bit bf16 loads
    const std::size_t aligned_offset = (mMemEffScratchOffset + kAlign - 1) & ~(kAlign - 1);
    if (aligned_offset + bytes > mMemEffScratchCapacity) {
        std::ostringstream oss;
        oss << "mem_eff scratch arena exhausted: offset=" << aligned_offset << " + req=" << bytes
            << " > cap=" << mMemEffScratchCapacity
            << ". Bump SUROGATE_MEM_EFF_ARENA_MB or prepare_mem_eff_scratch_for_capture size.";
        throw std::runtime_error(oss.str());
    }
    std::byte* p = mMemEffScratchArena + aligned_offset;
    mMemEffScratchOffset = aligned_offset + bytes;
    return p;
}

std::byte* CompiledExecutor::allocate_replay_persist(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    ensure_replay_persist_arena();
    // 16-byte alignment for the next bump (matches CUDA alignment expectations).
    constexpr std::size_t kAlign = 16;
    const std::size_t aligned_offset = (mReplayPersistOffset + kAlign - 1) & ~(kAlign - 1);
    if (aligned_offset + bytes > mReplayPersistCapacity) {
        // Arena exhausted — caller will fall back to cudaMallocAsync.
        return nullptr;
    }
    std::byte* p = mReplayPersistArena + aligned_offset;
    mReplayPersistOffset = aligned_offset + bytes;
    return p;
}

std::byte* CompiledExecutor::allocate_bwd_cross_layer(std::size_t nbytes) {
    if (nbytes == 0) return nullptr;
    if (!mPhaseArenas || !mPhaseArenas->allocated || !mPhaseArenas->bwd_cross_layer_ptr) {
        throw std::runtime_error("allocate_bwd_cross_layer: bwd_cross_layer arena not allocated");
    }
    constexpr std::size_t kAlign = 256;
    nbytes = (nbytes + kAlign - 1) & ~(kAlign - 1);
    auto note_live_alloc = [&](std::byte* ptr) {
        mBwdCrossLayerAllocBytes[ptr] = nbytes;
        mBwdCrossLayerCurrentLiveBytes += nbytes;
        if (mBwdCrossLayerCurrentLiveBytes > mBwdCrossLayerHighWaterBytes) {
            mBwdCrossLayerHighWaterBytes = mBwdCrossLayerCurrentLiveBytes;
        }
    };
    for (auto it = mBwdCrossLayerFreeBlocks.begin(); it != mBwdCrossLayerFreeBlocks.end(); ++it) {
        if (it->bytes < nbytes) continue;
        std::byte* ptr = it->ptr;
        const std::size_t remaining = it->bytes - nbytes;
        if (remaining >= kAlign) {
            it->ptr += nbytes;
            it->bytes = remaining;
        } else {
            mBwdCrossLayerFreeBlocks.erase(it);
        }
        note_live_alloc(ptr);
        return ptr;
    }
    if (mBwdCrossLayerBumpOffset + nbytes <= mPhaseArenas->bwd_cross_layer_bytes) {
        std::byte* ptr = mPhaseArenas->bwd_cross_layer_ptr + mBwdCrossLayerBumpOffset;
        mBwdCrossLayerBumpOffset += nbytes;
        note_live_alloc(ptr);
        return ptr;
    }
    // Arena exhausted. Under CUDA stream capture a cudaMalloc fallback
    // would invalidate the captured graph, so throw with a clear hint —
    // `prepare_bwd_cross_layer_for_capture` should have grown the arena
    // to fit the eager-measured high-water mark before capture began.
    cudaStream_t stream = mRunState.MainStream;
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_status);
    const bool capturing = (cap_status != cudaStreamCaptureStatusNone);
    if (capturing) {
        std::ostringstream oss;
        oss << "allocate_bwd_cross_layer: arena exhausted under CUDA graph capture " << "(req=" << nbytes
            << " bytes, used=" << mBwdCrossLayerBumpOffset << ", cap=" << mPhaseArenas->bwd_cross_layer_bytes
            << "). Set SUROGATE_BWD_CROSS_LAYER_MB above the eager high-water mark.";
        throw std::runtime_error(oss.str());
    }
    // Eager fallback (non-capture): cudaMalloc grows lazily so the user
    // doesn't pay for the worst case upfront. Tracked into the high-water
    // mark so capture preparation can size the arena correctly.
    std::byte* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    mBwdCrossLayerFallbacks.push_back(ptr);
    mBwdCrossLayerCurrentFallbackBytes += nbytes;
    note_live_alloc(ptr);
    return reinterpret_cast<std::byte*>(ptr);
}

void CompiledExecutor::release_bwd_cross_layer(std::byte* ptr) {
    if (!ptr) return;
    auto size_it = mBwdCrossLayerAllocBytes.find(ptr);
    if (size_it == mBwdCrossLayerAllocBytes.end()) return;
    const std::size_t nbytes = size_it->second;
    mBwdCrossLayerAllocBytes.erase(size_it);
    if (mBwdCrossLayerCurrentLiveBytes >= nbytes) {
        mBwdCrossLayerCurrentLiveBytes -= nbytes;
    } else {
        mBwdCrossLayerCurrentLiveBytes = 0;
    }

    const auto* arena_begin = mPhaseArenas ? mPhaseArenas->bwd_cross_layer_ptr : nullptr;
    const auto* arena_end = arena_begin ? arena_begin + mPhaseArenas->bwd_cross_layer_bytes : nullptr;
    if (arena_begin && ptr >= arena_begin && ptr < arena_end) {
        mBwdCrossLayerFreeBlocks.push_back({ptr, nbytes});
        std::sort(mBwdCrossLayerFreeBlocks.begin(),
                  mBwdCrossLayerFreeBlocks.end(),
                  [](const BwdCrossLayerFreeBlock& a, const BwdCrossLayerFreeBlock& b) { return a.ptr < b.ptr; });
        std::vector<BwdCrossLayerFreeBlock> merged;
        merged.reserve(mBwdCrossLayerFreeBlocks.size());
        for (const auto& block : mBwdCrossLayerFreeBlocks) {
            if (!block.ptr || block.bytes == 0) continue;
            if (!merged.empty() && merged.back().ptr + merged.back().bytes == block.ptr) {
                merged.back().bytes += block.bytes;
            } else {
                merged.push_back(block);
            }
        }
        mBwdCrossLayerFreeBlocks.swap(merged);
        return;
    }

    auto fb_it = std::find(mBwdCrossLayerFallbacks.begin(), mBwdCrossLayerFallbacks.end(), ptr);
    if (fb_it != mBwdCrossLayerFallbacks.end()) {
        (void)cudaFree(ptr);
        mBwdCrossLayerFallbacks.erase(fb_it);
        if (mBwdCrossLayerCurrentFallbackBytes >= nbytes) {
            mBwdCrossLayerCurrentFallbackBytes -= nbytes;
        } else {
            mBwdCrossLayerCurrentFallbackBytes = 0;
        }
    }
}

void CompiledExecutor::prepare_bwd_cross_layer_for_capture() {
    if (!mPhaseArenas || !mPhaseArenas->allocated) return;
    // Slack absorbs cross-step variance and free-list fragmentation in the persist
    // pattern. Without slack a small uptick on a later step would
    // exhaust the arena under capture and throw.
    constexpr std::size_t kSlackBytes = 128ULL * 1024 * 1024;
    const std::size_t needed = mBwdCrossLayerHighWaterBytes + kSlackBytes;
    if (needed <= mPhaseArenas->bwd_cross_layer_bytes) return;
    // Free fallback ptrs first (would otherwise leak on next backward
    // entry now that the new arena will absorb their work).
    for (std::byte* p : mBwdCrossLayerFallbacks) {
        if (p) (void)cudaFree(p);
    }
    mBwdCrossLayerFallbacks.clear();
    if (mPhaseArenas->bwd_cross_layer_ptr) {
        CUDA_CHECK(cudaFree(mPhaseArenas->bwd_cross_layer_ptr));
        mPhaseArenas->bwd_cross_layer_ptr = nullptr;
    }
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, needed));
    mPhaseArenas->bwd_cross_layer_ptr = static_cast<std::byte*>(raw);
    mPhaseArenas->bwd_cross_layer_bytes = needed;
    mBwdCrossLayerBumpOffset = 0;
    mBwdCrossLayerCurrentFallbackBytes = 0;
    mBwdCrossLayerCurrentLiveBytes = 0;
    mBwdCrossLayerFreeBlocks.clear();
    mBwdCrossLayerAllocBytes.clear();
}

void CompiledExecutor::set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache) {
    mFP8Cache = cache;
}

void CompiledExecutor::set_fp8_cache_transposed(std::unordered_map<std::string, FP8WeightCacheEntry>* cache_t) {
    mFP8CacheT = cache_t;
}

void CompiledExecutor::set_moe_fp8_cache(std::unordered_map<std::string, MoEFP8WeightCacheEntry>* cache) {
    mMoEFP8Cache = cache;
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

void CompiledExecutor::attach_moe_fp8_cache(modules::MoeMatmulContext& ctx, const std::string& weight_name) {
    const char* env = std::getenv("SUROGATE_FP8_MOE_WGRAD");
    if (!env || std::string_view(env) == "0" || !mMoEFP8Cache || weight_name.empty() || ctx.num_experts <= 0 ||
        ctx.N <= 0 || ctx.K <= 0) {
        return;
    }
    auto it = mMoEFP8Cache->find(weight_name);
    if (it == mMoEFP8Cache->end()) {
        MoEFP8WeightCacheEntry entry{};
        entry.weights_e4m3 = mRunState.Allocator->allocate(ETensorDType::FP8_E4M3,
                                                           ("fp8_moe_cache_" + weight_name).c_str(),
                                                           EAllocationType::ON_DEVICE,
                                                           {ctx.num_experts, ctx.N, ctx.K});
        entry.weight_amax = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                          ("fp8_moe_cache_" + weight_name + "_amax").c_str(),
                                                          EAllocationType::ON_DEVICE,
                                                          {ctx.num_experts});
        entry.weight_scales = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                            ("fp8_moe_cache_" + weight_name + "_scales").c_str(),
                                                            EAllocationType::ON_DEVICE,
                                                            {ctx.num_experts});
        auto [insert_it, _] = mMoEFP8Cache->emplace(weight_name, std::move(entry));
        it = insert_it;
    }
    ctx.cached_moe_weights_e4m3 = &it->second.weights_e4m3;
    ctx.cached_moe_weight_amax = &it->second.weight_amax;
    ctx.cached_moe_weight_scales = &it->second.weight_scales;
    ctx.cached_moe_weights_initialized = &it->second.initialized;
}

void CompiledExecutor::invalidate_moe_fp8_cache(const std::string& weight_name) {
    if (!mMoEFP8Cache || weight_name.empty()) {
        return;
    }
    auto it = mMoEFP8Cache->find(weight_name);
    if (it != mMoEFP8Cache->end()) {
        it->second.initialized = false;
    }
}

bool CompiledExecutor::try_moe_fp8_weight_grad(Tensor& d_weight,
                                               const Tensor& grad_output,
                                               const Tensor& input,
                                               const int* expert_offsets,
                                               int num_experts,
                                               int M,
                                               int N,
                                               const int* host_offsets,
                                               const int* active_experts,
                                               bool weight_is_compact,
                                               int num_active,
                                               float beta,
                                               const char* debug_tag) {
    const char* env = std::getenv("SUROGATE_FP8_MOE_WGRAD");
    if (!env || std::string_view(env) == "0") {
        return false;
    }
    if (grad_output.DType != ETensorDType::BF16 || input.DType != ETensorDType::BF16 ||
        d_weight.DType != ETensorDType::BF16 || !expert_offsets || !host_offsets || !mRunState.CublasLtHandle ||
        num_experts <= 0 || M <= 0 || N <= 0 || grad_output.Rank < 2 || input.Rank < 2 ||
        grad_output.Sizes[0] != input.Sizes[0] || grad_output.Sizes[1] != M || input.Sizes[1] != N ||
        (beta != 0.0f && beta != 1.0f)) {
        return false;
    }

    const long total_tokens = grad_output.Sizes[0];
    try {
        Tensor grad_e5m2 =
            mRunState.temp_alloc(ETensorDType::FP8_E5M2, {total_tokens, static_cast<long>(M)}, "moe_wgrad_dout_e5m2");
        Tensor grad_stats = mRunState.temp_alloc(ETensorDType::FP32, {2}, "moe_wgrad_dout_stats");
        grad_e5m2.Stats = grad_stats.get<float>();
        Tensor input_e4m3 =
            mRunState.temp_alloc(ETensorDType::FP8_E4M3, {total_tokens, static_cast<long>(N)}, "moe_wgrad_input_e4m3");
        Tensor input_stats = mRunState.temp_alloc(ETensorDType::FP32, {2}, "moe_wgrad_input_stats");
        input_e4m3.Stats = input_stats.get<float>();
        mTemps.push_back(grad_e5m2);
        mTemps.push_back(grad_stats);
        mTemps.push_back(input_e4m3);
        mTemps.push_back(input_stats);

        abs_max(grad_e5m2.abs_max(),
                grad_output.get<nv_bfloat16>(),
                total_tokens * static_cast<long>(M),
                mRunState.DeviceProp,
                mRunState.MainStream);
        quantize_with_abs_max(grad_e5m2,
                              grad_e5m2.scale(),
                              grad_output,
                              grad_e5m2.abs_max(),
                              total_tokens * static_cast<long>(M),
                              mRunState.DeviceProp,
                              mRunState.MainStream);
        abs_max(input_e4m3.abs_max(),
                input.get<nv_bfloat16>(),
                total_tokens * static_cast<long>(N),
                mRunState.DeviceProp,
                mRunState.MainStream);
        quantize_with_abs_max(input_e4m3,
                              input_e4m3.scale(),
                              input,
                              input_e4m3.abs_max(),
                              total_tokens * static_cast<long>(N),
                              mRunState.DeviceProp,
                              mRunState.MainStream);

        moe_grouped_gemm_weight_grad_fp8(d_weight.get<nv_bfloat16>(),
                                         grad_e5m2.get<__nv_fp8_e5m2>(),
                                         input_e4m3.get<__nv_fp8_e4m3>(),
                                         grad_e5m2.scale(),
                                         input_e4m3.scale(),
                                         expert_offsets,
                                         num_experts,
                                         M,
                                         N,
                                         mRunState.CublasLtHandle,
                                         mRunState.MainStream,
                                         host_offsets,
                                         1.0f,
                                         beta,
                                         active_experts,
                                         weight_is_compact,
                                         num_active);
        return true;
    } catch (const std::exception& e) {
        if (std::getenv("SUROGATE_DEBUG_FP8_MOE_WGRAD")) {
            std::fprintf(stderr, "[FP8 MoE wgrad] fallback in %s: %s\n", debug_tag ? debug_tag : "unknown", e.what());
        }
        return false;
    }
}

void CompiledExecutor::set_last_inputs_cpu(const Tensor* inputs_cpu) {
    mLastInputsCpu = inputs_cpu;
}

void CompiledExecutor::set_rng_seed_fn(std::function<unsigned int()> fn) {
    mRngSeedFn = std::move(fn);
}

const Tensor* CompiledExecutor::try_get_tensor(const std::string& name) const {
    if (auto it = mNamedTensors.find(name); it != mNamedTensors.end() && it->second.Data) {
        return &it->second;
    }

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
    // Backward-phase tensors (d_blocks[N].<base>, d_xF, d_encoded, ...) are
    // registered via bind_tensor in execute_backward; they land in
    // mNamedTensors even when the backward graph has no SSA id for the name.
    if (auto it = mNamedTensors.find(name); it != mNamedTensors.end() && it->second.Data) {
        return &it->second;
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

    std::string lookup_name = name;
    const bool is_grad_block_name = starts_with(name, "d_blocks[");
    if (is_grad_block_name) {
        lookup_name = name.substr(2);
    }

    int layer_idx = -1;
    const TensorSlot slot = resolve_block_slot(lookup_name, &layer_idx);
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) {
        return nullptr;
    }

    // Route through the shared slot-dispatch helpers: resolve_block_slot()
    // covers name parsing + slot lookup; block_activation_ptr /
    // block_gradient_ptr are the single slot->RunState-buffer tables. Any new
    // slot is added to tensor_slot_dispatch.cpp only; this function stays trivial.
    if (is_grad_block_name) {
        return block_gradient_ptr(mRunState, layer_idx, slot);
    }
    if (Tensor* t = block_activation_ptr(mRunState, layer_idx, slot)) {
        return t;
    }
    // Forward-scope references to gradient slots (e.g., `blocks[N].d_h_out`)
    // also resolve via the gradient helper — the enum is the source of truth
    // regardless of whether the name has a top-level `d_` prefix.
    return block_gradient_ptr(mRunState, layer_idx, slot);
}

void CompiledExecutor::handle_layer_start(int layer_idx) {
    if (mWeightManager && mWeightManager->needs_block_gather() && !mCapturing) {
        // Ensure the current layer's weights are present in a prefetch slot.
        // For strict-reverse backward orders (Q3 pure attention), the prior
        // layer_start's prefetch already did this and gather_block(layer_idx)
        // early-returns (but re-points entry.work to the holding slot).
        // For non-monotonic orders (Q3.5 hybrid: all attention layers
        // first, then linear), the prior prefetch is for a layer that
        // won't actually run next, so the current layer may not be in
        // any slot — this call synchronously gathers it.
        if (mComm) {
            mWeightManager->gather_block(layer_idx, *mComm, mRunState.side_stream());
        }
        mWeightManager->wait_for_gather(layer_idx, mRunState.MainStream);
    }

    // Prefetch next layer in the current traversal direction.
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

void CompiledExecutor::dispatch_forward_op(const CompiledOp& op, const modules::ForwardHook* hook) {
    if (!op.fn) {
        std::ostringstream oss;
        oss << "dispatch_forward_op: no dispatch fn for op type " << op_type_to_string(op.type) << " (id=" << op.op_id
            << ", semantic=" << op_semantic_kind_name(op.semantic_kind)
            << ", comm=" << communication_kind_name(op.comm_profile.kind)
            << ", distribution=" << distribution_kind_name(op.distribution_kind)
            << ", caps=" << op_capability_flags_string(op.default_caps)
            << ", matmul_caps=" << matmul_capability_flags_string(op.matmul_caps)
            << ", moe_caps=" << moe_capability_flags_string(op.moe_caps)
            << ", epilogue=" << epilogue_support_flags_string(op.epilogue_support)
            << ", storage=" << storage_compatibility_flags_string(op.storage_compat) << ")";
        throw std::runtime_error(oss.str());
    }
    op.fn(*this, op, static_cast<const void*>(hook));
}

void CompiledExecutor::dispatch_backward_op(const CompiledOp& op, const modules::BackwardHook* hook) {
    if (!op.fn) {
        std::ostringstream oss;
        oss << "dispatch_backward_op: no dispatch fn for op type " << op_type_to_string(op.type) << " (id=" << op.op_id
            << ", semantic=" << op_semantic_kind_name(op.semantic_kind)
            << ", comm=" << communication_kind_name(op.comm_profile.kind)
            << ", distribution=" << distribution_kind_name(op.distribution_kind)
            << ", caps=" << op_capability_flags_string(op.default_caps)
            << ", matmul_caps=" << matmul_capability_flags_string(op.matmul_caps)
            << ", moe_caps=" << moe_capability_flags_string(op.moe_caps)
            << ", epilogue=" << epilogue_support_flags_string(op.epilogue_support)
            << ", storage=" << storage_compatibility_flags_string(op.storage_compat) << ")";
        throw std::runtime_error(oss.str());
    }
    op.fn(*this, op, static_cast<const void*>(hook));
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

void CompiledExecutor::resize_segment_graphs(const CompiledGraph& fwd_graph, const CompiledGraph& bwd_graph) {
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
