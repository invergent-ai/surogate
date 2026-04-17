// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/executor/graph_executor.h"
#include "runtime/executor/graph_executor_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "kernels/kernels.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/lora/lora_utils.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/optimizers/flash_adamw_8bit.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

#include <iostream>
#include <optional>
#include <vector>

namespace dsl {

// ============================================================================
// Document masking: detect document boundaries from position_ids resets
// ============================================================================

struct DocMaskingInfo {
    std::vector<std::int32_t> cu_seqlens;  // (num_docs + 1,) cumulative offsets
    int num_docs;
    int max_seqlen;
    int total_q;
};

/// Scan position_ids for non-consecutive transitions to detect document
/// boundaries in packed sequences. Returns nullopt if no boundaries found (i.e.
/// single contiguous sequence per batch element — standard SFT/PT).
///
/// When \p mrope is true the position_ids come from a multimodal RoPE model
/// (e.g. Qwen3-VL / Qwen3.5-VL).  In that case image/video tokens share the
/// same temporal position (the value stays constant across visual tokens within
/// one image), so `curr - prev != 1` would create hundreds of false document
/// boundaries.  We detect boundaries only by strict *decreases* in position
/// (resets), which correctly identifies sample-packing boundaries while
/// ignoring the flat temporal positions inside image regions.
static std::optional<DocMaskingInfo>
compute_doc_masking(const std::int32_t* position_ids, int B, int T, bool mrope = false) {
    if (!position_ids) return std::nullopt;

    std::vector<std::int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int max_seqlen = 0;
    bool has_boundaries = false;

    for (int b = 0; b < B; ++b) {
        int doc_start = b * T;
        for (int t = 1; t < T; ++t) {
            int idx = b * T + t;
            const int prev = position_ids[idx - 1];
            const int curr = position_ids[idx];
            const bool is_boundary = mrope ? (curr < prev) : (curr - prev != 1);
            if (is_boundary) {
                // Document boundary: position_ids are not strictly consecutive.
                // Mirrors HF packed-sequence detection (diff != 1).
                int doc_len = (b * T + t) - doc_start;
                if (doc_len > 0) {
                    cu_seqlens.push_back(cu_seqlens.back() + doc_len);
                    max_seqlen = std::max(max_seqlen, doc_len);
                }
                doc_start = b * T + t;
                has_boundaries = true;
            }
        }
        // Last document in this batch element
        int last_len = (b + 1) * T - doc_start;
        if (last_len > 0) {
            cu_seqlens.push_back(cu_seqlens.back() + last_len);
            max_seqlen = std::max(max_seqlen, last_len);
        }
    }

    if (!has_boundaries) return std::nullopt;

    int num_docs = static_cast<int>(cu_seqlens.size()) - 1;
    int total_q = cu_seqlens.back();
    return DocMaskingInfo{std::move(cu_seqlens), num_docs, max_seqlen, total_q};
}

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    // Detect document boundaries for packed sequence masking.
    // Position_ids with per-document resets (e.g. [0,1,2, 0,1, 0,1,2,3])
    // trigger Flash Attention varlen with cu_seqlens instead of cuDNN full-attention.
    mDocMaskingActive = false;
    if (mOptions.DocMasking && position_ids.Data && position_ids.Device == -1) {
        const auto* pos_ptr = reinterpret_cast<const std::int32_t*>(position_ids.Data);
        const int B = static_cast<int>(inputs.Sizes[0]);
        const int T = static_cast<int>(inputs.Sizes[1]);
        const bool mrope = mModelConfig.Rope.is_multimodal();
        auto doc_info = compute_doc_masking(pos_ptr, B, T, mrope);
        if (doc_info) {
            mExecutor->set_doc_masking(doc_info->cu_seqlens.data(),
                                       doc_info->num_docs,
                                       doc_info->max_seqlen,
                                       doc_info->total_q);
            mDocMaskingActive = true;
        }
    }

    if (!lora_enabled()) {
        mExecutor->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);
    if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
        mQLoRAProvider->invalidate_cache();
    }

    // micro_step seeds per-projection dropout in the LoRA slice dispatcher.
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    mExecutor->forward(inputs, position_ids, comm, micro_step);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    // Eval mode disables LoRA dropout for deterministic scoring.
    mLoRARunState->is_training = false;
    mLoRARunState->micro_step = micro_step;

    return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
}

void DslModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward called before allocate_run_state()");
    }
    mUseTokenScale = true;

    if (!lora_enabled()) {
        mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        if (mDocMaskingActive) {
            mExecutor->clear_doc_masking();
            mDocMaskingActive = false;
        }
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);

    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }

    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

void DslModel::allocate_run_state(const RuntimeOptions& options,
                                  NCCLCommunicator& comm,
                                  int B,
                                  int T,
                                  bool allocate_optimizer) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mOptions = options;
    if (qlora_enabled() && mQLoRAConfig.is_fp4()) {
        mOptions.UseCudaGraphs = false;
    }
    const std::size_t dummy_stack_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;  // 1TB dummy stack
    const ActivationLayoutIR* layout = mModule->activation_layout.has_value() ? &*mModule->activation_layout : nullptr;
    mRunState = std::make_unique<DslRunState>(mModelConfig,
                                              mRuntimeConfig,
                                              mOptions,
                                              B,
                                              T,
                                              mAllocator,
                                              lora_enabled(),
                                              mQLoRAConfig.is_prequantized(),
                                              dummy_stack_bytes,
                                              /*allocate_stack=*/false,
                                              layout);
    mRunState->WorldSize = comm.world_size();
    if (mParams) {
        mParams->set_default_stream(mRunState->MainStream);
        if (mQLoRAProvider) {
            mParams->set_qlora_provider(mQLoRAProvider.get());
        }
    }

    const long base_size = static_cast<long>(mRunState->Stack.max_utilization());
    long moe_extra = 0;
    if (mModelConfig.NumExperts > 0) {
        const long moe_intermediate =
            (mModelConfig.MoeIntermediateSize > 0) ? mModelConfig.MoeIntermediateSize : mModelConfig.IntermediateSize;
        const long hidden = mModelConfig.HiddenSize;
        const long num_experts = mModelConfig.NumExperts;
        const long top_k = std::max(1, mModelConfig.NumExpertsPerTok);
        const long dtype_bytes = 2;  // BF16 bytes (matches modular sizing heuristic)
        const long up_factor = mModelConfig.mlp_up_factor();
        const long expert_gate_up_tp = num_experts * up_factor * moe_intermediate * hidden * dtype_bytes;
        const long expert_down_tp = num_experts * moe_intermediate * hidden * dtype_bytes;
        const long permuted_tokens = 2L * B * T * top_k * hidden * dtype_bytes;
        // MoE activation backward (gpt_oss_moe_act_backward) allocates d_inp buffers
        // in intermediate dimension ({N, up_factor * intermediate}), not hidden dimension.
        // Account for this larger BT-proportional backward buffer.
        const long moe_bwd_act = 2L * B * T * top_k * up_factor * moe_intermediate * dtype_bytes;
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens + moe_bwd_act;
    }
    ETensorDType act_dtype = mOptions.ModelType.value_or(mConfig->DType);
    if (is_fp8_dtype(act_dtype)) {
        act_dtype = ETensorDType::BF16;
    }
    const long dtype_bytes = static_cast<long>(get_dtype_size(act_dtype));
    const long BT = static_cast<long>(B) * static_cast<long>(T);
    const long C = mModelConfig.HiddenSize;
    long QKV = mModelConfig.head_size() * (mModelConfig.NumQueryHeads + 2 * mModelConfig.NumKeyValHeads);
    long MUp = static_cast<long>(mModelConfig.mlp_up_rows());
    // For hybrid models, use max dims across all block types
    if (mRuntimeConfig.has_per_layer_dims()) {
        for (const auto& pld : mRuntimeConfig.per_layer_dims) {
            QKV = std::max(QKV, pld.qkv_channels);
            MUp = std::max(MUp, pld.mlp_up);
        }
    }
    const long extra_tmp = std::max({BT * C, BT * QKV, BT * MUp}) * dtype_bytes;
    long attn_fallback_bytes = 0;
    const bool lora_stack_tight = lora_enabled();
    const long safety_floor = lora_stack_tight ? (32L * 1024 * 1024) : (64L * 1024 * 1024);
    const long safety_bytes = std::max(safety_floor, base_size / 8);
    // The sizing simulation captures ~55% of actual backward peak (flash attention
    // backward workspace and accumulated temps are not fully modeled), so we use
    // base_multiplier=2 for both LoRA and full fine-tune.
    // CPU-RAM centric: tighter multiplier since stack resets at each layer boundary.
    const long base_multiplier = mOptions.CpuTraining ? 1L : 2L;
    long required_size =
        std::max(1024L * 1024,
                 base_size * base_multiplier + moe_extra + safety_bytes + extra_tmp + attn_fallback_bytes);
    const long slack_bytes = mOptions.CpuTraining ? (128L * 1024 * 1024)
                             : lora_stack_tight   ? (256L * 1024 * 1024)
                                                  : (512L * 1024 * 1024);
    required_size += slack_bytes;  // extra slack for unmodeled temps
    const bool is_qwen3_hybrid_lora = lora_stack_tight &&
                                      (mModelConfig.architecture == modules::ArchitectureType::Hybrid) &&
                                      (mModelConfig.Architecture == PretrainedConfig::QWEN3);
    if (is_qwen3_hybrid_lora) {
        // Qwen3.5 hybrid blocks can hit an additional transient peak around SwiGLU
        // backward + LoRA hook temps that is not fully captured by the sizing simulation.
        // Reserve one extra BT x MUp buffer plus fixed margin.
        const long qwen35_swiglu_peak = BT * static_cast<long>(mModelConfig.mlp_up_rows()) * dtype_bytes;
        const long qwen35_extra_slack = std::max(128L * 1024 * 1024, qwen35_swiglu_peak + 64L * 1024 * 1024);
        required_size += qwen35_extra_slack;
    }
    long moe_stack_slack = 0;
    if (mModelConfig.NumExperts > 0) {
        moe_stack_slack = 2048L * 1024 * 1024;  // MoE backward temps can spike beyond simulated high-water mark
    }
    if (const char* env = std::getenv("SUROGATE_STACK_SLACK_MB")) {
        const long mb = std::max(0L, std::atol(env));
        moe_stack_slack = std::max(moe_stack_slack, mb * 1024 * 1024);
    }
    required_size += moe_stack_slack;
    long min_stack_base = mOptions.CpuTraining ? (512L * 1024 * 1024)
                          : lora_stack_tight   ? (512L * 1024 * 1024)
                                               : (3L * 1024 * 1024 * 1024);
    if (is_qwen3_hybrid_lora) {
        min_stack_base = std::max(min_stack_base, 1024L * 1024 * 1024);
    }
    if (mOptions.UseCudaGraphs) {
        // CUDA graph capture/replay requires stable temp addresses and tends to keep
        // additional transient allocations live during capture. Reserve extra headroom
        // to avoid capture-only stack OOMs.
        const long graph_extra_slack = lora_stack_tight ? (512L * 1024 * 1024) : (1024L * 1024 * 1024);
        required_size += graph_extra_slack;
        min_stack_base = std::max(min_stack_base, lora_stack_tight ? (1024L * 1024 * 1024) : (4L * 1024 * 1024 * 1024));
        if (is_qwen3_hybrid_lora) {
            // Qwen3.5 hybrid + LoRA + CUDA graphs retains additional transient tensors
            // during capture/replay that are not represented by simulation. Keep a
            // dedicated margin so mamba_gated_rmsnorm/swiglu backward peaks fit.
            required_size += 512L * 1024 * 1024;
            min_stack_base = std::max(min_stack_base, 1536L * 1024 * 1024);
        }
    }
    if (const char* env = std::getenv("SUROGATE_MIN_STACK_MB")) {
        const long mb = std::max(64L, std::atol(env));
        min_stack_base = mb * 1024 * 1024;
    }
    const long min_stack_bytes = min_stack_base + attn_fallback_bytes + moe_stack_slack;
    required_size =
        std::max(required_size, min_stack_bytes);  // Full fine-tune keeps 3GB+fallback; LoRA can use tighter floor.
    const auto high_mark = mRunState->Stack.get_high_mark();
    // DEBUG: Stack allocation size
    if (options.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-STACK] base_size=" << base_size / (1024 * 1024) << " MiB"
                  << ", required_size=" << required_size / (1024 * 1024) << " MiB"
                  << ", GPU used=" << (total_mem - free_mem) / (1024 * 1024) << " MiB"
                  << ", free=" << free_mem / (1024 * 1024) << " MiB" << std::endl;
    }
    Tensor stack_buffer =
        mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {required_size});
    mRunState->set_stack_buffer(std::move(stack_buffer), high_mark);
    comm.barrier();

    // Configure gradient manager for multi-GPU overlapped reduction
    if (mGrads && comm.world_size() > 1) {
        DslGradStoreConfig grad_config;
        grad_config.num_shards = comm.world_size();
        grad_config.shard_idx = comm.rank();
        grad_config.shard_gradients = mOptions.ShardGradients;  // ZeRO-2
        grad_config.use_all_to_all_reduce = mOptions.UseAllToAllReduce;
        grad_config.num_layers = mModelConfig.NumLayers;
        mGrads->configure(grad_config);
    }

    // CPU-RAM centric training: enable per-layer gradient streaming only for full fine-tune.
    // LoRA adapter gradients already live in the dedicated LoRA grad manager, so streaming
    // frozen base-model grads here is unnecessary and can interfere with the LoRA path.
    if (mGrads && mOptions.CpuTraining && !lora_enabled() && !mGrads->param_names().empty()) {
        // For single-GPU, configure() may not have been called yet (it requires world_size > 1).
        // Ensure the layer map is built by calling configure with minimal config.
        if (comm.world_size() == 1) {
            DslGradStoreConfig grad_config;
            grad_config.num_shards = 1;
            grad_config.shard_idx = 0;
            grad_config.num_layers = mModelConfig.NumLayers;
            mGrads->configure(grad_config);
        }
        mGrads->enable_streaming(*mParams);
    }

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    mExecutor =
        std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, mOptions, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
    }

    // Estimate actual backward stack peak from the compiled graph and resize if needed.
    // The heuristic sizing above may underestimate for architectures with heavy backward
    // ops (e.g. Qwen3.5 gated delta rule) that allocate many internal temps on the stack.
    if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
        const long bwd_peak = exec->estimate_backward_stack_peak(B, T);
        if (options.DebugMemoryBreakdown && comm.rank() == 0) {
            std::cerr << "[DEBUG-STACK] Backward peak estimate=" << bwd_peak / (1024 * 1024) << " MiB"
                      << ", heuristic=" << required_size / (1024 * 1024) << " MiB" << std::endl;
        }
        if (bwd_peak > 0) {
            // Safety margin for dispatch-internal temps not in the graph.
            // cpu_training: tighter margin since stack resets at each layer boundary.
            const long safety = mOptions.CpuTraining ? std::max(64L * 1024 * 1024, bwd_peak / 8)
                                                     : std::max(128L * 1024 * 1024, bwd_peak / 3);
            const long needed = bwd_peak + safety;
            if (needed > required_size) {
                if (options.DebugMemoryBreakdown && comm.rank() == 0) {
                    std::cerr << "[DEBUG-STACK] Resizing stack: " << required_size / (1024 * 1024) << " MiB" << " -> "
                              << needed / (1024 * 1024) << " MiB" << std::endl;
                }
                Tensor new_stack =
                    mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {needed});
                mRunState->set_stack_buffer(std::move(new_stack), high_mark);
                required_size = needed;
            }
        }
    }

    // Enable MoE routing stats tracking
    if (mModelConfig.NumExperts > 0) {
        float aux_coef = mModelConfig.moe_config.has_value() ? mModelConfig.moe_config->router_aux_loss_coef : 0.01f;
        mRunState->set_moe_config(mModelConfig.NumExperts, aux_coef);
    }

    // Wire weight manager for streaming/sharding
    if (mWeightManager) {
        if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
            exec->set_weight_manager(mWeightManager.get());
        }
    }

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(),
                                  mLoRAGrads.get(),
                                  mLoRARunState.get());
    }

    if (allocate_optimizer && lora_enabled()) {
        if (!mLoRAAdamW8BitState) {
            mLoRAAdamW8BitState = std::make_unique<modules::LoRAAdamW8BitState>();
        }
    }
}

void DslModel::zero_grads(cudaStream_t stream) {
    if (mGrads) {
        mGrads->zero_all(stream);
    }
}

void DslModel::set_internal_graphs_enabled(bool enabled) {
    if (mExecutor) {
        mExecutor->set_internal_graphs_enabled(enabled);
    }
}

bool DslModel::internal_graphs_enabled() const {
    return mExecutor ? mExecutor->internal_graphs_enabled() : false;
}

bool DslModel::has_capture_unsafe_ops() const {
    return mExecutor ? mExecutor->has_capture_unsafe_ops() : false;
}

std::vector<float> DslModel::compute_logprobs(const std::int32_t* input_ids,
                                              const std::int32_t* targets,
                                              int B,
                                              int T,
                                              bool use_lora,
                                              NCCLCommunicator& comm,
                                              const std::int32_t* position_ids,
                                              const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::compute_logprobs called before allocate_run_state()");
    }

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::compute_logprobs: executor is not a GraphExecutor");
    }

    const int BT = B * T;
    std::vector<float> result(static_cast<std::size_t>(BT), 0.0f);

    const modules::ForwardHook* hook_ptr = nullptr;
    if (use_lora && lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mLoRARunState->is_training = false;
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids, B, T, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                    doc_info->num_docs,
                                    doc_info->max_seqlen,
                                    doc_info->total_q);
    }

    graph_exec->execute_logprobs_forward((long)B,
                                         (long)T,
                                         input_ids,
                                         targets,
                                         result.data(),
                                         hook_ptr,
                                         comm,
                                         position_ids,
                                         temperatures);

    if (doc_info) {
        graph_exec->clear_doc_masking();
    }

    return result;
}

void DslModel::step_with_custom_loss(Tensor inputs,
                                     Tensor position_ids,
                                     Tensor targets,
                                     const float* per_token_grads_cpu,
                                     int grad_accum_steps,
                                     int micro_step,
                                     NCCLCommunicator& comm,
                                     const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::step_with_custom_loss called before allocate_run_state()");
    }
    mUseTokenScale = false;

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::step_with_custom_loss: executor is not a GraphExecutor");
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::int32_t* position_ids_ptr = (position_ids.Data && position_ids.Device == -1)
                                               ? reinterpret_cast<const std::int32_t*>(position_ids.Data)
                                               : nullptr;
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids_ptr, B_val, T_val, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                    doc_info->num_docs,
                                    doc_info->max_seqlen,
                                    doc_info->total_q);
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    float* inv_temperature_gpu = nullptr;
    if (temperatures) {
        const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
        std::vector<float> inv_temp(bt);
        for (std::size_t i = 0; i < bt; ++i) {
            inv_temp[i] = 1.0f / temperatures[i];
        }
        CUDA_CHECK(cudaMalloc(&inv_temperature_gpu, bt * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inv_temperature_gpu,
                                   inv_temp.data(),
                                   bt * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
        graph_exec->set_inv_temperature_context(inv_temperature_gpu);
    }

    // Forward pass (with LoRA hooks if enabled) — saves activations for backward.
    forward(inputs, position_ids, comm, micro_step);

    if (!lora_enabled()) {
        // No LoRA: plain backward with custom d_loss.
        graph_exec->backward_with_custom_dloss(inputs,
                                               targets,
                                               per_token_grads_cpu,
                                               comm,
                                               grad_accum_steps,
                                               micro_step,
                                               nullptr,
                                               nullptr);
        if (doc_info) graph_exec->clear_doc_masking();
        if (inv_temperature_gpu) {
            graph_exec->set_inv_temperature_context(nullptr);
            CUDA_CHECK(cudaFree(inv_temperature_gpu));
        }
        return;
    }

    // LoRA backward: mirror DslModel::backward() exactly, but use backward_with_custom_dloss.
    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    graph_exec->backward_with_custom_dloss(inputs,
                                           targets,
                                           per_token_grads_cpu,
                                           comm,
                                           grad_accum_steps,
                                           micro_step,
                                           nullptr,
                                           nullptr);

    if (doc_info) graph_exec->clear_doc_masking();
    if (inv_temperature_gpu) {
        graph_exec->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(inv_temperature_gpu));
    }

    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

// ============================================================================
// GRPO single-pass: forward (saves activations) + logprobs extraction
// ============================================================================

std::vector<float> DslModel::forward_for_grpo(Tensor inputs,
                                              Tensor position_ids,
                                              Tensor targets,
                                              int grad_accum_steps,
                                              int micro_step,
                                              NCCLCommunicator& comm,
                                              const float* temperatures) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward_for_grpo called before allocate_run_state()");
    }
    mUseTokenScale = false;

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::forward_for_grpo: executor is not a GraphExecutor");
    }

    // Detect document boundaries and enable flash varlen masking if needed.
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t BT = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    const std::int32_t* position_ids_ptr = (position_ids.Data && position_ids.Device == -1)
                                               ? reinterpret_cast<const std::int32_t*>(position_ids.Data)
                                               : nullptr;
    const bool mrope = mModelConfig.Rope.is_multimodal();
    std::optional<DocMaskingInfo> doc_info;
    if (mOptions.DocMasking) {
        doc_info = compute_doc_masking(position_ids_ptr, B_val, T_val, mrope);
    }
    if (doc_info) {
        graph_exec->set_doc_masking(doc_info->cu_seqlens.data(),
                                    doc_info->num_docs,
                                    doc_info->max_seqlen,
                                    doc_info->total_q);
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Set up per-token inverse temperatures (persists for backward_grpo).
    if (mGrpoInvTemperatureGpu) {
        CUDA_CHECK(cudaFree(mGrpoInvTemperatureGpu));
        mGrpoInvTemperatureGpu = nullptr;
    }
    if (temperatures) {
        std::vector<float> inv_temp(BT);
        for (std::size_t i = 0; i < BT; ++i) {
            inv_temp[i] = 1.0f / temperatures[i];
        }
        CUDA_CHECK(cudaMalloc(&mGrpoInvTemperatureGpu, BT * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(mGrpoInvTemperatureGpu,
                                   inv_temp.data(),
                                   BT * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
        graph_exec->set_inv_temperature_context(mGrpoInvTemperatureGpu);
    }

    // Always zero the losses buffer so we get per-micro-batch losses
    // (not accumulated from previous micro-steps).
    fill_zero(rs.Losses, main_stream);
    fill_zero(rs.ValidTokenCount, main_stream);
    fill_zero(rs.CorrectCount, main_stream);

    // Forward pass (with LoRA hooks if enabled) — saves activations for backward.
    forward(inputs, position_ids, comm, micro_step);

    // Extract logprobs from the Losses buffer.
    // cross_entropy_forward writes: losses[t] = logsumexp - logit[target[t]] = -logprob[t].
    // Masked positions (target == -100) have losses[t] = 0, so logprob[t] = 0.
    std::vector<float> logprobs(BT, 0.0f);
    CUDA_CHECK(
        cudaMemcpyAsync(logprobs.data(), rs.Losses.Data, BT * sizeof(float), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    for (std::size_t i = 0; i < BT; ++i) {
        logprobs[i] = -logprobs[i];
    }

    // Doc masking and temperature context persist for backward_grpo().
    return logprobs;
}

// ============================================================================
// GRPO backward pass (uses activations saved by forward_for_grpo)
// ============================================================================

void DslModel::backward_grpo(Tensor inputs,
                             Tensor targets,
                             const float* per_token_grads_cpu,
                             int grad_accum_steps,
                             int micro_step,
                             NCCLCommunicator& comm) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward_grpo called before allocate_run_state()");
    }

    auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get());
    if (!graph_exec) {
        throw std::runtime_error("DslModel::backward_grpo: executor is not a GraphExecutor");
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    if (!lora_enabled()) {
        // No LoRA: plain backward with custom d_loss.
        // Temperature context is already set from forward_for_grpo (pass nullptr to skip re-allocation).
        graph_exec->backward_with_custom_dloss(inputs,
                                               targets,
                                               per_token_grads_cpu,
                                               comm,
                                               grad_accum_steps,
                                               micro_step,
                                               nullptr,
                                               nullptr);
    } else {
        // LoRA backward: mirror step_with_custom_loss LoRA backward path exactly.
        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

        graph_exec->backward_with_custom_dloss(inputs,
                                               targets,
                                               per_token_grads_cpu,
                                               comm,
                                               grad_accum_steps,
                                               micro_step,
                                               nullptr,
                                               nullptr);

        mLoRAGrads->end_micro_step(main_stream, comm);
        internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
    }

    // Clean up state that was set by forward_for_grpo.
    if (mDocMaskingActive) {
        graph_exec->clear_doc_masking();
        mDocMaskingActive = false;
    }
    if (mGrpoInvTemperatureGpu) {
        graph_exec->set_inv_temperature_context(nullptr);
        CUDA_CHECK(cudaFree(mGrpoInvTemperatureGpu));
        mGrpoInvTemperatureGpu = nullptr;
    }
}

}  // namespace dsl
