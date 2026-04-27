// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/graph_compiler.h"
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
                                       doc_info->total_q,
                                       micro_step);
            mDocMaskingActive = true;
        }
    }
    if (mOptions.DocMasking && !mDocMaskingActive) {
        mExecutor->clear_doc_masking();
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

    // Packed-sequence validation must mirror forward() so debug/compare tools
    // see the same document-level attention masking as training.
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
                                       doc_info->total_q,
                                       micro_step);
            mDocMaskingActive = true;
        }
    }
    if (mOptions.DocMasking && !mDocMaskingActive) {
        mExecutor->clear_doc_masking();
    }

    if (!lora_enabled()) {
        const float loss = mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
        if (mDocMaskingActive) {
            mExecutor->clear_doc_masking();
            mDocMaskingActive = false;
        }
        return loss;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    // Eval mode disables LoRA dropout for deterministic scoring.
    mLoRARunState->is_training = false;
    mLoRARunState->micro_step = micro_step;

    const float loss = mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }
    return loss;
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
    const ActivationLayoutIR* layout = mModule->activation_layout.has_value() ? &*mModule->activation_layout : nullptr;

    // ------------------------------------------------------------------
    // Stack sizing — phase 1: plan-only estimate.
    //
    // Build a BufferPlan ahead of DslRunState so we can size the device
    // stack before any allocations happen. The plan DslRunState builds
    // internally from the same inputs is identical; this is two cheap
    // pure-function calls, not a real duplication.
    //
    // The backward compiled graph doesn't exist yet (the executor hasn't
    // been created), so the initial size is driven by `plan_stack_peak_bytes`
    // + the legacy safety/MoE/arch slacks. A second sizing pass after the
    // executor is ready resizes if the graph-walk peak is larger.
    // ------------------------------------------------------------------
    TensorSlotRegistry initial_registry;
    if (layout) {
        initial_registry.init_from_layout(*layout);
    }
    ETensorDType initial_act_dtype = mOptions.ModelType.value_or(mConfig->DType);
    if (is_fp8_dtype(initial_act_dtype)) {
        initial_act_dtype = ETensorDType::BF16;
    }
    const BufferPlan initial_plan = BufferPlan::build(mModelConfig,
                                                      mRuntimeConfig,
                                                      mOptions,
                                                      initial_registry,
                                                      lora_enabled(),
                                                      static_cast<long>(B),
                                                      static_cast<long>(T),
                                                      initial_act_dtype,
                                                      /*grad_dtype=*/initial_act_dtype);
    long required_size = required_stack_bytes(initial_plan, /*bwd_graph=*/nullptr, mModelConfig, mOptions);

    if (options.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "[DEBUG-STACK] plan_peak=" << initial_plan.plan_stack_peak_bytes() / (1024 * 1024) << " MiB"
                  << ", initial_required=" << required_size / (1024 * 1024) << " MiB"
                  << ", block_schema_records=" << mBlockSchemaPlanRecords.size()
                  << ", GPU used=" << (total_mem - free_mem) / (1024 * 1024) << " MiB"
                  << ", free=" << free_mem / (1024 * 1024) << " MiB" << std::endl;
    }

    mRunState = std::make_unique<DslRunState>(mModelConfig,
                                              mRuntimeConfig,
                                              mOptions,
                                              B,
                                              T,
                                              mAllocator,
                                              lora_enabled(),
                                              mQLoRAConfig.is_prequantized(),
                                              static_cast<std::size_t>(required_size),
                                              layout);
    mRunState->WorldSize = comm.world_size();
    if (mParams) {
        mParams->set_default_stream(mRunState->MainStream);
        if (mQLoRAProvider) {
            mParams->set_qlora_provider(mQLoRAProvider.get());
        }
    }
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

    // CPU-RAM centric training: per-layer base-model gradient streaming.
    //
    // Skipped under LoRA because the base model is frozen and produces no gradients —
    // streaming a never-written buffer would be a no-op. This is NOT a LoRA + CPU
    // incompatibility: LoRA + CPU training works correctly. Base weights are placed
    // on CPU pinned memory via `mOptions.CpuTraining` flowing into the weight manager's
    // `cpu_training` config (independent of this gate); LoRA adapter weights and
    // gradients live on GPU via the dedicated LoRA grad manager.
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

    // ------------------------------------------------------------------
    // Stack sizing — phase 2: re-run with the compiled backward graph.
    //
    // `required_stack_bytes` combines the plan-level peak with the graph-
    // walk peak and takes the larger. For architectures with heavy backward
    // ops (e.g. Qwen3.5 gated delta rule) the graph-walk is the binding
    // constraint — resize up if so. Shrinking is deliberately skipped to
    // avoid churn on the allocator.
    // ------------------------------------------------------------------
    // Wire LoRA weights and weight manager onto the executor BEFORE
    // compile, so compile_graphs can size the Persistent arena's LoRA
    // and DslWeightManager slabs upfront. set_lora_state /
    // set_weight_manager are idempotent pointer-setters — LoRA's full
    // state (grads / run_state) is populated with a second call below.
    if (lora_enabled()) {
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(),
                                  /*grads=*/nullptr,
                                  /*run_state=*/nullptr);
    }
    if (mWeightManager) {
        if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
            exec->set_weight_manager(mWeightManager.get());
        }
    }
    if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
        exec->ensure_graphs_compiled(B, T);
        const long needed =
            required_stack_bytes(mRunState->buffer_plan(), exec->compiled_backward(), mModelConfig, mOptions);
        if (options.DebugMemoryBreakdown && comm.rank() == 0) {
            const long graph_peak = graph_backward_stack_peak(exec->compiled_backward(), mRunState->buffer_plan());
            std::cerr << "[DEBUG-STACK] graph_peak=" << graph_peak / (1024 * 1024) << " MiB"
                      << ", final_required=" << needed / (1024 * 1024) << " MiB"
                      << ", currently_allocated=" << required_size / (1024 * 1024) << " MiB" << std::endl;
        }
        if (needed > required_size) {
            if (options.DebugMemoryBreakdown && comm.rank() == 0) {
                std::cerr << "[DEBUG-STACK] Resizing stack: " << required_size / (1024 * 1024) << " MiB" << " -> "
                          << needed / (1024 * 1024) << " MiB" << std::endl;
            }
            mRunState->resize_stack_to(needed);
            required_size = needed;
        }
    }

    // Enable MoE routing stats tracking
    if (mModelConfig.NumExperts > 0) {
        float aux_coef = mModelConfig.moe_config.has_value() ? mModelConfig.moe_config->router_aux_loss_coef : 0.01f;
        mRunState->set_moe_config(mModelConfig.NumExperts, aux_coef);
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
    if (mLoRAGrads) {
        mLoRAGrads->zero_all(stream);
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

void DslModel::prepare_bwd_cross_layer_for_capture() {
    if (mExecutor) {
        mExecutor->prepare_bwd_cross_layer_for_capture();
    }
}

void DslModel::enable_doc_masking_pad_to_max(int max_num_docs, int num_micro_steps) {
    if (mExecutor) {
        mExecutor->enable_doc_masking_pad_to_max(max_num_docs, num_micro_steps);
    }
}

void DslModel::stage_cu_seqlens_for_micro_step(int micro_step,
                                               const std::int32_t* cu_seqlens_cpu,
                                               int count,
                                               int total_q) {
    if (mExecutor) {
        mExecutor->stage_cu_seqlens_for_micro_step(micro_step, cu_seqlens_cpu, count, total_q);
    }
}

void DslModel::reissue_cu_seqlens_for_micro_step(int micro_step) {
    if (mExecutor) {
        mExecutor->reissue_cu_seqlens_for_micro_step(micro_step);
    }
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
    } else if (mOptions.DocMasking) {
        graph_exec->clear_doc_masking();
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
    } else if (mOptions.DocMasking) {
        graph_exec->clear_doc_masking();
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
    } else if (mOptions.DocMasking) {
        graph_exec->clear_doc_masking();
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
