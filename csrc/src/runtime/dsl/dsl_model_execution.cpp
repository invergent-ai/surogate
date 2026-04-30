// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/executor/causal_lm_execution_profile.h"
#include "runtime/executor/graph_executor.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"

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

static const CausalLMExecutionProfile& causal_lm_profile() {
    static const CausalLMExecutionProfile profile;
    return profile;
}

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    mDocMaskingActive =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

    if (!lora_enabled()) {
        auto request = causal_lm_profile()
                           .make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, micro_step);
        mExecutor->execute_forward(request, comm);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);
    if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
        mQLoRAProvider->invalidate_cache();
    }

    // micro_step seeds per-projection dropout in the LoRA slice dispatcher.
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    auto request =
        causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, micro_step);
    mExecutor->execute_forward(request, comm);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    mDocMaskingActive =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

    if (!lora_enabled()) {
        auto request =
            causal_lm_profile()
                .make_eval_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, targets, micro_step);
        const auto result = mExecutor->execute_eval(request, comm);
        const float loss = result.loss.value_or(0.0f);
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

    auto request =
        causal_lm_profile()
            .make_eval_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, targets, micro_step);
    const auto result = mExecutor->execute_eval(request, comm);
    const float loss = result.loss.value_or(0.0f);
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
        auto request = causal_lm_profile().make_backward_request(*mRunState,
                                                                 mModelConfig,
                                                                 mOptions,
                                                                 inputs,
                                                                 targets,
                                                                 grad_accum_steps,
                                                                 micro_step);
        mExecutor->execute_backward(request, comm);
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

    auto request =
        causal_lm_profile()
            .make_backward_request(*mRunState, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    mExecutor->execute_backward(request, comm);

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
                                                      /*grad_dtype=*/initial_act_dtype,
                                                      &mBlockSchemaPlanRecords);
    long required_size = required_stack_bytes(initial_plan, /*bwd_graph=*/nullptr, mModelConfig, mOptions);

    if (options.DebugMemoryBreakdown && comm.rank() == 0) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr
            << "[DEBUG-STACK] plan_peak=" << initial_plan.plan_stack_peak_bytes() / (1024 * 1024) << " MiB"
            << ", initial_required=" << required_size / (1024 * 1024) << " MiB"
            << ", block_schema_records=" << initial_plan.schema_record_count
            << ", block_schema_routing_layers=" << initial_plan.schema_routing_layers
            << ", block_schema_ep_layers=" << initial_plan.schema_ep_layers
            << ", block_schema_dense_layers=" << initial_plan.schema_dense_layers
            << ", block_schema_moe_layers=" << initial_plan.schema_moe_layers
            << ", block_schema_mamba_layers=" << initial_plan.schema_mamba_layers
            << ", block_schema_linear_mixer_layers=" << initial_plan.schema_linear_mixer_layers
            << ", block_schema_slots=" << initial_plan.schema_slot_count
            << ", block_schema_op_lifetime_slots=" << initial_plan.schema_op_lifetime_slots
            << ", block_schema_layer_lifetime_slots=" << initial_plan.schema_layer_lifetime_slots
            << ", block_schema_block_lifetime_slots=" << initial_plan.schema_block_lifetime_slots
            << ", block_schema_model_lifetime_slots=" << initial_plan.schema_model_lifetime_slots
            << ", block_schema_persistent_lifetime_slots=" << initial_plan.schema_persistent_lifetime_slots
            << ", block_schema_sharded_dim_slots=" << initial_plan.schema_sharded_dim_slots
            << ", block_schema_router_replicated_slots=" << initial_plan.schema_router_replicated_slots
            << ", block_schema_expert_parallel_slots=" << initial_plan.schema_expert_parallel_slots
            << ", block_schema_streaming_slots=" << initial_plan.schema_streaming_slots
            << ", block_schema_auto_resident_slots=" << initial_plan.schema_auto_resident_slots
            << ", block_schema_cpu_stream_slots=" << initial_plan.schema_cpu_pinned_stream_slots
            << ", block_schema_nvme_offload_slots=" << initial_plan.schema_nvme_offload_slots
            << ", block_schema_registry_registered_activation_slots="
            << initial_plan.schema_registry_registered_activation_slots
            << ", block_schema_registry_missing_activation_slots="
            << initial_plan.schema_registry_missing_activation_slots
            << ", block_schema_registry_save_for_backward_activation_slots="
            << initial_plan.schema_registry_save_for_backward_activation_slots
            << ", block_schema_registry_save_for_backward_mismatch_slots="
            << initial_plan.schema_registry_save_for_backward_mismatch_slots
            << ", block_schema_resolved_activation_shape_slots=" << initial_plan.schema_resolved_activation_shape_slots
            << ", block_schema_unresolved_activation_shape_slots="
            << initial_plan.schema_unresolved_activation_shape_slots
            << ", block_schema_dynamic_activation_shape_slots=" << initial_plan.schema_dynamic_activation_shape_slots
            << ", block_schema_resolved_activation_shape_bytes=" << initial_plan.schema_resolved_activation_shape_bytes
            << ", block_schema_save_for_backward_activation_slots="
            << initial_plan.schema_save_for_backward_activation_slots
            << ", block_schema_frame_activation_slots=" << initial_plan.schema_frame_activation_slots
            << ", block_schema_save_for_backward_activation_bytes="
            << initial_plan.schema_save_for_backward_activation_bytes
            << ", block_schema_frame_activation_bytes=" << initial_plan.schema_frame_activation_bytes
            << ", block_schema_allocation_authoritative=" << (initial_plan.schema_allocation_authoritative ? 1 : 0)
            << ", block_schema_allocation_authoritative_layers=" << initial_plan.schema_allocation_authoritative_layers
            << ", block_schema_allocation_unresolved_slots=" << initial_plan.schema_allocation_unresolved_slots
            << ", block_schema_authoritative_frame_arena_bytes=" << initial_plan.schema_authoritative_frame_arena_bytes
            << ", block_schema_authoritative_save_for_backward_arena_bytes="
            << initial_plan.schema_authoritative_save_for_backward_arena_bytes
            << ", block_schema_authoritative_persistent_activation_bytes="
            << initial_plan.schema_authoritative_persistent_activation_bytes
            << ", block_schema_authoritative_host_stream_activation_bytes="
            << initial_plan.schema_authoritative_host_stream_activation_bytes
            << ", block_schema_authoritative_total_activation_arena_bytes="
            << initial_plan.schema_authoritative_total_activation_arena_bytes
            << ", block_schema_max_layer_activation_shape_bytes="
            << initial_plan.schema_max_layer_activation_shape_bytes
            << ", block_schema_baseline_max_activation_shape_bytes="
            << initial_plan.schema_baseline_max_activation_shape_bytes
            << ", block_schema_activation_shape_savings_bytes=" << initial_plan.schema_activation_shape_savings_bytes
            << ", block_schema_resolved_param_shape_slots=" << initial_plan.schema_resolved_param_shape_slots
            << ", block_schema_unresolved_param_shape_slots=" << initial_plan.schema_unresolved_param_shape_slots
            << ", block_schema_expert_parallel_param_slots=" << initial_plan.schema_expert_parallel_param_slots
            << ", block_schema_resolved_param_shape_bytes=" << initial_plan.schema_resolved_param_shape_bytes
            << ", block_schema_resolved_param_shape_local_bytes="
            << initial_plan.schema_resolved_param_shape_local_bytes
            << ", block_schema_expert_parallel_param_shape_bytes="
            << initial_plan.schema_expert_parallel_param_shape_bytes
            << ", block_schema_expert_parallel_param_shape_local_bytes="
            << initial_plan.schema_expert_parallel_param_shape_local_bytes
            << ", block_schema_expert_parallel_param_shape_savings_bytes="
            << initial_plan.schema_expert_parallel_param_shape_savings_bytes
            << ", block_schema_scoring_bias_layers=" << initial_plan.schema_scoring_bias_routing_layers
            << ", block_schema_shared_expert_layers=" << initial_plan.schema_shared_expert_routing_layers
            << ", block_schema_weight_transfer_layers=" << initial_plan.schema_weight_transfer_layers
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
                                              layout,
                                              &mBlockSchemaPlanRecords);
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

    // CPU-RAM centric training: stream any trainable DSL parameter gradients
    // through rotating device buffers. LoRA-only runs usually have no DSL grads,
    // but train_router=true leaves router parameters trainable alongside LoRA,
    // so this must be keyed on the actual grad set rather than lora_enabled().
    if (mGrads && mOptions.CpuTraining && !mGrads->param_names().empty()) {
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
    exec_opts.excluded_save_tensors = causal_lm_profile().backward_save_exclusions();
    mExecutor =
        std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, mOptions, exec_opts);
    if (auto* graph_exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
        graph_exec->set_schema_hook_registry(&mHookRegistry);
    }
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
        float z_coef = mModelConfig.moe_config.has_value() ? mModelConfig.moe_config->router_z_loss_coef : 0.001f;
        mRunState->set_moe_config(mModelConfig.NumExperts, aux_coef, z_coef);
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

    const int BT = B * T;
    std::vector<float> result(static_cast<std::size_t>(BT), 0.0f);

    const modules::ForwardHook* hook_ptr = nullptr;
    if (use_lora && lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mLoRARunState->is_training = false;
    }

    auto& rs = *mRunState;
    const std::size_t token_count = static_cast<std::size_t>(BT);
    const std::size_t token_bytes = token_count * sizeof(std::int32_t);
    Tensor input_tensor = Tensor::from_pointer(reinterpret_cast<std::byte*>(const_cast<std::int32_t*>(input_ids)),
                                               -1,
                                               ETensorDType::INT32,
                                               std::vector<long>{B, T});
    Tensor target_tensor = Tensor::from_pointer(reinterpret_cast<std::byte*>(const_cast<std::int32_t*>(targets)),
                                                -1,
                                                ETensorDType::INT32,
                                                std::vector<long>{B, T});

    std::vector<std::int32_t> default_positions;
    const std::int32_t* position_source = position_ids;
    if (!position_source) {
        default_positions.resize(token_count);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < T; ++t) {
                default_positions[static_cast<std::size_t>(b * T + t)] = t;
            }
        }
        position_source = default_positions.data();
    }
    Tensor position_tensor =
        Tensor::from_pointer(reinterpret_cast<std::byte*>(const_cast<std::int32_t*>(position_source)),
                             -1,
                             ETensorDType::INT32,
                             std::vector<long>{B, T});
    const bool doc_masking_active =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, input_tensor, position_tensor);

    float* logprobs_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&logprobs_gpu, token_count * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(logprobs_gpu, 0, token_count * sizeof(float), rs.MainStream));

    float* inv_temperature_gpu = nullptr;
    if (temperatures) {
        std::vector<float> inv_temp(token_count);
        for (std::size_t i = 0; i < token_count; ++i) {
            inv_temp[i] = 1.0f / temperatures[i];
        }
        CUDA_CHECK(cudaMalloc(&inv_temperature_gpu, token_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(inv_temperature_gpu,
                                   inv_temp.data(),
                                   token_count * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   rs.MainStream));
    }

    auto request = causal_lm_profile()
                       .make_eval_request(rs, mModelConfig, mOptions, input_tensor, position_tensor, target_tensor, 0);
    request.mode = ExecutionMode::Forward;
    request.reduce_loss_on_completion = false;
    request.disable_forward_saves = true;
    request.logprobs_gpu = logprobs_gpu;
    request.inv_temperature_gpu = inv_temperature_gpu;
    request.forward_hook = hook_ptr;
    mExecutor->execute_forward(request, comm);

    CUDA_CHECK(cudaMemcpyAsync(result.data(), logprobs_gpu, token_bytes, cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
    CUDA_CHECK(cudaFree(logprobs_gpu));
    if (inv_temperature_gpu) {
        CUDA_CHECK(cudaFree(inv_temperature_gpu));
    }

    if (doc_masking_active) {
        mExecutor->clear_doc_masking();
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

    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const bool doc_masking_active =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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
    }

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B_val, T_val);
        if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
            mQLoRAProvider->invalidate_cache();
        }
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    auto forward_request =
        causal_lm_profile().make_eval_request(rs, mModelConfig, mOptions, inputs, position_ids, targets, micro_step);
    forward_request.mode = ExecutionMode::Forward;
    forward_request.reduce_loss_on_completion = false;
    forward_request.inv_temperature_gpu = inv_temperature_gpu;
    mExecutor->execute_forward(forward_request, comm);

    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    float* custom_dloss_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&custom_dloss_gpu, bt * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(custom_dloss_gpu,
                               per_token_grads_cpu,
                               bt * sizeof(float),
                               cudaMemcpyHostToDevice,
                               main_stream));

    if (!lora_enabled()) {
        auto backward_request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        backward_request.custom_dloss_gpu = custom_dloss_gpu;
        backward_request.inv_temperature_gpu = inv_temperature_gpu;
        mExecutor->execute_backward(backward_request, comm);
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
        CUDA_CHECK(cudaFree(custom_dloss_gpu));
        if (doc_masking_active) mExecutor->clear_doc_masking();
        if (inv_temperature_gpu) {
            CUDA_CHECK(cudaFree(inv_temperature_gpu));
        }
        return;
    }

    // LoRA backward: mirror DslModel::backward() exactly, but use backward_with_custom_dloss.
    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto backward_request =
        causal_lm_profile()
            .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    backward_request.custom_dloss_gpu = custom_dloss_gpu;
    backward_request.inv_temperature_gpu = inv_temperature_gpu;
    mExecutor->execute_backward(backward_request, comm);
    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    CUDA_CHECK(cudaFree(custom_dloss_gpu));

    if (doc_masking_active) mExecutor->clear_doc_masking();
    if (inv_temperature_gpu) {
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

    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t BT = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    mDocMaskingActive =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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
    }

    // Always zero the losses buffer so we get per-micro-batch losses
    // (not accumulated from previous micro-steps).
    fill_zero(rs.Losses, main_stream);
    fill_zero(rs.ValidTokenCount, main_stream);
    fill_zero(rs.CorrectCount, main_stream);

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B_val, T_val);
        if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
            mQLoRAProvider->invalidate_cache();
        }
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    auto request =
        causal_lm_profile().make_eval_request(rs, mModelConfig, mOptions, inputs, position_ids, targets, micro_step);
    request.mode = ExecutionMode::Forward;
    request.reduce_loss_on_completion = false;
    request.inv_temperature_gpu = mGrpoInvTemperatureGpu;
    mExecutor->execute_forward(request, comm);

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

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);

    float* custom_dloss_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&custom_dloss_gpu, bt * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(custom_dloss_gpu,
                               per_token_grads_cpu,
                               bt * sizeof(float),
                               cudaMemcpyHostToDevice,
                               main_stream));

    if (!lora_enabled()) {
        auto request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        request.custom_dloss_gpu = custom_dloss_gpu;
        request.inv_temperature_gpu = mGrpoInvTemperatureGpu;
        mExecutor->execute_backward(request, comm);
    } else {
        // LoRA backward: mirror step_with_custom_loss LoRA backward path exactly.
        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

        auto request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        request.custom_dloss_gpu = custom_dloss_gpu;
        request.inv_temperature_gpu = mGrpoInvTemperatureGpu;
        mExecutor->execute_backward(request, comm);

        mLoRAGrads->end_micro_step(main_stream, comm);
        internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(main_stream));
    CUDA_CHECK(cudaFree(custom_dloss_gpu));

    // Clean up state that was set by forward_for_grpo.
    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }
    if (mGrpoInvTemperatureGpu) {
        CUDA_CHECK(cudaFree(mGrpoInvTemperatureGpu));
        mGrpoInvTemperatureGpu = nullptr;
    }
}

}  // namespace dsl
