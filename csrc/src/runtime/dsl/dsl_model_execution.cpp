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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

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

namespace {
enum GrpoMetricOffset {
    GRPO_METRIC_POLICY_LOSS = 0,
    GRPO_METRIC_MISMATCH_KL = 1,
    GRPO_METRIC_MASKED_MISMATCH_KL = 2,
    GRPO_METRIC_UNMASKED_MISMATCH_KL = 3,
    GRPO_METRIC_IS_MASKED = 4,
    GRPO_METRIC_IS_MASKED_LOW = 5,
    GRPO_METRIC_IS_MASKED_HIGH = 6,
    GRPO_METRIC_TEACHER_KL = 7,
    GRPO_METRIC_SAMPLE_COUNT = 8,
    GRPO_METRIC_KEEP_TOKENS = 9,
    GRPO_METRIC_TOTAL_TOKENS = 10,
    GRPO_METRIC_COUNT = 11,
};

int acquire_grpo_host_staging_slot(modules::GrpoNativeScratch& scratch) {
    const int slot = scratch.next_host_slot;
    scratch.next_host_slot = (scratch.next_host_slot + 1) % modules::GrpoNativeScratch::kHostStagingSlots;
    if (scratch.host_copy_recorded[slot]) {
        CUDA_CHECK(cudaEventSynchronize(scratch.host_copy_done[slot]));
    }
    return slot;
}

void record_grpo_host_staging_slot(modules::GrpoNativeScratch& scratch, int slot, cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(scratch.host_copy_done[slot], stream));
    scratch.host_copy_recorded[slot] = true;
}

void upload_float_scratch(Tensor& host, Tensor& device, const float* src, std::size_t count, cudaStream_t stream) {
    std::memcpy(host.get<float>(), src, count * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(device.Data, host.Data, count * sizeof(float), cudaMemcpyHostToDevice, stream));
}

const float* upload_inv_temperature_scratch(modules::GrpoNativeScratch& scratch,
                                            int staging_slot,
                                            const float* temperatures,
                                            std::size_t count,
                                            cudaStream_t stream) {
    if (!temperatures) {
        return nullptr;
    }
    auto* inv_temp = scratch.host_temperatures[staging_slot].get<float>();
    for (std::size_t i = 0; i < count; ++i) {
        inv_temp[i] = 1.0f / temperatures[i];
    }
    CUDA_CHECK(cudaMemcpyAsync(scratch.inv_temperature.Data,
                               scratch.host_temperatures[staging_slot].Data,
                               count * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
    return scratch.inv_temperature.get<float>();
}
}  // namespace

static const CausalLMExecutionProfile& causal_lm_profile() {
    static const CausalLMExecutionProfile profile;
    return profile;
}

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    // Chunked-sequence mode carries per-document geometry in the chunk
    // metadata (per-chunk cu_seqlens on the kvprefix path) — the dense
    // doc-masking context must stay clear.
    mDocMaskingActive =
        mSequenceChunkActive
            ? false
            : causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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

void DslModel::set_sequence_chunk(int idx, int count, const ChunkPackMeta* pack) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::set_sequence_chunk called before allocate_run_state()");
    }
    mSequenceChunkActive = (idx >= 0 && count > 1);
    mExecutor->set_sequence_chunk(idx, count, pack);
}

void DslModel::zero_sequence_chunk_dkv() {
    if (!mExecutor) {
        throw std::logic_error("DslModel::zero_sequence_chunk_dkv called before allocate_run_state()");
    }
    mExecutor->zero_sequence_chunk_dkv();
}

void DslModel::forward_no_save(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward_no_save called before allocate_run_state()");
    }

    // Chunked-sequence mode carries per-document geometry in the chunk
    // metadata (per-chunk cu_seqlens on the kvprefix path) — the dense
    // doc-masking context must stay clear.
    mDocMaskingActive =
        mSequenceChunkActive
            ? false
            : causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

    if (lora_enabled()) {
        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);
        if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
            mQLoRAProvider->invalidate_cache();
        }
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    auto request =
        causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, micro_step);
    // KV sweep of the chunked schedule: this forward exists only to fill the
    // attention KV caches (the loss op lives in backward). Saved tensors are
    // written normally — every chunk shares the same saved-cache keys, so
    // this costs bandwidth, not memory, and disabling saves would diverge
    // from the compiled buffer plan's persistence layout (EP tensors).
    mExecutor->execute_forward(request, comm);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    // Chunked-sequence mode carries per-document geometry in the chunk
    // metadata (per-chunk cu_seqlens on the kvprefix path) — the dense
    // doc-masking context must stay clear.
    mDocMaskingActive =
        mSequenceChunkActive
            ? false
            : causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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
                                              &mBlockSchemaPlanRecords,
                                              causal_lm_profile().run_state_requirements());
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
        long needed = required_stack_bytes(mRunState->buffer_plan(), exec->compiled_backward(), mModelConfig, mOptions);
        // dispatch-PP runs sub-range stages with skip_finalize, which keeps the boundary
        // saves (res_att/mlp_down) live on the stack on top of the normal backward peak --
        // the auto-sizer doesn't model that, so add headroom (env, MB). There is ample free
        // VRAM here since the base weights are bounded to the streaming prefetch.
        if (const char* env = std::getenv("SUROGATE_DISPATCH_STACK_HEADROOM_MB")) {
            const long mb = std::atol(env);
            if (mb > 0) needed += mb * 1024L * 1024L;
        }
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
    const std::size_t token_bytes = token_count * sizeof(float);
    auto& scratch = rs.grpo_native_scratch();
    if (token_count > static_cast<std::size_t>(scratch.max_tokens)) {
        throw std::runtime_error("compute_logprobs inputs exceed allocated scratch capacity");
    }
    const int staging_slot = acquire_grpo_host_staging_slot(scratch);
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

    auto* logprobs_gpu = scratch.custom_dloss.get<float>();
    CUDA_CHECK(cudaMemsetAsync(logprobs_gpu, 0, token_bytes, rs.MainStream));

    const float* inv_temperature_gpu =
        upload_inv_temperature_scratch(scratch, staging_slot, temperatures, token_count, rs.MainStream);
    record_grpo_host_staging_slot(scratch, staging_slot, rs.MainStream);

    auto request = causal_lm_profile()
                       .make_eval_request(rs, mModelConfig, mOptions, input_tensor, position_tensor, target_tensor, 0);
    request.mode = ExecutionMode::Forward;
    request.reduce_loss_on_completion = false;
    request.disable_forward_saves = true;
    request.logprobs_gpu = logprobs_gpu;
    request.inv_temperature_gpu = inv_temperature_gpu;
    request.forward_hook = hook_ptr;
    mExecutor->execute_forward(request, comm);

    CUDA_CHECK(cudaMemcpyAsync(scratch.host_inference_logprobs[staging_slot].Data,
                               logprobs_gpu,
                               token_bytes,
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
    std::memcpy(result.data(), scratch.host_inference_logprobs[staging_slot].get<float>(), token_bytes);

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
    auto& scratch = rs.grpo_native_scratch();
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    const int staging_slot = acquire_grpo_host_staging_slot(scratch);
    const float* inv_temperature_gpu =
        upload_inv_temperature_scratch(scratch, staging_slot, temperatures, bt, main_stream);

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

    upload_float_scratch(scratch.host_inference_logprobs[staging_slot],
                         scratch.custom_dloss,
                         per_token_grads_cpu,
                         bt,
                         main_stream);
    record_grpo_host_staging_slot(scratch, staging_slot, main_stream);

    if (!lora_enabled()) {
        auto backward_request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
        backward_request.inv_temperature_gpu = inv_temperature_gpu;
        mExecutor->execute_backward(backward_request, comm);
        if (doc_masking_active) mExecutor->clear_doc_masking();
        return;
    }

    // LoRA backward: mirror DslModel::backward() exactly, but use backward_with_custom_dloss.
    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto backward_request =
        causal_lm_profile()
            .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
    backward_request.inv_temperature_gpu = inv_temperature_gpu;
    mExecutor->execute_backward(backward_request, comm);

    if (doc_masking_active) mExecutor->clear_doc_masking();

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
    // Chunked-sequence mode carries per-document geometry in the chunk
    // metadata (per-chunk cu_seqlens on the kvprefix path) — the dense
    // doc-masking context must stay clear.
    mDocMaskingActive =
        mSequenceChunkActive
            ? false
            : causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    auto& scratch = rs.grpo_native_scratch();
    const int staging_slot = acquire_grpo_host_staging_slot(scratch);

    // Set up per-token inverse temperatures (persists for backward_grpo).
    mGrpoInvTemperatureGpu =
        const_cast<float*>(upload_inv_temperature_scratch(scratch, staging_slot, temperatures, BT, main_stream));
    record_grpo_host_staging_slot(scratch, staging_slot, main_stream);

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
    auto& scratch = rs.grpo_native_scratch();
    const int staging_slot = acquire_grpo_host_staging_slot(scratch);

    upload_float_scratch(scratch.host_inference_logprobs[staging_slot],
                         scratch.custom_dloss,
                         per_token_grads_cpu,
                         bt,
                         main_stream);
    record_grpo_host_staging_slot(scratch, staging_slot, main_stream);

    if (!lora_enabled()) {
        auto request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
        request.inv_temperature_gpu = mGrpoInvTemperatureGpu;
        mExecutor->execute_backward(request, comm);
    } else {
        // LoRA backward: mirror step_with_custom_loss LoRA backward path exactly.
        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

        auto request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
        request.inv_temperature_gpu = mGrpoInvTemperatureGpu;
        mExecutor->execute_backward(request, comm);

        mLoRAGrads->end_micro_step(main_stream, comm);
        internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
    }

    // Clean up state that was set by forward_for_grpo.
    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }
    mGrpoInvTemperatureGpu = nullptr;
}

void DslModel::step_grpo_native(Tensor inputs,
                                Tensor position_ids,
                                Tensor targets,
                                const float* inference_logprobs_cpu,
                                const float* advantages_cpu,
                                const std::uint8_t* loss_mask_cpu,
                                const std::int32_t* sample_starts_cpu,
                                const std::int32_t* sample_ends_cpu,
                                int sample_count,
                                int grad_accum_steps,
                                int micro_step,
                                NCCLCommunicator& comm,
                                const GrpoNativeLossConfig& loss_config,
                                const float* temperatures_cpu,
                                const float* teacher_logprobs_cpu) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::step_grpo_native called before allocate_run_state()");
    }
    if (!inference_logprobs_cpu || !advantages_cpu || !loss_mask_cpu || !sample_starts_cpu || !sample_ends_cpu) {
        throw std::invalid_argument(
            "step_grpo_native requires inference_logprobs, advantages, loss_mask, sample ranges");
    }
    if (sample_count <= 0) {
        throw std::invalid_argument("step_grpo_native requires at least one sample range");
    }
    if (!(loss_config.loss_scale > 0.0f) || !std::isfinite(loss_config.loss_scale)) {
        throw std::invalid_argument("step_grpo_native loss_scale must be finite and positive");
    }

    auto& rs = *mRunState;
    auto& scratch = rs.grpo_native_scratch();
    cudaStream_t main_stream = rs.MainStream;
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    if (bt > static_cast<std::size_t>(scratch.max_tokens) ||
        static_cast<std::size_t>(sample_count) > static_cast<std::size_t>(scratch.max_samples)) {
        throw std::runtime_error("step_grpo_native inputs exceed allocated GRPO scratch capacity");
    }
    if (micro_step == 0) {
        CUDA_CHECK(cudaMemsetAsync(scratch.metrics.Data,
                                   0,
                                   static_cast<std::size_t>(GRPO_METRIC_COUNT) * sizeof(float),
                                   main_stream));
    }

    const int staging_slot = scratch.next_host_slot;
    scratch.next_host_slot = (scratch.next_host_slot + 1) % modules::GrpoNativeScratch::kHostStagingSlots;
    if (scratch.host_copy_recorded[staging_slot]) {
        CUDA_CHECK(cudaEventSynchronize(scratch.host_copy_done[staging_slot]));
    }

    auto copy_float_input = [main_stream, bt](Tensor& host, Tensor& device, const float* src) {
        std::memcpy(host.get<float>(), src, bt * sizeof(float));
        CUDA_CHECK(cudaMemcpyAsync(device.Data, host.Data, bt * sizeof(float), cudaMemcpyHostToDevice, main_stream));
    };

    copy_float_input(scratch.host_inference_logprobs[staging_slot], scratch.inference_logprobs, inference_logprobs_cpu);
    copy_float_input(scratch.host_advantages[staging_slot], scratch.advantages, advantages_cpu);
    std::memcpy(scratch.host_loss_mask[staging_slot].get<std::uint8_t>(), loss_mask_cpu, bt * sizeof(std::uint8_t));
    CUDA_CHECK(cudaMemcpyAsync(scratch.loss_mask.Data,
                               scratch.host_loss_mask[staging_slot].Data,
                               bt * sizeof(std::uint8_t),
                               cudaMemcpyHostToDevice,
                               main_stream));
    std::memcpy(scratch.host_sample_starts[staging_slot].get<std::int32_t>(),
                sample_starts_cpu,
                static_cast<std::size_t>(sample_count) * sizeof(std::int32_t));
    std::memcpy(scratch.host_sample_ends[staging_slot].get<std::int32_t>(),
                sample_ends_cpu,
                static_cast<std::size_t>(sample_count) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(scratch.sample_starts.Data,
                               scratch.host_sample_starts[staging_slot].Data,
                               static_cast<std::size_t>(sample_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice,
                               main_stream));
    CUDA_CHECK(cudaMemcpyAsync(scratch.sample_ends.Data,
                               scratch.host_sample_ends[staging_slot].Data,
                               static_cast<std::size_t>(sample_count) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice,
                               main_stream));

    const float* teacher_logprobs_gpu = nullptr;
    if (teacher_logprobs_cpu) {
        copy_float_input(scratch.host_teacher_logprobs[staging_slot], scratch.teacher_logprobs, teacher_logprobs_cpu);
        teacher_logprobs_gpu = scratch.teacher_logprobs.get<float>();
    }

    const float* inv_temperature_gpu = nullptr;
    if (temperatures_cpu) {
        auto* inv_temp = scratch.host_temperatures[staging_slot].get<float>();
        for (std::size_t i = 0; i < bt; ++i) {
            inv_temp[i] = 1.0f / temperatures_cpu[i];
        }
        CUDA_CHECK(cudaMemcpyAsync(scratch.inv_temperature.Data,
                                   scratch.host_temperatures[staging_slot].Data,
                                   bt * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
        inv_temperature_gpu = scratch.inv_temperature.get<float>();
    }
    CUDA_CHECK(cudaEventRecord(scratch.host_copy_done[staging_slot], main_stream));
    scratch.host_copy_recorded[staging_slot] = true;

    const bool doc_masking_active =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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

    compute_grpo_custom_dloss(scratch.custom_dloss.get<float>(),
                              scratch.metrics.get<float>(),
                              rs.Losses.get<float>(),
                              scratch.inference_logprobs.get<float>(),
                              scratch.advantages.get<float>(),
                              scratch.loss_mask.get<std::uint8_t>(),
                              teacher_logprobs_gpu,
                              scratch.sample_starts.get<std::int32_t>(),
                              scratch.sample_ends.get<std::int32_t>(),
                              sample_count,
                              static_cast<int>(bt),
                              loss_config.loss_scale,
                              loss_config.ipo_mask_low,
                              loss_config.ipo_mask_high,
                              loss_config.adv_tau,
                              loss_config.teacher_tau,
                              loss_config.kl_tau,
                              main_stream);

    if (!lora_enabled()) {
        auto backward_request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
        backward_request.inv_temperature_gpu = inv_temperature_gpu;
        mExecutor->execute_backward(backward_request, comm);
        if (doc_masking_active) mExecutor->clear_doc_masking();
        return;
    }

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    auto backward_request =
        causal_lm_profile()
            .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
    backward_request.inv_temperature_gpu = inv_temperature_gpu;
    mExecutor->execute_backward(backward_request, comm);
    if (doc_masking_active) mExecutor->clear_doc_masking();
    mLoRAGrads->end_micro_step(main_stream, comm);
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

GrpoNativeMetrics DslModel::consume_grpo_native_metrics() {
    if (!mRunState) {
        throw std::logic_error("DslModel::consume_grpo_native_metrics called before allocate_run_state()");
    }
    auto& rs = *mRunState;
    auto& scratch = rs.grpo_native_scratch();
    CUDA_CHECK(cudaMemcpyAsync(scratch.host_metrics.Data,
                               scratch.metrics.Data,
                               static_cast<std::size_t>(GRPO_METRIC_COUNT) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));

    const auto* values = scratch.host_metrics.get<float>();
    const float sample_count = std::max(values[GRPO_METRIC_SAMPLE_COUNT], 1.0f);
    GrpoNativeMetrics metrics;
    metrics.policy_loss = values[GRPO_METRIC_POLICY_LOSS] / sample_count;
    metrics.mismatch_kl = values[GRPO_METRIC_MISMATCH_KL] / sample_count;
    metrics.masked_mismatch_kl = values[GRPO_METRIC_MASKED_MISMATCH_KL] / sample_count;
    metrics.unmasked_mismatch_kl = values[GRPO_METRIC_UNMASKED_MISMATCH_KL] / sample_count;
    metrics.is_masked = values[GRPO_METRIC_IS_MASKED] / sample_count;
    metrics.is_masked_low = values[GRPO_METRIC_IS_MASKED_LOW] / sample_count;
    metrics.is_masked_high = values[GRPO_METRIC_IS_MASKED_HIGH] / sample_count;
    metrics.teacher_kl = values[GRPO_METRIC_TEACHER_KL] / sample_count;
    metrics.keep_tokens = values[GRPO_METRIC_KEEP_TOKENS];
    metrics.total_tokens = values[GRPO_METRIC_TOTAL_TOKENS];
    return metrics;
}

void DslModel::step_with_kd(Tensor inputs,
                            Tensor position_ids,
                            Tensor targets,
                            const std::int32_t* kd_ids_cpu,
                            const float* kd_logprobs_cpu,
                            int grad_accum_steps,
                            int micro_step,
                            NCCLCommunicator& comm,
                            const KdLossConfig& kd_config) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::step_with_kd called before allocate_run_state()");
    }
    if (!kd_ids_cpu || !kd_logprobs_cpu) {
        throw std::invalid_argument("step_with_kd requires teacher top-K ids and logprobs");
    }
    if (kd_config.top_k <= 0 || kd_config.top_k > 1024) {
        throw std::invalid_argument("step_with_kd: top_k must be in [1, 1024]");
    }
    if (!(kd_config.temperature > 0.0f) || !std::isfinite(kd_config.temperature)) {
        throw std::invalid_argument("step_with_kd: temperature must be finite and positive");
    }
    if (kd_config.kd_weight < 0.0f || kd_config.ce_weight < 0.0f) {
        throw std::invalid_argument("step_with_kd: loss weights must be non-negative");
    }

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);

    rs.set_kd_config(kd_config.top_k, static_cast<long>(bt));
    auto& kd = rs.kd_scratch();
    const std::size_t count = bt * static_cast<std::size_t>(kd_config.top_k);

    // Standard SFT loss semantics: CE loss / ValidTokenCount accumulate over
    // micro-steps and grads get 1/valid_token_count via global_norm_sqrt.
    mUseTokenScale = true;

    if (micro_step == 0) {
        CUDA_CHECK(cudaMemsetAsync(kd.loss_accum, 0, sizeof(float), main_stream));
    }

    const int staging_slot = kd.next_slot;
    kd.next_slot = (kd.next_slot + 1) % DslRunState::KdNativeScratch::kStagingSlots;
    if (kd.copy_recorded[staging_slot]) {
        CUDA_CHECK(cudaEventSynchronize(kd.copy_done[staging_slot]));
    }
    std::memcpy(kd.host_ids[staging_slot], kd_ids_cpu, count * sizeof(std::int32_t));
    std::memcpy(kd.host_logprobs[staging_slot], kd_logprobs_cpu, count * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(kd.topk_ids,
                               kd.host_ids[staging_slot],
                               count * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice,
                               main_stream));
    CUDA_CHECK(cudaMemcpyAsync(kd.topk_logprobs,
                               kd.host_logprobs[staging_slot],
                               count * sizeof(float),
                               cudaMemcpyHostToDevice,
                               main_stream));
    CUDA_CHECK(cudaEventRecord(kd.copy_done[staging_slot], main_stream));
    kd.copy_recorded[staging_slot] = true;

    // Chunked-sequence mode carries per-document geometry in the chunk
    // metadata (per-chunk cu_seqlens on the kvprefix path) — the dense
    // doc-masking context must stay clear.
    mDocMaskingActive =
        mSequenceChunkActive
            ? false
            : causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B_val, T_val);
        if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
            mQLoRAProvider->invalidate_cache();
        }
        mLoRARunState->micro_step = micro_step;
        mLoRARunState->is_training = true;
    }

    auto stamp_kd_fields = [&](ExecutionRequest& request) {
        request.kd_topk_ids_gpu = kd.topk_ids;
        request.kd_topk_logprobs_gpu = kd.topk_logprobs;
        request.kd_loss_accum_gpu = kd.loss_accum;
        request.kd_top_k = kd_config.top_k;
        request.kd_temperature = kd_config.temperature;
        request.kd_weight = kd_config.kd_weight;
        request.kd_ce_weight = kd_config.ce_weight;
    };

    // make_forward_request keeps initialize_loss_buffers = (micro_step == 0)
    // and reads targets from rs.Targets_CPU — the standard training forward.
    rs.Targets_CPU = targets;
    auto forward_request =
        causal_lm_profile().make_forward_request(rs, mModelConfig, mOptions, inputs, position_ids, micro_step);
    stamp_kd_fields(forward_request);
    mExecutor->execute_forward(forward_request, comm);

    if (!lora_enabled()) {
        auto backward_request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        stamp_kd_fields(backward_request);
        mExecutor->execute_backward(backward_request, comm);
        if (mDocMaskingActive) {
            mExecutor->clear_doc_masking();
            mDocMaskingActive = false;
        }
        return;
    }

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    auto backward_request =
        causal_lm_profile()
            .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    stamp_kd_fields(backward_request);
    mExecutor->execute_backward(backward_request, comm);
    if (mDocMaskingActive) {
        mExecutor->clear_doc_masking();
        mDocMaskingActive = false;
    }
    mLoRAGrads->end_micro_step(main_stream, comm);
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

float DslModel::consume_kd_loss_sum() {
    if (!mRunState) {
        throw std::logic_error("DslModel::consume_kd_loss_sum called before allocate_run_state()");
    }
    auto& rs = *mRunState;
    auto& kd = rs.kd_scratch();
    if (!kd.loss_accum) {
        return 0.0f;
    }
    float value = 0.0f;
    CUDA_CHECK(cudaMemcpy(&value, kd.loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(kd.loss_accum, 0, sizeof(float)));
    return value;
}

// ---- Debug-only dispatch-PP sub-range parity (BF16 full-FT, resident) ------

std::vector<float> DslModel::dispatch_pp_forward_hidden(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm) {
    if (lora_enabled()) {
        throw std::runtime_error("dispatch_pp_forward_hidden: BF16 full-FT only (no LoRA)");
    }
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_forward_hidden: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    const CompiledGraph* fwd = ge->compiled_forward();
    if (!fwd) {
        throw std::runtime_error("dispatch_pp_forward_hidden: forward graph not compiled");
    }
    // Run the whole forward eagerly but keep state resident (skip finalize) so the
    // final hidden state can be read before pruning, matching the sub-range path.
    ge->set_forward_op_range(0, fwd->ops.size(), /*skip_init=*/false, /*skip_finalize=*/true, /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    auto out = ge->last_block_hidden_f32();
    ge->clear_forward_op_range();
    return out;
}

std::vector<float> DslModel::dispatch_pp_forward_subranges(Tensor inputs,
                                                           Tensor position_ids,
                                                           NCCLCommunicator& comm,
                                                           int split_after_block) {
    if (lora_enabled()) {
        throw std::runtime_error("dispatch_pp_forward_subranges: BF16 full-FT only (no LoRA)");
    }
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_forward_subranges: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    const CompiledGraph* fwd = ge->compiled_forward();
    if (!fwd) {
        throw std::runtime_error("dispatch_pp_forward_subranges: forward graph not compiled");
    }
    const int num_layers = static_cast<int>(mModelConfig.NumLayers);
    if (split_after_block < 0 || split_after_block >= num_layers - 1) {
        throw std::runtime_error("dispatch_pp_forward_subranges: split must be in [0, num_layers-2]");
    }
    const std::size_t ops_n = fwd->ops.size();
    const std::size_t end_split = fwd->layer_end_indices[static_cast<std::size_t>(split_after_block)];
    const std::size_t start_next = fwd->layer_start_indices[static_cast<std::size_t>(split_after_block + 1)];

    // Segment 1: embedding + blocks [0 .. split]. Keep state resident for resume.
    ge->set_forward_op_range(0, end_split, /*skip_init=*/false, /*skip_finalize=*/true, /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }

    // CPU-boundary handoff: round-trip block ``split``'s output residual.
    ge->roundtrip_block_residual(split_after_block);

    // Segment 2: blocks [split+1 .. last] + final norm, resuming on shared state.
    // Keep state resident (skip finalize) so the final hidden is read before pruning.
    ge->set_forward_op_range(start_next, ops_n, /*skip_init=*/true, /*skip_finalize=*/true, /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    auto out = ge->last_block_hidden_f32();
    ge->clear_forward_op_range();
    return out;
}

void DslModel::dispatch_pp_forward_stage(Tensor inputs,
                                         Tensor position_ids,
                                         NCCLCommunicator& comm,
                                         int lo,
                                         int hi,
                                         std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named,
                                         bool preserve_output) {
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_forward_stage: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    if (lora_enabled()) {
        ensure_lora_run_state(comm, static_cast<int>(B), static_cast<int>(T));
    }
    const CompiledGraph* fwd = ge->compiled_forward();
    if (!fwd) {
        throw std::runtime_error("dispatch_pp_forward_stage: forward graph not compiled");
    }
    const int num_layers = static_cast<int>(mModelConfig.NumLayers);
    if (lo < 0 || hi < lo || hi >= num_layers) {
        throw std::runtime_error("dispatch_pp_forward_stage: invalid block range");
    }
    // Stage [0..] starts at the embedding; a resumed stage [lo>0..] starts at the
    // first op of block lo and consumes the injected boundary residual.
    const std::size_t op_lo = (lo == 0) ? 0 : fwd->layer_start_indices[static_cast<std::size_t>(lo)];
    const std::size_t op_hi =
        (hi == num_layers - 1) ? fwd->ops.size() : fwd->layer_end_indices[static_cast<std::size_t>(hi)];

    if (!inject_named.empty()) {
        ge->set_inject_named(std::move(inject_named));
    }
    // Keep block hi's output (x) live so the next stage's GPU can read it.
    if (preserve_output) {
        ge->set_preserve_layer(hi);
    }
    // skip_finalize keeps the stage's outputs (residual / final hidden) resident
    // for the caller's debug readers and the next stage's boundary read.
    ge->set_forward_op_range(op_lo,
                             op_hi,
                             /*skip_init=*/false,
                             /*skip_finalize=*/true,
                             /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    ge->clear_forward_op_range();
    ge->clear_inject_named();
}

void DslModel::dispatch_pp_backward_stage(Tensor inputs,
                                          Tensor targets,
                                          Tensor position_ids,
                                          NCCLCommunicator& comm,
                                          int lo,
                                          int hi,
                                          bool is_loss_stage,
                                          std::vector<std::pair<std::string, std::vector<std::byte>>> fwd_inject,
                                          std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named,
                                          int micro_step,
                                          int total_micro) {
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_backward_stage: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    ge->ensure_graphs_compiled(B, T);
    if (lora_enabled()) {
        // Per-block LoRA adapters train on this GPU. The pipeline runs total_micro
        // microbatches through this stage; start_micro_step zeros the grad accumulators
        // on micro 0 and accumulates thereafter, so a stage's grads sum over all its
        // microbatches before the optimizer reads them. GradAccumSteps drives the
        // optimizer's per-token normalization (effective batch = total_micro x B).
        ensure_lora_run_state(comm, static_cast<int>(B), static_cast<int>(T));
        lora_grads().start_micro_step(mRunState->MainStream, micro_step, total_micro);
        mRunState->GradAccumSteps = total_micro;
    }
    const CompiledGraph* fwd = ge->compiled_forward();
    const CompiledGraph* bwd = ge->compiled_backward();
    if (!fwd || !bwd) {
        throw std::runtime_error("dispatch_pp_backward_stage: graphs not compiled");
    }
    const int num_layers = static_cast<int>(mModelConfig.NumLayers);
    if (lo < 0 || hi < lo || hi >= num_layers) {
        throw std::runtime_error("dispatch_pp_backward_stage: invalid block range");
    }
    // Provide this stage's forward activations (and saved inputs for recompute) by
    // forwarding ONLY this stage's blocks [lo..hi] from the captured input boundary
    // (fwd_inject = block lo-1's residual). A resumed stage (lo>0) starts at block
    // lo's first op and consumes the injected boundary; stage 0 starts at the
    // embedding. This bounds resident activations to one stage instead of the whole
    // model (the whole-model forward overflowed the compute stack at longer seq).
    // Fallback: a caller that doesn't capture the input boundary (lo>0, empty
    // fwd_inject -- e.g. the grad-norms parity harness) gets a whole-from-start
    // forward, the original behaviour. The stream-driven schedule is avoided
    // (force_linear) -- its per-rank collectives would deadlock on this single GPU.
    const bool stage_bounded = (lo == 0) || !fwd_inject.empty();
    const std::size_t fwd_op_lo =
        (stage_bounded && lo > 0) ? fwd->layer_start_indices[static_cast<std::size_t>(lo)] : 0;
    const std::size_t fwd_op_hi =
        (hi == num_layers - 1) ? fwd->ops.size() : fwd->layer_end_indices[static_cast<std::size_t>(hi)];
    if (!fwd_inject.empty()) {
        ge->set_inject_named(
            fwd_inject);  // copy: re-injected below so the backward recompute of block lo finds its input
    }
    ge->set_forward_op_range(fwd_op_lo,
                             fwd_op_hi,
                             /*skip_init=*/false,
                             /*skip_finalize=*/false,
                             /*force_linear=*/true);
    {
        auto fwd_request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(fwd_request, comm);
    }
    ge->clear_forward_op_range();
    ge->clear_inject_named();
    // Stash the (mean) loss while the per-token Losses buffer is fresh. Only the
    // loss-owning stage's forward ran the loss ops; other stages leave Losses stale.
    // reduce_loss is skipped on the dispatch path, so sum locally (no all-reduce).
    if (is_loss_stage) {
        auto& rs = *mRunState;
        const std::size_t n = static_cast<std::size_t>(rs.B) * static_cast<std::size_t>(rs.T);
        if (rs.Losses.Data && rs.Losses.nelem() >= n && n > 0) {
            std::vector<float> h(n);
            CUDA_CHECK(cudaMemcpyAsync(h.data(),
                                       rs.Losses.template get<float>(),
                                       n * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       rs.MainStream));
            CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
            // Per-token CE is exactly 0 for masked/padding positions and strictly > 0 for
            // valid targets, so count(>0) is the local valid-token count. Normalize the loss
            // by valid (not total B*T) tokens to match the standard per-valid-token loss, and
            // accumulate the count across the step's microbatches for the grad-norm scale.
            double s = 0.0;
            std::size_t valid = 0;
            for (float x : h) {
                s += x;
                if (x > 0.0f) ++valid;
            }
            const std::size_t vc = std::max<std::size_t>(1, valid);
            mDispatchPpLastLoss = static_cast<float>(s / static_cast<double>(vc));
            mDispatchPpLossValidTokens = static_cast<int>(vc);
        }
    }

    // Inject the incoming boundary gradients (d_blocks[hi].*) and re-inject the
    // forward input boundary (blocks[lo-1].*) so the backward's per-block recompute
    // of block lo can rebuild it from its true input.
    for (auto& kv : fwd_inject)
        inject_named.push_back(std::move(kv));
    if (!inject_named.empty()) {
        ge->set_inject_named(std::move(inject_named));
    }
    // One GPU per stage on the full batch; skip the DDP grad all-reduce.
    ge->set_skip_grad_reduce(true);
    // Select this stage's ops by their owning block layer [lo..hi] (the loss-owning
    // stage also runs the lm-head/loss ops). This is robust to boundary view ops
    // (e.g. d_blocks[L].mlp_down -> .mlp_down_flat) whose op index sits between
    // layer_end[L+1] and layer_start[L]: by layer they belong to block L and so
    // run in block L's stage with no inter-stage gap. The op-index range stays
    // full; skip_finalize keeps the stage's input-boundary gradients
    // (d_blocks[lo-1].*) resident for the next (lower) stage's read.
    ge->set_backward_op_range(0,
                              bwd->ops.size(),
                              /*skip_init=*/false,
                              /*skip_finalize=*/true,
                              /*force_linear=*/true);
    // The loss-owning stage runs the leading loss/lm-head ops; the lowest stage
    // (lo == 0) runs the trailing embedding-backward op so its gradient is computed.
    ge->set_backward_layer_range(lo, hi, /*include_loss=*/is_loss_stage, /*include_embed=*/lo == 0);
    {
        auto request =
            causal_lm_profile().make_backward_request(*mRunState, mModelConfig, mOptions, inputs, targets, 1, 0);
        // One GPU holds the full batch; skip the DP loss/valid-token all-reduce
        // (it would deadlock waiting for idle GPUs).
        request.reduce_loss_on_completion = false;
        mExecutor->execute_backward(request, comm);
    }
    ge->clear_backward_op_range();
    ge->clear_backward_layer_range();
    ge->clear_inject_named();
    ge->set_skip_grad_reduce(false);
}

std::vector<float>
DslModel::dispatch_pp_grad_norms_whole(Tensor inputs, Tensor targets, Tensor position_ids, NCCLCommunicator& comm) {
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_grad_norms_whole: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    const CompiledGraph* fwd = ge->compiled_forward();
    const CompiledGraph* bwd = ge->compiled_backward();
    if (!fwd || !bwd) {
        throw std::runtime_error("dispatch_pp_grad_norms_whole: graphs not compiled");
    }
    // Run on one GPU over the full batch through the forced-eager (non-stream-driven)
    // path: the stream-driven schedule issues per-rank weight/grad collectives that
    // would deadlock here (only this GPU participates). Also skip the DDP grad
    // all-reduce for the same reason.
    ge->set_skip_grad_reduce(true);
    ge->set_forward_op_range(0,
                             fwd->ops.size(),
                             /*skip_init=*/false,
                             /*skip_finalize=*/false,
                             /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    ge->clear_forward_op_range();
    ge->set_backward_op_range(0,
                              bwd->ops.size(),
                              /*skip_init=*/false,
                              /*skip_finalize=*/false,
                              /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_backward_request(*mRunState, mModelConfig, mOptions, inputs, targets, 1, 0);
        // One GPU holds the full batch; skip the DP loss/valid-token all-reduce
        // (it would deadlock waiting for idle GPUs).
        request.reduce_loss_on_completion = false;
        mExecutor->execute_backward(request, comm);
    }
    ge->clear_backward_op_range();
    ge->set_skip_grad_reduce(false);
    return ge->block_grad_norms();
}

float DslModel::dispatch_pp_train_step(Tensor inputs,
                                       Tensor targets,
                                       Tensor position_ids,
                                       NCCLCommunicator& comm,
                                       const optimizers::OptimizerConfig& opt_config,
                                       int step_idx) {
    if (lora_enabled()) {
        throw std::runtime_error("dispatch_pp_train_step: BF16 full-FT only (no LoRA)");
    }
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_train_step: requires the DSL GraphExecutor");
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    const CompiledGraph* fwd = ge->compiled_forward();
    const CompiledGraph* bwd = ge->compiled_backward();
    if (!fwd || !bwd) {
        throw std::runtime_error("dispatch_pp_train_step: graphs not compiled");
    }
    // Forced-eager (non-stream-driven) sub-range executor — the single-GPU
    // dispatch-PP path. Forward computes the loss; backward fills the grad store;
    // the optimizer updates the weights from it. This is a single-GPU trainer
    // (world_size==1), so the DP loss/grad reductions are safe local no-ops — keep
    // them so reduce_loss populates ValidTokenCount and get_loss() is meaningful.
    // Match DslModel::backward's contract so the optimizer's grad-norm/scale uses
    // the per-token normalization the grads were produced with (else the scale and
    // the grads disagree and the update diverges).
    mUseTokenScale = true;
    ge->set_forward_op_range(0,
                             fwd->ops.size(),
                             /*skip_init=*/false,
                             /*skip_finalize=*/false,
                             /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    ge->clear_forward_op_range();

    ge->set_backward_op_range(0,
                              bwd->ops.size(),
                              /*skip_init=*/false,
                              /*skip_finalize=*/false,
                              /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_backward_request(*mRunState, mModelConfig, mOptions, inputs, targets, 1, 0);
        request.reduce_loss_on_completion = true;  // populates ValidTokenCount for get_loss()
        mExecutor->execute_backward(request, comm);
    }
    ge->clear_backward_op_range();

    const float loss = get_loss();  // mean loss (raw / valid tokens), available after reduce_loss
    // The optimizer step index is 1-based (Adam bias correction divides by
    // 1 - beta^step; step 0 would be a divide-by-zero -> NaN). Match the trainer's
    // update_with_config convention, which passes step + 1.
    update_with_config(comm, opt_config, step_idx + 1);
    return loss;
}

namespace {
// True if `name` is a parameter of transformer block `layer` ("blocks[L].*").
bool param_is_block(const std::string& name, int layer) {
    const std::string prefix = "blocks[" + std::to_string(layer) + "].";
    return name.rfind(prefix, 0) == 0;
}
bool param_is_any_block(const std::string& name) {
    return name.rfind("blocks[", 0) == 0;
}
std::vector<std::byte> dbg_tensor_bytes_to_host(const Tensor& t, cudaStream_t stream) {
    std::vector<std::byte> host;
    if (!t.Data || t.bytes() == 0) return host;
    host.resize(t.bytes());
    CUDA_CHECK(cudaMemcpyAsync(host.data(), t.Data, t.bytes(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}
}  // namespace

std::vector<std::pair<std::string, std::vector<std::byte>>>
DslModel::dispatch_pp_read_block_grads(int lo, int hi, bool include_head, bool include_embed) {
    std::vector<std::pair<std::string, std::vector<std::byte>>> out;
    // LoRA: only the per-block adapters train (base weights frozen). Collect this
    // stage's blocks' A/B gradients, keyed by (layer, iteration-ordinal, A|B) -- the
    // ordinal is stable across replicas (for_each_lora_layer_weight is deterministic),
    // so write_grads on the master reconstructs the same mapping. head/embed have no
    // LoRA adapters here, so include_head/include_embed don't apply.
    if (lora_enabled()) {
        cudaStream_t stream = mRunState->MainStream;
        auto& grads = lora_grads();
        for (int L = lo; L <= hi; ++L) {
            int ord = 0;
            modules::for_each_lora_layer_weight(grads.block_full(L), [&](modules::LoRATargetId, auto& layer) {
                const std::string base = "lora.g.L" + std::to_string(L) + "." + std::to_string(ord++);
                if (layer.A.Data && layer.A.bytes())
                    out.emplace_back(base + ".A", dbg_tensor_bytes_to_host(layer.A, stream));
                if (layer.B.Data && layer.B.bytes())
                    out.emplace_back(base + ".B", dbg_tensor_bytes_to_host(layer.B, stream));
            });
        }
        return out;
    }
    if (!mGrads) return out;
    cudaStream_t stream = mRunState->MainStream;
    for (const auto& name : mGrads->param_names()) {
        bool wanted = false;
        if (param_is_any_block(name)) {
            for (int L = lo; L <= hi && !wanted; ++L)
                wanted = param_is_block(name, L);
        } else if (name.find("embed") != std::string::npos) {
            wanted = include_embed;  // computed by the lowest stage's embedding backward
        } else {
            wanted = include_head;  // lm_head / final norm, computed by the loss stage
        }
        if (!wanted) continue;
        bool accumulate = false;
        Tensor* g = mGrads->get_param_grad(name, accumulate);
        if (!g || !g->Data || g->bytes() == 0) continue;
        out.emplace_back(name, dbg_tensor_bytes_to_host(*g, stream));
    }
    return out;
}

void DslModel::dispatch_pp_write_grads(const std::vector<std::pair<std::string, std::vector<std::byte>>>& items) {
    if (lora_enabled()) {
        cudaStream_t stream = mRunState->MainStream;
        auto& grads = lora_grads();
        // Cache each block's ordered A/B grad tensors (same stable order as read).
        std::unordered_map<int, std::vector<std::pair<Tensor*, Tensor*>>> block_ptrs;
        auto ptrs_for = [&](int L) -> std::vector<std::pair<Tensor*, Tensor*>>& {
            auto it = block_ptrs.find(L);
            if (it != block_ptrs.end()) return it->second;
            std::vector<std::pair<Tensor*, Tensor*>> v;
            modules::for_each_lora_layer_weight(grads.block_full(L), [&](modules::LoRATargetId, auto& layer) {
                v.emplace_back(&layer.A, &layer.B);
            });
            return block_ptrs.emplace(L, std::move(v)).first->second;
        };
        for (const auto& [name, bytes] : items) {
            int L = -1, ord = -1;
            char ab = 0;
            if (std::sscanf(name.c_str(), "lora.g.L%d.%d.%c", &L, &ord, &ab) != 3) {
                throw std::runtime_error("dispatch_pp_write_grads: bad LoRA grad key '" + name + "'");
            }
            auto& v = ptrs_for(L);
            if (ord < 0 || ord >= static_cast<int>(v.size())) {
                throw std::runtime_error("dispatch_pp_write_grads: ordinal out of range for '" + name + "'");
            }
            Tensor* t = (ab == 'A') ? v[static_cast<std::size_t>(ord)].first : v[static_cast<std::size_t>(ord)].second;
            if (!t || !t->Data || bytes.size() != t->bytes()) {
                throw std::runtime_error("dispatch_pp_write_grads: size mismatch for '" + name + "'");
            }
            CUDA_CHECK(cudaMemcpyAsync(t->Data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    if (!mGrads) return;
    cudaStream_t stream = mRunState->MainStream;
    for (const auto& [name, bytes] : items) {
        bool accumulate = false;
        Tensor* g = mGrads->get_param_grad(name, accumulate);
        if (!g || !g->Data) {
            throw std::runtime_error("dispatch_pp_write_grads: no grad slot for '" + name + "'");
        }
        if (bytes.size() != g->bytes()) {
            throw std::runtime_error("dispatch_pp_write_grads: size mismatch for '" + name + "'");
        }
        CUDA_CHECK(cudaMemcpyAsync(g->Data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<std::pair<std::string, std::vector<std::byte>>> DslModel::dispatch_pp_read_weights() {
    std::vector<std::pair<std::string, std::vector<std::byte>>> out;
    // LoRA: only the adapters change (base frozen), so the master-broadcast carries
    // just the per-block A/B master weights, keyed by (layer, ordinal, A|B).
    if (lora_enabled()) {
        cudaStream_t stream = mRunState->MainStream;
        auto& w = lora_weights();
        const int n = static_cast<int>(mModelConfig.NumLayers);
        for (int L = 0; L < n; ++L) {
            int ord = 0;
            modules::for_each_lora_layer_weight(w.get_master_block(L, stream), [&](modules::LoRATargetId, auto& layer) {
                const std::string base = "lora.w.L" + std::to_string(L) + "." + std::to_string(ord++);
                if (layer.A.Data && layer.A.bytes())
                    out.emplace_back(base + ".A", dbg_tensor_bytes_to_host(layer.A, stream));
                if (layer.B.Data && layer.B.bytes())
                    out.emplace_back(base + ".B", dbg_tensor_bytes_to_host(layer.B, stream));
            });
        }
        return out;
    }
    if (!mParams) return out;
    cudaStream_t stream = mRunState->MainStream;
    for (const auto& name : mParams->param_names()) {
        const Tensor& w = mParams->get(name);
        if (!w.Data || w.bytes() == 0) continue;
        out.emplace_back(name, dbg_tensor_bytes_to_host(w, stream));
    }
    return out;
}

void DslModel::dispatch_pp_write_weights(const std::vector<std::pair<std::string, std::vector<std::byte>>>& items) {
    if (lora_enabled()) {
        cudaStream_t stream = mRunState->MainStream;
        auto& w = lora_weights();
        std::unordered_map<int, std::vector<std::pair<Tensor*, Tensor*>>> block_ptrs;
        auto ptrs_for = [&](int L) -> std::vector<std::pair<Tensor*, Tensor*>>& {
            auto it = block_ptrs.find(L);
            if (it != block_ptrs.end()) return it->second;
            std::vector<std::pair<Tensor*, Tensor*>> v;
            modules::for_each_lora_layer_weight(w.get_master_block(L, stream), [&](modules::LoRATargetId, auto& layer) {
                v.emplace_back(&layer.A, &layer.B);
            });
            return block_ptrs.emplace(L, std::move(v)).first->second;
        };
        for (const auto& [name, bytes] : items) {
            int L = -1, ord = -1;
            char ab = 0;
            if (std::sscanf(name.c_str(), "lora.w.L%d.%d.%c", &L, &ord, &ab) != 3) {
                throw std::runtime_error("dispatch_pp_write_weights: bad LoRA weight key '" + name + "'");
            }
            auto& v = ptrs_for(L);
            if (ord < 0 || ord >= static_cast<int>(v.size())) {
                throw std::runtime_error("dispatch_pp_write_weights: ordinal out of range for '" + name + "'");
            }
            Tensor* t = (ab == 'A') ? v[static_cast<std::size_t>(ord)].first : v[static_cast<std::size_t>(ord)].second;
            if (!t || !t->Data || bytes.size() != t->bytes()) {
                throw std::runtime_error("dispatch_pp_write_weights: size mismatch for '" + name + "'");
            }
            CUDA_CHECK(cudaMemcpyAsync(t->Data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Master weights changed -> next get_block() must re-sync the work copies.
        w.advance_sync_generation();
        return;
    }
    if (!mParams) return;
    cudaStream_t stream = mRunState->MainStream;
    for (const auto& [name, bytes] : items) {
        if (!mParams->has(name)) continue;
        Tensor& w = mParams->get(name);
        if (!w.Data || bytes.size() != w.bytes()) {
            throw std::runtime_error("dispatch_pp_write_weights: size mismatch for '" + name + "'");
        }
        CUDA_CHECK(cudaMemcpyAsync(w.Data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void DslModel::dispatch_pp_zero_grads() {
    zero_grads(mRunState->MainStream);
    CUDA_CHECK(cudaStreamSynchronize(mRunState->MainStream));
}

float DslModel::dispatch_pp_raw_loss() const {
    return mDispatchPpLastLoss;
}

float DslModel::dispatch_pp_apply_optimizer(NCCLCommunicator& comm,
                                            const optimizers::OptimizerConfig& opt_config,
                                            int step_idx) {
    // The dispatch backward skips the DP loss all-reduce (would deadlock), but the loss
    // stage counted valid (non-pad) tokens locally over the step's microbatches. Publish
    // that into ValidTokenCount and use per-valid-token normalization so the grad norm (and
    // the optimizer's grad scale) match the standard path -- not the inflated total-token
    // (incl padding x world_size) fallback. step_idx is already 1-based.
    auto& rs = *mRunState;
    const int vc = std::max(1, mDispatchPpValidTokens);
    if (rs.ValidTokenCount.Data) {
        CUDA_CHECK(cudaMemcpyAsync(rs.ValidTokenCount.Data, &vc, sizeof(int), cudaMemcpyHostToDevice, rs.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
        mUseTokenScale = true;
    } else {
        mUseTokenScale = false;
    }
    // Grads are already complete on this GPU (collected by hand, not via DDP) — tell
    // the optimizer to skip the cross-GPU grad/norm all-reduce (would deadlock).
    mDispatchPpLocalGrads = true;
    update_with_config(comm, opt_config, step_idx);
    mDispatchPpLocalGrads = false;
    mDispatchPpValidTokens = 0;  // reset for the next step
    return get_norm();
}

std::vector<float> DslModel::dispatch_pp_grad_norms_subranges(Tensor inputs,
                                                              Tensor targets,
                                                              Tensor position_ids,
                                                              NCCLCommunicator& comm,
                                                              int split_after_block) {
    if (lora_enabled()) {
        throw std::runtime_error("dispatch_pp_grad_norms_subranges: BF16 full-FT only (no LoRA)");
    }
    GraphExecutor* ge = graph_executor();
    if (!ge) {
        throw std::runtime_error("dispatch_pp_grad_norms_subranges: requires the DSL GraphExecutor");
    }
    // One GPU, full batch; skip the DDP grad all-reduce (deadlocks on multi-GPU).
    ge->set_skip_grad_reduce(true);

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    ge->ensure_graphs_compiled(B, T);
    const CompiledGraph* fwd = ge->compiled_forward();
    const CompiledGraph* bwd = ge->compiled_backward();
    if (!fwd || !bwd) {
        throw std::runtime_error("dispatch_pp_grad_norms_subranges: graphs not compiled");
    }
    // Whole-graph forward (forced-eager: the stream-driven path issues per-rank
    // collectives that deadlock on a single GPU) saves the activations the backward
    // consumes.
    ge->set_forward_op_range(0,
                             fwd->ops.size(),
                             /*skip_init=*/false,
                             /*skip_finalize=*/false,
                             /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_forward_request(*mRunState, mModelConfig, mOptions, inputs, position_ids, 0);
        mExecutor->execute_forward(request, comm);
    }
    ge->clear_forward_op_range();
    const int num_layers = static_cast<int>(mModelConfig.NumLayers);
    if (split_after_block < 0 || split_after_block >= num_layers - 1) {
        throw std::runtime_error("dispatch_pp_grad_norms_subranges: split must be in [0, num_layers-2]");
    }
    // The backward op range [0, ops_n) covers loss + every transformer block in
    // reverse order. Running it through the *bounded, forced-eager* path proves
    // the sub-range executor produces grads identical to the whole-graph path.
    //
    // NOTE: running the block ranges as two SEPARATE backward invocations that
    // share accumulated-gradient state across a CPU boundary is a scheduler
    // concern (it requires multi-stage gradient-accumulation ownership that the
    // executor does not expose) and is deferred to the DispatchScheduler /
    // async-optimizer plans. The forward gate already demonstrates the
    // stop/resume + CPU activation handoff at a block boundary end-to-end.
    const std::size_t ops_n = bwd->ops.size();
    (void)split_after_block;

    ge->set_backward_op_range(0, ops_n, /*skip_init=*/false, /*skip_finalize=*/false, /*force_linear=*/true);
    {
        auto request =
            causal_lm_profile().make_backward_request(*mRunState, mModelConfig, mOptions, inputs, targets, 1, 0);
        // One GPU holds the full batch; skip the DP loss/valid-token all-reduce
        // (it would deadlock waiting for idle GPUs).
        request.reduce_loss_on_completion = false;
        mExecutor->execute_backward(request, comm);
    }
    ge->clear_backward_op_range();
    ge->set_skip_grad_reduce(false);

    return ge->block_grad_norms();
}
void DslModel::step_dpo_native(Tensor inputs,
                               Tensor position_ids,
                               Tensor targets,
                               const float* ref_logprobs_cpu,
                               const std::uint8_t* loss_mask_cpu,
                               const std::int32_t* sample_starts_cpu,
                               const std::int32_t* sample_ends_cpu,
                               int sample_count,
                               const std::int32_t* pair_chosen_cpu,
                               const std::int32_t* pair_rejected_cpu,
                               int pair_count,
                               int grad_accum_steps,
                               int micro_step,
                               NCCLCommunicator& comm,
                               const DpoNativeLossConfig& loss_config) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::step_dpo_native called before allocate_run_state()");
    }
    if (!ref_logprobs_cpu || !loss_mask_cpu || !sample_starts_cpu || !sample_ends_cpu) {
        throw std::invalid_argument("step_dpo_native requires ref_logprobs, loss_mask, sample ranges");
    }
    // pair_count == 0 is a valid degenerate micro-batch (an all-padding row in a
    // sharded layout): the dloss buffer is zeroed and forward/backward still run
    // so NCCL collectives stay matched across ranks.
    if (sample_count < 0 || pair_count < 0) {
        throw std::invalid_argument("step_dpo_native requires non-negative sample and pair counts");
    }
    if (pair_count > 0 && (!pair_chosen_cpu || !pair_rejected_cpu)) {
        throw std::invalid_argument("step_dpo_native requires pair indices when pair_count > 0");
    }
    if (!(loss_config.loss_scale > 0.0f) || !std::isfinite(loss_config.loss_scale)) {
        throw std::invalid_argument("step_dpo_native loss_scale must be finite and positive");
    }
    // Disable the optimizer's implicit divide-by-ValidTokenCount, matching the
    // GRPO native path: DPO normalizes explicitly via loss_scale.
    mUseTokenScale = false;

    auto& rs = *mRunState;
    auto& scratch = rs.grpo_native_scratch();
    cudaStream_t main_stream = rs.MainStream;
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);
    if (bt > static_cast<std::size_t>(scratch.max_tokens) ||
        static_cast<std::size_t>(sample_count) > static_cast<std::size_t>(scratch.max_samples) ||
        static_cast<std::size_t>(pair_count) > static_cast<std::size_t>(scratch.max_samples)) {
        throw std::runtime_error("step_dpo_native inputs exceed allocated GRPO scratch capacity");
    }
    // Reset the (shared) metric accumulators once per optimizer step; the kernel
    // atomicAdds across micro-steps. DPO uses the first 4 slots of the buffer.
    if (micro_step == 0) {
        CUDA_CHECK(cudaMemsetAsync(scratch.metrics.Data, 0, 4 * sizeof(float), main_stream));
    }

    const int staging_slot = scratch.next_host_slot;
    scratch.next_host_slot = (scratch.next_host_slot + 1) % modules::GrpoNativeScratch::kHostStagingSlots;
    if (scratch.host_copy_recorded[staging_slot]) {
        CUDA_CHECK(cudaEventSynchronize(scratch.host_copy_done[staging_slot]));
    }

    // Reference per-token logprobs ride the GRPO inference_logprobs buffer.
    std::memcpy(scratch.host_inference_logprobs[staging_slot].get<float>(), ref_logprobs_cpu, bt * sizeof(float));
    CUDA_CHECK(cudaMemcpyAsync(scratch.inference_logprobs.Data,
                               scratch.host_inference_logprobs[staging_slot].Data,
                               bt * sizeof(float),
                               cudaMemcpyHostToDevice,
                               main_stream));
    std::memcpy(scratch.host_loss_mask[staging_slot].get<std::uint8_t>(), loss_mask_cpu, bt * sizeof(std::uint8_t));
    CUDA_CHECK(cudaMemcpyAsync(scratch.loss_mask.Data,
                               scratch.host_loss_mask[staging_slot].Data,
                               bt * sizeof(std::uint8_t),
                               cudaMemcpyHostToDevice,
                               main_stream));
    if (sample_count > 0) {
        std::memcpy(scratch.host_sample_starts[staging_slot].get<std::int32_t>(),
                    sample_starts_cpu,
                    static_cast<std::size_t>(sample_count) * sizeof(std::int32_t));
        std::memcpy(scratch.host_sample_ends[staging_slot].get<std::int32_t>(),
                    sample_ends_cpu,
                    static_cast<std::size_t>(sample_count) * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpyAsync(scratch.sample_starts.Data,
                                   scratch.host_sample_starts[staging_slot].Data,
                                   static_cast<std::size_t>(sample_count) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
        CUDA_CHECK(cudaMemcpyAsync(scratch.sample_ends.Data,
                                   scratch.host_sample_ends[staging_slot].Data,
                                   static_cast<std::size_t>(sample_count) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
    }
    if (pair_count > 0) {
        std::memcpy(scratch.host_pair_chosen[staging_slot].get<std::int32_t>(),
                    pair_chosen_cpu,
                    static_cast<std::size_t>(pair_count) * sizeof(std::int32_t));
        std::memcpy(scratch.host_pair_rejected[staging_slot].get<std::int32_t>(),
                    pair_rejected_cpu,
                    static_cast<std::size_t>(pair_count) * sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpyAsync(scratch.pair_chosen.Data,
                                   scratch.host_pair_chosen[staging_slot].Data,
                                   static_cast<std::size_t>(pair_count) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
        CUDA_CHECK(cudaMemcpyAsync(scratch.pair_rejected.Data,
                                   scratch.host_pair_rejected[staging_slot].Data,
                                   static_cast<std::size_t>(pair_count) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice,
                                   main_stream));
    }
    CUDA_CHECK(cudaEventRecord(scratch.host_copy_done[staging_slot], main_stream));
    scratch.host_copy_recorded[staging_slot] = true;

    const bool doc_masking_active =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, micro_step);

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
    mExecutor->execute_forward(forward_request, comm);

    compute_dpo_custom_dloss(scratch.custom_dloss.get<float>(),
                             scratch.metrics.get<float>(),
                             rs.Losses.get<float>(),
                             scratch.inference_logprobs.get<float>(),
                             scratch.loss_mask.get<std::uint8_t>(),
                             scratch.sample_starts.get<std::int32_t>(),
                             scratch.sample_ends.get<std::int32_t>(),
                             scratch.pair_chosen.get<std::int32_t>(),
                             scratch.pair_rejected.get<std::int32_t>(),
                             pair_count,
                             static_cast<int>(bt),
                             loss_config.loss_scale,
                             loss_config.beta,
                             loss_config.length_norm ? 1 : 0,
                             main_stream);

    if (!lora_enabled()) {
        auto backward_request =
            causal_lm_profile()
                .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
        backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
        mExecutor->execute_backward(backward_request, comm);
        if (doc_masking_active) mExecutor->clear_doc_masking();
        return;
    }

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    auto backward_request =
        causal_lm_profile()
            .make_backward_request(rs, mModelConfig, mOptions, inputs, targets, grad_accum_steps, micro_step);
    backward_request.custom_dloss_gpu = scratch.custom_dloss.get<float>();
    mExecutor->execute_backward(backward_request, comm);
    if (doc_masking_active) mExecutor->clear_doc_masking();
    mLoRAGrads->end_micro_step(main_stream, comm);
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

DpoNativeMetrics DslModel::consume_dpo_native_metrics() {
    if (!mRunState) {
        throw std::logic_error("DslModel::consume_dpo_native_metrics called before allocate_run_state()");
    }
    auto& rs = *mRunState;
    auto& scratch = rs.grpo_native_scratch();
    CUDA_CHECK(cudaMemcpyAsync(scratch.host_metrics.Data,
                               scratch.metrics.Data,
                               4 * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));

    const auto* values = scratch.host_metrics.get<float>();
    const float pair_count = values[3];
    const float denom = std::max(pair_count, 1.0f);
    DpoNativeMetrics metrics;
    metrics.loss = values[0] / denom;
    metrics.accuracy = values[1] / denom;
    metrics.margin = values[2] / denom;
    metrics.pair_count = pair_count;
    return metrics;
}

std::vector<float>
DslModel::compute_ref_logprobs(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::compute_ref_logprobs called before allocate_run_state()");
    }
    auto& rs = *mRunState;
    const int B_val = static_cast<int>(inputs.Sizes[0]);
    const int T_val = static_cast<int>(inputs.Sizes[1]);
    const std::size_t bt = static_cast<std::size_t>(B_val) * static_cast<std::size_t>(T_val);

    const bool doc_masking_active =
        causal_lm_profile().apply_doc_masking(*mExecutor, mOptions, mModelConfig, inputs, position_ids, 0);

    // BASE-ONLY reference: the DPO reference is the FROZEN start checkpoint, so the LoRA
    // adapter must NOT be applied here — this runs inline per training step, AFTER the
    // policy adapter has diverged (nonzero). (Previously this ran the LoRA-on training
    // forward and only equalled the reference at init where the delta is zero; called
    // mid-training it returned the live policy, making the DPO margin identically 0.)
    // Disable LoRA for this one forward by nulling the executor's run_state
    // (apply_lora_slices_forward early-returns on a null run_state). fp8 activation scaling
    // lives in DslRunState, not the LoRA state, so this still shares the policy forward's
    // per-batch scale: margin == 0 at init (identical base activations) and an exact frozen
    // reference once the policy diverges. Restore the live state afterwards.
    if (lora_enabled()) {
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(),
                                  mLoRAGrads.get(),
                                  /*run_state=*/nullptr);
    }

    auto forward_request =
        causal_lm_profile().make_eval_request(rs, mModelConfig, mOptions, inputs, position_ids, targets, 0);
    forward_request.mode = ExecutionMode::Forward;
    forward_request.reduce_loss_on_completion = false;
    mExecutor->execute_forward(forward_request, comm);

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B_val, T_val);
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(),
                                  mLoRAGrads.get(),
                                  mLoRARunState.get());
    }

    if (doc_masking_active) {
        mExecutor->clear_doc_masking();
    }

    // rs.Losses holds per-token CE = -logprob(target). Return logprob = -CE.
    std::vector<float> logprobs(bt, 0.0f);
    CUDA_CHECK(cudaMemcpyAsync(logprobs.data(),
                               rs.Losses.get<float>(),
                               bt * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               rs.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(rs.MainStream));
    for (auto& v : logprobs) {
        v = -v;
    }
    return logprobs;
}

}  // namespace dsl
