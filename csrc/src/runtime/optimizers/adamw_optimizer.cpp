// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Full-precision FP32 AdamW optimizer. Owns its own state.

#include "runtime/optimizers/adamw_optimizer.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <cuda_bf16.h>

#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/optimizers/adamw.h"
#include "runtime/optimizers/adamw_8bit.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

namespace optimizers {

using namespace dsl::internal;

struct AdamWOptimizer::Impl {
    struct State {
        bool initialized = false;
        size_t total_params = 0;
        size_t total_state_elems = 0;
        bool offload_state = false;
        bool use_zero_copy = false;
        Tensor state1;  // FP32 momentum
        Tensor state2;  // FP32 variance
    };

    std::unique_ptr<State> state;

    void ensure_state() {
        if (!state) {
            state = std::make_unique<State>();
        }
    }
};

AdamWOptimizer::AdamWOptimizer()
    : mImpl(std::make_unique<Impl>()) {
}

AdamWOptimizer::~AdamWOptimizer() = default;

void AdamWOptimizer::init_state(dsl::DslModel& model, cudaStream_t stream) {
    mImpl->ensure_state();
    auto& state = *mImpl->state;
    if (state.initialized) {
        return;
    }

    constexpr size_t BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE;
    size_t total_params = 0;
    size_t state_elems = 0;
    const bool use_weight_manager = (model.mWeightManager != nullptr);
    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    for (const auto& name : model.mGrads->param_names()) {
        Tensor& param = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        add_tensor(param.nelem());
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;

    state.offload_state = model.mOptions.OffloadOptimizer;
    state.use_zero_copy = model.mOptions.UseZeroCopy;
    EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
    if (state.offload_state) {
        if (state.use_zero_copy) {
            alloc_kind = model.mOptions.offload_alloc();
        } else {
            alloc_kind = EAllocationType::ON_DEVICE;
        }
    }

    state.state1 = model.mAllocator->allocate(ETensorDType::FP32,
                                              "adamw_state1",
                                              alloc_kind,
                                              {static_cast<long>(state.total_state_elems)});
    state.state2 = model.mAllocator->allocate(ETensorDType::FP32,
                                              "adamw_state2",
                                              alloc_kind,
                                              {static_cast<long>(state.total_state_elems)});

    const std::size_t bytes1 = state.state1.bytes();
    const std::size_t bytes2 = state.state2.bytes();
    CUDA_CHECK(cudaMemsetAsync(state.state1.Data, 0, bytes1, stream));
    CUDA_CHECK(cudaMemsetAsync(state.state2.Data, 0, bytes2, stream));

    state.initialized = true;
}

void AdamWOptimizer::step(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config, int step_idx) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("AdamWOptimizer::step called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        model.update_lora_adamw(comm,
                                config.learning_rate,
                                config.adamw_beta1,
                                config.adamw_beta2,
                                step_idx,
                                config.adamw_epsilon,
                                config.weight_decay,
                                config.grad_clip);
        model.lora_weights().advance_sync_generation();
        return;
    }

    mImpl->ensure_state();

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (model.mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && model.mOptions.ShardWeights && (model.mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && model.mWeightManager->is_sharded(name);
    };

    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        if (model.mGrads->is_reduce_pending()) {
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            model.mGrads->clear_reduce_pending();
        } else {
            model.mGrads->reduce_all(comm, stream);
        }
    }

    model.calculate_gradient_norm(comm, config.grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (!mImpl->state->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("AdamWOptimizer::step: optimizer state must be initialized before capture");
        }
        init_state(model, stream);
    }

    auto& state = *mImpl->state;
    constexpr size_t BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE;
    size_t state_offset = 0;

    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& name : model.mGrads->param_names()) {
        Tensor& val = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        if (seen_grad_ptrs.count(grad->Data) > 0) {
            state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_offset += val.nelem();
            continue;
        }
        seen_grad_ptrs.insert(grad->Data);

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view =
            param_sharded ? static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards)) : *grad;
        if (param_sharded && grad_view.nelem() != val.nelem()) {
            throw std::runtime_error("AdamWOptimizer::step: sharded grad size mismatch for " + name);
        }

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad_view.Data) {
            state_offset += n;
            continue;
        }

        float* m = state.state1.template get<float>() + state_offset;
        float* v = state.state2.template get<float>() + state_offset;

        if (val.DType == ETensorDType::FP32) {
            if (grad_view.DType == ETensorDType::FP32) {
                adamw_update(val.template get<float>(),
                             grad_view.template get<float>(),
                             m,
                             v,
                             n,
                             config.learning_rate,
                             config.adamw_beta1,
                             config.adamw_beta2,
                             step_idx,
                             config.adamw_epsilon,
                             config.weight_decay,
                             grad_scale,
                             nullptr,
                             nullptr,
                             stream);
            } else if (grad_view.DType == ETensorDType::BF16) {
                adamw_update(val.template get<float>(),
                             grad_view.template get<nv_bfloat16>(),
                             m,
                             v,
                             n,
                             config.learning_rate,
                             config.adamw_beta1,
                             config.adamw_beta2,
                             step_idx,
                             config.adamw_epsilon,
                             config.weight_decay,
                             grad_scale,
                             nullptr,
                             nullptr,
                             stream);
            } else {
                throw std::runtime_error("AdamWOptimizer::step: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad_view.DType == ETensorDType::BF16) {
                adamw_update(val.template get<nv_bfloat16>(),
                             grad_view.template get<nv_bfloat16>(),
                             m,
                             v,
                             n,
                             config.learning_rate,
                             config.adamw_beta1,
                             config.adamw_beta2,
                             step_idx,
                             config.adamw_epsilon,
                             config.weight_decay,
                             grad_scale,
                             nullptr,
                             nullptr,
                             stream);
            } else if (grad_view.DType == ETensorDType::FP32) {
                adamw_update(val.template get<nv_bfloat16>(),
                             grad_view.template get<float>(),
                             m,
                             v,
                             n,
                             config.learning_rate,
                             config.adamw_beta1,
                             config.adamw_beta2,
                             step_idx,
                             config.adamw_epsilon,
                             config.weight_decay,
                             grad_scale,
                             nullptr,
                             nullptr,
                             stream);
            } else {
                throw std::runtime_error("AdamWOptimizer::step: unsupported grad dtype for " + name);
            }
        } else {
            throw std::runtime_error("AdamWOptimizer::step: unsupported param dtype for " + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("AdamWOptimizer::step: state buffer overflow");
        }
    }

    if (model.mWeightManager) {
        model.mWeightManager->invalidate();
        model.mWeightManager->sync_work_from_master(stream);
    }

    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(rs.NormDone));
        if (!std::isfinite(*rs.GradScaleHost)) {
            throw std::runtime_error("AdamWOptimizer::step: grad_scale is NaN/Inf");
        }
    }
    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void AdamWOptimizer::step_graph(dsl::DslModel& model,
                                NCCLCommunicator& comm,
                                const OptimizerConfig& config,
                                const float* opt_params,
                                const int* opt_step) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("AdamWOptimizer::step_graph called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        model.update_lora_adamw_graph(comm, config.grad_clip, opt_params, opt_step);
        model.lora_weights().advance_sync_generation();
        return;
    }

    mImpl->ensure_state();

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (model.mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && model.mOptions.ShardWeights && (model.mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && model.mWeightManager->is_sharded(name);
    };

    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        if (model.mGrads->is_reduce_pending()) {
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            model.mGrads->clear_reduce_pending();
        } else {
            model.mGrads->reduce_all(comm, stream);
        }
    }

    model.calculate_gradient_norm(comm, config.grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (!mImpl->state->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("AdamWOptimizer::step_graph: optimizer state must be initialized before capture");
        }
        init_state(model, stream);
    }

    auto& state = *mImpl->state;
    constexpr size_t BLOCK_SIZE = ADAMW8BIT_BLOCK_SIZE;
    size_t state_offset = 0;

    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& name : model.mGrads->param_names()) {
        Tensor& val = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        if (seen_grad_ptrs.count(grad->Data) > 0) {
            state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_offset += val.nelem();
            continue;
        }
        seen_grad_ptrs.insert(grad->Data);

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view =
            param_sharded ? static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards)) : *grad;
        if (param_sharded && grad_view.nelem() != val.nelem()) {
            throw std::runtime_error("AdamWOptimizer::step_graph: sharded grad size mismatch for " + name);
        }

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad_view.Data) {
            state_offset += n;
            continue;
        }

        float* m = state.state1.template get<float>() + state_offset;
        float* v = state.state2.template get<float>() + state_offset;

        const float wd_scale = 1.f;

        if (val.DType == ETensorDType::FP32) {
            if (grad_view.DType == ETensorDType::FP32) {
                adamw_update(val.template get<float>(),
                             grad_view.template get<float>(),
                             m,
                             v,
                             n,
                             /*lr=*/0.f,
                             /*beta1=*/0.f,
                             /*beta2=*/0.f,
                             /*step=*/1,
                             /*eps=*/0.f,
                             wd_scale,
                             grad_scale,
                             opt_params,
                             opt_step,
                             stream);
            } else if (grad_view.DType == ETensorDType::BF16) {
                adamw_update(val.template get<float>(),
                             grad_view.template get<nv_bfloat16>(),
                             m,
                             v,
                             n,
                             /*lr=*/0.f,
                             /*beta1=*/0.f,
                             /*beta2=*/0.f,
                             /*step=*/1,
                             /*eps=*/0.f,
                             wd_scale,
                             grad_scale,
                             opt_params,
                             opt_step,
                             stream);
            } else {
                throw std::runtime_error("AdamWOptimizer::step_graph: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad_view.DType == ETensorDType::BF16) {
                adamw_update(val.template get<nv_bfloat16>(),
                             grad_view.template get<nv_bfloat16>(),
                             m,
                             v,
                             n,
                             /*lr=*/0.f,
                             /*beta1=*/0.f,
                             /*beta2=*/0.f,
                             /*step=*/1,
                             /*eps=*/0.f,
                             wd_scale,
                             grad_scale,
                             opt_params,
                             opt_step,
                             stream);
            } else if (grad_view.DType == ETensorDType::FP32) {
                adamw_update(val.template get<nv_bfloat16>(),
                             grad_view.template get<float>(),
                             m,
                             v,
                             n,
                             /*lr=*/0.f,
                             /*beta1=*/0.f,
                             /*beta2=*/0.f,
                             /*step=*/1,
                             /*eps=*/0.f,
                             wd_scale,
                             grad_scale,
                             opt_params,
                             opt_step,
                             stream);
            } else {
                throw std::runtime_error("AdamWOptimizer::step_graph: unsupported grad dtype for " + name);
            }
        } else {
            throw std::runtime_error("AdamWOptimizer::step_graph: unsupported param dtype for " + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("AdamWOptimizer::step_graph: state buffer overflow");
        }
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    if (model.mWeightManager) {
        model.mWeightManager->invalidate();
        model.mWeightManager->sync_work_from_master(stream);
    }

    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(rs.NormDone));
        if (!std::isfinite(*rs.GradScaleHost)) {
            throw std::runtime_error("AdamWOptimizer::step_graph: grad_scale is NaN/Inf");
        }
    }
    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void AdamWOptimizer::prepare_for_graph(dsl::DslModel& model,
                                       NCCLCommunicator& comm,
                                       const OptimizerConfig& /*config*/) {
    if (!model.mRunState) {
        throw std::logic_error("AdamWOptimizer::prepare_for_graph called before allocate_run_state()");
    }

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (model.lora_enabled()) {
        if (!model.mLoRAAdamWState) {
            model.mLoRAAdamWState = std::make_unique<modules::LoRAAdamWState>();
        }
        if (!model.mLoRARunState->norm_ptrs_initialized) {
            model.populate_lora_norm_pointers(comm, stream);
            did_work = true;
        }
        if (!model.mLoRAAdamWState->initialized) {
            model.initialize_lora_adamw_state(comm, stream);
            did_work = true;
        }
        if (!model.mLoRAAdamWState->grad_ptrs_initialized) {
            model.update_lora_adamw_grad_pointers(comm, stream);
            model.mLoRAAdamWState->grad_ptrs_initialized = true;
            did_work = true;
        }
    } else {
        mImpl->ensure_state();
        if (!mImpl->state->initialized) {
            init_state(model, stream);
            did_work = true;
        }
    }

    if (did_work) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

}  // namespace optimizers
