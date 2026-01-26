// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model optimizer update functions.

#include "dsl/dsl_model.h"
#include "dsl/dsl_model_internal.h"
#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "kernels/kernels.h"
#include "modules/fp8_scaling_state.h"
#include "modules/optimizers/adamw_8bit.h"
#include "modules/optimizers/normuon.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

#include <algorithm>
#include <stdexcept>

namespace dsl {

using namespace internal;

void DslModel::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon,
                      float weight_decay, float grad_clip) {
    if (!mRunState || !mParams || !mGrads) {
        throw std::logic_error("DslModel::update called before allocate_run_state()");
    }
    if (lora_enabled()) {
        update_lora_adamw_8bit(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
        return;
    }
    if (!mAdamW8BitState) {
        throw std::logic_error("DslModel::update: optimizer state not allocated");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && mOptions.ShardWeights && (mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && mWeightManager->is_sharded(name);
    };

    // Check if async all-reduce was already started in backward()
    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        if (mGrads->is_reduce_pending()) {
            // Async reduce was started - wait for completion
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            mGrads->clear_reduce_pending();
        } else {
            // Fallback: sync reduce if async wasn't started (e.g., non-last micro-step called update)
            mGrads->reduce_all(comm, stream);
        }
    }

    calculate_gradient_norm(comm, grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (!mAdamW8BitState->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    size_t state_offset = 0;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view = param_sharded ? static_cast<Tensor>(shard_view(*grad, mShardIdx, mNumShards)) : *grad;
        if (param_sharded && grad_view.nelem() != val.nelem()) {
            throw std::runtime_error("DslModel::update: sharded grad size mismatch for " + name);
        }

        float wd = weight_decay;
        if (is_norm_param_name(name) || is_bias_param_name(name)) {
            wd = 0.f;
        }

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad_view.Data) {
            state_offset += n;
            continue;
        }
        const size_t block_offset = state_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()) + state_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + state_offset;
        float* am1 = state.absmax1.template get<float>() + block_offset;
        float* am2 = state.absmax2.template get<float>() + block_offset;
        float* q1 = state.quantiles1.template get<float>();
        float* q2 = state.quantiles2.template get<float>();

        if (val.DType == ETensorDType::FP32) {
            if (grad_view.DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad_view.template get<float>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, stream
                );
            } else if (grad_view.DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad_view.template get<nv_bfloat16>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, stream
                );
            } else {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad_view.DType != ETensorDType::BF16) {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad_view.template get<nv_bfloat16>(),
                s1, s2, n,
                learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                q1, q2, am1, am2, nullptr, nullptr, stream
            );
        } else {
            throw std::runtime_error("DslModel::update: unsupported param dtype for " + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("DslModel::update: state buffer overflow");
        }
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    if (mWeightManager) {
        mWeightManager->invalidate();
    }

    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void DslModel::update_adamw_8bit_graph(NCCLCommunicator& comm, float grad_clip,
                                       const float* opt_params, const int* opt_step) {
    if (!mRunState || !mParams || !mGrads) {
        throw std::logic_error("DslModel::update_adamw_8bit_graph called before allocate_run_state()");
    }
    if (!mAdamW8BitState) {
        throw std::logic_error("DslModel::update_adamw_8bit_graph: optimizer state not allocated");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && mOptions.ShardWeights && (mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && mWeightManager->is_sharded(name);
    };

    // Check if async all-reduce was already started in backward()
    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        if (mGrads->is_reduce_pending()) {
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            mGrads->clear_reduce_pending();
        } else {
            mGrads->reduce_all(comm, stream);
        }
    }

    calculate_gradient_norm(comm, grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    size_t state_offset = 0;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view = param_sharded ? static_cast<Tensor>(shard_view(*grad, mShardIdx, mNumShards)) : *grad;
        if (param_sharded && grad_view.nelem() != val.nelem()) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: sharded grad size mismatch for " + name);
        }

        const float wd_scale = (is_norm_param_name(name) || is_bias_param_name(name)) ? 0.f : 1.f;

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad_view.Data) {
            state_offset += n;
            continue;
        }
        const size_t block_offset = state_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()) + state_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + state_offset;
        float* am1 = state.absmax1.template get<float>() + block_offset;
        float* am2 = state.absmax2.template get<float>() + block_offset;
        float* q1 = state.quantiles1.template get<float>();
        float* q2 = state.quantiles2.template get<float>();

        if (val.DType == ETensorDType::FP32) {
            if (grad_view.DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad_view.template get<float>(),
                    s1, s2, n,
                    /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                    q1, q2, am1, am2, opt_params, opt_step, stream
                );
            } else if (grad_view.DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad_view.template get<nv_bfloat16>(),
                    s1, s2, n,
                    /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                    q1, q2, am1, am2, opt_params, opt_step, stream
                );
            } else {
                throw std::runtime_error("DslModel::update_adamw_8bit_graph: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad_view.DType != ETensorDType::BF16) {
                throw std::runtime_error("DslModel::update_adamw_8bit_graph: unsupported grad dtype for " + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad_view.template get<nv_bfloat16>(),
                s1, s2, n,
                /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                q1, q2, am1, am2, opt_params, opt_step, stream
            );
        } else {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: unsupported param dtype for " + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: state buffer overflow");
        }
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    if (mWeightManager) {
        mWeightManager->invalidate();
    }

    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void DslModel::update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    if (lora_enabled()) {
        switch (config.type) {
            case optimizers::OptimizerType::ADAMW_8BIT:
                update_lora_adamw_8bit(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                                       step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
                return;
            case optimizers::OptimizerType::NORMUON:
                update_lora_normuon(comm, config, step);
                return;
            default:
                throw std::logic_error("DslModel::update_with_config: unsupported optimizer type for LoRA");
        }
    }
    switch (config.type) {
        case optimizers::OptimizerType::ADAMW_8BIT:
            update(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                   step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
            break;
        default:
            throw std::logic_error("DslModel::update_with_config: unsupported optimizer type");
    }
}

void DslModel::update_with_graph_params(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config,
                                        const float* opt_params, const int* opt_step) {
    if (!opt_params || !opt_step) {
        throw std::logic_error("DslModel::update_with_graph_params: missing optimizer parameter buffers");
    }
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT) {
        throw std::logic_error("DslModel::update_with_graph_params: unsupported optimizer type");
    }
    if (lora_enabled()) {
        update_lora_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
        return;
    }
    update_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
}

void DslModel::prepare_optimizer_state_for_graph(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config) {
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT) {
        return;
    }
    if (!mRunState) {
        throw std::logic_error("DslModel::prepare_optimizer_state_for_graph called before allocate_run_state()");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (lora_enabled()) {
        if (!mLoRAAdamW8BitState) {
            throw std::logic_error("DslModel::prepare_optimizer_state_for_graph: LoRA optimizer state not allocated");
        }
        if (!mLoRAAdamW8BitState->initialized) {
            initialize_lora_multi_tensor_state(comm, stream);
            did_work = true;
        }
        if (!mLoRAAdamW8BitState->grad_ptrs_initialized) {
            update_lora_grad_pointers(comm, stream);
            mLoRAAdamW8BitState->grad_ptrs_initialized = true;
            did_work = true;
        }
    } else {
        if (!mAdamW8BitState) {
            throw std::logic_error("DslModel::prepare_optimizer_state_for_graph: optimizer state not allocated");
        }
        if (!mAdamW8BitState->initialized) {
            init_optimizer_state(stream);
            did_work = true;
        }
    }

    if (did_work) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void DslModel::init_optimizer_state(cudaStream_t stream) {
    if (!mAdamW8BitState) {
        throw std::runtime_error("DslModel::init_optimizer_state: optimizer state not allocated");
    }
    auto& state = *mAdamW8BitState;
    if (state.initialized) {
        return;
    }

    if (!state.quantiles1.Data) {
        state.quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
        state.quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
        std::vector<float> h_q1(256), h_q2(256);
        create_adamw8bit_quantiles1(h_q1.data());
        create_adamw8bit_quantiles2(h_q2.data());
        CUDA_CHECK(cudaMemcpy(state.quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    constexpr size_t BLOCK_SIZE = 2048;
    size_t total_params = 0;
    size_t state_elems = 0;
    const bool use_weight_manager = (mWeightManager != nullptr);
    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    for (const auto& name : mGrads->param_names()) {
        Tensor& param = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        add_tensor(param.nelem());
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", {(long)state.total_state_elems});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", {(long)state.total_state_elems});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", {(long)state.num_blocks});

    init_adamw8bit_state(reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()),
                         reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
                         state.absmax1.template get<float>(),
                         state.absmax2.template get<float>(),
                         state.total_state_elems, stream);

    state.initialized = true;
    mAdamWMomentumContainer.update_pointers(&state.state1, &state.absmax1);
    mAdamWVarianceContainer.update_pointers(&state.state2, &state.absmax2);
}

void DslModel::calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream, bool grads_reduced) {
    auto& rs = *mRunState;

    fill_zero(rs.scratch().norm_buffer, stream);
    for (const auto& kv : mGrads->grads()) {
        const Tensor& grad = kv.second;
        if (!grad.Data || grad.nelem() == 0) continue;
        global_norm_squared(rs.scratch().norm_buffer, grad, grad.nelem(), rs.DeviceProp, stream);
    }

    deterministic_sum(rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.nelem(),
                      stream);

    if (!grads_reduced && comm.world_size() > 1) {
        comm.reduce_norm(rs.scratch().norm_buffer.template get<float>(), stream);
    }

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));
    const bool capturing = stream_is_capturing(stream);
    global_norm_sqrt(rs.scratch().norm_buffer.template get<float>(), capturing ? nullptr : rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
    record_event_if_not_capturing(rs.NormDone, stream);
}

ITensorContainer& DslModel::weights() {
    if (lora_enabled()) {
        return *mLoRAWeights;
    }
    return mParams ? static_cast<ITensorContainer&>(*mParams) : mEmpty;
}

ITensorContainer& DslModel::opt_momentum() {
    if (lora_enabled()) {
        return mEmpty;
    }
    return mAdamWMomentumContainer;
}

ITensorContainer& DslModel::opt_momentum_scales() {
    return mEmpty;
}

ITensorContainer& DslModel::opt_variance() {
    if (lora_enabled()) {
        return mEmpty;
    }
    return mAdamWVarianceContainer;
}

ITensorContainer& DslModel::opt_variance_scales() {
    return mEmpty;
}

}  // namespace dsl
