// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model optimizer update functions.

#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/optimizers/adamw_8bit.h"
#include "runtime/optimizers/normuon.h"
#include "runtime/optimizers/polar_express.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "runtime/dsl/graph_executor_utils.h"

#include <cuda_bf16.h>

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
    if (!stream_is_capturing(stream)) {
        float grad_scale_host = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&grad_scale_host, grad_scale, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!std::isfinite(grad_scale_host)) {
            throw std::runtime_error("DslModel::update: grad_scale is NaN/Inf");
        }
    }

    if (!mAdamW8BitState->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = optimizers::ADAMW8BIT_BLOCK_SIZE;  // Must match optimizer block size
    size_t state_offset = 0;

    // Track seen gradient pointers to avoid double-updating tied weights (e.g., embedding/lm_head)
    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        // Skip if we've already updated with this gradient (tied weights share the same gradient pointer)
        if (seen_grad_ptrs.count(grad->Data) > 0) {
            // Still advance state_offset to maintain alignment for subsequent params
            state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_offset += val.nelem();
            continue;
        }
        seen_grad_ptrs.insert(grad->Data);

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
    if (!stream_is_capturing(stream)) {
        float grad_scale_host = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&grad_scale_host, grad_scale, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!std::isfinite(grad_scale_host)) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: grad_scale is NaN/Inf");
        }
    }

    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = optimizers::ADAMW8BIT_BLOCK_SIZE;  // Must match optimizer block size
    size_t state_offset = 0;

    // Track seen gradient pointers to avoid double-updating tied weights (e.g., embedding/lm_head)
    std::unordered_set<void*> seen_grad_ptrs_graph;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        // Skip if we've already updated with this gradient (tied weights share the same gradient pointer)
        if (seen_grad_ptrs_graph.count(grad->Data) > 0) {
            // Still advance state_offset to maintain alignment for subsequent params
            state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_offset += val.nelem();
            continue;
        }
        seen_grad_ptrs_graph.insert(grad->Data);

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
        case optimizers::OptimizerType::NORMUON:
            update_normuon(comm, config, step);
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
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT &&
        config.type != optimizers::OptimizerType::NORMUON) {
        throw std::logic_error("DslModel::update_with_graph_params: unsupported optimizer type");
    }
    if (lora_enabled()) {
        if (config.type == optimizers::OptimizerType::NORMUON) {
            throw std::logic_error("DslModel::update_with_graph_params: LoRA NorMuon graph capture not yet supported");
        }
        update_lora_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
        return;
    }
    if (config.type == optimizers::OptimizerType::NORMUON) {
        update_normuon_graph(comm, config.grad_clip, opt_params, opt_step);
    } else {
        update_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
    }
}

void DslModel::prepare_optimizer_state_for_graph(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config) {
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT &&
        config.type != optimizers::OptimizerType::NORMUON) {
        return;
    }
    if (!mRunState) {
        throw std::logic_error("DslModel::prepare_optimizer_state_for_graph called before allocate_run_state()");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (config.type == optimizers::OptimizerType::NORMUON) {
        // NorMuon optimizer state initialization
        if (lora_enabled()) {
            if (!mLoRANorMuonState || !mLoRANorMuonState->initialized) {
                // LoRA NorMuon state is initialized lazily in update_lora_normuon
                // We need to trigger initialization here before graph capture
                // Create a dummy config and call the update path which will initialize
                throw std::logic_error("DslModel::prepare_optimizer_state_for_graph: LoRA NorMuon requires eager initialization; not supported with CUDA graphs yet");
            }
        } else {
            if (!mFFTNorMuonState || !mFFTNorMuonState->initialized) {
                init_normuon_state(stream);
                did_work = true;
            }
        }
    } else {
        // AdamW optimizer state initialization
        if (lora_enabled()) {
            if (!mLoRAAdamW8BitState) {
                throw std::logic_error("DslModel::prepare_optimizer_state_for_graph: LoRA optimizer state not allocated");
            }
            if (!mLoRARunState->norm_ptrs_initialized) {
                populate_lora_norm_pointers(comm, stream);
                did_work = true;
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

    constexpr size_t BLOCK_SIZE = optimizers::ADAMW8BIT_BLOCK_SIZE;
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

    // Determine allocation location based on offload options
    // For adamw8bit, only zero-copy offloading is supported (state accessed directly by GPU).
    state.offload_state = mOptions.OffloadOptimizer;
    state.use_zero_copy = mOptions.UseZeroCopy;
    EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
    if (state.offload_state) {
        if (state.use_zero_copy) {
            alloc_kind = mOptions.offload_alloc();
        } else {
            alloc_kind = EAllocationType::ON_DEVICE;
        }
    }

    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", alloc_kind, {(long)state.total_state_elems});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", alloc_kind, {(long)state.total_state_elems});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", alloc_kind, {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", alloc_kind, {(long)state.num_blocks});

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

    // Track seen Data pointers to avoid double-counting tied gradients (e.g., embedding/lm_head)
    std::unordered_set<void*> seen_ptrs;
    for (const auto& kv : mGrads->grads()) {
        const Tensor& grad = kv.second;
        if (!grad.Data || grad.nelem() == 0) continue;
        // Skip if we've already counted this gradient (tied weights share the same Data pointer)
        if (seen_ptrs.count(grad.Data) > 0) {
            continue;
        }
        seen_ptrs.insert(grad.Data);
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

// ----------------------------------------------------------------------------
// NorMuon optimizer for full fine-tuning (hybrid: AdamW for 1D, NorMuon for 2D)
// ----------------------------------------------------------------------------

DslModel::FFTNorMuonState::~FFTNorMuonState() {
    if (cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

namespace {

// Helper to determine if a parameter should use NorMuon (2D weight matrix) or AdamW (1D/embedding/norm)
bool is_normuon_param(const std::string& name, const Tensor& param) {
    // NorMuon is only for 2D weight matrices, not embeddings, norms, biases, or lm_head
    if (param.Rank != 2) return false;
    if (is_norm_param_name(name)) return false;
    if (is_bias_param_name(name)) return false;

    // Embeddings and lm_head use AdamW
    auto lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("embed") != std::string::npos) return false;
    if (lower.find("lm_head") != std::string::npos) return false;

    // MoE router gates use AdamW (special case from study implementation)
    if (lower.find("router") != std::string::npos) return false;
    if (lower.find("gate") != std::string::npos && lower.find("mlp") == std::string::npos) return false;

    return true;
}

}  // namespace

void DslModel::init_normuon_state(cudaStream_t stream) {
    if (!mFFTNorMuonState) {
        mFFTNorMuonState = std::make_unique<FFTNorMuonState>();
    }
    auto& state = *mFFTNorMuonState;
    if (state.initialized) return;

    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;
    const bool use_weight_manager = (mWeightManager != nullptr);

    // Phase 1: Classify parameters and count state sizes
    state.adamw_total_params = 0;
    state.adamw_state_elems = 0;
    state.normuon_total_params = 0;
    state.normuon_state_elems = 0;
    state.max_weight_M = 0;
    state.max_weight_N = 0;
    state.param_classification.clear();
    state.variance_shapes.clear();

    for (const auto& name : mGrads->param_names()) {
        Tensor& param = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        size_t n = param.nelem();
        bool use_normuon = is_normuon_param(name, param);

        state.param_classification.push_back({name, use_normuon});

        if (use_normuon) {
            state.normuon_total_params += n;
            state.normuon_state_elems = (state.normuon_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state.normuon_state_elems += n;

            int M = static_cast<int>(param.Sizes[0]);
            int N = static_cast<int>(n / static_cast<size_t>(M));
            state.max_weight_M = std::max(state.max_weight_M, static_cast<size_t>(M));
            state.max_weight_N = std::max(state.max_weight_N, static_cast<size_t>(N));
            state.variance_shapes.push_back({M, N});
        } else {
            state.adamw_total_params += n;
            state.adamw_state_elems = (state.adamw_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state.adamw_state_elems += n;
        }
    }

    state.adamw_num_blocks = (state.adamw_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    state.normuon_num_blocks = (state.normuon_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 2: Allocate AdamW state tensors
    if (state.adamw_state_elems > 0) {
        state.adamw_quantiles1 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q1", {256});
        state.adamw_quantiles2 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q2", {256});

        std::vector<float> h_q1(256), h_q2(256);
        create_adamw8bit_quantiles1(h_q1.data());
        create_adamw8bit_quantiles2(h_q2.data());
        CUDA_CHECK(cudaMemcpy(state.adamw_quantiles1.Data, h_q1.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.adamw_quantiles2.Data, h_q2.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

        state.adamw_state1 = mAllocator->allocate(ETensorDType::BYTE, "normuon_adamw_s1", {static_cast<long>(state.adamw_state_elems)});
        state.adamw_state2 = mAllocator->allocate(ETensorDType::BYTE, "normuon_adamw_s2", {static_cast<long>(state.adamw_state_elems)});
        state.adamw_absmax1 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_am1", {static_cast<long>(state.adamw_num_blocks)});
        state.adamw_absmax2 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_am2", {static_cast<long>(state.adamw_num_blocks)});

        init_adamw8bit_state(
            reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()),
            reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()),
            state.adamw_absmax1.template get<float>(),
            state.adamw_absmax2.template get<float>(),
            state.adamw_state_elems, stream
        );
    }

    // Phase 3: Allocate NorMuon state tensors
    if (state.normuon_state_elems > 0) {
        state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "normuon_mom_q", {256});
        std::vector<float> h_mom_q(256);
        optimizers::create_normuon_quantiles(h_mom_q.data());
        CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_mom_q.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

        state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "normuon_mom_s", {static_cast<long>(state.normuon_state_elems)});
        state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "normuon_mom_am", {static_cast<long>(state.normuon_num_blocks)});

        optimizers::init_normuon_momentum_state(
            reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()),
            state.momentum_absmax.template get<float>(),
            state.normuon_state_elems, stream
        );

        // Allocate variance buffers for each 2D weight
        for (const auto& shape : state.variance_shapes) {
            int M = shape.first;
            int N = shape.second;
            size_t var_size = optimizers::normuon_variance_buffer_size(M, N);
            Tensor var_buf = mAllocator->allocate(ETensorDType::FP32, "normuon_var", {static_cast<long>(var_size)});
            fill_constant(var_buf.template get<float>(), 1.0f, var_size, stream);
            state.variance_buffers.push_back(std::move(var_buf));
        }

        // Allocate Polar Express workspace
        size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
        size_t polar_ws_size = optimizers::polar_express_workspace_size(1, static_cast<int>(state.max_weight_M), static_cast<int>(state.max_weight_N));
        size_t total_ws_elems = max_weight_elems + (polar_ws_size / sizeof(nv_bfloat16) + 1);
        state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "normuon_polar_ws", {static_cast<long>(total_ws_elems)});

        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        CUBLAS_CHECK(cublasSetStream(state.cublas_handle, stream));
    }

    state.initialized = true;
}

void DslModel::update_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    if (!mRunState || !mParams || !mGrads) {
        throw std::logic_error("DslModel::update_normuon called before allocate_run_state()");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && mOptions.ShardWeights && (mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && mWeightManager->is_sharded(name);
    };

    // Handle gradient reduction
    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        if (mGrads->is_reduce_pending()) {
            wait_event_if_not_capturing(stream, rs.all_reduce_done_event());
            mGrads->clear_reduce_pending();
        } else {
            mGrads->reduce_all(comm, stream);
        }
    }

    calculate_gradient_norm(comm, config.grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;
    if (!stream_is_capturing(stream)) {
        float grad_scale_host = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&grad_scale_host, grad_scale, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!std::isfinite(grad_scale_host)) {
            throw std::runtime_error("DslModel::update_normuon: grad_scale is NaN/Inf");
        }
    }

    // Initialize state if needed
    if (!mFFTNorMuonState || !mFFTNorMuonState->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_normuon: optimizer state must be initialized before capture");
        }
        init_normuon_state(stream);
    }

    auto& state = *mFFTNorMuonState;
    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

    // Extract hyperparameters
    const float normuon_lr = config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate;
    const float normuon_momentum = config.normuon_momentum;
    const float normuon_beta2 = config.normuon_beta2;
    const float weight_decay = config.weight_decay;
    const bool cautious_wd = config.normuon_cautious_wd;

    const float adamw_lr = config.learning_rate;
    const float adamw_beta1 = config.adamw_beta1;
    const float adamw_beta2 = config.adamw_beta2;
    const float adamw_eps = config.adamw_epsilon;

    // Track offsets
    size_t adamw_offset = 0;
    size_t normuon_offset = 0;
    size_t variance_idx = 0;

    // Track seen gradient pointers to avoid double-updating tied weights
    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& [name, use_normuon] : state.param_classification) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        if (!grad) continue;

        // Skip if we've already updated with this gradient (tied weights)
        if (seen_grad_ptrs.count(grad->Data) > 0) {
            if (use_normuon) {
                normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                normuon_offset += val.nelem();
                variance_idx++;
            } else {
                adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                adamw_offset += val.nelem();
            }
            continue;
        }
        seen_grad_ptrs.insert(grad->Data);

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view = param_sharded ? static_cast<Tensor>(shard_view(*grad, mShardIdx, mNumShards)) : *grad;

        float wd = weight_decay;
        if (is_norm_param_name(name) || is_bias_param_name(name)) {
            wd = 0.f;
        }

        if (use_normuon) {
            // NorMuon update for 2D weight matrices
            normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            int M = static_cast<int>(val.Sizes[0]);
            int N = static_cast<int>(val.nelem() / static_cast<size_t>(M));
            size_t n = val.nelem();
            size_t block_offset = normuon_offset / BLOCK_SIZE;

            unsigned char* mom_state = reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()) + normuon_offset;
            float* mom_absmax = state.momentum_absmax.template get<float>() + block_offset;
            float* quantiles = state.momentum_quantiles.template get<float>();
            float* var_buf = state.variance_buffers[variance_idx].template get<float>();
            nv_bfloat16* workspace = state.polar_workspace.template get<nv_bfloat16>();

            // Apply learning rate multiplier based on weight shape
            float lr_mult = optimizers::normuon_lr_multiplier(M, N);
            float effective_lr = normuon_lr * lr_mult;

            if (val.DType == ETensorDType::BF16 && grad_view.DType == ETensorDType::BF16) {
                optimizers::normuon_update_2d(
                    state.cublas_handle,
                    val.template get<nv_bfloat16>(),
                    grad_view.template get<nv_bfloat16>(),
                    mom_state,
                    var_buf,
                    workspace,
                    M, N,
                    effective_lr,
                    normuon_momentum,
                    normuon_beta2,
                    cautious_wd ? wd : 0.0f,
                    quantiles,
                    mom_absmax,
                    stream
                );
            } else {
                throw std::runtime_error("DslModel::update_normuon: NorMuon requires BF16 weights, got " + name);
            }

            normuon_offset += n;
            variance_idx++;
        } else {
            // AdamW update for 1D params (embeddings, norms, lm_head, etc.)
            adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            size_t n = val.nelem();
            size_t block_offset = adamw_offset / BLOCK_SIZE;

            unsigned char* s1 = reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()) + adamw_offset;
            unsigned char* s2 = reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()) + adamw_offset;
            float* am1 = state.adamw_absmax1.template get<float>() + block_offset;
            float* am2 = state.adamw_absmax2.template get<float>() + block_offset;
            float* q1 = state.adamw_quantiles1.template get<float>();
            float* q2 = state.adamw_quantiles2.template get<float>();

            if (val.DType == ETensorDType::FP32) {
                if (grad_view.DType == ETensorDType::FP32) {
                    adamw_update_8bit(
                        val.template get<float>(),
                        grad_view.template get<float>(),
                        s1, s2, n,
                        adamw_lr, adamw_beta1, adamw_beta2, step, adamw_eps, wd, grad_scale,
                        q1, q2, am1, am2, nullptr, nullptr, stream
                    );
                } else if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(
                        val.template get<float>(),
                        grad_view.template get<nv_bfloat16>(),
                        s1, s2, n,
                        adamw_lr, adamw_beta1, adamw_beta2, step, adamw_eps, wd, grad_scale,
                        q1, q2, am1, am2, nullptr, nullptr, stream
                    );
                }
            } else if (val.DType == ETensorDType::BF16) {
                if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(
                        val.template get<nv_bfloat16>(),
                        grad_view.template get<nv_bfloat16>(),
                        s1, s2, n,
                        adamw_lr, adamw_beta1, adamw_beta2, step, adamw_eps, wd, grad_scale,
                        q1, q2, am1, am2, nullptr, nullptr, stream
                    );
                }
            }

            adamw_offset += n;
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

void DslModel::update_normuon_graph(NCCLCommunicator& comm, float grad_clip,
                                    const float* opt_params, const int* opt_step) {
    if (!mRunState || !mParams || !mGrads) {
        throw std::logic_error("DslModel::update_normuon_graph called before allocate_run_state()");
    }
    if (!mFFTNorMuonState) {
        throw std::logic_error("DslModel::update_normuon_graph: optimizer state not allocated");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    wait_event_if_not_capturing(stream, rs.BackwardDone);

    const bool use_weight_manager = (mWeightManager != nullptr);
    const bool sharded_weights = use_weight_manager && mOptions.ShardWeights && (mNumShards > 1);
    auto param_is_sharded = [&](const std::string& name) -> bool {
        return sharded_weights && mWeightManager->is_sharded(name);
    };

    // Handle gradient reduction
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
    if (!stream_is_capturing(stream)) {
        float grad_scale_host = 0.0f;
        CUDA_CHECK(cudaMemcpyAsync(&grad_scale_host, grad_scale, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (!std::isfinite(grad_scale_host)) {
            throw std::runtime_error("DslModel::update_normuon_graph: grad_scale is NaN/Inf");
        }
    }

    if (!mFFTNorMuonState->initialized) {
        init_normuon_state(stream);
    }

    auto& state = *mFFTNorMuonState;
    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

    // Track offsets
    size_t adamw_offset = 0;
    size_t normuon_offset = 0;
    size_t variance_idx = 0;

    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& [name, use_normuon] : state.param_classification) {
        Tensor& val = use_weight_manager ? mWeightManager->get_master(name) : mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        if (!grad) continue;

        if (seen_grad_ptrs.count(grad->Data) > 0) {
            if (use_normuon) {
                normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                normuon_offset += val.nelem();
                variance_idx++;
            } else {
                adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                adamw_offset += val.nelem();
            }
            continue;
        }
        seen_grad_ptrs.insert(grad->Data);

        const bool param_sharded = param_is_sharded(name);
        Tensor grad_view = param_sharded ? static_cast<Tensor>(shard_view(*grad, mShardIdx, mNumShards)) : *grad;

        // Weight decay scale: 0 for norms/biases, 1 otherwise
        float wd_scale = (is_norm_param_name(name) || is_bias_param_name(name)) ? 0.f : 1.f;

        if (use_normuon) {
            // NorMuon update for 2D weight matrices (graph-compatible)
            normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            int M = static_cast<int>(val.Sizes[0]);
            int N = static_cast<int>(val.nelem() / static_cast<size_t>(M));
            size_t n = val.nelem();
            size_t block_offset = normuon_offset / BLOCK_SIZE;

            unsigned char* mom_state = reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()) + normuon_offset;
            float* mom_absmax = state.momentum_absmax.template get<float>() + block_offset;
            float* quantiles = state.momentum_quantiles.template get<float>();
            float* var_buf = state.variance_buffers[variance_idx].template get<float>();
            nv_bfloat16* workspace = state.polar_workspace.template get<nv_bfloat16>();

            // LR multiplier based on weight shape (applied in kernel)
            float lr_mult = optimizers::normuon_lr_multiplier(M, N);

            if (val.DType == ETensorDType::BF16 && grad_view.DType == ETensorDType::BF16) {
                optimizers::normuon_update_2d_graph(
                    state.cublas_handle,
                    val.template get<nv_bfloat16>(),
                    grad_view.template get<nv_bfloat16>(),
                    mom_state,
                    var_buf,
                    workspace,
                    M, N,
                    lr_mult,
                    wd_scale,
                    quantiles,
                    mom_absmax,
                    opt_params,
                    stream
                );
            } else {
                throw std::runtime_error("DslModel::update_normuon_graph: NorMuon requires BF16 weights, got " + name);
            }

            normuon_offset += n;
            variance_idx++;
        } else {
            // AdamW update for 1D params (graph-compatible)
            adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            size_t n = val.nelem();
            size_t block_offset = adamw_offset / BLOCK_SIZE;

            unsigned char* s1 = reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()) + adamw_offset;
            unsigned char* s2 = reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()) + adamw_offset;
            float* am1 = state.adamw_absmax1.template get<float>() + block_offset;
            float* am2 = state.adamw_absmax2.template get<float>() + block_offset;
            float* q1 = state.adamw_quantiles1.template get<float>();
            float* q2 = state.adamw_quantiles2.template get<float>();

            // AdamW params are at opt_params[4..7] for NorMuon hybrid mode
            // We create a shifted pointer for AdamW
            const float* adamw_opt_params = opt_params + 4;

            if (val.DType == ETensorDType::FP32) {
                if (grad_view.DType == ETensorDType::FP32) {
                    adamw_update_8bit(
                        val.template get<float>(),
                        grad_view.template get<float>(),
                        s1, s2, n,
                        /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                        q1, q2, am1, am2, adamw_opt_params, opt_step, stream
                    );
                } else if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(
                        val.template get<float>(),
                        grad_view.template get<nv_bfloat16>(),
                        s1, s2, n,
                        /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                        q1, q2, am1, am2, adamw_opt_params, opt_step, stream
                    );
                }
            } else if (val.DType == ETensorDType::BF16) {
                if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(
                        val.template get<nv_bfloat16>(),
                        grad_view.template get<nv_bfloat16>(),
                        s1, s2, n,
                        /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                        q1, q2, am1, am2, adamw_opt_params, opt_step, stream
                    );
                }
            }

            adamw_offset += n;
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

}  // namespace dsl
