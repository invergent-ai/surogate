// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_OPTIMIZER_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_OPTIMIZER_H

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include "kernels/kernels.h"
#include "lora_model_core.h"
#include "lora_optimizer_state.h"
#include "modules/optimizers/adamw_8bit.h"
#include "modules/optimizers/normuon.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {

template<typename Block>
void ModularLoRAModel<Block>::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                     int t, float epsilon, float weight_decay, float grad_clip) {
    if (!lora_enabled()) {
        mBaseModel->update(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
        return;
    }

    // Only 8-bit AdamW optimizer is supported
    if (!mLoRAAdamW8BitState) {
        throw std::logic_error("ModularLoRAModel::update: 8-bit optimizer state not allocated");
    }
    update_adamw_8bit(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
}

template<typename Block>
void ModularLoRAModel<Block>::update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    if (!lora_enabled()) {
        // Delegate to base model which supports all optimizers for full fine-tuning
        mBaseModel->update_with_config(comm, config, step);
        return;
    }

    switch (config.type) {
        case optimizers::OptimizerType::ADAMW_8BIT:
            if (!mLoRAAdamW8BitState) {
                throw std::logic_error("ModularLoRAModel::update_with_config(ADAMW_8BIT): "
                                       "optimizer state not allocated");
            }
            update_adamw_8bit(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                             step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
            break;

        case optimizers::OptimizerType::NORMUON:
            if (step == 1) {
                fmt::print("LoRA NorMuon optimizer selected (step {})\n", step);
            }
            update_normuon(comm, config, step);
            break;

        default:
            throw std::logic_error("ModularLoRAModel::update_with_config(): unsupported optimizer type");
    }
}

template<typename Block>
void ModularLoRAModel<Block>::update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                                int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = mBaseModel->run_state();
    cudaStream_t main_stream = rs.MainStream;
    auto& state = *mLoRAAdamW8BitState;

    // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
    calculate_lora_gradient_norm(comm, grad_clip);
    const float* grad_scale = mLoRARunState->norm_buffer.template get<float>() + 1;

    // Initialize multi-tensor optimizer state on first call
    if (!state.initialized) {
        initialize_multi_tensor_state(comm, main_stream);
    }

    // Update grad pointers each step (grads change between steps due to accumulation)
    update_grad_pointers(comm, main_stream);

    // Single kernel launch for all LoRA tensors - dispatch based on lora_dtype
    const ETensorDType lora_dtype = mLoRAConfig.dtype;
    if (lora_dtype == ETensorDType::FP32) {
        adamw_update_8bit_multi_tensor(
            reinterpret_cast<float**>(state.param_ptrs.Data),
            reinterpret_cast<float**>(state.grad_ptrs.Data),
            state.tensor_sizes.template get<int>(),
            state.num_tensors,
            reinterpret_cast<unsigned char*>(state.state1.Data),
            reinterpret_cast<unsigned char*>(state.state2.Data),
            state.absmax1.template get<float>(),
            state.absmax2.template get<float>(),
            state.state_offsets.template get<int>(),
            state.total_params,
            learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
            state.quantiles1.template get<float>(),
            state.quantiles2.template get<float>(),
            main_stream
        );
    } else if (lora_dtype == ETensorDType::BF16) {
        adamw_update_8bit_multi_tensor(
            reinterpret_cast<nv_bfloat16**>(state.param_ptrs.Data),
            reinterpret_cast<nv_bfloat16**>(state.grad_ptrs.Data),
            state.tensor_sizes.template get<int>(),
            state.num_tensors,
            reinterpret_cast<unsigned char*>(state.state1.Data),
            reinterpret_cast<unsigned char*>(state.state2.Data),
            state.absmax1.template get<float>(),
            state.absmax2.template get<float>(),
            state.state_offsets.template get<int>(),
            state.total_params,
            learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
            state.quantiles1.template get<float>(),
            state.quantiles2.template get<float>(),
            main_stream
        );
    } else {
        throw std::runtime_error(fmt::format(
            "ModularLoRAModel: unsupported lora_dtype {} for 8-bit AdamW optimizer",
            dtype_to_str(lora_dtype)));
    }

    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
}

template<typename Block>
void ModularLoRAModel<Block>::update_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    NVTX_RANGE_FN();
    auto& rs = mBaseModel->get_run_state();
    cudaStream_t main_stream = rs.MainStream;

    if (!mLoRANorMuonState) {
        mLoRANorMuonState = std::make_unique<LoRANorMuonState>();
    }
    auto& state = *mLoRANorMuonState;

    calculate_lora_gradient_norm(comm, config.grad_clip);

    const float lr = config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate;
    const float momentum = config.normuon_momentum;
    const float beta2 = config.normuon_beta2;
    const float weight_decay = config.weight_decay;
    const bool cautious_wd = config.normuon_cautious_wd;
    const int L = (int)mBaseModel->config().NumLayers;

    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

    if (!state.initialized) {
        state.total_params = 0;
        state.state_elems = 0;
        state.max_weight_M = 0;
        state.max_weight_N = 0;

        auto add_param = [&](const Tensor& weight) {
            if (!weight.Data) return;
            size_t n = weight.nelem();
            state.total_params += n;
            state.state_elems = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state.state_elems += n;

            int M = 1, N = (int)n;
            if (weight.Rank >= 2) {
                M = (int)weight.Sizes[0];
                N = (int)(n / M);
            }
            state.max_weight_M = std::max(state.max_weight_M, (size_t)M);
            state.max_weight_N = std::max(state.max_weight_N, (size_t)N);

            state.variance_shapes.push_back({M, N});
        };

        for (int l = 0; l < L; ++l) {
            auto& lora_w = mLoRAWeights->get_master_block(l, main_stream);

            if (lora_w.attention.q.has_value()) { add_param(lora_w.attention.q->A); add_param(lora_w.attention.q->B); }
            if (lora_w.attention.k.has_value()) { add_param(lora_w.attention.k->A); add_param(lora_w.attention.k->B); }
            if (lora_w.attention.v.has_value()) { add_param(lora_w.attention.v->A); add_param(lora_w.attention.v->B); }
            if (lora_w.attention.o.has_value()) { add_param(lora_w.attention.o->A); add_param(lora_w.attention.o->B); }
            if (lora_w.mlp.gate.has_value()) { add_param(lora_w.mlp.gate->A); add_param(lora_w.mlp.gate->B); }
            if (lora_w.mlp.up.has_value()) { add_param(lora_w.mlp.up->A); add_param(lora_w.mlp.up->B); }
            if (lora_w.mlp.down.has_value()) { add_param(lora_w.mlp.down->A); add_param(lora_w.mlp.down->B); }

            if (lora_w.moe.use_grouped) {
                if (lora_w.moe.grouped.gate.has_value()) { add_param(lora_w.moe.grouped.gate->A); add_param(lora_w.moe.grouped.gate->B); }
                if (lora_w.moe.grouped.up.has_value()) { add_param(lora_w.moe.grouped.up->A); add_param(lora_w.moe.grouped.up->B); }
                if (lora_w.moe.grouped.down.has_value()) { add_param(lora_w.moe.grouped.down->A); add_param(lora_w.moe.grouped.down->B); }
            } else {
                for (auto& expert : lora_w.moe.experts) {
                    if (expert.gate.has_value()) { add_param(expert.gate->A); add_param(expert.gate->B); }
                    if (expert.up.has_value()) { add_param(expert.up->A); add_param(expert.up->B); }
                    if (expert.down.has_value()) { add_param(expert.down->A); add_param(expert.down->B); }
                }
            }

            // Router LoRA (when train_router is enabled)
            if (lora_w.router.has_value() && lora_w.router->has_value()) {
                add_param(lora_w.router->A);
                add_param(lora_w.router->B);
            }
        }

        state.num_blocks = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_quantiles", {256});
        std::vector<float> h_quantiles(256);
        optimizers::create_normuon_quantiles(h_quantiles.data());
        CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_quantiles.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));

        state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "lora_normuon_momentum",
                                                     {static_cast<long>(state.state_elems)});
        state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_absmax",
                                                      {static_cast<long>(state.num_blocks)});

        optimizers::init_normuon_momentum_state(
            reinterpret_cast<unsigned char*>(state.momentum_state.Data),
            state.momentum_absmax.template get<float>(),
            state.state_elems,
            main_stream
        );

        for (const auto& shape : state.variance_shapes) {
            int M = shape.first;
            int N = shape.second;
            size_t var_size = optimizers::normuon_variance_buffer_size(M, N);
            Tensor var_buf = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_variance",
                                                   {static_cast<long>(var_size)});
            std::vector<float> ones(var_size, 1.0f);
            CUDA_CHECK(cudaMemcpyAsync(var_buf.Data, ones.data(),
                                       var_size * sizeof(float), cudaMemcpyHostToDevice, main_stream));
            state.variance_buffers.push_back(std::move(var_buf));
        }

        size_t max_dim = std::max(state.max_weight_M, state.max_weight_N);
        size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
        size_t polar_workspace_elems = 4 * max_dim * max_dim + 1;
        size_t polar_size = max_weight_elems + polar_workspace_elems;
        state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_polar",
                                                      {static_cast<long>(polar_size)});

        size_t max_weight_size = state.max_weight_M * state.max_weight_N;
        state.momentum_temp = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_temp",
                                                    {static_cast<long>(max_weight_size)});

        CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        CUBLAS_CHECK(cublasSetStream(state.cublas_handle, main_stream));

        state.initialized = true;
    }

    const ETensorDType lora_dtype = mLoRAConfig.dtype;
    size_t state_offset = 0;
    size_t var_idx = 0;
    bool unused_acc = false;

    auto update_param = [&](Tensor& param, Tensor& grad) {
        if (!param.Data) return;

        const auto& shape = state.variance_shapes[var_idx];
        int M = shape.first;
        int N = shape.second;
        size_t n = param.nelem();

        size_t aligned_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        unsigned char* momentum_ptr = reinterpret_cast<unsigned char*>(state.momentum_state.Data) + aligned_offset;
        float* absmax_ptr = state.momentum_absmax.template get<float>() + (aligned_offset / BLOCK_SIZE);
        float* variance_ptr = state.variance_buffers[var_idx].template get<float>();

        if (lora_dtype == ETensorDType::BF16) {
            optimizers::normuon_update_2d(
                state.cublas_handle,
                param.template get<nv_bfloat16>(),
                grad.template get<nv_bfloat16>(),
                momentum_ptr,
                variance_ptr,
                state.polar_workspace.template get<nv_bfloat16>(),
                M, N,
                lr,
                momentum,
                beta2,
                cautious_wd ? weight_decay : 0.0f,
                state.momentum_quantiles.template get<float>(),
                absmax_ptr,
                main_stream
            );
        } else {
            throw std::runtime_error("LoRA NorMuon optimizer currently only supports BF16 LoRA weights");
        }

        state_offset = aligned_offset + n;
        var_idx++;
    };

    for (int l = 0; l < L; ++l) {
        auto& lora_w = mLoRAWeights->get_master_block(l, main_stream);
        auto& lora_g = mLoRAGrads->get_block_full(l, main_stream, comm, unused_acc);

        if (lora_w.attention.q.has_value()) { update_param(lora_w.attention.q->A, lora_g.attention.q->A); update_param(lora_w.attention.q->B, lora_g.attention.q->B); }
        if (lora_w.attention.k.has_value()) { update_param(lora_w.attention.k->A, lora_g.attention.k->A); update_param(lora_w.attention.k->B, lora_g.attention.k->B); }
        if (lora_w.attention.v.has_value()) { update_param(lora_w.attention.v->A, lora_g.attention.v->A); update_param(lora_w.attention.v->B, lora_g.attention.v->B); }
        if (lora_w.attention.o.has_value()) { update_param(lora_w.attention.o->A, lora_g.attention.o->A); update_param(lora_w.attention.o->B, lora_g.attention.o->B); }
        if (lora_w.mlp.gate.has_value()) { update_param(lora_w.mlp.gate->A, lora_g.mlp.gate->A); update_param(lora_w.mlp.gate->B, lora_g.mlp.gate->B); }
        if (lora_w.mlp.up.has_value()) { update_param(lora_w.mlp.up->A, lora_g.mlp.up->A); update_param(lora_w.mlp.up->B, lora_g.mlp.up->B); }
        if (lora_w.mlp.down.has_value()) { update_param(lora_w.mlp.down->A, lora_g.mlp.down->A); update_param(lora_w.mlp.down->B, lora_g.mlp.down->B); }

        if (lora_w.moe.use_grouped) {
            if (lora_w.moe.grouped.gate.has_value() && lora_g.moe.grouped.gate.has_value()) {
                update_param(lora_w.moe.grouped.gate->A, lora_g.moe.grouped.gate->A);
                update_param(lora_w.moe.grouped.gate->B, lora_g.moe.grouped.gate->B);
            }
            if (lora_w.moe.grouped.up.has_value() && lora_g.moe.grouped.up.has_value()) {
                update_param(lora_w.moe.grouped.up->A, lora_g.moe.grouped.up->A);
                update_param(lora_w.moe.grouped.up->B, lora_g.moe.grouped.up->B);
            }
            if (lora_w.moe.grouped.down.has_value() && lora_g.moe.grouped.down.has_value()) {
                update_param(lora_w.moe.grouped.down->A, lora_g.moe.grouped.down->A);
                update_param(lora_w.moe.grouped.down->B, lora_g.moe.grouped.down->B);
            }
        } else {
            for (std::size_t e = 0; e < lora_w.moe.experts.size() && e < lora_g.moe.experts.size(); ++e) {
                auto& w_exp = lora_w.moe.experts[e];
                auto& g_exp = lora_g.moe.experts[e];
                if (w_exp.gate.has_value() && g_exp.gate.has_value()) { update_param(w_exp.gate->A, g_exp.gate->A); update_param(w_exp.gate->B, g_exp.gate->B); }
                if (w_exp.up.has_value() && g_exp.up.has_value()) { update_param(w_exp.up->A, g_exp.up->A); update_param(w_exp.up->B, g_exp.up->B); }
                if (w_exp.down.has_value() && g_exp.down.has_value()) { update_param(w_exp.down->A, g_exp.down->A); update_param(w_exp.down->B, g_exp.down->B); }
            }
        }

        // Router LoRA update (gradients computed in backward hook)
        if (lora_w.router.has_value() && lora_w.router->has_value()) {
            if (lora_g.router.has_value()) {
                update_param(lora_w.router->A, lora_g.router->A);
                update_param(lora_w.router->B, lora_g.router->B);
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
}

template<typename Block>
void ModularLoRAModel<Block>::calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = mBaseModel->get_run_state();
    cudaStream_t stream = rs.MainStream;

    // Ensure backward has completed before reading LoRA gradients.
    CUDA_CHECK(cudaStreamWaitEvent(stream, rs.BackwardDone));

    Tensor& buf = mLoRARunState->norm_buffer;
    fill_zero(buf, stream);

    auto norm_squared = [&](const Tensor& grad) {
        if (grad.Data) {
            global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
        }
    };

    const auto& cfg = mBaseModel->config();
    for (int l = 0; l < cfg.NumLayers; ++l) {
        bool unused_acc = false;
        auto& g = mLoRAGrads->get_block_full(l, stream, comm, unused_acc);

        if (g.attention.q.has_value()) { norm_squared(g.attention.q->A); norm_squared(g.attention.q->B); }
        if (g.attention.k.has_value()) { norm_squared(g.attention.k->A); norm_squared(g.attention.k->B); }
        if (g.attention.v.has_value()) { norm_squared(g.attention.v->A); norm_squared(g.attention.v->B); }
        if (g.attention.o.has_value()) { norm_squared(g.attention.o->A); norm_squared(g.attention.o->B); }
        if (g.mlp.gate.has_value()) { norm_squared(g.mlp.gate->A); norm_squared(g.mlp.gate->B); }
        if (g.mlp.up.has_value()) { norm_squared(g.mlp.up->A); norm_squared(g.mlp.up->B); }
        if (g.mlp.down.has_value()) { norm_squared(g.mlp.down->A); norm_squared(g.mlp.down->B); }

        if (g.moe.use_grouped) {
            if (g.moe.grouped.gate.has_value()) { norm_squared(g.moe.grouped.gate->A); norm_squared(g.moe.grouped.gate->B); }
            if (g.moe.grouped.up.has_value()) { norm_squared(g.moe.grouped.up->A); norm_squared(g.moe.grouped.up->B); }
            if (g.moe.grouped.down.has_value()) { norm_squared(g.moe.grouped.down->A); norm_squared(g.moe.grouped.down->B); }
        } else {
            for (const auto& expert : g.moe.experts) {
                if (expert.gate.has_value()) { norm_squared(expert.gate->A); norm_squared(expert.gate->B); }
                if (expert.up.has_value()) { norm_squared(expert.up->A); norm_squared(expert.up->B); }
                if (expert.down.has_value()) { norm_squared(expert.down->A); norm_squared(expert.down->B); }
            }
        }

        // Router LoRA gradients (when train_router is enabled)
        if (g.router.has_value()) { norm_squared(g.router->A); norm_squared(g.router->B); }
    }

    deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem() - 2, stream);

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));

    global_norm_sqrt(buf.template get<float>(), rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
    CUDA_CHECK(cudaEventRecord(rs.NormDone, stream));
}

template<typename Block>
void ModularLoRAModel<Block>::initialize_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream) {
    auto& state = *mLoRAAdamW8BitState;
    const int L = (int)mBaseModel->config().NumLayers;

    std::vector<void*> h_param_ptrs;
    std::vector<int> h_sizes;
    std::vector<int> h_state_offsets;
    size_t total_params = 0;

    auto collect_tensor = [&](Tensor& param) {
        if (!param.Data) return;
        h_param_ptrs.push_back(param.Data);
        int n = (int)param.nelem();
        h_sizes.push_back(n);
        h_state_offsets.push_back((int)total_params);
        total_params += n;
    };

    for (int l = 0; l < L; ++l) {
        auto& lora_w = mLoRAWeights->get_master_block(l, stream);

        if (lora_w.attention.q.has_value()) { collect_tensor(lora_w.attention.q->A); collect_tensor(lora_w.attention.q->B); }
        if (lora_w.attention.k.has_value()) { collect_tensor(lora_w.attention.k->A); collect_tensor(lora_w.attention.k->B); }
        if (lora_w.attention.v.has_value()) { collect_tensor(lora_w.attention.v->A); collect_tensor(lora_w.attention.v->B); }
        if (lora_w.attention.o.has_value()) { collect_tensor(lora_w.attention.o->A); collect_tensor(lora_w.attention.o->B); }
        if (lora_w.mlp.gate.has_value()) { collect_tensor(lora_w.mlp.gate->A); collect_tensor(lora_w.mlp.gate->B); }
        if (lora_w.mlp.up.has_value()) { collect_tensor(lora_w.mlp.up->A); collect_tensor(lora_w.mlp.up->B); }
        if (lora_w.mlp.down.has_value()) { collect_tensor(lora_w.mlp.down->A); collect_tensor(lora_w.mlp.down->B); }

        if (lora_w.moe.use_grouped) {
            if (lora_w.moe.grouped.gate.has_value()) { collect_tensor(lora_w.moe.grouped.gate->A); collect_tensor(lora_w.moe.grouped.gate->B); }
            if (lora_w.moe.grouped.up.has_value()) { collect_tensor(lora_w.moe.grouped.up->A); collect_tensor(lora_w.moe.grouped.up->B); }
            if (lora_w.moe.grouped.down.has_value()) { collect_tensor(lora_w.moe.grouped.down->A); collect_tensor(lora_w.moe.grouped.down->B); }
        } else {
            for (auto& expert : lora_w.moe.experts) {
                if (expert.gate.has_value()) { collect_tensor(expert.gate->A); collect_tensor(expert.gate->B); }
                if (expert.up.has_value()) { collect_tensor(expert.up->A); collect_tensor(expert.up->B); }
                if (expert.down.has_value()) { collect_tensor(expert.down->A); collect_tensor(expert.down->B); }
            }
        }

        // Router LoRA (when train_router is enabled)
        if (lora_w.router.has_value() && lora_w.router->has_value()) {
            collect_tensor(lora_w.router->A);
            collect_tensor(lora_w.router->B);
        }
    }

    state.num_tensors = (int)h_param_ptrs.size();
    state.total_params = total_params;
    constexpr size_t BLOCK_SIZE = 2048;
    state.num_blocks = (total_params + BLOCK_SIZE - 1) / BLOCK_SIZE;

    state.param_ptrs = mAllocator->allocate(ETensorDType::BYTE, "lora_mt_param_ptrs", EAllocationType::ON_DEVICE, {(long)(state.num_tensors * sizeof(void*))});
    state.grad_ptrs = mAllocator->allocate(ETensorDType::BYTE, "lora_mt_grad_ptrs", EAllocationType::ON_DEVICE, {(long)(state.num_tensors * sizeof(void*))});
    state.tensor_sizes = mAllocator->allocate(ETensorDType::INT32, "lora_mt_sizes", EAllocationType::ON_DEVICE, {(long)state.num_tensors});
    state.state_offsets = mAllocator->allocate(ETensorDType::INT32, "lora_mt_offsets", EAllocationType::ON_DEVICE, {(long)state.num_tensors});

    CUDA_CHECK(cudaMemcpyAsync(state.param_ptrs.Data, h_param_ptrs.data(), h_param_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.tensor_sizes.Data, h_sizes.data(), h_sizes.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.state_offsets.Data, h_state_offsets.data(), h_state_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state1", EAllocationType::ON_DEVICE, {(long)total_params});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state2", EAllocationType::ON_DEVICE, {(long)total_params});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax1", EAllocationType::ON_DEVICE, {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax2", EAllocationType::ON_DEVICE, {(long)state.num_blocks});

    init_adamw8bit_state(reinterpret_cast<unsigned char*>(state.state1.Data), reinterpret_cast<unsigned char*>(state.state2.Data),
        state.absmax1.template get<float>(), state.absmax2.template get<float>(), total_params, stream);

    state.initialized = true;
}

template<typename Block>
void ModularLoRAModel<Block>::update_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream) {
    auto& state = *mLoRAAdamW8BitState;
    const int L = (int)mBaseModel->config().NumLayers;

    std::vector<void*> h_grad_ptrs;
    h_grad_ptrs.reserve(state.num_tensors);
    bool unused_acc = false;

    auto collect_grad = [&](std::optional<LoRALayerWeights<Tensor>>& grad_opt) {
        if (!grad_opt.has_value()) return;
        if (grad_opt->A.Data) h_grad_ptrs.push_back(grad_opt->A.Data);
        if (grad_opt->B.Data) h_grad_ptrs.push_back(grad_opt->B.Data);
    };

    auto collect_grouped_grad = [&](std::optional<LoRAGroupedLayerWeights<Tensor>>& grad_opt) {
        if (!grad_opt.has_value()) return;
        if (grad_opt->A.Data) h_grad_ptrs.push_back(grad_opt->A.Data);
        if (grad_opt->B.Data) h_grad_ptrs.push_back(grad_opt->B.Data);
    };

    for (int l = 0; l < L; ++l) {
        auto& lora_g = mLoRAGrads->get_block_full(l, stream, comm, unused_acc);
        collect_grad(lora_g.attention.q);
        collect_grad(lora_g.attention.k);
        collect_grad(lora_g.attention.v);
        collect_grad(lora_g.attention.o);
        collect_grad(lora_g.mlp.gate);
        collect_grad(lora_g.mlp.up);
        collect_grad(lora_g.mlp.down);

        if (lora_g.moe.use_grouped) {
            collect_grouped_grad(lora_g.moe.grouped.gate);
            collect_grouped_grad(lora_g.moe.grouped.up);
            collect_grouped_grad(lora_g.moe.grouped.down);
        } else {
            for (auto& expert : lora_g.moe.experts) {
                collect_grad(expert.gate);
                collect_grad(expert.up);
                collect_grad(expert.down);
            }
        }

        // Router LoRA gradients (when train_router is enabled)
        collect_grad(lora_g.router);
    }

    CUDA_CHECK(cudaMemcpyAsync(state.grad_ptrs.Data, h_grad_ptrs.data(), h_grad_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_OPTIMIZER_H
