// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NorMuon hybrid optimizer: AdamW-8bit for 1D params + NorMuon with Polar
// Express orthogonalization for 2D weight matrices.

#include "runtime/optimizers/normuon_optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/optimizers/adamw_8bit.h"
#include "runtime/optimizers/normuon.h"
#include "runtime/optimizers/polar_express.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

namespace optimizers {

using namespace dsl::internal;

namespace {

// Helper to determine if a parameter should use NorMuon (2D weight matrix)
// or AdamW (1D / embedding / norm).
bool is_normuon_param(const std::string& name, const Tensor& param) {
    if (param.Rank != 2) return false;
    if (is_norm_param_name(name)) return false;
    if (is_bias_param_name(name)) return false;

    auto lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (dsl::tensor_role_is_embedding_name(lower)) return false;
    if (dsl::tensor_role_is_lm_head_name(lower)) return false;

    // MoE router gates use AdamW (special case from study implementation).
    if (dsl::tensor_role_is_router_name(lower)) return false;
    if (lower.find("gate") != std::string::npos && lower.find("mlp") == std::string::npos) return false;

    return true;
}

}  // namespace

struct NorMuonOptimizer::Impl {
    struct State {
        bool initialized = false;

        // AdamW state for 1D params (embeddings, norms, lm_head, etc.)
        size_t adamw_total_params = 0;
        size_t adamw_state_elems = 0;
        size_t adamw_num_blocks = 0;
        Tensor adamw_quantiles1;
        Tensor adamw_quantiles2;
        Tensor adamw_state1;
        Tensor adamw_state2;
        Tensor adamw_absmax1;
        Tensor adamw_absmax2;

        // NorMuon state for 2D params (attention, MLP weights)
        size_t normuon_total_params = 0;
        size_t normuon_state_elems = 0;
        size_t normuon_num_blocks = 0;
        Tensor momentum_quantiles;
        Tensor momentum_state;
        Tensor momentum_absmax;

        // Variance buffers — one per 2D weight
        std::vector<Tensor> variance_buffers;
        std::vector<std::pair<int, int>> variance_shapes;  // (M, N) per buffer

        // Polar Express workspace
        Tensor polar_workspace;
        size_t max_weight_M = 0;
        size_t max_weight_N = 0;

        // cuBLAS handle for Polar Express
        cublasHandle_t cublas_handle = nullptr;

        // Param classification: true = 2D (NorMuon), false = 1D (AdamW)
        std::vector<std::pair<std::string, bool>> param_classification;

        ~State() {
            if (cublas_handle) {
                cublasDestroy(cublas_handle);
                cublas_handle = nullptr;
            }
        }
    };

    std::unique_ptr<State> state;

    void ensure_state() {
        if (!state) {
            state = std::make_unique<State>();
        }
    }
};

NorMuonOptimizer::NorMuonOptimizer()
    : mImpl(std::make_unique<Impl>()) {
}

NorMuonOptimizer::~NorMuonOptimizer() = default;

void NorMuonOptimizer::init_state(dsl::DslModel& model, cudaStream_t stream) {
    mImpl->ensure_state();
    auto& state = *mImpl->state;
    if (state.initialized) return;

    constexpr size_t BLOCK_SIZE = NORMUON_BLOCK_SIZE;
    const bool use_weight_manager = (model.mWeightManager != nullptr);

    // Phase 1: classify params and count state sizes.
    state.adamw_total_params = 0;
    state.adamw_state_elems = 0;
    state.normuon_total_params = 0;
    state.normuon_state_elems = 0;
    state.max_weight_M = 0;
    state.max_weight_N = 0;
    state.param_classification.clear();
    state.variance_shapes.clear();

    for (const auto& name : model.mGrads->param_names()) {
        Tensor& param = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
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

    // Phase 2: allocate AdamW state tensors
    if (state.adamw_state_elems > 0) {
        state.adamw_quantiles1 = model.mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q1", {256});
        state.adamw_quantiles2 = model.mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q2", {256});

        std::vector<float> h_q1(256), h_q2(256);
        create_adamw8bit_quantiles1(h_q1.data());
        create_adamw8bit_quantiles2(h_q2.data());
        CUDA_CHECK(cudaMemcpy(state.adamw_quantiles1.Data, h_q1.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.adamw_quantiles2.Data, h_q2.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

        state.adamw_state1 = model.mAllocator->allocate(ETensorDType::BYTE,
                                                        "normuon_adamw_s1",
                                                        {static_cast<long>(state.adamw_state_elems)});
        state.adamw_state2 = model.mAllocator->allocate(ETensorDType::BYTE,
                                                        "normuon_adamw_s2",
                                                        {static_cast<long>(state.adamw_state_elems)});
        state.adamw_absmax1 = model.mAllocator->allocate(ETensorDType::FP32,
                                                         "normuon_adamw_am1",
                                                         {static_cast<long>(state.adamw_num_blocks)});
        state.adamw_absmax2 = model.mAllocator->allocate(ETensorDType::FP32,
                                                         "normuon_adamw_am2",
                                                         {static_cast<long>(state.adamw_num_blocks)});

        init_adamw8bit_state(reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()),
                             reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()),
                             state.adamw_absmax1.template get<float>(),
                             state.adamw_absmax2.template get<float>(),
                             state.adamw_state_elems,
                             stream);
    }

    // allocate NorMuon state tensors
    if (state.normuon_state_elems > 0) {
        state.momentum_quantiles = model.mAllocator->allocate(ETensorDType::FP32, "normuon_mom_q", {256});
        std::vector<float> h_mom_q(256);
        create_normuon_quantiles(h_mom_q.data());
        CUDA_CHECK(
            cudaMemcpy(state.momentum_quantiles.Data, h_mom_q.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

        state.momentum_state = model.mAllocator->allocate(ETensorDType::BYTE,
                                                          "normuon_mom_s",
                                                          {static_cast<long>(state.normuon_state_elems)});
        state.momentum_absmax = model.mAllocator->allocate(ETensorDType::FP32,
                                                           "normuon_mom_am",
                                                           {static_cast<long>(state.normuon_num_blocks)});

        init_normuon_momentum_state(reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()),
                                    state.momentum_absmax.template get<float>(),
                                    state.normuon_state_elems,
                                    stream);

        // Variance buffers — one per 2D weight
        for (const auto& shape : state.variance_shapes) {
            int M = shape.first;
            int N = shape.second;
            size_t var_size = normuon_variance_buffer_size(M, N);
            Tensor var_buf =
                model.mAllocator->allocate(ETensorDType::FP32, "normuon_var", {static_cast<long>(var_size)});
            fill_constant(var_buf.template get<float>(), 1.0f, var_size, stream);
            state.variance_buffers.push_back(std::move(var_buf));
        }

        // Polar Express workspace
        size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
        size_t polar_ws_size =
            polar_express_workspace_size(1, static_cast<int>(state.max_weight_M), static_cast<int>(state.max_weight_N));
        size_t total_ws_elems = max_weight_elems + (polar_ws_size / sizeof(nv_bfloat16) + 1);
        state.polar_workspace =
            model.mAllocator->allocate(ETensorDType::BF16, "normuon_polar_ws", {static_cast<long>(total_ws_elems)});

        CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        CUBLAS_CHECK(cublasSetStream(state.cublas_handle, stream));
    }

    state.initialized = true;
}

void NorMuonOptimizer::step(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config, int step_idx) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("NorMuonOptimizer::step called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        model.update_lora_normuon(comm, config, step_idx);
        model.lora_weights().advance_sync_generation();
        return;
    }

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

    if (!mImpl->state || !mImpl->state->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("NorMuonOptimizer::step: optimizer state must be initialized before capture");
        }
        init_state(model, stream);
    }

    auto& state = *mImpl->state;
    constexpr size_t BLOCK_SIZE = NORMUON_BLOCK_SIZE;

    const float normuon_lr = config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate;
    const float normuon_momentum = config.normuon_momentum;
    const float normuon_beta2 = config.normuon_beta2;
    const float weight_decay = config.weight_decay;
    const bool cautious_wd = config.normuon_cautious_wd;

    const float adamw_lr = config.learning_rate;
    const float adamw_beta1 = config.adamw_beta1;
    const float adamw_beta2 = config.adamw_beta2;
    const float adamw_eps = config.adamw_epsilon;

    size_t adamw_offset = 0;
    size_t normuon_offset = 0;
    size_t variance_idx = 0;

    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& [name, use_normuon] : state.param_classification) {
        Tensor& val = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
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
        Tensor grad_view =
            param_sharded ? static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards)) : *grad;

        float wd = weight_decay;

        if (use_normuon) {
            normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            int M = static_cast<int>(val.Sizes[0]);
            int N = static_cast<int>(val.nelem() / static_cast<size_t>(M));
            size_t n = val.nelem();
            size_t block_offset = normuon_offset / BLOCK_SIZE;

            unsigned char* mom_state =
                reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()) + normuon_offset;
            float* mom_absmax = state.momentum_absmax.template get<float>() + block_offset;
            float* quantiles = state.momentum_quantiles.template get<float>();
            float* var_buf = state.variance_buffers[variance_idx].template get<float>();
            nv_bfloat16* workspace = state.polar_workspace.template get<nv_bfloat16>();

            float lr_mult = normuon_lr_multiplier(M, N);
            float effective_lr = normuon_lr * lr_mult;

            if (val.DType == ETensorDType::BF16 && grad_view.DType == ETensorDType::BF16) {
                normuon_update_2d(state.cublas_handle,
                                  val.template get<nv_bfloat16>(),
                                  grad_view.template get<nv_bfloat16>(),
                                  mom_state,
                                  var_buf,
                                  workspace,
                                  M,
                                  N,
                                  effective_lr,
                                  normuon_momentum,
                                  normuon_beta2,
                                  cautious_wd ? wd : 0.0f,
                                  quantiles,
                                  mom_absmax,
                                  stream);
            } else {
                throw std::runtime_error("NorMuonOptimizer::step: NorMuon requires BF16 weights, got " + name);
            }

            normuon_offset += n;
            variance_idx++;
        } else {
            adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            size_t n = val.nelem();
            size_t block_offset = adamw_offset / BLOCK_SIZE;

            unsigned char* s1 =
                reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()) + adamw_offset;
            unsigned char* s2 =
                reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()) + adamw_offset;
            float* am1 = state.adamw_absmax1.template get<float>() + block_offset;
            float* am2 = state.adamw_absmax2.template get<float>() + block_offset;
            float* q1 = state.adamw_quantiles1.template get<float>();
            float* q2 = state.adamw_quantiles2.template get<float>();

            if (val.DType == ETensorDType::FP32) {
                if (grad_view.DType == ETensorDType::FP32) {
                    adamw_update_8bit(val.template get<float>(),
                                      grad_view.template get<float>(),
                                      s1,
                                      s2,
                                      n,
                                      adamw_lr,
                                      adamw_beta1,
                                      adamw_beta2,
                                      step_idx,
                                      adamw_eps,
                                      wd,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      nullptr,
                                      nullptr,
                                      stream);
                } else if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(val.template get<float>(),
                                      grad_view.template get<nv_bfloat16>(),
                                      s1,
                                      s2,
                                      n,
                                      adamw_lr,
                                      adamw_beta1,
                                      adamw_beta2,
                                      step_idx,
                                      adamw_eps,
                                      wd,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      nullptr,
                                      nullptr,
                                      stream);
                }
            } else if (val.DType == ETensorDType::BF16) {
                if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(val.template get<nv_bfloat16>(),
                                      grad_view.template get<nv_bfloat16>(),
                                      s1,
                                      s2,
                                      n,
                                      adamw_lr,
                                      adamw_beta1,
                                      adamw_beta2,
                                      step_idx,
                                      adamw_eps,
                                      wd,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      nullptr,
                                      nullptr,
                                      stream);
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

    if (model.mWeightManager) {
        model.mWeightManager->invalidate();
        model.mWeightManager->sync_work_from_master(stream);
    }

    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(rs.NormDone));
        if (!std::isfinite(*rs.GradScaleHost)) {
            throw std::runtime_error("NorMuonOptimizer::step: grad_scale is NaN/Inf");
        }
    }
    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void NorMuonOptimizer::step_graph(dsl::DslModel& model,
                                  NCCLCommunicator& comm,
                                  const OptimizerConfig& config,
                                  const float* opt_params,
                                  const int* opt_step) {
    if (!model.mRunState || !model.mParams || !model.mGrads) {
        throw std::logic_error("NorMuonOptimizer::step_graph called before allocate_run_state()");
    }
    if (model.lora_enabled()) {
        throw std::logic_error("NorMuonOptimizer::step_graph: LoRA NorMuon graph capture not yet supported");
    }
    if (!mImpl->state) {
        throw std::logic_error("NorMuonOptimizer::step_graph: optimizer state not allocated");
    }

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
        init_state(model, stream);
    }

    auto& state = *mImpl->state;
    constexpr size_t BLOCK_SIZE = NORMUON_BLOCK_SIZE;

    size_t adamw_offset = 0;
    size_t normuon_offset = 0;
    size_t variance_idx = 0;

    std::unordered_set<void*> seen_grad_ptrs;

    for (const auto& [name, use_normuon] : state.param_classification) {
        Tensor& val = use_weight_manager ? model.mWeightManager->get_master(name) : model.mParams->get(name);
        bool accumulate = false;
        Tensor* grad = model.mGrads->get_param_grad(name, accumulate);
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
        Tensor grad_view =
            param_sharded ? static_cast<Tensor>(shard_view(*grad, model.mShardIdx, model.mNumShards)) : *grad;

        float wd_scale = 1.f;

        if (use_normuon) {
            normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            int M = static_cast<int>(val.Sizes[0]);
            int N = static_cast<int>(val.nelem() / static_cast<size_t>(M));
            size_t n = val.nelem();
            size_t block_offset = normuon_offset / BLOCK_SIZE;

            unsigned char* mom_state =
                reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()) + normuon_offset;
            float* mom_absmax = state.momentum_absmax.template get<float>() + block_offset;
            float* quantiles = state.momentum_quantiles.template get<float>();
            float* var_buf = state.variance_buffers[variance_idx].template get<float>();
            nv_bfloat16* workspace = state.polar_workspace.template get<nv_bfloat16>();

            float lr_mult = normuon_lr_multiplier(M, N);

            if (val.DType == ETensorDType::BF16 && grad_view.DType == ETensorDType::BF16) {
                normuon_update_2d_graph(state.cublas_handle,
                                        val.template get<nv_bfloat16>(),
                                        grad_view.template get<nv_bfloat16>(),
                                        mom_state,
                                        var_buf,
                                        workspace,
                                        M,
                                        N,
                                        lr_mult,
                                        wd_scale,
                                        quantiles,
                                        mom_absmax,
                                        opt_params,
                                        stream);
            } else {
                throw std::runtime_error("NorMuonOptimizer::step_graph: NorMuon requires BF16 weights, got " + name);
            }

            normuon_offset += n;
            variance_idx++;
        } else {
            adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            size_t n = val.nelem();
            size_t block_offset = adamw_offset / BLOCK_SIZE;

            unsigned char* s1 =
                reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()) + adamw_offset;
            unsigned char* s2 =
                reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()) + adamw_offset;
            float* am1 = state.adamw_absmax1.template get<float>() + block_offset;
            float* am2 = state.adamw_absmax2.template get<float>() + block_offset;
            float* q1 = state.adamw_quantiles1.template get<float>();
            float* q2 = state.adamw_quantiles2.template get<float>();

            // AdamW params are at opt_params[4..7] for NorMuon hybrid mode
            const float* adamw_opt_params = opt_params + 4;

            if (val.DType == ETensorDType::FP32) {
                if (grad_view.DType == ETensorDType::FP32) {
                    adamw_update_8bit(val.template get<float>(),
                                      grad_view.template get<float>(),
                                      s1,
                                      s2,
                                      n,
                                      /*lr=*/0.f,
                                      /*beta1=*/0.f,
                                      /*beta2=*/0.f,
                                      /*step=*/1,
                                      /*eps=*/0.f,
                                      wd_scale,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      adamw_opt_params,
                                      opt_step,
                                      stream);
                } else if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(val.template get<float>(),
                                      grad_view.template get<nv_bfloat16>(),
                                      s1,
                                      s2,
                                      n,
                                      /*lr=*/0.f,
                                      /*beta1=*/0.f,
                                      /*beta2=*/0.f,
                                      /*step=*/1,
                                      /*eps=*/0.f,
                                      wd_scale,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      adamw_opt_params,
                                      opt_step,
                                      stream);
                }
            } else if (val.DType == ETensorDType::BF16) {
                if (grad_view.DType == ETensorDType::BF16) {
                    adamw_update_8bit(val.template get<nv_bfloat16>(),
                                      grad_view.template get<nv_bfloat16>(),
                                      s1,
                                      s2,
                                      n,
                                      /*lr=*/0.f,
                                      /*beta1=*/0.f,
                                      /*beta2=*/0.f,
                                      /*step=*/1,
                                      /*eps=*/0.f,
                                      wd_scale,
                                      grad_scale,
                                      q1,
                                      q2,
                                      am1,
                                      am2,
                                      adamw_opt_params,
                                      opt_step,
                                      stream);
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

    if (model.mWeightManager) {
        model.mWeightManager->invalidate();
        model.mWeightManager->sync_work_from_master(stream);
    }

    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventSynchronize(rs.NormDone));
        if (!std::isfinite(*rs.GradScaleHost)) {
            throw std::runtime_error("NorMuonOptimizer::step_graph: grad_scale is NaN/Inf");
        }
    }
    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void NorMuonOptimizer::prepare_for_graph(dsl::DslModel& model,
                                         NCCLCommunicator& /*comm*/,
                                         const OptimizerConfig& /*config*/) {
    if (!model.mRunState) {
        throw std::logic_error("NorMuonOptimizer::prepare_for_graph called before allocate_run_state()");
    }

    auto& rs = *model.mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (model.lora_enabled()) {
        if (!model.mLoRANorMuonState || !model.mLoRANorMuonState->initialized) {
            // LoRA NorMuon is lazily initialized in update_lora_normuon; capture
            // requires eager init, which is not yet wired.
            throw std::logic_error("NorMuonOptimizer::prepare_for_graph: LoRA NorMuon requires eager "
                                   "initialization; not supported with CUDA graphs yet");
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
