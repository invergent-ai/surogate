// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model LoRA implementation (training, checkpointing, optimizers).

#include "runtime/dsl/dsl_model.h"
#include "runtime/dsl/dsl_model_internal.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "runtime/dsl/dsl_run_state.h"
#include "runtime/lora/lora_utils.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/optimizers/adamw_8bit.h"
#include "runtime/optimizers/normuon.h"
#include "runtime/core/fp8_scaling_state.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"

namespace dsl {

// LoRA adapter export/import

void DslModel::export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;
    fs::path dir(directory);
    if (comm.rank() == 0) {
        fs::create_directories(dir);
    }
    comm.barrier();
    mLoRAWeights->export_to_file((dir / "adapter_model.safetensors").string(), comm);
    if (comm.rank() == 0) {
        nlohmann::json adapter_config;
        adapter_config["base_model_name_or_path"] = base_model_path;
        adapter_config["peft_type"] = "LORA";
        adapter_config["task_type"] = "CAUSAL_LM";
        adapter_config["r"] = mLoRAConfig->rank;
        adapter_config["lora_alpha"] = mLoRAConfig->alpha;
        adapter_config["lora_dropout"] = mLoRAConfig->dropout;
        adapter_config["fan_in_fan_out"] = false;
        adapter_config["bias"] = "none";
        adapter_config["use_rslora"] = mLoRAConfig->use_rs_lora;
        adapter_config["target_modules"] = modules::detail::targets_to_peft_names(*mLoRAConfig);
        std::ofstream config_file(dir / "adapter_config.json");
        config_file << adapter_config.dump(2);
    }
}

void DslModel::import_adapter(const std::string& file_name, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    mLoRAWeights->import_from_file(file_name, comm);
}

// LoRA checkpoint save/load

void DslModel::save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;

    export_adapter(checkpoint_dir, comm);

    if (mLoRAAdamW8BitState && mLoRAAdamW8BitState->initialized) {
        internal::LoRAAdamW8BitStateContainer container(mLoRAAdamW8BitState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container, comm);

        if (comm.rank() == 0) {
            nlohmann::json opt_meta;
            opt_meta["optimizer_type"] = "adamw_8bit";
            opt_meta["total_params"] = mLoRAAdamW8BitState->total_params;
            opt_meta["num_blocks"] = mLoRAAdamW8BitState->num_blocks;
            opt_meta["num_tensors"] = mLoRAAdamW8BitState->num_tensors;
            std::ofstream meta_file(fs::path(checkpoint_dir) / "lora_optimizer.json");
            meta_file << opt_meta.dump(2);
        }
    } else if (mLoRANorMuonState && mLoRANorMuonState->initialized) {
        internal::LoRANorMuonStateContainer container(mLoRANorMuonState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container, comm);

        if (comm.rank() == 0) {
            nlohmann::json opt_meta;
            opt_meta["optimizer_type"] = "normuon";
            opt_meta["total_params"] = mLoRANorMuonState->total_params;
            opt_meta["state_elems"] = mLoRANorMuonState->state_elems;
            opt_meta["num_blocks"] = mLoRANorMuonState->num_blocks;
            nlohmann::json shapes = nlohmann::json::array();
            for (const auto& shape : mLoRANorMuonState->variance_shapes) {
                shapes.push_back({shape.first, shape.second});
            }
            opt_meta["variance_shapes"] = shapes;
            std::ofstream meta_file(fs::path(checkpoint_dir) / "lora_optimizer.json");
            meta_file << opt_meta.dump(2);
        }
    }

    comm.barrier();
}

void DslModel::load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;

    fs::path adapter_file = fs::path(checkpoint_dir) / "adapter_model.safetensors";
    if (fs::exists(adapter_file)) {
        import_adapter(adapter_file.string(), comm);
    }

    fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
    fs::path opt_meta_file = fs::path(checkpoint_dir) / "lora_optimizer.json";
    if (!fs::exists(opt_file) || !fs::exists(opt_meta_file)) {
        return;
    }

    std::ifstream meta_stream(opt_meta_file);
    nlohmann::json opt_meta = nlohmann::json::parse(meta_stream);
    std::string optimizer_type = opt_meta["optimizer_type"].get<std::string>();

    if (optimizer_type == "adamw_8bit") {
        if (!mLoRAAdamW8BitState) {
            mLoRAAdamW8BitState = std::make_unique<modules::LoRAAdamW8BitState>();
        }
        auto& state = *mLoRAAdamW8BitState;
        state.total_params = opt_meta["total_params"].get<size_t>();
        state.num_blocks = opt_meta["num_blocks"].get<size_t>();
        state.num_tensors = opt_meta["num_tensors"].get<int>();

        if (!state.state1.Data) {
            state.state1 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state1", {static_cast<long>(state.total_params)});
            state.state2 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state2", {static_cast<long>(state.total_params)});
            state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax1", {static_cast<long>(state.num_blocks)});
            state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax2", {static_cast<long>(state.num_blocks)});

            state.quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
            state.quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});
            std::vector<float> h_q1(256), h_q2(256);
            optimizers::create_adamw8bit_quantiles1(h_q1.data());
            optimizers::create_adamw8bit_quantiles2(h_q2.data());
            CUDA_CHECK(cudaMemcpy(state.quantiles1.Data, h_q1.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(state.quantiles2.Data, h_q2.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
        }

        internal::LoRAAdamW8BitStateContainer container(&state);
        load_safetensors(opt_file.string(), container, /*allow_cast=*/false);
        state.values_restored = true;

    } else if (optimizer_type == "normuon") {
        if (!mLoRANorMuonState) {
            mLoRANorMuonState = std::make_unique<modules::LoRANorMuonState>();
        }
        auto& state = *mLoRANorMuonState;
        state.total_params = opt_meta["total_params"].get<size_t>();
        state.state_elems = opt_meta["state_elems"].get<size_t>();
        state.num_blocks = opt_meta["num_blocks"].get<size_t>();

        state.variance_shapes.clear();
        for (const auto& shape : opt_meta["variance_shapes"]) {
            state.variance_shapes.emplace_back(shape[0].get<int>(), shape[1].get<int>());
        }

        if (!state.momentum_state.Data) {
            state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_quantiles", {256});
            std::vector<float> h_quantiles(256);
            optimizers::create_normuon_quantiles(h_quantiles.data());
            CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_quantiles.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

            state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "lora_normuon_state", {static_cast<long>(state.state_elems)});
            state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_absmax", {static_cast<long>(state.num_blocks)});

            state.variance_buffers.clear();
            for (size_t i = 0; i < state.variance_shapes.size(); ++i) {
                const auto& shape = state.variance_shapes[i];
                state.variance_buffers.push_back(
                    mAllocator->allocate(ETensorDType::FP32, fmt::format("lora_normuon_var_{}", i).c_str(), {shape.first, shape.second}));
            }
        }

        internal::LoRANorMuonStateContainer container(&state);
        load_safetensors(opt_file.string(), container, /*allow_cast=*/false);
        state.values_restored = true;
    }

    comm.barrier();
}

// LoRA run state allocation

void DslModel::allocate_lora_run_state(NCCLCommunicator& comm, int B, int T) {
    (void)comm;
    if (!lora_enabled()) return;

    mLoRARunState = std::make_unique<modules::LoRARunState>();
    mLoRARunState->B = B;
    mLoRARunState->T = T;

    auto ctx = mAllocator->with_context("DSL_LoRA_RunState");

    const int rank = mLoRAConfig->rank;
    const int BT = B * T;
    const int qkv_features = std::max(0, mModelConfig.qkv_channels());
    const int max_features = std::max({mModelConfig.HiddenSize, mModelConfig.IntermediateSize, qkv_features});
    const ETensorDType work_dtype = mModelConfig.DType;

    mLoRARunState->intermediate = mAllocator->allocate(
        work_dtype, "lora_intermediate", EAllocationType::ON_DEVICE, {BT, rank});
    mLoRARunState->intermediate2 = mAllocator->allocate(
        work_dtype, "lora_intermediate2", EAllocationType::ON_DEVICE, {BT, rank});
    mLoRARunState->slice = mAllocator->allocate(
        work_dtype, "lora_slice", EAllocationType::ON_DEVICE, {BT, max_features});

    auto& rs = *mRunState;
    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(rs.DeviceProp)));
    mLoRARunState->norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "lora_norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums + 2});

    if (mOptions.recompute_enabled()) {
        const int C = mModelConfig.HiddenSize;
        mLoRARunState->recompute_ln = mAllocator->allocate(
            work_dtype, "lora_recompute_ln", EAllocationType::ON_DEVICE, {B, T, C});
        mLoRARunState->recompute_rstd = mAllocator->allocate(
            ETensorDType::FP32, "lora_recompute_rstd", EAllocationType::ON_DEVICE, {B, T});
    }

    if (mIsMoEModel && mModelConfig.moe_config.has_value()) {
        const auto& moe_cfg = *mModelConfig.moe_config;
        const int top_k = moe_cfg.top_k;
        const int total_tokens = BT * top_k;
        const int expert_D = moe_cfg.moe_intermediate_size > 0 ? moe_cfg.moe_intermediate_size : mModelConfig.IntermediateSize;
        const int moe_M = (is_gated_activation(mModelConfig.activation_type) ? 2 : 1) * expert_D;

        mLoRARunState->moe_lora_intermediate1 = mAllocator->allocate(
            work_dtype, "moe_lora_intermediate1", EAllocationType::ON_DEVICE, {total_tokens, rank});
        mLoRARunState->moe_lora_intermediate2 = mAllocator->allocate(
            work_dtype, "moe_lora_intermediate2", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        mLoRARunState->moe_lora_gate = mAllocator->allocate(
            work_dtype, "moe_lora_gate", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        mLoRARunState->moe_lora_up = mAllocator->allocate(
            work_dtype, "moe_lora_up", EAllocationType::ON_DEVICE, {total_tokens, expert_D});
        mLoRARunState->moe_lora_gate_up = mAllocator->allocate(
            work_dtype, "moe_lora_gate_up", EAllocationType::ON_DEVICE, {total_tokens, moe_M});
    }
}

void DslModel::ensure_lora_run_state(NCCLCommunicator& comm, int B, int T) {
    if (!lora_enabled()) return;
    if (!mLoRARunState || mLoRARunState->B != B || mLoRARunState->T != T) {
        allocate_lora_run_state(comm, B, T);
    }
}

// LoRA gradient norm calculation

void DslModel::calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    if (!mLoRARunState || !mLoRAGrads) {
        throw std::logic_error("DslModel::calculate_lora_gradient_norm: LoRA state not initialized");
    }
    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    internal::wait_event_if_not_capturing(stream, rs.BackwardDone);

    Tensor& buf = mLoRARunState->norm_buffer;
    fill_zero(buf, stream);

    static bool debug_printed = false;
    int grad_count = 0;
    int null_count = 0;
    const bool is_capturing = internal::stream_is_capturing(stream);

    auto norm_squared = [&](const Tensor& grad, const char* name) {
        if (grad.Data) {
            if (!debug_printed && !is_capturing) {
                // Check first few values of this gradient
                CUDA_CHECK(cudaStreamSynchronize(stream));
                float first_val = 0.f;
                if (grad.DType == ETensorDType::FP32) {
                    CUDA_CHECK(cudaMemcpy(&first_val, grad.Data, sizeof(float), cudaMemcpyDeviceToHost));
                } else if (grad.DType == ETensorDType::BF16) {
                    nv_bfloat16 bf_val;
                    CUDA_CHECK(cudaMemcpy(&bf_val, grad.Data, sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
                    first_val = __bfloat162float(bf_val);
                }
                fprintf(stderr, "[DEBUG norm] grad '%s': nelem=%ld dtype=%d first_val=%g\n",
                        name, grad.nelem(), (int)grad.DType, first_val);
            }
            grad_count++;
            global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
        } else {
            null_count++;
        }
    };

    for (int l = 0; l < mModelConfig.NumLayers; ++l) {
        bool unused_acc = false;
        auto& g = mLoRAGrads->get_block_full(l, stream, comm, unused_acc);

        if (g.attention.q.has_value()) { norm_squared(g.attention.q->A, "attn.q.A"); norm_squared(g.attention.q->B, "attn.q.B"); }
        if (g.attention.k.has_value()) { norm_squared(g.attention.k->A, "attn.k.A"); norm_squared(g.attention.k->B, "attn.k.B"); }
        if (g.attention.v.has_value()) { norm_squared(g.attention.v->A, "attn.v.A"); norm_squared(g.attention.v->B, "attn.v.B"); }
        if (g.attention.o.has_value()) { norm_squared(g.attention.o->A, "attn.o.A"); norm_squared(g.attention.o->B, "attn.o.B"); }
        if (g.mlp.gate.has_value()) { norm_squared(g.mlp.gate->A, "mlp.gate.A"); norm_squared(g.mlp.gate->B, "mlp.gate.B"); }
        if (g.mlp.up.has_value()) { norm_squared(g.mlp.up->A, "mlp.up.A"); norm_squared(g.mlp.up->B, "mlp.up.B"); }
        if (g.mlp.down.has_value()) { norm_squared(g.mlp.down->A, "mlp.down.A"); norm_squared(g.mlp.down->B, "mlp.down.B"); }
        if (g.moe.use_grouped) {
            if (g.moe.grouped.gate.has_value()) { norm_squared(g.moe.grouped.gate->A, "moe.gate.A"); norm_squared(g.moe.grouped.gate->B, "moe.gate.B"); }
            if (g.moe.grouped.up.has_value()) { norm_squared(g.moe.grouped.up->A, "moe.up.A"); norm_squared(g.moe.grouped.up->B, "moe.up.B"); }
            if (g.moe.grouped.down.has_value()) { norm_squared(g.moe.grouped.down->A, "moe.down.A"); norm_squared(g.moe.grouped.down->B, "moe.down.B"); }
        } else {
            for (const auto& expert : g.moe.experts) {
                if (expert.gate.has_value()) { norm_squared(expert.gate->A, "exp.gate.A"); norm_squared(expert.gate->B, "exp.gate.B"); }
                if (expert.up.has_value()) { norm_squared(expert.up->A, "exp.up.A"); norm_squared(expert.up->B, "exp.up.B"); }
                if (expert.down.has_value()) { norm_squared(expert.down->A, "exp.down.A"); norm_squared(expert.down->B, "exp.down.B"); }
                if (expert.down.has_value()) { norm_squared(expert.down->A, "exp.down.A"); norm_squared(expert.down->B, "exp.down.B"); }
            }
        }

        if (g.router.has_value()) { norm_squared(g.router->A, "router.A"); norm_squared(g.router->B, "router.B"); }
    }

    if (!debug_printed && !is_capturing) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> h_buf(buf.nelem());
        CUDA_CHECK(cudaMemcpy(h_buf.data(), buf.Data, buf.nelem() * sizeof(float), cudaMemcpyDeviceToHost));
        double total = 0;
        for (int i = 0; i < (int)h_buf.size() - 2; ++i) total += h_buf[i];
        fprintf(stderr, "[DEBUG norm] grad_count=%d null_count=%d sum_of_partials=%g buf_nelem=%ld\n",
                grad_count, null_count, total, buf.nelem());
        debug_printed = true;
    }

    deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem() - 2, stream);

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));
                        
    const bool capturing = internal::stream_is_capturing(stream);
    global_norm_sqrt(buf.template get<float>(), capturing ? nullptr : rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
    internal::record_event_if_not_capturing(rs.NormDone, stream);
}

// LoRA AdamW 8-bit optimizer

void DslModel::initialize_lora_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream) {
    (void)comm;
    auto& state = *mLoRAAdamW8BitState;
    state.grad_ptrs_initialized = false;

    std::vector<void*> h_param_ptrs;
    std::vector<int> h_sizes;
    std::vector<int> h_state_offsets;
    size_t total_params = 0;

    auto collect_tensor = [&](Tensor& param) {
        if (!param.Data) return;
        h_param_ptrs.push_back(param.Data);
        int n = static_cast<int>(param.nelem());
        h_sizes.push_back(n);
        h_state_offsets.push_back(static_cast<int>(total_params));
        total_params += n;
    };

    for (int l = 0; l < mModelConfig.NumLayers; ++l) {
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

        if (lora_w.router.has_value() && lora_w.router->has_value()) {
            collect_tensor(lora_w.router->A);
            collect_tensor(lora_w.router->B);
        }
    }

    state.num_tensors = static_cast<int>(h_param_ptrs.size());
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

    if (!state.state1.Data) {
        state.state1 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state1", EAllocationType::ON_DEVICE, {(long)total_params});
        state.state2 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state2", EAllocationType::ON_DEVICE, {(long)total_params});
        state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax1", EAllocationType::ON_DEVICE, {(long)state.num_blocks});
        state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax2", EAllocationType::ON_DEVICE, {(long)state.num_blocks});
    }

    if (!state.values_restored) {
        init_adamw8bit_state(reinterpret_cast<unsigned char*>(state.state1.Data),
                             reinterpret_cast<unsigned char*>(state.state2.Data),
                             state.absmax1.template get<float>(),
                             state.absmax2.template get<float>(),
                             total_params, stream);
    }

    state.initialized = true;
}

void DslModel::update_lora_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream) {
    auto& state = *mLoRAAdamW8BitState;
    std::vector<void*> h_grad_ptrs;
    h_grad_ptrs.reserve(state.num_tensors);
    bool unused_acc = false;

    auto collect_grad = [&](std::optional<modules::LoRALayerWeights<Tensor>>& grad_opt) {
        if (!grad_opt.has_value()) return;
        if (grad_opt->A.Data) h_grad_ptrs.push_back(grad_opt->A.Data);
        if (grad_opt->B.Data) h_grad_ptrs.push_back(grad_opt->B.Data);
    };
    auto collect_grouped_grad = [&](std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& grad_opt) {
        if (!grad_opt.has_value()) return;
        if (grad_opt->A.Data) h_grad_ptrs.push_back(grad_opt->A.Data);
        if (grad_opt->B.Data) h_grad_ptrs.push_back(grad_opt->B.Data);
    };

    for (int l = 0; l < mModelConfig.NumLayers; ++l) {
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

        collect_grad(lora_g.router);
    }

    if (h_grad_ptrs.size() != static_cast<std::size_t>(state.num_tensors)) {
        throw std::runtime_error(fmt::format(
            "DslModel::update_lora_grad_pointers: grad ptr count mismatch (expected {}, got {})",
            state.num_tensors, h_grad_ptrs.size()));
    }

    CUDA_CHECK(cudaMemcpyAsync(state.grad_ptrs.Data, h_grad_ptrs.data(), h_grad_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
}

void DslModel::update_lora_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                      int t, float epsilon, float weight_decay, float grad_clip) {
    if (!mLoRAAdamW8BitState) {
        throw std::logic_error("DslModel::update_lora_adamw_8bit: optimizer state not allocated");
    }
    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
   
    calculate_lora_gradient_norm(comm, grad_clip);
    const float* grad_scale = mLoRARunState->norm_buffer.template get<float>() + 1;

    if (!mLoRAAdamW8BitState->initialized) {
        if (internal::stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        initialize_lora_multi_tensor_state(comm, stream);
    }
    if (!mLoRAAdamW8BitState->grad_ptrs_initialized) {
        if (internal::stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: grad pointers must be initialized before capture");
        }
        update_lora_grad_pointers(comm, stream);
        mLoRAAdamW8BitState->grad_ptrs_initialized = true;
    }

    const ETensorDType lora_dtype = mLoRAConfig->dtype;
    if (lora_dtype == ETensorDType::FP32) {
        adamw_update_8bit_multi_tensor(
            reinterpret_cast<float**>(mLoRAAdamW8BitState->param_ptrs.Data),
            reinterpret_cast<float**>(mLoRAAdamW8BitState->grad_ptrs.Data),
            mLoRAAdamW8BitState->tensor_sizes.template get<int>(),
            mLoRAAdamW8BitState->num_tensors,
            reinterpret_cast<unsigned char*>(mLoRAAdamW8BitState->state1.Data),
            reinterpret_cast<unsigned char*>(mLoRAAdamW8BitState->state2.Data),
            mLoRAAdamW8BitState->absmax1.template get<float>(),
            mLoRAAdamW8BitState->absmax2.template get<float>(),
            mLoRAAdamW8BitState->state_offsets.template get<int>(),
            mLoRAAdamW8BitState->total_params,
            learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
            mLoRAAdamW8BitState->quantiles1.template get<float>(),
            mLoRAAdamW8BitState->quantiles2.template get<float>(),
            nullptr,
            nullptr,
            stream
        );
    } else if (lora_dtype == ETensorDType::BF16) {
        adamw_update_8bit_multi_tensor(
            reinterpret_cast<nv_bfloat16**>(mLoRAAdamW8BitState->param_ptrs.Data),
            reinterpret_cast<nv_bfloat16**>(mLoRAAdamW8BitState->grad_ptrs.Data),
            mLoRAAdamW8BitState->tensor_sizes.template get<int>(),
            mLoRAAdamW8BitState->num_tensors,
            reinterpret_cast<unsigned char*>(mLoRAAdamW8BitState->state1.Data),
            reinterpret_cast<unsigned char*>(mLoRAAdamW8BitState->state2.Data),
            mLoRAAdamW8BitState->absmax1.template get<float>(),
            mLoRAAdamW8BitState->absmax2.template get<float>(),
            mLoRAAdamW8BitState->state_offsets.template get<int>(),
            mLoRAAdamW8BitState->total_params,
            learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
            mLoRAAdamW8BitState->quantiles1.template get<float>(),
            mLoRAAdamW8BitState->quantiles2.template get<float>(),
            nullptr,
            nullptr,
            stream
        );
    } else {
        throw std::runtime_error("DslModel: unsupported LoRA dtype for AdamW 8-bit");
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    internal::record_event_if_not_capturing(rs.OptimizerDone, stream);
}

void DslModel::update_lora_adamw_8bit_graph(NCCLCommunicator& comm, float grad_clip,
                                           const float* opt_params, const int* opt_step) {
    if (!mLoRAAdamW8BitState) {
        throw std::logic_error("DslModel::update_lora_adamw_8bit_graph: optimizer state not allocated");
    }
    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    
    calculate_lora_gradient_norm(comm, grad_clip);
    const float* grad_scale = mLoRARunState->norm_buffer.template get<float>() + 1;

    if (!mLoRAAdamW8BitState->initialized) {
        if (internal::stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        initialize_lora_multi_tensor_state(comm, stream);
    }
    if (!mLoRAAdamW8BitState->grad_ptrs_initialized) {
        if (internal::stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: grad pointers must be initialized before capture");
        }
        update_lora_grad_pointers(comm, stream);
        mLoRAAdamW8BitState->grad_ptrs_initialized = true;
    }

    auto& state = *mLoRAAdamW8BitState;
    const ETensorDType lora_dtype = mLoRAConfig->dtype;

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
            /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, /*weight_decay=*/1.f,
            grad_scale,
            state.quantiles1.template get<float>(),
            state.quantiles2.template get<float>(),
            opt_params,
            opt_step,
            stream
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
            /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, /*weight_decay=*/1.f,
            grad_scale,
            state.quantiles1.template get<float>(),
            state.quantiles2.template get<float>(),
            opt_params,
            opt_step,
            stream
        );
    } else {
        throw std::runtime_error("DslModel: unsupported LoRA dtype for AdamW 8-bit");
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, stream);
        }
    }

    internal::record_event_if_not_capturing(rs.OptimizerDone, stream);
}

// LoRA NorMuon optimizer

void DslModel::update_lora_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    if (!mLoRANorMuonState) {
        mLoRANorMuonState = std::make_unique<modules::LoRANorMuonState>();
    }
    auto& state = *mLoRANorMuonState;

    calculate_lora_gradient_norm(comm, config.grad_clip);

    const float lr = config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate;
    const float momentum = config.normuon_momentum;
    const float beta2 = config.normuon_beta2;
    const float weight_decay = config.weight_decay;
    const bool cautious_wd = config.normuon_cautious_wd;
    const int L = mModelConfig.NumLayers;

    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

    if (!state.initialized) {
        state.total_params = 0;
        state.state_elems = 0;
        state.max_weight_M = 0;
        state.max_weight_N = 0;
        state.variance_shapes.clear();

        auto add_param = [&](const Tensor& weight) {
            if (!weight.Data) return;
            size_t n = weight.nelem();
            state.total_params += n;
            state.state_elems = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state.state_elems += n;

            int M = 1, N = static_cast<int>(n);
            if (weight.Rank >= 2) {
                M = static_cast<int>(weight.Sizes[0]);
                N = static_cast<int>(n / static_cast<size_t>(M));
            }
            state.max_weight_M = std::max(state.max_weight_M, static_cast<size_t>(M));
            state.max_weight_N = std::max(state.max_weight_N, static_cast<size_t>(N));
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

            if (lora_w.router.has_value() && lora_w.router->has_value()) {
                add_param(lora_w.router->A);
                add_param(lora_w.router->B);
            }
        }

        state.num_blocks = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_quantiles", {256});
        std::vector<float> h_quantiles(256);
        optimizers::create_normuon_quantiles(h_quantiles.data());
        CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_quantiles.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));

        state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "lora_normuon_momentum", {static_cast<long>(state.state_elems)});
        state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_absmax", {static_cast<long>(state.num_blocks)});

        optimizers::init_normuon_momentum_state(
            reinterpret_cast<unsigned char*>(state.momentum_state.Data),
            state.momentum_absmax.template get<float>(),
            state.state_elems,
            main_stream
        );

        state.variance_buffers.clear();
        for (const auto& shape : state.variance_shapes) {
            int M = shape.first;
            int N = shape.second;
            size_t var_size = optimizers::normuon_variance_buffer_size(M, N);
            Tensor var_buf = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_variance", {static_cast<long>(var_size)});
            std::vector<float> ones(var_size, 1.0f);
            CUDA_CHECK(cudaMemcpyAsync(var_buf.Data, ones.data(), var_size * sizeof(float), cudaMemcpyHostToDevice, main_stream));
            state.variance_buffers.push_back(std::move(var_buf));
        }

        size_t max_dim = std::max(state.max_weight_M, state.max_weight_N);
        size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
        size_t polar_workspace_elems = 4 * max_dim * max_dim + 1;
        size_t polar_size = max_weight_elems + polar_workspace_elems;
        state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_polar", {static_cast<long>(polar_size)});

        size_t max_weight_size = state.max_weight_M * state.max_weight_N;
        state.momentum_temp = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_temp", {static_cast<long>(max_weight_size)});

        CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        CUBLAS_CHECK(cublasSetStream(state.cublas_handle, main_stream));

        state.initialized = true;
    }

    const ETensorDType lora_dtype = mLoRAConfig->dtype;
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
            throw std::runtime_error("DSL LoRA NorMuon optimizer only supports BF16 LoRA weights");
        }

        state_offset = aligned_offset + n;
        var_idx++;
    };

    for (int l = 0; l < L; ++l) {
        auto& lora_w = mLoRAWeights->get_master_block(l, main_stream);
        auto& lora_g = mLoRAGrads->get_block_full(l, main_stream, comm, unused_acc);

        if (lora_w.attention.q.has_value() && lora_g.attention.q.has_value()) { update_param(lora_w.attention.q->A, lora_g.attention.q->A); update_param(lora_w.attention.q->B, lora_g.attention.q->B); }
        if (lora_w.attention.k.has_value() && lora_g.attention.k.has_value()) { update_param(lora_w.attention.k->A, lora_g.attention.k->A); update_param(lora_w.attention.k->B, lora_g.attention.k->B); }
        if (lora_w.attention.v.has_value() && lora_g.attention.v.has_value()) { update_param(lora_w.attention.v->A, lora_g.attention.v->A); update_param(lora_w.attention.v->B, lora_g.attention.v->B); }
        if (lora_w.attention.o.has_value() && lora_g.attention.o.has_value()) { update_param(lora_w.attention.o->A, lora_g.attention.o->A); update_param(lora_w.attention.o->B, lora_g.attention.o->B); }
        if (lora_w.mlp.gate.has_value() && lora_g.mlp.gate.has_value()) { update_param(lora_w.mlp.gate->A, lora_g.mlp.gate->A); update_param(lora_w.mlp.gate->B, lora_g.mlp.gate->B); }
        if (lora_w.mlp.up.has_value() && lora_g.mlp.up.has_value()) { update_param(lora_w.mlp.up->A, lora_g.mlp.up->A); update_param(lora_w.mlp.up->B, lora_g.mlp.up->B); }
        if (lora_w.mlp.down.has_value() && lora_g.mlp.down.has_value()) { update_param(lora_w.mlp.down->A, lora_g.mlp.down->A); update_param(lora_w.mlp.down->B, lora_g.mlp.down->B); }

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

        if (lora_w.router.has_value() && lora_g.router.has_value()) {
            update_param(lora_w.router->A, lora_g.router->A);
            update_param(lora_w.router->B, lora_g.router->B);
        }
    }

    if (rs.has_fp8_delayed_scaling()) {
        if (auto* fp8_state = rs.get_fp8_scaling_state()) {
            delayed_scaling_update(*fp8_state, main_stream);
        }
    }

    internal::record_event_if_not_capturing(rs.OptimizerDone, main_stream);
}

}  // namespace dsl
