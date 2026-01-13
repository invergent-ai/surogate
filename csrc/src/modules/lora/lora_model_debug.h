// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_DEBUG_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_DEBUG_H

#include "lora_model_core.h"
#include "kernels/kernels.h"
#include <limits>

namespace modules {

template<typename Block>
float ModularLoRAModel<Block>::debug_get_grad_scale() const {
    if (!lora_enabled() || !mLoRARunState) return std::numeric_limits<float>::quiet_NaN();
    float v = std::numeric_limits<float>::quiet_NaN();
    const float* device_ptr = mLoRARunState->norm_buffer.template get<float>() + 1;
    CUDA_CHECK(cudaMemcpy(&v, device_ptr, sizeof(v), cudaMemcpyDeviceToHost));
    return v;
}

template<typename Block>
int ModularLoRAModel<Block>::debug_get_valid_tokens() const {
    int v = 0;
    auto& rs = mBaseModel->get_run_state();
    CUDA_CHECK(cudaMemcpy(&v, rs.ValidTokenCount.template get<int>(), sizeof(v), cudaMemcpyDeviceToHost));
    return v;
}

template<typename Block>
std::vector<float> ModularLoRAModel<Block>::debug_get_grad_norms_by_layer(NCCLCommunicator& comm) const {
    if (!lora_enabled() || !mLoRARunState || !mLoRAGrads) {
        return {};
    }

    auto& rs = mBaseModel->get_run_state();
    cudaStream_t stream = rs.MainStream;
    const auto& cfg = mBaseModel->config();

    std::vector<float> norms;
    norms.resize((std::size_t)cfg.NumLayers, 0.0f);

    Tensor& buf = mLoRARunState->norm_buffer;

    auto add = [&](const Tensor& grad) {
        if (grad.Data) {
            global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
        }
    };

    for (int l = 0; l < cfg.NumLayers; ++l) {
        fill_zero(buf, stream);

        bool unused_accumulate = false;
        auto& g = mLoRAGrads->get_block_full(l, stream, comm, unused_accumulate);

        if (g.attention.q.has_value()) { add(g.attention.q->A); add(g.attention.q->B); }
        if (g.attention.k.has_value()) { add(g.attention.k->A); add(g.attention.k->B); }
        if (g.attention.v.has_value()) { add(g.attention.v->A); add(g.attention.v->B); }
        if (g.attention.o.has_value()) { add(g.attention.o->A); add(g.attention.o->B); }
        if (g.mlp.gate.has_value()) { add(g.mlp.gate->A); add(g.mlp.gate->B); }
        if (g.mlp.up.has_value()) { add(g.mlp.up->A); add(g.mlp.up->B); }
        if (g.mlp.down.has_value()) { add(g.mlp.down->A); add(g.mlp.down->B); }

        if (g.moe.use_grouped) {
            if (g.moe.grouped.gate.has_value()) { add(g.moe.grouped.gate->A); add(g.moe.grouped.gate->B); }
            if (g.moe.grouped.up.has_value()) { add(g.moe.grouped.up->A); add(g.moe.grouped.up->B); }
            if (g.moe.grouped.down.has_value()) { add(g.moe.grouped.down->A); add(g.moe.grouped.down->B); }
        } else {
            for (const auto& expert : g.moe.experts) {
                if (expert.gate.has_value()) { add(expert.gate->A); add(expert.gate->B); }
                if (expert.up.has_value()) { add(expert.up->A); add(expert.up->B); }
                if (expert.down.has_value()) { add(expert.down->A); add(expert.down->B); }
            }
        }

        deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem(), stream);
        if (comm.world_size() > 1) {
            comm.reduce_norm(buf.template get<float>(), stream);
        }
        CUDA_CHECK(cudaMemcpy(&norms[(std::size_t)l], buf.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
        norms[(std::size_t)l] = std::sqrt(std::max(0.0f, norms[(std::size_t)l]));
    }

    return norms;
}

template<typename Block>
std::vector<std::pair<std::string, float>> ModularLoRAModel<Block>::debug_get_grad_norms_by_module(int layer_idx, NCCLCommunicator& comm) const {
    if (!lora_enabled() || !mLoRARunState || !mLoRAGrads) {
        return {};
    }

    const auto& cfg = mBaseModel->config();
    if (layer_idx < 0 || layer_idx >= cfg.NumLayers) {
        return {};
    }

    auto& rs = mBaseModel->get_run_state();
    cudaStream_t stream = rs.MainStream;

    Tensor& buf = mLoRARunState->norm_buffer;

    auto add = [&](const Tensor& grad) {
        if (grad.Data) {
            global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
        }
    };

    auto module_norm = [&](const Tensor& A, const Tensor& B) -> float {
        fill_zero(buf, stream);
        add(A);
        add(B);
        deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem(), stream);
        if (comm.world_size() > 1) {
            comm.reduce_norm(buf.template get<float>(), stream);
        }
        float val = 0.0f;
        CUDA_CHECK(cudaMemcpy(&val, buf.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
        return std::sqrt(std::max(0.0f, val));
    };

    bool unused_accumulate = false;
    auto& g = mLoRAGrads->get_block_full(layer_idx, stream, comm, unused_accumulate);
    std::vector<std::pair<std::string, float>> out;

    if (g.attention.q.has_value()) out.emplace_back("q_proj", module_norm(g.attention.q->A, g.attention.q->B));
    if (g.attention.k.has_value()) out.emplace_back("k_proj", module_norm(g.attention.k->A, g.attention.k->B));
    if (g.attention.v.has_value()) out.emplace_back("v_proj", module_norm(g.attention.v->A, g.attention.v->B));
    if (g.attention.o.has_value()) out.emplace_back("o_proj", module_norm(g.attention.o->A, g.attention.o->B));
    if (g.mlp.gate.has_value()) out.emplace_back("gate_proj", module_norm(g.mlp.gate->A, g.mlp.gate->B));
    if (g.mlp.up.has_value()) out.emplace_back("up_proj", module_norm(g.mlp.up->A, g.mlp.up->B));
    if (g.mlp.down.has_value()) out.emplace_back("down_proj", module_norm(g.mlp.down->A, g.mlp.down->B));

    if (g.moe.use_grouped) {
        if (g.moe.grouped.gate.has_value()) out.emplace_back("moe.grouped.gate_proj", module_norm(g.moe.grouped.gate->A, g.moe.grouped.gate->B));
        if (g.moe.grouped.up.has_value()) out.emplace_back("moe.grouped.up_proj", module_norm(g.moe.grouped.up->A, g.moe.grouped.up->B));
        if (g.moe.grouped.down.has_value()) out.emplace_back("moe.grouped.down_proj", module_norm(g.moe.grouped.down->A, g.moe.grouped.down->B));
    } else {
        for (int e = 0; e < (int)g.moe.experts.size(); ++e) {
            const auto& expert = g.moe.experts[e];
            std::string prefix = "expert" + std::to_string(e) + ".";
            if (expert.gate.has_value()) out.emplace_back(prefix + "gate_proj", module_norm(expert.gate->A, expert.gate->B));
            if (expert.up.has_value()) out.emplace_back(prefix + "up_proj", module_norm(expert.up->A, expert.up->B));
            if (expert.down.has_value()) out.emplace_back(prefix + "down_proj", module_norm(expert.down->A, expert.down->B));
        }
    }

    return out;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_DEBUG_H
