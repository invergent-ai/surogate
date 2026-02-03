// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL-driven recompute plan implementation.

#include "runtime/dsl/recompute_plan.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime/dsl/graph_executor.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"

namespace dsl {
namespace {

bool is_optional_ref(const std::string& name, std::string& stripped) {
    if (!name.empty() && name.front() == '?') {
        stripped = name.substr(1);
        return true;
    }
    stripped = name;
    return false;
}

bool is_param_ref(const std::string& name, std::string& param_name) {
    constexpr std::string_view kPrefix = "@param:";
    if (!starts_with(name, kPrefix)) {
        return false;
    }
    param_name = std::string(name.substr(kPrefix.size()));
    return true;
}

bool is_input_ref(const std::string& name, std::string& input_name) {
    constexpr std::string_view kPrefix = "@input:";
    if (!starts_with(name, kPrefix)) {
        return false;
    }
    input_name = std::string(name.substr(kPrefix.size()));
    return true;
}

bool is_global_ref(const std::string& name, std::string& global_name) {
    constexpr std::string_view kPrefix = "@global:";
    if (!starts_with(name, kPrefix)) {
        return false;
    }
    global_name = std::string(name.substr(kPrefix.size()));
    return true;
}

struct MoeCompactInfo {
    std::vector<int> host_offsets;
    std::vector<int> active_experts;
    int num_active = 0;
    bool weight_is_compact = false;
};

MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream) {
    MoeCompactInfo info;
    if (!expert_offsets_dev || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.host_offsets.resize(num_experts + 1, 0);
    CUDA_CHECK(cudaMemcpyAsync(info.host_offsets.data(),
                               expert_offsets_dev,
                               static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (info.host_offsets[e + 1] > info.host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active > weight_experts) {
        info.active_experts.resize(weight_experts);
        info.num_active = weight_experts;
    }

    return info;
}

RecomputePolicy parse_policy(const std::string& policy) {
    if (policy == "lora_only") return RecomputePolicy::LoraOnly;
    if (policy == "fft_only") return RecomputePolicy::FftOnly;
    if (policy == "never") return RecomputePolicy::Never;
    return RecomputePolicy::Always;
}

bool eval_condition(const std::string& cond,
                    const AttrMap& module_config,
                    const modules::ModelConfig& model_cfg) {
    if (cond.empty()) {
        return true;
    }
    if (const auto* attr = find_attr(module_config, cond)) {
        if (auto v = attr_bool(*attr)) {
            return *v;
        }
        if (auto v = attr_int(*attr)) {
            return *v != 0;
        }
        if (auto v = attr_string(*attr)) {
            return *v == "true" || *v == "1" || *v == "yes";
        }
        if (auto v = std::get_if<double>(&attr->value)) {
            return *v != 0.0;
        }
    }
    if (cond == "use_qk_norm") {
        return model_cfg.use_qk_norm || model_cfg.UseQKNorm;
    }
    if (cond == "use_qkv_bias") {
        return model_cfg.UseQKVBias;
    }
    if (cond == "use_shared_expert") {
        return model_cfg.moe_config.has_value() && model_cfg.moe_config->use_shared_expert;
    }
    if (cond == "norm_topk_prob") {
        return model_cfg.moe_config.has_value() && model_cfg.moe_config->norm_topk_prob;
    }
    return false;
}

std::string canonicalize_name(const std::unordered_map<std::string, std::string>& alias_map,
                              const std::string& name) {
    auto it = alias_map.find(name);
    if (it != alias_map.end()) {
        return it->second;
    }
    return name;
}

std::string canonicalize_ref(const std::unordered_map<std::string, std::string>& alias_map,
                             const std::string& raw) {
    std::string stripped;
    const bool optional = is_optional_ref(raw, stripped);
    if (!starts_with(stripped, "@")) {
        stripped = canonicalize_name(alias_map, stripped);
    }
    return optional ? ("?" + stripped) : stripped;
}

bool is_residual_name(const std::string& name) {
    return name == "residual" || name == "res_ffn" || name == "res_att" ||
           name.find("residual") != std::string::npos;
}

bool is_rstd_name(const std::string& name) {
    return ends_with(name, "_rstd");
}

int resolve_rotary_dim(const AttrMap& attrs, const modules::ModelConfig& cfg) {
    if (const auto* attr = find_attr(attrs, "rotary_dim")) {
        if (auto v = attr_int(*attr)) {
            return static_cast<int>(*v);
        }
        if (auto s = attr_string(*attr)) {
            if (*s == "D" || *s == "head_dim" || *s == "Hs") {
                return static_cast<int>(cfg.head_size());
            }
            try {
                return std::stoi(*s);
            } catch (...) {
                return static_cast<int>(cfg.head_size());
            }
        }
    }
    return static_cast<int>(cfg.head_size());
}

float resolve_eps(const AttrMap& attrs, const modules::ModelConfig& cfg) {
    if (const auto* attr = find_attr(attrs, "eps")) {
        if (auto v = attr_int(*attr)) {
            return static_cast<float>(*v);
        }
        if (auto s = attr_string(*attr)) {
            try {
                return std::stof(*s);
            } catch (...) {
                return static_cast<float>(cfg.RmsNormEps);
            }
        }
        if (auto v = std::get_if<double>(&attr->value)) {
            return static_cast<float>(*v);
        }
    }
    return static_cast<float>(cfg.RmsNormEps);
}

int resolve_top_k(const AttrMap& attrs, const modules::ModelConfig& cfg) {
    if (const auto* attr = find_attr(attrs, "top_k")) {
        if (auto v = attr_int(*attr)) {
            return static_cast<int>(*v);
        }
        if (auto s = attr_string(*attr)) {
            if (*s == "K" || *s == "top_k") {
                return cfg.NumExpertsPerTok > 0 ? cfg.NumExpertsPerTok : 1;
            }
            try {
                return std::stoi(*s);
            } catch (...) {
                return cfg.NumExpertsPerTok > 0 ? cfg.NumExpertsPerTok : 1;
            }
        }
        if (auto v = std::get_if<double>(&attr->value)) {
            return static_cast<int>(*v);
        }
    }
    return cfg.NumExpertsPerTok > 0 ? cfg.NumExpertsPerTok : 1;
}

bool resolve_norm_topk(const AttrMap& attrs, const modules::ModelConfig& cfg) {
    if (const auto* attr = find_attr(attrs, "normalize")) {
        if (auto v = attr_bool(*attr)) {
            return *v;
        }
        if (auto v = attr_int(*attr)) {
            return *v != 0;
        }
        if (auto s = attr_string(*attr)) {
            if (*s == "norm_topk_prob") {
                return cfg.moe_config.has_value() && cfg.moe_config->norm_topk_prob;
            }
            return *s == "true" || *s == "1" || *s == "yes";
        }
        if (auto v = std::get_if<double>(&attr->value)) {
            return *v != 0.0;
        }
    }
    return cfg.moe_config.has_value() && cfg.moe_config->norm_topk_prob;
}

modules::MatmulOp parse_matmul_op(const AttrMap& attrs,
                                  const std::string& weight_name) {
    if (const auto* attr = find_attr(attrs, "matmul_op")) {
        if (auto s = attr_string(*attr)) {
            if (*s == "qkv") return modules::MatmulOp::QKV;
            if (*s == "attn_out" || *s == "out") return modules::MatmulOp::AttnOut;
            if (*s == "mlp_up" || *s == "up") return modules::MatmulOp::MLPUp;
            if (*s == "mlp_down" || *s == "down") return modules::MatmulOp::MLPDown;
        }
    }
    int layer_idx = -1;
    if (auto op = matmul_op_from_weight(weight_name, layer_idx)) {
        return *op;
    }
    throw std::runtime_error("DSL recompute: unable to infer matmul_op for " + weight_name);
}

const MatmulForwardPlan* plan_for_matmul(const LayerForwardPlan* plan, modules::MatmulOp op) {
    if (!plan) return nullptr;
    switch (op) {
        case modules::MatmulOp::QKV: return plan->qkv.valid ? &plan->qkv : nullptr;
        case modules::MatmulOp::AttnOut: return plan->out_proj.valid ? &plan->out_proj : nullptr;
        case modules::MatmulOp::MLPUp: return plan->mlp_up.valid ? &plan->mlp_up : nullptr;
        case modules::MatmulOp::MLPDown: return plan->mlp_down.valid ? &plan->mlp_down : nullptr;
        default: return nullptr;
    }
}

struct InputMap {
    std::unordered_map<std::string, Tensor*> tensors;
    std::unordered_map<std::string, std::string> param_names;
};

struct RecomputeScratch {
    std::vector<Tensor> temps;
    Tensor zero_residual{};
    bool zero_ready = false;
    Tensor moe_expert_offsets{};
    Tensor moe_gather_indices{};
};

struct RecomputeContext {
    DslRunState& rs;
    DslParamStore& weights;
    const modules::ModelConfig& cfg;
    const RuntimeOptions& options;
    const modules::ModularLoRAConfig* lora_config = nullptr;
    modules::ModularLoRAWeightsManager* lora_weights = nullptr;
    modules::LoRARunState* lora_run_state = nullptr;
    const std::unordered_map<std::string, Tensor>* saved = nullptr;
    const LayerForwardPlan* layer_plan = nullptr;
    std::function<const Tensor*(const std::string&, Tensor&, cudaStream_t)> get_fp8_cached_weight;
    std::function<const FP4WeightCacheEntry*(const std::string&, Tensor&, cudaStream_t)> get_fp4_cached_weight;
    bool lora_only_mode = false;  // FFT mode when false
    int layer_idx = -1;           // Current layer being processed
};

Tensor& ensure_activation(RecomputeContext& ctx, int layer_idx, const std::string& name, cudaStream_t stream) {
    auto& rs = ctx.rs;
    auto& acts = rs.simplified_acts(layer_idx);

    auto ensure = [&](Tensor& t) -> Tensor& {
        if (!t.Data) {
            rs.temp_acquire(t);
        }
        return t;
    };

    if (name == "ln1" || name == "ln1_flat") return ensure(acts.ln1);
    if (name == "ln1_rstd") return ensure(acts.ln1_rstd);
    if (name == "ln2" || name == "ln2_flat") return ensure(acts.ln2);
    if (name == "ln2_rstd") return ensure(acts.ln2_rstd);
    if (name == "q_rstd") return ensure(acts.q_rstd);
    if (name == "k_rstd") return ensure(acts.k_rstd);
    if (name == "qkv" || name == "qkv_flat" || name == "qkv_biased") return ensure(acts.qkv);
    if (name == "qkv_rope") {
        if (acts.qkv_rope.Data) {
            return ensure(acts.qkv_rope);
        }
        return ensure(acts.qkv);
    }
    if (name == "lse") return ensure(acts.lse);
    if (name == "att" || name == "att_flat" || name == "attn") return ensure(acts.att);
    if (name == "att_out" || name == "att_out_flat") return ensure(acts.att_out);
    if (name == "res_att" || name == "residual_att") return ensure(acts.residual_att);
    if (name == "mlp_up" || name == "mlp_up_flat") return ensure(acts.mlp_up);
    if (name == "swiglu" || name == "swiglu_flat") return ensure(acts.swiglu);
    if (name == "mlp_down" || name == "mlp_down_flat") return ensure(acts.mlp_down);
    if (name == "router_logits") return ensure(acts.router_logits);
    if (name == "router_probs") return ensure(acts.router_probs);
    if (name == "routing_weights") return ensure(acts.routing_weights);
    if (name == "routing_indices") return ensure(acts.routing_indices);
    if (name == "permuted_input") return ensure(acts.permuted_input);
    if (name == "scatter_indices") return ensure(acts.scatter_indices);
    if (name == "expert_gate_up") return ensure(acts.expert_gate_up);
    if (name == "expert_act") return ensure(acts.expert_act);
    if (name == "expert_down") return ensure(acts.expert_down);
    if (name == "moe_out" || name == "moe_out_flat") return ensure(acts.moe_out);
    if (name == "res_ffn" || name == "residual_ffn") return rs.get_residual(layer_idx, stream);
    throw std::runtime_error("DSL recompute: unknown activation output: " + name);
}

Tensor* resolve_activation(RecomputeContext& ctx, int layer_idx, const std::string& name) {
    auto& rs = ctx.rs;
    auto& acts = rs.simplified_acts(layer_idx);

    auto get = [&](Tensor& t) -> Tensor* {
        return t.Data ? &t : nullptr;
    };

    auto try_saved = [&](const std::string& key) -> Tensor* {
        if (!ctx.saved) {
            return nullptr;
        }
        auto it = ctx.saved->find(key);
        if (it != ctx.saved->end() && it->second.Data) {
            return const_cast<Tensor*>(&it->second);
        }
        return nullptr;
    };
    if (ctx.lora_only_mode) {
        if (Tensor* t = try_saved(name)) {
            return t;
        }
        if (layer_idx >= 0) {
            std::string scoped = "blocks[" + std::to_string(layer_idx) + "]." + name;
            if (Tensor* t = try_saved(scoped)) {
                return t;
            }
        }
    }

    if (name == "ln1" || name == "ln1_flat") return get(acts.ln1);
    if (name == "ln1_rstd") return get(acts.ln1_rstd);
    if (name == "ln2" || name == "ln2_flat") return get(acts.ln2);
    if (name == "ln2_rstd") return get(acts.ln2_rstd);
    if (name == "q_rstd") return get(acts.q_rstd);
    if (name == "k_rstd") return get(acts.k_rstd);
    if (name == "qkv" || name == "qkv_flat" || name == "qkv_biased") return get(acts.qkv);
    if (name == "qkv_rope") {
        if (acts.qkv_rope.Data) return get(acts.qkv_rope);
        return get(acts.qkv);
    }
    if (name == "lse") return get(acts.lse);
    if (name == "att" || name == "att_flat" || name == "attn") return get(acts.att);
    if (name == "att_out" || name == "att_out_flat") return get(acts.att_out);
    if (name == "res_att" || name == "residual_att") return get(acts.residual_att);
    if (name == "mlp_up" || name == "mlp_up_flat") return get(acts.mlp_up);
    if (name == "swiglu" || name == "swiglu_flat") return get(acts.swiglu);
    if (name == "mlp_down" || name == "mlp_down_flat") return get(acts.mlp_down);
    if (name == "router_logits") return get(acts.router_logits);
    if (name == "router_probs") return get(acts.router_probs);
    if (name == "routing_weights") return get(acts.routing_weights);
    if (name == "routing_indices") return get(acts.routing_indices);
    if (name == "permuted_input") return get(acts.permuted_input);
    if (name == "scatter_indices") return get(acts.scatter_indices);
    if (name == "expert_gate_up") return get(acts.expert_gate_up);
    if (name == "expert_act") return get(acts.expert_act);
    if (name == "expert_down") return get(acts.expert_down);
    if (name == "moe_out" || name == "moe_out_flat") return get(acts.moe_out);
    if (name == "res_ffn" || name == "residual_ffn") return &rs.get_residual(layer_idx, rs.MainStream);
    if (Tensor* t = try_saved(name)) {
        return t;
    }
    if (layer_idx >= 0) {
        std::string scoped = "blocks[" + std::to_string(layer_idx) + "]." + name;
        if (Tensor* t = try_saved(scoped)) {
            return t;
        }
    }
    return nullptr;
}

Tensor& resolve_param(RecomputeContext& ctx, int layer_idx, const std::string& name, std::string& resolved_name) {
    auto& weights = ctx.weights;
    resolved_name = "blocks[" + std::to_string(layer_idx) + "]." + name;
    if (weights.has(resolved_name)) {
        return weights.get(resolved_name);
    }
    if (weights.has(name)) {
        resolved_name = name;
        return weights.get(name);
    }
    throw std::runtime_error("DSL recompute: missing parameter " + name);
}

Tensor* resolve_input_ref(RecomputeContext& ctx,
                          int layer_idx,
                          long B,
                          long T,
                          const std::string& name,
                          RecomputeScratch& scratch) {
    auto& rs = ctx.rs;
    if (name == "position_ids") {
        return &rs.PositionIDs;
    }
    if (name == "x") {
        if (layer_idx == 0) {
            return &rs.non_block_activations().encoded;
        }
        auto& prev = rs.simplified_acts(layer_idx - 1);
        return prev.mlp_down.Data ? &prev.mlp_down : nullptr;
    }
    if (name == "residual") {
        if (layer_idx == 0) {
            if (!scratch.zero_ready) {
                scratch.zero_residual = rs.temp_alloc(rs.non_block_activations().encoded.DType,
                                                      {B, T, static_cast<long>(ctx.cfg.HiddenSize)});
                fill_zero(scratch.zero_residual, rs.MainStream);
                scratch.temps.push_back(scratch.zero_residual);
                scratch.zero_ready = true;
            }
            return &scratch.zero_residual;
        }
        auto& prev = rs.simplified_acts(layer_idx - 1);
        return prev.residual_att.Data ? &prev.residual_att : nullptr;
    }
    if (name == "residual_in") {
        if (layer_idx == 0) {
            return &rs.non_block_activations().encoded;
        }
        return &rs.get_residual(layer_idx - 1, rs.MainStream);
    }
    if (name == "token_ids") {
        return &rs.Inputs;
    }
    if (name == "targets") {
        return &rs.Targets;
    }
    return nullptr;
}

Tensor* resolve_global_ref(RecomputeContext& ctx, const std::string& name) {
    auto& rs = ctx.rs;
    if (name == "freq_cis" || name == "rope_freqs") {
        return &rs.non_block_activations().freq_cis;
    }
    if (name == "encoded" || name == "x0") {
        return &rs.non_block_activations().encoded;
    }
    return nullptr;
}

InputMap resolve_inputs(RecomputeContext& ctx,
                        const RecomputeOp& op,
                        int layer_idx,
                        long B,
                        long T,
                        RecomputeScratch& scratch) {
    InputMap out{};
    for (const auto& raw : op.inputs) {
        std::string stripped;
        const bool optional = is_optional_ref(raw, stripped);
        std::string param_name;
        std::string input_name;
        std::string global_name;

        if (is_param_ref(stripped, param_name)) {
            std::string resolved_name;
            Tensor* tensor = nullptr;
            try {
                tensor = &resolve_param(ctx, layer_idx, param_name, resolved_name);
            } catch (...) {
                if (!optional) {
                    throw;
                }
            }
            if (tensor) {
                out.tensors[param_name] = tensor;
                out.param_names[param_name] = resolved_name;
            }
            continue;
        }

        if (is_input_ref(stripped, input_name)) {
            Tensor* tensor = resolve_input_ref(ctx, layer_idx, B, T, input_name, scratch);
            if (!tensor && !optional) {
                throw std::runtime_error("DSL recompute: missing @input:" + input_name);
            }
            if (tensor) {
                out.tensors[input_name] = tensor;
            }
            continue;
        }

        if (is_global_ref(stripped, global_name)) {
            Tensor* tensor = resolve_global_ref(ctx, global_name);
            if (!tensor && !optional) {
                throw std::runtime_error("DSL recompute: missing @global:" + global_name);
            }
            if (tensor) {
                out.tensors[global_name] = tensor;
            }
            continue;
        }

        Tensor* tensor = resolve_activation(ctx, layer_idx, stripped);
        if (!tensor && !optional) {
            throw std::runtime_error("DSL recompute: missing activation input " + stripped);
        }
        if (tensor) {
            out.tensors[stripped] = tensor;
        }
    }
    return out;
}

std::unordered_map<std::string, Tensor*> resolve_outputs(RecomputeContext& ctx,
                                                         const RecomputeOp& op,
                                                         int layer_idx,
                                                         cudaStream_t stream) {
    std::unordered_map<std::string, Tensor*> outputs;
    outputs.reserve(op.outputs.size());
    for (const auto& name : op.outputs) {
        Tensor& out = ensure_activation(ctx, layer_idx, name, stream);
        outputs.emplace(name, &out);
    }
    return outputs;
}

void apply_lora_targets(RecomputeContext& ctx,
                        const RecomputeOp& op,
                        int layer_idx,
                        long B,
                        long T,
                        const Tensor& input,
                        Tensor& output) {
    auto& rs = ctx.rs;
    if (!ctx.lora_config || !ctx.lora_weights || !ctx.lora_run_state) {
        return;
    }
    if (!ctx.lora_config->enabled() || !ctx.lora_weights->enabled()) {
        return;
    }
    if (op.lora_targets.empty()) {
        return;
    }

    const int Hq = static_cast<int>(ctx.cfg.NumQueryHeads);
    const int Hkv = static_cast<int>(ctx.cfg.NumKeyValHeads);
    const int Hs = static_cast<int>(ctx.cfg.head_size());
    const int C = static_cast<int>(ctx.cfg.HiddenSize);
    const int D = static_cast<int>(ctx.cfg.IntermediateSize);
    const int BT = static_cast<int>(B * T);

    const int rank = ctx.lora_config->rank;
    const float scaling = ctx.lora_config->scaling();
    const float dropout = ctx.lora_config->dropout;
    const bool training = ctx.lora_run_state->is_training;

    auto get_dropout_seed = [&](int proj_type) -> unsigned int {
        return ctx.lora_run_state->dropout_base_seed
               + static_cast<unsigned int>(layer_idx) * 1000000u
               + static_cast<unsigned int>(proj_type) * 100000u
               + static_cast<unsigned int>(ctx.lora_run_state->micro_step) * 10000u;
    };

    auto& lora_block = ctx.lora_weights->get_block(layer_idx, rs.MainStream);

    for (const auto& target : op.lora_targets) {
        if (target == "q" && lora_block.attention.q.has_value()) {
            modules::detail::apply_lora_contribution(
                output, 0, input, lora_block.attention.q.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(0), training,
                BT, C, Hq * Hs, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "k" && lora_block.attention.k.has_value()) {
            modules::detail::apply_lora_contribution(
                output, Hq * Hs, input, lora_block.attention.k.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(1), training,
                BT, C, Hkv * Hs, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "v" && lora_block.attention.v.has_value()) {
            modules::detail::apply_lora_contribution(
                output, (Hq + Hkv) * Hs, input, lora_block.attention.v.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(2), training,
                BT, C, Hkv * Hs, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "o" && lora_block.attention.o.has_value()) {
            modules::detail::apply_lora_contribution(
                output, 0, input, lora_block.attention.o.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(3), training,
                BT, Hq * Hs, C, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "up" && lora_block.mlp.up.has_value()) {
            modules::detail::apply_lora_contribution(
                output, 0, input, lora_block.mlp.up.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(4), training,
                BT, C, D, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "gate" && lora_block.mlp.gate.has_value()) {
            modules::detail::apply_lora_contribution(
                output, D, input, lora_block.mlp.gate.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(5), training,
                BT, C, D, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        } else if (target == "down" && lora_block.mlp.down.has_value()) {
            modules::detail::apply_lora_contribution(
                output, 0, input, lora_block.mlp.down.value(),
                ctx.lora_run_state->intermediate, ctx.lora_run_state->slice,
                scaling, dropout, get_dropout_seed(6), training,
                BT, D, C, rank,
                rs.CublasLtHandle, rs.CuBlasWorkspace, rs.MainStream);
        }
    }
}

void execute_matmul(RecomputeContext& ctx,
                    const RecomputeOp& op,
                    int layer_idx,
                    long B,
                    long T,
                    const InputMap& inputs,
                    const std::unordered_map<std::string, Tensor*>& outputs) {
    auto& rs = ctx.rs;
    const auto& cfg = ctx.cfg;
    const int Bv = static_cast<int>(B);
    const int Tv = static_cast<int>(T);

    std::string weight_key;
    std::string bias_key;
    std::string input_key;

    for (const auto& raw : op.inputs) {
        std::string stripped;
        is_optional_ref(raw, stripped);
        std::string param_name;
        if (is_param_ref(stripped, param_name)) {
            if (weight_key.empty()) {
                weight_key = param_name;
            } else if (bias_key.empty()) {
                bias_key = param_name;
            }
            continue;
        }
        std::string input_name;
        if (is_input_ref(stripped, input_name)) {
            if (input_key.empty()) {
                input_key = input_name;
            }
            continue;
        }
        std::string global_name;
        if (is_global_ref(stripped, global_name)) {
            if (input_key.empty()) {
                input_key = global_name;
            }
            continue;
        }
        if (input_key.empty()) {
            input_key = stripped;
        }
    }

    if (weight_key.empty() || input_key.empty()) {
        throw std::runtime_error("DSL recompute: matmul missing weight or input");
    }

    Tensor& weight = *inputs.tensors.at(weight_key);
    Tensor& inp = *inputs.tensors.at(input_key);

    if (op.outputs.empty()) {
        throw std::runtime_error("DSL recompute: matmul missing output");
    }
    const std::string& out_name = op.outputs.front();
    auto out_it = outputs.find(out_name);
    if (out_it == outputs.end()) {
        throw std::runtime_error("DSL recompute: matmul missing output");
    }
    Tensor& out = *out_it->second;

    std::optional<Tensor> bias;
    if (!bias_key.empty()) {
        auto it = inputs.tensors.find(bias_key);
        if (it != inputs.tensors.end()) {
            bias = *it->second;
        }
    }

    const std::string& weight_name = inputs.param_names.at(weight_key);
    const bool force_basic_matmul =
        (weight_name.find("router_weight") != std::string::npos) ||
        (weight_name.find("shared_expert_") != std::string::npos);
    std::optional<modules::MatmulOp> matmul_op;
    const MatmulForwardPlan* plan = nullptr;

    if (!force_basic_matmul) {
        try {
            matmul_op = parse_matmul_op(op.attrs, weight_name);
            plan = plan_for_matmul(ctx.layer_plan, *matmul_op);
        } catch (...) {
            // Unknown matmul op (e.g., router weights) -> fall back to basic matmul.
        }
    }

    const bool allow_quant = allow_quant_layer(ctx.options, cfg, layer_idx);
    const bool have_recipe = ctx.options.TrainingRecipe != nullptr;
    // Match forward path: if the training recipe is enabled, use it during recompute.
    // This ensures recompute activations match non-recompute forward (even with FP8).
    const bool use_recipe = (matmul_op.has_value() && (plan ? (plan->use_recipe && have_recipe) : have_recipe));

    const int C_in = (inp.Rank >= 3) ? static_cast<int>(inp.Sizes[2]) : static_cast<int>(inp.Sizes[1]);
    const int C_out = (out.Rank >= 3) ? static_cast<int>(out.Sizes[2]) : static_cast<int>(out.Sizes[1]);

    Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {B * T, C_in});
    Tensor out_flat = (out.Rank == 2) ? out : view_tensor(out, {B * T, C_out});

    if (use_recipe) {
        const recipes::Recipe& recipe = *ctx.options.TrainingRecipe;
        modules::MatmulContext mm_ctx;
        mm_ctx.out = &out_flat;
        mm_ctx.inp = &inp_flat;
        mm_ctx.weight = &weight;
        mm_ctx.bias = bias ? &*bias : nullptr;
        mm_ctx.B = Bv;
        mm_ctx.T = Tv;
        mm_ctx.C_in = C_in;
        mm_ctx.C_out = C_out;
        mm_ctx.run_state = &rs;
        mm_ctx.stream = rs.MainStream;
        mm_ctx.layer_idx = layer_idx;
        mm_ctx.op = *matmul_op;
        mm_ctx.allow_fp8 = plan ? plan->allow_fp8 : allow_quant;
        mm_ctx.allow_fp4 = plan ? plan->allow_fp4 : (allow_quant && ctx.options.fp4_enabled());
        if (mm_ctx.allow_fp8) {
            mm_ctx.inp_quant = fp8_forward_buffer(rs, mm_ctx.op);
            mm_ctx.delayed_quantizer_idx = plan ? plan->delayed_quantizer_idx
                                                : fp8_quantizer_index(rs, mm_ctx.op, layer_idx);
            if ((!plan || plan->use_fp8_cache) && ctx.get_fp8_cached_weight) {
                mm_ctx.cached_weight = ctx.get_fp8_cached_weight(weight_name, weight, rs.MainStream);
            }
        }
        if (mm_ctx.allow_fp4 && (!plan || plan->use_fp4_cache) && ctx.get_fp4_cached_weight) {
            if (const auto* fp4_cache = ctx.get_fp4_cached_weight(weight_name, weight, rs.MainStream)) {
                mm_ctx.cached_fp4_data = &fp4_cache->data;
                mm_ctx.cached_fp4_scales = &fp4_cache->scales;
                mm_ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
            }
        }
        recipe.forward_matmul(mm_ctx);
    } else {
        matmul(out_flat, weight, inp_flat, bias, nullptr, nullptr,
               rs.CublasLtHandle, rs.CuBlasWorkspace,
               C_out, static_cast<int>(B * T), C_in,
               swap_transpose(parse_transpose(op.attrs)), false, rs.MainStream);
    }

    apply_lora_targets(ctx, op, layer_idx, B, T, inp, out);
}

void execute_fused_residual(RecomputeContext& ctx,
                            const RecomputeOp& op,
                            int layer_idx,
                            long B,
                            long T,
                            const InputMap& inputs,
                            const std::unordered_map<std::string, Tensor*>& outputs) {
    (void)op;
    Tensor* residual = nullptr;
    Tensor* input = nullptr;
    Tensor* rstd = nullptr;
    Tensor* weight = nullptr;

    for (const auto& kv : inputs.tensors) {
        const std::string& key = kv.first;
        if (inputs.param_names.count(key)) {
            weight = kv.second;
            continue;
        }
        if (is_rstd_name(key)) {
            rstd = kv.second;
            continue;
        }
        if (!residual && is_residual_name(key)) {
            residual = kv.second;
            continue;
        }
        if (!input) {
            input = kv.second;
        }
    }

    if (!weight || !rstd) {
        throw std::runtime_error("DSL recompute: fused_residual_rmsnorm_apply_saved missing weight/rstd");
    }

    Tensor* res_out = nullptr;
    Tensor* y_out = nullptr;
    std::string res_out_name;
    for (const auto& kv : outputs) {
        if (is_residual_name(kv.first)) {
            res_out = kv.second;
            res_out_name = kv.first;
        } else {
            y_out = kv.second;
        }
    }
    if (!y_out) {
        throw std::runtime_error("DSL recompute: fused_residual_rmsnorm_apply_saved missing y output");
    }

    const int C = static_cast<int>(ctx.cfg.HiddenSize);

    // Both FFT mode and LoRA mode: handle recomputation for ln1_fused specially.
    // In backward mode, @input:residual and @input:x refer to previous layer's outputs which
    // are stored in shared buffers and contain garbage. Instead, we use the saved res_ffn
    // (stored in get_residual during forward) and recompute ln1 only.
    // This applies to BOTH FFT mode and LoRA mode because both have the shared buffer problem.
    const bool is_res_ffn_output = (res_out_name == "res_ffn" || res_out_name == "residual_ffn");

    if (is_res_ffn_output) {
        // ln1_fused group: Recompute ln1 only from saved res_ffn[layer_idx] and ln1_rstd[layer_idx].
        // - res_ffn[layer_idx] = residual + x (stored in get_residual(layer_idx) during forward)
        // - ln1 = rmsnorm(res_ffn) using saved ln1_rstd
        auto& acts = ctx.rs.simplified_acts(layer_idx);
        if (!acts.ln1.Data) {
            ctx.rs.temp_acquire(acts.ln1);
        }
        // Use the current layer's res_ffn (stored during forward pass)
        Tensor& res_ffn = ctx.rs.get_residual(layer_idx, ctx.rs.MainStream);

        rmsnorm_apply_saved(acts.ln1, res_ffn, *weight, acts.ln1_rstd,
                            static_cast<int>(B), static_cast<int>(T), C, ctx.rs.MainStream);

        return;
    }

    // Use full fused op for ln2_fused cases
    if (!residual || !input) {
        throw std::runtime_error("DSL recompute: fused_residual_rmsnorm_apply_saved missing residual/input");
    }
    if (!res_out) {
        throw std::runtime_error("DSL recompute: fused_residual_rmsnorm_apply_saved missing res output");
    }

    const int BT = static_cast<int>(B * T);

    // Use apply_saved with saved rstd for all cases.
    // In FFT mode: att is now saved (not recomputed due to cuDNN non-determinism),
    // so att_out is recomputed from saved att (deterministic matmul), and res_att
    // matches forward. This means saved ln2_rstd is valid and should be used.
    // In LoRA mode: inputs match forward, so saved rstd is valid.
    fused_residual_rmsnorm_apply_saved(*res_out, *y_out, *residual, *input, *weight, *rstd,
                                       BT, C, ctx.rs.MainStream);

}

void execute_rmsnorm(RecomputeContext& ctx,
                     const RecomputeOp& op,
                     const InputMap& inputs,
                     const std::unordered_map<std::string, Tensor*>& outputs,
                     int layer_idx,
                     long B,
                     long T) {
    (void)layer_idx;
    Tensor* input = nullptr;
    Tensor* rstd = nullptr;
    Tensor* weight = nullptr;

    for (const auto& kv : inputs.tensors) {
        const std::string& key = kv.first;
        if (inputs.param_names.count(key)) {
            weight = kv.second;
            continue;
        }
        if (is_rstd_name(key)) {
            rstd = kv.second;
            continue;
        }
        input = kv.second;
    }

    if (!input || !weight || !rstd) {
        throw std::runtime_error("DSL recompute: rmsnorm_apply_saved missing inputs");
    }

    if (op.outputs.empty()) {
        throw std::runtime_error("DSL recompute: rmsnorm_apply_saved missing output");
    }
    const std::string& out_name = op.outputs.front();
    auto out_it = outputs.find(out_name);
    if (out_it == outputs.end()) {
        throw std::runtime_error("DSL recompute: rmsnorm_apply_saved missing output");
    }
    Tensor& out = *out_it->second;
    const int C = static_cast<int>(ctx.cfg.HiddenSize);
    rmsnorm_apply_saved(out, *input, *weight, *rstd, static_cast<int>(B), static_cast<int>(T),
                        C, ctx.rs.MainStream);

}

void execute_qkv_qk_norm_rope(RecomputeContext& ctx,
                              const RecomputeOp& op,
                              int layer_idx,
                              long B,
                              long T,
                              const InputMap& inputs,
                              const std::unordered_map<std::string, Tensor*>& outputs) {
    (void)layer_idx;
    Tensor* qkv_in = nullptr;
    Tensor* q_norm = nullptr;
    Tensor* k_norm = nullptr;
    Tensor* freqs = nullptr;
    Tensor* pos_ids = nullptr;

    for (const auto& kv : inputs.tensors) {
        const std::string& key = kv.first;
        if (key == "qkv" || key == "qkv_rope") {
            qkv_in = kv.second;
        } else if (key == "q_norm_weight") {
            q_norm = kv.second;
        } else if (key == "k_norm_weight") {
            k_norm = kv.second;
        } else if (key == "freq_cis" || key == "rope_freqs") {
            freqs = kv.second;
        } else if (key == "position_ids") {
            pos_ids = kv.second;
        }
    }

    if (!qkv_in || !freqs || !pos_ids) {
        throw std::runtime_error("DSL recompute: qkv_qk_norm_rope missing inputs");
    }

    auto it_qkv = outputs.find("qkv_rope");
    Tensor* qkv_out = (it_qkv != outputs.end()) ? it_qkv->second : nullptr;
    if (!qkv_out) {
        auto it_qkv2 = outputs.find("qkv");
        if (it_qkv2 != outputs.end()) {
            qkv_out = it_qkv2->second;
        }
    }
    if (!qkv_out) {
        throw std::runtime_error("DSL recompute: qkv_qk_norm_rope missing qkv output");
    }

    Tensor* q_rstd = nullptr;
    Tensor* k_rstd = nullptr;
    if (auto it = outputs.find("q_rstd"); it != outputs.end()) q_rstd = it->second;
    if (auto it = outputs.find("k_rstd"); it != outputs.end()) k_rstd = it->second;

    if (qkv_in->Data != qkv_out->Data) {
        cudaMemcpyAsync(qkv_out->Data, qkv_in->Data, qkv_in->bytes(),
                        cudaMemcpyDeviceToDevice, ctx.rs.MainStream);
    }

    const int Hq = static_cast<int>(ctx.cfg.NumQueryHeads);
    const int Hkv = static_cast<int>(ctx.cfg.NumKeyValHeads);
    const int Hs = static_cast<int>(ctx.cfg.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    Tensor qkv_view = (qkv_out->Rank == 4)
        ? view_tensor(*qkv_out, {B, T, qkv_channels})
        : *qkv_out;

    const LayerForwardPlan* fwd_plan = ctx.layer_plan;
    bool use_qk_norm = ctx.cfg.use_qk_norm || ctx.cfg.UseQKNorm;
    bool rope_fused = false;
    int rotary_dim = resolve_rotary_dim(op.attrs, ctx.cfg);

    if (fwd_plan && fwd_plan->attn.valid) {
        use_qk_norm = fwd_plan->attn.use_qk_norm;
        rope_fused = fwd_plan->attn.rope_fused;
        if (fwd_plan->attn.rotary_dim > 0) {
            rotary_dim = fwd_plan->attn.rotary_dim;
        }
    }

    const float eps = resolve_eps(op.attrs, ctx.cfg);

    if (use_qk_norm) {
        if (!q_norm || !k_norm || !q_rstd || !k_rstd) {
            throw std::runtime_error("DSL recompute: qkv_qk_norm_rope missing qk-norm buffers");
        }
        if (rope_fused) {
            qkv_qk_norm_rope_forward(qkv_view, *q_rstd, *k_rstd, *q_norm, *k_norm,
                                     *freqs, reinterpret_cast<int*>(pos_ids->Data),
                                     eps,
                                     static_cast<int>(B), static_cast<int>(T),
                                     Hq, Hkv, Hs, ctx.rs.MainStream);
        } else {
            const int q_rows = Hq * Hs;
            qkv_head_rmsnorm_forward(qkv_view, *q_rstd, *q_norm,
                                     eps,
                                     static_cast<int>(B), static_cast<int>(T),
                                     qkv_channels, Hq, Hs, 0, ctx.rs.MainStream);
            qkv_head_rmsnorm_forward(qkv_view, *k_rstd, *k_norm,
                                     eps,
                                     static_cast<int>(B), static_cast<int>(T),
                                     qkv_channels, Hkv, Hs, q_rows, ctx.rs.MainStream);
            Tensor qkv_rope = (qkv_out->Rank == 4)
                ? *qkv_out
                : view_tensor(*qkv_out, {B, T, Hq + 2 * Hkv, Hs});
            rope_forward(qkv_rope, qkv_rope, *freqs, reinterpret_cast<int*>(pos_ids->Data), nullptr,
                         static_cast<int>(B), static_cast<int>(T),
                         Hq, Hkv, Hs, rotary_dim, ctx.rs.MainStream);
        }
    } else {
        Tensor qkv_rope = (qkv_out->Rank == 4)
            ? *qkv_out
            : view_tensor(*qkv_out, {B, T, Hq + 2 * Hkv, Hs});
        rope_forward(qkv_rope, qkv_rope, *freqs, reinterpret_cast<int*>(pos_ids->Data), nullptr,
                     static_cast<int>(B), static_cast<int>(T),
                     Hq, Hkv, Hs, rotary_dim, ctx.rs.MainStream);
    }

}

void execute_rope(RecomputeContext& ctx,
                  const RecomputeOp& op,
                  int layer_idx,
                  long B,
                  long T,
                  const InputMap& inputs,
                  const std::unordered_map<std::string, Tensor*>& outputs) {
    (void)layer_idx;
    Tensor* qkv_in = nullptr;
    Tensor* freqs = nullptr;
    Tensor* pos_ids = nullptr;

    for (const auto& kv : inputs.tensors) {
        const std::string& key = kv.first;
        if (key == "qkv" || key == "qkv_rope") {
            qkv_in = kv.second;
        } else if (key == "freq_cis" || key == "rope_freqs") {
            freqs = kv.second;
        } else if (key == "position_ids") {
            pos_ids = kv.second;
        }
    }

    if (!qkv_in || !freqs || !pos_ids) {
        throw std::runtime_error("DSL recompute: rope missing inputs (qkv, freq_cis, position_ids)");
    }

    auto it_qkv = outputs.find("qkv_rope");
    Tensor* qkv_out = (it_qkv != outputs.end()) ? it_qkv->second : nullptr;
    if (!qkv_out) {
        auto it_qkv2 = outputs.find("qkv");
        if (it_qkv2 != outputs.end()) {
            qkv_out = it_qkv2->second;
        }
    }
    if (!qkv_out) {
        throw std::runtime_error("DSL recompute: rope missing qkv_rope output");
    }

    // Copy input to output if different buffers
    if (qkv_in->Data != qkv_out->Data) {
        cudaMemcpyAsync(qkv_out->Data, qkv_in->Data, qkv_in->bytes(),
                        cudaMemcpyDeviceToDevice, ctx.rs.MainStream);
    }

    const int Hq = static_cast<int>(ctx.cfg.NumQueryHeads);
    const int Hkv = static_cast<int>(ctx.cfg.NumKeyValHeads);
    const int Hs = static_cast<int>(ctx.cfg.head_size());
    int rotary_dim = resolve_rotary_dim(op.attrs, ctx.cfg);

    // Check if forward plan has rotary_dim override
    const LayerForwardPlan* fwd_plan = ctx.layer_plan;
    if (fwd_plan && fwd_plan->attn.valid && fwd_plan->attn.rotary_dim > 0) {
        rotary_dim = fwd_plan->attn.rotary_dim;
    }

    // Reshape to 4D if needed for rope_forward
    Tensor qkv_rope = (qkv_out->Rank == 4)
        ? *qkv_out
        : view_tensor(*qkv_out, {B, T, Hq + 2 * Hkv, Hs});

    rope_forward(qkv_rope, qkv_rope, *freqs, reinterpret_cast<int*>(pos_ids->Data), nullptr,
                 static_cast<int>(B), static_cast<int>(T),
                 Hq, Hkv, Hs, rotary_dim, ctx.rs.MainStream);
}

void execute_flash_attention(RecomputeContext& ctx,
                             long B,
                             long T,
                             const InputMap& inputs,
                             const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* qkv = nullptr;
    if (auto it = inputs.tensors.find("qkv_rope"); it != inputs.tensors.end()) {
        qkv = it->second;
    } else if (auto it2 = inputs.tensors.find("qkv"); it2 != inputs.tensors.end()) {
        qkv = it2->second;
    }
    if (!qkv) {
        throw std::runtime_error("DSL recompute: flash_attention missing qkv");
    }

    auto it_att = outputs.find("att");
    auto it_lse = outputs.find("lse");
    if (it_att == outputs.end() || it_lse == outputs.end()) {
        throw std::runtime_error("DSL recompute: flash_attention missing outputs");
    }

    Tensor& att = *it_att->second;
    Tensor& lse = *it_lse->second;

    const int Hq = static_cast<int>(ctx.cfg.NumQueryHeads);
    const int Hkv = static_cast<int>(ctx.cfg.NumKeyValHeads);
    const int Hs = static_cast<int>(ctx.cfg.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);

    if (!ctx.rs.scratch().cudnn_workspace.Data) {
        ctx.rs.temp_acquire(ctx.rs.scratch().cudnn_workspace);
    }

    Tensor att_view = view_tensor(att, {B, T, Hq, Hs});
    Tensor lse_view = view_tensor(lse, {B, Hq, T});
    Tensor qkv_view = (qkv->Rank == 4)
        ? *qkv
        : view_tensor(*qkv, {B, T, Hq + 2 * Hkv, Hs});

    if (!cudnn_gqa_ok) {
        attention_forward_custom(att_view, lse_view, qkv_view,
                                 static_cast<int>(B), static_cast<int>(T),
                                 Hq, Hkv, Hs, ctx.rs.MainStream);
    } else {
        attention_forward_cudnn(att_view, lse_view, qkv_view, ctx.rs.scratch().cudnn_workspace,
                                ctx.rs.CudnnHandle, static_cast<int>(B), static_cast<int>(T),
                                Hq, Hkv, Hs, ctx.rs.MainStream);
    }
}

void execute_swiglu(RecomputeContext& ctx,
                    const RecomputeOp& op,
                    long B,
                    long T,
                    const InputMap& inputs,
                    const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* inp = nullptr;
    if (auto it = inputs.tensors.find("mlp_up"); it != inputs.tensors.end()) {
        inp = it->second;
    } else if (auto it2 = inputs.tensors.find("mlp_up_flat"); it2 != inputs.tensors.end()) {
        inp = it2->second;
    } else if (auto it3 = inputs.tensors.find("expert_gate_up"); it3 != inputs.tensors.end()) {
        inp = it3->second;
    } else if (auto it4 = inputs.tensors.find("expert_gate_up_flat"); it4 != inputs.tensors.end()) {
        inp = it4->second;
    } else {
        for (const auto& kv : inputs.tensors) {
            if (kv.second && kv.second->DType != ETensorDType::INT32) {
                inp = kv.second;
                break;
            }
        }
    }
    if (!inp) {
        throw std::runtime_error("DSL recompute: swiglu missing input");
    }
    if (op.outputs.empty()) {
        throw std::runtime_error("DSL recompute: swiglu missing output");
    }
    const std::string& out_name = op.outputs.front();
    auto out_it = outputs.find(out_name);
    if (out_it == outputs.end()) {
        throw std::runtime_error("DSL recompute: swiglu missing output");
    }
    Tensor& out = *out_it->second;
    if (inp->Rank == 2) {
        const long N = inp->Sizes[0];
        const long D = inp->Sizes[1] / 2;
        Tensor inp_view = view_tensor(*inp, {1, N, 2 * D});
        Tensor out_view = (out.Rank == 2) ? view_tensor(out, {1, N, D}) : out;
        swiglu_forward(out_view, inp_view, nullptr, 1, static_cast<int>(N), static_cast<int>(D), ctx.rs.MainStream);
        return;
    }
    int D = static_cast<int>(ctx.cfg.IntermediateSize);
    if (inp->Rank >= 3 && inp->Sizes[2] > 0) {
        D = static_cast<int>(inp->Sizes[2] / 2);
    }
    swiglu_forward(out, *inp, nullptr, static_cast<int>(B), static_cast<int>(T), D, ctx.rs.MainStream);
}

void execute_moe_router_probs(RecomputeContext& ctx,
                              const InputMap& inputs,
                              const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* logits = nullptr;
    if (auto it = inputs.tensors.find("router_logits"); it != inputs.tensors.end()) {
        logits = it->second;
    } else if (!inputs.tensors.empty()) {
        logits = inputs.tensors.begin()->second;
    }
    if (!logits) {
        throw std::runtime_error("DSL recompute: moe_router_probs missing input");
    }
    if (outputs.empty()) {
        throw std::runtime_error("DSL recompute: moe_router_probs missing output");
    }
    Tensor& out = *outputs.begin()->second;

    const bool use_sigmoid = ctx.cfg.moe_config.has_value() && ctx.cfg.moe_config->norm_topk_prob;

    if (use_sigmoid) {
        const int num_elements = static_cast<int>(out.nelem());
        if (logits->DType == ETensorDType::BF16) {
            moe_sigmoid_forward(out.get<nv_bfloat16>(),
                                logits->get<nv_bfloat16>(),
                                num_elements, ctx.rs.MainStream);
        } else {
            moe_sigmoid_forward(out.get<float>(),
                                logits->get<float>(),
                                num_elements, ctx.rs.MainStream);
        }
        return;
    }

    const int num_tokens = static_cast<int>(logits->Sizes[0]);
    const int num_experts = static_cast<int>(logits->Sizes[1]);
    if (logits->DType == ETensorDType::BF16) {
        moe_softmax_forward(out.get<nv_bfloat16>(),
                            logits->get<nv_bfloat16>(),
                            num_tokens, num_experts, ctx.rs.MainStream);
    } else {
        moe_softmax_forward(out.get<float>(),
                            logits->get<float>(),
                            num_tokens, num_experts, ctx.rs.MainStream);
    }
}

void execute_moe_topk(RecomputeContext& ctx,
                      const RecomputeOp& op,
                      const InputMap& inputs,
                      const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* probs = nullptr;
    if (auto it = inputs.tensors.find("router_probs"); it != inputs.tensors.end()) {
        probs = it->second;
    } else if (!inputs.tensors.empty()) {
        probs = inputs.tensors.begin()->second;
    }
    if (!probs) {
        throw std::runtime_error("DSL recompute: moe_topk missing input");
    }
    auto out_w = outputs.find("routing_weights");
    auto out_i = outputs.find("routing_indices");
    if (out_w == outputs.end() || out_i == outputs.end()) {
        throw std::runtime_error("DSL recompute: moe_topk missing outputs");
    }
    Tensor& weights = *out_w->second;
    Tensor& indices = *out_i->second;

    const int num_tokens = static_cast<int>(probs->Sizes[0]);
    const int num_experts = static_cast<int>(probs->Sizes[1]);
    const int top_k = resolve_top_k(op.attrs, ctx.cfg);
    const bool normalize = resolve_norm_topk(op.attrs, ctx.cfg);

    if (probs->DType == ETensorDType::BF16) {
        moe_topk_forward(indices.get<int>(),
                         weights.get<nv_bfloat16>(),
                         probs->get<nv_bfloat16>(),
                         num_tokens, num_experts, top_k, normalize, ctx.rs.MainStream);
    } else {
        moe_topk_forward(indices.get<int>(),
                         weights.get<float>(),
                         probs->get<float>(),
                         num_tokens, num_experts, top_k, normalize, ctx.rs.MainStream);
    }
}

void execute_moe_permute(RecomputeContext& ctx,
                         RecomputeScratch& scratch,
                         const RecomputeOp& op,
                         const InputMap& inputs,
                         const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* inp = nullptr;
    if (auto it = inputs.tensors.find("ln2"); it != inputs.tensors.end()) {
        inp = it->second;
    } else if (auto it2 = inputs.tensors.find("ln2_flat"); it2 != inputs.tensors.end()) {
        inp = it2->second;
    } else if (!inputs.tensors.empty()) {
        inp = inputs.tensors.begin()->second;
    }
    auto idx_it = inputs.tensors.find("routing_indices");
    if (!inp || idx_it == inputs.tensors.end()) {
        throw std::runtime_error("DSL recompute: moe_permute missing inputs");
    }
    Tensor& routing_indices = *idx_it->second;

    auto out_p = outputs.find("permuted_input");
    auto out_s = outputs.find("scatter_indices");
    if (out_p == outputs.end() || out_s == outputs.end()) {
        throw std::runtime_error("DSL recompute: moe_permute missing outputs");
    }
    Tensor& permuted = *out_p->second;
    Tensor& scatter_indices = *out_s->second;

    const int top_k = resolve_top_k(op.attrs, ctx.cfg);
    const int num_tokens = static_cast<int>((inp->Rank == 2) ? inp->Sizes[0] : (inp->Sizes[0] * inp->Sizes[1]));
    const int hidden_size = static_cast<int>((inp->Rank == 2) ? inp->Sizes[1] : inp->Sizes[2]);
    const int total_tokens = num_tokens * top_k;
    const int num_experts = static_cast<int>(ctx.cfg.NumExperts);

    Tensor inp_flat = (inp->Rank == 2) ? *inp : view_tensor(*inp, {num_tokens, hidden_size});

    Tensor expert_counts = ctx.rs.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_counts");
    Tensor expert_offsets = ctx.rs.Stack.allocate(ETensorDType::INT32, {num_experts + 1}, "moe_expert_offsets");
    Tensor expert_positions = ctx.rs.Stack.allocate(ETensorDType::INT32, {num_experts}, "moe_expert_positions");
    Tensor gather_indices = ctx.rs.Stack.allocate(ETensorDType::INT32, {total_tokens}, "moe_gather_indices");

    fill_zero(expert_positions, ctx.rs.MainStream);

    moe_compute_expert_counts(expert_counts.get<int>(),
                              routing_indices.get<int>(),
                              num_tokens, top_k, num_experts, ctx.rs.MainStream);

    moe_compute_expert_offsets(expert_offsets.get<int>(),
                               expert_counts.get<int>(),
                               num_experts, ctx.rs.MainStream);

    moe_build_indices(gather_indices.get<int>(),
                      scatter_indices.get<int>(),
                      routing_indices.get<int>(),
                      expert_offsets.get<int>(),
                      expert_positions.get<int>(),
                      num_tokens, top_k, num_experts, ctx.rs.MainStream);

    if (inp_flat.DType == ETensorDType::BF16) {
        moe_permute_tokens(permuted.get<nv_bfloat16>(),
                           inp_flat.get<nv_bfloat16>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, ctx.rs.MainStream);
    } else {
        moe_permute_tokens(permuted.get<float>(),
                           inp_flat.get<float>(),
                           gather_indices.get<int>(),
                           total_tokens, num_tokens, hidden_size, top_k, ctx.rs.MainStream);
    }

    scratch.moe_expert_offsets = expert_offsets;
    scratch.moe_gather_indices = gather_indices;
}

void execute_moe_grouped_gemm_gate_up(RecomputeContext& ctx,
                                      RecomputeScratch& scratch,
                                      const InputMap& inputs,
                                      const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* inp = nullptr;
    Tensor* weights = nullptr;
    if (auto it = inputs.tensors.find("permuted_input"); it != inputs.tensors.end()) {
        inp = it->second;
    } else if (auto it = inputs.tensors.find("permuted_input_flat"); it != inputs.tensors.end()) {
        inp = it->second;
    }
    for (const auto& kv : inputs.param_names) {
        auto it = inputs.tensors.find(kv.first);
        if (it != inputs.tensors.end()) {
            weights = it->second;
            break;
        }
    }
    if (!inp) {
        for (const auto& kv : inputs.tensors) {
            if (inputs.param_names.count(kv.first)) {
                continue;
            }
            if (kv.second && kv.second->DType != ETensorDType::INT32) {
                inp = kv.second;
                break;
            }
        }
    }
    if (!inp || !weights || !scratch.moe_expert_offsets.Data) {
        throw std::runtime_error("DSL recompute: moe_grouped_gemm_gate_up missing inputs");
    }
    if (outputs.empty()) {
        throw std::runtime_error("DSL recompute: moe_grouped_gemm_gate_up missing output");
    }
    Tensor& out = *outputs.begin()->second;

    const int num_experts = static_cast<int>(ctx.cfg.NumExperts);
    const int hidden_size = static_cast<int>(ctx.cfg.HiddenSize);
    const int intermediate_size = (ctx.cfg.MoeIntermediateSize > 0)
        ? static_cast<int>(ctx.cfg.MoeIntermediateSize)
        : static_cast<int>(ctx.cfg.IntermediateSize);
    const int weight_experts = (weights->Rank > 0) ? static_cast<int>(weights->Sizes[0]) : num_experts;
    MoeCompactInfo compact = build_moe_compact_info(scratch.moe_expert_offsets.get<int>(),
                                                    num_experts,
                                                    weight_experts,
                                                    ctx.rs.MainStream);
    const int* host_offsets_ptr = compact.host_offsets.empty() ? nullptr : compact.host_offsets.data();
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, ctx.rs.MainStream);
    } else if (weights->DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(out.get<nv_bfloat16>(),
                                 inp->get<nv_bfloat16>(),
                                 weights->get<nv_bfloat16>(),
                                 scratch.moe_expert_offsets.get<int>(),
                                 num_experts, hidden_size, intermediate_size,
                                 ctx.rs.cublas_handle(), ctx.rs.MainStream,
                                 host_offsets_ptr,
                                 active_ptr,
                                 weight_is_compact,
                                 num_active);
    } else {
        moe_grouped_gemm_gate_up(out.get<float>(),
                                 inp->get<float>(),
                                 weights->get<float>(),
                                 scratch.moe_expert_offsets.get<int>(),
                                 num_experts, hidden_size, intermediate_size,
                                 ctx.rs.cublas_handle(), ctx.rs.MainStream,
                                 host_offsets_ptr,
                                 active_ptr,
                                 weight_is_compact,
                                 num_active);
    }
}

void execute_moe_grouped_gemm_down(RecomputeContext& ctx,
                                   RecomputeScratch& scratch,
                                   const InputMap& inputs,
                                   const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* inp = nullptr;
    Tensor* weights = nullptr;
    if (auto it = inputs.tensors.find("expert_act"); it != inputs.tensors.end()) {
        inp = it->second;
    } else if (auto it = inputs.tensors.find("expert_act_flat"); it != inputs.tensors.end()) {
        inp = it->second;
    }
    for (const auto& kv : inputs.param_names) {
        auto it = inputs.tensors.find(kv.first);
        if (it != inputs.tensors.end()) {
            weights = it->second;
            break;
        }
    }
    if (!inp) {
        for (const auto& kv : inputs.tensors) {
            if (inputs.param_names.count(kv.first)) {
                continue;
            }
            if (kv.second && kv.second->DType != ETensorDType::INT32) {
                inp = kv.second;
                break;
            }
        }
    }
    if (!inp || !weights || !scratch.moe_expert_offsets.Data) {
        throw std::runtime_error("DSL recompute: moe_grouped_gemm_down missing inputs");
    }
    if (outputs.empty()) {
        throw std::runtime_error("DSL recompute: moe_grouped_gemm_down missing output");
    }
    Tensor& out = *outputs.begin()->second;

    const int num_experts = static_cast<int>(ctx.cfg.NumExperts);
    const int hidden_size = static_cast<int>(ctx.cfg.HiddenSize);
    const int intermediate_size = (ctx.cfg.MoeIntermediateSize > 0)
        ? static_cast<int>(ctx.cfg.MoeIntermediateSize)
        : static_cast<int>(ctx.cfg.IntermediateSize);
    const int weight_experts = (weights->Rank > 0) ? static_cast<int>(weights->Sizes[0]) : num_experts;
    MoeCompactInfo compact = build_moe_compact_info(scratch.moe_expert_offsets.get<int>(),
                                                    num_experts,
                                                    weight_experts,
                                                    ctx.rs.MainStream);
    const int* host_offsets_ptr = compact.host_offsets.empty() ? nullptr : compact.host_offsets.data();
    const int* active_ptr = compact.active_experts.empty() ? nullptr : compact.active_experts.data();
    const int num_active = compact.active_experts.empty() ? -1 : compact.num_active;
    const bool weight_is_compact = compact.weight_is_compact;

    if (weight_is_compact && compact.active_experts.empty()) {
        fill_zero(out, ctx.rs.MainStream);
    } else if (inp->DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(out.get<nv_bfloat16>(),
                              inp->get<nv_bfloat16>(),
                              weights->get<nv_bfloat16>(),
                              scratch.moe_expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              ctx.rs.cublas_handle(), ctx.rs.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    } else {
        moe_grouped_gemm_down(out.get<float>(),
                              inp->get<float>(),
                              weights->get<float>(),
                              scratch.moe_expert_offsets.get<int>(),
                              num_experts, hidden_size, intermediate_size,
                              ctx.rs.cublas_handle(), ctx.rs.MainStream,
                              host_offsets_ptr,
                              active_ptr,
                              weight_is_compact,
                              num_active);
    }
}

void execute_moe_unpermute(RecomputeContext& ctx,
                           const RecomputeOp& op,
                           const InputMap& inputs,
                           const std::unordered_map<std::string, Tensor*>& outputs) {
    Tensor* expert_out = nullptr;
    Tensor* routing_weights = nullptr;
    Tensor* scatter_indices = nullptr;
    for (const auto& kv : inputs.tensors) {
        if (kv.first == "routing_weights") {
            routing_weights = kv.second;
        } else if (kv.first == "scatter_indices") {
            scatter_indices = kv.second;
        } else if (!expert_out) {
            expert_out = kv.second;
        }
    }
    if (!expert_out || !routing_weights || !scatter_indices) {
        throw std::runtime_error("DSL recompute: moe_unpermute missing inputs");
    }
    if (outputs.empty()) {
        throw std::runtime_error("DSL recompute: moe_unpermute missing output");
    }
    Tensor& out = *outputs.begin()->second;

    const int top_k = resolve_top_k(op.attrs, ctx.cfg);
    const int num_tokens = static_cast<int>(routing_weights->Sizes[0]);
    const int total_tokens = num_tokens * top_k;
    const int hidden_size = static_cast<int>(ctx.cfg.HiddenSize);

    if (expert_out->DType == ETensorDType::BF16) {
        moe_unpermute_and_combine(out.get<nv_bfloat16>(),
                                  expert_out->get<nv_bfloat16>(),
                                  routing_weights->get<nv_bfloat16>(),
                                  scatter_indices->get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  ctx.rs.MainStream);
    } else {
        moe_unpermute_and_combine(out.get<float>(),
                                  expert_out->get<float>(),
                                  routing_weights->get<float>(),
                                  scatter_indices->get<int>(),
                                  num_tokens, total_tokens, hidden_size, top_k,
                                  ctx.rs.MainStream);
    }
}

}  // namespace

void RecomputePlan::init_from_layout(const ActivationLayoutIR& layout,
                                     const AttrMap& module_config,
                                     const modules::ModelConfig& model_cfg) {
    mPlan = {};
    mDependencies.clear();

    std::unordered_map<std::string, std::string> alias_map = layout.build_alias_map();

    std::unordered_set<std::string> enabled_slots;
    enabled_slots.reserve(layout.slots.size());
    for (const auto& slot : layout.slots) {
        if (slot.scope != ActivationScope::Block) {
            continue;
        }
        if (!eval_condition(slot.condition, module_config, model_cfg)) {
            continue;
        }
        enabled_slots.insert(slot.name);
    }

    std::unordered_map<std::string, std::vector<const ActivationSlotIR*>> groups;
    std::vector<const ActivationSlotIR*> singles;

    for (const auto& slot : layout.slots) {
        if (slot.scope != ActivationScope::Block) {
            continue;
        }
        if (!eval_condition(slot.condition, module_config, model_cfg)) {
            continue;
        }
        if (!slot.recompute_in_backward &&
            slot.memory_hint != ActivationMemoryHint::Recompute) {
            continue;
        }
        if (!slot.recompute_group.empty()) {
            groups[slot.recompute_group].push_back(&slot);
        } else {
            singles.push_back(&slot);
        }
    }

    std::vector<RecomputeOp> ops;

    auto build_outputs = [&](const ActivationSlotIR& leader,
                             const std::vector<const ActivationSlotIR*>& slots) -> std::vector<std::string> {
        std::vector<std::string> outputs;
        if (!leader.recompute_outputs.empty()) {
            outputs = leader.recompute_outputs;
        } else {
            outputs.reserve(slots.size());
            for (const auto* slot : slots) {
                outputs.push_back(slot->name);
            }
        }
        for (auto& out : outputs) {
            out = canonicalize_name(alias_map, out);
        }
        outputs.erase(std::remove_if(outputs.begin(), outputs.end(),
                                     [&](const std::string& name) {
                                         return enabled_slots.find(name) == enabled_slots.end();
                                     }),
                      outputs.end());
        return outputs;
    };

    for (auto& [group_id, slots] : groups) {
        const ActivationSlotIR* leader = nullptr;
        for (const auto* slot : slots) {
            if (!slot->recompute_op.empty()) {
                leader = slot;
                break;
            }
        }
        if (!leader) {
            throw std::runtime_error("DSL recompute: group missing recompute_op: " + group_id);
        }
        for (const auto* slot : slots) {
            if (!slot->recompute_op.empty() && slot->recompute_op != leader->recompute_op) {
                throw std::runtime_error("DSL recompute: group has mismatched recompute_op: " + group_id);
            }
        }
        std::vector<std::string> outputs = build_outputs(*leader, slots);
        if (outputs.empty()) {
            continue;
        }
        if (!leader->recompute_outputs.empty()) {
            for (const auto* slot : slots) {
                if (enabled_slots.find(slot->name) == enabled_slots.end()) {
                    continue;
                }
                const std::string canon = canonicalize_name(alias_map, slot->name);
                if (std::find(outputs.begin(), outputs.end(), canon) == outputs.end()) {
                    throw std::runtime_error("DSL recompute: group outputs missing slot " + canon);
                }
            }
        }
        RecomputeOp op;
        op.op_type = leader->recompute_op;
        op.inputs.reserve(leader->recompute_from.size());
        for (const auto& dep : leader->recompute_from) {
            op.inputs.push_back(canonicalize_ref(alias_map, dep));
        }
        op.outputs = std::move(outputs);
        op.attrs = leader->recompute_attrs;
        op.policy = parse_policy(leader->recompute_policy);
        op.lora_targets = leader->lora_targets;
        ops.push_back(std::move(op));
    }

    for (const auto* slot : singles) {
        if (slot->recompute_op.empty()) {
            throw std::runtime_error("DSL recompute: slot missing recompute_op: " + slot->name);
        }
        RecomputeOp op;
        op.op_type = slot->recompute_op;
        op.inputs.reserve(slot->recompute_from.size());
        for (const auto& dep : slot->recompute_from) {
            op.inputs.push_back(canonicalize_ref(alias_map, dep));
        }
        std::vector<std::string> outputs;
        if (!slot->recompute_outputs.empty()) {
            outputs = slot->recompute_outputs;
        } else {
            outputs.push_back(slot->name);
        }
        for (auto& out : outputs) {
            out = canonicalize_name(alias_map, out);
        }
        outputs.erase(std::remove_if(outputs.begin(), outputs.end(),
                                     [&](const std::string& name) {
                                         return enabled_slots.find(name) == enabled_slots.end();
                                     }),
                      outputs.end());
        if (outputs.empty()) {
            continue;
        }
        if (!slot->recompute_outputs.empty()) {
            const std::string canon = canonicalize_name(alias_map, slot->name);
            if (std::find(outputs.begin(), outputs.end(), canon) == outputs.end()) {
                throw std::runtime_error("DSL recompute: slot outputs missing itself: " + canon);
            }
        }
        op.outputs = std::move(outputs);
        op.attrs = slot->recompute_attrs;
        op.policy = parse_policy(slot->recompute_policy);
        op.lora_targets = slot->lora_targets;
        ops.push_back(std::move(op));
    }

    if (ops.empty()) {
        return;
    }

    std::unordered_map<std::string, std::size_t> producer;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        for (const auto& out : ops[i].outputs) {
            if (producer.count(out)) {
                throw std::runtime_error("DSL recompute: duplicate producer for output " + out);
            }
            producer[out] = i;
        }
    }

    std::vector<std::vector<std::size_t>> adj(ops.size());
    std::vector<int> indegree(ops.size(), 0);

    for (std::size_t i = 0; i < ops.size(); ++i) {
        std::unordered_set<std::size_t> deps;
        for (const auto& raw : ops[i].inputs) {
            std::string stripped;
            is_optional_ref(raw, stripped);
            if (starts_with(stripped, "@")) {
                continue;
            }
            auto it = producer.find(stripped);
            if (it != producer.end() && it->second != i) {
                deps.insert(it->second);
            }
        }
        indegree[i] = static_cast<int>(deps.size());
        for (auto dep : deps) {
            adj[dep].push_back(i);
        }
    }

    std::deque<std::size_t> queue;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (indegree[i] == 0) {
            queue.push_back(i);
        }
    }

    std::vector<RecomputeOp> ordered;
    ordered.reserve(ops.size());
    while (!queue.empty()) {
        std::size_t idx = queue.front();
        queue.pop_front();
        ordered.push_back(ops[idx]);
        for (auto succ : adj[idx]) {
            if (--indegree[succ] == 0) {
                queue.push_back(succ);
            }
        }
    }

    if (ordered.size() != ops.size()) {
        throw std::runtime_error("DSL recompute: cycle detected in recompute graph");
    }

    mPlan.topo_ops = std::move(ordered);
    mPlan.producer_index.clear();
    for (std::size_t i = 0; i < mPlan.topo_ops.size(); ++i) {
        for (const auto& out : mPlan.topo_ops[i].outputs) {
            mPlan.producer_index[out] = i;
            mDependencies[out] = mPlan.topo_ops[i].inputs;
        }
    }

}

bool RecomputePlan::can_recompute(const std::string& name) const {
    return mPlan.producer_index.find(name) != mPlan.producer_index.end();
}

const std::vector<std::string>& RecomputePlan::get_dependencies(const std::string& name) const {
    static const std::vector<std::string> kEmpty;
    auto it = mDependencies.find(name);
    if (it == mDependencies.end()) {
        return kEmpty;
    }
    return it->second;
}

/**
 * @brief Execute the recomputation plan for a single transformer layer.
 *
 * Replays a precomputed, topologically-sorted list of DSL ops (@c mPlan.topo_ops) to
 * regenerate intermediate activations needed during backward (or other phases),
 * optionally restricting execution to LoRA-related ops.
 *
 * High-level behavior:
 * - Validates that the recompute plan is non-empty and that @p layer_idx is in range.
 * - If residual offloading is enabled, scans op inputs to detect whether a residual
 *   tensor (e.g., "res_ffn"/"residual_ffn") is required, and fetches it on a side stream.
 * - Iterates through all ops, applying per-op recompute policies:
 *   - @c Never: skip
 *   - @c LoraOnly: execute only when @p lora_only_mode is true
 * - Resolves op inputs/outputs, dispatches to the appropriate kernel implementation
 * - Frees temporary tensors allocated during recompute
 */
void RecomputePlan::execute_layer(GraphExecutor& executor,
                                  int layer_idx,
                                  long B,
                                  long T,
                                  bool lora_only_mode,
                                  cudaStream_t stream) {
    if (mPlan.topo_ops.empty()) {
        throw std::runtime_error("DSL recompute plan is empty");
    }
    if (layer_idx < 0 || layer_idx >= executor.mConfig.NumLayers) {
        return;
    }

    if (executor.mRunState.has_residual_offloading()) {
        bool needs_residual = false;
        bool needs_prev_residual = false;
        bool has_ln1_fused_group = false;
        for (const auto& op : mPlan.topo_ops) {
            // Check for ln1_fused group (res_ffn output) - these need prev residual in FFT mode
            for (const auto& out : op.outputs) {
                if (out == "res_ffn" || out == "residual_ffn") {
                    has_ln1_fused_group = true;
                    break;
                }
            }
            for (const auto& raw : op.inputs) {
                std::string stripped;
                is_optional_ref(raw, stripped);
                if (stripped == "res_ffn" || stripped == "residual_ffn") {
                    needs_residual = true;
                }
                std::string input_name;
                if (is_input_ref(stripped, input_name) && input_name == "residual_in") {
                    needs_prev_residual = true;
                }
            }
        }
        // FFT mode and LoRA mode both use get_residual(layer_idx) for res_ffn
        if (has_ln1_fused_group && !lora_only_mode) {
            executor.mRunState.fetch_residual(layer_idx, executor.mRunState.side_stream());
        }
        if (needs_residual && lora_only_mode) {
            executor.mRunState.fetch_residual(layer_idx, executor.mRunState.side_stream());
        }
        if (needs_prev_residual && layer_idx > 0) {
            executor.mRunState.fetch_residual(layer_idx - 1, executor.mRunState.side_stream());
        }
    }

    RecomputeContext ctx{
        executor.mRunState,
        executor.mWeights,
        executor.mConfig,
        executor.mOptions,
        executor.mLoRAConfig,
        executor.mLoRAWeights,
        executor.mLoRARunState,
        &executor.mSaved,
        executor.forward_plan(layer_idx),
        [&](const std::string& name, Tensor& weight, cudaStream_t s) {
            return executor.get_fp8_cached_weight(name, weight, s);
        },
        [&](const std::string& name, Tensor& weight, cudaStream_t s) {
            return executor.get_fp4_cached_weight(name, weight, s);
        },
        lora_only_mode,
        layer_idx
    };

    RecomputeScratch scratch;

    for (const auto& op : mPlan.topo_ops) {
        if (op.policy == RecomputePolicy::Never) {
            continue;
        }
        if (op.policy == RecomputePolicy::FftOnly && lora_only_mode) {
            // Avoid recomputing attention outputs in LoRA mode.
            // LoRA O-proj backward expects the original forward attention activations.
            if (op.op_type == "flash_attention") {
                continue;
            }
        }
        // In FFT mode, we MUST recompute all activations for each micro-step.
        // The saved tensors contain micro-step 0's activations, which are WRONG for subsequent
        // micro-steps during gradient accumulation. Skipping recompute causes flash attention
        // backward to receive stale activations (from micro-step 0) with correct d_out (from
        // current micro-step), resulting in gradient explosion.
        //
        // The original logic skipped lora_only ops in FFT mode to "use saved tensors instead",
        // but this only works with gradient_accumulation_steps=1.
        //
        // if (op.policy == RecomputePolicy::LoraOnly && !lora_only_mode) {
        //     continue;  // WRONG: causes gradient explosion with grad accum > 1
        // }
        // In LoRA mode, only skip recompute for non-deterministic ops that must reuse
        // the original forward activations (e.g., flash_attention outputs for O-proj).
        if (lora_only_mode && !op.lora_targets.empty() && op.op_type == "flash_attention") {
            continue;
        }

        const InputMap inputs = resolve_inputs(ctx, op, layer_idx, B, T, scratch);
        const auto outputs = resolve_outputs(ctx, op, layer_idx, stream);

        if (op.op_type == "fused_residual_rmsnorm_apply_saved") {
            execute_fused_residual(ctx, op, layer_idx, B, T, inputs, outputs);
        } else if (op.op_type == "rmsnorm_apply_saved") {
            execute_rmsnorm(ctx, op, inputs, outputs, layer_idx, B, T);
        } else if (op.op_type == "matmul") {
            execute_matmul(ctx, op, layer_idx, B, T, inputs, outputs);
        } else if (op.op_type == "moe_router_probs") {
            execute_moe_router_probs(ctx, inputs, outputs);
        } else if (op.op_type == "moe_topk") {
            execute_moe_topk(ctx, op, inputs, outputs);
        } else if (op.op_type == "moe_permute") {
            execute_moe_permute(ctx, scratch, op, inputs, outputs);
        } else if (op.op_type == "moe_grouped_gemm_gate_up") {
            execute_moe_grouped_gemm_gate_up(ctx, scratch, inputs, outputs);
        } else if (op.op_type == "moe_grouped_gemm_down") {
            execute_moe_grouped_gemm_down(ctx, scratch, inputs, outputs);
        } else if (op.op_type == "moe_unpermute") {
            execute_moe_unpermute(ctx, op, inputs, outputs);
        } else if (op.op_type == "qkv_qk_norm_rope") {
            execute_qkv_qk_norm_rope(ctx, op, layer_idx, B, T, inputs, outputs);
        } else if (op.op_type == "rope") {
            execute_rope(ctx, op, layer_idx, B, T, inputs, outputs);
        } else if (op.op_type == "flash_attention") {
            execute_flash_attention(ctx, B, T, inputs, outputs);
        } else if (op.op_type == "swiglu") {
            execute_swiglu(ctx, op, B, T, inputs, outputs);
        } else {
            throw std::runtime_error("DSL recompute: unsupported op " + op.op_type);
        }
    }

    // NOTE: We do NOT free scratch.temps here. The zero_residual and other scratch
    // allocations will remain on the stack until the step ends. Trying to free them
    // here would violate LIFO order since temp_acquire may have allocated activations
    // on top of scratch allocations during the op execution loop.
}

}  // namespace dsl
