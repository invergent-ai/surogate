// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL parameter store implementation.

#include "runtime/dsl/dsl_param_store.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "runtime/dsl/ir.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/training/runtime_options.h"
#include "runtime/training/model.h"
#include "runtime/lora/lora_config.h"
#include "utilities/dtype.h"

namespace dsl {
namespace {

bool is_rope_param(const std::string& name) {
    return name.find("rope_freqs") != std::string::npos;
}

void augment_shape_env(ShapeEnv& env, const AttrMap& config) {
    auto get_long = [&](std::string_view key) -> std::optional<long> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::int64_t>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        if (auto v = std::get_if<double>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        return std::nullopt;
    };
    auto get_string = [&](std::string_view key) -> std::optional<std::string> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::string>(&it->second.value)) {
            return *v;
        }
        return std::nullopt;
    };

    auto d_model = get_long("d_model");
    if (!d_model) {
        d_model = get_long("hidden_size");
    }
    auto num_q = get_long("num_query_heads");
    if (!num_q) {
        num_q = get_long("num_attention_heads");
    }
    auto num_kv = get_long("num_kv_heads");
    if (!num_kv) {
        num_kv = get_long("num_key_value_heads");
    }
    auto head_size = get_long("head_size");
    if (!head_size) {
        head_size = get_long("head_dim");
    }
    auto d_ff = get_long("d_ff");
    if (!d_ff) {
        d_ff = get_long("intermediate_size");
    }
    auto mlp_activation = get_string("mlp_activation");
    if (!mlp_activation) mlp_activation = get_string("mlp_hidden_act");
    if (!mlp_activation) mlp_activation = get_string("activation");
    int up_factor = 2;
    if (mlp_activation) {
        std::string act = *mlp_activation;
        std::transform(act.begin(), act.end(), act.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (act == "swiglu" || act == "geglu") {
            up_factor = 2;
        } else if (act == "relu" || act == "relu2" || act == "gelu" || act == "gelu_new" ||
                   act == "gelu_fast" || act == "silu" || act == "swish") {
            up_factor = 1;
        }
    }
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }
    auto max_seq = get_long("max_seq");
    if (!max_seq) {
        max_seq = get_long("max_position_embeddings");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
    }
    if (max_seq) {
        env.values.emplace("MaxSeq", *max_seq);
    }
    if (num_q) {
        env.values.emplace("Hq", *num_q);
    }
    if (num_kv) {
        env.values.emplace("Hkv", *num_kv);
    } else if (num_q) {
        env.values.emplace("Hkv", *num_q);
    }
    long Hq = env.values.count("Hq") ? env.values.at("Hq") : 0;
    long Hkv = env.values.count("Hkv") ? env.values.at("Hkv") : 0;
    long C = env.values.count("C") ? env.values.at("C") : 0;
    if (!head_size && Hq > 0 && C > 0) {
        head_size = C / Hq;
    }
    if (head_size) {
        env.values.emplace("D", *head_size);
    }
    if (d_ff) {
        env.values.emplace("M", *d_ff);
        env.values.emplace("MUp", up_factor * (*d_ff));
    }
    if (vocab) {
        env.values.emplace("V", *vocab);
    }
    if (Hq > 0 && head_size) {
        env.values.emplace("AttnDim", Hq * (*head_size));
    }
    if (head_size && Hq > 0 && Hkv > 0) {
        env.values.emplace("QKV", (Hq + 2 * Hkv) * (*head_size));
    }

    // MoE dimensions
    auto num_experts = get_long("num_experts");
    auto num_experts_per_tok = get_long("num_experts_per_tok");
    if (!num_experts_per_tok) num_experts_per_tok = get_long("num_selected_experts");
    auto shared_expert_intermediate = get_long("shared_expert_intermediate");
    if (!shared_expert_intermediate) shared_expert_intermediate = get_long("shared_expert_intermediate_size");

    if (num_experts) {
        env.values.emplace("E", *num_experts);
    }
    if (num_experts_per_tok) {
        env.values.emplace("K", *num_experts_per_tok);
    }
    if (shared_expert_intermediate && *shared_expert_intermediate > 0) {
        env.values.emplace("SharedM", *shared_expert_intermediate);
        env.values.emplace("SharedMUp", up_factor * (*shared_expert_intermediate));
    } else if (d_ff) {
        // Default shared expert size to regular intermediate size if not specified
        env.values.emplace("SharedM", *d_ff);
        env.values.emplace("SharedMUp", up_factor * (*d_ff));
    }
}

} // namespace

DslParamStore::DslParamStore(const Module& module,
                             const Graph& graph,
                             const RuntimeOptions& options,
                             const PretrainedConfig& config,
                             const std::shared_ptr<TensorAllocator>& allocator,
                             const modules::ModularLoRAConfig* lora_config,
                             const std::unordered_set<std::string>* external_params,
                             bool use_weight_manager)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslParamStore: allocator is null");
    }

    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

    const bool freeze_base = lora_config && lora_config->enabled();
    const bool train_router = freeze_base && lora_config->train_router;
    auto is_router_param = [&](const std::string& name) -> bool {
        return name.find("router") != std::string::npos;
    };

    if (external_params) {
        mExternalParams = *external_params;
    }

    mUsesWeightManager = use_weight_manager;

    for (const auto& kv : graph.params) {
        const std::string& name = kv.first;
        const TensorInfo& info = kv.second;

        if (is_rope_param(name)) {
            // RoPE frequencies are provided by the run state.
            continue;
        }

        ETensorDType dtype = info.dtype.value_or(config.DType);
        std::vector<long> shape = resolve_shape(info.shape, env);

        Entry entry;
        entry.external = mExternalParams.find(name) != mExternalParams.end();
        entry.managed_by_weight_manager = (!entry.external && mUsesWeightManager);
        if (entry.external || entry.managed_by_weight_manager) {
            entry.tensor = Tensor::empty(dtype, shape);
        } else {
            entry.tensor = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
        }
        entry.trainable = !is_rope_param(name);
        if (freeze_base) {
            entry.trainable = train_router && is_router_param(name);
        }

        mParams.emplace(name, entry);
        mParamOrder.push_back(name);
    }

    // Deterministic ordering for optimizer updates/checkpointing.
    std::sort(mParamOrder.begin(), mParamOrder.end());
}

Tensor& DslParamStore::get(const std::string& name) {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        std::cerr << "[ERROR] DslParamStore::get: parameter '" << name << "' not found. Available params: ";
        for (auto& p : mParams) {
            if (p.first.find("mlp_up") != std::string::npos || p.first.find("experts") != std::string::npos) {
                std::cerr << p.first << ", ";
            }
        }
        std::cerr << std::endl;
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    if (it->second.external) {
        if (!mQLoRAProvider) {
            throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
        }
        return mQLoRAProvider->resolve_param(name, mDefaultStream);
    }
    if (it->second.managed_by_weight_manager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for parameter " + name);
        }
        return mWeightManager->get(name);
    }
    return it->second.tensor;
}

const Tensor& DslParamStore::get(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    if (it->second.external) {
        if (!mQLoRAProvider) {
            throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
        }
        return mQLoRAProvider->resolve_param(name, mDefaultStream);
    }
    if (it->second.managed_by_weight_manager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for parameter " + name);
        }
        return mWeightManager->get(name);
    }
    return it->second.tensor;
}

bool DslParamStore::has(const std::string& name) const {
    return mParams.find(name) != mParams.end();
}

bool DslParamStore::is_trainable(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return false;
    return it->second.trainable;
}

bool DslParamStore::is_external(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return false;
    return it->second.external;
}

const Tensor& DslParamStore::template_tensor(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    return it->second.tensor;
}

void DslParamStore::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    if (mUsesWeightManager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for iterate_tensors");
        }
        mWeightManager->iterate_tensors(callback);
        return;
    }
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        if (it->second.external) {
            if (!mQLoRAProvider) {
                throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
            }
            callback(name, TensorShard(mQLoRAProvider->resolve_param(name, mDefaultStream)));
        } else {
            callback(name, TensorShard(it->second.tensor));
        }
    }
}

} // namespace dsl
