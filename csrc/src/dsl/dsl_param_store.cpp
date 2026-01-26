// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL parameter store implementation.

#include "dsl/dsl_param_store.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "dsl/ir.h"
#include "training/runtime_options.h"
#include "training/model.h"
#include "modules/lora/lora_config.h"
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
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
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
        env.values.emplace("MUp", 2 * (*d_ff));
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
}

} // namespace

DslParamStore::DslParamStore(const Module& module,
                             const Graph& graph,
                             const RuntimeOptions& options,
                             const PretrainedConfig& config,
                             const std::shared_ptr<TensorAllocator>& allocator,
                             const modules::ModularLoRAConfig* lora_config)
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
        entry.tensor = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
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
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    return it->second.tensor;
}

const Tensor& DslParamStore::get(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
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

void DslParamStore::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        callback(name, TensorShard(it->second.tensor));
    }
}

} // namespace dsl
