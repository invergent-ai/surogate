// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder DSL model wrapper (validation-only).

#include "dsl/dsl_model.h"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

#include "utilities/comm.h"

namespace dsl {
namespace {

struct HfValue {
    enum class Kind { Int, Float, Bool };
    Kind kind;
    long i = 0;
    double f = 0.0;
    bool b = false;
};

std::optional<HfValue> get_hf_value(const PretrainedConfig& cfg, const std::string& key) {
    if (key == "hidden_size") return HfValue{HfValue::Kind::Int, cfg.HiddenSize, 0.0, false};
    if (key == "intermediate_size") return HfValue{HfValue::Kind::Int, cfg.IntermediateSize, 0.0, false};
    if (key == "vocab_size") return HfValue{HfValue::Kind::Int, cfg.VocabSize, 0.0, false};
    if (key == "num_attention_heads") return HfValue{HfValue::Kind::Int, cfg.NumQueryHeads, 0.0, false};
    if (key == "num_key_value_heads") return HfValue{HfValue::Kind::Int, cfg.NumKeyValHeads, 0.0, false};
    if (key == "num_hidden_layers") return HfValue{HfValue::Kind::Int, cfg.NumLayers, 0.0, false};
    if (key == "max_position_embeddings") return HfValue{HfValue::Kind::Int, cfg.MaxPositionEmbeddings, 0.0, false};
    if (key == "head_dim") return HfValue{HfValue::Kind::Int, cfg.head_size(), 0.0, false};
    if (key == "attention_bias") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.UseQKVBias};
    if (key == "use_qk_norm") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.UseQKNorm};
    if (key == "rms_norm_eps") return HfValue{HfValue::Kind::Float, 0, cfg.RmsNormEps, false};
    if (key == "rope_theta") return HfValue{HfValue::Kind::Float, 0, cfg.RopeTheta, false};
    if (key == "tie_word_embeddings") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.TiedWordEmbeddings};
    return std::nullopt;
}

std::optional<HfValue> attr_to_value(const AttrValue& value) {
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return HfValue{HfValue::Kind::Int, static_cast<long>(*v), 0.0, false};
    }
    if (auto v = std::get_if<double>(&value.value)) {
        return HfValue{HfValue::Kind::Float, 0, *v, false};
    }
    if (auto v = std::get_if<bool>(&value.value)) {
        return HfValue{HfValue::Kind::Bool, 0, 0.0, *v};
    }
    return std::nullopt;
}

bool values_match(const HfValue& expected, const HfValue& actual) {
    if (expected.kind == HfValue::Kind::Bool || actual.kind == HfValue::Kind::Bool) {
        if (expected.kind != HfValue::Kind::Bool || actual.kind != HfValue::Kind::Bool) {
            return false;
        }
        return expected.b == actual.b;
    }
    const double lhs = (expected.kind == HfValue::Kind::Int) ? static_cast<double>(expected.i) : expected.f;
    const double rhs = (actual.kind == HfValue::Kind::Int) ? static_cast<double>(actual.i) : actual.f;
    return std::abs(lhs - rhs) <= 1e-6;
}

[[noreturn]] void throw_unimplemented(std::string_view name) {
    throw std::runtime_error(std::string("DSL model placeholder: ") + std::string(name) + " not implemented");
}

} // namespace

DslModel::DslModel(const PretrainedConfig& config,
                   const RuntimeOptions& /*options*/,
                   const std::string& ir_json,
                   const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config.clone()), mAllocator(allocator) {
    if (ir_json.empty()) {
        throw std::runtime_error("DSL model placeholder: IR JSON is empty");
    }
    nlohmann::json root = nlohmann::json::parse(ir_json);
    mIr = load_ir_from_json(root);
    if (!mIr.success) {
        throw std::runtime_error("DSL model placeholder: IR JSON indicates compilation failure");
    }
    mModule = &pick_model_module(mIr);
    validate_ir();
}

void DslModel::forward(Tensor, Tensor, NCCLCommunicator&, int) {
    throw_unimplemented("forward");
}

float DslModel::validate(Tensor, Tensor, Tensor, NCCLCommunicator&, int) {
    throw_unimplemented("validate");
}

void DslModel::backward(Tensor, Tensor, NCCLCommunicator&, int, int) {
    throw_unimplemented("backward");
}

void DslModel::update(NCCLCommunicator&, float, float, float, int, float, float, float) {
    throw_unimplemented("update");
}

void DslModel::init_weights(NCCLCommunicator&) {
    throw_unimplemented("init_weights");
}

void DslModel::import_weights(const std::string&, bool, NCCLCommunicator&) {
    throw_unimplemented("import_weights");
}

void DslModel::on_restore_checkpoint(NCCLCommunicator&) {
    throw_unimplemented("on_restore_checkpoint");
}

void DslModel::export_weights(const std::string&, NCCLCommunicator&) {
    throw_unimplemented("export_weights");
}

void DslModel::allocate_run_state(const RuntimeOptions&, NCCLCommunicator& comm, int B, int T, bool) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mRunState = std::make_unique<IRunState>(mConfig->clone(), B, T, mAllocator);
    mRunState->WorldSize = comm.world_size();
}

std::string_view DslModel::model_type() const {
    return mConfig ? mConfig->model_name() : "DSL";
}

IRunState& DslModel::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("DslModel::get_run_state() called before allocate_run_state()");
    }
    return *mRunState;
}

void DslModel::validate_ir() {
    if (!mModule) {
        throw std::runtime_error("DSL model placeholder: no module selected");
    }
    validate_config_mapping(*mModule);
    validate_param_shapes(*mModule);
}

const Module& DslModel::pick_model_module(const IRFile& ir) const {
    const Module* candidate = nullptr;
    for (const auto& mod : ir.modules) {
        if (mod.kind != "model") {
            continue;
        }
        if (candidate) {
            throw std::runtime_error("DSL model placeholder: multiple model modules in IR");
        }
        candidate = &mod;
    }
    if (!candidate) {
        throw std::runtime_error("DSL model placeholder: no model module in IR");
    }
    return *candidate;
}

void DslModel::validate_config_mapping(const Module& module) const {
    const AttrMap* mapping = nullptr;
    auto it = module.hf_config.find("param_mapping");
    if (it != module.hf_config.end()) {
        if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&it->second.value)) {
            if (*map_ptr) {
                mapping = map_ptr->get();
            }
        }
    }
    if (!mapping) {
        mapping = &module.hf_config;
    }

    for (const auto& kv : *mapping) {
        const auto* hf_key = std::get_if<std::string>(&kv.second.value);
        if (!hf_key) {
            continue;
        }
        auto expected = get_hf_value(*mConfig, *hf_key);
        if (!expected) {
            continue;
        }
        auto it = module.config.find(kv.first);
        if (it == module.config.end()) {
            throw std::runtime_error("DSL model placeholder: missing module config param " + kv.first);
        }
        auto actual = attr_to_value(it->second);
        if (!actual || !values_match(*expected, *actual)) {
            throw std::runtime_error("DSL model placeholder: config mismatch for param " + kv.first);
        }
    }
}

void DslModel::validate_param_shapes(const Module& module) const {
    auto env = make_shape_env(module, /*B=*/1, /*T=*/1);
    for (const auto& kv : module.params) {
        const auto& info = kv.second;
        if (info.shape.empty()) {
            continue;
        }
        auto resolved = resolve_shape(info.shape, env);
        for (const auto dim : resolved) {
            if (dim <= 0) {
                throw std::runtime_error("DSL model placeholder: invalid shape for param " + kv.first);
            }
        }
    }
}

} // namespace dsl
