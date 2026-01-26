// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#include "dsl/dsl_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "dsl/graph_executor.h"
#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/dsl_model_internal.h"
#include "kernels/kernels.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "modules/lora/lora_utils.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/model_config.h"
#include "modules/optimizers/adamw_8bit.h"
#include "modules/optimizers/normuon.h"
#include "modules/fp8_scaling_state.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/safetensors.h"

namespace dsl {

DslModel::DslModel(const PretrainedConfig& config,
                   const RuntimeOptions& options,
                   const std::string& ir_json,
                   const std::shared_ptr<TensorAllocator>& allocator,
                   const std::optional<modules::ModularLoRAConfig>& lora_config)
    : mConfig(config.clone()),
      mAllocator(allocator ? allocator : std::make_shared<TensorAllocator>()),
      mOptions(options),
      mModelConfig(modules::ModelConfig::from_pretrained_config(config)) {
    if (ir_json.empty()) {
        throw std::runtime_error("DSL model: IR JSON is empty");
    }
    nlohmann::json root = nlohmann::json::parse(ir_json);
    mIr = load_ir_from_json(root);
    if (!mIr.success) {
        throw std::runtime_error("DSL model: IR JSON indicates compilation failure");
    }
    mModule = &pick_model_module(mIr);
    validate_ir();

    if (!mModule->forward.has_value()) {
        throw std::runtime_error("DSL model: module missing forward graph");
    }

    mParams = std::make_unique<DslParamStore>(*mModule, mModule->forward.value(),
                                              options, *mConfig, mAllocator,
                                              lora_config ? &*lora_config : nullptr);
    mGrads = std::make_unique<DslGradStore>(*mParams, mAllocator);

    // Create weight manager for streaming/sharding if enabled
    if (options.ShardWeights || options.OffloadMaster) {
        mWeightManager = std::make_unique<DslWeightManager>(
            *mModule, mModule->forward.value(), options, *mConfig, mAllocator,
            lora_config ? &*lora_config : nullptr);
    }

    if (lora_config.has_value() && lora_config->enabled()) {
        mLoRAConfig = lora_config;
        mIsMoEModel = (mModelConfig.architecture == modules::ArchitectureType::MoE) || mModelConfig.moe_config.has_value();

        modules::ModularLoRAWeightsManager::Config wm{};
        wm.num_layers = mModelConfig.NumLayers;
        wm.hidden_size = mModelConfig.HiddenSize;
        wm.intermediate_size = mModelConfig.IntermediateSize;
        wm.num_query_heads = mModelConfig.NumQueryHeads;
        wm.num_kv_heads = mModelConfig.NumKeyValHeads;
        wm.head_size = mModelConfig.head_size();
        wm.lora_config = *mLoRAConfig;
        wm.work_dtype = mModelConfig.DType;
        wm.shard_idx = 0;
        wm.num_shards = 1;
        wm.is_moe = mIsMoEModel;
        if (mIsMoEModel && mModelConfig.moe_config.has_value()) {
            wm.num_experts = mModelConfig.moe_config->num_experts;
            wm.moe_intermediate_size = mModelConfig.moe_config->moe_intermediate_size > 0
                                        ? mModelConfig.moe_config->moe_intermediate_size
                                        : mModelConfig.IntermediateSize;
            wm.train_router = mLoRAConfig->train_router;
        }
        mLoRAWeights = std::make_unique<modules::ModularLoRAWeightsManager>(wm, *mAllocator);

        modules::ModularLoRAGradsManager::Config gm{};
        gm.num_layers = mModelConfig.NumLayers;
        gm.hidden_size = mModelConfig.HiddenSize;
        gm.intermediate_size = mModelConfig.IntermediateSize;
        gm.num_query_heads = mModelConfig.NumQueryHeads;
        gm.num_kv_heads = mModelConfig.NumKeyValHeads;
        gm.head_size = mModelConfig.head_size();
        gm.lora_config = *mLoRAConfig;
        gm.grad_dtype = mLoRAConfig->dtype;
        gm.shard_idx = 0;
        gm.num_shards = 1;
        gm.is_moe = mIsMoEModel;
        if (mIsMoEModel && mModelConfig.moe_config.has_value()) {
            gm.num_experts = mModelConfig.moe_config->num_experts;
            gm.moe_intermediate_size = mModelConfig.moe_config->moe_intermediate_size > 0
                                        ? mModelConfig.moe_config->moe_intermediate_size
                                        : mModelConfig.IntermediateSize;
            gm.train_router = mLoRAConfig->train_router;
        }
        mLoRAGrads = std::make_unique<modules::ModularLoRAGradsManager>(gm, mAllocator);
    }

    for (const auto& kv : mModule->hf_mapping) {
        mHfMapping.emplace(kv.first, internal::parse_mapping_spec(kv.second));
    }
    for (const auto& kv : mModule->hf_export) {
        mHfExport.emplace(kv.first, internal::parse_mapping_spec(kv.second));
    }
}

DslModel::~DslModel() = default;

modules::ModularLoRAWeightsManager& DslModel::lora_weights() {
    if (!mLoRAWeights) {
        throw std::runtime_error("DSL model: LoRA not enabled");
    }
    return *mLoRAWeights;
}

modules::ModularLoRAGradsManager& DslModel::lora_grads() {
    if (!mLoRAGrads) {
        throw std::runtime_error("DSL model: LoRA not enabled");
    }
    return *mLoRAGrads;
}

modules::LoRARunState& DslModel::lora_run_state() {
    if (!mLoRARunState) {
        throw std::runtime_error("DSL model: LoRA run state not allocated");
    }
    return *mLoRARunState;
}

void DslModel::validate_ir() {
    if (!mModule) {
        throw std::runtime_error("DSL model: no module selected");
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
            throw std::runtime_error("DSL model: multiple model modules in IR");
        }
        candidate = &mod;
    }
    if (!candidate) {
        throw std::runtime_error("DSL model: no model module in IR");
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
        auto expected = internal::get_hf_value(*mConfig, *hf_key);
        if (!expected) {
            continue;
        }
        auto it = module.config.find(kv.first);
        if (it == module.config.end()) {
            throw std::runtime_error("DSL model: missing module config param " + kv.first);
        }
        auto actual = internal::attr_to_value(it->second);
        if (!actual || !internal::values_match(*expected, *actual)) {
            throw std::runtime_error("DSL model: config mismatch for param " + kv.first);
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
                throw std::runtime_error("DSL model: invalid shape for param " + kv.first);
            }
        }
    }
}

} // namespace dsl
