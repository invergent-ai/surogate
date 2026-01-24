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
namespace {

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && *value;
}

float env_float(const char* name, float fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    float out = std::strtof(value, &end);
    if (end == value) return fallback;
    return out;
}

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    long out = std::strtol(value, &end, 10);
    if (end == value) return fallback;
    return static_cast<int>(out);
}

bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

inline void wait_event_if_not_capturing(cudaStream_t stream, cudaEvent_t event) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
}

inline void record_event_if_not_capturing(cudaEvent_t event, cudaStream_t stream) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
}

struct HfValue {
    enum class Kind { Int, Float, Bool };
    Kind kind;
    long i = 0;
    double f = 0.0;
    bool b = false;
};

class LoRAAdamW8BitStateContainer final : public ITensorContainer {
public:
    explicit LoRAAdamW8BitStateContainer(modules::LoRAAdamW8BitState* state) : mState(state) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState) return;
        if (!mState->state1.Data) return;
        callback("lora_adamw8bit.state1", TensorShard(mState->state1));
        callback("lora_adamw8bit.state2", TensorShard(mState->state2));
        callback("lora_adamw8bit.absmax1", TensorShard(mState->absmax1));
        callback("lora_adamw8bit.absmax2", TensorShard(mState->absmax2));
    }

private:
    modules::LoRAAdamW8BitState* mState = nullptr;
};

class LoRANorMuonStateContainer final : public ITensorContainer {
public:
    explicit LoRANorMuonStateContainer(modules::LoRANorMuonState* state) : mState(state) {}

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override {
        if (!mState) return;
        if (!mState->momentum_state.Data) return;
        callback("lora_normuon.momentum_state", TensorShard(mState->momentum_state));
        callback("lora_normuon.momentum_absmax", TensorShard(mState->momentum_absmax));
        for (size_t i = 0; i < mState->variance_buffers.size(); ++i) {
            callback(fmt::format("lora_normuon.variance_{}", i), TensorShard(mState->variance_buffers[i]));
        }
    }

private:
    modules::LoRANorMuonState* mState = nullptr;
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

const AttrMap* as_map(const AttrValue& value) {
    if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&value.value)) {
        if (*map_ptr) return map_ptr->get();
    }
    return nullptr;
}

const AttrList* as_list(const AttrValue& value) {
    if (const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value)) {
        if (*list_ptr) return list_ptr->get();
    }
    return nullptr;
}

std::optional<std::string> as_string(const AttrValue& value) {
    if (const auto* str = std::get_if<std::string>(&value.value)) {
        return *str;
    }
    return std::nullopt;
}

std::optional<long> as_int(const AttrValue& value) {
    if (const auto* i64 = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<long>(*i64);
    }
    if (const auto* f64 = std::get_if<double>(&value.value)) {
        return static_cast<long>(*f64);
    }
    return std::nullopt;
}

std::optional<bool> as_bool(const AttrValue& value) {
    if (const auto* b = std::get_if<bool>(&value.value)) {
        return *b;
    }
    return std::nullopt;
}

const AttrValue* find_key(const AttrMap* map, const std::string& key) {
    if (!map) return nullptr;
    auto it = map->find(key);
    if (it == map->end()) return nullptr;
    return &it->second;
}

bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    if (prefix.find("blocks[") == 0) {
        auto close = prefix.find(']');
        if (close == std::string_view::npos) return false;
        auto idx_str = prefix.substr(7, close - 7);
        try {
            layer_idx = std::stoi(std::string(idx_str));
        } catch (...) {
            return false;
        }
        param_name = std::string(rest);
        return true;
    }

    if (prefix == "blocks") {
        auto idx_str = name.substr(dot + 1);
        auto dot2 = idx_str.find('.');
        if (dot2 == std::string_view::npos) return false;
        try {
            layer_idx = std::stoi(std::string(idx_str.substr(0, dot2)));
        } catch (...) {
            return false;
        }
        param_name = std::string(idx_str.substr(dot2 + 1));
        return true;
    }

    return false;
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
    auto h = to_lower(haystack);
    auto n = to_lower(needle);
    return h.find(n) != std::string::npos;
}

void replace_all(std::string& str, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::size_t start = 0;
    while ((start = str.find(from, start)) != std::string::npos) {
        str.replace(start, from.size(), to);
        start += to.size();
    }
}

std::string format_hf_name(std::string templ, int layer_idx, int expert_idx = -1) {
    if (templ.find("{layer}") != std::string::npos) {
        if (layer_idx < 0) {
            throw std::runtime_error("DSL model: HF mapping uses {layer} but no layer index available");
        }
        replace_all(templ, "{layer}", std::to_string(layer_idx));
    }
    if (templ.find("{expert}") != std::string::npos) {
        if (expert_idx < 0) {
            throw std::runtime_error("DSL model: HF mapping uses {expert} but no expert index available");
        }
        replace_all(templ, "{expert}", std::to_string(expert_idx));
    }
    return templ;
}

DslModel::MappingSpec parse_mapping_spec(const AttrValue& value) {
    DslModel::MappingSpec spec;

    if (auto direct = as_string(value)) {
        spec.kind = DslModel::MappingSpec::Kind::Direct;
        spec.source = *direct;
        return spec;
    }

    const AttrMap* map = as_map(value);
    if (!map) {
        return spec;
    }

    if (const auto* opt_val = find_key(map, "optional")) {
        if (auto opt = as_bool(*opt_val)) {
            spec.optional = *opt;
        }
    }

    std::string type;
    if (const auto* type_val = find_key(map, "type")) {
        if (auto t = as_string(*type_val)) {
            type = *t;
        }
    }

    auto get_source = [&]() -> std::string {
        if (const auto* src_val = find_key(map, "source")) {
            if (auto src = as_string(*src_val)) {
                return *src;
            }
        }
        if (const auto* path_val = find_key(map, "path")) {
            if (auto path = as_string(*path_val)) {
                return *path;
            }
        }
        return {};
    };

    if (type.empty() || type == "direct") {
        spec.kind = DslModel::MappingSpec::Kind::Direct;
        spec.source = get_source();
        return spec;
    }

    if (type == "fuse") {
        spec.kind = DslModel::MappingSpec::Kind::Fuse;
        if (const auto* dim_val = find_key(map, "dim")) {
            if (auto dim = as_int(*dim_val)) {
                spec.dim = static_cast<int>(*dim);
            }
        }
        if (const auto* list_val = find_key(map, "sources")) {
            if (const auto* list = as_list(*list_val)) {
                for (const auto& item : *list) {
                    if (auto src = as_string(item)) {
                        spec.sources.push_back(*src);
                    }
                }
            }
        }
        return spec;
    }

    if (type == "split") {
        spec.kind = DslModel::MappingSpec::Kind::Split;
        spec.source = get_source();
        if (const auto* dim_val = find_key(map, "dim")) {
            if (auto dim = as_int(*dim_val)) {
                spec.dim = static_cast<int>(*dim);
            }
        }
        if (const auto* list_val = find_key(map, "ranges")) {
            if (const auto* list = as_list(*list_val)) {
                for (const auto& item : *list) {
                    if (const auto* pair_list = as_list(item)) {
                        if (pair_list->size() >= 2) {
                            auto start = as_int(pair_list->at(0));
                            auto end = as_int(pair_list->at(1));
                            if (start && end) {
                                spec.ranges.emplace_back(*start, *end);
                            }
                        }
                    }
                }
            }
        }
        return spec;
    }

    if (type == "transform") {
        spec.kind = DslModel::MappingSpec::Kind::Transform;
        spec.source = get_source();
        if (const auto* fn_val = find_key(map, "fn")) {
            if (auto fn = as_string(*fn_val)) {
                spec.fn = *fn;
            }
        }
        return spec;
    }

    if (type == "tied_to") {
        spec.kind = DslModel::MappingSpec::Kind::TiedTo;
        if (const auto* tgt_val = find_key(map, "target")) {
            if (auto tgt = as_string(*tgt_val)) {
                spec.target = *tgt;
            }
        }
        return spec;
    }

    return spec;
}

const DslModel::MappingSpec* find_mapping_spec(
    const std::unordered_map<std::string, DslModel::MappingSpec>& mapping,
    const std::string& internal_name,
    int& layer_idx) {
    layer_idx = -1;
    auto it = mapping.find(internal_name);
    if (it != mapping.end()) {
        return &it->second;
    }

    std::string base;
    if (parse_block_param(internal_name, layer_idx, base)) {
        std::string placeholder = std::string("blocks[{layer}].") + base;
        it = mapping.find(placeholder);
        if (it != mapping.end()) {
            return &it->second;
        }
        it = mapping.find(base);
        if (it != mapping.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

Tensor slice_dim0(const Tensor& base, long offset, long length) {
    Tensor slice = base;
    if (slice.Rank < 1) {
        throw std::runtime_error("DSL model: cannot slice rank-0 tensor");
    }
    long stride = 1;
    for (int i = 1; i < slice.Rank; ++i) {
        stride *= slice.Sizes[i];
    }
    const std::size_t elem_size = get_dtype_size(slice.DType);
    const std::size_t byte_offset = static_cast<std::size_t>(offset) * static_cast<std::size_t>(stride) * elem_size;
    slice.Data = static_cast<std::byte*>(slice.Data) + byte_offset;
    slice.Sizes[0] = length;
    return slice;
}

bool is_norm_param_name(const std::string& name) {
    auto lower = to_lower(name);
    return lower.find("norm") != std::string::npos || lower.find("ln1") != std::string::npos || lower.find("ln2") != std::string::npos;
}

bool is_bias_param_name(const std::string& name) {
    return contains_ci(name, "bias");
}

std::vector<long> infer_fuse_slices(const std::string& name, const PretrainedConfig& cfg, int num_sources) {
    if (contains_ci(name, "qkv")) {
        const long hs = cfg.head_size();
        const long q_rows = static_cast<long>(cfg.NumQueryHeads) * hs;
        const long kv_rows = static_cast<long>(cfg.NumKeyValHeads) * hs;
        return {q_rows, kv_rows, kv_rows};
    }
    if (contains_ci(name, "mlp_up") || contains_ci(name, "gate_up")) {
        const long m = cfg.IntermediateSize;
        return std::vector<long>(num_sources, m);
    }
    return {};
}

} // namespace

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
        mHfMapping.emplace(kv.first, parse_mapping_spec(kv.second));
    }
    for (const auto& kv : mModule->hf_export) {
        mHfExport.emplace(kv.first, parse_mapping_spec(kv.second));
    }
}

DslModel::~DslModel() = default;

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        mExecutor->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    // Store micro_step for dropout seed computation (needed by backward pass)
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    auto hook = [this, micro_step](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            // seed = base_seed + layer_idx * 1000000 + proj_type * 100000 + micro_step * 10000
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                // Projection types: 0=Q, 1=K, 2=V, 3=O, 4=Up, 5=Gate, 6=Down
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(0), is_training,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(1), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(2), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(3), is_training,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(4), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(5), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(6), is_training,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };

    mExecutor->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto hook = [this](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                // Validation: no dropout (is_training=false)
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };

    return mExecutor->validate_with_hook(inputs, position_ids, targets, comm, micro_step, hook);
}

void DslModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, modules::BackwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const int micro_step = mLoRARunState->micro_step;

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::BackwardHookPoint::AfterMLPDownBackward: {
                if (!lora_block.mlp.down.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.mlp.down.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Projection type 6 = Down
                const unsigned int dropout_seed = get_dropout_seed(6);

                modules::detail::backward_lora_layer(
                    lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                    da.d_swiglu,
                    da.d_res_ffn, 0,
                    a.swiglu,
                    lora_block.mlp.down->A, lora_block.mlp.down->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, D, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterMLPUpBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Get ln2 input: either from stored activation or recompute from residual_att
                Tensor ln2_input;
                if (mOptions.RecomputeLoRA) {
                    // Recompute ln2 from residual_att
                    // TODO: Implement recomputation for DSL path
                    ln2_input = a.ln2;
                } else {
                    ln2_input = a.ln2;
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

                if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                    dA_up = lora_grads.mlp.up->A;
                    dB_up = lora_grads.mlp.up->B;
                    lora_up = *lora_block.mlp.up;
                }
                if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                    dA_gate = lora_grads.mlp.gate->A;
                    dB_gate = lora_grads.mlp.gate->B;
                    lora_gate = *lora_block.mlp.gate;
                }

                if (!dA_up.Data && !dA_gate.Data) break;

                // Projection types: 4=Up, 5=Gate
                modules::detail::backward_lora_mlp_up_gate_fused(
                    dA_up, dB_up,
                    dA_gate, dB_gate,
                    da.d_ln2,
                    da.d_mlp_up,
                    ln2_input,
                    lora_up, lora_gate,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(4), get_dropout_seed(5), is_training,
                    B * T,
                    C,
                    D,
                    rank,
                    lora_accum,
                    mLoRARunState->intermediate,
                    mLoRARunState->intermediate2,
                    mLoRARunState->slice,
                    rs.CublasLtHandle,
                    rs.CuBlasWorkspace,
                    stream);
            } break;
            case modules::BackwardHookPoint::AfterAttnOutBackward: {
                if (!lora_block.attention.o.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.attention.o.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Projection type 3 = O
                const unsigned int dropout_seed = get_dropout_seed(3);

                modules::detail::backward_lora_layer(
                    lora_grads.attention.o->A, lora_grads.attention.o->B,
                    da.d_att,
                    da.d_res_att, 0,
                    a.att,
                    lora_block.attention.o->A, lora_block.attention.o->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, Hq * Hs, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
            } break;
            case modules::BackwardHookPoint::AfterQKVBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);

                // Get ln1 input: either from stored activation or recompute from residual
                Tensor ln1_input;
                if (mOptions.RecomputeLoRA) {
                    // Recompute ln1 from the residual input
                    // TODO: Implement recomputation for DSL path
                    ln1_input = a.ln1;
                } else {
                    ln1_input = a.ln1;
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

                if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                    dA_q = lora_grads.attention.q->A;
                    dB_q = lora_grads.attention.q->B;
                    lora_q = *lora_block.attention.q;
                }
                if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                    dA_k = lora_grads.attention.k->A;
                    dB_k = lora_grads.attention.k->B;
                    lora_k = *lora_block.attention.k;
                }
                if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                    dA_v = lora_grads.attention.v->A;
                    dB_v = lora_grads.attention.v->B;
                    lora_v = *lora_block.attention.v;
                }

                if (!dA_q.Data && !dA_k.Data && !dA_v.Data) break;

                // Projection types: 0=Q, 1=K, 2=V
                modules::detail::backward_lora_qkv_fused(
                    dA_q, dB_q,
                    dA_k, dB_k,
                    dA_v, dB_v,
                    da.d_ln1,
                    da.d_qkv,
                    ln1_input,
                    lora_q, lora_k, lora_v,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                    B * T,
                    C,
                    Hq * Hs,
                    Hkv * Hs,
                    rank,
                    lora_accum,
                    mLoRARunState->intermediate,
                    mLoRARunState->intermediate2,
                    mLoRARunState->slice,
                    rs.CublasLtHandle,
                    rs.CuBlasWorkspace,
                    stream);

                mLoRAGrads->notify_block(layer_idx, stream, comm);
            } break;
            default:
                break;
        }
    };

    mExecutor->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);
    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

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

    if (!mAdamW8BitState->initialized) {
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    size_t state_offset = 0;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        float wd = weight_decay;
        if (is_norm_param_name(name) || is_bias_param_name(name)) {
            wd = 0.f;
        }

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad->Data) {
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
            if (grad->DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<float>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, stream
                );
            } else if (grad->DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<nv_bfloat16>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, stream
                );
            } else {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad->DType != ETensorDType::BF16) {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad->template get<nv_bfloat16>(),
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

    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    size_t state_offset = 0;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        const float wd_scale = (is_norm_param_name(name) || is_bias_param_name(name)) ? 0.f : 1.f;

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad->Data) {
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
            if (grad->DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<float>(),
                    s1, s2, n,
                    /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                    q1, q2, am1, am2, opt_params, opt_step, stream
                );
            } else if (grad->DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<nv_bfloat16>(),
                    s1, s2, n,
                    /*lr=*/0.f, /*beta1=*/0.f, /*beta2=*/0.f, /*step=*/1, /*eps=*/0.f, wd_scale, grad_scale,
                    q1, q2, am1, am2, opt_params, opt_step, stream
                );
            } else {
                throw std::runtime_error("DslModel::update_adamw_8bit_graph: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad->DType != ETensorDType::BF16) {
                throw std::runtime_error("DslModel::update_adamw_8bit_graph: unsupported grad dtype for " + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad->template get<nv_bfloat16>(),
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
        default:
            throw std::logic_error("DslModel::update_with_config: unsupported optimizer type");
    }
}

void DslModel::update_with_graph_params(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config,
                                        const float* opt_params, const int* opt_step) {
    if (!opt_params || !opt_step) {
        throw std::logic_error("DslModel::update_with_graph_params: missing optimizer parameter buffers");
    }
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT) {
        throw std::logic_error("DslModel::update_with_graph_params: unsupported optimizer type");
    }
    if (lora_enabled()) {
        update_lora_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
        return;
    }
    update_adamw_8bit_graph(comm, config.grad_clip, opt_params, opt_step);
}

void DslModel::prepare_optimizer_state_for_graph(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config) {
    if (config.type != optimizers::OptimizerType::ADAMW_8BIT) {
        return;
    }
    if (!mRunState) {
        throw std::logic_error("DslModel::prepare_optimizer_state_for_graph called before allocate_run_state()");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    bool did_work = false;

    if (lora_enabled()) {
        if (!mLoRAAdamW8BitState) {
            throw std::logic_error("DslModel::prepare_optimizer_state_for_graph: LoRA optimizer state not allocated");
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

    if (did_work) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

void DslModel::zero_grads(cudaStream_t stream) {
    if (mGrads) {
        mGrads->zero_all(stream);
    }
}

void DslModel::set_internal_graphs_enabled(bool enabled) {
    if (mExecutor) {
        mExecutor->set_internal_graphs_enabled(enabled);
    }
}

bool DslModel::internal_graphs_enabled() const {
    return mExecutor ? mExecutor->internal_graphs_enabled() : false;
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

modules::ModularLoRAWeightsManager& DslModel::lora_weights() {
    if (!mLoRAWeights) {
        throw std::logic_error("DslModel::lora_weights: LoRA weights not initialized");
    }
    return *mLoRAWeights;
}

modules::ModularLoRAGradsManager& DslModel::lora_grads() {
    if (!mLoRAGrads) {
        throw std::logic_error("DslModel::lora_grads: LoRA grads not initialized");
    }
    return *mLoRAGrads;
}

modules::LoRARunState& DslModel::lora_run_state() {
    if (!mLoRARunState) {
        throw std::logic_error("DslModel::lora_run_state: LoRA run state not initialized");
    }
    return *mLoRARunState;
}

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

void DslModel::save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) {
    if (!lora_enabled()) return;
    namespace fs = std::filesystem;

    export_adapter(checkpoint_dir, comm);

    if (mLoRAAdamW8BitState && mLoRAAdamW8BitState->initialized) {
        LoRAAdamW8BitStateContainer container(mLoRAAdamW8BitState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container);

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
        LoRANorMuonStateContainer container(mLoRANorMuonState.get());
        fs::path opt_file = fs::path(checkpoint_dir) / "lora_optimizer.safetensors";
        write_safetensors(opt_file.string(), container);

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

        LoRAAdamW8BitStateContainer container(&state);
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

        LoRANorMuonStateContainer container(&state);
        load_safetensors(opt_file.string(), container, /*allow_cast=*/false);
        state.values_restored = true;
    }

    comm.barrier();
}

std::vector<std::byte> DslModel::rng_state() const {
    if (mExecutor) {
        return mExecutor->rng_state();
    }
    return mRngState;
}

void DslModel::set_rng_state(const std::vector<std::byte>& state) {
    mRngState = state;
    if (mExecutor) {
        mExecutor->set_rng_state(state);
    }
}

void DslModel::init_weights(NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::init_weights called before parameters are initialized");
    }

    const float scale = 0.02f;
    const float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(mConfig->NumLayers));
    unsigned long long seed = 42ULL;
    unsigned long long subseq = 0ULL;

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        if (is_bias_param_name(name)) {
            fill_zero(param, nullptr);
            continue;
        }
        if (is_norm_param_name(name)) {
            fill_constant(param, 1.f, param.nelem(), nullptr);
            continue;
        }
        float stddev = scale;
        if (contains_ci(name, "out_weight") || contains_ci(name, "mlp_down_weight") || contains_ci(name, "down_proj")) {
            stddev *= residual_scale;
        }
        fill_normal(param, param.nelem(), 0.f, stddev, seed, subseq++, nullptr);
    }

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    comm.barrier();
}

void DslModel::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::import_weights called before parameters are initialized");
    }

    SafeTensorsReader reader(file_name);
    std::vector<std::pair<std::string, std::string>> tied_params;

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        int layer_idx = -1;
        const MappingSpec* spec = find_mapping_spec(mHfMapping, name, layer_idx);
        MappingSpec direct_fallback;
        if (!spec) {
            direct_fallback.kind = MappingSpec::Kind::Direct;
            direct_fallback.source = name;
            spec = &direct_fallback;
        }

        if (spec->kind == MappingSpec::Kind::TiedTo) {
            tied_params.emplace_back(name, spec->target);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            entry.read_tensor(param, allow_cast);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse mapping only supports dim=0 for " + name);
            }
            long offset = 0;
            for (const auto& src : spec->sources) {
                const std::string hf_name = format_hf_name(src, layer_idx);
                const auto& entry = reader.find_entry(hf_name);
                if (entry.shape().empty()) {
                    throw std::runtime_error("DSL model: empty shape for " + hf_name);
                }
                if (static_cast<int>(entry.shape().size()) != param.Rank) {
                    throw std::runtime_error("DSL model: rank mismatch for " + hf_name);
                }
                for (int i = 1; i < param.Rank; ++i) {
                    if (entry.shape().at(i) != param.Sizes[i]) {
                        throw std::runtime_error("DSL model: shape mismatch for " + hf_name);
                    }
                }
                const long slice_len = entry.shape().at(0);
                Tensor slice = slice_dim0(param, offset, slice_len);
                entry.read_raw(slice, 0, slice.nelem(), allow_cast);
                offset += slice_len;
            }
            if (offset != param.Sizes[0]) {
                throw std::runtime_error("DSL model: fuse mapping size mismatch for " + name);
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Split) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: split mapping only supports dim=0 for " + name);
            }
            if (spec->ranges.empty()) {
                throw std::runtime_error("DSL model: split mapping missing ranges for " + name);
            }
            auto [start, end] = spec->ranges.front();
            if (start < 0 || end <= start) {
                throw std::runtime_error("DSL model: unsupported split range for " + name);
            }
            const long expected = end - start;
            if (param.Sizes[0] != expected) {
                throw std::runtime_error("DSL model: split range size mismatch for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            long stride = 1;
            for (int i = 1; i < param.Rank; ++i) {
                stride *= param.Sizes[i];
            }
            const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(start) * stride;
            entry.read_raw(param, offset, param.nelem(), allow_cast);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported transform '" + spec->fn + "' for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            if (entry.shape().size() != 2 || param.Rank != 2) {
                throw std::runtime_error("DSL model: transpose expects 2D tensors for " + name);
            }
            Tensor tmp = mAllocator->allocate(param.DType, ("hf_tmp_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {entry.shape().at(0), entry.shape().at(1)});
            entry.read_tensor(tmp, allow_cast);
            cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
            transpose(param, tmp, static_cast<int>(entry.shape().at(0)),
                      static_cast<int>(entry.shape().at(1)), stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF mapping for " + name);
    }

    for (const auto& tie : tied_params) {
        Tensor& dst = mParams->get(tie.first);
        Tensor& src = mParams->get(tie.second);
        CUDA_CHECK(cudaMemcpy(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToDevice));
    }

    if (lora_enabled()) {
        mLoRAWeights->random_init(42, comm);
    }

    comm.barrier();
}

void DslModel::on_restore_checkpoint(NCCLCommunicator& comm) {
    (void)comm;
    if (mAdamW8BitState && mAdamW8BitState->state1.Data) {
        mAdamW8BitState->initialized = true;
        mAdamWMomentumContainer.update_pointers(&mAdamW8BitState->state1, &mAdamW8BitState->absmax1);
        mAdamWVarianceContainer.update_pointers(&mAdamW8BitState->state2, &mAdamW8BitState->absmax2);
    }
}

void DslModel::prepare_optimizer_for_checkpoint_load() {
    if (lora_enabled()) {
        return;
    }
    if (!mAdamW8BitState) {
        mAdamW8BitState = std::make_unique<AdamW8BitState>();
    }
    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }
}

void DslModel::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::export_weights called before parameters are initialized");
    }

    const auto& mapping = !mHfExport.empty() ? mHfExport : mHfMapping;
    SafeTensorWriter writer(file_name);

    struct ExportEntry {
        std::string name;
        Tensor tensor;
        bool needs_transpose = false;
        Tensor source;
    };

    std::vector<ExportEntry> exports;
    exports.reserve(mParams->param_names().size());

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        int layer_idx = -1;
        const MappingSpec* spec = find_mapping_spec(mapping, name, layer_idx);
        if (!spec) {
            MappingSpec fallback;
            fallback.kind = MappingSpec::Kind::Direct;
            fallback.source = name;
            spec = &fallback;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);
            exports.push_back({hf_name, param, false, {}});
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse export only supports dim=0 for " + name);
            }
            std::vector<long> slice_sizes = infer_fuse_slices(name, *mConfig, static_cast<int>(spec->sources.size()));
            if (slice_sizes.empty()) {
                if (param.Sizes[0] % static_cast<long>(spec->sources.size()) == 0) {
                    const long chunk = param.Sizes[0] / static_cast<long>(spec->sources.size());
                    slice_sizes.assign(spec->sources.size(), chunk);
                } else {
                    throw std::runtime_error("DSL model: cannot infer fuse slices for " + name);
                }
            } else if (slice_sizes.size() != spec->sources.size()) {
                throw std::runtime_error("DSL model: fuse slice count mismatch for " + name);
            }
            long offset = 0;
            for (std::size_t i = 0; i < spec->sources.size(); ++i) {
                const auto& src = spec->sources[i];
                const std::string hf_name = format_hf_name(src, layer_idx);
                const long slice_len = slice_sizes.at(i);
                if (slice_len <= 0) {
                    throw std::runtime_error("DSL model: invalid fuse slice for " + name);
                }
                Tensor slice = slice_dim0(param, offset, slice_len);
                exports.push_back({hf_name, slice, false, {}});
                offset += slice_len;
            }
            if (offset != param.Sizes[0]) {
                throw std::runtime_error("DSL model: fuse slices do not cover full tensor for " + name);
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported export transform '" + spec->fn + "' for " + name);
            }
            if (param.Rank != 2) {
                throw std::runtime_error("DSL model: transpose export expects 2D tensor for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            Tensor tmp = mAllocator->allocate(param.DType, ("export_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {param.Sizes[1], param.Sizes[0]});
            exports.push_back({hf_name, tmp, true, param});
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF export mapping for " + name);
    }

    for (const auto& entry : exports) {
        writer.register_tensor(entry.name, TensorShard(entry.tensor));
    }
    writer.prepare_metadata(&comm);

    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    for (auto& entry : exports) {
        if (entry.needs_transpose) {
            transpose(entry.tensor, entry.source,
                      static_cast<int>(entry.source.Sizes[0]),
                      static_cast<int>(entry.source.Sizes[1]),
                      stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        writer.write_tensor(entry.name, TensorShard(entry.tensor), &comm);
    }

    writer.finalize(&comm);
}

float DslModel::get_loss() const {
    if (!mRunState) {
        return 0.0f;
    }
    float raw_loss = mRunState->get_loss();
    int valid_tokens = 0;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    }
    return 0.0f;
}

float DslModel::get_accuracy() const {
    return IModel::get_accuracy();
}

void DslModel::allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T,
                                  bool allocate_optimizer) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mOptions = options;
    const std::size_t dummy_stack_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;  // 1TB dummy stack
    mRunState = std::make_unique<DslRunState>(*mConfig, options, B, T, mAllocator, lora_enabled(),
                                              dummy_stack_bytes, /*allocate_stack=*/false);
    mRunState->WorldSize = comm.world_size();

    const long base_size = static_cast<long>(mRunState->Stack.max_utilization());
    long moe_extra = 0;
    if (mModelConfig.NumExperts > 0) {
        const long moe_intermediate = (mModelConfig.MoeIntermediateSize > 0)
                                          ? mModelConfig.MoeIntermediateSize
                                          : mModelConfig.IntermediateSize;
        const long hidden = mModelConfig.HiddenSize;
        const long num_experts = mModelConfig.NumExperts;
        const long top_k = std::max(1, mModelConfig.NumExpertsPerTok);
        const long dtype_bytes = 2;  // BF16 bytes (matches modular sizing heuristic)
        const long up_factor = mModelConfig.mlp_up_factor();
        const long expert_gate_up_tp = num_experts * up_factor * moe_intermediate * hidden * dtype_bytes;
        const long expert_down_tp = num_experts * moe_intermediate * hidden * dtype_bytes;
        const long permuted_tokens = 2L * B * T * top_k * hidden * dtype_bytes;
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens;
    }
    ETensorDType act_dtype = options.ModelType.value_or(mConfig->DType);
    if (is_fp8_dtype(act_dtype)) {
        act_dtype = ETensorDType::BF16;
    }
    const long dtype_bytes = static_cast<long>(get_dtype_size(act_dtype));
    const long BT = static_cast<long>(B) * static_cast<long>(T);
    const long C = mModelConfig.HiddenSize;
    const long QKV = mModelConfig.head_size() * (mModelConfig.NumQueryHeads + 2 * mModelConfig.NumKeyValHeads);
    const long MUp = static_cast<long>(mModelConfig.mlp_up_rows());
    const long extra_tmp = std::max({BT * C, BT * QKV, BT * MUp}) * dtype_bytes;
    const long safety_bytes = std::max(64L * 1024 * 1024, base_size / 8);
    long required_size = std::max(1024L * 1024, base_size + base_size + moe_extra + safety_bytes + extra_tmp);
    required_size += 512L * 1024 * 1024;  // extra slack for unmodeled temps
    required_size = std::max(required_size, 3L * 1024 * 1024 * 1024);  // 3GB minimum for full fine-tune stability
    const auto high_mark = mRunState->Stack.get_high_mark();
    Tensor stack_buffer = mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {required_size});
    mRunState->set_stack_buffer(std::move(stack_buffer), high_mark);
    comm.barrier();

    // Configure gradient manager for multi-GPU overlapped reduction
    if (mGrads && comm.world_size() > 1) {
        DslGradStoreConfig grad_config;
        grad_config.num_shards = comm.world_size();
        grad_config.shard_idx = comm.rank();
        grad_config.shard_gradients = options.ShardGradients;  // ZeRO-2
        grad_config.use_all_to_all_reduce = options.UseAllToAllReduce;
        grad_config.num_layers = mModelConfig.NumLayers;
        mGrads->configure(grad_config);
    }

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    exec_opts.use_compiled_execution = options.UseCompiledDsl;
    mExecutor = std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, options, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
    }

    // Wire weight manager for streaming/sharding
    if (mWeightManager) {
        if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
            exec->set_weight_manager(mWeightManager.get());
        }
    }

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(), mLoRAGrads.get(), mLoRARunState.get());
    }

    if (allocate_optimizer) {
        if (lora_enabled()) {
            if (!mLoRAAdamW8BitState) {
                mLoRAAdamW8BitState = std::make_unique<modules::LoRAAdamW8BitState>();
            }
            if (!mLoRAAdamW8BitState->quantiles1.Data) {
                mLoRAAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
                mLoRAAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        } else {
            if (!mAdamW8BitState) {
                mAdamW8BitState = std::make_unique<AdamW8BitState>();
            }
            if (!mAdamW8BitState->quantiles1.Data) {
                mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
                mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }
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

bool DslModel::is_weight_streaming_enabled() const {
    return mWeightManager && mWeightManager->is_streaming_enabled();
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

    constexpr size_t BLOCK_SIZE = 2048;
    size_t total_params = 0;
    size_t state_elems = 0;
    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    for (const auto& name : mGrads->param_names()) {
        Tensor& param = mParams->get(name);
        add_tensor(param.nelem());
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", {(long)state.total_state_elems});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", {(long)state.total_state_elems});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", {(long)state.num_blocks});

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
    for (const auto& kv : mGrads->grads()) {
        const Tensor& grad = kv.second;
        if (!grad.Data || grad.nelem() == 0) continue;
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

void DslModel::allocate_lora_run_state(NCCLCommunicator& comm, int B, int T) {
    (void)comm;
    if (!lora_enabled()) return;

    mLoRARunState = std::make_unique<modules::LoRARunState>();
    mLoRARunState->B = B;
    mLoRARunState->T = T;

    auto ctx = mAllocator->with_context("DSL_LoRA_RunState");

    const int rank = mLoRAConfig->rank;
    const int BT = B * T;
    const int max_features = std::max(mModelConfig.HiddenSize, mModelConfig.IntermediateSize);
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

    if (mOptions.RecomputeLoRA) {
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

void DslModel::calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    if (!mLoRARunState || !mLoRAGrads) {
        throw std::logic_error("DslModel::calculate_lora_gradient_norm: LoRA state not initialized");
    }
    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    wait_event_if_not_capturing(stream, rs.BackwardDone);

    Tensor& buf = mLoRARunState->norm_buffer;
    fill_zero(buf, stream);

    auto norm_squared = [&](const Tensor& grad) {
        if (grad.Data) {
            global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
        }
    };

    for (int l = 0; l < mModelConfig.NumLayers; ++l) {
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

        if (g.router.has_value()) { norm_squared(g.router->A); norm_squared(g.router->B); }
    }

    deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem() - 2, stream);

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));

    const bool capturing = stream_is_capturing(stream);
    global_norm_sqrt(buf.template get<float>(), capturing ? nullptr : rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
    record_event_if_not_capturing(rs.NormDone, stream);
}

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
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        initialize_lora_multi_tensor_state(comm, stream);
    }
    if (!mLoRAAdamW8BitState->grad_ptrs_initialized) {
        if (stream_is_capturing(stream)) {
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

    record_event_if_not_capturing(rs.OptimizerDone, stream);
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
        if (stream_is_capturing(stream)) {
            throw std::runtime_error("DslModel::update_lora_adamw_8bit_graph: optimizer state must be initialized before capture");
        }
        initialize_lora_multi_tensor_state(comm, stream);
    }
    if (!mLoRAAdamW8BitState->grad_ptrs_initialized) {
        if (stream_is_capturing(stream)) {
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

    record_event_if_not_capturing(rs.OptimizerDone, stream);
}

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

    record_event_if_not_capturing(rs.OptimizerDone, main_stream);
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
        auto expected = get_hf_value(*mConfig, *hf_key);
        if (!expected) {
            continue;
        }
        auto it = module.config.find(kv.first);
        if (it == module.config.end()) {
            throw std::runtime_error("DSL model: missing module config param " + kv.first);
        }
        auto actual = attr_to_value(it->second);
        if (!actual || !values_match(*expected, *actual)) {
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
