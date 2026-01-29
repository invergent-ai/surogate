// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model helper functions (config mapping, string utilities, HF mapping).

#include "dsl/dsl_model_internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "utilities/dtype.h"

namespace dsl {
namespace internal {

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

void wait_event_if_not_capturing(cudaStream_t stream, cudaEvent_t event) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
    }
}

void record_event_if_not_capturing(cudaEvent_t event, cudaStream_t stream) {
    if (!stream_is_capturing(stream)) {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }
}

LoRAAdamW8BitStateContainer::LoRAAdamW8BitStateContainer(modules::LoRAAdamW8BitState* state)
    : mState(state) {}

void LoRAAdamW8BitStateContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!mState) return;
    if (!mState->state1.Data) return;
    callback("lora_adamw8bit.state1", TensorShard(mState->state1));
    callback("lora_adamw8bit.state2", TensorShard(mState->state2));
    callback("lora_adamw8bit.absmax1", TensorShard(mState->absmax1));
    callback("lora_adamw8bit.absmax2", TensorShard(mState->absmax2));
}

LoRANorMuonStateContainer::LoRANorMuonStateContainer(modules::LoRANorMuonState* state)
    : mState(state) {}

void LoRANorMuonStateContainer::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!mState) return;
    if (!mState->momentum_state.Data) return;
    callback("lora_normuon.momentum_state", TensorShard(mState->momentum_state));
    callback("lora_normuon.momentum_absmax", TensorShard(mState->momentum_absmax));
    for (size_t i = 0; i < mState->variance_buffers.size(); ++i) {
        callback(fmt::format("lora_normuon.variance_{}", i), TensorShard(mState->variance_buffers[i]));
    }
}

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

std::string format_hf_name(std::string templ, int layer_idx, int expert_idx) {
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

    if (type == "stack_experts") {
        spec.kind = DslModel::MappingSpec::Kind::StackExperts;
        if (const auto* pattern_val = find_key(map, "pattern")) {
            if (auto pattern = as_string(*pattern_val)) {
                spec.source = *pattern;
            }
        }
        if (const auto* fuse_val = find_key(map, "fuse_gate_up")) {
            if (auto fuse = as_bool(*fuse_val)) {
                spec.fuse_gate_up = *fuse;
            }
        }
        if (const auto* num_val = find_key(map, "num_experts")) {
            if (auto num = as_int(*num_val)) {
                spec.num_experts = static_cast<int>(*num);
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

}  // namespace internal
}  // namespace dsl
