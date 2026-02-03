// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL-driven weight mapping implementation.

#include "dsl/weight_mapping.h"

#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dsl/weight_mapping_base.h"

namespace dsl {
namespace {

struct WeightSpec {
    enum class Kind { Direct, Fuse, Transform, TiedTo, Split, StackExperts, Unknown };
    Kind kind = Kind::Unknown;
    std::string source;
    std::vector<std::string> sources;
    std::string fn;
    int dim = 0;
    bool optional = false;
    bool fuse_gate_up = false;  // For StackExperts: fuse gate+up into gate_up format
    int num_experts = 0;        // For StackExperts: number of experts (0 = auto)
};

struct WeightEntry {
    std::string internal_name;
    WeightSpec spec;
};

const AttrMap* as_map(const AttrValue& value) {
    if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&value.value)) {
        if (*map_ptr) {
            return map_ptr->get();
        }
    }
    return nullptr;
}

const AttrList* as_list(const AttrValue& value) {
    if (const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value)) {
        if (*list_ptr) {
            return list_ptr->get();
        }
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

WeightSpec parse_weight_spec(const AttrValue& value) {
    WeightSpec spec;
    if (auto direct = as_string(value)) {
        spec.kind = WeightSpec::Kind::Direct;
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

    if (type == "direct" || (!type.empty() && type == "tied_to") || type.empty()) {
        if (const auto* src_val = find_key(map, "source")) {
            if (auto src = as_string(*src_val)) {
                spec.source = *src;
            }
        }
        if (!spec.source.empty()) {
            spec.kind = (type == "tied_to") ? WeightSpec::Kind::TiedTo : WeightSpec::Kind::Direct;
            return spec;
        }
    }

    if (type == "fuse") {
        spec.kind = WeightSpec::Kind::Fuse;
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

    if (type == "transform") {
        spec.kind = WeightSpec::Kind::Transform;
        if (const auto* src_val = find_key(map, "source")) {
            if (auto src = as_string(*src_val)) {
                spec.source = *src;
            }
        }
        if (const auto* fn_val = find_key(map, "fn")) {
            if (auto fn = as_string(*fn_val)) {
                spec.fn = *fn;
            }
        }
        return spec;
    }

    if (type == "split") {
        spec.kind = WeightSpec::Kind::Split;
        if (const auto* dim_val = find_key(map, "dim")) {
            if (auto dim = as_int(*dim_val)) {
                spec.dim = static_cast<int>(*dim);
            }
        }
        if (const auto* list_val = find_key(map, "targets")) {
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

    if (type == "stack_experts") {
        spec.kind = WeightSpec::Kind::StackExperts;
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

std::vector<std::string> split_dotted(const std::string& name) {
    std::vector<std::string> parts;
    std::string current;
    for (char c : name) {
        if (c == '.') {
            if (!current.empty()) parts.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) parts.push_back(current);
    return parts;
}

std::string trim_optional(const std::string& name) {
    if (!name.empty() && name.back() == '?') {
        return name.substr(0, name.size() - 1);
    }
    return name;
}

std::optional<modules::TensorTarget> resolve_tensor_target(const std::string& internal_name) {
    const auto clean = trim_optional(internal_name);
    const auto parts = split_dotted(clean);
    if (parts.empty()) {
        return std::nullopt;
    }

    const std::string& last = parts.back();
    const std::string prev = (parts.size() >= 2) ? parts[parts.size() - 2] : std::string();

    if (last == "embedding" || last == "embeddings" || last == "embed_tokens") {
        return modules::TensorTarget::Embeddings;
    }
    if (last == "final_norm" || last == "norm" || last == "final_norm_weight") {
        return modules::TensorTarget::FinalNorm;
    }
    if (last == "lm_head" || last == "lm_head_weight") {
        return modules::TensorTarget::LMHead;
    }

    if (last == "ln1_weight" || (prev == "ln1" && last == "weight")) {
        return modules::TensorTarget::LN1Weight;
    }
    if (last == "ln2_weight" || (prev == "ln2" && last == "weight")) {
        return modules::TensorTarget::LN2Weight;
    }
    if (last == "qkv_weight" || (prev == "attention" && last == "qkv_weight")) {
        return modules::TensorTarget::QKVWeight;
    }
    if (last == "qkv_bias") {
        return modules::TensorTarget::QKVBias;
    }
    if (last == "out_weight" || last == "o_proj_weight" || (prev == "attention" && last == "out_weight")) {
        return modules::TensorTarget::OutWeight;
    }
    if (last == "q_norm_weight") {
        return modules::TensorTarget::QNormWeight;
    }
    if (last == "k_norm_weight") {
        return modules::TensorTarget::KNormWeight;
    }
    if (last == "mlp_up_weight" || (prev == "mlp" && last == "up_weight")) {
        return modules::TensorTarget::MLPUpWeight;
    }
    if (last == "mlp_down_weight" || (prev == "mlp" && last == "down_weight")) {
        return modules::TensorTarget::MLPDownWeight;
    }

    if (last == "router_weight" || (prev == "router" && last == "weight")) {
        return modules::TensorTarget::RouterGate;
    }
    if (last == "router_bias" || (prev == "router" && last == "bias")) {
        return modules::TensorTarget::RouterBias;
    }
    if (last == "experts_gate_up" || last == "gate_up_weight") {
        return modules::TensorTarget::ExpertsGateUp;
    }
    if (last == "experts_down" || last == "down_weight") {
        return modules::TensorTarget::ExpertsDown;
    }
    if (last == "gate_proj" && prev == "experts") {
        return modules::TensorTarget::ExpertGate;
    }
    if (last == "up_proj" && prev == "experts") {
        return modules::TensorTarget::ExpertUp;
    }
    if (last == "down_proj" && prev == "experts") {
        return modules::TensorTarget::ExpertDown;
    }
    if (last == "shared_expert_gate") {
        return modules::TensorTarget::SharedExpertGate;
    }
    if (last == "shared_expert_up") {
        return modules::TensorTarget::SharedExpertUp;
    }
    if (last == "shared_expert_down") {
        return modules::TensorTarget::SharedExpertDown;
    }

    if (last == "in_proj_weight") {
        return modules::TensorTarget::MambaInProjWeight;
    }
    if (last == "in_proj_bias") {
        return modules::TensorTarget::MambaInProjBias;
    }
    if (last == "out_proj_weight") {
        return modules::TensorTarget::MambaOutProjWeight;
    }
    if (last == "out_proj_bias") {
        return modules::TensorTarget::MambaOutProjBias;
    }
    if (last == "conv1d_weight") {
        return modules::TensorTarget::MambaConv1dWeight;
    }
    if (last == "conv1d_bias") {
        return modules::TensorTarget::MambaConv1dBias;
    }
    if (last == "A_log") {
        return modules::TensorTarget::MambaALog;
    }
    if (last == "D") {
        return modules::TensorTarget::MambaD;
    }
    if (last == "dt_bias") {
        return modules::TensorTarget::MambaDtBias;
    }
    if (last == "norm_weight") {
        return modules::TensorTarget::MambaNormWeight;
    }

    return std::nullopt;
}

modules::RangeComputeFn resolve_fuse_range(modules::TensorTarget target, const std::string& source_name) {
    using modules::ranges::qkv_k_bias;
    using modules::ranges::qkv_k_weight;
    using modules::ranges::qkv_q_bias;
    using modules::ranges::qkv_q_weight;
    using modules::ranges::qkv_v_bias;
    using modules::ranges::qkv_v_weight;
    using modules::ranges::mlp_gate_weight;
    using modules::ranges::mlp_up_weight;

    if (target == modules::TensorTarget::QKVWeight) {
        if (contains_ci(source_name, "q_proj")) return qkv_q_weight;
        if (contains_ci(source_name, "k_proj")) return qkv_k_weight;
        if (contains_ci(source_name, "v_proj")) return qkv_v_weight;
    }
    if (target == modules::TensorTarget::QKVBias) {
        if (contains_ci(source_name, "q_proj")) return qkv_q_bias;
        if (contains_ci(source_name, "k_proj")) return qkv_k_bias;
        if (contains_ci(source_name, "v_proj")) return qkv_v_bias;
    }
    if (target == modules::TensorTarget::MLPUpWeight) {
        if (contains_ci(source_name, "up_proj")) return mlp_up_weight;
        if (contains_ci(source_name, "gate_proj")) return mlp_gate_weight;
    }
    return nullptr;
}

class DslWeightMapping final : public modules::BaseWeightMapping {
public:
    explicit DslWeightMapping(std::vector<WeightEntry> entries)
        : mEntries(std::move(entries)) {}

    void register_patterns() override {
        for (const auto& entry : mEntries) {
            auto target = resolve_tensor_target(entry.internal_name);
            if (!target.has_value()) {
                throw std::runtime_error("DSL weight mapping: unknown internal target " + entry.internal_name);
            }

            const auto& spec = entry.spec;
            switch (spec.kind) {
                case WeightSpec::Kind::Direct:
                    if (spec.source.empty()) {
                        throw std::runtime_error("DSL weight mapping: missing source for " + entry.internal_name);
                    }
                    add_pattern_for_source(spec.source, *target, nullptr, spec.optional);
                    break;
                case WeightSpec::Kind::Fuse: {
                    if (spec.dim != 0) {
                        throw std::runtime_error("DSL weight mapping: fuse dim other than 0 is not supported yet");
                    }
                    if (spec.sources.empty()) {
                        throw std::runtime_error("DSL weight mapping: missing fuse sources for " + entry.internal_name);
                    }
                    for (const auto& src : spec.sources) {
                        auto range_fn = resolve_fuse_range(*target, src);
                        if (!range_fn) {
                            throw std::runtime_error("DSL weight mapping: unsupported fuse source " + src);
                        }
                        add_pattern_for_source(src, *target, range_fn, spec.optional);
                    }
                    break;
                }
                case WeightSpec::Kind::TiedTo:
                    // Tied weights do not require direct loading here.
                    break;
                case WeightSpec::Kind::Transform:
                    throw std::runtime_error("DSL weight mapping: transform mappings are not supported yet");
                case WeightSpec::Kind::Split:
                    // Split is export-only; ignore for import.
                    break;
                case WeightSpec::Kind::StackExperts: {
                    // stack_experts: loads per-expert HF tensors and stacks them into batched format.
                    // The pattern has {expert} placeholder which add_expert_pattern handles.
                    if (spec.source.empty()) {
                        throw std::runtime_error("DSL weight mapping: missing pattern for stack_experts " + entry.internal_name);
                    }
                    if (spec.fuse_gate_up) {
                        // For gate_up: register both gate_proj and up_proj patterns
                        // The pattern is for gate_proj; derive up_proj by replacing gate_proj with up_proj
                        std::string gate_pattern = spec.source;
                        std::string up_pattern = spec.source;
                        // Replace gate_proj with up_proj in the pattern
                        std::size_t pos = up_pattern.find("gate_proj");
                        if (pos != std::string::npos) {
                            up_pattern.replace(pos, 9, "up_proj");
                        }
                        add_expert_pattern(gate_pattern, modules::TensorTarget::ExpertGate, nullptr, spec.optional);
                        add_expert_pattern(up_pattern, modules::TensorTarget::ExpertUp, nullptr, spec.optional);
                    } else {
                        // For down_proj: single pattern
                        add_expert_pattern(spec.source, modules::TensorTarget::ExpertDown, nullptr, spec.optional);
                    }
                    break;
                }
                default:
                    throw std::runtime_error("DSL weight mapping: unsupported mapping spec for " + entry.internal_name);
            }
        }
    }

private:
    void add_pattern_for_source(const std::string& hf_name,
                                modules::TensorTarget target,
                                modules::RangeComputeFn range_fn,
                                bool optional) {
        if (hf_name.find("{expert}") != std::string::npos) {
            add_expert_pattern(hf_name, target, std::move(range_fn), optional);
        } else if (hf_name.find("{layer}") != std::string::npos) {
            add_layer_pattern(hf_name, target, std::move(range_fn), optional);
        } else {
            add_pattern(hf_name, target, std::move(range_fn), optional);
        }
    }

    std::vector<WeightEntry> mEntries;
};

} // namespace

std::unique_ptr<modules::BaseWeightMapping> build_weight_mapping(const Module& module) {
    if (module.hf_mapping.empty()) {
        return nullptr;
    }

    std::vector<WeightEntry> entries;
    entries.reserve(module.hf_mapping.size());
    for (const auto& kv : module.hf_mapping) {
        WeightSpec spec = parse_weight_spec(kv.second);
        entries.push_back({kv.first, std::move(spec)});
    }

    auto mapping = std::make_unique<DslWeightMapping>(std::move(entries));
    mapping->register_patterns();
    return mapping;
}

} // namespace dsl
