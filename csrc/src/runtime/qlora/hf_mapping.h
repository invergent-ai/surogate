// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Lightweight HuggingFace mapping spec for QLoRA weight loading.

#ifndef SUROGATE_SRC_MODULES_QLORA_HF_MAPPING_H
#define SUROGATE_SRC_MODULES_QLORA_HF_MAPPING_H

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace modules {

struct HfMappingSpec {
    enum class Kind { Direct, Fuse, Split, Transform, TiedTo, StackExperts, Unknown };
    Kind kind = Kind::Unknown;
    std::string source;
    std::vector<std::string> sources;
    std::vector<std::pair<long, long>> ranges;
    std::string fn;
    std::string target;
    int dim = 0;
    bool optional = false;
    bool fuse_gate_up = false;
    int num_experts = 0;
};

struct HfMapping {
    std::unordered_map<std::string, HfMappingSpec> mapping;

    static std::string format_name(std::string templ, int layer_idx, int expert_idx = -1) {
        auto replace_all = [](std::string& str, std::string_view from, std::string_view to) {
            if (from.empty()) return;
            std::size_t start = 0;
            while ((start = str.find(from, start)) != std::string::npos) {
                str.replace(start, from.size(), to);
                start += to.size();
            }
        };

        if (templ.find("{layer}") != std::string::npos) {
            if (layer_idx < 0) {
                throw std::runtime_error("HF mapping uses {layer} but no layer index available");
            }
            replace_all(templ, "{layer}", std::to_string(layer_idx));
        }
        if (templ.find("{expert}") != std::string::npos) {
            if (expert_idx < 0) {
                throw std::runtime_error("HF mapping uses {expert} but no expert index available");
            }
            replace_all(templ, "{expert}", std::to_string(expert_idx));
        }
        return templ;
    }

    const HfMappingSpec* find(std::string_view internal_name, int& layer_idx) const {
        auto trim_optional = [](std::string_view name) -> std::string_view {
            if (!name.empty() && name.back() == '?') {
                return name.substr(0, name.size() - 1);
            }
            return name;
        };
        auto parse_block_param = [](std::string_view name, int& layer_idx_out, std::string& param_name) {
            auto dot = name.find('.');
            if (dot == std::string_view::npos) return false;
            auto prefix = name.substr(0, dot);
            auto rest = name.substr(dot + 1);

            if (prefix.find("blocks[") == 0) {
                auto close = prefix.find(']');
                if (close == std::string_view::npos) return false;
                auto idx_str = prefix.substr(7, close - 7);
                try {
                    layer_idx_out = std::stoi(std::string(idx_str));
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
                    layer_idx_out = std::stoi(std::string(idx_str.substr(0, dot2)));
                } catch (...) {
                    return false;
                }
                param_name = std::string(idx_str.substr(dot2 + 1));
                return true;
            }

            return false;
        };

        layer_idx = -1;
        const std::string clean(trim_optional(internal_name));
        auto it = mapping.find(clean);
        if (it != mapping.end()) {
            return &it->second;
        }

        std::string base;
        if (parse_block_param(clean, layer_idx, base)) {
            const std::string placeholder = std::string("blocks[{layer}].") + base;
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
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_QLORA_HF_MAPPING_H
