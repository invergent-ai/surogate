// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Optional runtime gradient masks for deliberate non-LoRA partial-update runs.
// Unset environment variables leave optimizer behavior unchanged.

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_GRADIENT_MASK_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_GRADIENT_MASK_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace optimizers::gradient_mask {

struct Spec {
    bool enabled = false;
    bool mlp_growth_new_half = false;
    std::vector<std::string> trainable_substrings;
};

inline std::string lower(std::string_view value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

inline bool contains_ci(std::string_view haystack, std::string_view needle) {
    const std::string h = lower(haystack);
    const std::string n = lower(needle);
    return h.find(n) != std::string::npos;
}

inline std::vector<std::string> split_csv(const char* raw) {
    std::vector<std::string> out;
    if (!raw || raw[0] == '\0') {
        return out;
    }
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(item.begin(),
                   std::find_if(item.begin(), item.end(), [](unsigned char c) { return !std::isspace(c); }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char c) { return !std::isspace(c); }).base(),
                   item.end());
        if (!item.empty()) {
            out.push_back(lower(item));
        }
    }
    return out;
}

inline const Spec& spec() {
    static const Spec parsed = [] {
        Spec s;
        if (const char* allow = std::getenv("SUROGATE_TRAINABLE_PARAM_SUBSTRINGS")) {
            s.trainable_substrings = split_csv(allow);
            s.enabled = !s.trainable_substrings.empty();
        }

        const char* raw_mode = std::getenv("SUROGATE_GRADIENT_MASK_MODE");
        const std::string mode = raw_mode ? lower(raw_mode) : "";
        if (mode.empty() || mode == "none" || mode == "off" || mode == "0") {
            return s;
        }
        s.enabled = true;
        if (mode == "allow_substrings") {
            if (s.trainable_substrings.empty()) {
                throw std::runtime_error("SUROGATE_GRADIENT_MASK_MODE=allow_substrings requires "
                                         "SUROGATE_TRAINABLE_PARAM_SUBSTRINGS");
            }
            return s;
        }
        if (mode == "mlp_growth_new_half") {
            s.mlp_growth_new_half = true;
            return s;
        }
        throw std::runtime_error("Unknown SUROGATE_GRADIENT_MASK_MODE: " + mode);
    }();
    return parsed;
}

inline bool enabled() {
    return spec().enabled;
}

inline bool is_growth_mlp_up_or_gate(std::string_view name) {
    return (contains_ci(name, "mlp") || contains_ci(name, "mixer")) &&
           (contains_ci(name, "gate_proj") || contains_ci(name, "up_proj") || contains_ci(name, "mlp_gate_weight") ||
            contains_ci(name, "mlp_up_weight") || contains_ci(name, "gate_weight") || contains_ci(name, "up_weight"));
}

inline bool is_growth_mlp_down(std::string_view name) {
    return (contains_ci(name, "mlp") || contains_ci(name, "mixer")) &&
           (contains_ci(name, "down_proj") || contains_ci(name, "mlp_down_weight") || contains_ci(name, "down_weight"));
}

inline bool is_growth_mlp_param(std::string_view name) {
    return is_growth_mlp_up_or_gate(name) || is_growth_mlp_down(name);
}

inline bool whole_param_trainable(std::string_view name) {
    const Spec& s = spec();
    if (!s.enabled) {
        return true;
    }

    if (!s.trainable_substrings.empty()) {
        for (const std::string& needle : s.trainable_substrings) {
            if (contains_ci(name, needle)) {
                return true;
            }
        }
        return false;
    }

    if (s.mlp_growth_new_half) {
        return is_growth_mlp_param(name);
    }

    return true;
}

inline void apply_partial_mask(std::string_view name, Tensor& grad, cudaStream_t stream) {
    const Spec& s = spec();
    if (!s.mlp_growth_new_half || !is_growth_mlp_param(name)) {
        return;
    }
    if (grad.Rank != 2) {
        throw std::runtime_error("mlp_growth_new_half mask expected rank-2 gradient for " + std::string(name));
    }
    const long rows = grad.Sizes[0];
    const long cols = grad.Sizes[1];
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("mlp_growth_new_half mask got empty gradient for " + std::string(name));
    }
    if (is_growth_mlp_up_or_gate(name)) {
        if ((rows % 2) != 0) {
            throw std::runtime_error("mlp_growth_new_half expected even row count for " + std::string(name));
        }
        const std::size_t bytes =
            static_cast<std::size_t>(rows / 2) * static_cast<std::size_t>(cols) * get_dtype_size(grad.DType);
        CUDA_CHECK(cudaMemsetAsync(grad.Data, 0, bytes, stream));
        return;
    }
    if (is_growth_mlp_down(name)) {
        if ((cols % 2) != 0) {
            throw std::runtime_error("mlp_growth_new_half expected even column count for " + std::string(name));
        }
        zero_matrix_columns(grad, 0, cols / 2, stream);
    }
}

inline float weight_decay_for(std::string_view name, float configured_weight_decay) {
    const Spec& s = spec();
    if (s.mlp_growth_new_half && is_growth_mlp_param(name)) {
        // Weight decay would modify the frozen inherited half of a partially
        // masked tensor. Disable it for the whole tensor in this mode.
        return 0.0f;
    }
    return configured_weight_decay;
}

}  // namespace optimizers::gradient_mask

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_GRADIENT_MASK_H
