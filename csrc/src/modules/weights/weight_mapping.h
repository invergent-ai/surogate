// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Weight Mapping Base Infrastructure
//
// This file defines the base weight mapping system that maps HuggingFace tensor names
// to internal tensor destinations with optional fusion. Model-specific mappings are
// defined in their respective model directories (e.g., models/llama/weight_mapping.h)
// and registered through the models registry (models/registry.h).
//
// The registry pattern enables:
// 1. Easy addition of new model architectures
// 2. Inherited weight patterns from parent models
// 3. Custom loaders for weight fusion/transformation

#ifndef SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_H
#define SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_H

#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/pretrained_config.h"
#include "utilities/tensor.h"

namespace modules {

// ============================================================================
// Weight Pattern Types
// ============================================================================

/**
 * @brief Describes where to load a tensor within a fused destination
 *
 * For direct loads: begin=0, end=total_elements
 * For fusion loads: begin/end specify the slice within the fused tensor
 */
struct LoadRange {
    std::ptrdiff_t begin = 0;   ///< Start offset in elements
    std::ptrdiff_t end = 0;     ///< End offset in elements (exclusive)

    [[nodiscard]] bool is_full_tensor() const { return begin == 0 && end == 0; }
    [[nodiscard]] std::ptrdiff_t size() const { return end - begin; }
};

/**
 * @brief Function type for computing load range from config
 *
 * Used for patterns where the range depends on model dimensions
 * (e.g., QKV fusion where K offset depends on num_query_heads).
 */
using RangeComputeFn = std::function<LoadRange(const PretrainedConfig& cfg, long hidden_size)>;

/**
 * @brief Identifies which internal tensor to target
 */
enum class TensorTarget {
    // Non-block tensors
    Embeddings,
    FinalNorm,
    LMHead,

    // Block tensors (require layer index)
    LN1Weight,
    LN2Weight,
    QKVWeight,
    QKVBias,
    OutWeight,
    QNormWeight,
    KNormWeight,
    MLPUpWeight,
    MLPDownWeight,

    // MoE tensors
    RouterGate,
    RouterBias,
    ExpertsGateUp,
    ExpertsDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,

    // Per-expert (non-batched)
    ExpertGate,
    ExpertUp,
    ExpertDown,
};

/**
 * @brief Complete pattern entry for weight loading
 */
struct WeightPattern {
    std::string hf_regex;           ///< Regex to match HuggingFace tensor name
    TensorTarget target;            ///< Which internal tensor to target
    RangeComputeFn range_fn;        ///< Function to compute load range (nullptr = full tensor)
    bool optional = false;          ///< Whether this tensor is optional
    int expert_group = -1;          ///< For per-expert patterns: capture group index for expert_idx

    WeightPattern() = default;
    WeightPattern(std::string regex, TensorTarget tgt,
                  RangeComputeFn fn = nullptr, bool opt = false, int exp_grp = -1)
        : hf_regex(std::move(regex)), target(tgt),
          range_fn(std::move(fn)), optional(opt), expert_group(exp_grp) {}
};

/**
 * @brief Result of matching a tensor name against patterns
 */
struct PatternMatch {
    const WeightPattern* pattern = nullptr;
    int layer_idx = -1;             ///< Extracted layer index (-1 for non-block)
    int expert_idx = -1;            ///< Extracted expert index (-1 if not per-expert)
    LoadRange range;                ///< Computed load range

    explicit operator bool() const { return pattern != nullptr; }
};

// ============================================================================
// Export Pattern Types
// ============================================================================

/**
 * @brief Describes how to export an internal tensor to HuggingFace format
 *
 * For direct exports: row_begin=0, row_end=0 (full tensor)
 * For split exports: row_begin/row_end specify the row slice
 */
struct ExportPattern {
    std::string hf_name_template;   ///< HuggingFace tensor name template ({layer}, {expert})
    TensorTarget source;            ///< Which internal tensor to export from
    long row_begin = 0;             ///< Start row for slicing (0 = from start)
    long row_end = 0;               ///< End row for slicing (0 = to end, meaning full tensor)
    bool optional = false;          ///< Whether this export is optional (e.g., bias)
    int expert_idx = -1;            ///< For per-expert: which expert (-1 = not per-expert)

    [[nodiscard]] bool is_full_tensor() const { return row_begin == 0 && row_end == 0; }

    [[nodiscard]] std::string expand_name(int layer_idx, int exp_idx = -1) const {
        std::string result = hf_name_template;
        // Replace {layer}
        size_t pos = result.find("{layer}");
        if (pos != std::string::npos) {
            result.replace(pos, 7, std::to_string(layer_idx));
        }
        // Replace {expert}
        pos = result.find("{expert}");
        if (pos != std::string::npos) {
            result.replace(pos, 8, std::to_string(exp_idx >= 0 ? exp_idx : expert_idx));
        }
        return result;
    }
};

// ============================================================================
// Base Weight Mapping
// ============================================================================

/**
 * @brief Base class for model-specific weight mappings
 *
 * Provides the framework for mapping HuggingFace tensor names to internal
 * tensor destinations. Derived classes register patterns specific to their
 * model architecture.
 */
class BaseWeightMapping {
public:
    virtual ~BaseWeightMapping() = default;

    /**
     * @brief Register all weight patterns for this model type (import)
     */
    virtual void register_patterns() = 0;

    /**
     * @brief Register all export patterns for this model type
     *
     * Override in derived classes to add model-specific export patterns.
     * Default implementation provides patterns for base Llama-style models.
     */
    virtual void register_export_patterns() {}

    /**
     * @brief Match a tensor name and compute load parameters
     *
     * @param tensor_name HuggingFace tensor name from file
     * @param cfg Model configuration for computing ranges
     * @param hidden_size Hidden dimension
     * @return Match result with pattern, indices, and range
     */
    [[nodiscard]] PatternMatch match(const std::string& tensor_name,
                                     const PretrainedConfig& cfg,
                                     long hidden_size) const {
        for (const auto& [regex, pattern] : mCompiledPatterns) {
            std::smatch m;
            if (std::regex_match(tensor_name, m, regex)) {
                PatternMatch result;
                result.pattern = &pattern;

                // Extract layer index if present (group 1 for layer patterns)
                if (m.size() > 1 && m[1].matched) {
                    result.layer_idx = std::stoi(m[1].str());
                }

                // Extract expert index if this is a per-expert pattern
                if (pattern.expert_group >= 0 && m.size() > static_cast<size_t>(pattern.expert_group) &&
                    m[pattern.expert_group].matched) {
                    result.expert_idx = std::stoi(m[pattern.expert_group].str());
                }

                // Compute load range
                if (pattern.range_fn) {
                    result.range = pattern.range_fn(cfg, hidden_size);
                }
                // else range stays {0,0} meaning full tensor

                return result;
            }
        }
        return PatternMatch{};
    }

    /**
     * @brief Get all registered patterns (for debugging/introspection)
     */
    [[nodiscard]] const std::vector<WeightPattern>& patterns() const { return mPatterns; }

    /**
     * @brief Get export patterns for non-block tensors
     */
    [[nodiscard]] const std::vector<ExportPattern>& export_nonblock_patterns() const {
        return mExportNonBlockPatterns;
    }

    /**
     * @brief Get export patterns for per-layer tensors
     */
    [[nodiscard]] const std::vector<ExportPattern>& export_layer_patterns() const {
        return mExportLayerPatterns;
    }

    /**
     * @brief Get export patterns for per-expert tensors (MoE only)
     */
    [[nodiscard]] const std::vector<ExportPattern>& export_expert_patterns() const {
        return mExportExpertPatterns;
    }

protected:
    /**
     * @brief Register a pattern for non-block weights (exact match)
     */
    void add_pattern(const std::string& hf_name, TensorTarget target,
                     RangeComputeFn range_fn = nullptr, bool optional = false) {
        // Escape dots for regex
        std::string escaped;
        for (char c : hf_name) {
            if (c == '.') escaped += "\\.";
            else escaped += c;
        }
        WeightPattern p{escaped, target, std::move(range_fn), optional};
        mPatterns.push_back(p);
        mCompiledPatterns.emplace_back(std::regex(escaped), p);
    }

    /**
     * @brief Register a pattern for per-layer weights
     *
     * Use {layer} as placeholder for layer index.
     */
    void add_layer_pattern(const std::string& hf_template, TensorTarget target,
                           RangeComputeFn range_fn = nullptr, bool optional = false) {
        std::string regex_str = escape_and_replace(hf_template, "{layer}", R"((\d+))");
        WeightPattern p{regex_str, target, std::move(range_fn), optional};
        mPatterns.push_back(p);
        mCompiledPatterns.emplace_back(std::regex(regex_str), p);
    }

    /**
     * @brief Register a pattern for per-expert weights
     *
     * Use {layer} for layer index and {expert} for expert index.
     */
    void add_expert_pattern(const std::string& hf_template, TensorTarget target,
                            RangeComputeFn range_fn = nullptr, bool optional = false) {
        // First replace {layer}, then {expert}
        std::string regex_str = escape_and_replace(hf_template, "{layer}", R"((\d+))");
        // Find position of {expert} to determine capture group
        size_t expert_pos = regex_str.find("{expert}");
        int expert_group = 2;  // Layer is group 1, expert is group 2
        (void)expert_pos;  // Suppress unused warning
        regex_str = replace_placeholder(regex_str, "{expert}", R"((\d+))");

        WeightPattern p{regex_str, target, std::move(range_fn), optional, expert_group};
        mPatterns.push_back(p);
        mCompiledPatterns.emplace_back(std::regex(regex_str), p);
    }

    // ========================================================================
    // Export pattern registration
    // ========================================================================

    /**
     * @brief Register an export pattern for non-block tensors
     */
    void add_export_nonblock(const std::string& hf_name, TensorTarget source, bool optional = false) {
        mExportNonBlockPatterns.push_back({hf_name, source, 0, 0, optional, -1});
    }

    /**
     * @brief Register an export pattern for per-layer tensors (full tensor)
     */
    void add_export_layer(const std::string& hf_template, TensorTarget source, bool optional = false) {
        mExportLayerPatterns.push_back({hf_template, source, 0, 0, optional, -1});
    }

    /**
     * @brief Register an export pattern for per-layer tensors with row slicing
     *
     * @param hf_template HF name template with {layer} placeholder
     * @param source Internal tensor to export from
     * @param row_begin Start row (inclusive)
     * @param row_end End row (exclusive), 0 means compute at export time
     * @param optional Whether the tensor is optional
     */
    void add_export_layer_slice(const std::string& hf_template, TensorTarget source,
                                long row_begin, long row_end, bool optional = false) {
        mExportLayerPatterns.push_back({hf_template, source, row_begin, row_end, optional, -1});
    }

    /**
     * @brief Register an export pattern for per-expert tensors
     */
    void add_export_expert(const std::string& hf_template, TensorTarget source, int expert_idx) {
        mExportExpertPatterns.push_back({hf_template, source, 0, 0, false, expert_idx});
    }

private:
    static std::string escape_and_replace(const std::string& input,
                                          const std::string& placeholder,
                                          const std::string& replacement) {
        std::string result;
        size_t i = 0;
        while (i < input.size()) {
            // Check for placeholder
            if (input.compare(i, placeholder.size(), placeholder) == 0) {
                result += replacement;
                i += placeholder.size();
            } else if (input[i] == '.') {
                result += "\\.";
                ++i;
            } else {
                result += input[i];
                ++i;
            }
        }
        return result;
    }

    static std::string replace_placeholder(const std::string& input,
                                           const std::string& placeholder,
                                           const std::string& replacement) {
        std::string result = input;
        size_t pos = result.find(placeholder);
        if (pos != std::string::npos) {
            result.replace(pos, placeholder.size(), replacement);
        }
        return result;
    }

    std::vector<WeightPattern> mPatterns;
    std::vector<std::pair<std::regex, WeightPattern>> mCompiledPatterns;

    // Export patterns (stored directly, no regex needed)
    std::vector<ExportPattern> mExportNonBlockPatterns;
    std::vector<ExportPattern> mExportLayerPatterns;
    std::vector<ExportPattern> mExportExpertPatterns;
};

// ============================================================================
// Range Computation Functions
// ============================================================================

namespace ranges {

// QKV weight ranges (2D: rows x hidden_size)
inline LoadRange qkv_q_weight(const PretrainedConfig& cfg, long C) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    return {0, q_rows * C};
}

inline LoadRange qkv_k_weight(const PretrainedConfig& cfg, long C) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    const long kv_rows = cfg.head_size() * cfg.NumKeyValHeads;
    return {q_rows * C, (q_rows + kv_rows) * C};
}

inline LoadRange qkv_v_weight(const PretrainedConfig& cfg, long C) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    const long kv_rows = cfg.head_size() * cfg.NumKeyValHeads;
    return {(q_rows + kv_rows) * C, (q_rows + 2 * kv_rows) * C};
}

// QKV bias ranges (1D: rows)
inline LoadRange qkv_q_bias(const PretrainedConfig& cfg, long /*C*/) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    return {0, q_rows};
}

inline LoadRange qkv_k_bias(const PretrainedConfig& cfg, long /*C*/) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    const long kv_rows = cfg.head_size() * cfg.NumKeyValHeads;
    return {q_rows, q_rows + kv_rows};
}

inline LoadRange qkv_v_bias(const PretrainedConfig& cfg, long /*C*/) {
    const long q_rows = cfg.head_size() * cfg.NumQueryHeads;
    const long kv_rows = cfg.head_size() * cfg.NumKeyValHeads;
    return {q_rows + kv_rows, q_rows + 2 * kv_rows};
}

// Gate+Up MLP ranges (2D: rows x hidden_size)
// Layout: [up, gate] each of size (intermediate_size, hidden_size)
inline LoadRange mlp_up_weight(const PretrainedConfig& cfg, long C) {
    return {0, cfg.IntermediateSize * C};
}

inline LoadRange mlp_gate_weight(const PretrainedConfig& cfg, long C) {
    const long D = cfg.IntermediateSize;
    return {D * C, 2 * D * C};
}

// Row-based ranges for slicing (used in export for proper tensor shapes)
// These return row indices rather than element indices

inline long qkv_q_rows(const PretrainedConfig& cfg) {
    return cfg.head_size() * cfg.NumQueryHeads;
}

inline long qkv_kv_rows(const PretrainedConfig& cfg) {
    return cfg.head_size() * cfg.NumKeyValHeads;
}

inline long mlp_intermediate(const PretrainedConfig& cfg) {
    return cfg.IntermediateSize;
}

} // namespace ranges

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_H
