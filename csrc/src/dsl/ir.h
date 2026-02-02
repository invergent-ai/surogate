// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL IR structures and JSON loader.

#ifndef SUROGATE_SRC_DSL_IR_H
#define SUROGATE_SRC_DSL_IR_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "utilities/dtype.h"

namespace dsl {

enum class DimKind {
    Concrete,
    Symbolic,
    Computed,
    Variadic,
};

struct Dim {
    DimKind kind = DimKind::Concrete;
    long value = 0;
    std::string expr;

    static Dim concrete(long v) {
        Dim d;
        d.kind = DimKind::Concrete;
        d.value = v;
        return d;
    }

    static Dim symbolic(std::string name) {
        Dim d;
        d.kind = DimKind::Symbolic;
        d.expr = std::move(name);
        return d;
    }

    static Dim computed(std::string expr) {
        Dim d;
        d.kind = DimKind::Computed;
        d.expr = std::move(expr);
        return d;
    }

    static Dim variadic() {
        Dim d;
        d.kind = DimKind::Variadic;
        d.expr = "*";
        return d;
    }
};

struct AttrValue;

using AttrList = std::vector<AttrValue>;
using AttrMap = std::unordered_map<std::string, AttrValue>;

struct AttrValue {
    using ListPtr = std::shared_ptr<AttrList>;
    using MapPtr = std::shared_ptr<AttrMap>;

    std::variant<
        std::monostate,
        bool,
        std::int64_t,
        double,
        std::string,
        ListPtr,
        MapPtr
    > value;

    AttrValue() = default;

    template<typename T>
    AttrValue(T v) : value(std::move(v)) {}
};

struct TensorInfo {
    std::vector<Dim> shape;
    std::optional<ETensorDType> dtype;
    bool is_param = false;
    bool is_input = false;
    bool is_output = false;
};

// ============================================================================
// Activation Slot IR (generated from Python DSL)
// ============================================================================

/// @brief Scope of an activation slot
enum class ActivationScope : std::uint8_t {
    Block,           ///< Per-layer activation (in SimplifiedLayerActivations)
    Global,          ///< Global activation (in NonBlockActivations)
    Gradient,        ///< Per-layer gradient (in SimplifiedLayerGradients)
    GlobalGradient,  ///< Global gradient (in NonBlockGradientBuffers)
};

/// @brief Memory management hints for activation slots
enum class ActivationMemoryHint : std::uint8_t {
    Persistent,      ///< Keep in memory across forward/backward
    Save,            ///< Save for backward pass
    Recompute,       ///< Can be recomputed in backward
    Temporary,       ///< Stack-allocated, freed after use
    Shared,          ///< Shares memory with another slot
};

/// @brief Specification for a single activation tensor slot
struct ActivationSlotIR {
    std::string name;                          ///< Canonical slot name (e.g., "ln1", "qkv")
    ActivationScope scope = ActivationScope::Block;
    std::vector<Dim> shape;                    ///< Shape expression using symbolic dims
    std::optional<ETensorDType> dtype;         ///< Override dtype (nullopt = inherit)
    std::vector<std::string> aliases;          ///< Alternative names mapping to this slot
    ActivationMemoryHint memory_hint = ActivationMemoryHint::Persistent;
    std::string shares_with;                   ///< If memory_hint == Shared, slot to share with
    bool save_for_backward = false;            ///< Add to forward save list
    bool recompute_in_backward = false;        ///< Can be recomputed instead of saved
    std::vector<std::string> recompute_from;   ///< Dependencies for recompute
    std::string recompute_op;                  ///< Recompute op type (dispatch key)
    AttrMap recompute_attrs;                   ///< Recompute op attributes
    std::string recompute_policy;              ///< "always", "lora_only", "never"
    std::string recompute_group;               ///< Group ID for multi-output ops
    std::vector<std::string> recompute_outputs;///< Explicit recompute outputs
    std::vector<std::string> lora_targets;     ///< LoRA targets for matmul recompute
    std::string gradient_of;                   ///< For gradient slots: corresponding forward activation
    std::string alias_of;                      ///< Optional alias target (reuse existing buffer)
    std::string condition;                     ///< Condition expression (e.g., "use_qk_norm")
    std::string description;                   ///< Documentation
};

/// @brief Complete activation layout for a block or model
struct ActivationLayoutIR {
    std::string name;                          ///< Layout name (e.g., "Qwen3BlockActivations")
    std::vector<ActivationSlotIR> slots;       ///< Forward activation slots
    std::vector<ActivationSlotIR> gradient_slots;  ///< Backward gradient slots
    std::string extends;                       ///< Base layout to extend (optional)

    /// @brief Get slot by name or alias (returns nullptr if not found)
    const ActivationSlotIR* get_slot(const std::string& name) const;

    /// @brief Get slot index by name or alias (-1 if not found)
    int get_slot_index(const std::string& name) const;

    /// @brief Build mapping from aliases to canonical slot names
    std::unordered_map<std::string, std::string> build_alias_map() const;

    /// @brief Get list of slots that should be saved for backward
    std::vector<std::string> get_save_list() const;
};

struct Operation {
    std::string id;
    std::string name;
    std::string kernel_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    AttrMap attrs;
};

struct Graph {
    std::string name;
    std::unordered_map<std::string, TensorInfo> inputs;
    std::unordered_map<std::string, TensorInfo> outputs;
    std::unordered_map<std::string, TensorInfo> params;
    std::unordered_map<std::string, TensorInfo> intermediates;
    std::vector<std::string> save;
    std::vector<std::string> recompute;
    std::vector<Operation> operations;
};

struct Module {
    std::string name;
    std::string kind;
    std::optional<std::string> extends;
    AttrMap config;
    AttrMap hf_config;
    AttrMap hf_mapping;
    AttrMap hf_export;
    std::unordered_map<std::string, TensorInfo> params;
    std::optional<Graph> forward;
    std::optional<Graph> backward;
    std::optional<ActivationLayoutIR> activation_layout;  ///< Activation slots (from DSL)
};

struct IRFile {
    std::string source_file;
    bool success = false;
    std::vector<std::string> warnings;
    std::vector<Module> modules;
};

struct ShapeEnv {
    std::unordered_map<std::string, long> values;
};

IRFile load_ir_from_json(const nlohmann::json& root);
IRFile load_ir_file(const std::string& path);

ShapeEnv make_shape_env(const Module& module, long B, long T);
long resolve_dim(const Dim& dim, const ShapeEnv& env);
std::vector<long> resolve_shape(const std::vector<Dim>& dims, const ShapeEnv& env);

} // namespace dsl

#endif // SUROGATE_SRC_DSL_IR_H
