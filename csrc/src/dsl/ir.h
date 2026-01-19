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
