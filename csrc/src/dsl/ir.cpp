// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL IR JSON loader and shape resolution helpers.

#include "dsl/ir.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

namespace dsl {
namespace {

bool is_identifier(std::string_view text) {
    if (text.empty()) {
        return false;
    }
    const unsigned char c0 = static_cast<unsigned char>(text[0]);
    if (!(std::isalpha(c0) || text[0] == '_')) {
        return false;
    }
    for (std::size_t i = 1; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (!(std::isalnum(c) || text[i] == '_')) {
            return false;
        }
    }
    return true;
}

Dim parse_dim(const nlohmann::json& dim_json) {
    if (dim_json.is_number_integer()) {
        return Dim::concrete(dim_json.get<long>());
    }
    if (dim_json.is_number_unsigned()) {
        return Dim::concrete(static_cast<long>(dim_json.get<std::uint64_t>()));
    }
    if (dim_json.is_string()) {
        std::string text = dim_json.get<std::string>();
        if (text == "*") {
            return Dim::variadic();
        }
        if (is_identifier(text)) {
            return Dim::symbolic(std::move(text));
        }
        return Dim::computed(std::move(text));
    }
    throw std::runtime_error("Invalid dim value in DSL IR");
}

AttrValue parse_attr(const nlohmann::json& value) {
    if (value.is_null()) {
        return {};
    }
    if (value.is_boolean()) {
        return AttrValue{value.get<bool>()};
    }
    if (value.is_number_integer()) {
        return AttrValue{value.get<std::int64_t>()};
    }
    if (value.is_number_unsigned()) {
        return AttrValue{static_cast<std::int64_t>(value.get<std::uint64_t>())};
    }
    if (value.is_number_float()) {
        return AttrValue{value.get<double>()};
    }
    if (value.is_string()) {
        return AttrValue{value.get<std::string>()};
    }
    if (value.is_array()) {
        AttrList items;
        items.reserve(value.size());
        for (const auto& el : value) {
            items.push_back(parse_attr(el));
        }
        return AttrValue{std::make_shared<AttrList>(std::move(items))};
    }
    if (value.is_object()) {
        auto obj = std::make_shared<AttrMap>();
        for (auto it = value.begin(); it != value.end(); ++it) {
            obj->emplace(it.key(), parse_attr(it.value()));
        }
        return AttrValue{std::move(obj)};
    }
    throw std::runtime_error("Unsupported attr value type in DSL IR");
}

AttrMap parse_attr_map(const nlohmann::json& obj) {
    AttrMap out;
    if (!obj.is_object()) {
        return out;
    }
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        out.emplace(it.key(), parse_attr(it.value()));
    }
    return out;
}

TensorInfo parse_tensor_info(const nlohmann::json& obj) {
    TensorInfo info;
    if (obj.contains("shape") && !obj["shape"].is_null()) {
        for (const auto& dim_json : obj["shape"]) {
            info.shape.push_back(parse_dim(dim_json));
        }
    }
    if (obj.contains("dtype") && !obj["dtype"].is_null()) {
        info.dtype = dtype_from_str(obj["dtype"].get<std::string>());
    }
    info.is_param = obj.value("is_param", false);
    info.is_input = obj.value("is_input", false);
    info.is_output = obj.value("is_output", false);
    return info;
}

std::unordered_map<std::string, TensorInfo> parse_tensor_map(const nlohmann::json& obj) {
    std::unordered_map<std::string, TensorInfo> out;
    if (!obj.is_object()) {
        return out;
    }
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        out.emplace(it.key(), parse_tensor_info(it.value()));
    }
    return out;
}

Operation parse_operation(const nlohmann::json& op_json) {
    Operation op;
    if (op_json.contains("id") && !op_json["id"].is_null()) {
        op.id = op_json["id"].get<std::string>();
    }
    if (op_json.contains("name") && !op_json["name"].is_null()) {
        op.name = op_json["name"].get<std::string>();
    }
    if (op_json.contains("kernel_type") && !op_json["kernel_type"].is_null()) {
        op.kernel_type = op_json["kernel_type"].get<std::string>();
    }
    if (op_json.contains("inputs")) {
        for (const auto& item : op_json["inputs"]) {
            op.inputs.push_back(item.get<std::string>());
        }
    }
    if (op_json.contains("outputs")) {
        for (const auto& item : op_json["outputs"]) {
            op.outputs.push_back(item.get<std::string>());
        }
    }
    if (op_json.contains("attrs") && !op_json["attrs"].is_null()) {
        op.attrs = parse_attr_map(op_json["attrs"]);
    }
    return op;
}

Graph parse_graph(const nlohmann::json& graph_json) {
    Graph graph;
    if (graph_json.contains("name") && !graph_json["name"].is_null()) {
        graph.name = graph_json["name"].get<std::string>();
    }
    if (graph_json.contains("inputs")) {
        graph.inputs = parse_tensor_map(graph_json["inputs"]);
    }
    if (graph_json.contains("outputs")) {
        graph.outputs = parse_tensor_map(graph_json["outputs"]);
    }
    if (graph_json.contains("params")) {
        graph.params = parse_tensor_map(graph_json["params"]);
    }
    if (graph_json.contains("intermediates")) {
        graph.intermediates = parse_tensor_map(graph_json["intermediates"]);
    }
    if (graph_json.contains("save")) {
        for (const auto& item : graph_json["save"]) {
            graph.save.push_back(item.get<std::string>());
        }
    }
    if (graph_json.contains("recompute")) {
        for (const auto& item : graph_json["recompute"]) {
            graph.recompute.push_back(item.get<std::string>());
        }
    }
    if (graph_json.contains("operations")) {
        for (const auto& op_json : graph_json["operations"]) {
            graph.operations.push_back(parse_operation(op_json));
        }
    }
    return graph;
}

class ExprParser {
public:
    ExprParser(std::string_view expr, const ShapeEnv& env)
        : mExpr(expr), mEnv(env) {}

    long parse() {
        long value = parse_expr();
        skip_ws();
        if (mPos != mExpr.size()) {
            throw std::runtime_error("Unexpected token in DSL shape expression");
        }
        return value;
    }

private:
    long parse_expr() {
        long value = parse_term();
        while (true) {
            skip_ws();
            if (match('+')) {
                value += parse_term();
                continue;
            }
            if (match('-')) {
                value -= parse_term();
                continue;
            }
            break;
        }
        return value;
    }

    long parse_term() {
        long value = parse_factor();
        while (true) {
            skip_ws();
            if (match('*')) {
                value *= parse_factor();
                continue;
            }
            if (match_div()) {
                long rhs = parse_factor();
                if (rhs == 0) {
                    throw std::runtime_error("Division by zero in DSL shape expression");
                }
                value /= rhs;
                continue;
            }
            break;
        }
        return value;
    }

    long parse_factor() {
        skip_ws();
        if (match('+')) {
            return parse_factor();
        }
        if (match('-')) {
            return -parse_factor();
        }
        if (match('(')) {
            long value = parse_expr();
            skip_ws();
            if (!match(')')) {
                throw std::runtime_error("Unclosed '(' in DSL shape expression");
            }
            return value;
        }
        if (mPos >= mExpr.size()) {
            throw std::runtime_error("Unexpected end of DSL shape expression");
        }
        if (std::isdigit(static_cast<unsigned char>(mExpr[mPos]))) {
            return parse_number();
        }
        if (std::isalpha(static_cast<unsigned char>(mExpr[mPos])) || mExpr[mPos] == '_') {
            return parse_identifier();
        }
        throw std::runtime_error("Invalid token in DSL shape expression");
    }

    long parse_number() {
        std::size_t start = mPos;
        while (mPos < mExpr.size() && std::isdigit(static_cast<unsigned char>(mExpr[mPos]))) {
            ++mPos;
        }
        std::string_view token = mExpr.substr(start, mPos - start);
        return std::stol(std::string(token));
    }

    long parse_identifier() {
        std::size_t start = mPos;
        ++mPos;
        while (mPos < mExpr.size()) {
            const unsigned char c = static_cast<unsigned char>(mExpr[mPos]);
            if (!(std::isalnum(c) || mExpr[mPos] == '_')) {
                break;
            }
            ++mPos;
        }
        std::string name(mExpr.substr(start, mPos - start));
        auto it = mEnv.values.find(name);
        if (it == mEnv.values.end()) {
            throw std::runtime_error("Unknown symbol in DSL shape expression: " + name);
        }
        return it->second;
    }

    bool match(char c) {
        if (mPos < mExpr.size() && mExpr[mPos] == c) {
            ++mPos;
            return true;
        }
        return false;
    }

    bool match_div() {
        if (mPos >= mExpr.size() || mExpr[mPos] != '/') {
            return false;
        }
        if (mPos + 1 < mExpr.size() && mExpr[mPos + 1] == '/') {
            mPos += 2;
            return true;
        }
        ++mPos;
        return true;
    }

    void skip_ws() {
        while (mPos < mExpr.size() && std::isspace(static_cast<unsigned char>(mExpr[mPos]))) {
            ++mPos;
        }
    }

    std::string_view mExpr;
    const ShapeEnv& mEnv;
    std::size_t mPos = 0;
};

} // namespace

IRFile load_ir_from_json(const nlohmann::json& root) {
    IRFile ir;
    if (root.contains("source_file") && !root["source_file"].is_null()) {
        ir.source_file = root["source_file"].get<std::string>();
    }
    if (root.contains("success")) {
        ir.success = root["success"].get<bool>();
    }
    if (root.contains("warnings") && root["warnings"].is_array()) {
        for (const auto& warning : root["warnings"]) {
            ir.warnings.push_back(warning.get<std::string>());
        }
    }
    if (root.contains("modules")) {
        for (const auto& mod_json : root["modules"]) {
            Module mod;
            mod.name = mod_json.value("name", "");
            mod.kind = mod_json.value("kind", "");
            if (mod_json.contains("extends") && !mod_json["extends"].is_null()) {
                mod.extends = mod_json["extends"].get<std::string>();
            }
            if (mod_json.contains("config")) {
                mod.config = parse_attr_map(mod_json["config"]);
            }
            if (mod_json.contains("hf_config")) {
                mod.hf_config = parse_attr_map(mod_json["hf_config"]);
            }
            if (mod_json.contains("hf_mapping")) {
                mod.hf_mapping = parse_attr_map(mod_json["hf_mapping"]);
            }
            if (mod_json.contains("hf_export")) {
                mod.hf_export = parse_attr_map(mod_json["hf_export"]);
            }
            if (mod_json.contains("params")) {
                mod.params = parse_tensor_map(mod_json["params"]);
            }
            if (mod_json.contains("forward") && !mod_json["forward"].is_null()) {
                mod.forward = parse_graph(mod_json["forward"]);
            }
            if (mod_json.contains("backward") && !mod_json["backward"].is_null()) {
                mod.backward = parse_graph(mod_json["backward"]);
            }
            ir.modules.push_back(std::move(mod));
        }
    }
    return ir;
}

IRFile load_ir_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open DSL IR file: " + path);
    }
    nlohmann::json root = nlohmann::json::parse(file);
    return load_ir_from_json(root);
}

ShapeEnv make_shape_env(const Module& module, long B, long T) {
    ShapeEnv env;
    env.values.emplace("B", B);
    env.values.emplace("T", T);

    for (const auto& kv : module.config) {
        const auto* i64 = std::get_if<std::int64_t>(&kv.second.value);
        if (i64) {
            env.values.emplace(kv.first, static_cast<long>(*i64));
            continue;
        }
        const auto* f64 = std::get_if<double>(&kv.second.value);
        if (f64) {
            env.values.emplace(kv.first, static_cast<long>(*f64));
        }
    }
    return env;
}

long resolve_dim(const Dim& dim, const ShapeEnv& env) {
    switch (dim.kind) {
        case DimKind::Concrete:
            return dim.value;
        case DimKind::Symbolic: {
            auto it = env.values.find(dim.expr);
            if (it == env.values.end()) {
                throw std::runtime_error("Unknown shape symbol: " + dim.expr);
            }
            return it->second;
        }
        case DimKind::Computed: {
            ExprParser parser(dim.expr, env);
            return parser.parse();
        }
        case DimKind::Variadic:
            throw std::runtime_error("Cannot resolve variadic dimension in DSL IR");
    }
    throw std::runtime_error("Unsupported dim kind in DSL IR");
}

std::vector<long> resolve_shape(const std::vector<Dim>& dims, const ShapeEnv& env) {
    std::vector<long> resolved;
    resolved.reserve(dims.size());
    for (const auto& dim : dims) {
        resolved.push_back(resolve_dim(dim, env));
    }
    return resolved;
}

} // namespace dsl
