// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Utility functions for DSL Graph executor.

#include "dsl/graph_executor_utils.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "dsl/dsl_runtime.h"
#include "dsl/ir.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

// Forward declaration for local helper
Tensor* try_get_tensor(ExecState& st, const std::string& name, std::unordered_map<std::string, Tensor>& saved);

}  // namespace

// String utilities
bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Environment variable check
bool env_enabled(const char* name) {
    if (!name || !*name) {
        return false;
    }
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    return std::string_view(value) != "0" && std::string_view(value) != "false";
}

// Gradient name parsing
std::optional<std::string> base_param_from_grad(std::string_view name) {
    if (!starts_with(name, "d_")) {
        return std::nullopt;
    }
    std::string base(name.substr(2));
    const std::string_view accum_tag = "_accum_";
    const std::string_view from_tag = "_from_";
    std::size_t pos = std::string::npos;
    std::size_t pos_accum = base.find(accum_tag);
    std::size_t pos_from = base.find(from_tag);
    if (pos_accum != std::string::npos) {
        pos = pos_accum;
    }
    if (pos_from != std::string::npos) {
        if (pos == std::string::npos || pos_from < pos) {
            pos = pos_from;
        }
    }
    if (pos != std::string::npos) {
        base = base.substr(0, pos);
    }
    return base;
}

// Block parameter parsing (e.g., "blocks[0].qkv_weight" -> layer_idx=0, param_name="qkv_weight")
bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    // blocks[<idx>]
    if (starts_with(prefix, "blocks[")) {
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

    // blocks.<idx>
    if (starts_with(prefix, "blocks")) {
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

// Shape inference and utilities
bool infer_block_tensor_shape(const ExecState& st, std::string_view name, std::vector<long>& shape) {
    int layer_idx = -1;
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return false;
    }
    const long B = st.B;
    const long T = st.T;
    const long C = st.config.HiddenSize;
    const long D = st.config.IntermediateSize;
    const long Hq = st.config.NumQueryHeads;
    const long Hkv = st.config.NumKeyValHeads;
    const long Hs = st.config.head_size();
    const long QKV = Hs * (Hq + 2 * Hkv);
    const long AttnDim = Hq * Hs;
    const long MUp = 2 * D;

    if (field == "qkv_flat" || field == "qkv_biased") {
        shape = {B * T, QKV};
        return true;
    }
    if (field == "ln1_flat" || field == "ln2_flat") {
        shape = {B * T, C};
        return true;
    }
    if (field == "att_out_flat") {
        shape = {B * T, C};
        return true;
    }
    if (field == "att_flat") {
        shape = {B * T, AttnDim};
        return true;
    }
    if (field == "mlp_up_flat") {
        shape = {B * T, MUp};
        return true;
    }
    if (field == "mlp_down_flat") {
        shape = {B * T, C};
        return true;
    }
    if (field == "ln1" || field == "ln2" || field == "res_att" || field == "res_ffn" || field == "att_out" || field == "mlp_down") {
        shape = {B, T, C};
        return true;
    }
    if (field == "ln1_rstd" || field == "ln2_rstd") {
        shape = {B, T};
        return true;
    }
    if (field == "mlp_up") {
        shape = {B, T, MUp};
        return true;
    }
    if (field == "swiglu") {
        shape = {B, T, D};
        return true;
    }
    if (field == "qkv" || field == "qkv_rope" || field == "qkv_flat" || field == "qkv_biased") {
        shape = {B, T, QKV};
        return true;
    }
    if (field == "att") {
        shape = {B, T, AttnDim};
        return true;
    }
    if (field == "q_rstd") {
        shape = {B, T, Hq};
        return true;
    }
    if (field == "k_rstd") {
        shape = {B, T, Hkv};
        return true;
    }
    if (field == "lse") {
        shape = {B, Hq, T};
        return true;
    }
    return false;
}

std::string tensor_shape_str(const Tensor& t) {
    std::string out = "[";
    for (int i = 0; i < t.Rank; ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(t.Sizes[i]);
    }
    out += "]";
    return out;
}

bool tensor_shape_matches(const Tensor& t, const std::vector<long>& shape) {
    if (t.Rank != static_cast<int>(shape.size())) {
        return false;
    }
    for (int i = 0; i < t.Rank; ++i) {
        if (t.Sizes[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

std::size_t shape_nelem(const std::vector<long>& shape) {
    std::size_t total = 1;
    for (long dim : shape) {
        total *= static_cast<std::size_t>(dim);
    }
    return total;
}

// Attribute access helpers
const AttrValue* find_attr(const AttrMap& attrs, std::string_view key) {
    auto it = attrs.find(std::string(key));
    if (it == attrs.end()) {
        return nullptr;
    }
    return &it->second;
}

std::optional<std::string> attr_string(const AttrValue& value) {
    if (auto v = std::get_if<std::string>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

std::optional<long> attr_int(const AttrValue& value) {
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<long>(*v);
    }
    if (auto v = std::get_if<double>(&value.value)) {
        return static_cast<long>(*v);
    }
    return std::nullopt;
}

std::optional<double> attr_double(const AttrValue& value) {
    if (auto v = std::get_if<double>(&value.value)) {
        return *v;
    }
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*v);
    }
    return std::nullopt;
}

std::optional<bool> attr_bool(const AttrValue& value) {
    if (auto v = std::get_if<bool>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

// Shape environment augmentation
void augment_shape_env(ShapeEnv& env, const AttrMap& config) {
    auto get_long = [&](std::string_view key) -> std::optional<long> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::int64_t>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        if (auto v = std::get_if<double>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        return std::nullopt;
    };

    auto d_model = get_long("d_model");
    if (!d_model) {
        d_model = get_long("hidden_size");
    }
    auto num_q = get_long("num_query_heads");
    if (!num_q) {
        num_q = get_long("num_attention_heads");
    }
    auto num_kv = get_long("num_kv_heads");
    if (!num_kv) {
        num_kv = get_long("num_key_value_heads");
    }
    auto head_size = get_long("head_size");
    if (!head_size) {
        head_size = get_long("head_dim");
    }
    auto d_ff = get_long("d_ff");
    if (!d_ff) {
        d_ff = get_long("intermediate_size");
    }
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
    }
    if (num_q) {
        env.values.emplace("Hq", *num_q);
    }
    if (num_kv) {
        env.values.emplace("Hkv", *num_kv);
    } else if (num_q) {
        env.values.emplace("Hkv", *num_q);
    }
    long Hq = env.values.count("Hq") ? env.values.at("Hq") : 0;
    long Hkv = env.values.count("Hkv") ? env.values.at("Hkv") : 0;
    long C = env.values.count("C") ? env.values.at("C") : 0;
    if (!head_size && Hq > 0 && C > 0) {
        head_size = C / Hq;
    }
    if (head_size) {
        env.values.emplace("D", *head_size);
    }
    if (d_ff) {
        env.values.emplace("M", *d_ff);
        env.values.emplace("MUp", 2 * (*d_ff));
    }
    if (vocab) {
        env.values.emplace("V", *vocab);
    }
    if (Hq > 0 && head_size) {
        env.values.emplace("AttnDim", Hq * (*head_size));
    }
    if (head_size && Hq > 0 && Hkv > 0) {
        env.values.emplace("QKV", (Hq + 2 * Hkv) * (*head_size));
    }
}

std::vector<long> resolve_attr_shape(const AttrValue& value, const ShapeEnv& env) {
    const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value);
    if (!list_ptr || !*list_ptr) {
        throw std::runtime_error("DSL graph executor: shape attr is not a list");
    }
    std::vector<long> shape;
    shape.reserve((*list_ptr)->size());
    for (const auto& item : **list_ptr) {
        if (auto v = std::get_if<std::int64_t>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<double>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<std::string>(&item.value)) {
            shape.push_back(resolve_dim(Dim::computed(*v), env));
            continue;
        }
        throw std::runtime_error("DSL graph executor: unsupported shape attr item");
    }
    return shape;
}

// Tensor view utilities
Tensor view_tensor(const Tensor& src, const std::vector<long>& shape) {
    if (shape.size() > MAX_TENSOR_DIM) {
        throw std::runtime_error("DSL graph executor: view rank too large");
    }
    Tensor out = src;
    out.Rank = static_cast<int>(shape.size());
    for (int i = 0; i < out.Rank; ++i) {
        out.Sizes[i] = shape[i];
    }
    for (int i = out.Rank; i < MAX_TENSOR_DIM; ++i) {
        out.Sizes[i] = 1;
    }
    return out;
}

Tensor view_for_shape(const Tensor& src, const std::vector<long>& shape, const std::string& name) {
    if (shape.empty()) {
        return src;
    }
    if (shape_nelem(shape) != src.nelem()) {
        throw std::runtime_error("DSL graph executor: shape mismatch for tensor " + name);
    }
    return view_tensor(src, shape);
}

// Matmul utilities
std::optional<modules::MatmulOp> matmul_op_from_weight(std::string_view name, int& layer_idx) {
    std::string field;
    if (!parse_block_param(name, layer_idx, field)) {
        return std::nullopt;
    }
    if (field == "qkv_weight") return modules::MatmulOp::QKV;
    if (field == "out_weight") return modules::MatmulOp::AttnOut;
    if (field == "mlp_up_weight") return modules::MatmulOp::MLPUp;
    if (field == "mlp_down_weight") return modules::MatmulOp::MLPDown;
    return std::nullopt;
}

EMMTranspose parse_transpose(const AttrMap& attrs) {
    auto attr = find_attr(attrs, "transpose");
    if (!attr) {
        return EMMTranspose::NN;
    }
    if (auto s = attr_string(*attr)) {
        if (*s == "NN") return EMMTranspose::NN;
        if (*s == "NT") return EMMTranspose::NT;
        if (*s == "TN") return EMMTranspose::TN;
        if (*s == "TT") return EMMTranspose::TT;
    }
    throw std::runtime_error("DSL graph executor: invalid transpose attr");
}

EMMTranspose swap_transpose(EMMTranspose mode) {
    // Row-major GEMM mapping to column-major: swap A/B, swap M/N, and swap transpose flags.
    // NN -> NN, NT -> TN, TN -> NT, TT -> TT.
    switch (mode) {
        case EMMTranspose::NN:
            return EMMTranspose::NN;
        case EMMTranspose::NT:
            return EMMTranspose::TN;
        case EMMTranspose::TN:
            return EMMTranspose::NT;
        case EMMTranspose::TT:
            return EMMTranspose::TT;
    }
    return EMMTranspose::NN;
}

void matmul_dims(const Tensor& a, const Tensor& b, EMMTranspose mode, int& M, int& N, int& K) {
    if (a.Rank != 2 || b.Rank != 2) {
        throw std::runtime_error("DSL graph executor: matmul expects rank-2 tensors");
    }
    const long a0 = a.Sizes[0];
    const long a1 = a.Sizes[1];
    const long b0 = b.Sizes[0];
    const long b1 = b.Sizes[1];
    const bool transA = (mode == EMMTranspose::TN || mode == EMMTranspose::TT);
    const bool transB = (mode == EMMTranspose::NT || mode == EMMTranspose::TT);
    const long a_rows = transA ? a1 : a0;
    const long a_cols = transA ? a0 : a1;
    const long b_rows = transB ? b1 : b0;
    const long b_cols = transB ? b0 : b1;
    if (a_cols != b_rows) {
        throw std::runtime_error("DSL graph executor: matmul dimension mismatch");
    }
    M = static_cast<int>(a_rows);
    N = static_cast<int>(b_cols);
    K = static_cast<int>(a_cols);
}

// Graph utilities
bool is_required_op(const Operation& op, const std::unordered_set<std::string>& needed) {
    for (const auto& out : op.outputs) {
        if (needed.count(out) > 0) {
            return true;
        }
    }
    return false;
}

std::vector<char> compute_required_ops(const Graph& graph, const std::vector<std::string>& outputs) {
    std::unordered_set<std::string> needed(outputs.begin(), outputs.end());
    std::vector<char> required(graph.operations.size(), 0);
    for (std::size_t idx = graph.operations.size(); idx-- > 0;) {
        const auto& op = graph.operations[idx];
        if (!is_required_op(op, needed)) {
            continue;
        }
        required[idx] = 1;
        for (const auto& inp : op.inputs) {
            needed.insert(inp);
        }
    }
    return required;
}

// Temporary memory management
void free_temps(ExecState& st) {
    for (auto it = st.temps.rbegin(); it != st.temps.rend(); ++it) {
        st.rs.temp_free(*it);
    }
    st.temps.clear();
}

}  // namespace dsl
