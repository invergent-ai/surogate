// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Golden tests for compiled DSL ops (GPU).

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "dsl/compiled_ops.h"
#include "dsl/dsl_grad_store.h"
#include "dsl/dsl_param_store.h"
#include "dsl/dsl_run_state.h"
#include "dsl/graph_executor_utils.h"
#include "dsl/ir.h"
#include "modules/model_config.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct GoldenTensor {
    std::string dtype;
    std::vector<long> shape;
    std::vector<double> f64;
    std::vector<long long> i64;

    std::size_t numel() const {
        std::size_t n = 1;
        for (long d : shape) {
            n *= static_cast<std::size_t>(d);
        }
        return n;
    }

    bool is_int() const {
        return dtype.rfind("int", 0) == 0 || dtype.rfind("uint", 0) == 0;
    }
};

struct GoldenCase {
    std::string op;
    std::string case_id;
    json meta;
    dsl::AttrMap attrs;
    std::unordered_map<std::string, GoldenTensor> inputs;
    std::unordered_map<std::string, GoldenTensor> outputs;
};

struct OpSpec {
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

constexpr std::size_t kStackBytes = 64ULL * 1024ULL * 1024ULL;

bool is_special_input_name(const std::string& name) {
    return name == "token_ids" || name == "position_ids" || name == "targets" ||
           name == "labels" || name == "loss" || name == "losses" || name == "d_loss";
}

bool is_special_output_name(const std::string& name) {
    return name == "loss" || name == "losses";
}

GoldenTensor parse_tensor(const json& jt) {
    GoldenTensor t;
    t.dtype = jt.at("dtype").get<std::string>();

    for (const auto& dim : jt.at("shape")) {
        if (dim.is_number_integer()) {
            t.shape.push_back(static_cast<long>(dim.get<long long>()));
        } else if (dim.is_number_float()) {
            t.shape.push_back(static_cast<long>(dim.get<double>()));
        }
    }

    const auto& data = jt.at("data");
    if (t.is_int()) {
        t.i64.reserve(data.size());
        for (const auto& v : data) {
            t.i64.push_back(v.get<long long>());
        }
    } else {
        t.f64.reserve(data.size());
        for (const auto& v : data) {
            t.f64.push_back(v.get<double>());
        }
    }

    return t;
}

std::vector<dsl::Dim> to_dims(const std::vector<long>& shape) {
    std::vector<dsl::Dim> dims;
    dims.reserve(shape.size());
    for (long d : shape) {
        dims.push_back(dsl::Dim::concrete(d));
    }
    return dims;
}

ETensorDType device_dtype_for(const std::string& dtype) {
    if (dtype == "fp32" || dtype == "float32" || dtype == "float") {
        return ETensorDType::FP32;
    }
    if (dtype == "fp64" || dtype == "float64" || dtype == "double") {
        return ETensorDType::FP32;
    }
    if (dtype == "bf16" || dtype == "bfloat16") {
        return ETensorDType::BF16;
    }
    if (dtype == "int32" || dtype == "i32") {
        return ETensorDType::INT32;
    }
    if (dtype == "int64" || dtype == "i64") {
        return ETensorDType::INT32;
    }
    throw std::runtime_error("Unsupported dtype in golden: " + dtype);
}

void copy_tensor_to_device(Tensor& dst, const GoldenTensor& src) {
    const std::size_t n = dst.nelem();
    if (src.numel() != n) {
        throw std::runtime_error("Golden tensor element count mismatch");
    }

    if (dst.DType == ETensorDType::FP32) {
        std::vector<float> host(n);
        if (src.is_int()) {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = static_cast<float>(src.i64[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = static_cast<float>(src.f64[i]);
            }
        }
        CUDA_CHECK(cudaMemcpy(dst.Data, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        return;
    }

    if (dst.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> host(n);
        if (src.is_int()) {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = __float2bfloat16(static_cast<float>(src.i64[i]));
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = __float2bfloat16(static_cast<float>(src.f64[i]));
            }
        }
        CUDA_CHECK(cudaMemcpy(dst.Data, host.data(), n * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        return;
    }

    if (dst.DType == ETensorDType::INT32) {
        std::vector<std::int32_t> host(n);
        if (src.is_int()) {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = static_cast<std::int32_t>(src.i64[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                host[i] = static_cast<std::int32_t>(src.f64[i]);
            }
        }
        CUDA_CHECK(cudaMemcpy(dst.Data, host.data(), n * sizeof(std::int32_t), cudaMemcpyHostToDevice));
        return;
    }

    throw std::runtime_error("copy_tensor_to_device: unsupported dtype");
}

std::vector<double> read_tensor_as_double(const Tensor& t) {
    const std::size_t n = t.nelem();
    std::vector<double> out(n);

    if (t.DType == ETensorDType::FP32) {
        std::vector<float> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), t.Data, n * sizeof(float), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<double>(host[i]);
        }
        return out;
    }

    if (t.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), t.Data, n * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<double>(__bfloat162float(host[i]));
        }
        return out;
    }

    if (t.DType == ETensorDType::INT32) {
        std::vector<std::int32_t> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), t.Data, n * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<double>(host[i]);
        }
        return out;
    }

    throw std::runtime_error("read_tensor_as_double: unsupported dtype");
}

void copy_int32_to_host(Tensor& dst, const GoldenTensor& src) {
    const std::size_t n = src.numel();
    std::vector<std::int32_t> host(n);
    if (src.is_int()) {
        for (std::size_t i = 0; i < n; ++i) {
            host[i] = static_cast<std::int32_t>(src.i64[i]);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            host[i] = static_cast<std::int32_t>(src.f64[i]);
        }
    }
    std::memcpy(dst.Data, host.data(), n * sizeof(std::int32_t));
}

std::optional<long> meta_long(const json& meta, const char* key) {
    if (!meta.contains(key)) {
        return std::nullopt;
    }
    const auto& v = meta.at(key);
    if (v.is_number_integer()) {
        return static_cast<long>(v.get<long long>());
    }
    if (v.is_number_float()) {
        return static_cast<long>(v.get<double>());
    }
    return std::nullopt;
}

dsl::AttrValue parse_attr_value(const json& j) {
    if (j.is_boolean()) {
        return dsl::AttrValue(j.get<bool>());
    }
    if (j.is_number_integer()) {
        return dsl::AttrValue(static_cast<std::int64_t>(j.get<long long>()));
    }
    if (j.is_number_float()) {
        return dsl::AttrValue(j.get<double>());
    }
    if (j.is_string()) {
        return dsl::AttrValue(j.get<std::string>());
    }
    if (j.is_array()) {
        auto list = std::make_shared<dsl::AttrList>();
        list->reserve(j.size());
        for (const auto& el : j) {
            list->push_back(parse_attr_value(el));
        }
        return dsl::AttrValue(list);
    }
    if (j.is_object()) {
        auto map = std::make_shared<dsl::AttrMap>();
        for (auto it = j.begin(); it != j.end(); ++it) {
            (*map)[it.key()] = parse_attr_value(it.value());
        }
        return dsl::AttrValue(map);
    }
    return dsl::AttrValue();
}

OpSpec op_spec_for(const std::string& op) {
    static const std::unordered_map<std::string, OpSpec> kSpecs = {
        {"add", {{"a", "b"}, {"out"}}},
        {"view", {{"x"}, {"out"}}},
        {"zeros", {{}, {"out"}}},
        {"bias_add", {{"x", "bias"}, {"out"}}},
        {"matmul", {{"a", "b"}, {"out"}}},
        {"matmul_bias", {{"a", "b", "bias"}, {"out"}}},
        {"swiglu", {{"inp"}, {"out"}}},
        {"matmul_swiglu", {{"a", "b"}, {"out", "up_out"}}},
        {"embedding", {{"token_ids", "embedding"}, {"out"}}},
        {"fused_residual_rmsnorm", {{"residual_in", "input", "weight"}, {"residual_out", "y", "rstd"}}},
        {"rope", {{"qkv", "freqs", "position_ids"}, {"out"}}},
        {"qkv_qk_norm_rope", {{"qkv", "q_norm", "k_norm", "freqs", "position_ids"}, {"qkv_out", "q_rstd", "k_rstd"}}},
        {"flash_attention", {{"qkv"}, {"out", "lse"}}},
        {"cross_entropy_loss", {{"logits", "targets"}, {"loss"}}},
        {"fused_lm_head_loss", {{"xF_flat", "weight", "targets"}, {"loss"}}},
        {"add_backward", {{"d_out"}, {"d_a", "d_b"}}},
        {"view_backward", {{"d_out"}, {"d_inp"}}},
        {"bias_add_backward", {{"d_out"}, {"d_x", "d_bias"}}},
        {"matmul_backward", {{"d_out", "a", "b"}, {"d_a", "d_b"}}},
        {"swiglu_backward", {{"d_out", "inp"}, {"d_inp"}}},
        {"matmul_swiglu_backward", {{"d_out", "ln2", "weight", "mlp_up"}, {"d_inp", "d_weight"}}},
        {"embedding_backward", {{"d_out"}, {"d_embedding"}}},
        {"rope_backward", {{"d_out", "freqs", "position_ids"}, {"d_qkv"}}},
        {"qkv_qk_norm_rope_backward", {{"d_out", "qkv", "q_norm", "k_norm", "q_rstd", "k_rstd", "freqs", "position_ids"}, {"d_qkv"}}},
        {"flash_attention_backward", {{"d_out", "out", "lse", "qkv"}, {"d_qkv"}}},
        {"cross_entropy_backward", {{"d_loss", "logits", "targets"}, {"d_logits"}}},
        {"fused_lm_head_loss_backward", {{"d_loss", "xF_flat", "weight", "targets"}, {"d_xF", "d_weight"}}},
        {"fused_residual_rmsnorm_backward", {{"d_y", "d_residual_next", "residual_out", "weight", "rstd"}, {"d_residual", "d_input", "d_weight"}}},
        {"zeros_backward", {{}, {}}},
    };

    auto it = kSpecs.find(op);
    if (it == kSpecs.end()) {
        throw std::runtime_error("Unknown op in golden: " + op);
    }
    return it->second;
}

bool is_backward_op(const std::string& op) {
    const std::string suffix = "_backward";
    if (op.size() < suffix.size()) {
        return false;
    }
    return op.compare(op.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::pair<double, double> tolerance_for_op(const std::string& op) {
    if (op == "flash_attention" || op == "flash_attention_backward" ||
        op == "qkv_qk_norm_rope" || op == "qkv_qk_norm_rope_backward") {
        return {1e-3, 1e-3};
    }
    if (op == "rope" || op == "rope_backward") {
        return {1e-4, 1e-4};
    }
    if (op == "fused_lm_head_loss" || op == "fused_lm_head_loss_backward") {
        return {1e-4, 1e-4};
    }
    return {1e-5, 1e-5};
}

GoldenCase load_case(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open golden file: " + path.string());
    }

    json root;
    in >> root;

    GoldenCase gc;
    gc.op = root.at("op").get<std::string>();
    gc.case_id = root.at("case").get<std::string>();
    if (root.contains("meta")) {
        gc.meta = root.at("meta");
    }

    if (root.contains("attrs")) {
        const auto& attrs = root.at("attrs");
        for (auto it = attrs.begin(); it != attrs.end(); ++it) {
            gc.attrs[it.key()] = parse_attr_value(it.value());
        }
    }

    if (root.contains("inputs")) {
        for (auto it = root.at("inputs").begin(); it != root.at("inputs").end(); ++it) {
            gc.inputs.emplace(it.key(), parse_tensor(it.value()));
        }
    }

    if (root.contains("outputs")) {
        for (auto it = root.at("outputs").begin(); it != root.at("outputs").end(); ++it) {
            gc.outputs.emplace(it.key(), parse_tensor(it.value()));
        }
    }

    return gc;
}

std::pair<long, long> infer_B_T(const GoldenCase& gc) {
    const auto B_meta = meta_long(gc.meta, "B");
    const auto T_meta = meta_long(gc.meta, "T");
    if (B_meta && T_meta) {
        return {*B_meta, *T_meta};
    }

    auto find_shape = [&](const std::string& name) -> std::optional<std::vector<long>> {
        auto it = gc.inputs.find(name);
        if (it != gc.inputs.end()) {
            return it->second.shape;
        }
        return std::nullopt;
    };

    if (auto shape = find_shape("token_ids")) {
        if (shape->size() >= 2) {
            return {(*shape)[0], (*shape)[1]};
        }
    }
    if (auto shape = find_shape("position_ids")) {
        if (shape->size() >= 2) {
            return {(*shape)[0], (*shape)[1]};
        }
    }
    if (auto shape = find_shape("d_out")) {
        if (shape->size() >= 2) {
            return {(*shape)[0], (*shape)[1]};
        }
    }
    if (auto shape = find_shape("qkv")) {
        if (shape->size() >= 2) {
            return {(*shape)[0], (*shape)[1]};
        }
    }
    if (auto shape = find_shape("targets")) {
        if (shape->size() == 2) {
            return {(*shape)[0], (*shape)[1]};
        }
        if (shape->size() == 1) {
            return {1, (*shape)[0]};
        }
    }
    if (auto shape = find_shape("logits")) {
        if (shape->size() >= 1) {
            return {1, (*shape)[0]};
        }
    }

    // Fallback
    return {1, 1};
}

PretrainedConfig build_config(const GoldenCase& gc, long B, long T) {
    PretrainedConfig cfg;
    cfg.DType = ETensorDType::FP32;
    cfg.NumLayers = 1;
    cfg.NumQueryHeads = 1;
    cfg.NumKeyValHeads = 1;
    cfg.HiddenSize = 1;
    cfg.IntermediateSize = 1;
    cfg.VocabSize = 1;
    cfg.MaxPositionEmbeddings = static_cast<int>(std::max<long>(T, 8));
    cfg.RmsNormEps = 1e-5f;

    if (auto eps_it = gc.attrs.find("eps"); eps_it != gc.attrs.end()) {
        if (auto v = std::get_if<double>(&eps_it->second.value)) {
            cfg.RmsNormEps = static_cast<float>(*v);
        } else if (auto v = std::get_if<std::int64_t>(&eps_it->second.value)) {
            cfg.RmsNormEps = static_cast<float>(*v);
        }
    }

    if (auto v = meta_long(gc.meta, "Hq")) {
        cfg.NumQueryHeads = static_cast<int>(*v);
    }
    if (auto v = meta_long(gc.meta, "Hkv")) {
        cfg.NumKeyValHeads = static_cast<int>(*v);
    } else if (meta_long(gc.meta, "Hq")) {
        cfg.NumKeyValHeads = cfg.NumQueryHeads;
    }

    if (auto v = meta_long(gc.meta, "head_dim")) {
        cfg.HeadDim = static_cast<int>(*v);
    }

    if (auto v = meta_long(gc.meta, "C")) {
        cfg.HiddenSize = static_cast<int>(*v);
    } else if (auto v = meta_long(gc.meta, "hidden")) {
        cfg.HiddenSize = static_cast<int>(*v);
    } else if (auto v = meta_long(gc.meta, "hidden_size")) {
        cfg.HiddenSize = static_cast<int>(*v);
    } else if (cfg.HeadDim > 0 && cfg.NumQueryHeads > 0) {
        cfg.HiddenSize = cfg.HeadDim * cfg.NumQueryHeads;
    }

    bool set_intermediate = false;
    if (auto v = meta_long(gc.meta, "D")) {
        cfg.IntermediateSize = static_cast<int>(*v);
        set_intermediate = true;
    } else if (gc.op == "swiglu" || gc.op == "swiglu_backward") {
        auto it = gc.inputs.find("inp");
        if (it != gc.inputs.end() && !it->second.shape.empty()) {
            cfg.IntermediateSize = static_cast<int>(it->second.shape.back() / 2);
            set_intermediate = true;
        }
    } else if (gc.op == "matmul_swiglu" || gc.op == "matmul_swiglu_backward") {
        auto it = gc.outputs.find("out");
        if (it != gc.outputs.end() && !it->second.shape.empty()) {
            cfg.IntermediateSize = static_cast<int>(it->second.shape.back());
            set_intermediate = true;
        }
    }
    if (!set_intermediate) {
        cfg.IntermediateSize = std::max(1, cfg.HiddenSize);
    }

    if (auto v = meta_long(gc.meta, "vocab_size")) {
        cfg.VocabSize = static_cast<int>(*v);
    } else if (auto v = meta_long(gc.meta, "V")) {
        cfg.VocabSize = static_cast<int>(*v);
    } else if (auto it = gc.inputs.find("weight"); it != gc.inputs.end() && it->second.shape.size() >= 1) {
        cfg.VocabSize = static_cast<int>(it->second.shape[0]);
    } else if (auto it = gc.outputs.find("d_embedding"); it != gc.outputs.end() && it->second.shape.size() >= 1) {
        cfg.VocabSize = static_cast<int>(it->second.shape[0]);
    } else if (auto it = gc.inputs.find("logits"); it != gc.inputs.end() && it->second.shape.size() >= 2) {
        cfg.VocabSize = static_cast<int>(it->second.shape[1]);
    }

    if (gc.op == "qkv_qk_norm_rope" || gc.op == "qkv_qk_norm_rope_backward") {
        cfg.UseQKNorm = true;
    }

    return cfg;
}

fs::path find_goldens_dir() {
    fs::path cwd = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        fs::path candidate = cwd / "tests" / "ops" / "goldens";
        if (fs::exists(candidate)) {
            return candidate;
        }
        if (!cwd.has_parent_path()) {
            break;
        }
        cwd = cwd.parent_path();
    }
    throw std::runtime_error("Could not locate tests/ops/goldens directory");
}

void expect_allclose(const std::string& label,
                     const GoldenTensor& expected,
                     const Tensor& actual,
                     double rtol,
                     double atol) {
    const auto actual_vals = read_tensor_as_double(actual);
    REQUIRE(actual_vals.size() == expected.numel());

    double max_abs = 0.0;
    double max_rel = 0.0;
    std::size_t first_bad = actual_vals.size();

    for (std::size_t i = 0; i < actual_vals.size(); ++i) {
        const double exp_val = expected.is_int() ? static_cast<double>(expected.i64[i]) : expected.f64[i];
        const double act_val = actual_vals[i];
        const double diff = std::abs(act_val - exp_val);
        const double rel = diff / (std::abs(exp_val) + 1e-12);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        if (diff > atol + rtol * std::abs(exp_val)) {
            if (first_bad == actual_vals.size()) {
                first_bad = i;
            }
        }
    }

    INFO(label << ": max_abs=" << max_abs << " max_rel=" << max_rel);
    if (first_bad != actual_vals.size()) {
        const double exp_val = expected.is_int() ? static_cast<double>(expected.i64[first_bad]) : expected.f64[first_bad];
        const double act_val = actual_vals[first_bad];
        INFO(label << ": first_bad idx=" << first_bad << " expected=" << exp_val << " actual=" << act_val);
    }
    REQUIRE(first_bad == actual_vals.size());
}

}  // namespace

TEST_CASE("dsl compiled ops match goldens", "[dsl][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(goldens_dir)) {
        if (entry.path().extension() == ".json") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    REQUIRE(!files.empty());

    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        for (const auto& path : files) {
            const GoldenCase gc = load_case(path);
            const OpSpec spec = op_spec_for(gc.op);
            const auto [B, T] = infer_B_T(gc);

            INFO("golden=" << path.filename().string());
            INFO("op=" << gc.op << " case=" << gc.case_id << " B=" << B << " T=" << T);

            PretrainedConfig cfg = build_config(gc, B, T);
            modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

            RuntimeOptions options;
            options.UseCudaGraphs = false;
            options.RecomputeBlock = false;
            options.ModelType = cfg.DType;
            options.MatmulType = cfg.DType;
            options.GradientType = cfg.DType;

            auto allocator = std::make_shared<TensorAllocator>();

            dsl::Module module;
            module.name = "golden";
            module.kind = "model";

            dsl::Graph graph;
            graph.name = gc.op;

            std::unordered_map<std::string, GoldenTensor> param_inputs;

            auto input_from_json = [&](const std::string& name) -> const GoldenTensor& {
                if (gc.op == "embedding" && name == "embedding") {
                    auto it = gc.inputs.find("weight");
                    if (it == gc.inputs.end()) {
                        throw std::runtime_error("embedding golden missing weight");
                    }
                    return it->second;
                }
                auto it = gc.inputs.find(name);
                if (it == gc.inputs.end()) {
                    throw std::runtime_error("golden missing input: " + name);
                }
                return it->second;
            };

            // Inputs
            for (const auto& name : spec.inputs) {
                const auto& gt = input_from_json(name);
                dsl::TensorInfo info;
                info.shape = to_dims(gt.shape);
                info.dtype = device_dtype_for(gt.dtype);
                info.is_input = true;

                if (is_special_input_name(name)) {
                    graph.inputs.emplace(name, info);
                } else {
                    graph.params.emplace(name, info);
                    param_inputs.emplace(name, gt);
                }
            }

            // Outputs
            for (const auto& name : spec.outputs) {
                auto it = gc.outputs.find(name);
                if (it == gc.outputs.end()) {
                    throw std::runtime_error("golden missing output: " + name);
                }
                const auto& gt = it->second;
                dsl::TensorInfo info;
                info.shape = to_dims(gt.shape);
                info.dtype = device_dtype_for(gt.dtype);
                info.is_output = true;
                graph.outputs.emplace(name, info);
            }

            // Build operation
            dsl::Operation op;
            op.id = gc.op + \"_\" + gc.case_id;
            op.name = gc.op;
            op.kernel_type = gc.op;
            op.inputs = spec.inputs;
            op.outputs = spec.outputs;
            op.attrs = gc.attrs;
            graph.operations.push_back(op);

            // Ensure embedding params exist for embedding_backward (needed for d_embedding grads)
            if (gc.op == "embedding_backward" && graph.params.find("embedding") == graph.params.end()) {
                auto it = gc.outputs.find("d_embedding");
                if (it == gc.outputs.end()) {
                    throw std::runtime_error("embedding_backward missing d_embedding output");
                }
                dsl::TensorInfo info;
                info.shape = to_dims(it->second.shape);
                info.dtype = device_dtype_for(it->second.dtype);
                info.is_param = true;
                graph.params.emplace("embedding", info);
            }

            // Fill module and graph
            module.forward = graph;

            dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
            dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

            dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                       false, kStackBytes, true);

            // Copy parameter inputs
            for (const auto& kv : param_inputs) {
                Tensor& dst = params.get(kv.first);
                copy_tensor_to_device(dst, kv.second);
            }

            // Special inputs: token_ids / position_ids / targets / d_loss
            if (auto it = gc.inputs.find("token_ids"); it != gc.inputs.end()) {
                copy_tensor_to_device(run_state.Inputs, it->second);
                copy_int32_to_host(run_state.Inputs_CPU, it->second);
            }
            if (auto it = gc.inputs.find("position_ids"); it != gc.inputs.end()) {
                copy_tensor_to_device(run_state.PositionIDs, it->second);
                copy_int32_to_host(run_state.PositionIDs_CPU, it->second);
            }
            if (auto it = gc.inputs.find("targets"); it != gc.inputs.end()) {
                copy_tensor_to_device(run_state.Targets, it->second);
                copy_int32_to_host(run_state.Targets_CPU, it->second);
            }
            if (auto it = gc.inputs.find("d_loss"); it != gc.inputs.end()) {
                copy_tensor_to_device(run_state.scratch().cross_entropy_dloss, it->second);
            }

            dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
            auto compiled = compiler.compile(graph, B, T);

            dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
            exec.set_dimensions(B, T);

            if (gc.op == "embedding_backward") {
                exec.set_last_inputs_cpu(&run_state.Inputs_CPU);
            }

            if (is_backward_op(gc.op)) {
                exec.execute_backward(compiled, comm, 1, 0, nullptr);
            } else {
                exec.execute_forward(compiled, comm, true, nullptr);
            }

            CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

            const auto [rtol, atol] = tolerance_for_op(gc.op);
            for (const auto& name : spec.outputs) {
                auto exp_it = gc.outputs.find(name);
                if (exp_it == gc.outputs.end()) {
                    FAIL("Missing expected output for " + name);
                }
                const auto& expected = exp_it->second;

                if (is_special_output_name(name)) {
                    Tensor loss_view = view_tensor(run_state.Losses, expected.shape);
                    expect_allclose(name, expected, loss_view, rtol, atol);
                    continue;
                }

                const Tensor* actual = exec.try_get_tensor(name);
                if (!actual) {
                    FAIL("Missing actual tensor for output: " + name);
                }
                expect_allclose(name, expected, *actual, rtol, atol);
            }
        }
    });
}
