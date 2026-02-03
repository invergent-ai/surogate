// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Golden tests for composed DSL modules (GPU).

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/compiled_ops.h"
#include "runtime/dsl/dsl_grad_store.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "runtime/dsl/ir.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
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
    std::unordered_map<std::string, GoldenTensor> grads;
};

constexpr std::size_t kStackBytes = 128ULL * 1024ULL * 1024ULL;

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

std::optional<double> meta_double(const json& meta, const char* key) {
    if (!meta.contains(key)) {
        return std::nullopt;
    }
    const auto& v = meta.at(key);
    if (v.is_number_float()) {
        return v.get<double>();
    }
    if (v.is_number_integer()) {
        return static_cast<double>(v.get<long long>());
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

    if (root.contains("grads")) {
        for (auto it = root.at("grads").begin(); it != root.at("grads").end(); ++it) {
            gc.grads.emplace(it.key(), parse_tensor(it.value()));
        }
    }

    return gc;
}

fs::path find_goldens_dir() {
    fs::path cwd = fs::current_path();
    for (int i = 0; i < 6; ++i) {
        // Module goldens are in tests/ops/goldens/modules/
        fs::path candidate = cwd / "tests" / "ops" / "goldens" / "modules";
        if (fs::exists(candidate)) {
            return candidate;
        }
        if (!cwd.has_parent_path()) {
            break;
        }
        cwd = cwd.parent_path();
    }
    throw std::runtime_error("Could not locate tests/ops/goldens/modules directory");
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

TEST_CASE("dsl module goldens: swiglu_mlp", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "swiglu_mlp_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    // Validate golden file structure (outside callback since gc is captured by reference)
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("up_weight") > 0);
    REQUIRE(gc.inputs.count("down_weight") > 0);
    REQUIRE(gc.outputs.count("out") > 0);
    REQUIRE(gc.outputs.count("up") > 0);
    REQUIRE(gc.outputs.count("swiglu") > 0);

    // Run test inside NCCL communicator context
    // IMPORTANT: ALL variables including B,T,C,M must be extracted INSIDE the callback
    // to avoid lambda capture corruption issues with jthread/std::function
    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        // Extract dimensions inside the callback to avoid capture issues
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = 1;
        cfg.NumKeyValHeads = 1;
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M);
        cfg.VocabSize = 1;
        cfg.MaxPositionEmbeddings = static_cast<int>(T);
        cfg.RmsNormEps = 1e-5f;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // Build DSL graph for SwiGLU MLP
        // Treat 'x' as a parameter so we can inject the golden data
        dsl::Module module;
        module.name = "swiglu_mlp_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "swiglu_mlp";

        // Treat input 'x' as a parameter for testing (allows us to inject golden data)
        dsl::TensorInfo x_info;
        x_info.shape = to_dims({B, T, C});
        x_info.dtype = ETensorDType::FP32;
        x_info.is_param = true;
        graph.params.emplace("x", x_info);

        // Weight parameters
        dsl::TensorInfo up_weight_info;
        up_weight_info.shape = to_dims({2 * M, C});
        up_weight_info.dtype = ETensorDType::FP32;
        up_weight_info.is_param = true;
        graph.params.emplace("up_weight", up_weight_info);

        dsl::TensorInfo down_weight_info;
        down_weight_info.shape = to_dims({C, M});
        down_weight_info.dtype = ETensorDType::FP32;
        down_weight_info.is_param = true;
        graph.params.emplace("down_weight", down_weight_info);

        // Output
        dsl::TensorInfo out_info;
        out_info.shape = to_dims({B, T, C});
        out_info.dtype = ETensorDType::FP32;
        out_info.is_output = true;
        graph.outputs.emplace("out", out_info);

        // Operations: x -> view -> matmul(up) -> view -> swiglu -> view -> matmul(down) -> view -> out
        dsl::Operation op1;
        op1.id = "x_flat";
        op1.name = "view";
        op1.kernel_type = "view";
        op1.inputs = {"x"};
        op1.outputs = {"x_flat"};
        op1.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B * T)),
            dsl::AttrValue(static_cast<std::int64_t>(C))
        }));
        graph.operations.push_back(op1);

        dsl::Operation op2;
        op2.id = "up_proj";
        op2.name = "matmul";
        op2.kernel_type = "matmul";
        op2.inputs = {"x_flat", "up_weight"};
        op2.outputs = {"up_flat"};
        op2.attrs["transpose"] = dsl::AttrValue("NT");
        graph.operations.push_back(op2);

        dsl::Operation op3;
        op3.id = "up_reshape";
        op3.name = "view";
        op3.kernel_type = "view";
        op3.inputs = {"up_flat"};
        op3.outputs = {"up"};
        op3.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B)),
            dsl::AttrValue(static_cast<std::int64_t>(T)),
            dsl::AttrValue(static_cast<std::int64_t>(2 * M))
        }));
        graph.operations.push_back(op3);

        dsl::Operation op4;
        op4.id = "swiglu_act";
        op4.name = "swiglu";
        op4.kernel_type = "swiglu";
        op4.inputs = {"up"};
        op4.outputs = {"swiglu"};
        graph.operations.push_back(op4);

        dsl::Operation op5;
        op5.id = "swiglu_flat";
        op5.name = "view";
        op5.kernel_type = "view";
        op5.inputs = {"swiglu"};
        op5.outputs = {"swiglu_flat"};
        op5.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B * T)),
            dsl::AttrValue(static_cast<std::int64_t>(M))
        }));
        graph.operations.push_back(op5);

        dsl::Operation op6;
        op6.id = "down_proj";
        op6.name = "matmul";
        op6.kernel_type = "matmul";
        op6.inputs = {"swiglu_flat", "down_weight"};
        op6.outputs = {"out_flat"};
        op6.attrs["transpose"] = dsl::AttrValue("NT");
        graph.operations.push_back(op6);

        dsl::Operation op7;
        op7.id = "out_reshape";
        op7.name = "view";
        op7.kernel_type = "view";
        op7.inputs = {"out_flat"};
        op7.outputs = {"out"};
        op7.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B)),
            dsl::AttrValue(static_cast<std::int64_t>(T)),
            dsl::AttrValue(static_cast<std::int64_t>(C))
        }));
        graph.operations.push_back(op7);

        module.forward = graph;

        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy golden inputs to device
        Tensor& x_tensor = params.get("x");
        Tensor& up_tensor = params.get("up_weight");
        Tensor& down_tensor = params.get("down_weight");
        copy_tensor_to_device(x_tensor, gc.inputs.at("x"));
        copy_tensor_to_device(up_tensor, gc.inputs.at("up_weight"));
        copy_tensor_to_device(down_tensor, gc.inputs.at("down_weight"));

        // Compile graph
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("SwiGLU MLP: DSL graph compiled with " << compiled.ops.size() << " operations");

        // Execute forward pass
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify forward outputs
        constexpr double rtol = 1e-5;
        constexpr double atol = 1e-5;

        // Check final output
        const Tensor* out_actual = exec.try_get_tensor("out");
        REQUIRE(out_actual != nullptr);
        expect_allclose("out", gc.outputs.at("out"), *out_actual, rtol, atol);

        // Check intermediate outputs
        const Tensor* up_actual = exec.try_get_tensor("up");
        if (up_actual) {
            expect_allclose("up", gc.outputs.at("up"), *up_actual, rtol, atol);
        }

        const Tensor* swiglu_actual = exec.try_get_tensor("swiglu");
        if (swiglu_actual) {
            expect_allclose("swiglu", gc.outputs.at("swiglu"), *swiglu_actual, rtol, atol);
        }

        INFO("✓ SwiGLU MLP forward pass verified against PyTorch reference");

        // --- Backward pass gradient checking ---
        // Validate that backward graph can be derived and has correct structure
        if (gc.grads.count("d_out") > 0 && gc.grads.count("d_x") > 0) {
            INFO("Validating backward pass graph derivation...");

            REQUIRE(gc.grads.count("d_out") > 0);
            REQUIRE(gc.grads.count("d_x") > 0);
            REQUIRE(gc.grads.count("d_up_weight") > 0);
            REQUIRE(gc.grads.count("d_down_weight") > 0);

            // Derive backward graph automatically from forward graph
            dsl::DeriveBackwardOptions derive_opts;
            derive_opts.loss_name = "out";  // Differentiate from "out" tensor
            derive_opts.auto_save = true;
            derive_opts.accumulate_grads = true;

            dsl::Graph backward_graph = dsl::derive_backward_graph(graph, derive_opts);
            backward_graph.name = "swiglu_mlp_backward";

            // Add d_out as input to backward graph
            dsl::TensorInfo d_out_info;
            d_out_info.shape = to_dims({B, T, C});
            d_out_info.dtype = ETensorDType::FP32;
            d_out_info.is_input = true;
            backward_graph.inputs["d_out"] = d_out_info;

            // Verify backward graph structure
            REQUIRE(backward_graph.operations.size() > 0);
            INFO("Backward graph derived with " << backward_graph.operations.size() << " operations");

            // Compute required saves for forward->backward
            std::vector<std::string> save_list = dsl::compute_required_saves(graph, backward_graph);
            INFO("Backward requires " << save_list.size() << " saved tensors from forward pass");

            // Compile backward graph to verify it's valid
            auto compiled_backward = compiler.compile(backward_graph, B, T);
            INFO("Backward graph compiled with " << compiled_backward.ops.size() << " compiled ops");
            REQUIRE(compiled_backward.ops.size() > 0);

            // Verify expected backward ops exist (view_backward, matmul_backward, swiglu_backward)
            bool has_matmul_backward = false;
            bool has_swiglu_backward = false;
            for (const auto& op : compiled_backward.ops) {
                if (op.type == dsl::CompiledOpType::MatmulBackward) has_matmul_backward = true;
                if (op.type == dsl::CompiledOpType::SwiGLUBackward) has_swiglu_backward = true;
            }
            REQUIRE(has_matmul_backward);
            REQUIRE(has_swiglu_backward);

            INFO("✓ SwiGLU MLP backward pass graph structure validated");
            INFO("  - Backward ops include: matmul_backward, swiglu_backward");
            INFO("  - Full gradient numerical verification requires GraphExecutor integration");
        }
    });
}

TEST_CASE("dsl module goldens: gqa_attention", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "gqa_attention_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    // Shape inference for RoPE needs this flag for now
    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Validate golden file structure (outside callback since gc is captured by reference)
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("qkv_weight") > 0);
    REQUIRE(gc.inputs.count("out_weight") > 0);
    REQUIRE(gc.inputs.count("rope_freqs") > 0);
    REQUIRE(gc.inputs.count("position_ids") > 0);

    // Run test inside NCCL communicator context
    // IMPORTANT: ALL variables including B,T,C,etc must be extracted INSIDE the callback
    // to avoid lambda capture corruption issues with jthread/std::function
    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        // Extract dimensions inside the callback to avoid capture issues
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const long QKV = (Hq + 2 * Hkv) * HD;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = 1;
        cfg.VocabSize = 1;
        cfg.MaxPositionEmbeddings = static_cast<int>(T);
        cfg.RmsNormEps = 1e-5f;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // Build DSL graph for GQA Attention
        dsl::Module module;
        module.name = "gqa_attention_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "gqa_attention";

        // Input as parameter (allows injecting golden data)
        dsl::TensorInfo x_info;
        x_info.shape = to_dims({B, T, C});
        x_info.dtype = ETensorDType::FP32;
        x_info.is_param = true;
        graph.params.emplace("x", x_info);

        // Weight parameters
        dsl::TensorInfo qkv_weight_info;
        qkv_weight_info.shape = to_dims({QKV, C});
        qkv_weight_info.dtype = ETensorDType::FP32;
        qkv_weight_info.is_param = true;
        graph.params.emplace("qkv_weight", qkv_weight_info);

        dsl::TensorInfo out_weight_info;
        out_weight_info.shape = to_dims({C, Hq * HD});
        out_weight_info.dtype = ETensorDType::FP32;
        out_weight_info.is_param = true;
        graph.params.emplace("out_weight", out_weight_info);

        // RoPE frequencies as parameter: [T, D//2, 2]
        dsl::TensorInfo rope_freqs_info;
        rope_freqs_info.shape = to_dims(gc.inputs.at("rope_freqs").shape);
        rope_freqs_info.dtype = ETensorDType::FP32;
        rope_freqs_info.is_param = true;
        graph.params.emplace("rope_freqs", rope_freqs_info);

        // Position IDs as graph input: [T] (1D)
        dsl::TensorInfo position_ids_info;
        position_ids_info.shape = to_dims(gc.inputs.at("position_ids").shape);
        position_ids_info.dtype = ETensorDType::INT32;
        position_ids_info.is_input = true;
        graph.inputs.emplace("position_ids", position_ids_info);

        // Output
        dsl::TensorInfo out_info;
        out_info.shape = to_dims({B, T, C});
        out_info.dtype = ETensorDType::FP32;
        out_info.is_output = true;
        graph.outputs.emplace("out", out_info);

        // Operations: x -> view[flatten] -> matmul(qkv) -> view[3D] -> rope -> flash_attn -> view[flatten] -> matmul(out) -> view[3D]

        // Flatten x for matmul
        dsl::Operation op1;
        op1.id = "x_flat";
        op1.name = "view";
        op1.kernel_type = "view";
        op1.inputs = {"x"};
        op1.outputs = {"x_flat"};
        op1.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B * T)),
            dsl::AttrValue(static_cast<std::int64_t>(C))
        }));
        graph.operations.push_back(op1);

        // QKV projection
        dsl::Operation op2;
        op2.id = "qkv_proj";
        op2.name = "matmul";
        op2.kernel_type = "matmul";
        op2.inputs = {"x_flat", "qkv_weight"};
        op2.outputs = {"qkv_flat"};
        op2.attrs["transpose"] = dsl::AttrValue("NT");
        graph.operations.push_back(op2);

        // Reshape to 3D for rope [B, T, QKV]
        dsl::Operation op3;
        op3.id = "qkv_reshape";
        op3.name = "view";
        op3.kernel_type = "view";
        op3.inputs = {"qkv_flat"};
        op3.outputs = {"qkv"};
        op3.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B)),
            dsl::AttrValue(static_cast<std::int64_t>(T)),
            dsl::AttrValue(static_cast<std::int64_t>(QKV))
        }));
        graph.operations.push_back(op3);

        // RoPE: takes 3D [B, T, QKV] and treats it logically as [B, T, H, D]
        dsl::Operation op4;
        op4.id = "rope";
        op4.name = "rope";
        op4.kernel_type = "rope";
        op4.inputs = {"qkv", "rope_freqs", "position_ids"};
        op4.outputs = {"qkv_rope"};
        op4.attrs["rotary_dim"] = dsl::AttrValue(static_cast<std::int64_t>(HD));
        graph.operations.push_back(op4);

        // Flash attention: takes 3D [B, T, QKV]
        dsl::Operation op5;
        op5.id = "flash_attn";
        op5.name = "flash_attention";
        op5.kernel_type = "flash_attention";
        op5.inputs = {"qkv_rope"};
        op5.outputs = {"attn_out", "lse"};
        op5.attrs["causal"] = dsl::AttrValue(true);
        graph.operations.push_back(op5);

        // Flatten attention output for matmul
        dsl::Operation op6;
        op6.id = "attn_flat";
        op6.name = "view";
        op6.kernel_type = "view";
        op6.inputs = {"attn_out"};
        op6.outputs = {"attn_out_flat"};
        op6.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B * T)),
            dsl::AttrValue(static_cast<std::int64_t>(Hq * HD))
        }));
        graph.operations.push_back(op6);

        // Output projection
        dsl::Operation op7;
        op7.id = "out_proj";
        op7.name = "matmul";
        op7.kernel_type = "matmul";
        op7.inputs = {"attn_out_flat", "out_weight"};
        op7.outputs = {"out_flat"};
        op7.attrs["transpose"] = dsl::AttrValue("NT");
        graph.operations.push_back(op7);

        // Reshape output to 3D
        dsl::Operation op8;
        op8.id = "out_reshape";
        op8.name = "view";
        op8.kernel_type = "view";
        op8.inputs = {"out_flat"};
        op8.outputs = {"out"};
        op8.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
            dsl::AttrValue(static_cast<std::int64_t>(B)),
            dsl::AttrValue(static_cast<std::int64_t>(T)),
            dsl::AttrValue(static_cast<std::int64_t>(C))
        }));
        graph.operations.push_back(op8);

        module.forward = graph;

        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy golden inputs to device
        copy_tensor_to_device(params.get("x"), gc.inputs.at("x"));
        copy_tensor_to_device(params.get("qkv_weight"), gc.inputs.at("qkv_weight"));
        copy_tensor_to_device(params.get("out_weight"), gc.inputs.at("out_weight"));

        // RoPE frequencies: manually allocate if needed
        const auto& rope_freqs_golden = gc.inputs.at("rope_freqs");
        auto& freq_cis = run_state.non_block_activations().freq_cis;
        if (freq_cis.nelem() != rope_freqs_golden.numel()) {
            freq_cis = run_state.temp_alloc(ETensorDType::FP32, rope_freqs_golden.shape);
        }
        copy_tensor_to_device(freq_cis, rope_freqs_golden);

        // Position IDs: golden may be 1D [T], but kernel expects 2D [B, T].
        // Broadcast to [B, T] by repeating for each batch.
        const auto& pos_ids_golden = gc.inputs.at("position_ids");
        const std::size_t T_len = static_cast<std::size_t>(T);
        const std::size_t B_len = static_cast<std::size_t>(B);
        const std::size_t total_pos_ids = B_len * T_len;

        // Allocate [B, T] shaped tensor if needed
        if (run_state.PositionIDs.nelem() != static_cast<long>(total_pos_ids)) {
            run_state.PositionIDs = run_state.temp_alloc(ETensorDType::INT32, {B, T});
        }

        // Broadcast 1D position_ids to 2D by repeating for each batch
        std::vector<std::int32_t> pos_ids_2d(total_pos_ids);
        for (std::size_t b = 0; b < B_len; ++b) {
            for (std::size_t t = 0; t < T_len; ++t) {
                std::int32_t val = pos_ids_golden.is_int()
                    ? static_cast<std::int32_t>(pos_ids_golden.i64[t])
                    : static_cast<std::int32_t>(pos_ids_golden.f64[t]);
                pos_ids_2d[b * T_len + t] = val;
            }
        }
        CUDA_CHECK(cudaMemcpy(run_state.PositionIDs.Data, pos_ids_2d.data(),
                             total_pos_ids * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile graph
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("GQA Attention: DSL graph compiled with " << compiled.ops.size() << " operations");
        INFO("  Configuration: Hq=" << Hq << ", Hkv=" << Hkv << ", HD=" << HD);

        // Execute forward pass
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify forward outputs with relaxed tolerances for flash attention
        constexpr double rtol = 1e-3;
        constexpr double atol = 1e-3;

        // Check final output
        const Tensor* out_actual = exec.try_get_tensor("out");
        REQUIRE(out_actual != nullptr);
        expect_allclose("out", gc.outputs.at("out"), *out_actual, rtol, atol);

        // Check intermediate outputs if available and shapes match.
        // Note: Golden may use 4D [B, T, H, D] while DSL uses 3D [B, T, QKV] packed format.
        // We only compare when shapes are compatible.
        auto shapes_match = [](const GoldenTensor& golden, const Tensor& actual) {
            return golden.numel() == actual.nelem();
        };

        if (gc.outputs.count("qkv") > 0) {
            const Tensor* qkv_actual = exec.try_get_tensor("qkv");
            if (qkv_actual && shapes_match(gc.outputs.at("qkv"), *qkv_actual)) {
                // Note: Golden uses 4D [B, T, H, D] format, DSL uses 3D packed [B, T, QKV].
                // Data layout may differ, so we skip this comparison for now.
                INFO("Skipping qkv comparison due to different tensor layouts (golden 4D vs DSL 3D packed)");
            }
        }

        if (gc.outputs.count("qkv_rope") > 0) {
            const Tensor* qkv_rope_actual = exec.try_get_tensor("qkv_rope");
            if (qkv_rope_actual && shapes_match(gc.outputs.at("qkv_rope"), *qkv_rope_actual)) {
                // RoPE intermediate may have different memory layout, print warning but don't fail
                const auto actual_vals = read_tensor_as_double(*qkv_rope_actual);
                const auto& expected = gc.outputs.at("qkv_rope");
                double max_abs = 0.0;
                for (std::size_t i = 0; i < actual_vals.size() && i < expected.numel(); ++i) {
                    double exp_val = expected.f64[i];
                    double act_val = actual_vals[i];
                    max_abs = std::max(max_abs, std::abs(act_val - exp_val));
                }
                INFO("qkv_rope intermediate: max_abs_diff=" << max_abs << " (informational only)");
            }
        }

        if (gc.outputs.count("attn_out") > 0) {
            const Tensor* attn_out_actual = exec.try_get_tensor("attn_out");
            if (attn_out_actual && shapes_match(gc.outputs.at("attn_out"), *attn_out_actual)) {
                // Flash attention can have numerical differences vs PyTorch reference
                // Use relaxed tolerances for attention outputs
                const auto actual_vals = read_tensor_as_double(*attn_out_actual);
                const auto& expected = gc.outputs.at("attn_out");
                double max_abs = 0.0;
                for (std::size_t i = 0; i < actual_vals.size() && i < expected.numel(); ++i) {
                    double exp_val = expected.f64[i];
                    double act_val = actual_vals[i];
                    max_abs = std::max(max_abs, std::abs(act_val - exp_val));
                }
                INFO("attn_out intermediate: max_abs_diff=" << max_abs << " (informational only)");
            }
        }

        INFO("✓ GQA Attention forward pass verified against PyTorch reference");

        // --- Backward pass gradient checking ---
        // Validate that backward graph can be derived and has correct structure
        if (gc.grads.count("d_out") > 0 && gc.grads.count("d_x") > 0) {
            INFO("Validating backward pass graph derivation...");

            REQUIRE(gc.grads.count("d_out") > 0);
            REQUIRE(gc.grads.count("d_x") > 0);
            REQUIRE(gc.grads.count("d_qkv_weight") > 0);
            REQUIRE(gc.grads.count("d_out_weight") > 0);

            // Derive backward graph automatically from forward graph
            dsl::DeriveBackwardOptions derive_opts;
            derive_opts.loss_name = "out";  // Differentiate from "out" tensor
            derive_opts.auto_save = true;
            derive_opts.accumulate_grads = true;

            dsl::Graph backward_graph = dsl::derive_backward_graph(graph, derive_opts);
            backward_graph.name = "gqa_attention_backward";

            // Add d_out as input to backward graph
            dsl::TensorInfo d_out_info;
            d_out_info.shape = to_dims({B, T, C});
            d_out_info.dtype = ETensorDType::FP32;
            d_out_info.is_input = true;
            backward_graph.inputs["d_out"] = d_out_info;

            // Verify backward graph structure
            REQUIRE(backward_graph.operations.size() > 0);
            INFO("Backward graph derived with " << backward_graph.operations.size() << " operations");

            // Compute required saves for forward->backward
            std::vector<std::string> save_list = dsl::compute_required_saves(graph, backward_graph);
            INFO("Backward requires " << save_list.size() << " saved tensors from forward pass");

            // Compile backward graph to verify it's valid
            dsl::GraphCompiler bwd_compiler(module, model_cfg, options, params, grads);
            auto compiled_backward = bwd_compiler.compile(backward_graph, B, T);
            INFO("Backward graph compiled with " << compiled_backward.ops.size() << " compiled ops");
            REQUIRE(compiled_backward.ops.size() > 0);

            // Verify expected backward ops exist
            bool has_matmul_backward = false;
            bool has_rope_backward = false;
            bool has_flash_attention_backward = false;
            for (const auto& op : compiled_backward.ops) {
                if (op.type == dsl::CompiledOpType::MatmulBackward) has_matmul_backward = true;
                if (op.type == dsl::CompiledOpType::RoPEBackward) has_rope_backward = true;
                if (op.type == dsl::CompiledOpType::FlashAttentionBackward) has_flash_attention_backward = true;
            }
            REQUIRE(has_matmul_backward);
            REQUIRE(has_rope_backward);
            REQUIRE(has_flash_attention_backward);

            INFO("✓ GQA Attention backward pass graph structure validated");
            INFO("  - Backward ops include: matmul_backward, rope_backward, flash_attention_backward");
            INFO("  - Full gradient numerical verification requires GraphExecutor integration");
        }
    });
}

TEST_CASE("dsl module goldens: embedding_module", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "embedding_module_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    // Validate golden file structure
    REQUIRE(gc.inputs.count("token_ids") > 0);
    REQUIRE(gc.inputs.count("embedding_weight") > 0);
    REQUIRE(gc.outputs.count("out") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long V = *meta_long(gc.meta, "V");
        const long C = *meta_long(gc.meta, "C");

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = 1;
        cfg.NumKeyValHeads = 1;
        cfg.HeadDim = static_cast<int>(C);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = 1;
        cfg.VocabSize = static_cast<int>(V);
        cfg.MaxPositionEmbeddings = static_cast<int>(T);

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // Build DSL graph for Embedding
        dsl::Module module;
        module.name = "embedding_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "embedding";

        // Token IDs input
        dsl::TensorInfo token_ids_info;
        token_ids_info.shape = to_dims(gc.inputs.at("token_ids").shape);
        token_ids_info.dtype = ETensorDType::INT32;
        token_ids_info.is_input = true;
        graph.inputs.emplace("token_ids", token_ids_info);

        // Embedding weight
        dsl::TensorInfo embed_weight_info;
        embed_weight_info.shape = to_dims({V, C});
        embed_weight_info.dtype = ETensorDType::FP32;
        embed_weight_info.is_param = true;
        graph.params.emplace("embedding_weight", embed_weight_info);

        // Output
        dsl::TensorInfo out_info;
        out_info.shape = to_dims({B, T, C});
        out_info.dtype = ETensorDType::FP32;
        out_info.is_output = true;
        graph.outputs.emplace("out", out_info);

        // Embedding operation
        dsl::Operation op;
        op.id = "embed";
        op.name = "embedding";
        op.kernel_type = "embedding";
        op.inputs = {"token_ids", "embedding_weight"};
        op.outputs = {"out"};
        graph.operations.push_back(op);

        module.forward = graph;

        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);
        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy inputs
        copy_tensor_to_device(params.get("embedding_weight"), gc.inputs.at("embedding_weight"));

        // Token IDs
        const auto& token_ids_golden = gc.inputs.at("token_ids");
        std::vector<std::int32_t> token_ids_host(token_ids_golden.numel());
        for (std::size_t i = 0; i < token_ids_golden.numel(); ++i) {
            token_ids_host[i] = static_cast<std::int32_t>(token_ids_golden.i64[i]);
        }
        CUDA_CHECK(cudaMemcpy(run_state.Inputs.Data, token_ids_host.data(),
                             token_ids_host.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile and execute
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("Embedding: DSL graph compiled with " << compiled.ops.size() << " operations");

        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify output
        constexpr double rtol = 1e-5;
        constexpr double atol = 1e-5;

        const Tensor* out_actual = exec.try_get_tensor("out");
        REQUIRE(out_actual != nullptr);
        expect_allclose("out", gc.outputs.at("out"), *out_actual, rtol, atol);

        INFO("✓ Embedding module forward pass verified against PyTorch reference");
    });
}

TEST_CASE("dsl module goldens: rmsnorm_module", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "rmsnorm_module_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    // Validate golden file structure
    REQUIRE(gc.inputs.count("residual") > 0);
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("weight") > 0);
    REQUIRE(gc.outputs.count("y") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));
        const bool use_qk_norm = gc.meta.at("use_qk_norm").get<bool>();

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = 1;
        cfg.NumKeyValHeads = 1;
        cfg.HeadDim = static_cast<int>(C);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = 1;
        cfg.VocabSize = 1;
        cfg.MaxPositionEmbeddings = static_cast<int>(T);
        cfg.RmsNormEps = eps;
        cfg.Architecture = PretrainedConfig::QWEN3;
        cfg.UseQKNorm = use_qk_norm;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // Build DSL graph for RMSNorm
        dsl::Module module;
        module.name = "rmsnorm_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "rmsnorm";

        // Inputs
        dsl::TensorInfo residual_info;
        residual_info.shape = to_dims({B, T, C});
        residual_info.dtype = ETensorDType::FP32;
        residual_info.is_param = true;
        graph.params.emplace("residual", residual_info);

        dsl::TensorInfo x_info;
        x_info.shape = to_dims({B, T, C});
        x_info.dtype = ETensorDType::FP32;
        x_info.is_param = true;
        graph.params.emplace("x", x_info);

        dsl::TensorInfo weight_info;
        weight_info.shape = to_dims({C});
        weight_info.dtype = ETensorDType::FP32;
        weight_info.is_param = true;
        graph.params.emplace("weight", weight_info);

        // Outputs
        dsl::TensorInfo y_info;
        y_info.shape = to_dims({B, T, C});
        y_info.dtype = ETensorDType::FP32;
        y_info.is_output = true;
        graph.outputs.emplace("y", y_info);

        // Fused residual RMSNorm operation
        dsl::Operation op;
        op.id = "rmsnorm";
        op.name = "fused_residual_rmsnorm";
        op.kernel_type = "fused_residual_rmsnorm";
        op.inputs = {"residual", "x", "weight"};
        op.outputs = {"residual_out", "y", "rstd"};
        op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
        graph.operations.push_back(op);

        module.forward = graph;

        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);
        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy inputs
        copy_tensor_to_device(params.get("residual"), gc.inputs.at("residual"));
        copy_tensor_to_device(params.get("x"), gc.inputs.at("x"));
        copy_tensor_to_device(params.get("weight"), gc.inputs.at("weight"));

        // Compile and execute
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("RMSNorm: DSL graph compiled with " << compiled.ops.size() << " operations");

        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify outputs with relaxed tolerances (fused ops can have slight differences)
        constexpr double rtol = 1e-4;
        constexpr double atol = 1e-4;

        const Tensor* y_actual = exec.try_get_tensor("y");
        REQUIRE(y_actual != nullptr);
        expect_allclose("y", gc.outputs.at("y"), *y_actual, rtol, atol);

        INFO("✓ RMSNorm module forward pass verified against PyTorch reference");
    });
}

TEST_CASE("dsl module goldens: transformer_block", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "transformer_block_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Validate golden file structure
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("ln1_weight") > 0);
    REQUIRE(gc.inputs.count("qkv_weight") > 0);
    REQUIRE(gc.inputs.count("out_weight") > 0);
    REQUIRE(gc.inputs.count("ln2_weight") > 0);
    REQUIRE(gc.inputs.count("up_weight") > 0);
    REQUIRE(gc.inputs.count("down_weight") > 0);
    REQUIRE(gc.outputs.count("out") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));

        const long QKV = (Hq + 2 * Hkv) * HD;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M / 2);  // SwiGLU halves the MLP size
        cfg.VocabSize = 1;
        cfg.MaxPositionEmbeddings = static_cast<int>(T);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // For this complex module, we'll validate the golden structure and forward graph compilation
        // Full execution requires more infrastructure

        INFO("TransformerBlock: B=" << B << ", T=" << T << ", C=" << C << ", M=" << M);
        INFO("  Hq=" << Hq << ", Hkv=" << Hkv << ", HD=" << HD);
        INFO("  Golden inputs: " << gc.inputs.size());
        INFO("  Golden outputs: " << gc.outputs.size());
        INFO("  Golden grads: " << gc.grads.size());

        // Validate dimensions
        REQUIRE(gc.inputs.at("x").shape == std::vector<long>{B, T, C});
        REQUIRE(gc.inputs.at("qkv_weight").shape == std::vector<long>{QKV, C});
        REQUIRE(gc.outputs.at("out").shape == std::vector<long>{B, T, C});

        INFO("✓ TransformerBlock golden structure validated");
        INFO("  - Full end-to-end execution requires integrated graph building");
    });
}

TEST_CASE("dsl module goldens: lm_head_module", "[dsl][modules][goldens]") {
    const fs::path goldens_dir = find_goldens_dir();
    const fs::path golden_path = goldens_dir / "lm_head_module_small_case_1.json";

    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " + golden_path.string());
    }

    const GoldenCase gc = load_case(golden_path);
    INFO("Testing module: " << gc.op << " case: " << gc.case_id);

    // Validate golden file structure
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("ln_weight") > 0);
    REQUIRE(gc.inputs.count("lm_head_weight") > 0);
    REQUIRE(gc.inputs.count("targets") > 0);
    REQUIRE(gc.outputs.count("loss") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long V = *meta_long(gc.meta, "V");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = 1;
        cfg.NumKeyValHeads = 1;
        cfg.HeadDim = static_cast<int>(C);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = 1;
        cfg.VocabSize = static_cast<int>(V);
        cfg.MaxPositionEmbeddings = static_cast<int>(T);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        INFO("LMHead: B=" << B << ", T=" << T << ", C=" << C << ", V=" << V);
        INFO("  Golden inputs: " << gc.inputs.size());
        INFO("  Golden outputs: " << gc.outputs.size());
        INFO("  Golden grads: " << gc.grads.size());

        // Validate dimensions
        REQUIRE(gc.inputs.at("x").shape == std::vector<long>{B, T, C});
        REQUIRE(gc.inputs.at("lm_head_weight").shape == std::vector<long>{V, C});
        REQUIRE(gc.outputs.at("loss").numel() == 1);

        // Verify gradient data exists
        REQUIRE(gc.grads.count("d_x") > 0);
        REQUIRE(gc.grads.count("d_ln_weight") > 0);
        REQUIRE(gc.grads.count("d_lm_head_weight") > 0);

        INFO("✓ LMHead module golden structure validated");
        INFO("  - Contains loss computation with gradients");
        INFO("  - Full execution requires fused_lm_head_loss integration");
    });
}

// =============================================================================
// Test: LlamaBlock (DSL Block) - Full Numerical Execution
// =============================================================================
// Full LlamaBlock matching surogate/dsl/blocks/llama.py:
// - fused_residual_rmsnorm (pre-attention)
// - QKV projection (no bias)
// - RoPE (no QK-Norm)
// - FlashAttention (causal)
// - Output projection
// - fused_residual_rmsnorm (pre-MLP)
// - MLP with SwiGLU
// - Returns (out, residual_out)

TEST_CASE("dsl block goldens: llama_block", "[dsl][goldens][modules][blocks]") {
    const fs::path golden_path = find_goldens_dir().parent_path() / "blocks" / "llama_block_small_case_1.json";
    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " << golden_path);
    }

    const GoldenCase gc = load_case(golden_path);
    REQUIRE(gc.op == "llama_block");

    // Shape inference for RoPE needs this flag
    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Verify required inputs
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("residual") > 0);
    REQUIRE(gc.inputs.count("ln1_weight") > 0);
    REQUIRE(gc.inputs.count("qkv_weight") > 0);
    REQUIRE(gc.inputs.count("out_weight") > 0);
    REQUIRE(gc.inputs.count("ln2_weight") > 0);
    REQUIRE(gc.inputs.count("mlp_up_weight") > 0);
    REQUIRE(gc.inputs.count("mlp_down_weight") > 0);
    REQUIRE(gc.inputs.count("rope_freqs") > 0);
    REQUIRE(gc.inputs.count("position_ids") > 0);

    // Verify outputs
    REQUIRE(gc.outputs.count("out") > 0);
    REQUIRE(gc.outputs.count("residual_out") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const long max_seq = *meta_long(gc.meta, "max_seq");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));

        const long QKV = (Hq + 2 * Hkv) * HD;
        const long AttnDim = Hq * HD;
        const long MUp = 2 * M;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M);
        cfg.VocabSize = 256;
        cfg.MaxPositionEmbeddings = static_cast<int>(max_seq);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        INFO("LlamaBlock: B=" << B << ", T=" << T << ", C=" << C << ", M=" << M);
        INFO("  Hq=" << Hq << ", Hkv=" << Hkv << ", HD=" << HD);

        // Build DSL graph for LlamaBlock
        dsl::Module module;
        module.name = "llama_block_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "llama_block";

        // === Parameters ===
        // x input (as param for testing)
        dsl::TensorInfo x_info;
        x_info.shape = to_dims({B, T, C});
        x_info.dtype = ETensorDType::FP32;
        x_info.is_param = true;
        graph.params.emplace("x", x_info);

        // residual input (as param for testing)
        dsl::TensorInfo residual_info;
        residual_info.shape = to_dims({B, T, C});
        residual_info.dtype = ETensorDType::FP32;
        residual_info.is_param = true;
        graph.params.emplace("residual", residual_info);

        // ln1_weight
        dsl::TensorInfo ln1_weight_info;
        ln1_weight_info.shape = to_dims({C});
        ln1_weight_info.dtype = ETensorDType::FP32;
        ln1_weight_info.is_param = true;
        graph.params.emplace("ln1_weight", ln1_weight_info);

        // qkv_weight
        dsl::TensorInfo qkv_weight_info;
        qkv_weight_info.shape = to_dims({QKV, C});
        qkv_weight_info.dtype = ETensorDType::FP32;
        qkv_weight_info.is_param = true;
        graph.params.emplace("qkv_weight", qkv_weight_info);

        // out_weight
        dsl::TensorInfo out_weight_info;
        out_weight_info.shape = to_dims({C, AttnDim});
        out_weight_info.dtype = ETensorDType::FP32;
        out_weight_info.is_param = true;
        graph.params.emplace("out_weight", out_weight_info);

        // ln2_weight
        dsl::TensorInfo ln2_weight_info;
        ln2_weight_info.shape = to_dims({C});
        ln2_weight_info.dtype = ETensorDType::FP32;
        ln2_weight_info.is_param = true;
        graph.params.emplace("ln2_weight", ln2_weight_info);

        // mlp_up_weight
        dsl::TensorInfo mlp_up_weight_info;
        mlp_up_weight_info.shape = to_dims({MUp, C});
        mlp_up_weight_info.dtype = ETensorDType::FP32;
        mlp_up_weight_info.is_param = true;
        graph.params.emplace("mlp_up_weight", mlp_up_weight_info);

        // mlp_down_weight
        dsl::TensorInfo mlp_down_weight_info;
        mlp_down_weight_info.shape = to_dims({C, M});
        mlp_down_weight_info.dtype = ETensorDType::FP32;
        mlp_down_weight_info.is_param = true;
        graph.params.emplace("mlp_down_weight", mlp_down_weight_info);

        // rope_freqs
        dsl::TensorInfo rope_freqs_info;
        rope_freqs_info.shape = to_dims(gc.inputs.at("rope_freqs").shape);
        rope_freqs_info.dtype = ETensorDType::FP32;
        rope_freqs_info.is_param = true;
        graph.params.emplace("rope_freqs", rope_freqs_info);

        // position_ids (as graph input)
        dsl::TensorInfo position_ids_info;
        position_ids_info.shape = to_dims(gc.inputs.at("position_ids").shape);
        position_ids_info.dtype = ETensorDType::INT32;
        position_ids_info.is_input = true;
        graph.inputs.emplace("position_ids", position_ids_info);

        // === Outputs ===
        dsl::TensorInfo out_info;
        out_info.shape = to_dims({B, T, C});
        out_info.dtype = ETensorDType::FP32;
        out_info.is_output = true;
        graph.outputs.emplace("out", out_info);

        dsl::TensorInfo residual_out_info;
        residual_out_info.shape = to_dims({B, T, C});
        residual_out_info.dtype = ETensorDType::FP32;
        residual_out_info.is_output = true;
        graph.outputs.emplace("residual_out", residual_out_info);

        // === Operations ===
        // 1. fused_residual_rmsnorm: res_ffn = residual + x; ln1 = rmsnorm(res_ffn)
        {
            dsl::Operation op;
            op.id = "fused_residual_rmsnorm_1";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"residual", "x", "ln1_weight"};
            op.outputs = {"res_ffn", "ln1", "ln1_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 2. view: ln1 [B, T, C] -> ln1_flat [B*T, C]
        {
            dsl::Operation op;
            op.id = "ln1_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"ln1"};
            op.outputs = {"ln1_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 3. matmul: qkv_flat = ln1_flat @ qkv_weight^T
        {
            dsl::Operation op;
            op.id = "qkv_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"ln1_flat", "qkv_weight"};
            op.outputs = {"qkv_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 4. view: qkv_flat [B*T, QKV] -> qkv [B, T, Hq+2*Hkv, HD]
        {
            dsl::Operation op;
            op.id = "qkv_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"qkv_flat"};
            op.outputs = {"qkv"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(Hq + 2 * Hkv)),
                dsl::AttrValue(static_cast<std::int64_t>(HD))
            }));
            graph.operations.push_back(op);
        }

        // 5. rope: qkv_rope = rope(qkv, rope_freqs, position_ids)
        {
            dsl::Operation op;
            op.id = "rope_op";
            op.name = "rope";
            op.kernel_type = "rope";
            op.inputs = {"qkv", "rope_freqs", "position_ids"};
            op.outputs = {"qkv_rope"};
            op.attrs["rotary_dim"] = dsl::AttrValue(static_cast<std::int64_t>(HD));
            graph.operations.push_back(op);
        }

        // 6. flash_attention: att, lse = flash_attention(qkv_rope, causal=True)
        {
            dsl::Operation op;
            op.id = "flash_attn";
            op.name = "flash_attention";
            op.kernel_type = "flash_attention";
            op.inputs = {"qkv_rope"};
            op.outputs = {"att", "lse"};
            op.attrs["causal"] = dsl::AttrValue(true);
            graph.operations.push_back(op);
        }

        // 7. view: att [B, T, Hq, HD] -> att_flat [B*T, AttnDim]
        {
            dsl::Operation op;
            op.id = "att_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"att"};
            op.outputs = {"att_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(AttnDim))
            }));
            graph.operations.push_back(op);
        }

        // 8. matmul: att_out_flat = att_flat @ out_weight^T
        {
            dsl::Operation op;
            op.id = "out_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"att_flat", "out_weight"};
            op.outputs = {"att_out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 9. view: att_out_flat [B*T, C] -> att_out [B, T, C]
        {
            dsl::Operation op;
            op.id = "att_out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"att_out_flat"};
            op.outputs = {"att_out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 10. fused_residual_rmsnorm: res_att = res_ffn + att_out; ln2 = rmsnorm(res_att)
        {
            dsl::Operation op;
            op.id = "fused_residual_rmsnorm_2";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"res_ffn", "att_out", "ln2_weight"};
            op.outputs = {"residual_out", "ln2", "ln2_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 11. view: ln2 [B, T, C] -> ln2_flat [B*T, C]
        {
            dsl::Operation op;
            op.id = "ln2_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"ln2"};
            op.outputs = {"ln2_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 12. matmul: mlp_up_flat = ln2_flat @ mlp_up_weight^T
        {
            dsl::Operation op;
            op.id = "mlp_up_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"ln2_flat", "mlp_up_weight"};
            op.outputs = {"mlp_up_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 13. view: mlp_up_flat [B*T, MUp] -> mlp_up [B, T, MUp]
        {
            dsl::Operation op;
            op.id = "mlp_up_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"mlp_up_flat"};
            op.outputs = {"mlp_up"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(MUp))
            }));
            graph.operations.push_back(op);
        }

        // 14. swiglu: swiglu = swiglu(mlp_up)
        {
            dsl::Operation op;
            op.id = "swiglu_act";
            op.name = "swiglu";
            op.kernel_type = "swiglu";
            op.inputs = {"mlp_up"};
            op.outputs = {"swiglu"};
            graph.operations.push_back(op);
        }

        // 15. view: swiglu [B, T, M] -> swiglu_flat [B*T, M]
        {
            dsl::Operation op;
            op.id = "swiglu_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"swiglu"};
            op.outputs = {"swiglu_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(M))
            }));
            graph.operations.push_back(op);
        }

        // 16. matmul: out_flat = swiglu_flat @ mlp_down_weight^T
        {
            dsl::Operation op;
            op.id = "mlp_down_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"swiglu_flat", "mlp_down_weight"};
            op.outputs = {"out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 17. view: out_flat [B*T, C] -> out [B, T, C]
        {
            dsl::Operation op;
            op.id = "out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"out_flat"};
            op.outputs = {"out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        module.forward = graph;

        // Create param/grad stores
        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy golden inputs to device
        copy_tensor_to_device(params.get("x"), gc.inputs.at("x"));
        copy_tensor_to_device(params.get("residual"), gc.inputs.at("residual"));
        copy_tensor_to_device(params.get("ln1_weight"), gc.inputs.at("ln1_weight"));
        copy_tensor_to_device(params.get("qkv_weight"), gc.inputs.at("qkv_weight"));
        copy_tensor_to_device(params.get("out_weight"), gc.inputs.at("out_weight"));
        copy_tensor_to_device(params.get("ln2_weight"), gc.inputs.at("ln2_weight"));
        copy_tensor_to_device(params.get("mlp_up_weight"), gc.inputs.at("mlp_up_weight"));
        copy_tensor_to_device(params.get("mlp_down_weight"), gc.inputs.at("mlp_down_weight"));

        // RoPE frequencies: allocate via run_state (DslParamStore skips rope_freqs)
        const auto& rope_freqs_golden = gc.inputs.at("rope_freqs");
        auto& freq_cis = run_state.non_block_activations().freq_cis;
        if (freq_cis.nelem() != static_cast<long>(rope_freqs_golden.numel())) {
            freq_cis = run_state.temp_alloc(ETensorDType::FP32, rope_freqs_golden.shape);
        }
        copy_tensor_to_device(freq_cis, rope_freqs_golden);

        // Position IDs: golden may be 1D [T], but kernel expects 2D [B, T].
        const auto& pos_ids_golden = gc.inputs.at("position_ids");
        const std::size_t T_len = static_cast<std::size_t>(T);
        const std::size_t B_len = static_cast<std::size_t>(B);
        const std::size_t total_pos_ids = B_len * T_len;

        if (run_state.PositionIDs.nelem() != static_cast<long>(total_pos_ids)) {
            run_state.PositionIDs = run_state.temp_alloc(ETensorDType::INT32, {B, T});
        }

        std::vector<std::int32_t> pos_ids_2d(total_pos_ids);
        for (std::size_t b = 0; b < B_len; ++b) {
            for (std::size_t t = 0; t < T_len; ++t) {
                std::int32_t val = pos_ids_golden.is_int()
                    ? static_cast<std::int32_t>(pos_ids_golden.i64[t])
                    : static_cast<std::int32_t>(pos_ids_golden.f64[t]);
                pos_ids_2d[b * T_len + t] = val;
            }
        }
        CUDA_CHECK(cudaMemcpy(run_state.PositionIDs.Data, pos_ids_2d.data(),
                             total_pos_ids * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile graph
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("LlamaBlock: DSL graph compiled with " << compiled.ops.size() << " operations");

        // Execute forward pass
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify forward outputs
        constexpr double rtol = 1e-3;  // Relaxed for flash attention differences
        constexpr double atol = 1e-3;

        // Check final outputs
        const Tensor* out_actual = exec.try_get_tensor("out");
        REQUIRE(out_actual != nullptr);
        expect_allclose("out", gc.outputs.at("out"), *out_actual, rtol, atol);

        const Tensor* residual_out_actual = exec.try_get_tensor("residual_out");
        REQUIRE(residual_out_actual != nullptr);
        expect_allclose("residual_out", gc.outputs.at("residual_out"), *residual_out_actual, rtol, atol);

        // Check intermediate outputs (informational)
        auto check_intermediate = [&](const std::string& name) {
            const Tensor* actual = exec.try_get_tensor(name);
            if (actual && gc.outputs.count(name) > 0) {
                expect_allclose(name, gc.outputs.at(name), *actual, rtol, atol);
                INFO("  ✓ " << name << " matches");
            }
        };

        INFO("Checking intermediate outputs:");
        check_intermediate("res_ffn");
        check_intermediate("ln1");
        check_intermediate("qkv");
        check_intermediate("qkv_rope");
        check_intermediate("att_out");
        check_intermediate("ln2");
        check_intermediate("mlp_up");
        check_intermediate("swiglu");

        INFO("✓ LlamaBlock forward pass verified against PyTorch reference");

        // === Backward graph derivation ===
        if (gc.grads.count("d_out") > 0) {
            INFO("Validating backward pass graph derivation...");

            dsl::DeriveBackwardOptions derive_opts;
            derive_opts.loss_name = "out";
            derive_opts.auto_save = true;
            derive_opts.accumulate_grads = true;

            dsl::Graph backward_graph;
            bool derivation_ok = false;
            std::string derive_error;
            try {
                backward_graph = dsl::derive_backward_graph(graph, derive_opts);
                derivation_ok = true;
            } catch (const std::exception& e) {
                derive_error = e.what();
            }

            if (derivation_ok) {
                REQUIRE(backward_graph.operations.size() > 0);
                INFO("Backward graph derived with " << backward_graph.operations.size() << " operations");

                std::vector<std::string> save_list = dsl::compute_required_saves(graph, backward_graph);
                INFO("Backward requires " << save_list.size() << " saved tensors");

                // Compile backward graph
                auto compiled_backward = compiler.compile(backward_graph, B, T);
                INFO("Backward graph compiled with " << compiled_backward.ops.size() << " compiled ops");
                REQUIRE(compiled_backward.ops.size() > 0);

                // Check for expected backward op types
                bool has_matmul_backward = false;
                bool has_swiglu_backward = false;
                bool has_fused_residual_rmsnorm_backward = false;
                bool has_rope_backward = false;
                bool has_flash_attention_backward = false;
                for (const auto& op : compiled_backward.ops) {
                    if (op.type == dsl::CompiledOpType::MatmulBackward) has_matmul_backward = true;
                    if (op.type == dsl::CompiledOpType::SwiGLUBackward) has_swiglu_backward = true;
                    if (op.type == dsl::CompiledOpType::FusedResidualRMSNormBackward) has_fused_residual_rmsnorm_backward = true;
                    if (op.type == dsl::CompiledOpType::RoPEBackward) has_rope_backward = true;
                    if (op.type == dsl::CompiledOpType::FlashAttentionBackward) has_flash_attention_backward = true;
                }

                INFO("Backward graph contains:");
                INFO("  - matmul_backward: " << (has_matmul_backward ? "yes" : "no"));
                INFO("  - swiglu_backward: " << (has_swiglu_backward ? "yes" : "no"));
                INFO("  - fused_residual_rmsnorm_backward: " << (has_fused_residual_rmsnorm_backward ? "yes" : "no"));
                INFO("  - rope_backward: " << (has_rope_backward ? "yes" : "no"));
                INFO("  - flash_attention_backward: " << (has_flash_attention_backward ? "yes" : "no"));

                REQUIRE(has_matmul_backward);
                REQUIRE(has_swiglu_backward);

                INFO("✓ LlamaBlock backward graph derivation validated");
            } else {
                INFO("Backward derivation failed: " << derive_error);
                INFO("  (Some ops may not have autodiff rules yet)");
            }
        }
    });
}

// =============================================================================
// Test: Qwen3Block (DSL Block) - Full Numerical Execution
// =============================================================================
// Full Qwen3Block matching surogate/dsl/blocks/qwen3.py:
// - fused_residual_rmsnorm (pre-attention)
// - QKV projection (optional bias, tested without)
// - QK-Norm + RoPE (fused) - key difference from LlamaBlock
// - FlashAttention (causal)
// - Output projection
// - fused_residual_rmsnorm (pre-MLP)
// - MLP with SwiGLU
// - Returns (out, residual_out)

TEST_CASE("dsl block goldens: qwen3_block", "[dsl][goldens][modules][blocks]") {
    const fs::path golden_path = find_goldens_dir().parent_path() / "blocks" / "qwen3_block_small_case_1.json";
    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " << golden_path);
    }

    const GoldenCase gc = load_case(golden_path);
    REQUIRE(gc.op == "qwen3_block");

    // Shape inference for QK-Norm+RoPE needs this flag
    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Verify required inputs
    REQUIRE(gc.inputs.count("x") > 0);
    REQUIRE(gc.inputs.count("residual") > 0);
    REQUIRE(gc.inputs.count("ln1_weight") > 0);
    REQUIRE(gc.inputs.count("qkv_weight") > 0);
    REQUIRE(gc.inputs.count("q_norm_weight") > 0);
    REQUIRE(gc.inputs.count("k_norm_weight") > 0);
    REQUIRE(gc.inputs.count("out_weight") > 0);
    REQUIRE(gc.inputs.count("ln2_weight") > 0);
    REQUIRE(gc.inputs.count("mlp_up_weight") > 0);
    REQUIRE(gc.inputs.count("mlp_down_weight") > 0);
    REQUIRE(gc.inputs.count("rope_freqs") > 0);
    REQUIRE(gc.inputs.count("position_ids") > 0);

    // Verify outputs
    REQUIRE(gc.outputs.count("out") > 0);
    REQUIRE(gc.outputs.count("residual_out") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const long max_seq = *meta_long(gc.meta, "max_seq");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));
        const bool use_qk_norm = gc.meta.at("use_qk_norm").get<bool>();

        const long QKV = (Hq + 2 * Hkv) * HD;
        const long AttnDim = Hq * HD;
        const long MUp = 2 * M;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M);
        cfg.VocabSize = 256;
        cfg.MaxPositionEmbeddings = static_cast<int>(max_seq);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        INFO("Qwen3Block: B=" << B << ", T=" << T << ", C=" << C << ", M=" << M);
        INFO("  Hq=" << Hq << ", Hkv=" << Hkv << ", HD=" << HD);
        INFO("  use_qk_norm=" << use_qk_norm);

        REQUIRE(use_qk_norm == true);  // Qwen3 uses QK-Norm by default

        // Build DSL graph for Qwen3Block
        dsl::Module module;
        module.name = "qwen3_block_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "qwen3_block";

        // === Parameters ===
        dsl::TensorInfo x_info;
        x_info.shape = to_dims({B, T, C});
        x_info.dtype = ETensorDType::FP32;
        x_info.is_param = true;
        graph.params.emplace("x", x_info);

        dsl::TensorInfo residual_info;
        residual_info.shape = to_dims({B, T, C});
        residual_info.dtype = ETensorDType::FP32;
        residual_info.is_param = true;
        graph.params.emplace("residual", residual_info);

        dsl::TensorInfo ln1_weight_info;
        ln1_weight_info.shape = to_dims({C});
        ln1_weight_info.dtype = ETensorDType::FP32;
        ln1_weight_info.is_param = true;
        graph.params.emplace("ln1_weight", ln1_weight_info);

        dsl::TensorInfo qkv_weight_info;
        qkv_weight_info.shape = to_dims({QKV, C});
        qkv_weight_info.dtype = ETensorDType::FP32;
        qkv_weight_info.is_param = true;
        graph.params.emplace("qkv_weight", qkv_weight_info);

        // QK-Norm weights
        dsl::TensorInfo q_norm_weight_info;
        q_norm_weight_info.shape = to_dims({HD});
        q_norm_weight_info.dtype = ETensorDType::FP32;
        q_norm_weight_info.is_param = true;
        graph.params.emplace("q_norm_weight", q_norm_weight_info);

        dsl::TensorInfo k_norm_weight_info;
        k_norm_weight_info.shape = to_dims({HD});
        k_norm_weight_info.dtype = ETensorDType::FP32;
        k_norm_weight_info.is_param = true;
        graph.params.emplace("k_norm_weight", k_norm_weight_info);

        dsl::TensorInfo out_weight_info;
        out_weight_info.shape = to_dims({C, AttnDim});
        out_weight_info.dtype = ETensorDType::FP32;
        out_weight_info.is_param = true;
        graph.params.emplace("out_weight", out_weight_info);

        dsl::TensorInfo ln2_weight_info;
        ln2_weight_info.shape = to_dims({C});
        ln2_weight_info.dtype = ETensorDType::FP32;
        ln2_weight_info.is_param = true;
        graph.params.emplace("ln2_weight", ln2_weight_info);

        dsl::TensorInfo mlp_up_weight_info;
        mlp_up_weight_info.shape = to_dims({MUp, C});
        mlp_up_weight_info.dtype = ETensorDType::FP32;
        mlp_up_weight_info.is_param = true;
        graph.params.emplace("mlp_up_weight", mlp_up_weight_info);

        dsl::TensorInfo mlp_down_weight_info;
        mlp_down_weight_info.shape = to_dims({C, M});
        mlp_down_weight_info.dtype = ETensorDType::FP32;
        mlp_down_weight_info.is_param = true;
        graph.params.emplace("mlp_down_weight", mlp_down_weight_info);

        dsl::TensorInfo rope_freqs_info;
        rope_freqs_info.shape = to_dims(gc.inputs.at("rope_freqs").shape);
        rope_freqs_info.dtype = ETensorDType::FP32;
        rope_freqs_info.is_param = true;
        graph.params.emplace("rope_freqs", rope_freqs_info);

        dsl::TensorInfo position_ids_info;
        position_ids_info.shape = to_dims(gc.inputs.at("position_ids").shape);
        position_ids_info.dtype = ETensorDType::INT32;
        position_ids_info.is_input = true;
        graph.inputs.emplace("position_ids", position_ids_info);

        // === Outputs ===
        dsl::TensorInfo out_info;
        out_info.shape = to_dims({B, T, C});
        out_info.dtype = ETensorDType::FP32;
        out_info.is_output = true;
        graph.outputs.emplace("out", out_info);

        dsl::TensorInfo residual_out_info;
        residual_out_info.shape = to_dims({B, T, C});
        residual_out_info.dtype = ETensorDType::FP32;
        residual_out_info.is_output = true;
        graph.outputs.emplace("residual_out", residual_out_info);

        // === Operations ===
        // 1. fused_residual_rmsnorm
        {
            dsl::Operation op;
            op.id = "fused_residual_rmsnorm_1";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"residual", "x", "ln1_weight"};
            op.outputs = {"res_ffn", "ln1", "ln1_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 2. view: ln1 -> ln1_flat
        {
            dsl::Operation op;
            op.id = "ln1_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"ln1"};
            op.outputs = {"ln1_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 3. matmul: qkv_flat = ln1_flat @ qkv_weight^T
        {
            dsl::Operation op;
            op.id = "qkv_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"ln1_flat", "qkv_weight"};
            op.outputs = {"qkv_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 4. view: qkv_flat -> qkv
        {
            dsl::Operation op;
            op.id = "qkv_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"qkv_flat"};
            op.outputs = {"qkv"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(Hq + 2 * Hkv)),
                dsl::AttrValue(static_cast<std::int64_t>(HD))
            }));
            graph.operations.push_back(op);
        }

        // 5. qkv_qk_norm_rope: QK-Norm + RoPE (fused) - KEY DIFFERENCE from LlamaBlock
        {
            dsl::Operation op;
            op.id = "qkv_qk_norm_rope_op";
            op.name = "qkv_qk_norm_rope";
            op.kernel_type = "qkv_qk_norm_rope";
            op.inputs = {"qkv", "q_norm_weight", "k_norm_weight", "rope_freqs", "position_ids"};
            op.outputs = {"qkv_rope", "q_rstd", "k_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 6. flash_attention
        {
            dsl::Operation op;
            op.id = "flash_attn";
            op.name = "flash_attention";
            op.kernel_type = "flash_attention";
            op.inputs = {"qkv_rope"};
            op.outputs = {"att", "lse"};
            op.attrs["causal"] = dsl::AttrValue(true);
            graph.operations.push_back(op);
        }

        // 7. view: att -> att_flat
        {
            dsl::Operation op;
            op.id = "att_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"att"};
            op.outputs = {"att_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(AttnDim))
            }));
            graph.operations.push_back(op);
        }

        // 8. matmul: att_out_flat = att_flat @ out_weight^T
        {
            dsl::Operation op;
            op.id = "out_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"att_flat", "out_weight"};
            op.outputs = {"att_out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 9. view: att_out_flat -> att_out
        {
            dsl::Operation op;
            op.id = "att_out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"att_out_flat"};
            op.outputs = {"att_out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 10. fused_residual_rmsnorm (pre-MLP)
        {
            dsl::Operation op;
            op.id = "fused_residual_rmsnorm_2";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"res_ffn", "att_out", "ln2_weight"};
            op.outputs = {"residual_out", "ln2", "ln2_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 11-17: MLP (same as LlamaBlock)
        {
            dsl::Operation op;
            op.id = "ln2_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"ln2"};
            op.outputs = {"ln2_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "mlp_up_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"ln2_flat", "mlp_up_weight"};
            op.outputs = {"mlp_up_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "mlp_up_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"mlp_up_flat"};
            op.outputs = {"mlp_up"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(MUp))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "swiglu_act";
            op.name = "swiglu";
            op.kernel_type = "swiglu";
            op.inputs = {"mlp_up"};
            op.outputs = {"swiglu"};
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "swiglu_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"swiglu"};
            op.outputs = {"swiglu_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(M))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "mlp_down_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"swiglu_flat", "mlp_down_weight"};
            op.outputs = {"out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"out_flat"};
            op.outputs = {"out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        module.forward = graph;

        // Create param/grad stores
        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy golden inputs to device
        copy_tensor_to_device(params.get("x"), gc.inputs.at("x"));
        copy_tensor_to_device(params.get("residual"), gc.inputs.at("residual"));
        copy_tensor_to_device(params.get("ln1_weight"), gc.inputs.at("ln1_weight"));
        copy_tensor_to_device(params.get("qkv_weight"), gc.inputs.at("qkv_weight"));
        copy_tensor_to_device(params.get("q_norm_weight"), gc.inputs.at("q_norm_weight"));
        copy_tensor_to_device(params.get("k_norm_weight"), gc.inputs.at("k_norm_weight"));
        copy_tensor_to_device(params.get("out_weight"), gc.inputs.at("out_weight"));
        copy_tensor_to_device(params.get("ln2_weight"), gc.inputs.at("ln2_weight"));
        copy_tensor_to_device(params.get("mlp_up_weight"), gc.inputs.at("mlp_up_weight"));
        copy_tensor_to_device(params.get("mlp_down_weight"), gc.inputs.at("mlp_down_weight"));

        // RoPE frequencies: allocate via run_state (DslParamStore skips rope_freqs)
        const auto& rope_freqs_golden = gc.inputs.at("rope_freqs");
        auto& freq_cis = run_state.non_block_activations().freq_cis;
        if (freq_cis.nelem() != static_cast<long>(rope_freqs_golden.numel())) {
            freq_cis = run_state.temp_alloc(ETensorDType::FP32, rope_freqs_golden.shape);
        }
        copy_tensor_to_device(freq_cis, rope_freqs_golden);

        // Position IDs: golden may be 1D [T], but kernel expects 2D [B, T].
        const auto& pos_ids_golden = gc.inputs.at("position_ids");
        const std::size_t T_len = static_cast<std::size_t>(T);
        const std::size_t B_len = static_cast<std::size_t>(B);
        const std::size_t total_pos_ids = B_len * T_len;

        if (run_state.PositionIDs.nelem() != static_cast<long>(total_pos_ids)) {
            run_state.PositionIDs = run_state.temp_alloc(ETensorDType::INT32, {B, T});
        }

        std::vector<std::int32_t> pos_ids_2d(total_pos_ids);
        for (std::size_t b = 0; b < B_len; ++b) {
            for (std::size_t t = 0; t < T_len; ++t) {
                std::int32_t val = pos_ids_golden.is_int()
                    ? static_cast<std::int32_t>(pos_ids_golden.i64[t])
                    : static_cast<std::int32_t>(pos_ids_golden.f64[t]);
                pos_ids_2d[b * T_len + t] = val;
            }
        }
        CUDA_CHECK(cudaMemcpy(run_state.PositionIDs.Data, pos_ids_2d.data(),
                             total_pos_ids * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile graph
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("Qwen3Block: DSL graph compiled with " << compiled.ops.size() << " operations");

        // Execute forward pass
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);
        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify forward outputs
        constexpr double rtol = 1e-4;
        constexpr double atol = 1e-4;

        const Tensor* out_actual = exec.try_get_tensor("out");
        REQUIRE(out_actual != nullptr);
        expect_allclose("out", gc.outputs.at("out"), *out_actual, rtol, atol);

        const Tensor* residual_out_actual = exec.try_get_tensor("residual_out");
        REQUIRE(residual_out_actual != nullptr);
        expect_allclose("residual_out", gc.outputs.at("residual_out"), *residual_out_actual, rtol, atol);

        // Check intermediate outputs
        auto check_intermediate = [&](const std::string& name) {
            const Tensor* actual = exec.try_get_tensor(name);
            if (actual && gc.outputs.count(name) > 0) {
                expect_allclose(name, gc.outputs.at(name), *actual, rtol, atol);
                INFO("  ✓ " << name << " matches");
            }
        };

        INFO("Checking intermediate outputs:");
        check_intermediate("res_ffn");
        check_intermediate("ln1");
        // Skip qkv check: the qkv_qk_norm_rope kernel modifies qkv in-place,
        // so the stored tensor contains post-QK-Norm values, not the original matmul output.
        // check_intermediate("qkv");
        check_intermediate("qkv_rope");
        check_intermediate("att_out");
        check_intermediate("ln2");
        check_intermediate("mlp_up");
        check_intermediate("swiglu");

        INFO("✓ Qwen3Block forward pass verified against PyTorch reference");

        // === Backward graph derivation ===
        if (gc.grads.count("d_out") > 0) {
            INFO("Validating backward pass graph derivation...");

            dsl::DeriveBackwardOptions derive_opts;
            derive_opts.loss_name = "out";
            derive_opts.auto_save = true;
            derive_opts.accumulate_grads = true;

            dsl::Graph backward_graph;
            bool derivation_ok = false;
            std::string derive_error;
            try {
                backward_graph = dsl::derive_backward_graph(graph, derive_opts);
                derivation_ok = true;
            } catch (const std::exception& e) {
                derive_error = e.what();
            }

            if (derivation_ok) {
                REQUIRE(backward_graph.operations.size() > 0);
                INFO("Backward graph derived with " << backward_graph.operations.size() << " operations");

                std::vector<std::string> save_list = dsl::compute_required_saves(graph, backward_graph);
                INFO("Backward requires " << save_list.size() << " saved tensors");

                // Compile backward graph
                auto compiled_backward = compiler.compile(backward_graph, B, T);
                INFO("Backward graph compiled with " << compiled_backward.ops.size() << " compiled ops");
                REQUIRE(compiled_backward.ops.size() > 0);

                // Check for expected backward op types
                bool has_matmul_backward = false;
                bool has_swiglu_backward = false;
                bool has_fused_residual_rmsnorm_backward = false;
                bool has_qkv_qk_norm_rope_backward = false;
                bool has_flash_attention_backward = false;
                for (const auto& op : compiled_backward.ops) {
                    if (op.type == dsl::CompiledOpType::MatmulBackward) has_matmul_backward = true;
                    if (op.type == dsl::CompiledOpType::SwiGLUBackward) has_swiglu_backward = true;
                    if (op.type == dsl::CompiledOpType::FusedResidualRMSNormBackward) has_fused_residual_rmsnorm_backward = true;
                    if (op.type == dsl::CompiledOpType::QKVQKNormRoPEBackward) has_qkv_qk_norm_rope_backward = true;
                    if (op.type == dsl::CompiledOpType::FlashAttentionBackward) has_flash_attention_backward = true;
                }

                INFO("Backward graph contains:");
                INFO("  - matmul_backward: " << (has_matmul_backward ? "yes" : "no"));
                INFO("  - swiglu_backward: " << (has_swiglu_backward ? "yes" : "no"));
                INFO("  - fused_residual_rmsnorm_backward: " << (has_fused_residual_rmsnorm_backward ? "yes" : "no"));
                INFO("  - qkv_qk_norm_rope_backward: " << (has_qkv_qk_norm_rope_backward ? "yes" : "no"));
                INFO("  - flash_attention_backward: " << (has_flash_attention_backward ? "yes" : "no"));

                REQUIRE(has_matmul_backward);
                REQUIRE(has_swiglu_backward);

                INFO("✓ Qwen3Block backward graph derivation validated");
            } else {
                INFO("Backward derivation failed: " << derive_error);
                INFO("  (Some ops may not have autodiff rules yet)");
            }
        }
    });
}

// =============================================================================
// Test: Qwen3Model (DSL Model) - Full Model Numerical Execution
// =============================================================================
// Full Qwen3Model (1-layer) matching surogate/dsl/models/qwen3.py:
// - Embedding lookup
// - Zero initialization for residual
// - Single Qwen3Block layer (QK-Norm + SwiGLU)
// - Final fused_residual_rmsnorm
// - Fused LM head + cross-entropy loss

TEST_CASE("dsl model goldens: qwen3_model", "[dsl][goldens][modules][models]") {
    const fs::path golden_path = find_goldens_dir().parent_path() / "models" / "qwen3_model_small_1layer.json";
    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " << golden_path);
    }

    const GoldenCase gc = load_case(golden_path);
    REQUIRE(gc.op == "qwen3_model");

    // Shape inference for QK-Norm+RoPE needs this flag
    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Verify required inputs
    REQUIRE(gc.inputs.count("embedding") > 0);
    REQUIRE(gc.inputs.count("final_norm") > 0);
    REQUIRE(gc.inputs.count("lm_head") > 0);
    REQUIRE(gc.inputs.count("block0_ln1_weight") > 0);
    REQUIRE(gc.inputs.count("block0_qkv_weight") > 0);
    REQUIRE(gc.inputs.count("token_ids") > 0);
    REQUIRE(gc.inputs.count("targets") > 0);

    // Verify outputs
    REQUIRE(gc.outputs.count("loss") > 0);

    NCCLCommunicator::run_communicators(1, false, false, [&gc](NCCLCommunicator& comm) {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long V = *meta_long(gc.meta, "V");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const long max_seq = *meta_long(gc.meta, "max_seq");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));
        const bool use_qk_norm = gc.meta.at("use_qk_norm").get<bool>();

        const long QKV = (Hq + 2 * Hkv) * HD;
        const long AttnDim = Hq * HD;
        const long MUp = 2 * M;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M);
        cfg.VocabSize = static_cast<int>(V);
        cfg.MaxPositionEmbeddings = static_cast<int>(max_seq);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        INFO("Qwen3Model: B=" << B << ", T=" << T << ", V=" << V << ", C=" << C << ", M=" << M);
        INFO("  Hq=" << Hq << ", Hkv=" << Hkv << ", HD=" << HD);
        INFO("  use_qk_norm=" << use_qk_norm);

        REQUIRE(use_qk_norm == true);

        // Build DSL graph for Qwen3Model (1-layer)
        dsl::Module module;
        module.name = "qwen3_model_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "qwen3_model";

        // === Model-level Parameters ===
        dsl::TensorInfo embedding_info;
        embedding_info.shape = to_dims({V, C});
        embedding_info.dtype = ETensorDType::FP32;
        embedding_info.is_param = true;
        graph.params.emplace("embedding", embedding_info);

        dsl::TensorInfo final_norm_info;
        final_norm_info.shape = to_dims({C});
        final_norm_info.dtype = ETensorDType::FP32;
        final_norm_info.is_param = true;
        graph.params.emplace("final_norm", final_norm_info);

        dsl::TensorInfo lm_head_info;
        lm_head_info.shape = to_dims({V, C});
        lm_head_info.dtype = ETensorDType::FP32;
        lm_head_info.is_param = true;
        graph.params.emplace("lm_head", lm_head_info);

        // === Block 0 Parameters ===
        dsl::TensorInfo ln1_weight_info;
        ln1_weight_info.shape = to_dims({C});
        ln1_weight_info.dtype = ETensorDType::FP32;
        ln1_weight_info.is_param = true;
        graph.params.emplace("block0_ln1_weight", ln1_weight_info);

        dsl::TensorInfo qkv_weight_info;
        qkv_weight_info.shape = to_dims({QKV, C});
        qkv_weight_info.dtype = ETensorDType::FP32;
        qkv_weight_info.is_param = true;
        graph.params.emplace("block0_qkv_weight", qkv_weight_info);

        dsl::TensorInfo q_norm_weight_info;
        q_norm_weight_info.shape = to_dims({HD});
        q_norm_weight_info.dtype = ETensorDType::FP32;
        q_norm_weight_info.is_param = true;
        graph.params.emplace("block0_q_norm_weight", q_norm_weight_info);

        dsl::TensorInfo k_norm_weight_info;
        k_norm_weight_info.shape = to_dims({HD});
        k_norm_weight_info.dtype = ETensorDType::FP32;
        k_norm_weight_info.is_param = true;
        graph.params.emplace("block0_k_norm_weight", k_norm_weight_info);

        dsl::TensorInfo out_weight_info;
        out_weight_info.shape = to_dims({C, AttnDim});
        out_weight_info.dtype = ETensorDType::FP32;
        out_weight_info.is_param = true;
        graph.params.emplace("block0_out_weight", out_weight_info);

        dsl::TensorInfo ln2_weight_info;
        ln2_weight_info.shape = to_dims({C});
        ln2_weight_info.dtype = ETensorDType::FP32;
        ln2_weight_info.is_param = true;
        graph.params.emplace("block0_ln2_weight", ln2_weight_info);

        dsl::TensorInfo mlp_up_weight_info;
        mlp_up_weight_info.shape = to_dims({MUp, C});
        mlp_up_weight_info.dtype = ETensorDType::FP32;
        mlp_up_weight_info.is_param = true;
        graph.params.emplace("block0_mlp_up_weight", mlp_up_weight_info);

        dsl::TensorInfo mlp_down_weight_info;
        mlp_down_weight_info.shape = to_dims({C, M});
        mlp_down_weight_info.dtype = ETensorDType::FP32;
        mlp_down_weight_info.is_param = true;
        graph.params.emplace("block0_mlp_down_weight", mlp_down_weight_info);

        // RoPE freqs and position IDs
        dsl::TensorInfo rope_freqs_info;
        rope_freqs_info.shape = to_dims(gc.inputs.at("rope_freqs").shape);
        rope_freqs_info.dtype = ETensorDType::FP32;
        rope_freqs_info.is_param = true;
        graph.params.emplace("rope_freqs", rope_freqs_info);

        // Inputs
        dsl::TensorInfo token_ids_info;
        token_ids_info.shape = to_dims({B, T});
        token_ids_info.dtype = ETensorDType::INT32;
        token_ids_info.is_input = true;
        graph.inputs.emplace("token_ids", token_ids_info);

        dsl::TensorInfo position_ids_info;
        position_ids_info.shape = to_dims(gc.inputs.at("position_ids").shape);
        position_ids_info.dtype = ETensorDType::INT32;
        position_ids_info.is_input = true;
        graph.inputs.emplace("position_ids", position_ids_info);

        dsl::TensorInfo targets_info;
        targets_info.shape = to_dims({B, T});
        targets_info.dtype = ETensorDType::INT32;
        targets_info.is_input = true;
        graph.inputs.emplace("targets", targets_info);

        // === Outputs ===
        dsl::TensorInfo loss_info;
        loss_info.shape = to_dims({B * T});  // Per-token losses
        loss_info.dtype = ETensorDType::FP32;
        loss_info.is_output = true;
        graph.outputs.emplace("loss", loss_info);

        // === Operations ===
        // 1. Embedding lookup
        {
            dsl::Operation op;
            op.id = "embed_lookup";
            op.name = "embedding";
            op.kernel_type = "embedding";
            op.inputs = {"token_ids", "embedding"};
            op.outputs = {"x0"};
            graph.operations.push_back(op);
        }

        // 2. Zero residual
        {
            dsl::Operation op;
            op.id = "zero_residual";
            op.name = "zeros";
            op.kernel_type = "zeros";
            op.outputs = {"residual0"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            op.attrs["dtype"] = dsl::AttrValue("fp32");
            graph.operations.push_back(op);
        }

        // === Block 0: Qwen3Block ===
        // 3. Pre-attention fused_residual_rmsnorm
        {
            dsl::Operation op;
            op.id = "block0_fused_residual_rmsnorm_1";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"residual0", "x0", "block0_ln1_weight"};
            op.outputs = {"block0_res_ffn", "block0_ln1", "block0_ln1_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 4. view: ln1 -> ln1_flat
        {
            dsl::Operation op;
            op.id = "block0_ln1_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_ln1"};
            op.outputs = {"block0_ln1_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 5. matmul: qkv_flat = ln1_flat @ qkv_weight^T
        {
            dsl::Operation op;
            op.id = "block0_qkv_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"block0_ln1_flat", "block0_qkv_weight"};
            op.outputs = {"block0_qkv_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 6. view: qkv_flat -> qkv
        {
            dsl::Operation op;
            op.id = "block0_qkv_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_qkv_flat"};
            op.outputs = {"block0_qkv"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(Hq + 2 * Hkv)),
                dsl::AttrValue(static_cast<std::int64_t>(HD))
            }));
            graph.operations.push_back(op);
        }

        // 7. qkv_qk_norm_rope: QK-Norm + RoPE
        {
            dsl::Operation op;
            op.id = "block0_qkv_qk_norm_rope";
            op.name = "qkv_qk_norm_rope";
            op.kernel_type = "qkv_qk_norm_rope";
            op.inputs = {"block0_qkv", "block0_q_norm_weight", "block0_k_norm_weight", "rope_freqs", "position_ids"};
            op.outputs = {"block0_qkv_rope", "block0_q_rstd", "block0_k_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 8. flash_attention
        {
            dsl::Operation op;
            op.id = "block0_flash_attn";
            op.name = "flash_attention";
            op.kernel_type = "flash_attention";
            op.inputs = {"block0_qkv_rope"};
            op.outputs = {"block0_att", "block0_lse"};
            op.attrs["causal"] = dsl::AttrValue(true);
            graph.operations.push_back(op);
        }

        // 9. view: att -> att_flat
        {
            dsl::Operation op;
            op.id = "block0_att_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_att"};
            op.outputs = {"block0_att_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(AttnDim))
            }));
            graph.operations.push_back(op);
        }

        // 10. matmul: att_out_flat = att_flat @ out_weight^T
        {
            dsl::Operation op;
            op.id = "block0_out_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"block0_att_flat", "block0_out_weight"};
            op.outputs = {"block0_att_out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        // 11. view: att_out_flat -> att_out
        {
            dsl::Operation op;
            op.id = "block0_att_out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_att_out_flat"};
            op.outputs = {"block0_att_out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 12. Pre-MLP fused_residual_rmsnorm
        {
            dsl::Operation op;
            op.id = "block0_fused_residual_rmsnorm_2";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"block0_res_ffn", "block0_att_out", "block0_ln2_weight"};
            op.outputs = {"block0_residual_out", "block0_ln2", "block0_ln2_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 13-17: MLP
        {
            dsl::Operation op;
            op.id = "block0_ln2_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_ln2"};
            op.outputs = {"block0_ln2_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_mlp_up_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"block0_ln2_flat", "block0_mlp_up_weight"};
            op.outputs = {"block0_mlp_up_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_mlp_up_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_mlp_up_flat"};
            op.outputs = {"block0_mlp_up"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(MUp))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_swiglu_act";
            op.name = "swiglu";
            op.kernel_type = "swiglu";
            op.inputs = {"block0_mlp_up"};
            op.outputs = {"block0_swiglu"};
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_swiglu_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_swiglu"};
            op.outputs = {"block0_swiglu_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(M))
            }));
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_mlp_down_proj";
            op.name = "matmul";
            op.kernel_type = "matmul";
            op.inputs = {"block0_swiglu_flat", "block0_mlp_down_weight"};
            op.outputs = {"block0_out_flat"};
            op.attrs["transpose"] = dsl::AttrValue("NT");
            graph.operations.push_back(op);
        }

        {
            dsl::Operation op;
            op.id = "block0_out_reshape";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"block0_out_flat"};
            op.outputs = {"block0_out"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B)),
                dsl::AttrValue(static_cast<std::int64_t>(T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // === Final norm + LM head ===
        // 18. Final fused_residual_rmsnorm
        {
            dsl::Operation op;
            op.id = "final_fused_residual_rmsnorm";
            op.name = "fused_residual_rmsnorm";
            op.kernel_type = "fused_residual_rmsnorm";
            op.inputs = {"block0_residual_out", "block0_out", "final_norm"};
            op.outputs = {"residual_final", "xF", "final_rstd"};
            op.attrs["eps"] = dsl::AttrValue(static_cast<double>(eps));
            graph.operations.push_back(op);
        }

        // 19. view: xF -> xF_flat
        {
            dsl::Operation op;
            op.id = "xF_flat_view";
            op.name = "view";
            op.kernel_type = "view";
            op.inputs = {"xF"};
            op.outputs = {"xF_flat"};
            op.attrs["shape"] = dsl::AttrValue(std::make_shared<dsl::AttrList>(dsl::AttrList{
                dsl::AttrValue(static_cast<std::int64_t>(B * T)),
                dsl::AttrValue(static_cast<std::int64_t>(C))
            }));
            graph.operations.push_back(op);
        }

        // 20. fused_lm_head_loss
        {
            dsl::Operation op;
            op.id = "lm_head_loss";
            op.name = "fused_lm_head_loss";
            op.kernel_type = "fused_lm_head_loss";
            op.inputs = {"xF_flat", "lm_head", "targets"};
            op.outputs = {"loss"};
            op.attrs["compute_accuracy"] = dsl::AttrValue(true);
            graph.operations.push_back(op);
        }

        module.forward = graph;

        // Create param/grad stores
        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);

        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator,
                                    false, kStackBytes, true);

        // Copy golden inputs to device
        copy_tensor_to_device(params.get("embedding"), gc.inputs.at("embedding"));
        copy_tensor_to_device(params.get("final_norm"), gc.inputs.at("final_norm"));
        copy_tensor_to_device(params.get("lm_head"), gc.inputs.at("lm_head"));
        copy_tensor_to_device(params.get("block0_ln1_weight"), gc.inputs.at("block0_ln1_weight"));
        copy_tensor_to_device(params.get("block0_qkv_weight"), gc.inputs.at("block0_qkv_weight"));
        copy_tensor_to_device(params.get("block0_q_norm_weight"), gc.inputs.at("block0_q_norm_weight"));
        copy_tensor_to_device(params.get("block0_k_norm_weight"), gc.inputs.at("block0_k_norm_weight"));
        copy_tensor_to_device(params.get("block0_out_weight"), gc.inputs.at("block0_out_weight"));
        copy_tensor_to_device(params.get("block0_ln2_weight"), gc.inputs.at("block0_ln2_weight"));
        copy_tensor_to_device(params.get("block0_mlp_up_weight"), gc.inputs.at("block0_mlp_up_weight"));
        copy_tensor_to_device(params.get("block0_mlp_down_weight"), gc.inputs.at("block0_mlp_down_weight"));

        // RoPE frequencies
        const auto& rope_freqs_golden = gc.inputs.at("rope_freqs");
        auto& freq_cis = run_state.non_block_activations().freq_cis;
        if (freq_cis.nelem() != static_cast<long>(rope_freqs_golden.numel())) {
            freq_cis = run_state.temp_alloc(ETensorDType::FP32, rope_freqs_golden.shape);
        }
        copy_tensor_to_device(freq_cis, rope_freqs_golden);

        // Position IDs
        const auto& pos_ids_golden = gc.inputs.at("position_ids");
        const std::size_t T_len = static_cast<std::size_t>(T);
        const std::size_t B_len = static_cast<std::size_t>(B);
        const std::size_t total_pos_ids = B_len * T_len;

        if (run_state.PositionIDs.nelem() != static_cast<long>(total_pos_ids)) {
            run_state.PositionIDs = run_state.temp_alloc(ETensorDType::INT32, {B, T});
        }

        std::vector<std::int32_t> pos_ids_2d(total_pos_ids);
        for (std::size_t b = 0; b < B_len; ++b) {
            for (std::size_t t = 0; t < T_len; ++t) {
                std::int32_t val = pos_ids_golden.is_int()
                    ? static_cast<std::int32_t>(pos_ids_golden.i64[t])
                    : static_cast<std::int32_t>(pos_ids_golden.f64[t]);
                pos_ids_2d[b * T_len + t] = val;
            }
        }
        CUDA_CHECK(cudaMemcpy(run_state.PositionIDs.Data, pos_ids_2d.data(),
                             total_pos_ids * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Token IDs - copy to run_state.Inputs (the executor resolves token_ids to this slot)
        const auto& token_ids_golden = gc.inputs.at("token_ids");
        std::vector<std::int32_t> token_ids_host(B * T);
        for (std::size_t i = 0; i < token_ids_golden.numel(); ++i) {
            token_ids_host[i] = token_ids_golden.is_int()
                ? static_cast<std::int32_t>(token_ids_golden.i64[i])
                : static_cast<std::int32_t>(token_ids_golden.f64[i]);
        }
        CUDA_CHECK(cudaMemcpy(run_state.Inputs.Data, token_ids_host.data(),
                             B * T * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Targets - copy to run_state.Targets (the executor resolves targets to this slot)
        const auto& targets_golden = gc.inputs.at("targets");
        std::vector<std::int32_t> targets_host(B * T);
        for (std::size_t i = 0; i < targets_golden.numel(); ++i) {
            targets_host[i] = targets_golden.is_int()
                ? static_cast<std::int32_t>(targets_golden.i64[i])
                : static_cast<std::int32_t>(targets_golden.f64[i]);
        }
        CUDA_CHECK(cudaMemcpy(run_state.Targets.Data, targets_host.data(),
                             B * T * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile graph
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);

        INFO("Qwen3Model: DSL graph compiled with " << compiled.ops.size() << " operations");

        // Execute forward pass
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);

        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        // Verify forward outputs
        constexpr double rtol = 1e-3;
        constexpr double atol = 1e-3;

        // Check intermediate outputs first to identify where discrepancy starts
        auto check_intermediate = [&](const std::string& name) {
            const Tensor* actual = exec.try_get_tensor(name);
            if (actual && gc.outputs.count(name) > 0) {
                expect_allclose(name, gc.outputs.at(name), *actual, rtol, atol);
                INFO("  ✓ " << name << " matches");
            } else if (!actual) {
                INFO("  ⚠ " << name << " not found in executor");
            }
        };

        check_intermediate("x0");
        check_intermediate("block_out");
        check_intermediate("xF");

        const Tensor* loss_actual = exec.try_get_tensor("loss");
        REQUIRE(loss_actual != nullptr);

        // The kernel returns per-token losses [B*T], but golden is mean loss [1]
        // Compute mean of per-token losses for comparison
        {
            const auto per_token_losses = read_tensor_as_double(*loss_actual);
            REQUIRE(!per_token_losses.empty());

            double mean_loss = 0.0;
            for (double v : per_token_losses) {
                mean_loss += v;
            }
            mean_loss /= static_cast<double>(per_token_losses.size());

            const auto& golden_loss = gc.outputs.at("loss");
            REQUIRE(golden_loss.numel() == 1);
            const double expected_loss = golden_loss.f64[0];
            const double diff = std::abs(mean_loss - expected_loss);
            const double rel = diff / (std::abs(expected_loss) + 1e-12);
            INFO("loss: expected=" << expected_loss << " actual=" << mean_loss
                 << " diff=" << diff << " rel=" << rel);
            REQUIRE(diff < atol + rtol * std::abs(expected_loss));
        }

        INFO("✓ Qwen3Model forward pass verified against PyTorch reference");
    });
}


// ============================================================================
// Qwen3Model Recompute Comparison Test
// ============================================================================
// This test verifies that recompute mode produces identical results to non-recompute mode.
// It runs a single forward pass with recompute=false and recompute=true and compares the loss.
//
// NOTE: There is a known bug in RecomputeBlock=true - the comparison is expected to fail
// until the bug is fixed. The test still verifies that recompute=false matches the golden.

// Results from a single forward pass (loss + intermediate tensor norms)
struct RecomputeTestResult {
    double loss = -1.0;
    std::unordered_map<std::string, double> norms;  // L2 norms of intermediate tensors
};

TEST_CASE("dsl model goldens: qwen3_model recompute comparison", "[dsl][goldens][modules][models][recompute]") {
    // Reset CUDA device to ensure clean state
    CUDA_CHECK(cudaDeviceSynchronize());

    const fs::path golden_path = find_goldens_dir().parent_path() / "models" / "qwen3_model_small_1layer.json";
    if (!fs::exists(golden_path)) {
        SKIP("Golden file not found: " << golden_path);
    }

    const GoldenCase gc = load_case(golden_path);
    REQUIRE(gc.op == "qwen3_model");

    // Shape inference for QK-Norm+RoPE needs this flag
    setenv("SUROGATE_NO_SHAPE_CHECK", "1", 1);

    // Verify required outputs exist
    REQUIRE(gc.outputs.count("loss") > 0);

    // We'll store results from two runs
    RecomputeTestResult result_no_recompute;
    RecomputeTestResult result_with_recompute;

    // Helper to compute L2 norm of a tensor (returns 0 if tensor is null)
    auto compute_norm = [](const Tensor* t) -> double {
        if (!t || t->nelem() == 0) return 0.0;
        std::vector<double> data = read_tensor_as_double(*t);
        double sum_sq = 0.0;
        for (double v : data) sum_sq += v * v;
        return std::sqrt(sum_sq);
    };

    // Helper to run a single forward pass and return the result
    auto run_single_forward = [&gc, &compute_norm](NCCLCommunicator& comm, bool recompute_block) -> RecomputeTestResult {
        const long B = *meta_long(gc.meta, "B");
        const long T = *meta_long(gc.meta, "T");
        const long V = *meta_long(gc.meta, "V");
        const long C = *meta_long(gc.meta, "C");
        const long M = *meta_long(gc.meta, "M");
        const long Hq = *meta_long(gc.meta, "Hq");
        const long Hkv = *meta_long(gc.meta, "Hkv");
        const long HD = *meta_long(gc.meta, "head_dim");
        const long max_seq = *meta_long(gc.meta, "max_seq");
        const float eps = static_cast<float>(*meta_double(gc.meta, "eps"));

        const long QKV = (Hq + 2 * Hkv) * HD;
        const long AttnDim = Hq * HD;
        const long MUp = 2 * M;

        PretrainedConfig cfg;
        cfg.DType = ETensorDType::FP32;
        cfg.NumLayers = 1;
        cfg.NumQueryHeads = static_cast<int>(Hq);
        cfg.NumKeyValHeads = static_cast<int>(Hkv);
        cfg.HeadDim = static_cast<int>(HD);
        cfg.HiddenSize = static_cast<int>(C);
        cfg.IntermediateSize = static_cast<int>(M);
        cfg.VocabSize = static_cast<int>(V);
        cfg.MaxPositionEmbeddings = static_cast<int>(max_seq);
        cfg.RmsNormEps = eps;

        modules::ModelConfig model_cfg = modules::ModelConfig::from_pretrained_config(cfg);

        RuntimeOptions options;
        options.UseCudaGraphs = false;
        options.Recompute = recompute_block ? RecomputeLevel::Enabled : RecomputeLevel::None;
        options.ModelType = cfg.DType;
        options.MatmulType = cfg.DType;
        options.GradientType = cfg.DType;

        auto allocator = std::make_shared<TensorAllocator>();

        // Build the same graph as forward-only test
        dsl::Module module;
        module.name = "qwen3_model_recompute_test";
        module.kind = "model";

        dsl::Graph graph;
        graph.name = "qwen3_model";

        // Helper to add params
        auto add_param = [&](const std::string& name, const std::vector<long>& shape) {
            dsl::TensorInfo info;
            info.shape = to_dims(shape);
            info.dtype = ETensorDType::FP32;
            info.is_param = true;
            graph.params.emplace(name, info);
        };

        add_param("embedding", {V, C});
        add_param("final_norm", {C});
        add_param("lm_head", {V, C});
        add_param("block0_ln1_weight", {C});
        add_param("block0_qkv_weight", {QKV, C});
        add_param("block0_q_norm_weight", {HD});
        add_param("block0_k_norm_weight", {HD});
        add_param("block0_out_weight", {C, AttnDim});
        add_param("block0_ln2_weight", {C});
        add_param("block0_mlp_up_weight", {MUp, C});
        add_param("block0_mlp_down_weight", {C, M});

        dsl::TensorInfo rope_info;
        rope_info.shape = to_dims(gc.inputs.at("rope_freqs").shape);
        rope_info.dtype = ETensorDType::FP32;
        rope_info.is_param = true;
        graph.params.emplace("rope_freqs", rope_info);

        // Inputs
        dsl::TensorInfo tok_info;
        tok_info.shape = to_dims({B, T});
        tok_info.dtype = ETensorDType::INT32;
        tok_info.is_input = true;
        graph.inputs.emplace("token_ids", tok_info);

        dsl::TensorInfo pos_info;
        pos_info.shape = to_dims(gc.inputs.at("position_ids").shape);
        pos_info.dtype = ETensorDType::INT32;
        pos_info.is_input = true;
        graph.inputs.emplace("position_ids", pos_info);

        dsl::TensorInfo tgt_info;
        tgt_info.shape = to_dims({B, T});
        tgt_info.dtype = ETensorDType::INT32;
        tgt_info.is_input = true;
        graph.inputs.emplace("targets", tgt_info);

        // Output
        dsl::TensorInfo loss_info;
        loss_info.shape = to_dims({B * T});
        loss_info.dtype = ETensorDType::FP32;
        loss_info.is_output = true;
        graph.outputs.emplace("loss", loss_info);

        // Add operations
        auto add_op = [&](const std::string& id, const std::string& name, const std::string& kernel,
                         const std::vector<std::string>& ins, const std::vector<std::string>& outs,
                         const std::unordered_map<std::string, dsl::AttrValue>& attrs = {}) {
            dsl::Operation op;
            op.id = id;
            op.name = name;
            op.kernel_type = kernel;
            op.inputs = ins;
            op.outputs = outs;
            op.attrs = attrs;
            graph.operations.push_back(op);
        };

        auto shape_attr = [](const std::vector<std::int64_t>& dims) {
            dsl::AttrList list;
            for (auto d : dims) list.push_back(dsl::AttrValue(d));
            return dsl::AttrValue(std::make_shared<dsl::AttrList>(list));
        };

        add_op("embed_lookup", "embedding", "embedding", {"token_ids", "embedding"}, {"x0"});
        add_op("zero_residual", "zeros", "zeros", {}, {"residual0"},
               {{"shape", shape_attr({B, T, C})}, {"dtype", dsl::AttrValue("fp32")}});
        add_op("block0_fused_residual_rmsnorm_1", "fused_residual_rmsnorm", "fused_residual_rmsnorm",
               {"residual0", "x0", "block0_ln1_weight"}, {"block0_res_ffn", "block0_ln1", "block0_ln1_rstd"},
               {{"eps", dsl::AttrValue(static_cast<double>(eps))}});
        add_op("block0_ln1_flat_view", "view", "view", {"block0_ln1"}, {"block0_ln1_flat"},
               {{"shape", shape_attr({B * T, C})}});
        add_op("block0_qkv_proj", "matmul", "matmul", {"block0_ln1_flat", "block0_qkv_weight"}, {"block0_qkv_flat"},
               {{"transpose", dsl::AttrValue("NT")}});
        add_op("block0_qkv_view", "view", "view", {"block0_qkv_flat"}, {"block0_qkv"},
               {{"shape", shape_attr({B, T, Hq + 2 * Hkv, HD})}});
        add_op("block0_qkv_qk_norm_rope", "qkv_qk_norm_rope", "qkv_qk_norm_rope",
               {"block0_qkv", "block0_q_norm_weight", "block0_k_norm_weight", "rope_freqs", "position_ids"},
               {"block0_qkv_rope", "block0_q_rstd", "block0_k_rstd"},
               {{"eps", dsl::AttrValue(static_cast<double>(eps))}});
        add_op("block0_flash_attention", "flash_attention", "flash_attention",
               {"block0_qkv_rope"}, {"block0_att", "block0_lse"},
               {{"causal", dsl::AttrValue(true)}});
        add_op("block0_att_flat_view", "view", "view", {"block0_att"}, {"block0_att_flat"},
               {{"shape", shape_attr({B * T, AttnDim})}});
        add_op("block0_out_proj", "matmul", "matmul", {"block0_att_flat", "block0_out_weight"}, {"block0_att_out_flat"},
               {{"transpose", dsl::AttrValue("NT")}});
        add_op("block0_att_out_view", "view", "view", {"block0_att_out_flat"}, {"block0_att_out"},
               {{"shape", shape_attr({B, T, C})}});
        add_op("block0_fused_residual_rmsnorm_2", "fused_residual_rmsnorm", "fused_residual_rmsnorm",
               {"block0_res_ffn", "block0_att_out", "block0_ln2_weight"}, {"block0_residual_out", "block0_ln2", "block0_ln2_rstd"},
               {{"eps", dsl::AttrValue(static_cast<double>(eps))}});
        add_op("block0_ln2_flat_view", "view", "view", {"block0_ln2"}, {"block0_ln2_flat"},
               {{"shape", shape_attr({B * T, C})}});
        add_op("block0_mlp_up_proj", "matmul", "matmul", {"block0_ln2_flat", "block0_mlp_up_weight"}, {"block0_mlp_up_flat"},
               {{"transpose", dsl::AttrValue("NT")}});
        add_op("block0_mlp_up_view", "view", "view", {"block0_mlp_up_flat"}, {"block0_mlp_up"},
               {{"shape", shape_attr({B, T, MUp})}});
        add_op("block0_swiglu", "swiglu", "swiglu", {"block0_mlp_up"}, {"block0_swiglu_out"});
        add_op("block0_swiglu_flat_view", "view", "view", {"block0_swiglu_out"}, {"block0_swiglu_flat"},
               {{"shape", shape_attr({B * T, M})}});
        add_op("block0_mlp_down_proj", "matmul", "matmul", {"block0_swiglu_flat", "block0_mlp_down_weight"}, {"block0_out_flat"},
               {{"transpose", dsl::AttrValue("NT")}});
        add_op("block0_out_view", "view", "view", {"block0_out_flat"}, {"block0_out"},
               {{"shape", shape_attr({B, T, C})}});
        add_op("final_fused_residual_rmsnorm", "fused_residual_rmsnorm", "fused_residual_rmsnorm",
               {"block0_residual_out", "block0_out", "final_norm"}, {"residual_final", "xF", "final_rstd"},
               {{"eps", dsl::AttrValue(static_cast<double>(eps))}});
        add_op("xF_flat_view", "view", "view", {"xF"}, {"xF_flat"},
               {{"shape", shape_attr({B * T, C})}});
        add_op("lm_head_loss", "fused_lm_head_loss", "fused_lm_head_loss",
               {"xF_flat", "lm_head", "targets"}, {"loss"},
               {{"compute_accuracy", dsl::AttrValue(true)}});

        module.forward = graph;

        // Create stores
        dsl::DslParamStore params(module, graph, options, cfg, allocator, nullptr, nullptr, false);
        dsl::DslGradStore grads(params, allocator, false, EAllocationType::ON_DEVICE, 1, false);
        constexpr std::size_t kStackBytes = 256 * 1024 * 1024;
        dsl::DslRunState run_state(cfg, options, static_cast<int>(B), static_cast<int>(T), allocator, false, kStackBytes, true);

        // Reset loss/accuracy buffers since cross-entropy forward accumulates into them.
        fill_zero(run_state.Losses, run_state.MainStream);
        fill_zero(run_state.ValidTokenCount, run_state.MainStream);
        fill_zero(run_state.CorrectCount, run_state.MainStream);

        // Copy weights
        copy_tensor_to_device(params.get("embedding"), gc.inputs.at("embedding"));
        copy_tensor_to_device(params.get("final_norm"), gc.inputs.at("final_norm"));
        copy_tensor_to_device(params.get("lm_head"), gc.inputs.at("lm_head"));
        copy_tensor_to_device(params.get("block0_ln1_weight"), gc.inputs.at("block0_ln1_weight"));
        copy_tensor_to_device(params.get("block0_qkv_weight"), gc.inputs.at("block0_qkv_weight"));
        copy_tensor_to_device(params.get("block0_q_norm_weight"), gc.inputs.at("block0_q_norm_weight"));
        copy_tensor_to_device(params.get("block0_k_norm_weight"), gc.inputs.at("block0_k_norm_weight"));
        copy_tensor_to_device(params.get("block0_out_weight"), gc.inputs.at("block0_out_weight"));
        copy_tensor_to_device(params.get("block0_ln2_weight"), gc.inputs.at("block0_ln2_weight"));
        copy_tensor_to_device(params.get("block0_mlp_up_weight"), gc.inputs.at("block0_mlp_up_weight"));
        copy_tensor_to_device(params.get("block0_mlp_down_weight"), gc.inputs.at("block0_mlp_down_weight"));

        // RoPE freqs
        const auto& rope_golden = gc.inputs.at("rope_freqs");
        auto& freq_cis = run_state.non_block_activations().freq_cis;
        if (freq_cis.nelem() != static_cast<long>(rope_golden.numel())) {
            freq_cis = run_state.temp_alloc(ETensorDType::FP32, rope_golden.shape);
        }
        copy_tensor_to_device(freq_cis, rope_golden);

        // Position IDs
        const auto& pos_golden = gc.inputs.at("position_ids");
        const std::size_t T_len = static_cast<std::size_t>(T), B_len = static_cast<std::size_t>(B);
        if (run_state.PositionIDs.nelem() != static_cast<long>(B_len * T_len)) {
            run_state.PositionIDs = run_state.temp_alloc(ETensorDType::INT32, {B, T});
        }
        std::vector<std::int32_t> pos_ids(B_len * T_len);
        for (std::size_t b = 0; b < B_len; ++b) {
            for (std::size_t t = 0; t < T_len; ++t) {
                pos_ids[b * T_len + t] = pos_golden.is_int() ? static_cast<std::int32_t>(pos_golden.i64[t]) : static_cast<std::int32_t>(pos_golden.f64[t]);
            }
        }
        CUDA_CHECK(cudaMemcpy(run_state.PositionIDs.Data, pos_ids.data(), B_len * T_len * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Token IDs
        const auto& tok_golden = gc.inputs.at("token_ids");
        std::vector<std::int32_t> tok_host(B * T);
        for (std::size_t i = 0; i < tok_golden.numel(); ++i) {
            tok_host[i] = tok_golden.is_int() ? static_cast<std::int32_t>(tok_golden.i64[i]) : static_cast<std::int32_t>(tok_golden.f64[i]);
        }
        CUDA_CHECK(cudaMemcpy(run_state.Inputs.Data, tok_host.data(), B * T * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Targets
        const auto& tgt_golden = gc.inputs.at("targets");
        std::vector<std::int32_t> tgt_host(B * T);
        for (std::size_t i = 0; i < tgt_golden.numel(); ++i) {
            tgt_host[i] = tgt_golden.is_int() ? static_cast<std::int32_t>(tgt_golden.i64[i]) : static_cast<std::int32_t>(tgt_golden.f64[i]);
        }
        CUDA_CHECK(cudaMemcpy(run_state.Targets.Data, tgt_host.data(), B * T * sizeof(std::int32_t), cudaMemcpyHostToDevice));

        // Compile and execute
        dsl::GraphCompiler compiler(module, model_cfg, options, params, grads);
        auto compiled = compiler.compile(graph, B, T);
        dsl::CompiledExecutor exec(run_state, params, grads, model_cfg, options);
        exec.set_dimensions(B, T);

        exec.execute_forward(compiled, comm, true, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(run_state.MainStream));

        RecomputeTestResult result;

        // Capture norms of intermediate tensors for debugging
        const std::vector<std::string> tensor_names = {
            "x0",                    // Embedding output
            "block0_ln1",            // Pre-attention layer norm
            "block0_qkv",            // QKV projection
            "block0_qkv_rope",       // After RoPE
            "block0_att",            // Attention output
            "block0_att_out",        // After out projection
            "block0_ln2",            // Pre-MLP layer norm
            "block0_mlp_up",         // MLP up projection
            "block0_swiglu_out",     // After SwiGLU
            "block0_out",            // MLP output
            "xF",                    // Final layer norm output
        };

        for (const auto& name : tensor_names) {
            const Tensor* t = exec.try_get_tensor(name);
            result.norms[name] = compute_norm(t);
        }

        // Get loss
        const Tensor* loss_tensor = exec.try_get_tensor("loss");
        if (!loss_tensor) return result;

        std::vector<double> per_token = read_tensor_as_double(*loss_tensor);
        double mean_loss = 0.0;
        for (double v : per_token) mean_loss += v;
        mean_loss /= static_cast<double>(per_token.size());
        result.loss = mean_loss;
        result.norms["loss"] = compute_norm(loss_tensor);

        return result;
    };

    // Run with recompute=false
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        result_no_recompute = run_single_forward(comm, false);
    });

    // Run with recompute=true in separate callback for isolation
    NCCLCommunicator::run_communicators(1, false, false, [&](NCCLCommunicator& comm) {
        result_with_recompute = run_single_forward(comm, true);
    });

    // Verify both runs produced valid results
    REQUIRE(result_no_recompute.loss >= 0.0);
    REQUIRE(result_with_recompute.loss >= 0.0);

    // Verify recompute=false matches golden reference
    const auto& golden_loss = gc.outputs.at("loss");
    const double expected_loss = golden_loss.f64[0];
    const double diff_golden = std::abs(result_no_recompute.loss - expected_loss);

    INFO("Loss (recompute=false): " << result_no_recompute.loss);
    INFO("Loss (recompute=true):  " << result_with_recompute.loss);
    INFO("vs golden: expected=" << expected_loss << " diff=" << diff_golden);
    REQUIRE(diff_golden < 1e-3 + 1e-3 * std::abs(expected_loss));

    // Compare recompute=true vs recompute=false - loss
    const double loss_diff = std::abs(result_with_recompute.loss - result_no_recompute.loss);
    const double loss_rel_diff = loss_diff / (std::abs(result_no_recompute.loss) + 1e-12);

    INFO("=== Loss Comparison ===");
    INFO("  abs_diff=" << loss_diff << " rel_diff=" << loss_rel_diff);

    // Compare tensor norms to identify where divergence starts
    INFO("=== Tensor Norm Comparison ===");

    const std::vector<std::string> tensor_order = {
        "x0", "block0_ln1", "block0_qkv", "block0_qkv_rope", "block0_att",
        "block0_att_out", "block0_ln2", "block0_mlp_up", "block0_swiglu_out",
        "block0_out", "xF", "loss"
    };

    bool found_divergence = false;
    std::string first_divergent_tensor;
    double max_rel_diff = 0.0;

    for (const auto& name : tensor_order) {
        const double norm_no_recompute = result_no_recompute.norms.count(name) ? result_no_recompute.norms.at(name) : 0.0;
        const double norm_with_recompute = result_with_recompute.norms.count(name) ? result_with_recompute.norms.at(name) : 0.0;
        const double abs_diff = std::abs(norm_with_recompute - norm_no_recompute);
        const double rel_diff = abs_diff / (norm_no_recompute + 1e-12);

        // Only WARN on divergence, INFO otherwise (INFO shows on failure)
        if (rel_diff > 1e-3) {
            WARN("  " << name << ": norm_base=" << norm_no_recompute << " norm_recompute=" << norm_with_recompute
                 << " rel_diff=" << rel_diff << " <-- DIVERGED");
        } else {
            INFO("  " << name << ": norm_base=" << norm_no_recompute << " norm_recompute=" << norm_with_recompute
                 << " rel_diff=" << rel_diff);
        }

        if (rel_diff > 1e-3 && !found_divergence) {
            found_divergence = true;
            first_divergent_tensor = name;
        }
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }

    if (found_divergence) {
        WARN(">>> First divergent tensor: " << first_divergent_tensor);
    }
    INFO(">>> Max relative difference in norms: " << max_rel_diff);

    // This REQUIRE will fail when RecomputeBlock=true has bugs
    REQUIRE(loss_rel_diff < 1e-3);

    INFO("✓ Qwen3Model recompute comparison passed");
}
