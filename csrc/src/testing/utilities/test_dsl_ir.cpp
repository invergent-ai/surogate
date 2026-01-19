// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for DSL IR JSON loader + shape resolution.

#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "dsl/ir.h"

TEST_CASE("DSL IR loader parses module and resolves shapes") {
    const char* kJson = R"JSON(
{
  "source_file": "std/models/qwen3.module",
  "success": true,
  "modules": [
    {
      "name": "Qwen3Model",
      "kind": "model",
      "config": {
        "d_model": 8,
        "n_layers": 2,
        "vocab_size": 16
      },
      "params": {
        "embedding": {"shape": [16, 8], "dtype": "bf16", "is_param": true},
        "lm_head": {"shape": [16, 8], "dtype": "bf16", "is_param": true}
      },
      "forward": {
        "name": "Qwen3Model.forward",
        "inputs": {
          "token_ids": {"shape": ["B", "T"], "dtype": "int32", "is_input": true}
        },
        "outputs": {
          "logits": {"shape": ["B", "T", "vocab_size"], "dtype": "bf16", "is_output": true}
        },
        "intermediates": {
          "x0": {"shape": ["B", "T", "d_model"], "dtype": "bf16"}
        },
        "operations": [
          {"id": "node_1", "name": "embedding", "kernel_type": "embedding", "inputs": ["token_ids", "embedding"], "outputs": ["x0"]},
          {"id": "node_2", "name": "view", "kernel_type": "view", "inputs": ["x0"], "outputs": ["x0_flat"], "attrs": {"shape": ["(B * T)", "d_model"]}}
        ]
      }
    }
  ]
}
)JSON";

    nlohmann::json root = nlohmann::json::parse(kJson);
    auto ir = dsl::load_ir_from_json(root);

    REQUIRE(ir.success);
    REQUIRE(ir.modules.size() == 1);
    const auto& module = ir.modules.front();
    REQUIRE(module.name == "Qwen3Model");
    REQUIRE(module.kind == "model");
    REQUIRE(module.forward.has_value());
    REQUIRE(module.forward->operations.size() == 2);

    auto env = dsl::make_shape_env(module, /*B=*/2, /*T=*/3);
    const auto& x0 = module.forward->intermediates.at("x0");
    auto resolved = dsl::resolve_shape(x0.shape, env);
    REQUIRE(resolved.size() == 3);
    REQUIRE(resolved[0] == 2);
    REQUIRE(resolved[1] == 3);
    REQUIRE(resolved[2] == 8);

    const auto& op = module.forward->operations[1];
    auto it = op.attrs.find("shape");
    REQUIRE(it != op.attrs.end());
    const auto* list_ptr = std::get_if<dsl::AttrValue::ListPtr>(&it->second.value);
    REQUIRE(list_ptr);
    REQUIRE(*list_ptr);
    const auto& list = **list_ptr;
    REQUIRE(list.size() == 2);
    const auto* dim_expr = std::get_if<std::string>(&list[0].value);
    REQUIRE(dim_expr);
    dsl::Dim dim = dsl::Dim::computed(*dim_expr);
    REQUIRE(dsl::resolve_dim(dim, env) == 6);
}
