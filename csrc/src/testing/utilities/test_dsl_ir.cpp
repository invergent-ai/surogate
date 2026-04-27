// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for DSL IR JSON loader + shape resolution.

#include <string>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/ir.h"

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
        ],
        "metadata": {
          "block_schemas": [
            {
              "layer": 0,
              "block_index": 0,
              "block_type": "Qwen3Block",
              "blocks_param": "blocks",
              "block_name": "Qwen3Block",
              "schema": {
                "routing": {"kind": "none"},
                "ep_topology": {"ep_size_param": "ep_size"},
                "slots": [
                  {
                    "name": "qkv_weight",
                    "kind": "param",
                    "distribution": {"kind": "replicated"}
                  },
                  {
                    "name": "experts_gate_up",
                    "kind": "param",
                    "distribution": {"kind": "expert_parallel"},
                    "streaming_hint": {"prefetch_distance": 1}
                  },
                  {
                    "name": "permuted_input",
                    "kind": "activation",
                    "distribution": {"kind": "expert_parallel"}
                  }
                ],
                "attrs": {"block_family": "qwen3_dense"}
              }
            }
          ]
        }
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

    auto meta_it = module.forward->metadata.find("block_schemas");
    REQUIRE(meta_it != module.forward->metadata.end());
    const auto* records_ptr = std::get_if<dsl::AttrValue::ListPtr>(&meta_it->second.value);
    REQUIRE(records_ptr);
    REQUIRE(*records_ptr);
    REQUIRE((**records_ptr).size() == 1);

    const auto* record_ptr = std::get_if<dsl::AttrValue::MapPtr>(&(**records_ptr)[0].value);
    REQUIRE(record_ptr);
    REQUIRE(*record_ptr);
    const auto& record = **record_ptr;
    REQUIRE(std::get<std::int64_t>(record.at("layer").value) == 0);
    REQUIRE(std::get<std::string>(record.at("block_type").value) == "Qwen3Block");

    const auto* schema_ptr = std::get_if<dsl::AttrValue::MapPtr>(&record.at("schema").value);
    REQUIRE(schema_ptr);
    REQUIRE(*schema_ptr);
    const auto* attrs_ptr = std::get_if<dsl::AttrValue::MapPtr>(&(**schema_ptr).at("attrs").value);
    REQUIRE(attrs_ptr);
    REQUIRE(*attrs_ptr);
    REQUIRE(std::get<std::string>((**attrs_ptr).at("block_family").value) == "qwen3_dense");

    const auto schema_records = dsl::collect_block_schema_plan_records(*module.forward);
    REQUIRE(schema_records.size() == 1);
    REQUIRE(schema_records[0].layer == 0);
    REQUIRE(schema_records[0].block_index == 0);
    REQUIRE(schema_records[0].block_type == "Qwen3Block");
    REQUIRE(schema_records[0].blocks_param == "blocks");
    REQUIRE(schema_records[0].block_name == "Qwen3Block");
    REQUIRE(schema_records[0].block_family == "qwen3_dense");
    REQUIRE(schema_records[0].family_kind == dsl::BlockSchemaFamilyKind::Dense);
    REQUIRE(schema_records[0].slot_count == 3);
    REQUIRE(schema_records[0].param_slots == 2);
    REQUIRE(schema_records[0].activation_slots == 1);
    REQUIRE(schema_records[0].expert_parallel_slots == 2);
    REQUIRE(schema_records[0].streaming_slots == 1);
    REQUIRE(schema_records[0].has_routing);
    REQUIRE(schema_records[0].has_ep_topology);

    auto coverage = dsl::validate_block_schema_plan_coverage(schema_records, /*num_layers=*/1);
    REQUIRE(coverage.ok);
    coverage = dsl::validate_block_schema_plan_coverage({}, /*num_layers=*/1);
    REQUIRE(coverage.ok);

    auto duplicate_records = schema_records;
    duplicate_records.push_back(schema_records[0]);
    coverage = dsl::validate_block_schema_plan_coverage(duplicate_records, /*num_layers=*/2);
    REQUIRE_FALSE(coverage.ok);
    REQUIRE(coverage.message.find("duplicate block schema layer 0") != std::string::npos);

    auto missing_records = schema_records;
    coverage = dsl::validate_block_schema_plan_coverage(missing_records, /*num_layers=*/2);
    REQUIRE_FALSE(coverage.ok);
    REQUIRE(coverage.message.find("record count 1 != NumLayers 2") != std::string::npos);

    auto out_of_range_records = schema_records;
    out_of_range_records[0].layer = 2;
    coverage = dsl::validate_block_schema_plan_coverage(out_of_range_records, /*num_layers=*/1);
    REQUIRE_FALSE(coverage.ok);
    REQUIRE(coverage.message.find("outside [0, 1)") != std::string::npos);

    PretrainedConfig cfg;
    cfg.HiddenSize = 8;
    cfg.NumQueryHeads = 2;
    cfg.NumKeyValHeads = 1;
    cfg.IntermediateSize = 16;
    cfg.NumLayers = 1;
    dsl::DslRuntimeConfig runtime_config;
    RuntimeOptions options;
    dsl::TensorSlotRegistry registry;
    const auto plan = dsl::BufferPlan::build(cfg,
                                             runtime_config,
                                             options,
                                             registry,
                                             /*lora_only_mode=*/false,
                                             /*B=*/2,
                                             /*T=*/3,
                                             ETensorDType::BF16,
                                             ETensorDType::BF16,
                                             &schema_records);
    REQUIRE(plan.schema_record_count == 1);
    REQUIRE(plan.schema_routing_layers == 1);
    REQUIRE(plan.schema_ep_layers == 1);
    REQUIRE(plan.schema_dense_layers == 1);
    REQUIRE(plan.schema_moe_layers == 0);
    REQUIRE(plan.schema_mamba_layers == 0);
    REQUIRE(plan.schema_linear_mixer_layers == 0);
    REQUIRE(plan.schema_slot_count == 3);
    REQUIRE(plan.schema_param_slots == 2);
    REQUIRE(plan.schema_activation_slots == 1);
    REQUIRE(plan.schema_expert_parallel_slots == 2);
    REQUIRE(plan.schema_streaming_slots == 1);
    REQUIRE(plan.schema_layers.size() == 1);
    REQUIRE(plan.schema_layers[0].layer == 0);
    REQUIRE(plan.schema_layers[0].has_schema);
    REQUIRE(plan.schema_layers[0].block_family == "qwen3_dense");
    REQUIRE(plan.schema_layers[0].family_kind == dsl::BlockSchemaFamilyKind::Dense);
    REQUIRE(plan.schema_layers[0].slot_count == 3);
    REQUIRE(plan.schema_layers[0].param_slots == 2);
    REQUIRE(plan.schema_layers[0].activation_slots == 1);
    REQUIRE(plan.schema_layers[0].expert_parallel_slots == 2);
    REQUIRE(plan.schema_layers[0].streaming_slots == 1);
    REQUIRE(plan.schema_layers[0].has_routing);
    REQUIRE(plan.schema_layers[0].has_ep_topology);
}
