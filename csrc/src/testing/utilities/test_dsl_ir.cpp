// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for DSL IR JSON loader + shape resolution.

#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/hook_registry.h"
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
                "routing": {
                  "kind": "topk_softmax",
                  "topk": 2,
                  "norm_topk_prob": true,
                  "scoring_bias": false,
                  "shared_experts": 0
                },
                "ep_topology": {"ep_size_param": "ep_size", "weight_transfer_eligible": false},
                "slots": [
                  {
                    "name": "qkv_weight",
                    "kind": "param",
                    "residency": "gpu",
                    "shape": ["QKV", "C"],
                    "distribution": {"kind": "replicated"}
                  },
                  {
                    "name": "experts_gate_up",
                    "kind": "param",
                    "residency": "auto",
                    "shape": ["E", "2M", "C"],
                    "grouped": true,
                    "save_for_backward": false,
                    "distribution": {"kind": "expert_parallel"},
                    "streaming_hint": {"prefetch_distance": 1}
                  },
                  {
                    "name": "permuted_input",
                    "kind": "activation",
                    "residency": "cpu_pinned_stream",
                    "shape": ["B", "T", "C"],
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
    REQUIRE(schema_records[0].op_lifetime_slots == 0);
    REQUIRE(schema_records[0].layer_lifetime_slots == 3);
    REQUIRE(schema_records[0].block_lifetime_slots == 0);
    REQUIRE(schema_records[0].model_lifetime_slots == 0);
    REQUIRE(schema_records[0].persistent_lifetime_slots == 0);
    REQUIRE(schema_records[0].replicated_slots == 1);
    REQUIRE(schema_records[0].sharded_dim_slots == 0);
    REQUIRE(schema_records[0].router_replicated_slots == 0);
    REQUIRE(schema_records[0].expert_parallel_slots == 2);
    REQUIRE(schema_records[0].streaming_slots == 1);
    REQUIRE(schema_records[0].gpu_resident_slots == 1);
    REQUIRE(schema_records[0].auto_resident_slots == 1);
    REQUIRE(schema_records[0].cpu_pinned_stream_slots == 1);
    REQUIRE(schema_records[0].cpu_pageable_slots == 0);
    REQUIRE(schema_records[0].nvme_offload_slots == 0);
    REQUIRE(schema_records[0].slots.size() == 3);
    REQUIRE(schema_records[0].slots[1].name == "experts_gate_up");
    REQUIRE(schema_records[0].slots[1].kind == "param");
    REQUIRE(schema_records[0].slots[1].lifetime == "layer");
    REQUIRE(schema_records[0].slots[1].residency == "auto");
    REQUIRE(schema_records[0].slots[1].distribution_kind == "expert_parallel");
    REQUIRE(schema_records[0].slots[1].shape_rank == 3);
    REQUIRE(schema_records[0].slots[1].shape_dims == std::vector<std::string>{"E", "2M", "C"});
    REQUIRE(schema_records[0].slots[1].grouped);
    REQUIRE_FALSE(schema_records[0].slots[1].save_for_backward);
    REQUIRE(schema_records[0].slots[1].streaming_prefetch_distance == 1);
    REQUIRE(schema_records[0].slots[2].name == "permuted_input");
    REQUIRE(schema_records[0].slots[2].shape_dims == std::vector<std::string>{"B", "T", "C"});
    REQUIRE(schema_records[0].routing_kind == "topk_softmax");
    REQUIRE(schema_records[0].routing_topk == 2);
    REQUIRE(schema_records[0].routing_topk_param.empty());
    REQUIRE(schema_records[0].routing_norm_topk_prob);
    REQUIRE(schema_records[0].routing_norm_topk_prob_param.empty());
    REQUIRE_FALSE(schema_records[0].routing_scoring_bias);
    REQUIRE(schema_records[0].routing_shared_experts == 0);
    REQUIRE(schema_records[0].routing_shared_experts_param.empty());
    REQUIRE(schema_records[0].ep_size_param == "ep_size");
    REQUIRE_FALSE(schema_records[0].ep_weight_transfer_eligible);
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
    runtime_config.num_experts = 4;
    runtime_config.linear_conv_kernel_dim = 4;
    runtime_config.linear_key_head_dim = 32;
    runtime_config.linear_value_head_dim = 16;
    runtime_config.linear_num_key_heads = 4;
    runtime_config.linear_num_value_heads = 8;
    runtime_config.d_per_layer_input = 12;
    runtime_config.mamba_num_heads = 8;
    runtime_config.mamba_head_dim = 32;
    runtime_config.ssm_state_size = 16;
    runtime_config.n_groups = 4;
    RuntimeOptions options;
    options.EPSize = 2;
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
    REQUIRE(plan.schema_op_lifetime_slots == 0);
    REQUIRE(plan.schema_layer_lifetime_slots == 3);
    REQUIRE(plan.schema_block_lifetime_slots == 0);
    REQUIRE(plan.schema_model_lifetime_slots == 0);
    REQUIRE(plan.schema_persistent_lifetime_slots == 0);
    REQUIRE(plan.schema_replicated_slots == 1);
    REQUIRE(plan.schema_sharded_dim_slots == 0);
    REQUIRE(plan.schema_router_replicated_slots == 0);
    REQUIRE(plan.schema_expert_parallel_slots == 2);
    REQUIRE(plan.schema_streaming_slots == 1);
    REQUIRE(plan.schema_gpu_resident_slots == 1);
    REQUIRE(plan.schema_auto_resident_slots == 1);
    REQUIRE(plan.schema_cpu_pinned_stream_slots == 1);
    REQUIRE(plan.schema_cpu_pageable_slots == 0);
    REQUIRE(plan.schema_nvme_offload_slots == 0);
    REQUIRE(plan.schema_registry_registered_activation_slots == 0);
    REQUIRE(plan.schema_registry_missing_activation_slots == 0);
    REQUIRE(plan.schema_resolved_activation_shape_slots == 1);
    REQUIRE(plan.schema_unresolved_activation_shape_slots == 0);
    REQUIRE(plan.schema_resolved_activation_shape_bytes == 96);
    REQUIRE(plan.schema_save_for_backward_activation_slots == 0);
    REQUIRE(plan.schema_frame_activation_slots == 1);
    REQUIRE(plan.schema_save_for_backward_activation_bytes == 0);
    REQUIRE(plan.schema_frame_activation_bytes == 96);
    REQUIRE(plan.schema_max_layer_activation_shape_bytes == 96);
    REQUIRE(plan.schema_legacy_max_activation_shape_bytes == 96);
    REQUIRE(plan.schema_activation_shape_savings_bytes == 0);
    REQUIRE(plan.schema_resolved_param_shape_slots == 2);
    REQUIRE(plan.schema_unresolved_param_shape_slots == 0);
    REQUIRE(plan.schema_expert_parallel_param_slots == 1);
    REQUIRE(plan.schema_resolved_param_shape_bytes == 2304);
    REQUIRE(plan.schema_resolved_param_shape_local_bytes == 1280);
    REQUIRE(plan.schema_expert_parallel_param_shape_bytes == 2048);
    REQUIRE(plan.schema_expert_parallel_param_shape_local_bytes == 1024);
    REQUIRE(plan.schema_expert_parallel_param_shape_savings_bytes == 1024);
    REQUIRE(plan.schema_scoring_bias_routing_layers == 0);
    REQUIRE(plan.schema_shared_expert_routing_layers == 0);
    REQUIRE(plan.schema_weight_transfer_layers == 0);
    REQUIRE(plan.schema_layers.size() == 1);
    REQUIRE(plan.schema_layers[0].layer == 0);
    REQUIRE(plan.schema_layers[0].has_schema);
    REQUIRE(plan.schema_layers[0].block_family == "qwen3_dense");
    REQUIRE(plan.schema_layers[0].family_kind == dsl::BlockSchemaFamilyKind::Dense);
    REQUIRE(plan.schema_layers[0].slot_count == 3);
    REQUIRE(plan.schema_layers[0].param_slots == 2);
    REQUIRE(plan.schema_layers[0].activation_slots == 1);
    REQUIRE(plan.schema_layers[0].op_lifetime_slots == 0);
    REQUIRE(plan.schema_layers[0].layer_lifetime_slots == 3);
    REQUIRE(plan.schema_layers[0].block_lifetime_slots == 0);
    REQUIRE(plan.schema_layers[0].model_lifetime_slots == 0);
    REQUIRE(plan.schema_layers[0].persistent_lifetime_slots == 0);
    REQUIRE(plan.schema_layers[0].replicated_slots == 1);
    REQUIRE(plan.schema_layers[0].sharded_dim_slots == 0);
    REQUIRE(plan.schema_layers[0].router_replicated_slots == 0);
    REQUIRE(plan.schema_layers[0].expert_parallel_slots == 2);
    REQUIRE(plan.schema_layers[0].streaming_slots == 1);
    REQUIRE(plan.schema_layers[0].gpu_resident_slots == 1);
    REQUIRE(plan.schema_layers[0].auto_resident_slots == 1);
    REQUIRE(plan.schema_layers[0].cpu_pinned_stream_slots == 1);
    REQUIRE(plan.schema_layers[0].cpu_pageable_slots == 0);
    REQUIRE(plan.schema_layers[0].nvme_offload_slots == 0);
    REQUIRE(plan.schema_layers[0].resolved_activation_shape_slots == 1);
    REQUIRE(plan.schema_layers[0].unresolved_activation_shape_slots == 0);
    REQUIRE(plan.schema_layers[0].resolved_activation_shape_bytes == 96);
    REQUIRE(plan.schema_layers[0].save_for_backward_activation_slots == 0);
    REQUIRE(plan.schema_layers[0].frame_activation_slots == 1);
    REQUIRE(plan.schema_layers[0].save_for_backward_activation_bytes == 0);
    REQUIRE(plan.schema_layers[0].frame_activation_bytes == 96);
    REQUIRE(plan.schema_layers[0].resolved_param_shape_slots == 2);
    REQUIRE(plan.schema_layers[0].unresolved_param_shape_slots == 0);
    REQUIRE(plan.schema_layers[0].expert_parallel_param_slots == 1);
    REQUIRE(plan.schema_layers[0].resolved_param_shape_bytes == 2304);
    REQUIRE(plan.schema_layers[0].resolved_param_shape_local_bytes == 1280);
    REQUIRE(plan.schema_layers[0].slots.size() == 3);
    REQUIRE(plan.schema_layers[0].slots[1].name == "experts_gate_up");
    REQUIRE(plan.schema_layers[0].slots[1].shape_dims == std::vector<std::string>{"E", "2M", "C"});
    REQUIRE(plan.schema_layers[0].slots[1].resolved_shape == std::vector<long>{4, 32, 8});
    REQUIRE(plan.schema_layers[0].slots[1].resolved_bytes == 2048);
    REQUIRE(plan.schema_layers[0].slots[1].resolved_local_bytes == 1024);
    REQUIRE(plan.schema_layers[0].slots[2].shape_resolved);
    REQUIRE(plan.schema_layers[0].slots[2].resolved_shape == std::vector<long>{2, 3, 8});
    REQUIRE(plan.schema_layers[0].slots[2].resolved_numel == 48);
    REQUIRE(plan.schema_layers[0].slots[2].resolved_bytes == 96);

    auto alias_records = schema_records;
    alias_records[0].slots.clear();
    dsl::BlockSchemaSlotSummary q_proj_slot;
    q_proj_slot.name = "full_q_proj_weight";
    q_proj_slot.kind = "param";
    q_proj_slot.distribution_kind = "replicated";
    q_proj_slot.shape_dims = {"QProjDim", "C"};
    q_proj_slot.shape_rank = 2;
    alias_records[0].slots.push_back(q_proj_slot);
    dsl::BlockSchemaSlotSummary kv_slot;
    kv_slot.name = "full_k_proj_weight";
    kv_slot.kind = "param";
    kv_slot.distribution_kind = "replicated";
    kv_slot.shape_dims = {"KVDim", "C"};
    kv_slot.shape_rank = 2;
    alias_records[0].slots.push_back(kv_slot);
    dsl::BlockSchemaSlotSummary conv_slot;
    conv_slot.name = "lin_in_proj_qkv_weight";
    conv_slot.kind = "param";
    conv_slot.distribution_kind = "replicated";
    conv_slot.shape_dims = {"ConvDim", "C"};
    conv_slot.shape_rank = 2;
    alias_records[0].slots.push_back(conv_slot);
    dsl::BlockSchemaSlotSummary value_slot;
    value_slot.name = "lin_out_weight";
    value_slot.kind = "param";
    value_slot.distribution_kind = "replicated";
    value_slot.shape_dims = {"C", "ValueDim"};
    value_slot.shape_rank = 2;
    alias_records[0].slots.push_back(value_slot);
    dsl::BlockSchemaSlotSummary conv_state_slot;
    conv_state_slot.name = "lin_conv_state";
    conv_state_slot.kind = "activation";
    conv_state_slot.distribution_kind = "replicated";
    conv_state_slot.shape_dims = {"B", "ConvDim", "ConvK"};
    conv_state_slot.shape_rank = 3;
    alias_records[0].slots.push_back(conv_state_slot);
    dsl::BlockSchemaSlotSummary pli_slot;
    pli_slot.name = "per_layer_input";
    pli_slot.kind = "activation";
    pli_slot.distribution_kind = "replicated";
    pli_slot.shape_dims = {"B", "T", "PLI_D"};
    pli_slot.shape_rank = 3;
    alias_records[0].slots.push_back(pli_slot);
    dsl::BlockSchemaSlotSummary dispatched_slot;
    dispatched_slot.name = "permuted_input";
    dispatched_slot.kind = "activation";
    dispatched_slot.distribution_kind = "expert_parallel";
    dispatched_slot.shape_dims = {"dispatched_tokens", "C"};
    dispatched_slot.shape_rank = 2;
    alias_records[0].slots.push_back(dispatched_slot);
    dsl::BlockSchemaSlotSummary mamba_projected_slot;
    mamba_projected_slot.name = "projected";
    mamba_projected_slot.kind = "activation";
    mamba_projected_slot.distribution_kind = "replicated";
    mamba_projected_slot.shape_dims = {"B", "T", "P"};
    mamba_projected_slot.shape_rank = 3;
    alias_records[0].slots.push_back(mamba_projected_slot);
    dsl::BlockSchemaSlotSummary mamba_hidden_slot;
    mamba_hidden_slot.name = "hidden_states";
    mamba_hidden_slot.kind = "activation";
    mamba_hidden_slot.distribution_kind = "replicated";
    mamba_hidden_slot.shape_dims = {"B", "I", "T"};
    mamba_hidden_slot.shape_rank = 3;
    alias_records[0].slots.push_back(mamba_hidden_slot);
    dsl::BlockSchemaSlotSummary mamba_conv_slot;
    mamba_conv_slot.name = "conv_out";
    mamba_conv_slot.kind = "activation";
    mamba_conv_slot.distribution_kind = "replicated";
    mamba_conv_slot.shape_dims = {"B", "D_conv", "T"};
    mamba_conv_slot.shape_rank = 3;
    alias_records[0].slots.push_back(mamba_conv_slot);
    dsl::BlockSchemaSlotSummary mamba_state_slot;
    mamba_state_slot.name = "ssm_state";
    mamba_state_slot.kind = "activation";
    mamba_state_slot.distribution_kind = "replicated";
    mamba_state_slot.shape_dims = {"B", "H", "D", "N"};
    mamba_state_slot.shape_rank = 4;
    alias_records[0].slots.push_back(mamba_state_slot);
    alias_records[0].slot_count = 11;
    alias_records[0].param_slots = 4;
    alias_records[0].activation_slots = 7;
    const auto alias_plan = dsl::BufferPlan::build(cfg,
                                                   runtime_config,
                                                   options,
                                                   registry,
                                                   /*lora_only_mode=*/false,
                                                   /*B=*/2,
                                                   /*T=*/3,
                                                   ETensorDType::BF16,
                                                   ETensorDType::BF16,
                                                   &alias_records);
    REQUIRE(alias_plan.schema_layers[0].slots[0].resolved_shape == std::vector<long>{16, 8});
    REQUIRE(alias_plan.schema_layers[0].slots[1].resolved_shape == std::vector<long>{4, 8});
    REQUIRE(alias_plan.schema_layers[0].slots[2].resolved_shape == std::vector<long>{384, 8});
    REQUIRE(alias_plan.schema_layers[0].slots[3].resolved_shape == std::vector<long>{8, 128});
    REQUIRE(alias_plan.schema_layers[0].slots[4].resolved_shape == std::vector<long>{2, 384, 4});
    REQUIRE(alias_plan.schema_layers[0].slots[5].resolved_shape == std::vector<long>{2, 3, 12});
    REQUIRE(alias_plan.schema_layers[0].slots[6].shape_dynamic);
    REQUIRE(alias_plan.schema_layers[0].slots[7].resolved_shape == std::vector<long>{2, 3, 648});
    REQUIRE(alias_plan.schema_layers[0].slots[8].resolved_shape == std::vector<long>{2, 256, 3});
    REQUIRE(alias_plan.schema_layers[0].slots[9].resolved_shape == std::vector<long>{2, 384, 3});
    REQUIRE(alias_plan.schema_layers[0].slots[10].resolved_shape == std::vector<long>{2, 8, 32, 16});
    REQUIRE(alias_plan.schema_dynamic_activation_shape_slots == 1);
    REQUIRE(alias_plan.schema_unresolved_activation_shape_slots == 1);

    auto savings_records = schema_records;
    savings_records.push_back(schema_records[0]);
    savings_records[1].layer = 1;
    savings_records[1].slots.clear();
    dsl::BlockSchemaSlotSummary small_activation;
    small_activation.name = "small_state";
    small_activation.kind = "activation";
    small_activation.distribution_kind = "replicated";
    small_activation.shape_dims = {"B", "T", "4"};
    small_activation.shape_rank = 3;
    savings_records[1].slots.push_back(small_activation);
    savings_records[1].slot_count = 1;
    savings_records[1].param_slots = 0;
    savings_records[1].activation_slots = 1;
    PretrainedConfig two_layer_cfg = cfg;
    two_layer_cfg.NumLayers = 2;
    const auto savings_plan = dsl::BufferPlan::build(two_layer_cfg,
                                                     runtime_config,
                                                     options,
                                                     registry,
                                                     /*lora_only_mode=*/false,
                                                     /*B=*/2,
                                                     /*T=*/3,
                                                     ETensorDType::BF16,
                                                     ETensorDType::BF16,
                                                     &savings_records);
    REQUIRE(savings_plan.schema_resolved_activation_shape_bytes == 144);
    REQUIRE(savings_plan.schema_max_layer_activation_shape_bytes == 96);
    REQUIRE(savings_plan.schema_legacy_max_activation_shape_bytes == 192);
    REQUIRE(savings_plan.schema_activation_shape_savings_bytes == 48);

    REQUIRE(plan.schema_layer(0) == &plan.schema_layers[0]);
    REQUIRE(plan.schema_layer(1) == nullptr);
    REQUIRE(plan.schema_layer_has_slot(0, "experts_gate_up"));
    REQUIRE_FALSE(plan.schema_layer_has_slot(0, "missing_slot"));
    const auto* gate_up_slot = plan.schema_slot(0, "experts_gate_up");
    REQUIRE(gate_up_slot);
    REQUIRE(gate_up_slot->grouped);
    REQUIRE(gate_up_slot->streaming_prefetch_distance == 1);
    dsl::ActivationLayoutIR layout;
    dsl::ActivationSlotIR permuted;
    permuted.name = "permuted_input";
    layout.slots.push_back(permuted);
    dsl::TensorSlotRegistry parity_registry;
    parity_registry.init_from_layout(layout);
    REQUIRE(plan.schema_activation_slots_missing_from_registry(parity_registry).empty());
    const auto parity_plan = dsl::BufferPlan::build(cfg,
                                                    runtime_config,
                                                    options,
                                                    parity_registry,
                                                    /*lora_only_mode=*/false,
                                                    /*B=*/2,
                                                    /*T=*/3,
                                                    ETensorDType::BF16,
                                                    ETensorDType::BF16,
                                                    &schema_records);
    REQUIRE(parity_plan.schema_registry_registered_activation_slots == 1);
    REQUIRE(parity_plan.schema_registry_missing_activation_slots == 0);
    REQUIRE(parity_plan.schema_registry_save_for_backward_activation_slots == 0);
    REQUIRE(parity_plan.schema_registry_save_for_backward_mismatch_slots == 0);
    REQUIRE(parity_plan.schema_layers[0].registry_registered_activation_slots == 1);
    REQUIRE(parity_plan.schema_layers[0].registry_missing_activation_slots == 0);
    REQUIRE(parity_plan.schema_layers[0].registry_save_for_backward_activation_slots == 0);
    REQUIRE(parity_plan.schema_layers[0].registry_save_for_backward_mismatch_slots == 0);

    auto save_records = schema_records;
    save_records[0].slots[2].save_for_backward = true;
    save_records[0].slots[2].lifetime = "block";
    auto save_layout = layout;
    save_layout.slots[0].save_for_backward = true;
    dsl::TensorSlotRegistry save_registry;
    save_registry.init_from_layout(save_layout);
    const auto save_plan = dsl::BufferPlan::build(cfg,
                                                  runtime_config,
                                                  options,
                                                  save_registry,
                                                  /*lora_only_mode=*/false,
                                                  /*B=*/2,
                                                  /*T=*/3,
                                                  ETensorDType::BF16,
                                                  ETensorDType::BF16,
                                                  &save_records);
    REQUIRE(save_plan.schema_registry_save_for_backward_activation_slots == 1);
    REQUIRE(save_plan.schema_registry_save_for_backward_mismatch_slots == 0);
    dsl::TensorSlotRegistry no_save_registry;
    no_save_registry.init_from_layout(layout);
    const auto save_mismatch_plan = dsl::BufferPlan::build(cfg,
                                                           runtime_config,
                                                           options,
                                                           no_save_registry,
                                                           /*lora_only_mode=*/false,
                                                           /*B=*/2,
                                                           /*T=*/3,
                                                           ETensorDType::BF16,
                                                           ETensorDType::BF16,
                                                           &save_records);
    REQUIRE(save_mismatch_plan.schema_registry_save_for_backward_activation_slots == 0);
    REQUIRE(save_mismatch_plan.schema_registry_save_for_backward_mismatch_slots == 1);
    const auto save_mismatches =
        save_mismatch_plan.schema_save_for_backward_slots_not_saved_in_registry(no_save_registry);
    REQUIRE(save_mismatches.size() == 1);
    REQUIRE(save_mismatches[0] == "layer0.permuted_input");

    dsl::ActivationLayoutIR missing_layout;
    dsl::TensorSlotRegistry missing_registry;
    missing_registry.init_from_layout(missing_layout);
    const auto missing_schema_slots = plan.schema_activation_slots_missing_from_registry(missing_registry);
    REQUIRE(missing_schema_slots.size() == 1);
    REQUIRE(missing_schema_slots[0] == "layer0.permuted_input");
    const auto missing_plan = dsl::BufferPlan::build(cfg,
                                                     runtime_config,
                                                     options,
                                                     missing_registry,
                                                     /*lora_only_mode=*/false,
                                                     /*B=*/2,
                                                     /*T=*/3,
                                                     ETensorDType::BF16,
                                                     ETensorDType::BF16,
                                                     &schema_records);
    REQUIRE(missing_plan.schema_registry_registered_activation_slots == 0);
    REQUIRE(missing_plan.schema_registry_missing_activation_slots == 1);
    REQUIRE(missing_plan.schema_layers[0].registry_registered_activation_slots == 0);
    REQUIRE(missing_plan.schema_layers[0].registry_missing_activation_slots == 1);
    REQUIRE(plan.schema_layers[0].routing_kind == "topk_softmax");
    REQUIRE(plan.schema_layers[0].routing_topk == 2);
    REQUIRE(plan.schema_layers[0].routing_norm_topk_prob);
    REQUIRE_FALSE(plan.schema_layers[0].routing_scoring_bias);
    REQUIRE(plan.schema_layers[0].routing_shared_experts == 0);
    REQUIRE(plan.schema_layers[0].ep_size_param == "ep_size");
    REQUIRE_FALSE(plan.schema_layers[0].ep_weight_transfer_eligible);
    REQUIRE(plan.schema_layers[0].has_routing);
    REQUIRE(plan.schema_layers[0].has_ep_topology);

    REQUIRE(dsl::hook_event_name(dsl::HookEventKind::AfterAllToAll) == std::string("after_all_to_all"));
    REQUIRE(dsl::schema_id_for_hook_target(schema_records[0]) == "qwen3_dense");

    const auto prefetch_targets = dsl::collect_schema_hook_targets(schema_records, dsl::HookEventKind::BeforeConsume);
    REQUIRE(prefetch_targets.size() == 1);
    REQUIRE(prefetch_targets[0].schema_id == "qwen3_dense");
    REQUIRE(prefetch_targets[0].slot_name == "experts_gate_up");

    const auto comm_targets = dsl::collect_schema_hook_targets(schema_records, dsl::HookEventKind::AfterAllToAll);
    REQUIRE(comm_targets.size() == 1);
    REQUIRE(comm_targets[0].schema_id == "qwen3_dense");
    REQUIRE(comm_targets[0].slot_name == "permuted_input");

    const auto grad_targets = dsl::collect_schema_hook_targets(schema_records, dsl::HookEventKind::AfterReduceScatter);
    REQUIRE(grad_targets.size() == 1);
    REQUIRE(grad_targets[0].schema_id == "qwen3_dense");
    REQUIRE(grad_targets[0].slot_name == "experts_gate_up");

    dsl::HookRegistry hook_registry;
    int callback_count = 0;
    const dsl::HookTarget comm_target{"qwen3_dense", "permuted_input"};
    hook_registry.on_after_all_to_all(
        comm_target,
        "post_dispatch_quantize",
        [&](dsl::HookContext& ctx) {
            REQUIRE(ctx.layer_idx == 0);
            REQUIRE(ctx.target.schema_id == "qwen3_dense");
            REQUIRE(ctx.target.slot_name == "permuted_input");
            ++callback_count;
        },
        /*priority=*/10);
    hook_registry.on_after_all_to_all(
        comm_target,
        "offload_accounting",
        [&](dsl::HookContext&) { ++callback_count; },
        /*priority=*/0);
    hook_registry.on_before_consume({"qwen3_dense", "experts_gate_up"}, "ensure_streamed");

    REQUIRE(hook_registry.size() == 3);
    const auto matches = hook_registry.find(dsl::HookEventKind::AfterAllToAll, "qwen3_dense", "permuted_input");
    REQUIRE(matches.size() == 2);
    REQUIRE(matches[0]->name == "post_dispatch_quantize");
    REQUIRE(matches[0]->distribution_aware);
    REQUIRE(matches[1]->name == "offload_accounting");

    dsl::HookContext hook_ctx;
    hook_ctx.layer_idx = 0;
    hook_ctx.target = comm_target;
    hook_ctx.event = dsl::HookEventKind::AfterAllToAll;
    REQUIRE(hook_registry.dispatch(hook_ctx) == 2);
    REQUIRE(callback_count == 2);

    REQUIRE_THROWS_AS(hook_registry.on_after_produce({"", "qkv"}, "broken"), std::invalid_argument);
}
