// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "runtime/core/matmul_context.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/lora/lora_types.h"

using namespace dsl;

namespace {

struct FakeLoRATensor {
    void* Data = nullptr;
};

modules::LoRALayerWeights<FakeLoRATensor> fake_lora_layer() {
    return {FakeLoRATensor{reinterpret_cast<void*>(0x1)}, FakeLoRATensor{reinterpret_cast<void*>(0x2)}};
}

modules::LoRAGroupedLayerWeights<FakeLoRATensor> fake_grouped_lora_layer() {
    return {FakeLoRATensor{reinterpret_cast<void*>(0x3)}, FakeLoRATensor{reinterpret_cast<void*>(0x4)}};
}

}  // namespace

TEST_CASE("TensorRole classifies MoE ownership and distribution conservatively", "[tensor_role]") {
    SECTION("router tensors are MoE-owned and router-replicated") {
        TensorRole role = infer_tensor_role_from_name("blocks[2].router_logits", 2);

        REQUIRE(role.block_layer == 2);
        REQUIRE(role.ownership == TensorOwnership::MoE);
        REQUIRE(role.dist.kind == DistributionKind::RouterReplicated);
        REQUIRE(role.is_moe_owned());
        REQUIRE_FALSE(role.is_expert_parallel());
        REQUIRE(tensor_role_is_moe_name("blocks[2].router_logits"));
        REQUIRE(tensor_role_is_router_name("blocks[2].router_logits"));
        REQUIRE(tensor_role_is_router_name("router"));
        REQUIRE(tensor_role_is_router_name("router_weight"));
        REQUIRE_FALSE(tensor_role_is_expert_parallel_name("blocks[2].router_logits"));
    }

    SECTION("expert tensors are MoE-owned and expert-parallel eligible") {
        TensorRole role = infer_tensor_role_from_name("blocks[4].expert_gate_up", 4);

        REQUIRE(role.block_layer == 4);
        REQUIRE(role.ownership == TensorOwnership::MoE);
        REQUIRE(role.dist.kind == DistributionKind::ExpertParallel);
        REQUIRE(role.is_moe_owned());
        REQUIRE(role.is_expert_parallel());
        REQUIRE(tensor_role_is_expert_parallel_name("blocks[4].expert_gate_up"));
        REQUIRE(tensor_role_is_expert_parallel_name("d_blocks[4].experts_gate_up"));
        REQUIRE(tensor_role_is_shared_expert_name("blocks[4].shared_expert_up"));
        REQUIRE_FALSE(tensor_role_is_shared_expert_name("blocks[4].expert_gate_up"));
        REQUIRE(tensor_role_is_expert_weight_name("blocks[4].expert_down"));
        REQUIRE(tensor_role_is_expert_weight_name("blocks[4].expert_up"));
        REQUIRE(tensor_role_is_expert_weight_name("blocks[4].experts_up"));
        REQUIRE(tensor_role_is_expert_bias_name("blocks[4].experts_down_bias"));
        REQUIRE_FALSE(tensor_role_is_expert_bias_name("blocks[4].shared_expert_up"));
    }

    SECTION("EP side-channel tensors are EP-owned and expert-parallel") {
        TensorRole role = infer_tensor_role_from_name("ep_recv_scatter_indices");

        REQUIRE(role.ownership == TensorOwnership::EP);
        REQUIRE(role.dist.kind == DistributionKind::ExpertParallel);
        REQUIRE(role.is_moe_owned());
        REQUIRE(role.is_expert_parallel());
    }

    SECTION("global MoE side-channels remain MoE-owned") {
        TensorRole role = infer_tensor_role_from_name("moe_expert_offsets");

        REQUIRE(role.ownership == TensorOwnership::MoE);
        REQUIRE(role.dist.kind == DistributionKind::ExpertParallel);
        REQUIRE(role.is_moe_owned());
        REQUIRE(role.is_expert_parallel());
        REQUIRE(tensor_role_is_moe_side_channel_name("moe_expert_offsets"));
        REQUIRE(tensor_role_is_moe_side_channel_name("gather_indices"));
        REQUIRE(tensor_role_is_moe_side_channel_name("expert_offsets"));
        REQUIRE(tensor_role_is_moe_side_channel_name("blocks[0].routing_indices"));
        REQUIRE(tensor_role_is_moe_side_channel_name("ep_recv_scatter_indices"));
        REQUIRE_FALSE(tensor_role_is_moe_side_channel_name("blocks[0].expert_gate_up"));
    }

    SECTION("non-MoE special tensors keep explicit ownership") {
        TensorRole rope = infer_tensor_role_from_name("blocks[1].rope_freqs", 1);
        TensorRole embedding = infer_tensor_role_from_name("embed_tokens");
        TensorRole lm_head = infer_tensor_role_from_name("lm_head");
        TensorRole activation = infer_tensor_role_from_name("blocks[0].attn_out", 0);

        REQUIRE(rope.ownership == TensorOwnership::RopeFreqs);
        REQUIRE(rope.is_rope_freq());
        REQUIRE_FALSE(rope.is_moe_owned());
        REQUIRE(tensor_role_is_rope_name("blocks[1].rope_freqs"));
        REQUIRE(tensor_role_is_rope_name("freq_cis"));
        REQUIRE(embedding.ownership == TensorOwnership::Embedding);
        REQUIRE_FALSE(embedding.is_moe_owned());
        REQUIRE_FALSE(embedding.is_rope_freq());
        REQUIRE(tensor_role_is_embedding_name("embed_tokens"));
        REQUIRE(tensor_role_is_embedding_name("embed_1"));
        REQUIRE_FALSE(tensor_role_is_embedding_name("blocks[0].attn_out"));
        REQUIRE(lm_head.ownership == TensorOwnership::LMHead);
        REQUIRE(tensor_role_is_lm_head_name("lm_head"));
        REQUIRE(tensor_role_is_lm_head_name("model.lm_head.weight"));
        REQUIRE_FALSE(tensor_role_is_lm_head_name("blocks[0].attn_out"));
        REQUIRE(tensor_role_is_standalone_gate_name("gate_weight"));
        REQUIRE_FALSE(tensor_role_is_standalone_gate_name("mlp_gate_weight"));
        REQUIRE(activation.ownership == TensorOwnership::Stack);
        REQUIRE_FALSE(activation.is_moe_owned());
        REQUIRE_FALSE(tensor_role_is_rope_name("blocks[0].attn_out"));
    }
}

TEST_CASE("FP8 ready flag mapping covers dense matmul quant producers", "[tensor_role]") {
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::QKV) == DslRunState::FP8Ready_LN1);
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::MLPUp) == DslRunState::FP8Ready_LN2);
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::AttnOut) == DslRunState::FP8Ready_Att);
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::MLPDown) == DslRunState::FP8Ready_SwiGLU);
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::Embedding) == DslRunState::FP8Ready_None);
    REQUIRE(fp8_ready_flag_for_matmul_op(modules::MatmulOp::LMHead) == DslRunState::FP8Ready_None);
    REQUIRE(std::string(fp8_ready_flag_name(DslRunState::FP8Ready_LN1)) == "LN1");
    REQUIRE(std::string(fp8_ready_flag_name(DslRunState::FP8Ready_Att)) == "AttnOut");
    REQUIRE(quant_state_for_fp8_ready_flag(DslRunState::FP8Ready_None) == QuantState::None);
    REQUIRE(quant_state_for_fp8_ready_flag(DslRunState::FP8Ready_SwiGLU) == QuantState::FP8Ready);
    REQUIRE(std::string(quant_state_name(QuantState::None)) == "None");
    REQUIRE(std::string(quant_state_name(QuantState::FP8Pending)) == "FP8Pending");
    REQUIRE(std::string(quant_state_name(QuantState::FP8Ready)) == "FP8Ready");
    REQUIRE(std::string(quant_state_name(QuantState::FP4Ready)) == "FP4Ready");
}

TEST_CASE("MatmulContext carries optional input TensorRole metadata", "[tensor_role][matmul]") {
    modules::MatmulContext ctx{};
    REQUIRE_FALSE(ctx.has_input_role);
    REQUIRE(ctx.input_role.kind == TensorRoleKind::Unknown);

    ctx.input_role = infer_tensor_role_from_name("blocks[0].ln1_out", 0);
    ctx.input_role.quant_state = QuantState::FP8Ready;
    ctx.has_input_role = true;

    REQUIRE(ctx.has_input_role);
    REQUIRE(ctx.input_role.block_layer == 0);
    REQUIRE(ctx.input_role.quant_state == QuantState::FP8Ready);

    modules::MoeMatmulContext moe_ctx{};
    REQUIRE_FALSE(moe_ctx.has_token_role);
    REQUIRE(moe_ctx.token_role.kind == TensorRoleKind::Unknown);

    moe_ctx.token_role = infer_tensor_role_from_name("blocks[0].moe_x_flat", 0);
    moe_ctx.has_token_role = true;

    REQUIRE(moe_ctx.has_token_role);
    REQUIRE(moe_ctx.token_role.is_moe_owned());
    REQUIRE(moe_ctx.token_role.block_layer == 0);
}

TEST_CASE("LoRA target iteration follows structural block order", "[tensor_role][lora]") {
    modules::LoRABlockWeights<FakeLoRATensor> block;
    block.attention.q = fake_lora_layer();
    block.mlp.down = fake_lora_layer();
    block.moe.experts.resize(1);
    block.moe.experts[0].gate = fake_lora_layer();
    block.moe.experts[0].up = fake_lora_layer();
    block.moe.shared = modules::LoRAMLPWeights<FakeLoRATensor>{};
    block.moe.shared->down = fake_lora_layer();
    block.router = fake_lora_layer();

    std::vector<modules::LoRATargetId> ids;
    modules::for_each_lora_layer_weight(block, [&](modules::LoRATargetId id, auto&) { ids.push_back(id); });

    REQUIRE(ids == std::vector<modules::LoRATargetId>{
                       modules::LoRATargetId::Q,
                       modules::LoRATargetId::Down,
                       modules::LoRATargetId::ExpertGate,
                       modules::LoRATargetId::ExpertUp,
                       modules::LoRATargetId::SharedDown,
                       modules::LoRATargetId::Router,
                   });

    block.moe.use_grouped = true;
    block.moe.grouped.gate_up = fake_grouped_lora_layer();
    ids.clear();
    modules::for_each_lora_layer_weight(block, [&](modules::LoRATargetId id, auto&) { ids.push_back(id); });

    REQUIRE(ids == std::vector<modules::LoRATargetId>{
                       modules::LoRATargetId::Q,
                       modules::LoRATargetId::Down,
                       modules::LoRATargetId::ExpertGateUp,
                       modules::LoRATargetId::SharedDown,
                       modules::LoRATargetId::Router,
                   });

    modules::LoRABlockWeights<FakeLoRATensor> grads;
    grads.attention.q = fake_lora_layer();
    grads.mlp.down = fake_lora_layer();
    grads.moe.use_grouped = true;
    grads.moe.grouped.gate_up = fake_grouped_lora_layer();
    grads.router = fake_lora_layer();

    ids.clear();
    modules::for_each_lora_layer_weight_pair(block, grads, [&](modules::LoRATargetId id, auto&, auto&) {
        ids.push_back(id);
    });

    REQUIRE(ids == std::vector<modules::LoRATargetId>{
                       modules::LoRATargetId::Q,
                       modules::LoRATargetId::Down,
                       modules::LoRATargetId::ExpertGateUp,
                       modules::LoRATargetId::Router,
                   });

    REQUIRE(modules::lora_target_is_expert(modules::LoRATargetId::ExpertGateUp));
    REQUIRE_FALSE(modules::lora_target_is_expert(modules::LoRATargetId::Router));
}

TEST_CASE("CompiledGraph exposes tensor roles by id and name", "[tensor_role][graph]") {
    CompiledGraph graph;
    graph.tensor_meta.resize(2);
    graph.tensor_name_to_id["blocks[0].routing_weights"] = 0;
    graph.tensor_name_to_id["blocks[0].attn_out"] = 1;
    graph.tensor_meta[0].role = infer_tensor_role_from_name("blocks[0].routing_weights", 0);
    graph.tensor_meta[1].role = infer_tensor_role_from_name("blocks[0].attn_out", 0);

    const TensorRole* moe_by_id = graph.role_for_tensor_id(0);
    const TensorRole* moe_by_name = graph.role_for_name("blocks[0].routing_weights");
    const TensorRole* dense_by_name = graph.role_for_name("blocks[0].attn_out");

    REQUIRE(moe_by_id != nullptr);
    REQUIRE(moe_by_id->is_moe_owned());
    REQUIRE(moe_by_name != nullptr);
    REQUIRE(moe_by_name->is_moe_owned());
    REQUIRE(dense_by_name != nullptr);
    REQUIRE_FALSE(dense_by_name->is_moe_owned());
    REQUIRE(graph.role_for_tensor_id(-1) == nullptr);
    REQUIRE(graph.role_for_tensor_id(2) == nullptr);
    REQUIRE(graph.role_for_name("missing") == nullptr);

    graph.tensor_meta[0].role.quant_state = QuantState::FP8Ready;
    graph.tensor_meta[1].role.quant_state = QuantState::FP4Ready;
    REQUIRE(graph.count_tensors_with_quant_state(QuantState::FP8Ready) == 1);
    REQUIRE(graph.count_tensors_with_quant_state(QuantState::FP4Ready) == 1);
    REQUIRE(graph.count_tensors_with_quant_state(QuantState::FP8Pending) == 0);
}

TEST_CASE("CompiledGraph summarizes op descriptor facets", "[tensor_role][graph]") {
    CompiledGraph graph;
    graph.ops.resize(3);
    graph.ops[0].type = CompiledOpType::Matmul;
    graph.ops[0].comm_profile.kind = CommunicationKind::NoComm;
    graph.ops[0].default_caps.flags = OpCapabilityDenseMatmul | OpCapabilityFp8Eligible;
    graph.ops[0].matmul_caps.flags = MatmulCapabilityFp8ForwardEligible | MatmulCapabilityWeightCacheEligible;
    graph.ops[1].type = CompiledOpType::BiasAdd;
    graph.ops[1].comm_profile.kind = CommunicationKind::AllToAllIn;
    graph.ops[1].grouped_semantics.routes_tokens = true;
    graph.ops[1].storage_compat.flags |= StorageCompatibilityCpuPinnedStream;
    graph.ops[1].default_caps.flags = OpCapabilityDenseMatmul;
    graph.ops[2].comm_profile.kind = CommunicationKind::ExpertParallelRouted;
    graph.ops[2].grouped_semantics.is_grouped = true;
    graph.ops[2].grouped_semantics.expert_dim = 0;
    graph.ops[2].default_caps.flags = OpCapabilityGroupedMatmul | OpCapabilityFp8Eligible;
    graph.ops[2].epilogue_support.flags = EpilogueSupportActivation;
    graph.ops[2].moe_caps.flags = MoECapabilityGroupedGemmEligible | MoECapabilityFp8GroupedEligible;

    REQUIRE(graph.count_ops_with_comm(CommunicationKind::NoComm) == 1);
    REQUIRE(graph.count_ops_with_comm(CommunicationKind::AllToAllIn) == 1);
    REQUIRE(graph.count_ops_with_comm(CommunicationKind::ExpertParallelRouted) == 1);
    REQUIRE(graph.count_ops_with_comm(CommunicationKind::AllReduceAfter) == 0);
    REQUIRE(graph.count_grouped_ops() == 1);
    REQUIRE(graph.count_ops_with_capability(OpCapabilityGroupedMatmul) == 1);
    REQUIRE(graph.count_ops_with_capability(OpCapabilityFp8Eligible) == 2);
    REQUIRE(graph.count_ops_with_capability(OpCapabilityFp4Eligible) == 0);
    REQUIRE(graph.count_ops_with_matmul_capability(MatmulCapabilityFp8ForwardEligible) == 1);
    REQUIRE(graph.count_ops_with_matmul_capability(MatmulCapabilityWeightCacheEligible) == 1);
    REQUIRE(graph.count_ops_with_matmul_capability(MatmulCapabilityFp4ForwardEligible) == 0);
    REQUIRE(graph.count_ops_with_epilogue(EpilogueSupportActivation) == 1);
    REQUIRE(graph.count_ops_with_moe_capability(MoECapabilityGroupedGemmEligible) == 1);
    REQUIRE(graph.count_ops_with_moe_capability(MoECapabilityFp8GroupedEligible) == 1);
    REQUIRE(graph.count_ops_with_moe_capability(MoECapabilityFp8BackwardImplemented) == 0);
    REQUIRE(graph.count_ops_supporting_storage(StorageTier::GpuResident) == 3);
    REQUIRE(graph.count_ops_supporting_storage(StorageTier::CpuPinnedStream) == 1);
    REQUIRE(graph.count_fusion_candidate_starts() == 0);

    graph.ops[1].comm_profile.kind = CommunicationKind::NoComm;
    REQUIRE(graph.count_fusion_candidate_starts() == 1);
}
