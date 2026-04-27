// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/executor/graph_executor_helpers.h"

using namespace dsl;

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
        REQUIRE(tensor_role_is_moe_side_channel_name("blocks[0].routing_indices"));
        REQUIRE(tensor_role_is_moe_side_channel_name("ep_recv_scatter_indices"));
        REQUIRE_FALSE(tensor_role_is_moe_side_channel_name("blocks[0].expert_gate_up"));
    }

    SECTION("non-MoE special tensors keep explicit ownership") {
        TensorRole rope = infer_tensor_role_from_name("blocks[1].rope_freqs", 1);
        TensorRole embedding = infer_tensor_role_from_name("embed_tokens");
        TensorRole activation = infer_tensor_role_from_name("blocks[0].attn_out", 0);

        REQUIRE(rope.ownership == TensorOwnership::RopeFreqs);
        REQUIRE(rope.is_rope_freq());
        REQUIRE_FALSE(rope.is_moe_owned());
        REQUIRE(tensor_role_is_rope_name("blocks[1].rope_freqs"));
        REQUIRE(tensor_role_is_rope_name("freq_cis"));
        REQUIRE(embedding.ownership == TensorOwnership::Embedding);
        REQUIRE_FALSE(embedding.is_moe_owned());
        REQUIRE_FALSE(embedding.is_rope_freq());
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
