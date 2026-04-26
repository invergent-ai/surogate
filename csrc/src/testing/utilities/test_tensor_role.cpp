// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/tensor_role.h"

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
}
