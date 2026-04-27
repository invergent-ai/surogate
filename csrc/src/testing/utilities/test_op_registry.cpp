// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <utility>

#include "recipes/capability_predicates.h"
#include "runtime/executor/op_registry.h"

namespace dsl {
namespace {

TEST_CASE("op registry descriptor metadata merges without dispatch churn", "[op_registry]") {
    OpDescriptor base;
    base.name = "__test_descriptor_metadata";
    base.type = CompiledOpType::Unknown;
    base.semantic_kind = OpSemanticKind::Dense;
    REQUIRE(OpRegistry::instance().register_op(std::move(base)) == 0);

    OpDescriptor meta;
    meta.name = "__test_descriptor_metadata";
    meta.distribution_kind = DistributionKind::ExpertParallel;
    meta.default_caps.flags = OpCapabilityGroupedMatmul | OpCapabilityFp8Eligible;
    meta.epilogue_support.flags = EpilogueSupportActivation;
    meta.storage_compat.flags = StorageCompatibilityGpuResident | StorageCompatibilityCpuPinnedStream;
    meta.moe_caps.flags = MoECapabilityGroupedGemmEligible | MoECapabilityFp8GroupedEligible;
    meta.moe_caps.expert_storage.flags = StorageCompatibilityGpuResident;
    meta.moe_caps.ep_awareness = EpAwareness::Routed;
    meta.matmul_caps.flags = MatmulCapabilityFp8ForwardEligible | MatmulCapabilityWeightCacheEligible;
    meta.matmul_caps.supported_epilogues.flags = EpilogueSupportActivation;
    meta.matmul_caps.colocate_input = QuantColocation::PrecedingNorm;
    meta.matmul_caps.weight_storage.flags = StorageCompatibilityGpuResident | StorageCompatibilityCpuPinnedStream;
    meta.matmul_caps.recipe_priority = 7;
    meta.comm_profile.kind = CommunicationKind::ExpertParallelRouted;
    meta.grouped_semantics.is_grouped = true;
    meta.grouped_semantics.expert_dim = 0;
    meta.descriptor_flags = 0x4;
    REQUIRE(OpRegistry::instance().register_op(std::move(meta)) == 0);

    const OpDescriptor* desc = OpRegistry::instance().find_by_name("__test_descriptor_metadata");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(desc->default_caps.has(OpCapabilityGroupedMatmul));
    REQUIRE(desc->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE_FALSE(desc->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(desc->epilogue_support.has(EpilogueSupportActivation));
    REQUIRE(desc->storage_compat.supports(StorageTier::GpuResident));
    REQUIRE(desc->storage_compat.supports(StorageTier::CpuPinnedStream));
    REQUIRE_FALSE(desc->storage_compat.supports(StorageTier::NvmeOffload));
    REQUIRE(desc->moe_caps.has(MoECapabilityGroupedGemmEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityFp8GroupedEligible));
    REQUIRE(desc->moe_caps.expert_storage.supports(StorageTier::GpuResident));
    REQUIRE(desc->moe_caps.ep_awareness == EpAwareness::Routed);
    REQUIRE(desc->matmul_caps.has(MatmulCapabilityFp8ForwardEligible));
    REQUIRE(desc->matmul_caps.has(MatmulCapabilityWeightCacheEligible));
    REQUIRE_FALSE(desc->matmul_caps.has(MatmulCapabilityFp8BackwardEligible));
    REQUIRE(desc->matmul_caps.supported_epilogues.has(EpilogueSupportActivation));
    REQUIRE(desc->matmul_caps.colocate_input == QuantColocation::PrecedingNorm);
    REQUIRE(desc->matmul_caps.weight_storage.supports(StorageTier::CpuPinnedStream));
    REQUIRE(desc->matmul_caps.recipe_priority == 7);
    REQUIRE(desc->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(desc->grouped_semantics.is_grouped);
    REQUIRE(desc->grouped_semantics.expert_dim == 0);
    REQUIRE(desc->descriptor_flags == 0x4);
    REQUIRE(std::string(op_semantic_kind_name(desc->semantic_kind)) == "Dense");
    REQUIRE(std::string(communication_kind_name(desc->comm_profile.kind)) == "ExpertParallelRouted");
    REQUIRE(op_capability_flags_string(desc->default_caps) == "GroupedMatmul|FP8Eligible");
    REQUIRE(moe_capability_flags_string(desc->moe_caps) == "GroupedGemmEligible|FP8GroupedEligible");
    REQUIRE(std::string(ep_awareness_name(desc->moe_caps.ep_awareness)) == "Routed");
    REQUIRE(matmul_capability_flags_string(desc->matmul_caps) == "FP8ForwardEligible|WeightCacheEligible");
    REQUIRE(std::string(quant_colocation_name(desc->matmul_caps.colocate_input)) == "PrecedingNorm");
    REQUIRE(epilogue_support_flags_string(desc->epilogue_support) == "Activation");
    REQUIRE(storage_compatibility_flags_string(desc->storage_compat) == "GpuResident|CpuPinnedStream");
}

TEST_CASE("moe ops carry first-month descriptor metadata", "[op_registry]") {
    const OpDescriptor* desc = OpRegistry::instance().find_by_name("moe_grouped_gemm");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::MoE);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(desc->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(desc->grouped_semantics.is_grouped);
    REQUIRE_FALSE(desc->grouped_semantics.routes_tokens);
    REQUIRE(desc->grouped_semantics.expert_dim == 0);
    REQUIRE(desc->grouped_semantics.ep_aware);
    REQUIRE(desc->default_caps.has(OpCapabilityGroupedMatmul));
    REQUIRE(desc->default_caps.has(OpCapabilityMoeRouted));
    REQUIRE(desc->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(desc->default_caps.has(OpCapabilityWeightCacheEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityGroupedGemmEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityFp8GroupedEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityFp4GroupedEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityCudnnMoeGraphEligible));
    REQUIRE(desc->moe_caps.has(MoECapabilityPerExpertQuant));
    REQUIRE_FALSE(desc->moe_caps.has(MoECapabilityFp8BackwardImplemented));
    REQUIRE_FALSE(desc->moe_caps.has(MoECapabilityNvfp4NoFallback));
    REQUIRE(desc->moe_caps.ep_awareness == EpAwareness::Routed);
    REQUIRE(desc->storage_compat.supports(StorageTier::GpuResident));
    REQUIRE_FALSE(desc->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* gate_up = OpRegistry::instance().find_by_name("moe_grouped_gemm_gate_up");
    REQUIRE(gate_up != nullptr);
    REQUIRE(gate_up == OpRegistry::instance().find(CompiledOpType::MoEGroupedGemmGateUp));
    REQUIRE(gate_up->forward_fn != nullptr);
    REQUIRE(gate_up->epilogue_support.has(EpilogueSupportActivation));

    const OpDescriptor* grouped_bwd = OpRegistry::instance().find_by_name("moe_grouped_gemm_backward");
    REQUIRE(grouped_bwd != nullptr);
    REQUIRE(grouped_bwd == OpRegistry::instance().find(CompiledOpType::MoEGroupedGemmBackward));
    REQUIRE(grouped_bwd->backward_fn != nullptr);
    REQUIRE(grouped_bwd->semantic_kind == OpSemanticKind::MoE);
    REQUIRE(grouped_bwd->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(grouped_bwd->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(grouped_bwd->grouped_semantics.is_grouped);
    REQUIRE(grouped_bwd->grouped_semantics.ep_aware);
    REQUIRE(grouped_bwd->default_caps.has(OpCapabilityGroupedMatmul));
    REQUIRE(grouped_bwd->default_caps.has(OpCapabilityMoeRouted));
    REQUIRE(grouped_bwd->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(grouped_bwd->default_caps.has(OpCapabilityWeightCacheEligible));
    REQUIRE(grouped_bwd->moe_caps.has(MoECapabilityGroupedGemmEligible));
    REQUIRE(grouped_bwd->moe_caps.has(MoECapabilityFp8GroupedEligible));
    REQUIRE_FALSE(grouped_bwd->moe_caps.has(MoECapabilityFp8BackwardImplemented));
}

TEST_CASE("moe utility ops carry routing descriptor metadata", "[op_registry]") {
    const OpDescriptor* router = OpRegistry::instance().find_by_name("moe_topk");
    REQUIRE(router != nullptr);
    REQUIRE(router == OpRegistry::instance().find(CompiledOpType::MoETopK));
    REQUIRE(router->forward_fn != nullptr);
    REQUIRE(router->semantic_kind == OpSemanticKind::MoE);
    REQUIRE(router->distribution_kind == DistributionKind::RouterReplicated);
    REQUIRE(router->comm_profile.kind == CommunicationKind::NoComm);
    REQUIRE_FALSE(router->grouped_semantics.routes_tokens);

    const OpDescriptor* permute = OpRegistry::instance().find_by_name("moe_permute");
    REQUIRE(permute != nullptr);
    REQUIRE(permute == OpRegistry::instance().find(CompiledOpType::MoEPermute));
    REQUIRE(permute->forward_fn != nullptr);
    REQUIRE(permute->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(permute->comm_profile.kind == CommunicationKind::NoComm);
    REQUIRE(permute->grouped_semantics.routes_tokens);
    REQUIRE(permute->grouped_semantics.ep_aware);

    const OpDescriptor* bias = OpRegistry::instance().find_by_name("moe_expert_bias_add");
    REQUIRE(bias != nullptr);
    REQUIRE(bias == OpRegistry::instance().find(CompiledOpType::MoEExpertBiasAdd));
    REQUIRE(bias->forward_fn != nullptr);
    REQUIRE(bias->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(bias->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(bias->grouped_semantics.ep_aware);
    REQUIRE(bias->epilogue_support.has(EpilogueSupportBias));
}

TEST_CASE("ep ops carry communication profile metadata", "[op_registry]") {
    const OpDescriptor* dispatch = OpRegistry::instance().find_by_name("ep_dispatch");
    REQUIRE(dispatch != nullptr);
    REQUIRE(dispatch == OpRegistry::instance().find(CompiledOpType::EpDispatch));
    REQUIRE(dispatch->forward_fn != nullptr);
    REQUIRE(dispatch->semantic_kind == OpSemanticKind::Collective);
    REQUIRE(dispatch->comm_profile.kind == CommunicationKind::AllToAllIn);
    REQUIRE(dispatch->comm_profile.can_overlap_with_compute);
    REQUIRE(dispatch->grouped_semantics.routes_tokens);
    REQUIRE(dispatch->grouped_semantics.ep_aware);

    const OpDescriptor* combine = OpRegistry::instance().find_by_name("ep_combine");
    REQUIRE(combine != nullptr);
    REQUIRE(combine == OpRegistry::instance().find(CompiledOpType::EpCombine));
    REQUIRE(combine->forward_fn != nullptr);
    REQUIRE(combine->comm_profile.kind == CommunicationKind::AllToAllOut);
}

TEST_CASE("mamba and gated-delta ops carry no-comm descriptor metadata", "[op_registry]") {
    const OpDescriptor* mamba_scan = OpRegistry::instance().find_by_name("mamba_ssm_scan");
    REQUIRE(mamba_scan != nullptr);
    REQUIRE(mamba_scan == OpRegistry::instance().find(CompiledOpType::MambaSsmScan));
    REQUIRE(mamba_scan->forward_fn != nullptr);
    REQUIRE(mamba_scan->semantic_kind == OpSemanticKind::Sequence);
    REQUIRE(mamba_scan->distribution_kind == DistributionKind::Replicated);
    REQUIRE(mamba_scan->comm_profile.kind == CommunicationKind::NoComm);

    const OpDescriptor* mamba_out = OpRegistry::instance().find_by_name("mamba_out_proj");
    REQUIRE(mamba_out != nullptr);
    REQUIRE(mamba_out->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(mamba_out->comm_profile.kind == CommunicationKind::NoComm);
    REQUIRE(mamba_out->default_caps.has(OpCapabilityDenseMatmul));
    REQUIRE(mamba_out->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(mamba_out->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(mamba_out->matmul_caps.has(MatmulCapabilityFp8ForwardEligible));
    REQUIRE(mamba_out->matmul_caps.has(MatmulCapabilityFp4ForwardEligible));
    REQUIRE(mamba_out->matmul_caps.has(MatmulCapabilityWeightCacheEligible));
    REQUIRE(mamba_out->matmul_caps.colocate_input == QuantColocation::PrecedingActivation);
    REQUIRE(mamba_out->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* gated_delta = OpRegistry::instance().find_by_name("chunk_gated_delta_rule");
    REQUIRE(gated_delta != nullptr);
    REQUIRE(gated_delta == OpRegistry::instance().find(CompiledOpType::ChunkGatedDeltaRule));
    REQUIRE(gated_delta->forward_fn != nullptr);
    REQUIRE(gated_delta->semantic_kind == OpSemanticKind::Sequence);
    REQUIRE(gated_delta->comm_profile.kind == CommunicationKind::NoComm);
}

TEST_CASE("core transformer ops carry descriptor metadata", "[op_registry]") {
    const OpDescriptor* matmul = OpRegistry::instance().find_by_name("matmul");
    REQUIRE(matmul != nullptr);
    REQUIRE(matmul == OpRegistry::instance().find(CompiledOpType::Matmul));
    REQUIRE(matmul->forward_fn != nullptr);
    REQUIRE(matmul->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(matmul->comm_profile.kind == CommunicationKind::NoComm);
    REQUIRE(matmul->default_caps.has(OpCapabilityDenseMatmul));
    REQUIRE(matmul->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(matmul->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(matmul->default_caps.has(OpCapabilityLoRACompatible));
    REQUIRE(matmul->matmul_caps.has(MatmulCapabilityFp8ForwardEligible));
    REQUIRE(matmul->matmul_caps.has(MatmulCapabilityFp4ForwardEligible));
    REQUIRE(matmul->matmul_caps.has(MatmulCapabilityWeightCacheEligible));
    REQUIRE(matmul->matmul_caps.colocate_input == QuantColocation::None);
    REQUIRE(matmul->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* matmul_bias = OpRegistry::instance().find_by_name("matmul_bias");
    REQUIRE(matmul_bias != nullptr);
    REQUIRE(matmul_bias->epilogue_support.has(EpilogueSupportBias));
    REQUIRE(matmul_bias->matmul_caps.supported_epilogues.has(EpilogueSupportBias));

    const OpDescriptor* matmul_backward = OpRegistry::instance().find_by_name("matmul_backward");
    REQUIRE(matmul_backward != nullptr);
    REQUIRE(matmul_backward == OpRegistry::instance().find(CompiledOpType::MatmulBackward));
    REQUIRE(matmul_backward->backward_fn != nullptr);
    REQUIRE(matmul_backward->default_caps.has(OpCapabilityDenseMatmul));
    REQUIRE(matmul_backward->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(matmul_backward->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(matmul_backward->matmul_caps.has(MatmulCapabilityFp8BackwardEligible));
    REQUIRE(matmul_backward->matmul_caps.has(MatmulCapabilityFp4BackwardEligible));
    REQUIRE_FALSE(matmul_backward->matmul_caps.has(MatmulCapabilityFp8ForwardEligible));
    REQUIRE(matmul_backward->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* matmul_swiglu = OpRegistry::instance().find_by_name("matmul_swiglu");
    REQUIRE(matmul_swiglu != nullptr);
    REQUIRE(matmul_swiglu == OpRegistry::instance().find(CompiledOpType::MatmulSwiGLU));
    REQUIRE(matmul_swiglu->forward_fn != nullptr);
    REQUIRE(matmul_swiglu->default_caps.has(OpCapabilityDenseMatmul));
    REQUIRE(matmul_swiglu->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(matmul_swiglu->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(matmul_swiglu->epilogue_support.has(EpilogueSupportActivation));
    REQUIRE(matmul_swiglu->matmul_caps.has(MatmulCapabilityFp8ForwardEligible));
    REQUIRE(matmul_swiglu->matmul_caps.supported_epilogues.has(EpilogueSupportActivation));
    REQUIRE(matmul_swiglu->matmul_caps.colocate_input == QuantColocation::PrecedingNorm);
    REQUIRE(matmul_swiglu->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* rmsnorm = OpRegistry::instance().find_by_name("rmsnorm");
    REQUIRE(rmsnorm != nullptr);
    REQUIRE(rmsnorm->semantic_kind == OpSemanticKind::Normalization);

    const OpDescriptor* flash = OpRegistry::instance().find_by_name("flash_attention");
    REQUIRE(flash != nullptr);
    REQUIRE(flash->semantic_kind == OpSemanticKind::Attention);

    const OpDescriptor* loss = OpRegistry::instance().find_by_name("cross_entropy_loss");
    REQUIRE(loss != nullptr);
    REQUIRE(loss->semantic_kind == OpSemanticKind::Loss);
}

TEST_CASE("recipe capability predicates preserve legacy fallback semantics", "[op_registry]") {
    OpCapabilities unannotated{};
    REQUIRE(recipes::descriptor_allows_fp8(unannotated));
    REQUIRE(recipes::descriptor_allows_fp4(unannotated));
    REQUIRE(recipes::descriptor_allows_fp8(unannotated, "test"));
    REQUIRE(recipes::descriptor_allows_fp4(unannotated, "test"));

    OpCapabilities fp8_only{OpCapabilityFp8Eligible};
    REQUIRE(recipes::descriptor_allows_fp8(fp8_only));
    REQUIRE_FALSE(recipes::descriptor_allows_fp4(fp8_only));
    REQUIRE(recipes::descriptor_allows_fp8(fp8_only, "test"));
    REQUIRE_FALSE(recipes::descriptor_allows_fp4(fp8_only, "test"));

    OpCapabilities fp4_only{OpCapabilityFp4Eligible};
    REQUIRE_FALSE(recipes::descriptor_allows_fp8(fp4_only));
    REQUIRE(recipes::descriptor_allows_fp4(fp4_only));
    REQUIRE_FALSE(recipes::descriptor_allows_fp8(fp4_only, "test"));
    REQUIRE(recipes::descriptor_allows_fp4(fp4_only, "test"));

    MoECapabilities moe_unannotated{};
    REQUIRE(recipes::descriptor_allows_moe_fp8_grouped(moe_unannotated));
    REQUIRE(recipes::descriptor_allows_moe_fp4_grouped(moe_unannotated));
    REQUIRE(recipes::descriptor_has_moe_fp8_backward(moe_unannotated, "test"));

    MoECapabilities moe_forward_only{MoECapabilityFp8GroupedEligible | MoECapabilityFp4GroupedEligible};
    REQUIRE(recipes::descriptor_allows_moe_fp8_grouped(moe_forward_only));
    REQUIRE(recipes::descriptor_allows_moe_fp4_grouped(moe_forward_only));
    REQUIRE_FALSE(recipes::descriptor_has_moe_fp8_backward(moe_forward_only, "test"));

    MatmulCapabilities matmul_unannotated{};
    REQUIRE(recipes::descriptor_allows_matmul_fp8_forward(matmul_unannotated, "test"));
    REQUIRE(recipes::descriptor_allows_matmul_fp8_backward(matmul_unannotated, "test"));
    REQUIRE(recipes::descriptor_allows_matmul_fp4_forward(matmul_unannotated, "test"));
    REQUIRE(recipes::descriptor_allows_matmul_fp4_backward(matmul_unannotated, "test"));

    MatmulCapabilities matmul_forward_only{MatmulCapabilityFp8ForwardEligible | MatmulCapabilityFp4ForwardEligible |
                                           MatmulCapabilityWeightCacheEligible};
    REQUIRE(recipes::descriptor_allows_matmul_fp8_forward(matmul_forward_only, "test"));
    REQUIRE_FALSE(recipes::descriptor_allows_matmul_fp8_backward(matmul_forward_only, "test"));
    REQUIRE(recipes::descriptor_allows_matmul_fp4_forward(matmul_forward_only, "test"));
    REQUIRE_FALSE(recipes::descriptor_allows_matmul_fp4_backward(matmul_forward_only, "test"));

    TensorRole fp8_ready_input{};
    fp8_ready_input.quant_state = QuantState::FP8Ready;
    TensorRole bf16_input{};
    bf16_input.quant_state = QuantState::None;
    MatmulCapabilities colocated_forward{MatmulCapabilityFp8ForwardEligible};
    colocated_forward.colocate_input = QuantColocation::PrecedingNorm;
    REQUIRE(recipes::descriptor_allows_matmul_fp8_colocated_forward(colocated_forward, &fp8_ready_input, "test"));
    REQUIRE_FALSE(recipes::descriptor_allows_matmul_fp8_colocated_forward(colocated_forward, &bf16_input, "test"));
    REQUIRE_FALSE(recipes::descriptor_allows_matmul_fp8_colocated_forward(colocated_forward, nullptr, "test"));
}

}  // namespace
}  // namespace dsl
