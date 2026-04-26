// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/tensor_role.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace dsl {
namespace {

bool contains(std::string_view s, std::string_view needle) {
    return s.find(needle) != std::string_view::npos;
}

std::string strip_saved_prefix(std::string_view name) {
    constexpr std::string_view kSaved = "saved.";
    if (name.rfind(kSaved, 0) == 0) {
        name.remove_prefix(kSaved.size());
    }
    return std::string(name);
}

}  // namespace

const char* tensor_ownership_name(TensorOwnership ownership) {
    switch (ownership) {
        case TensorOwnership::Unknown: return "Unknown";
        case TensorOwnership::Persistent: return "Persistent";
        case TensorOwnership::Stack: return "Stack";
        case TensorOwnership::LoRA: return "LoRA";
        case TensorOwnership::MoE: return "MoE";
        case TensorOwnership::EP: return "EP";
        case TensorOwnership::RopeFreqs: return "RopeFreqs";
        case TensorOwnership::Embedding: return "Embedding";
    }
    return "Unknown";
}

const char* distribution_kind_name(DistributionKind kind) {
    switch (kind) {
        case DistributionKind::Replicated: return "Replicated";
        case DistributionKind::ShardedDim: return "ShardedDim";
        case DistributionKind::ExpertParallel: return "ExpertParallel";
        case DistributionKind::RouterReplicated: return "RouterReplicated";
    }
    return "Replicated";
}

TensorRole infer_tensor_role_from_name(std::string_view raw_name, int block_layer) {
    const std::string owned = strip_saved_prefix(raw_name);
    std::string_view name(owned);

    TensorRole role{};
    role.block_layer = block_layer;

    if (name.rfind("d_", 0) == 0 || name.rfind("d_blocks[", 0) == 0 || name.rfind("d_layer", 0) == 0) {
        role.kind = TensorRoleKind::ActivationGrad;
    } else {
        role.kind = TensorRoleKind::Activation;
    }

    if (contains(name, "rope_freqs") || contains(name, "freq_cis")) {
        role.ownership = TensorOwnership::RopeFreqs;
        return role;
    }
    if (name == "embedding" || name == "embeddings" || name == "embed_tokens" || contains(name, "embed")) {
        role.ownership = TensorOwnership::Embedding;
        return role;
    }

    if (name.rfind("ep_", 0) == 0) {
        role.ownership = TensorOwnership::EP;
        role.dist.kind = DistributionKind::ExpertParallel;
        return role;
    }

    if (name == "moe_expert_offsets" || name == "moe_gather_indices" || contains(name, "moe_") ||
        contains(name, "scatter_indices") || contains(name, "routing_weights") || contains(name, "routing_indices") ||
        contains(name, "router_") || contains(name, "permuted") || contains(name, "expert_") ||
        contains(name, "experts_")) {
        role.ownership = TensorOwnership::MoE;
        if (contains(name, "router_")) {
            role.dist.kind = DistributionKind::RouterReplicated;
        } else {
            role.dist.kind = DistributionKind::ExpertParallel;
        }
        return role;
    }

    role.ownership = (name.rfind("blocks[", 0) == 0 || name.rfind("d_blocks[", 0) == 0) ? TensorOwnership::Stack
                                                                                        : TensorOwnership::Persistent;
    return role;
}

bool tensor_role_is_moe_name(std::string_view name) {
    return infer_tensor_role_from_name(name).is_moe_owned();
}

void tensor_role_parity_check(std::string_view name, bool legacy_value, bool role_value, const char* context) {
    const char* env = std::getenv("SUROGATE_TENSOR_ROLE_PARITY");
    if (!env || std::string_view(env) == "0") {
        return;
    }
    if (legacy_value == role_value) {
        return;
    }
    std::fprintf(stderr,
                 "[TensorRole parity] context=%s name=%.*s legacy=%d role=%d\n",
                 context ? context : "unknown",
                 static_cast<int>(name.size()),
                 name.data(),
                 legacy_value ? 1 : 0,
                 role_value ? 1 : 0);
    if (std::string_view(env) == "abort") {
        std::abort();
    }
}

}  // namespace dsl
