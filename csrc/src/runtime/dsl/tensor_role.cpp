// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/tensor_role.h"

#include <cctype>
#include <string>

namespace dsl {
namespace {

bool contains(std::string_view s, std::string_view needle) {
    return s.find(needle) != std::string_view::npos;
}

bool contains_ci(std::string_view s, std::string_view needle) {
    if (needle.empty()) return true;
    if (needle.size() > s.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= s.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            const auto a = static_cast<unsigned char>(s[i + j]);
            const auto b = static_cast<unsigned char>(needle[j]);
            if (std::tolower(a) != std::tolower(b)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
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
        case TensorOwnership::LMHead: return "LMHead";
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

const char* quant_state_name(QuantState state) {
    switch (state) {
        case QuantState::None: return "None";
        case QuantState::FP8Pending: return "FP8Pending";
        case QuantState::FP8Ready: return "FP8Ready";
        case QuantState::FP4Ready: return "FP4Ready";
    }
    return "None";
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
    if (name == "lm_head" || name == "lm_head_weight" || contains(name, "lm_head")) {
        role.ownership = TensorOwnership::LMHead;
        return role;
    }

    if (name.rfind("ep_", 0) == 0) {
        role.ownership = TensorOwnership::EP;
        role.dist.kind = DistributionKind::ExpertParallel;
        return role;
    }

    if (name == "moe_expert_offsets" || name == "moe_gather_indices" || contains(name, "moe_") ||
        contains(name, "scatter_indices") || contains(name, "routing_weights") || contains(name, "routing_indices") ||
        contains(name, "gather_indices") || contains(name, "expert_offsets") || contains(name, "router") ||
        contains(name, "permuted") || contains(name, "expert_") || contains(name, "experts_") ||
        contains(name, "experts.")) {
        role.ownership = TensorOwnership::MoE;
        if (contains(name, "router")) {
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

bool tensor_role_is_rope_name(std::string_view name) {
    return infer_tensor_role_from_name(name).is_rope_freq();
}

bool tensor_role_is_expert_parallel_name(std::string_view name) {
    return infer_tensor_role_from_name(name).is_expert_parallel();
}

bool tensor_role_is_moe_side_channel_name(std::string_view name) {
    return contains(name, "scatter_indices") || contains(name, "routing_indices") || contains(name, "gather_indices") ||
           contains(name, "expert_offsets") || contains(name, "ep_recv_scatter");
}

bool tensor_role_is_router_name(std::string_view name) {
    const TensorRole role = infer_tensor_role_from_name(name);
    return role.dist.kind == DistributionKind::RouterReplicated;
}

bool tensor_role_is_embedding_name(std::string_view name) {
    return infer_tensor_role_from_name(name).ownership == TensorOwnership::Embedding;
}

bool tensor_role_is_lm_head_name(std::string_view name) {
    return infer_tensor_role_from_name(name).ownership == TensorOwnership::LMHead;
}

bool tensor_role_is_standalone_gate_name(std::string_view name) {
    return contains(name, "gate") && !contains(name, "mlp");
}

bool tensor_role_is_shared_expert_name(std::string_view name) {
    const TensorRole role = infer_tensor_role_from_name(name);
    return role.is_moe_owned() && contains(name, "shared_expert");
}

bool tensor_role_is_expert_weight_name(std::string_view name) {
    const TensorRole role = infer_tensor_role_from_name(name);
    return role.is_expert_parallel() && (contains(name, "experts") || contains(name, "expert_gate_up") ||
                                         contains(name, "expert_up") || contains(name, "expert_down"));
}

bool tensor_role_is_expert_gate_up_name(std::string_view name) {
    return tensor_role_is_expert_weight_name(name) && contains(name, "gate_up");
}

bool tensor_role_is_expert_up_name(std::string_view name) {
    return tensor_role_is_expert_weight_name(name) && !contains(name, "shared_expert") && !contains(name, "gate_up") &&
           contains(name, "_up");
}

bool tensor_role_is_expert_down_name(std::string_view name) {
    return tensor_role_is_expert_weight_name(name) && contains(name, "down");
}

bool tensor_role_is_expert_bias_name(std::string_view name) {
    const TensorRole role = infer_tensor_role_from_name(name);
    return role.is_expert_parallel() && contains(name, "experts_") && contains(name, "_bias");
}

bool tensor_role_is_fused_qkv_name(std::string_view name) {
    return contains_ci(name, "qkv");
}

bool tensor_role_is_fused_mlp_up_name(std::string_view name) {
    return contains_ci(name, "mlp_up") || contains_ci(name, "gate_up");
}

}  // namespace dsl
