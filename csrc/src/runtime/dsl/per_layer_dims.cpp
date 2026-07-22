// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/per_layer_dims.h"

#include <algorithm>
#include <string>

#include "runtime/executor/graph_executor_utils.h"

namespace dsl {

void derive_per_layer_block_dims(const Graph& graph,
                                 int num_layers,
                                 long global_hq,
                                 long default_hkv,
                                 long default_hs,
                                 long default_dff,
                                 std::vector<BlockTypeDims>& dims) {
    dims.assign(static_cast<std::size_t>(std::max(num_layers, 0)), BlockTypeDims{});
    if (num_layers <= 0) {
        return;
    }
    for (auto& d : dims) {
        d.head_size = default_hs;
        d.qkv_channels = default_hs * (global_hq + 2 * default_hkv);
        d.attn_dim = global_hq * default_hs;
        d.intermediate = default_dff;
        d.mlp_up = 2 * default_dff;
    }

    // Pass 1: out_weight is authoritative for attn_dim/head_size — its
    // columns are exactly Hq*Hs regardless of how Q/K/V are packed. The
    // qkv_weight row count is ambiguous (k_eq_v layers pack Hq+Hkv heads,
    // not Hq+2*Hkv), so derive from it only when no out_weight was seen.
    std::vector<bool> attn_from_out(static_cast<std::size_t>(num_layers), false);
    for (const auto& [name, info] : graph.params) {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) continue;
        if (layer_idx < 0 || layer_idx >= num_layers || info.shape.size() < 2) continue;
        long s0 = (info.shape[0].kind == DimKind::Concrete) ? info.shape[0].value : 0;
        long s1 = (info.shape[1].kind == DimKind::Concrete) ? info.shape[1].value : 0;
        if (s0 == 0 || s1 == 0) continue;
        auto& d = dims[static_cast<std::size_t>(layer_idx)];
        if (field == "out_weight") {
            d.attn_dim = s1;
            if (global_hq > 0) d.head_size = s1 / global_hq;
            attn_from_out[static_cast<std::size_t>(layer_idx)] = true;
        } else if (field == "mlp_down_weight") {
            d.intermediate = s1;
            d.mlp_up = s1;
        } else if (field == "mlp_gate_weight") {
            d.intermediate = s0;
            d.mlp_up = s0;
        }
    }

    // Pass 2: fused QKV / Q-only projections.
    for (const auto& [name, info] : graph.params) {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) continue;
        if (layer_idx < 0 || layer_idx >= num_layers || info.shape.size() < 2) continue;
        long s0 = (info.shape[0].kind == DimKind::Concrete) ? info.shape[0].value : 0;
        long s1 = (info.shape[1].kind == DimKind::Concrete) ? info.shape[1].value : 0;
        if (s0 == 0 || s1 == 0) continue;
        auto& d = dims[static_cast<std::size_t>(layer_idx)];
        const bool from_out = attn_from_out[static_cast<std::size_t>(layer_idx)];
        if (field == "qkv_weight") {
            d.qkv_channels = s0;
            if (from_out) {
                if (global_hq > 0) d.head_size = d.attn_dim / global_hq;
            } else {
                long total_heads = global_hq + 2 * default_hkv;
                if (total_heads > 0) d.head_size = s0 / total_heads;
                d.attn_dim = global_hq * d.head_size;
            }
        } else if (field == "self_attn_q_weight") {
            // Shared-KV block: Q-only projection
            d.qkv_channels = s0;
            if (!from_out) {
                if (global_hq > 0) d.head_size = s0 / global_hq;
                d.attn_dim = s0;
            }
        }
    }

    // Pass 3: separate-projection attention (Laguna) and MLP fixups.
    // - q_norm_weight rows give the exact per-layer head size; dividing
    //   out_weight columns by the *global* head count is wrong for layers
    //   that override the query head count (e.g. Laguna sliding layers).
    // - k/g projections give kv_dim / gate_dim.
    // - mlp_up_weight rows are authoritative for mlp_up (2*M when the
    //   gate+up projections are fused).
    for (const auto& [name, info] : graph.params) {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) continue;
        if (layer_idx < 0 || layer_idx >= num_layers || info.shape.empty()) continue;
        long s0 = (info.shape[0].kind == DimKind::Concrete) ? info.shape[0].value : 0;
        if (s0 == 0) continue;
        auto& d = dims[static_cast<std::size_t>(layer_idx)];
        if (field == "q_norm_weight" && info.shape.size() == 1) {
            d.head_size = s0;
        } else if (field == "k_proj_weight" && info.shape.size() >= 2) {
            d.kv_dim = s0;
        } else if (field == "g_proj_weight" && info.shape.size() >= 2) {
            d.gate_dim = s0;
        } else if (field == "mlp_up_weight" && info.shape.size() >= 2) {
            d.mlp_up = s0;
        }
    }
    // Layers with separate K/V projections have no fused qkv_weight; derive
    // their total QKV width from the actual projection widths.
    for (auto& d : dims) {
        if (d.kv_dim > 0 && d.attn_dim > 0) {
            d.qkv_channels = d.attn_dim + 2 * d.kv_dim;
        }
    }
}

bool per_layer_dims_vary(const std::vector<BlockTypeDims>& dims) {
    for (std::size_t i = 1; i < dims.size(); ++i) {
        if (dims[i].head_size != dims[0].head_size || dims[i].qkv_channels != dims[0].qkv_channels ||
            dims[i].attn_dim != dims[0].attn_dim || dims[i].kv_dim != dims[0].kv_dim ||
            dims[i].gate_dim != dims[0].gate_dim || dims[i].intermediate != dims[0].intermediate) {
            return true;
        }
    }
    return false;
}

}  // namespace dsl
