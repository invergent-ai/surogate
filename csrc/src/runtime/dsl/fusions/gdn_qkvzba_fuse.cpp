// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Fusion pass: GDN QKVZ+BA projection merge for Qwen3.5 linear-attention decode
//
// Detects the 4-matmul pattern in the GatedDeltaNet linear block:
//   matmul(ln1_flat, qkv_weight)  [role=gdn_in_proj_qkv]  → qkv_flat → view
//   matmul(ln1_flat, z_weight)    [role=gdn_in_proj_z]     → z_flat   → view
//   matmul(ln1_flat, b_weight)    [role=gdn_in_proj_b]     → b_flat   → view
//   matmul(ln1_flat, a_weight)    [role=gdn_in_proj_a]     → a_flat   → view
//
// Rewrites into a single compound op:
//   gdn_fused_proj(ln1_flat, qkv_weight, z_weight, b_weight, a_weight)
//     → mixed_qkv, z_flat, b_flat, a_flat
//
// The dispatch internally merges weights (cached), does 2 matmuls instead of 4,
// and uses a JIT Triton kernel for the post-matmul split/reshape.

#include "runtime/dsl/fusions/fusion_pass.h"
#include "runtime/dsl/graph_executor_utils.h"

#include <algorithm>
#include <cstdio>
#include <set>

namespace dsl {
namespace {

/// Check if an operation has a specific role attr.
bool has_role(const Operation& op, const char* role) {
    if (auto* attr = find_attr(op.attrs, "role")) {
        if (auto s = attr_string(*attr)) {
            return *s == role;
        }
    }
    return false;
}

/// Find the index of the first op with a given role, starting at `from`.
int find_op_with_role(const std::vector<Operation>& ops, const char* role,
                      std::size_t from = 0) {
    for (std::size_t i = from; i < ops.size(); ++i) {
        if (has_role(ops[i], role)) return static_cast<int>(i);
    }
    return -1;
}

/// Find the view op that consumes a matmul's output (matmul → view chain).
/// Returns index or -1.
int find_view_of(const std::vector<Operation>& ops, int matmul_idx) {
    if (matmul_idx < 0 || ops[matmul_idx].outputs.empty()) return -1;
    const std::string& matmul_out = ops[matmul_idx].outputs[0];
    for (std::size_t i = matmul_idx + 1; i < ops.size(); ++i) {
        const std::string type = op_type_for_rewrite(ops[i]);
        if ((type == "view" || type == "reshape") &&
            ops[i].inputs.size() == 1 &&
            ops[i].inputs[0] == matmul_out) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool run_gdn_qkvzba_fuse(Graph& graph, const FusionContext& ctx) {
    const auto& ops = graph.operations;
    if (ops.size() < 8) return false;

    const bool debug = (std::getenv("SUROGATE_FUSION_DEBUG") != nullptr);

    // Find the 4 matmuls by role
    int idx_qkv = find_op_with_role(ops, "gdn_in_proj_qkv");
    if (idx_qkv < 0) return false;

    int idx_z = find_op_with_role(ops, "gdn_in_proj_z", idx_qkv + 1);
    int idx_b = find_op_with_role(ops, "gdn_in_proj_b", idx_qkv + 1);
    int idx_a = find_op_with_role(ops, "gdn_in_proj_a", idx_qkv + 1);

    if (idx_z < 0 || idx_b < 0 || idx_a < 0) return false;

    // Verify all share the same first input (ln1_flat)
    const std::string& shared_input = ops[idx_qkv].inputs[0];
    if (ops[idx_z].inputs[0] != shared_input ||
        ops[idx_b].inputs[0] != shared_input ||
        ops[idx_a].inputs[0] != shared_input) {
        return false;
    }

    // Find the view ops that consume each matmul's output
    // (z → view, b → view, a → view; qkv → view is before conv1d)
    int view_z = find_view_of(ops, idx_z);
    int view_b = find_view_of(ops, idx_b);
    int view_a = find_view_of(ops, idx_a);

    if (view_z < 0 || view_b < 0 || view_a < 0) return false;

    if (debug) {
        fprintf(stderr, "[fusion] gdn_qkvzba_fuse: pattern matched "
                "(qkv=%d, z=%d[v%d], b=%d[v%d], a=%d[v%d])\n",
                idx_qkv, idx_z, view_z, idx_b, view_b, idx_a, view_a);
        fprintf(stderr, "[fusion]   shared_input: %s\n", shared_input.c_str());
        fprintf(stderr, "[fusion]   qkv_weight: %s -> %s\n",
                ops[idx_qkv].inputs[1].c_str(), ops[idx_qkv].outputs[0].c_str());
        fprintf(stderr, "[fusion]   z_weight: %s -> %s (view-> %s)\n",
                ops[idx_z].inputs[1].c_str(), ops[idx_z].outputs[0].c_str(),
                ops[view_z].outputs[0].c_str());
        fprintf(stderr, "[fusion]   b_weight: %s -> %s (view-> %s)\n",
                ops[idx_b].inputs[1].c_str(), ops[idx_b].outputs[0].c_str(),
                ops[view_b].outputs[0].c_str());
        fprintf(stderr, "[fusion]   a_weight: %s -> %s (view-> %s)\n",
                ops[idx_a].inputs[1].c_str(), ops[idx_a].outputs[0].c_str(),
                ops[view_a].outputs[0].c_str());
    }

    // --- Build the fused op ---
    // gdn_fused_proj(ln1_flat, qkv_weight, z_weight, b_weight, a_weight)
    //   → mixed_qkv, z_flat, b_flat, a_flat
    //
    // mixed_qkv replaces the original qkv matmul output (feeds into conv1d)
    // z_flat, b_flat, a_flat replace the original view outputs
    // (the original matmul outputs and intermediate views are removed)

    Operation fused;
    fused.name = "gdn_fused_proj";
    fused.kernel_type = "gdn_fused_proj";
    fused.id = ops[idx_qkv].id;

    // Inputs: ln1_flat, qkv_weight, z_weight, b_weight, a_weight
    fused.inputs.push_back(shared_input);
    fused.inputs.push_back(ops[idx_qkv].inputs[1]);  // qkv_weight
    fused.inputs.push_back(ops[idx_z].inputs[1]);     // z_weight
    fused.inputs.push_back(ops[idx_b].inputs[1]);     // b_weight
    fused.inputs.push_back(ops[idx_a].inputs[1]);     // a_weight

    // Outputs: flat tensors matching what the original matmuls produced.
    // The existing view ops (which we keep) will reshape to [B, T, ...].
    fused.outputs.push_back(ops[idx_qkv].outputs[0]);    // qkv_flat [BT, ConvDim]
    fused.outputs.push_back(ops[idx_z].outputs[0]);       // z_flat [BT, Hv*Vd]
    fused.outputs.push_back(ops[idx_b].outputs[0]);       // b_flat [BT, Hv]
    fused.outputs.push_back(ops[idx_a].outputs[0]);       // a_flat [BT, Hv]

    // Carry forward transpose attr from the matmul
    fused.attrs = ops[idx_qkv].attrs;

    // --- Rewrite the op list ---
    // Remove the 3 extra matmuls (z, b, a) but KEEP their view ops.
    // The fused op produces flat outputs; existing views reshape to [B, T, ...].
    std::set<int> remove_indices;
    remove_indices.insert(idx_z);
    remove_indices.insert(idx_b);
    remove_indices.insert(idx_a);

    std::vector<Operation> rewritten;
    rewritten.reserve(ops.size());
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (static_cast<int>(i) == idx_qkv) {
            // Replace the qkv matmul with the fused op
            rewritten.push_back(std::move(fused));
        } else if (remove_indices.count(static_cast<int>(i))) {
            // Skip removed ops
            continue;
        } else {
            rewritten.push_back(ops[i]);
        }
    }

    graph.operations = std::move(rewritten);

    if (debug) {
        fprintf(stderr, "[fusion] gdn_qkvzba_fuse: rewrote %zu ops (removed %zu)\n",
                graph.operations.size(), remove_indices.size());
    }

    return true;
}

static FusionPassRegistrar _reg({
    .id = "gdn_qkvzba_fuse",
    .description = "Fuse Qwen3.5 GDN 4-matmul projections into 2 matmuls + JIT split kernel",
    .required_kernels = {"jit:gdn_fused_proj_contiguous"},
    .run = run_gdn_qkvzba_fuse,
});

} // namespace
} // namespace dsl
