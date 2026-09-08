#include "family/impl/storage_workspace.h"
#include "targets/qwen3_5/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/gdn_gating_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include <cstdlib>

#include "family/impl/lora_hook.h"
#include "family/impl/mlp_swiglu.h"
#include "api/ops/linear_pair.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/mtp_pack.h"
#include "api/ops/residual_add.h"
#include "api/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen3_5::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_5_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"

namespace sinfer::targets::qwen3_5::detail {
namespace {

constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;

ops::LinearPolicy text_policy(QType format) {
    switch (format) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    case QType::FP8_E4M3FN_BLK128_F32S:
    case QType::FP8_E4M3FN_ROW_F32S:
        return kFp8TextPolicy; // informational: the route quantises per token per 128 regardless
    // W8 opts into A8: the wrappers run the W8A8-int IMMA path at
    // T >= kW8A8MinTokens and A16 below it.
    case QType::W8G32_F16S:
        return ops::LinearPolicy::AllowA8;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

ops::LinearPolicy text_policy(const Weight& weight) { return text_policy(weight.qtype); }

using family::matrix_workspace;
using family::matrix_pair_workspace;
using family::text_layers_workspace;
using family::WorkspaceLayers;

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (const auto* split =
            std::get_if<QkvPlusZGdnInputProjectionPayload>(&weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(
                            split->query_key_value.qtype, split->z.qtype, split->query_key_value.n, split->z.n,
                            split->query_key_value.k,
                            text_policy(split->query_key_value), text_policy(split->z), batch, width, width));
    }
    if (const auto* pair =
            std::get_if<QkPlusVzGdnInputProjectionPayload>(&weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_pair_workspace_capacity_bytes(
                            pair->query_key.qtype, pair->value_z.qtype, pair->query_key.n,
                            pair->value_z.n / 2, pair->value_z.n / 2, pair->query_key.k,
                            text_policy(pair->query_key), text_policy(pair->value_z), batch, width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (const auto* split =
            std::get_if<QkvPlusZGdnInputProjectionPayload>(&weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_split_workspace_capacity_bytes(
                            split->query_key_value.qtype, split->z.qtype, split->query_key_value.n, split->z.n,
                            split->query_key_value.k,
                            text_policy(split->query_key_value), text_policy(split->z), batch, width, width));
    }
    if (const auto* pair =
            std::get_if<QkPlusVzGdnInputProjectionPayload>(&weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_pair_workspace_capacity_bytes(
                            pair->query_key.qtype, pair->value_z.qtype, pair->query_key.n,
                            pair->value_z.n / 2, pair->value_z.n / 2, pair->query_key.k,
                            text_policy(pair->query_key), text_policy(pair->value_z), batch, width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& g, QType gate_up_qtype,
                                       QType down_qtype, ops::LinearPolicy gate_up_policy,
                                       ops::LinearPolicy down_policy,
                                       std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {g.intermediate, last});
    family::swiglu_mlp_layout(layout, g.intermediate, g.hidden, gate_up_qtype,
                              gate_up_policy, first, last);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, g.hidden, g.intermediate, down_policy, first, last));
    }
    std::size_t peak = layout.peak_bytes(1);
    if (gate_up_qtype == QType::NVFP4 && down_qtype == QType::NVFP4 &&
        gate_up_policy == ops::LinearPolicy::AllowA4 && down_policy == ops::LinearPolicy::AllowA4) {
        WorkspaceLayoutBuilder fused;
        family::swiglu_mlp_down_add_layout(fused, g.intermediate, g.hidden, gate_up_policy, first, last);
        peak = std::max(peak, fused.peak_bytes(1));
    }
    return peak;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return family::graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}


namespace {

/// Adds the round's adapter deltas to a projection that has just been written.
///
/// Keyed by the base weight's device pointer, which is this layer's identity --
/// the variant methods take weights, not a layer index. Every token in the round
/// may name a different adapter (or none), which is why this reads the round's
/// slot vector rather than a single active adapter.
///
/// The ids and the scratch both come from the decode frame, published by the
/// schedule. They must: a captured graph records their addresses once and replays
/// against whatever the round wrote, and a buffer allocated per call inside the
/// hook is baked in by address instead -- which is what made an earlier version
/// of this give a different answer on every replay.
using family::apply_lora;
using family::apply_lora_qkv;

} // namespace

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             text_policy(split->query_key), workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
    apply_lora_qkv(fused, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
    apply_lora(weight, 3, attention, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    // A native GGUF parent is projected per component, from the row views the loader cut:
    // its rows may come in more than one format (a UD mixture's q beside its k), and each
    // view is one format the kernel can take. The packed launch and its split stay for the
    // formats whose fused parent is one plane.
    if (weights.packed.layout == QuantLayout::GgmlBlocks) {
        ops::linear(hidden, weights.query, query, stream);
        ops::linear(hidden, weights.key, key, stream);
        ops::linear(hidden, weights.output_gate, gate, stream);
        ops::linear(hidden, weights.value, value, stream);
        return;
    }
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {weights.packed.n, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({weights.head_dim, query.ne[0] / weights.head_dim, cols});
    Tensor key_heads   = key.view({weights.head_dim, key.ne[0] / weights.head_dim, cols});
    Tensor gate_heads  = gate.view({weights.head_dim, gate.ne[0] / weights.head_dim, cols});
    Tensor value_heads = value.view({weights.head_dim, value.ne[0] / weights.head_dim, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    // The fused pair carries tuned route tables for the 27B and 35B geometries
    // only (K 5120/2048 into 1024 rows). This target's MTP block projects 2 kv
    // heads -- 512 rows -- so it takes the unfused pair, exactly as its q/gate
    // sibling below already does. Correctness first; the fusion can follow if a
    // route table is ever measured for this shape.
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // The GDN value width is the output gate's own: its element count over the columns it
    // is viewed as. Reading it off the tensor rather than a compiled constant is what lets
    // one target run a checkpoint with more value heads than the one it compiled.
    const std::int32_t columns = static_cast<std::int32_t>(hidden.ne[1]);
    Tensor output_gate_flat =
        output_gate.view({static_cast<std::int32_t>(output_gate.numel() / columns), columns});
    if (const auto* split =
            std::get_if<QkvPlusZGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_split(hidden, split->query_key_value, split->z, qkv, output_gate_flat,
                                  text_policy(split->query_key_value), text_policy(split->z), workspace, stream);
        return;
    }
    // The 27B-class groupwise export splits one component earlier: query|key, then value|z.
    if (const auto* pair =
            std::get_if<QkPlusVzGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_pair(hidden, pair->query_key, pair->value_z, qkv, output_gate_flat,
                                 text_policy(pair->query_key), text_policy(pair->value_z), workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    const std::int32_t gate_rows =
        static_cast<std::int32_t>(output_gate.numel() / (hidden.ne[1] * hidden.ne[2]));
    Tensor output_gate_view = output_gate.view({gate_rows, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<QkvPlusZGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot_split(
            hidden, split->query_key_value, split->z, conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, output_gate_view,
            text_policy(split->query_key_value), text_policy(split->z), leaf_workspace, stream);
        return;
    }
    if (const auto* pair =
            std::get_if<QkPlusVzGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot_pair(
            hidden, pair->query_key, pair->value_z, conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, output_gate_view,
            text_policy(pair->query_key), text_policy(pair->value_z), leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    const std::int32_t gate_rows =
        static_cast<std::int32_t>(output_gate.numel() / (hidden.ne[1] * hidden.ne[2]));
    Tensor output_gate_view = output_gate.view({gate_rows, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<QkvPlusZGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record_split(
            hidden, split->query_key_value, split->z, conv_weight, conv_states, valid_columns,
            initial_slots, conv_record, query, key, value, output_gate_view,
            text_policy(split->query_key_value), text_policy(split->z), leaf_workspace, stream);
        return;
    }
    if (const auto* pair =
            std::get_if<QkPlusVzGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record_pair(
            hidden, pair->query_key, pair->value_z, conv_weight, conv_states, valid_columns,
            initial_slots, conv_record, query, key, value, output_gate_view,
            text_policy(pair->query_key), text_policy(pair->value_z), leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    if (family::swiglu_mlp_down_add(hidden, weights.gate_up, weights.down, residual,
                                    text_policy(weights.gate_up), /*limit=*/0.0F, workspace,
                                    stream)) {
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, hidden.ne[1]});
    family::swiglu_mlp(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
    // down reads the SwiGLU activation, which is exactly the input its adapter was
    // trained against.
    apply_lora(weights.down, 4, activation, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {weights.gate_up.n, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    const std::int32_t intermediate = weights.gate_up.n / 2;
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, intermediate),
                  gate_up.slice(0, intermediate, intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {hidden.ne[0], cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                       std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.mtp_attention_input_rows(), last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                std::int32_t last) {
    family::validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                    std::int32_t last) {
    family::validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Attention, [&](const std::string& prefix) {
        if (geometry.linear_storage.contains(prefix + "attention/query_key")) {
            // The two-parent attention op uses row views at A16.
            const auto rows = [&](QType type, int n, int k) {
                return ops::linear_workspace_capacity_bytes(type, n, k, ops::LinearPolicy::A16Only, first, last);
            };
            return std::max(matrix_workspace(geometry, prefix + "attention/query_key", rows),
                            matrix_workspace(geometry, prefix + "attention/gate_value", rows));
        }
        return matrix_workspace(geometry, prefix + "attention/query_key_gate_value", [&](QType type, int n, int k) {
            return ops::attn_input_proj_workspace_capacity_bytes(type, n, k, text_policy(type), first, last);
        });
    });
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Attention, [&](const std::string& prefix) {
        return matrix_workspace(geometry, prefix + "attention/output", [&](QType type, int n, int k) {
            return ops::linear_add_workspace_capacity_bytes(type, n, k, text_policy(type), first, last);
        });
    });
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Linear, [&](const std::string& prefix) {
        return matrix_workspace(geometry, prefix + "gdn/output", [&](QType type, int n, int k) {
            return ops::linear_add_workspace_capacity_bytes(type, n, k, text_policy(type), first, last);
        });
    });
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Linear, [&](const std::string& prefix) {
        if (geometry.linear_storage.contains(prefix + "gdn/query_key_value")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key_value", prefix + "gdn/z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_split_workspace_capacity_bytes(
                        a, b, an, bn, k, text_policy(a), text_policy(b), first, last);
                });
        }
        if (geometry.linear_storage.contains(prefix + "gdn/query_key")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key", prefix + "gdn/value_z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_pair_workspace_capacity_bytes(
                        a, b, an, bn / 2, bn / 2, k, text_policy(a), text_policy(b), first, last);
                });
        }
        return matrix_workspace(geometry, prefix + "gdn/query_key_value_z", [&](QType type, int n, int k) {
            return ops::gdn_input_proj_workspace_capacity_bytes(type, n, k, text_policy(type), first, last);
        });
    });
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t batch_size, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Linear, [&](const std::string& prefix) {
        const auto capacity = [&]() -> std::size_t {
        if (geometry.linear_storage.contains(prefix + "gdn/query_key_value")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key_value", prefix + "gdn/z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(
                        a, b, an, bn, k, text_policy(a), text_policy(b), batch_size, first, last);
                });
        }
        if (geometry.linear_storage.contains(prefix + "gdn/query_key")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key", prefix + "gdn/value_z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_conv_snapshot_pair_workspace_capacity_bytes(
                        a, b, an, bn / 2, bn / 2, k, text_policy(a), text_policy(b), batch_size, first, last);
                });
        }
        return matrix_workspace(geometry, prefix + "gdn/query_key_value_z", [&](QType type, int n, int k) {
            return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(type, n, k, text_policy(type), batch_size, first, last);
        });
        };
        return std::max(kMinimumLeafWorkspaceBytes, capacity());
    });
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t batch_size, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::Linear, [&](const std::string& prefix) {
        const auto capacity = [&]() -> std::size_t {
        if (geometry.linear_storage.contains(prefix + "gdn/query_key_value")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key_value", prefix + "gdn/z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_conv_record_split_workspace_capacity_bytes(
                        a, b, an, bn, k, text_policy(a), text_policy(b), batch_size, first, last);
                });
        }
        if (geometry.linear_storage.contains(prefix + "gdn/query_key")) {
            return matrix_pair_workspace(geometry, prefix + "gdn/query_key", prefix + "gdn/value_z",
                [&](QType a, QType b, int an, int bn, int k) {
                    return ops::gdn_input_proj_conv_record_pair_workspace_capacity_bytes(
                        a, b, an, bn / 2, bn / 2, k, text_policy(a), text_policy(b), batch_size, first, last);
                });
        }
        return matrix_workspace(geometry, prefix + "gdn/query_key_value_z", [&](QType type, int n, int k) {
            return ops::gdn_input_proj_conv_record_workspace_capacity_bytes(type, n, k, text_policy(type), batch_size, first, last);
        });
        };
        return std::max(kMinimumLeafWorkspaceBytes, capacity());
    });
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(geometry.gdn_value_heads,
                                                              geometry.hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry,
    WeightsProfile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return text_layers_workspace(geometry, WorkspaceLayers::All, [&](const std::string& prefix) {
        const auto& gate_up = family::require_linear_storage(geometry, prefix + "mlp/gate_up");
        const auto& down = family::require_linear_storage(geometry, prefix + "mlp/down");
        std::size_t peak = 0;
        for (const auto a : gate_up.formats) {
            for (const auto b : down.formats) {
                peak = std::max(peak, post_mixer_workspace_bytes(geometry, a, b, text_policy(a), text_policy(b), first, last));
            }
        }
        return peak;
    });
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                             std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.mtp_mlp_gate_up_rows(), last});
    (void)layout.alloc(DType::BF16, {geometry.intermediate, last});
    (void)layout.alloc(DType::BF16, {geometry.hidden, last});
    return layout.peak_bytes(1);
}


// This family's non-attending layers run a gated delta net, not a short convolution. The leaf
// is declared because the runtime is a template over the whole Variant interface, and it
// refuses rather than returning quietly: a mixer that contributed nothing to the residual reads
// as a model that merely answers badly.
void Variant::short_conv_projection(const Tensor&, const Tensor&, float,
                                    const GdnProjectionWeights&, Tensor&, family::TextPhase,
                                    WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("short_conv_projection: this target's linear mixer is a gated delta net");
}

std::size_t Variant::short_conv_projection_workspace_capacity_bytes(const family::TextGeometry&,
                                                                    WeightsProfile,
                                                                    family::TextPhase,
                                                                    std::int32_t, std::int32_t) {
    return 0;
}

} // namespace sinfer::targets::qwen3_5::detail
