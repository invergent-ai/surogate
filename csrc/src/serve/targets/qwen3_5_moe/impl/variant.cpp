#include "targets/qwen3_5_moe/impl/variant.h"

#include "family/impl/lora_hook.h"
#include "api/ops/attn_input_proj.h"
#include "api/ops/gdn_gating_proj.h"
#include "api/ops/gdn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/sparse_moe.h"

#include <algorithm>
#include <stdexcept>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen3_5_moe::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_5_moe_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"

namespace sinfer::targets::qwen3_5_moe::detail {
namespace {


std::vector<GraphExecutionProfile> dflash_base_profiles(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return family::graph_profiles_through(max_frontier, ends);
}

bool dflash_target_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                        std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

/// The policy a generic linear accepts for a weight in the format the artifact stored it in:
/// BF16 takes A16 only; NVFP4 routes W4A4 at every width on these shapes either way.
ops::LinearPolicy linear_policy_for(const Weight& weight) {
    return weight.qtype == QType::NVFP4 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::A16Only;
}

void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights, Tensor& residual,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        ops::kSparseMoeQwen36Geometry, weights.routed_gate_up.qtype, weights.routed_down.qtype,
        hidden.ne[1], hidden.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, residual, leaf_workspace,
                    stream);
}


constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_record_workspace_bytes(const Tensor& hidden) {
    return std::max(kMinimumLeafWorkspaceBytes,
                    ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                        hidden.ne[2], hidden.ne[1], hidden.ne[1]));
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return family::graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t batch_size) {
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (weights.split) {
        // One projection per stored Linear. The fused op infers its q/k/v split from the
        // parent's row count and has no NVFP4 route at this geometry; four generic linears
        // have both, and a quantized export's constituents carry their own global scales, so
        // this is also the only exact way to run them. LoRA registers against the fused
        // parent's pointer and is not offered on the split path yet.
        const SplitAttentionWeights& split = *weights.split;
        ops::linear(hidden, split.query, query, linear_policy_for(split.query), workspace, stream);
        ops::linear(hidden, split.gate, gate, linear_policy_for(split.gate), workspace, stream);
        ops::linear(hidden, split.key, key, linear_policy_for(split.key), workspace, stream);
        ops::linear(hidden, split.value, value, linear_policy_for(split.value), workspace, stream);
        return;
    }
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
    family::apply_lora_qkv(weights.query_key_gate_value, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, workspace, stream);
    family::apply_lora(weight, 3, attention, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor query   = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    Tensor gate    = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor key     = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    Tensor value   = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    ops::attn_input_proj(hidden, weights.query_key_gate_value, query, gate, key, value, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1] * hidden.ne[2])});
    if (weights.split) {
        // in_proj_qkv and in_proj_z as the checkpoint stores them, each in its own format.
        const SplitGdnInputWeights& split = *weights.split;
        ops::linear(hidden, split.query_key_value, qkv, linear_policy_for(split.query_key_value),
                    workspace, stream);
        ops::linear(hidden, split.z, output_gate_flat, linear_policy_for(split.z), workspace,
                    stream);
        return;
    }
    ops::gdn_input_proj(hidden, weights.query_key_value_z, qkv, output_gate_flat, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (weights.split) {
        const SplitGdnInputWeights& split = *weights.split;
        ops::gdn_input_proj_conv_snapshot_split(
            hidden, split.query_key_value, split.z, conv_weight, conv_states, valid_columns,
            initial_slot, snapshot_base_slot, query, key, value, output_gate_view,
            linear_policy_for(split.query_key_value), workspace, stream);
        return;
    }
    ops::gdn_input_proj_conv_snapshot(hidden, weights.query_key_value_z, conv_weight, conv_states,
                                      valid_columns, initial_slot, snapshot_base_slot, query, key,
                                      value, output_gate_view, workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (weights.split) {
        const SplitGdnInputWeights& split = *weights.split;
        ops::gdn_input_proj_conv_record_split(
            hidden, split.query_key_value, split.z, conv_weight, conv_states, valid_columns,
            initial_slots, conv_record, query, key, value, output_gate_view,
            linear_policy_for(split.query_key_value), leaf_workspace, stream);
        return;
    }
    ops::gdn_input_proj_conv_record(hidden, weights.query_key_value_z, conv_weight, conv_states,
                                    valid_columns, initial_slots, conv_record, query, key, value,
                                    output_gate_view, leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_b_projection, weights.a_log,
                              weights.dt_bias, workspace, hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                       std::int32_t last) {
    family::validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                    std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        // Four generic linears over the split. The NVFP4 W4A4 workspace is a function of K and
        // the token count, not N, so the query's is the widest of the four.
        return ops::linear_workspace_capacity_bytes(QType::NVFP4, TextConfig::query_size,
                                                    TextConfig::hidden, ops::LinearPolicy::AllowA4,
                                                    first, last);
    }
    return ops::attn_input_proj_workspace_capacity_bytes(
        QType::W8G32_F16S, TextConfig::mtp_attention_input_rows, TextConfig::hidden,
        ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                          family::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::AllowA4, first, last);
    }
    return ops::linear_add_workspace_capacity_bytes(QType::W8G32_F16S, TextConfig::hidden,
                                                    TextConfig::query_size,
                                                    ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        // Two generic linears over the split halves; the NVFP4 workspace bounds any format the
        // export gave them.
        return std::max(ops::linear_workspace_capacity_bytes(
                            QType::NVFP4, TextConfig::convolution_dim, TextConfig::hidden,
                            ops::LinearPolicy::AllowA4, first, last),
                        ops::linear_workspace_capacity_bytes(QType::NVFP4, TextConfig::value_dim,
                                                             TextConfig::hidden,
                                                             ops::LinearPolicy::AllowA4, first, last));
    }
    return ops::gdn_input_proj_workspace_capacity_bytes(
        QType::W8G32_F16S, TextConfig::convolution_dim + TextConfig::value_dim,
        TextConfig::hidden, ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                            family::TextPhase,
                                                                            std::int32_t batch_size,
                                                                            std::int32_t first,
                                                                            std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        return ops::gdn_input_proj_conv_snapshot_split_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::convolution_dim, TextConfig::value_dim, TextConfig::hidden,
            ops::LinearPolicy::AllowA4, batch_size, first, last);
    }
    return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size, first, last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                          family::TextPhase,
                                                                          std::int32_t batch_size,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_split_workspace_capacity_bytes(
                            QType::NVFP4, TextConfig::convolution_dim, TextConfig::value_dim,
                            TextConfig::hidden, ops::LinearPolicy::AllowA4, batch_size, first,
                            last));
    }
    return std::max(kMinimumLeafWorkspaceBytes,
                    ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch_size,
                        first, last));
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights,
                                                                    family::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    if (weights == WeightsProfile::CompressedTensors) {
        // Whatever format the export gave gdn/output, the NVFP4 workspace bounds it.
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::AllowA4, first, last);
    }
    return ops::linear_add_workspace_capacity_bytes(QType::W8G32_F16S, TextConfig::hidden,
                                                    TextConfig::value_dim,
                                                    ops::LinearPolicy::A16Only, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    if (weights_profile == WeightsProfile::RoutedNvfp4 ||
        weights_profile == WeightsProfile::CompressedTensors) {
        return ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry,
                                                        QType::NVFP4, QType::NVFP4, first, last);
    }
    // A groupwise-int artifact mixes Q5 and Q6 down projections across its layers, and the
    // scratch is one arena for all of them.
    return std::max(ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry,
                                                             QType::Q4G64_F16S, QType::Q5G64_F16S,
                                                             first, last),
                    ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry,
                                                             QType::Q4G64_F16S, QType::Q6G64_F16S,
                                                             first, last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                             std::int32_t last) {
    return ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeQwen36Geometry,
                                                    QType::W8G32_F16S, QType::W8G32_F16S, first,
                                                    last);
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

} // namespace sinfer::targets::qwen3_5_moe::detail
