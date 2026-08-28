#include "targets/qwen4exp/impl/variant.h"

#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/hyper_connection.h"
#include "api/ops/linear.h"
#include "api/ops/ngram_ple.h"
#include "api/ops/scatter.h"
#include "api/ops/sparse_moe.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen4exp::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen4exp_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen4exp::detail {
namespace {

constexpr std::int32_t kStreams  = TextConfig::hc_count;
constexpr std::int32_t kHidden   = TextConfig::hidden;
constexpr std::int32_t kResidual = TextConfig::hc_width;
constexpr std::int32_t kLowRank  = TextConfig::hc_low_rank;
constexpr float kEps             = TextConfig::rms_epsilon;
constexpr auto kPolicy           = ops::LinearPolicy::A16Only;

// The mix hook of a block computes the inject gates its output projection scatters with. Both
// run on the same thread inside one arena scope of the family runtime, so the gates travel
// through a thread-local handle rather than a family-visible parameter.
thread_local Tensor t_inject;

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::size_t round_up(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::size_t plane_bytes(std::int32_t rows, std::int32_t tokens, DType dtype) {
    return round_up(static_cast<std::size_t>(rows) * static_cast<std::size_t>(tokens) *
                    dtype_size(dtype));
}

std::size_t mix_capacity(std::int32_t first, std::int32_t last) {
    return ops::hyper_connection_mix_workspace_capacity_bytes(kStreams, kHidden, kLowRank, first,
                                                              last) +
           plane_bytes(kStreams, last, DType::FP32);
}

std::size_t w8_capacity(std::int32_t rows, std::int32_t columns, std::int32_t first,
                        std::int32_t last) {
    return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, rows, columns, kPolicy, first,
                                                last);
}

// Mixes the residual streams into `hidden` and keeps the inject gates for the combine.
void mix_into(const Tensor& residual, const ops::HyperConnectionWeights& weights, Tensor& hidden,
              WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    Tensor inject             = workspace.alloc(DType::FP32, {kStreams, tokens});
    ops::hyper_connection_mix(residual, weights, kStreams, kEps, hidden, &inject, workspace,
                              stream);
    t_inject = inject;
}

// Scatters a block output into the residual streams with the gates the mix kept.
void combine_into(const Tensor& block_output, Tensor& residual, cudaStream_t stream) {
    if (t_inject.data == nullptr || t_inject.ne[1] != residual.ne[1]) {
        throw std::logic_error("qwen4exp: combine without a matching mix");
    }
    ops::hyper_connection_combine(block_output, t_inject, residual, stream);
    t_inject = Tensor{};
}

Tensor rows_of(const Tensor& fused, std::int32_t begin, std::int32_t count, WorkspaceArena& work,
               cudaStream_t stream) {
    Tensor out = work.alloc(DType::BF16, {count, fused.ne[1]});
    ops::extract_bf16_columns(fused, begin, out, stream);
    return out;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return graph_profiles_through(capacity - 1, {127, 511, 2047});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

// --- residual hooks ---------------------------------------------------------------------------

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    Tensor embedded           = workspace.alloc(DType::BF16, {kHidden, tokens});
    ops::embedding(ids, model.token_embedding, embedded, stream);
    ops::broadcast_streams(embedded, kStreams, residual, stream);
}

void Variant::final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    ops::hyper_connection_mix(residual, model.output_mix, kStreams, kEps, hidden, nullptr,
                              workspace, stream);
}

void Variant::attention_norm(const Tensor& residual, const FullAttentionProjectionWeights& weights,
                             Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
}

void Variant::post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                              Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
}

void Variant::layer_prologue(const ModelView& model, int layer, Tensor& residual,
                             const qwen3_6::detail::PrologueColumns& columns,
                             NgramPleStatePool* ple_state, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    if (layer != model.ple.layer) { return; }
    if (ple_state == nullptr || ple_state->empty()) {
        throw std::logic_error("qwen4exp: the PLE layer needs its state pool");
    }
    ops::NgramPleColumns ple_columns{columns.ids, columns.segment_begin, columns.slots,
                                     columns.segment_last};
    ops::NgramPleState state{ple_state->history, ple_state->conv_state};
    ops::ngram_ple_forward(residual, ple_columns, model.ple.hash, model.ple.table, model.ple.op,
                           state, kStreams, TextConfig::ple_conv_kernel,
                           TextConfig::ple_conv_dilation, kEps, workspace, stream);
}

NgramPleStatePoolSpec Variant::ple_state_spec(std::int32_t slot_count) {
    return NgramPleStatePoolSpec{
        .history_tokens = TextConfig::ple_ngram - 1,
        .conv_history   = TextConfig::ple_conv_history,
        .channels       = kResidual,
        .slot_count     = slot_count,
        .eos_token      = TextConfig::eos_token,
    };
}

std::size_t Variant::layer_prologue_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    return ops::ngram_ple_workspace_capacity_bytes(kStreams, kHidden, TextConfig::ple_embed,
                                                   TextConfig::ple_heads, first, last) +
           plane_bytes(kHidden, last, DType::BF16); // the embedding before its broadcast
}

// --- projections -------------------------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    Tensor fused = workspace.alloc(DType::BF16, {TextConfig::query_projection_rows, tokens});
    ops::linear(hidden, weights.query_key_gate_value, fused, kPolicy, workspace, stream);
    // Row order from the converter: q | k | gate | v.
    ops::extract_bf16_columns(fused, 0, query, stream);
    ops::extract_bf16_columns(fused, TextConfig::query_size, key, stream);
    ops::extract_bf16_columns(fused, TextConfig::query_size + TextConfig::kv_size, gate, stream);
    ops::extract_bf16_columns(fused, 2 * TextConfig::query_size + TextConfig::kv_size, value,
                              stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    Tensor output  = workspace.alloc(DType::BF16, {kHidden, attention.ne[1]});
    ops::linear(attention, weight, output, kPolicy, workspace, stream);
    combine_into(output, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&,
                                       Tensor&, Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                       cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = static_cast<std::int32_t>(hidden.ne[1] * hidden.ne[2]);
    Tensor flat_hidden        = hidden.view({kHidden, tokens});
    Tensor fused = workspace.alloc(DType::BF16, {TextConfig::gdn_projection_rows, tokens});
    ops::linear(flat_hidden, weights.query_key_value_z, fused, kPolicy, workspace, stream);
    Tensor qkv_flat  = qkv.view({TextConfig::convolution_dim, tokens});
    Tensor gate_flat = output_gate.view({TextConfig::value_dim, tokens});
    ops::extract_bf16_columns(fused, 0, qkv_flat, stream);
    ops::extract_bf16_columns(fused, TextConfig::convolution_dim, gate_flat, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase phase, WorkspaceArena& workspace,
    cudaStream_t stream) {
    auto scope               = workspace.scope();
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t tokens = width * batch;
    Tensor projected = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, width, batch});
    gdn_input_projection(hidden, weights, projected, output_gate, phase, workspace, stream);
    Tensor convolved = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, width, batch});
    ops::causal_conv1d_silu_snapshot(projected, conv_weight, conv_states, valid_columns,
                                     initial_slot, snapshot_base_slot, convolved, stream);
    Tensor convolved_flat = convolved.view({TextConfig::convolution_dim, tokens});
    Tensor query_flat     = query.view({TextConfig::key_dim, tokens});
    Tensor key_flat       = key.view({TextConfig::key_dim, tokens});
    Tensor value_flat     = value.view({TextConfig::value_dim, tokens});
    ops::extract_bf16_columns(convolved_flat, 0, query_flat, stream);
    ops::extract_bf16_columns(convolved_flat, TextConfig::key_dim, key_flat, stream);
    ops::extract_bf16_columns(convolved_flat, 2 * TextConfig::key_dim, value_flat, stream);
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&,
                                          const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                          Tensor&, qwen3_6::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    throw std::logic_error("qwen4exp: speculative replay records are not served");
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    auto scope    = workspace.scope();
    Tensor output = workspace.alloc(DType::BF16, {kHidden, hidden.ne[1]});
    ops::linear(hidden, weight, output, kPolicy, workspace, stream);
    combine_into(output, residual, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor&, float,
                                          const GdnProjectionWeights& weights, Tensor& hidden,
                                          Tensor& g, Tensor& beta, WorkspaceArena& workspace,
                                          cudaStream_t stream) {
    mix_into(residual, weights.mix, hidden, workspace, stream);
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t heads  = TextConfig::gdn_value_heads;
    Tensor ab                 = workspace.alloc(DType::BF16, {2 * heads, tokens});
    ops::detail::bf16_cublaslt_gemm(weights.a_b_projection, hidden, ab, stream);
    Tensor a = rows_of(ab, 0, heads, workspace, stream);
    Tensor b = rows_of(ab, heads, heads, workspace, stream);
    ops::gdn_gating(a, b, weights.a_log, weights.dt_bias, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                = workspace.scope();
    const std::int32_t tokens = hidden.ne[1];
    // The MoE op adds into its destination; a zeroed plane turns that into a plain store.
    Tensor output = workspace.alloc(DType::BF16, {kHidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(output.data, 0, output.bytes(), stream));
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        ops::kSparseMoeFlashNextGeometry, weights.op.routed_gate_up.qtype,
        weights.op.routed_down.qtype, tokens, tokens));
    WorkspaceArena leaf(storage);
    ops::sparse_moe(hidden, weights.op, ops::SparseMoeEpilogue::AddResidual, output, leaf, stream);
    combine_into(output, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&,
                             cudaStream_t) {
    throw std::logic_error("qwen4exp: MTP is not served");
}

// --- workspace capacities ----------------------------------------------------------------------

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    // The mix hook's planes live in the same mixer scope as the projection.
    return mix_capacity(first, last) +
           plane_bytes(TextConfig::query_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::query_projection_rows, kHidden, first, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::query_size, first, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    return plane_bytes(TextConfig::gdn_projection_rows, last, DType::BF16) +
           w8_capacity(TextConfig::gdn_projection_rows, kHidden, first, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width) {
    const std::int32_t tokens = batch_size * max_width;
    return 2 * plane_bytes(TextConfig::convolution_dim, tokens, DType::BF16) +
           gdn_input_projection_workspace_capacity_bytes(profile, phase, batch_size * min_width,
                                                         tokens);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile,
                                                                          qwen3_6::TextPhase,
                                                                          std::int32_t,
                                                                          std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    return plane_bytes(kHidden, last, DType::BF16) +
           w8_capacity(kHidden, TextConfig::value_dim, first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    const std::int32_t heads = TextConfig::gdn_value_heads;
    return mix_capacity(first, last) + plane_bytes(2 * heads, last, DType::BF16) +
           2 * plane_bytes(heads, last, DType::BF16);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile, qwen3_6::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    return mix_capacity(first, last) + plane_bytes(kHidden, last, DType::BF16) +
           round_up(ops::sparse_moe_workspace_capacity_bytes(ops::kSparseMoeFlashNextGeometry,
                                                             QType::W8G32_F16S, QType::W8G32_F16S,
                                                             first, last));
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

} // namespace ninfer::targets::qwen4exp::detail
