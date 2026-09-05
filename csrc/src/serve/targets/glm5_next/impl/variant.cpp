#include "targets/glm5_next/impl/variant.h"

#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/hyper_connection.h"
#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/manifold_hyper_connection.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scatter.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::glm5_next::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS glm5_next_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::glm5_next::detail {
namespace {

using family::apply_lora;

constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::SparseMoeGeometry kMoeGeometry = ops::kSparseMoeGlm53Geometry;
constexpr std::int32_t kStreams = TextConfig::hc_streams;

// --------------------------------------------------------------------------------------------
// The mixings, from a site's collapse to its scatter
//
// A layer collapses the residual streams into one block input and, after the block has run,
// recombines them with the matrix that collapse produced. The two are separate leaves in the
// family's interface, so what the first computes has to reach the second. It travels in
// thread-local views of a device buffer this target owns per device -- the same handoff the
// other hyper-connected target makes, and with the same check that the two ends ran on one
// device.
// --------------------------------------------------------------------------------------------

struct MixingScratch {
    void* data       = nullptr;
    std::size_t bytes = 0;
};

/// Enough for `post` [S,T] and `comb` [S,S,T] at the widest round this engine plans.
constexpr std::size_t kMaximumMixingTokens = 65536;
constexpr std::size_t kMixingScratchBytes =
    static_cast<std::size_t>(kStreams) * (1 + kStreams) * kMaximumMixingTokens * sizeof(float);

std::mutex& mixing_mutex() {
    static std::mutex value;
    return value;
}

std::unordered_map<int, MixingScratch>& mixing_scratch() {
    static std::unordered_map<int, MixingScratch> value;
    return value;
}

MixingScratch& mixing_scratch_for_current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mixing_mutex());
    return mixing_scratch()[device];
}

thread_local Tensor t_post;
thread_local Tensor t_comb;
thread_local int t_mixing_device = -1;

void check_device_handoff(const char* what, int recorded) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (recorded >= 0 && recorded != device) {
        throw std::logic_error(std::string("glm5_next: ") + what +
                               " were produced on another device than the one consuming them");
    }
}

/// Collapse the streams into `hidden`, keeping the mixings the matching scatter needs, then
/// apply the site's own norm to what the block will read.
void mix_into(const Tensor& residual, const HyperConnectionPayload& hc, const Tensor& norm,
              float eps, Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    const MixingScratch& scratch = mixing_scratch_for_current_device();
    const std::size_t needed = static_cast<std::size_t>(kStreams) * (1 + kStreams) *
                               static_cast<std::size_t>(tokens) * sizeof(float);
    if (scratch.data == nullptr || needed > scratch.bytes) {
        throw std::logic_error("glm5_next: the mixing scratch is missing or too small for this "
                               "forward");
    }
    Tensor post(scratch.data, DType::FP32, {kStreams, tokens});
    Tensor comb(static_cast<char*>(scratch.data) +
                    static_cast<std::size_t>(kStreams) * tokens * sizeof(float),
                DType::FP32, {kStreams, kStreams, tokens});
    auto scope = workspace.scope();
    Tensor collapsed = workspace.alloc(DType::BF16, {residual.ne[0] / kStreams, tokens});
    ops::manifold_hyper_connection_mix(residual, hc.weights, kStreams, eps,
                                       TextConfig::hc_epsilon,
                                       TextConfig::hc_sinkhorn_iterations, collapsed, post, comb,
                                       workspace, stream);
    ops::rmsnorm(collapsed, norm, eps, /*unit_offset=*/false, hidden, stream);
    t_post = post;
    t_comb = comb;
    CUDA_CHECK(cudaGetDevice(&t_mixing_device));
}

/// Recombine the streams with the block's output and the matrix the collapse produced.
void combine_into(const Tensor& block_output, Tensor& residual, cudaStream_t stream) {
    if (t_post.data == nullptr || t_post.ne[1] != residual.ne[1]) {
        throw std::logic_error("glm5_next: a combine without a matching collapse");
    }
    check_device_handoff("the stream mixings", t_mixing_device);
    ops::manifold_hyper_connection_combine(block_output, t_post, t_comb, kStreams, residual,
                                           stream);
    t_post          = Tensor{};
    t_comb          = Tensor{};
    t_mixing_device = -1;
}

[[noreturn]] void no_short_conv(const char* leaf) {
    throw std::logic_error(
        std::string("glm5_next: ") + leaf +
        " was called, but this target's non-attending layers run Kimi Delta Attention, not a "
        "short convolution. Reaching here means the family runtime took the short-conv branch "
        "for a target whose declared mixer is KimiDelta.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("glm5_next: ") + leaf +
        " was called, but this target binds no draft head, and Kimi Delta Attention has no "
        "replay-record form to verify a speculative round with; --spec is refused when the "
        "artifact is bound. Reaching here means a speculative round started without one.");
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("glm5_next: invalid weights profile");
}

/// The larger of the two routes an artifact of this target may carry for one projection: the
/// profile's row-split int8, or the K-quants a GGUF served natively keeps. The layout is planned
/// before the weights are read, so it has to hold either.
std::size_t linear_capacity(WeightsProfile profile, std::int32_t rows, std::int32_t columns,
                            std::int32_t first, std::int32_t last) {
    return std::max({
        ops::linear_workspace_capacity_bytes(profile_qtype(profile), rows, columns, kTextPolicy,
                                             first, last),
        ops::linear_workspace_capacity_bytes(QType::Q4_K, rows, columns, kTextPolicy, first, last),
        ops::linear_workspace_capacity_bytes(QType::Q6_K, rows, columns, kTextPolicy, first, last),
    });
}

std::size_t plane_bytes(std::int32_t rows, std::int32_t tokens, DType dtype) {
    return (static_cast<std::size_t>(rows) * tokens * dtype_size(dtype) + 255U) / 256U * 256U;
}

/// The mixture, over an arena of its own, into a destination it adds to -- so the caller zeroes
/// what it hands over when it wants the block's output rather than a sum with the residual.
void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights, Tensor& out,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        kMoeGeometry, weights.routed_gate_up.qtype, weights.routed_down.qtype, hidden.ne[1],
        hidden.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, out, leaf_workspace,
                    stream);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

void Variant::prewarm_device_scratch() {
    MixingScratch& scratch = mixing_scratch_for_current_device();
    if (scratch.data == nullptr) {
        CUDA_CHECK(cudaMalloc(&scratch.data, kMixingScratchBytes));
        scratch.bytes = kMixingScratchBytes;
    }
}

// ---- The residual: four streams in, four streams out ------------------------

void Variant::embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor embedded =
        workspace.alloc(DType::BF16, {model.geometry.hidden, ids.ne[0]});
    ops::embedding(ids, model.token_embedding, embedded, stream);
    ops::broadcast_streams(embedded, kStreams, residual, stream);
}

void Variant::final_residual_mix(const ModelView& model, const Tensor& residual, Tensor& hidden,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor mean = workspace.alloc(DType::BF16, {model.geometry.hidden, residual.ne[1]});
    // The stack ends with an unweighted mean of the streams, not another mixing.
    ops::collapse_streams_mean(residual, kStreams, mean, stream);
    ops::rmsnorm(mean, model.final_norm, model.geometry.rms_epsilon, /*unit_offset=*/false, hidden,
                 stream);
}

void Variant::attention_norm(const Tensor& residual,
                             const FullAttentionProjectionWeights& weights, Tensor& hidden,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.hc, weights.norm, weights.rms_epsilon, hidden, workspace, stream);
}

void Variant::post_mixer_norm(const Tensor& residual, const PostMixerWeights& weights,
                              Tensor& hidden, WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.hc, weights.norm, weights.rms_epsilon, hidden, workspace, stream);
}

// ---- Multi-head latent attention, NoPE --------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // `gate` is deliberately untouched: this attention writes no output-gate rows and the family
    // skips the multiply (`Variant::attention_output_gate == false`).
    (void)gate;
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    Tensor q_low  = workspace.alloc(DType::BF16, {weights.query_a.n, tokens});
    Tensor kv_low = workspace.alloc(DType::BF16, {weights.kv_a.n, tokens});
    ops::linear(hidden, weights.query_a, q_low, kTextPolicy, workspace, stream);
    ops::linear(hidden, weights.kv_a, kv_low, kTextPolicy, workspace, stream);
    // Each low rank is normalised whole, which is what replaces the per-head query and key
    // norms every other target in this family carries.
    ops::rmsnorm(q_low, weights.query_a_norm, weights.rms_epsilon, /*unit_offset=*/false, q_low,
                 stream);
    ops::rmsnorm(kv_low, weights.kv_a_norm, weights.rms_epsilon, /*unit_offset=*/false, kv_low,
                 stream);
    ops::linear(q_low, weights.query_b, query, kTextPolicy, workspace, stream);
    // The expansion, one head at a time in the weight's own row order: the latent becomes this
    // layer's keys and values, which the cache then holds as an ordinary attention's.
    ops::linear(kv_low, weights.k_b, key, kTextPolicy, workspace, stream);
    ops::linear(kv_low, weights.v_b, value, kTextPolicy, workspace, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {weight.n, attention.ne[1]});
    ops::linear(attention, weight, out, kTextPolicy, workspace, stream);
    apply_lora(weight, 3, attention, out, stream);
    combine_into(out, residual, stream);
}

// ---- Kimi Delta Attention ---------------------------------------------------

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    mix_into(residual, weights.hc, norm_weight, eps, hidden, workspace, stream);
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    Tensor decay_low = workspace.alloc(DType::BF16, {weights.decay_a.n, tokens});
    Tensor decay     = workspace.alloc(DType::BF16, {weights.decay_b.n, tokens});
    Tensor update    = workspace.alloc(DType::BF16, {weights.beta.n, tokens});
    ops::linear(hidden, weights.decay_a, decay_low, kTextPolicy, workspace, stream);
    ops::linear(decay_low, weights.decay_b, decay, kTextPolicy, workspace, stream);
    ops::linear(hidden, weights.beta, update, kTextPolicy, workspace, stream);
    ops::kda_gating(decay, update, weights.a_log, weights.decay_bias, weights.gate_lower_bound, g,
                    beta, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    Tensor gate_low = workspace.alloc(DType::BF16, {weights.gate_a.n, tokens});
    ops::linear(hidden, weights.query_key_value, qkv, kTextPolicy, workspace, stream);
    // The output gate is low-rank, so it is two projections rather than rows of the fused one.
    ops::linear(hidden, weights.gate_a, gate_low, kTextPolicy, workspace, stream);
    ops::linear(gate_low, weights.gate_b, output_gate, kTextPolicy, workspace, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t width   = hidden.ne[1];
    const std::int32_t batch   = hidden.ne[2];
    const std::int32_t rows    = weights.query_key_value.n;
    const std::int32_t per     = rows / 3;
    auto scope                 = workspace.scope();
    Tensor flat_hidden = hidden.view({hidden.ne[0], width * batch});
    Tensor projected   = workspace.alloc(DType::BF16, {rows, width * batch});
    Tensor convolved   = workspace.alloc(DType::BF16, {rows, width * batch});
    Tensor gate_low    = workspace.alloc(DType::BF16, {weights.gate_a.n, width * batch});
    ops::linear(flat_hidden, weights.query_key_value, projected, kTextPolicy, workspace, stream);
    // One convolution over the fused q|k|v, checkpointed per lane, then the three spans are
    // taken apart. The fused snapshot op the delta-net targets use projects the output gate as
    // rows of the same parent; here the gate is low-rank and cannot ride along.
    Tensor projected_lanes = projected.view({rows, width, batch});
    Tensor convolved_lanes = convolved.view({rows, width, batch});
    ops::causal_conv1d_silu_snapshot(projected_lanes, conv_weight, conv_states, valid_columns,
                                     initial_slot, snapshot_base_slot, convolved_lanes, stream);
    ops::extract_bf16_columns(convolved, 0, query, stream);
    ops::extract_bf16_columns(convolved, per, key, stream);
    ops::extract_bf16_columns(convolved, 2 * per, value, stream);
    Tensor flat_gate = output_gate.view({output_gate.ne[0], width * batch});
    ops::linear(flat_hidden, weights.gate_a, gate_low, kTextPolicy, workspace, stream);
    ops::linear(gate_low, weights.gate_b, flat_gate, kTextPolicy, workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {weight.n, hidden.ne[1]});
    ops::linear(hidden, weight, out, kTextPolicy, workspace, stream);
    apply_lora(weight, 3, hidden, out, stream);
    combine_into(out, residual, stream);
}

// ---- The feed-forward -------------------------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    const std::int32_t hidden_rows = hidden.ne[0];
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {hidden_rows, tokens});
    if (weights.sparse) {
        // The mixture's epilogue adds into its destination, and what the recombination needs is
        // the block's own output, so the destination starts at zero.
        CUDA_CHECK(cudaMemsetAsync(out.data, 0, out.bytes(), stream));
        run_sparse_moe(hidden, weights.moe, out, workspace, stream);
    } else {
        Tensor activated = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, tokens});
        ops::linear_swiglu(hidden, weights.gate_up, activated, kTextPolicy, workspace, stream);
        ops::linear(activated, weights.down, out, kTextPolicy, workspace, stream);
    }
    combine_into(out, residual, stream);
}

// ---- Workspace capacities ---------------------------------------------------

std::size_t
Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                       WeightsProfile weights_profile,
                                                       family::TextPhase, std::int32_t first,
                                                       std::int32_t last) {
    family::validate_token_interval(first, last);
    return plane_bytes(geometry.q_lora_rank, last, DType::BF16) +
           plane_bytes(geometry.kv_lora_rank, last, DType::BF16) +
           std::max({linear_capacity(weights_profile, geometry.q_lora_rank, geometry.hidden, first,
                                     last),
                     linear_capacity(weights_profile, geometry.kv_lora_rank, geometry.hidden,
                                     first, last),
                     linear_capacity(weights_profile, geometry.query_size(), geometry.q_lora_rank,
                                     first, last),
                     linear_capacity(weights_profile, geometry.kv_size(), geometry.kv_lora_rank,
                                     first, last)});
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return plane_bytes(geometry.hidden, last, DType::BF16) +
           linear_capacity(weights_profile, geometry.hidden, geometry.query_size(), first, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    // The collapse's own scratch, then the three control projections'.
    const std::size_t mix = plane_bytes(geometry.hidden, last, DType::BF16) +
                            ops::manifold_hyper_connection_mix_workspace_capacity_bytes(
                                geometry.hc_streams, geometry.hidden, first, last);
    const std::size_t control =
        plane_bytes(geometry.kda_gate_rank, last, DType::BF16) +
        plane_bytes(geometry.value_dim(), last, DType::BF16) +
        plane_bytes(geometry.gdn_value_heads, last, DType::BF16) +
        std::max({linear_capacity(WeightsProfile::GroupwiseInt, geometry.kda_gate_rank,
                                  geometry.hidden, first, last),
                  linear_capacity(WeightsProfile::GroupwiseInt, geometry.value_dim(),
                                  geometry.kda_gate_rank, first, last),
                  linear_capacity(WeightsProfile::GroupwiseInt, geometry.gdn_value_heads,
                                  geometry.hidden, first, last)});
    return std::max(mix, control);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return plane_bytes(geometry.kda_gate_rank, last, DType::BF16) +
           std::max({linear_capacity(weights_profile, geometry.convolution_dim(), geometry.hidden,
                                     first, last),
                     linear_capacity(weights_profile, geometry.kda_gate_rank, geometry.hidden,
                                     first, last),
                     linear_capacity(weights_profile, geometry.value_dim(),
                                     geometry.kda_gate_rank, first, last)});
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t batch_size, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    const std::int32_t columns = last * std::max(batch_size, 1);
    return 2 * plane_bytes(geometry.convolution_dim(), columns, DType::BF16) +
           plane_bytes(geometry.kda_gate_rank, columns, DType::BF16) +
           std::max({linear_capacity(weights_profile, geometry.convolution_dim(), geometry.hidden,
                                     1, columns),
                     linear_capacity(weights_profile, geometry.kda_gate_rank, geometry.hidden, 1,
                                     columns),
                     linear_capacity(weights_profile, geometry.value_dim(),
                                     geometry.kda_gate_rank, 1, columns)});
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return plane_bytes(geometry.hidden, last, DType::BF16) +
           linear_capacity(weights_profile, geometry.hidden, geometry.value_dim(), first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                         WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    // The block's own output plane, then whichever of the two feed-forwards is larger. A layer
    // is one or the other, but the layout is planned for the stack rather than per layer.
    const std::size_t mixture = std::max({
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, qtype, qtype, first, last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q4_K, first,
                                                 last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q6_K, first,
                                                 last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q5_K, QType::Q6_K, first,
                                                 last),
    });
    const std::size_t dense =
        plane_bytes(geometry.dense_intermediate, last, DType::BF16) +
        std::max(linear_capacity(weights_profile, 2 * geometry.dense_intermediate, geometry.hidden,
                                 first, last),
                 linear_capacity(weights_profile, geometry.hidden, geometry.dense_intermediate,
                                 first, last));
    return plane_bytes(geometry.hidden, last, DType::BF16) + std::max(mixture, dense);
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has to exist. None
// can be reached by this topology, and each says so rather than returning quietly: a silent
// no-op would be a layer that contributed nothing to the residual, which reads as a model that
// merely answers badly.

// This target runs the delta-net leaves and the mixer output projection, so only the short
// convolution's and the draft head's are refused. The record form is its own refusal below:
// Kimi Delta Attention has no replay, which is a property of the recurrence rather than of the
// mixer's presence.
SINFER_FAMILY_UNRUNNABLE_SHORT_CONV_LEAVES(no_short_conv)
SINFER_FAMILY_UNRUNNABLE_MTP_LEAVES(no_speculation)

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&,
                                          const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                          Tensor&, family::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    no_speculation("gdn_input_projection_record");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    const family::TextGeometry&, WeightsProfile, family::TextPhase, std::int32_t, std::int32_t,
    std::int32_t) {
    return 0;
}

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    // Only the magic is this target's: 'G53F'.
    family::debug_probe_dump(0x47353346, tag, tensor, TextConfig::layers, stream);
}

} // namespace sinfer::targets::glm5_next::detail
