#include "targets/glm5_next/impl/variant.h"

#include "api/ops/causal_conv1d_silu.h"
#include "api/ops/embedding.h"
#include "api/ops/gdn_gating.h"
#include "api/ops/hyper_connection.h"
#include "api/ops/linear.h"
#include "api/ops/silu_mul.h"
#include "api/ops/head_linear.h"
#include "api/ops/manifold_hyper_connection.h"
#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scatter.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "family/impl/lora_hook.h"
#include "family/impl/moe/expert_cache.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"

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
/// The value half of the layer's latent expansion, set by `attention_projection` for the
/// output hook that follows it: the family hands that hook the output weight alone, and the
/// attended latent has to be unfolded through this before the output weight can read it.
thread_local const Weight* t_value_unfold = nullptr;

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
    if (hc.weights.mix.n <= 0) {
        throw std::logic_error(
            "glm5_next: the hyper-connection of layer " + std::to_string(hc.layer) +
            " is not bound on this stage, so this round is running a layer the stage does not "
            "hold");
    }
    ops::manifold_hyper_connection_mix(residual, hc.weights, kStreams, eps,
                                       TextConfig::hc_epsilon,
                                       TextConfig::hc_sinkhorn_iterations, collapsed, post, comb,
                                       workspace, stream);
    ops::rmsnorm(collapsed, norm, eps, /*unit_offset=*/false, hidden, stream);
    // Parity probes. Under SUROGATE_SERVE_DUMP_RESIDUAL these are the only tensors of this
    // stack a reference implementation can be compared against directly: the stream collapse,
    // the two mixings it produced, and (below) the residual the recombination left.
    Variant::debug_probe("hc_collapsed", collapsed, stream);
    Variant::debug_probe("hc_normed", hidden, stream);
    Variant::debug_probe("hc_post", post, stream);
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
    Variant::debug_probe("block_output", block_output, stream);
    ops::manifold_hyper_connection_combine(block_output, t_post, t_comb, kStreams, residual,
                                           stream);
    Variant::debug_probe("residual", residual, stream);
    t_post          = Tensor{};
    t_comb          = Tensor{};
    t_mixing_device = -1;
}

/// The output-gate plane as a matrix. The family hands it over shaped by head -- `[value_dim,
/// heads, T]` in a prefill and `[value_dim, W, B]` in a decode round -- and a projection writes
/// `[N, T]`, so the rows the weight produces name the split.
Tensor gate_matrix(const Tensor& output_gate, std::int32_t rows) {
    const std::int64_t total = output_gate.numel();
    if (rows <= 0 || total % rows != 0) {
        throw std::logic_error("glm5_next: the output gate plane is not a whole number of rows");
    }
    return output_gate.view({rows, static_cast<std::int32_t>(total / rows)});
}

/// One projection, named. An unbound weight is a binding mistake -- a stage that did not
/// upload something it runs, or a layer whose inventory index went astray -- and the op it
/// reaches sees only `n` and `k`, so it can say the shape is wrong but not whose. Every linear
/// leaf of this target goes through here so the answer is in the message.
void project(const char* what, const Tensor& x, const Weight& w, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    if (w.n <= 0 || w.k <= 0 || w.qdata == nullptr) {
        throw std::logic_error(std::string("glm5_next: ") + what +
                               " is not bound on this stage (n " + std::to_string(w.n) + ", k " +
                               std::to_string(w.k) + ")");
    }
    ops::linear(x, w, out, kTextPolicy, workspace, stream);
}

[[noreturn]] void no_short_conv(const char* leaf) {
    throw std::logic_error(
        std::string("glm5_next: ") + leaf +
        " was called, but this target's non-attending layers run Kimi Delta Attention, not a "
        "short convolution. Reaching here means the family runtime took the short-conv branch "
        "for a target whose declared mixer is KimiDelta.");
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

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // The same cuts the other draft heads capture at: the final AR window E+2K at the
    // split-policy transitions until the grid reaches its cap, and one concrete kernel
    // implementation per range at the T=4/5/6 launch boundaries.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
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

namespace {

/// The latent: `kv_a` then its norm, straight into the key buffer. In the absorbed form it *is*
/// the layer's key, and its value, and the cache holds it as such -- 512 wide, one head.
template <class Payload>
void latent_key(const Tensor& hidden, const Payload& weights, Tensor& key,
                WorkspaceArena& workspace, cudaStream_t stream) {
    project("mla/kv_a", hidden, weights.kv_a, key, workspace, stream);
    ops::rmsnorm(key, weights.kv_a_norm, weights.rms_epsilon, /*unit_offset=*/false, key, stream);
}

/// The absorbed query: the low rank, its norm, the heads, and every head folded through its own
/// key half into the latent, carrying the sqrt 2 that turns the kernels' 1/sqrt(512) into the
/// model's 1/sqrt(256). Each low rank is normalised whole, which is what replaces the per-head
/// query and key norms every other target in this family carries.
template <class Payload>
void absorbed_query(const Tensor& hidden, const Payload& weights, Tensor& query,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    Tensor q_low   = workspace.alloc(DType::BF16, {weights.query_a.n, tokens});
    Tensor q_heads = workspace.alloc(DType::BF16, {weights.query_b.n, tokens});
    project("mla/query_a", hidden, weights.query_a, q_low, workspace, stream);
    ops::rmsnorm(q_low, weights.query_a_norm, weights.rms_epsilon, /*unit_offset=*/false, q_low,
                 stream);
    project("mla/query_b", q_low, weights.query_b, q_heads, workspace, stream);
    const std::int32_t heads = weights.k_b.n / weights.kv_a.n;
    ops::head_linear(q_heads, weights.k_b, heads, kAbsorbScale, query, stream);
}

/// The attended latent back through the value half, per head, then the output projection.
void unfold_and_project(const Tensor& attention, const Weight& value_unfold, const Weight& output,
                        Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens  = attention.ne[1];
    const std::int32_t heads   = static_cast<std::int32_t>(attention.ne[0] / value_unfold.k);
    auto scope      = workspace.scope();
    Tensor unfolded = workspace.alloc(DType::BF16, {value_unfold.n, tokens});
    ops::head_linear(attention, value_unfold, heads, 1.0F, unfolded, stream);
    project("mla/output", unfolded, output, out, workspace, stream);
    apply_lora(output, 3, unfolded, out, stream);
}

std::size_t absorbed_query_capacity(const family::TextGeometry& geometry,
                                    WeightsProfile weights_profile, std::int32_t first,
                                    std::int32_t last) {
    // The query low rank and the heads `query_b` produces; the absorption is a per-head kernel
    // with no transient of its own.
    return plane_bytes(geometry.q_lora_rank, last, DType::BF16) +
           plane_bytes(geometry.query_heads * TextConfig::qk_head_dim, last, DType::BF16) +
           std::max(linear_capacity(weights_profile, geometry.q_lora_rank, geometry.hidden, first,
                                    last),
                    linear_capacity(weights_profile,
                                    geometry.query_heads * TextConfig::qk_head_dim,
                                    geometry.q_lora_rank, first, last));
}

std::size_t latent_key_capacity(const family::TextGeometry& geometry,
                                WeightsProfile weights_profile, std::int32_t first,
                                std::int32_t last) {
    return linear_capacity(weights_profile, geometry.kv_lora_rank, geometry.hidden, first, last);
}

std::size_t unfold_capacity(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                            std::int32_t first, std::int32_t last) {
    // The unfolded heads, then the output projection's transient.
    return plane_bytes(geometry.query_heads * TextConfig::v_head_dim, last, DType::BF16) +
           linear_capacity(weights_profile, geometry.hidden,
                           geometry.query_heads * TextConfig::v_head_dim, first, last);
}

} // namespace

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // `gate` is deliberately untouched: this attention writes no output-gate rows and the family
    // skips the multiply (`Variant::attention_output_gate == false`).
    (void)gate;
    latent_key(hidden, weights, key, workspace, stream);
    absorbed_query(hidden, weights, query, workspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(value.data, key.data,
                               static_cast<std::size_t>(key.numel()) * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
    t_value_unfold = &weights.v_b;
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (t_value_unfold == nullptr) {
        throw std::logic_error("glm5_next: attention output before its projection on this thread");
    }
    // The attended latent, per head, back through the value half: the heads at their own
    // width, which is what the output projection was trained to read.
    const std::int32_t tokens = attention.ne[1];
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {weight.n, tokens});
    unfold_and_project(attention, *t_value_unfold, weight, out, workspace, stream);
    combine_into(out, residual, stream);
}

// ---- The draft head's attention: the same latent attention, no hyper-connection ----

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights,
                                       Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    (void)gate; // no output-gate rows; the family's tail skips the multiply
    latent_key(hidden, weights, key, workspace, stream);
    absorbed_query(hidden, weights, query, workspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(value.data, key.data,
                               static_cast<std::size_t>(key.numel()) * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    latent_key(hidden, weights, key, workspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(value.data, key.data,
                               static_cast<std::size_t>(key.numel()) * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToDevice, stream));
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    (void)gate;
    absorbed_query(hidden, weights, query, workspace, stream);
}

void Variant::mtp_attention_output_projection(const Tensor& attention,
                                              const MtpAttentionProjectionWeights& weights,
                                              const Weight& output, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    unfold_and_project(attention, weights.v_b, output, out, workspace, stream);
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
    project("kda/decay_a", hidden, weights.decay_a, decay_low, workspace, stream);
    project("kda/decay_b", decay_low, weights.decay_b, decay, workspace, stream);
    project("kda/beta", hidden, weights.beta, update, workspace, stream);
    ops::kda_gating(decay, update, weights.a_log, weights.decay_bias, weights.gate_lower_bound, g,
                    beta, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, family::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    Tensor gate_low = workspace.alloc(DType::BF16, {weights.gate_a.n, tokens});
    Tensor gate     = gate_matrix(output_gate, weights.gate_b.n);
    project("kda/query_key_value", hidden, weights.query_key_value, qkv, workspace, stream);
    // The output gate is low-rank, so it is two projections rather than rows of the fused one.
    project("kda/gate_a", hidden, weights.gate_a, gate_low, workspace, stream);
    project("kda/gate_b", gate_low, weights.gate_b, gate, workspace, stream);
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
    project("kda/query_key_value", flat_hidden, weights.query_key_value, projected, workspace,
            stream);
    // One convolution over the fused q|k|v, checkpointed per lane, then the three spans are
    // taken apart. The fused snapshot op the delta-net targets use projects the output gate as
    // rows of the same parent; here the gate is low-rank and cannot ride along.
    Tensor projected_lanes = projected.view({rows, width, batch});
    Tensor convolved_lanes = convolved.view({rows, width, batch});
    ops::causal_conv1d_silu_snapshot(projected_lanes, conv_weight, conv_states, valid_columns,
                                     initial_slot, snapshot_base_slot, convolved_lanes, stream);
    // The family hands q, k and v over shaped by lane; the extract wants [D', T].
    Tensor flat_query = query.view({query.ne[0], width * batch});
    Tensor flat_key   = key.view({key.ne[0], width * batch});
    Tensor flat_value = value.view({value.ne[0], width * batch});
    ops::extract_bf16_columns(convolved, 0, flat_query, stream);
    ops::extract_bf16_columns(convolved, per, flat_key, stream);
    ops::extract_bf16_columns(convolved, 2 * per, flat_value, stream);
    Tensor flat_gate = gate_matrix(output_gate, weights.gate_b.n);
    project("kda/gate_a", flat_hidden, weights.gate_a, gate_low, workspace, stream);
    project("kda/gate_b", gate_low, weights.gate_b, flat_gate, workspace, stream);
}

void Variant::gdn_input_projection_record(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    // A speculative round records the projection the convolution reads instead of advancing
    // the convolution's state: a rejected draft is undone by folding the accepted prefix from
    // the record, so the recurrent state is re-derived rather than rolled back. The projection
    // is this layer's own; the convolution from a state it does not write is the family's,
    // the same one the delta-net targets compose with.
    const std::int32_t columns = hidden.ne[1] * hidden.ne[2];
    Tensor flat_hidden         = hidden.view({hidden.ne[0], columns});
    Tensor flat_record         = conv_record.view({conv_record.ne[0], columns});
    gdn_input_projection(flat_hidden, weights, flat_record, output_gate, phase, workspace,
                         stream);
    ops::detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                  valid_columns, initial_slots, query, key, value,
                                                  stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {weight.n, hidden.ne[1]});
    project("kda/output", hidden, weight, out, workspace, stream);
    apply_lora(weight, 3, hidden, out, stream);
    combine_into(out, residual, stream);
}

// ---- The feed-forward -------------------------------------------------------

namespace {

/// The dense feed-forward as two row-range projections and a pointwise, not the fused SwiGLU.
/// That op's kernels are instantiated per (intermediate, k), and this shape -- 12,288 over
/// 4,096 -- would be three more instantiations for three of forty-five layers. `linear_rows`
/// reads the same fused parent and is shape-free. Written to `out`, not added.
void dense_feed_forward(const Tensor& hidden, const FeedForwardPayload& weights, Tensor& out,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    auto scope                = workspace.scope();
    const std::int32_t width  = weights.gate_up.n / 2;
    Tensor gate = workspace.alloc(DType::BF16, {width, tokens});
    Tensor up   = workspace.alloc(DType::BF16, {width, tokens});
    ops::linear_rows(hidden, weights.gate_up, 0, gate, &workspace, stream);
    ops::linear_rows(hidden, weights.gate_up, width, up, &workspace, stream);
    // Clamped, like every other SwiGLU this model has.
    ops::silu_mul(gate, up, gate, kMoeGeometry.swiglu_limit, stream);
    project("mlp/down", gate, weights.down, out, workspace, stream);
}

std::size_t feed_forward_capacity(const family::TextGeometry& geometry,
                                  WeightsProfile weights_profile, std::int32_t first,
                                  std::int32_t last) {
    const QType qtype = profile_qtype(weights_profile);
    // Whichever of the two feed-forwards is larger. A layer is one or the other, but the
    // layout is planned for the stack rather than per layer.
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
        2 * plane_bytes(geometry.dense_intermediate, last, DType::BF16) +
        std::max(linear_capacity(weights_profile, 2 * geometry.dense_intermediate, geometry.hidden,
                                 first, last),
                 linear_capacity(weights_profile, geometry.hidden, geometry.dense_intermediate,
                                 first, last));
    return std::max(mixture, dense);
}

} // namespace

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
        const family::BankedMixture mixture{weights.layer,         weights.layers,
                                            &weights.moe,          weights.host_bank_q4,
                                            weights.host_gate_up,  weights.host_down};
        // Experts in the host bank go through the expert cache: resident ones from the device
        // pool, the rest fetched over PCIe or -- the split's share -- computed on the host and
        // added back here before the recombination. A layer whose experts are on the card
        // takes the plain route; the pool holds nothing it would want.
        family::ExpertCache* cache =
            mixture.banked()
                ? &family::ExpertCache::for_current_device(kMoeGeometry, weights.layers)
                : nullptr;
        if (cache != nullptr && cache->enabled()) {
            cache->run(mixture, hidden, out, workspace, stream);
            cache->add_pending_partial(out, stream);
        } else if (mixture.host_bank_q4) {
            // The Q4 bank's routed Weights are a base pointer and a shape; only the cache
            // decodes them, so without a pool there is nothing the plain route could read.
            throw std::logic_error("glm5_next: the Q4 host expert bank needs the expert cache "
                                   "(the pool could not be sized; pass --expert-slots)");
        } else {
            run_sparse_moe(hidden, weights.moe, out, workspace, stream);
        }
    } else {
        dense_feed_forward(hidden, weights, out, workspace, stream);
    }
    combine_into(out, residual, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    // The head's residual is one stream, so the block's output is added rather than
    // recombined -- which for the mixture is its epilogue's own destination.
    if (weights.sparse) {
        run_sparse_moe(hidden, weights.moe, residual, workspace, stream);
        return;
    }
    auto scope = workspace.scope();
    Tensor out = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    dense_feed_forward(hidden, weights, out, workspace, stream);
    ops::residual_add(out, residual, stream);
}

// ---- Workspace capacities ---------------------------------------------------

std::size_t
Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                       WeightsProfile weights_profile,
                                                       family::TextPhase, std::int32_t first,
                                                       std::int32_t last) {
    family::validate_token_interval(first, last);
    // The latent goes straight into the key buffer; the query's transients are its own.
    return std::max(latent_key_capacity(geometry, weights_profile, first, last),
                    absorbed_query_capacity(geometry, weights_profile, first, last));
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    // The block's output plane, then the unfold's own.
    return plane_bytes(geometry.hidden, last, DType::BF16) +
           unfold_capacity(geometry, weights_profile, first, last);
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
    // The block's own output plane, then the feed-forward's.
    return plane_bytes(geometry.hidden, last, DType::BF16) +
           feed_forward_capacity(geometry, weights_profile, first, last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase phase,
    std::int32_t batch_size, std::int32_t first, std::int32_t last) {
    // The record is the projection itself, written where the fold will read it; only the
    // projection's own transients are the leaf's.
    family::validate_token_interval(first, last);
    const std::int32_t columns = last * std::max(batch_size, 1);
    return gdn_input_projection_workspace_capacity_bytes(geometry, weights_profile, phase, 1,
                                                         columns);
}

// ---- The draft head's own capacities ----------------------------------------

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(latent_key_capacity(geometry, WeightsProfile::GroupwiseInt, first, last),
                    absorbed_query_capacity(geometry, WeightsProfile::GroupwiseInt, first, last));
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return latent_key_capacity(geometry, WeightsProfile::GroupwiseInt, first, last);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return absorbed_query_capacity(geometry, WeightsProfile::GroupwiseInt, first, last);
}

std::size_t Variant::mtp_attention_output_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return unfold_capacity(geometry, WeightsProfile::GroupwiseInt, first, last);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                             std::int32_t first,
                                                             std::int32_t last) {
    family::validate_token_interval(first, last);
    // A dense head writes its output to a plane of its own before the add; the mixture adds in
    // place. Planned for either.
    return plane_bytes(geometry.hidden, last, DType::BF16) +
           feed_forward_capacity(geometry, WeightsProfile::GroupwiseInt, first, last);
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has to exist. None
// can be reached by this topology, and each says so rather than returning quietly: a silent
// no-op would be a layer that contributed nothing to the residual, which reads as a model that
// merely answers badly. Only the short convolution's are left: this target's mixer is a delta
// rule, and its draft head runs above.
SINFER_FAMILY_UNRUNNABLE_SHORT_CONV_LEAVES(no_short_conv)

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    // Only the magic is this target's: 'G53F'.
    family::debug_probe_dump(0x47353346, tag, tensor, 2 * TextConfig::layers, stream);
}

} // namespace sinfer::targets::glm5_next::detail
