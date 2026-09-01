#include "targets/qwen3/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_swiglu.h"

#include "targets/qwen3_6/impl/lora_hook.h"

#include <algorithm>
#include <stdexcept>

#define SINFER_QWEN36_VARIANT    ::sinfer::targets::qwen3::detail::Variant
#define SINFER_QWEN36_RUNTIME_NS qwen3_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace sinfer::targets::qwen3::detail {
namespace {

using qwen3_6::apply_lora;
using qwen3_6::apply_lora_qkv;

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

/// This target has one export profile and it is W8. The ungated
/// `attn_input_proj` overload it uses admits A16 only, and keeping the linear
/// leaves on the same policy is what makes the workspace figures below exact
/// rather than optimistic: a capacity sized for A16 does not hold the quantized
/// activation an A8 route would ask for.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3: ") + leaf +
        " was called, but every one of this target's 28 layers is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the GDN branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3: invalid weights profile");
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t, std::uint32_t) {
    return {};
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t,
                                                                  std::uint32_t) {
    return {};
}

// ---- Attention -------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    // `gate` is deliberately untouched. Qwen3's projection has no output-gate
    // rows, and the family skips the sigmoid multiply for this target
    // (Variant::attention_output_gate == false), so nothing reads the plane.
    (void)gate;
    ops::attn_input_proj(hidden, weights.query_key_value, query, key, value, stream);
    apply_lora_qkv(weights.query_key_value, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, kTextPolicy, workspace, stream);
    apply_lora(weight, 3, attention, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    (void)profile_qtype(weights_profile);
    // The ungated three-output `attn_input_proj` overload writes q, k and v
    // directly from the row-split parent; it materializes no packed parent and
    // takes no transient storage.
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return ops::linear_add_workspace_capacity_bytes(profile_qtype(weights_profile),
                                                    TextConfig::hidden, TextConfig::query_size,
                                                    kTextPolicy, first, last);
}

// ---- Post-mixer (SwiGLU MLP) ----------------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, kTextPolicy, workspace, stream);
    ops::linear_add(activation, weights.down, residual, kTextPolicy, workspace, stream);
    // down reads the SwiGLU activation, which is exactly the input its adapter
    // was trained against.
    apply_lora(weights.down, 4, activation, residual, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    return post_mixer_workspace_bytes(qtype, qtype, kTextPolicy, first, last);
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Qwen3 topology, and each says so
// rather than returning quietly: a silent no-op here would be a layer that
// contributed nothing to the residual, which reads as a model that merely
// answers badly.

void Variant::gdn_input_projection(const Tensor&, const GdnProjectionWeights&, Tensor&, Tensor&,
                                   qwen3_6::TextPhase, WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection");
}

void Variant::gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&,
                                            const Tensor&, Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            qwen3_6::TextPhase, WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection_snapshot");
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, Tensor&,
                                          Tensor&, Tensor&, Tensor&, Tensor&, qwen3_6::TextPhase,
                                          WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_input_projection_record");
}

void Variant::gdn_output_projection(const Tensor&, const Weight&, Tensor&, qwen3_6::TextPhase,
                                    WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_output_projection");
}

void Variant::gdn_norm_control_projection(const Tensor&, const Tensor&, float,
                                          const GdnProjectionWeights&, Tensor&, Tensor&, Tensor&,
                                          WorkspaceArena&, cudaStream_t) {
    no_linear_layers("gdn_norm_control_projection");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile,
                                                                            qwen3_6::TextPhase,
                                                                            std::int32_t,
                                                                            std::int32_t,
                                                                            std::int32_t) {
    return 0;
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
                                                                    std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                       Tensor&, Tensor&, Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_attention_projection");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_kv_projection");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    no_speculation("mtp_q_gate_projection");
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&, WorkspaceArena&,
                             cudaStream_t) {
    no_speculation("mtp_post_mixer");
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

} // namespace sinfer::targets::qwen3::detail
