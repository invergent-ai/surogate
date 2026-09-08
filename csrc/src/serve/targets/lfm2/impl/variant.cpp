#include "targets/lfm2/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear.h"
#include "api/ops/linear_swiglu.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "family/impl/lora_hook.h"
#include "family/impl/mlp_swiglu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::lfm2::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS lfm2_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::lfm2::detail {
namespace {

using family::apply_lora;
using family::apply_lora_qkv;



/// One export profile, and it is W8. The ungated `attn_input_proj` overload admits A16 only,
/// and keeping every linear leaf on the same policy is what makes the workspace figures below
/// exact rather than optimistic: a capacity sized for A16 does not hold the quantised
/// activation an A8 route would ask for.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

[[noreturn]] void no_delta_net(const char* leaf) {
    throw std::logic_error(
        std::string("lfm2: ") + leaf +
        " was called, but this target's non-attending layers run a short convolution, not a "
        "gated delta net: there is no recurrent state, no gating projection and no fused q|k|v "
        "to convolve. Reaching here means the family runtime took the delta-net branch for a "
        "target whose declared mixer is ShortConv.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("lfm2: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& geometry, QType gate_up_qtype,
                                       QType down_qtype, ops::LinearPolicy policy,
                                       std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.intermediate, last});
    family::swiglu_mlp_layout(layout, geometry.intermediate, geometry.hidden, gate_up_qtype,
                              policy, first, last);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, geometry.hidden, geometry.intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("lfm2: invalid weights profile");
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap.
    return family::graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
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
                                   Tensor& gate, Tensor& key, Tensor& value, family::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    // `gate` is deliberately untouched. LFM2's projection has no output-gate rows, and the
    // family skips the sigmoid multiply for this target (attention_output_gate == false), so
    // nothing reads the plane.
    (void)gate;
    ops::attn_input_proj(hidden, weights.query_key_value, query, key, value, stream);
    apply_lora_qkv(weights.query_key_value, hidden, query, key, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, kTextPolicy, workspace, stream);
    apply_lora(weight, 3, attention, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    family::validate_token_interval(first, last);
    (void)profile_qtype(weights_profile);
    // The ungated three-output `attn_input_proj` overload writes q, k and v
    // directly from the row-split parent; it materializes no packed parent and
    // takes no transient storage.
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    // The larger of the two routes this artifact may carry: the profile's row-split format,
    // or the K-quants a GGUF served natively keeps. The layout is planned before the weights
    // are read, so it must hold either.
    return std::max(
        ops::linear_add_workspace_capacity_bytes(profile_qtype(weights_profile), geometry.hidden,
                                                 geometry.query_size(), kTextPolicy, first, last),
        ops::linear_add_workspace_capacity_bytes(QType::Q4_K, geometry.hidden,
                                                 geometry.query_size(), ops::LinearPolicy::A16Only,
                                                 first, last));
}

// ---- Post-mixer (SwiGLU MLP) ----------------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    if (weights.moe.experts_per_token) {
        auto storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
            ops::sparse_moe_geometry(weights.moe), weights.moe.routed_gate_up.qtype,
            weights.moe.routed_down.qtype, hidden.ne[1], hidden.ne[1]));
        WorkspaceArena leaf(storage);
        ops::sparse_moe(hidden, weights.moe, ops::SparseMoeEpilogue::AddResidual,
                        residual, leaf, stream);
        return;
    }
    // The width comes from the weight the SwiGLU reads, so this one function serves whatever
    // size of LFM2 was bound: gate and up are fused, hence half the rows.
    Tensor activation = workspace.alloc(DType::BF16, {weights.gate_up.n / 2, hidden.ne[1]});
    family::swiglu_mlp(hidden, weights.gate_up, activation, kTextPolicy, workspace, stream);
    ops::linear_add(activation, weights.down, residual, kTextPolicy, workspace, stream);
    // down reads the SwiGLU activation, which is exactly the input its adapter
    // was trained against.
    apply_lora(weights.down, 4, activation, residual, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    auto dense = geometry;
    if (geometry.experts) { dense.intermediate = geometry.dense_intermediate; }
    auto bytes = std::max(
        post_mixer_workspace_bytes(dense, qtype, qtype, kTextPolicy, first, last),
        post_mixer_workspace_bytes(dense, QType::Q4_K, QType::Q4_K,
                                   ops::LinearPolicy::A16Only, first, last));
    if (geometry.experts) {
        const ops::SparseMoeGeometry moe{geometry.hidden, geometry.experts,
            geometry.experts_per_token, geometry.intermediate,
            ops::SparseMoeGating::SigmoidBiasTopK, geometry.routed_scale};
        bytes = std::max(bytes, ops::sparse_moe_workspace_capacity_bytes(
            moe, qtype, qtype, first, last));
    }
    return bytes;
}

// ---- The short-convolution mixer -------------------------------------------

void Variant::short_conv_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                                    const GdnProjectionWeights& weights, Tensor& bcx,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    // Normalise the residual, then one projection to B, C and x at once. They stay stacked:
    // the convolution slices them itself, and one matmul against one weight is what the
    // checkpoint stores.
    auto scope       = workspace.scope();
    Tensor normalised = workspace.alloc(DType::BF16, {residual.ne[0], residual.ne[1]});
    ops::rmsnorm(residual, norm_weight, eps, Variant::norm_unit_offset, normalised, stream);
    ops::linear(normalised, weights.in_projection, bcx, kTextPolicy, workspace, stream);
}

std::size_t Variant::short_conv_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.hidden, last});
    {
        auto scope = layout.scope();
        // Either format the artifact may carry: the profile's row-split W8, or the K-quants a
        // GGUF served natively keeps. The layout is planned before the weights are read.
        (void)layout.alloc_bytes(std::max(
            ops::linear_workspace_capacity_bytes(profile_qtype(weights_profile),
                                                 3 * geometry.hidden, geometry.hidden,
                                                 kTextPolicy, first, last),
            ops::linear_workspace_capacity_bytes(QType::Q4_K, 3 * geometry.hidden, geometry.hidden,
                                                 ops::LinearPolicy::A16Only, first, last)));
    }
    return layout.peak_bytes(1);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    family::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    // The mixer's output projection, which the family reaches through the delta net's name
    // because an output projection is the same leaf whichever mixer produced its input.
    ops::linear_add(hidden, weight, residual, kTextPolicy, workspace, stream);
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(
    const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase,
    std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(ops::linear_add_workspace_capacity_bytes(profile_qtype(weights_profile),
                                                             geometry.hidden, geometry.hidden,
                                                             kTextPolicy, first, last),
                    ops::linear_add_workspace_capacity_bytes(QType::Q4_K, geometry.hidden,
                                                             geometry.hidden,
                                                             ops::LinearPolicy::A16Only, first,
                                                             last));
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has to exist. None
// can be reached by an LFM2 topology, and each says so rather than returning quietly: a silent
// no-op would be a layer that contributed nothing to the residual, which reads as a model that
// merely answers badly.
//
// The delta net's leaves and the draft head's, defined once in the family. The short
// convolution's are not among them -- this is the target that runs it -- and the mixer's output
// projection is written above rather than refused, because the family reaches it by the delta
// net's name.
SINFER_FAMILY_UNRUNNABLE_GDN_LEAVES(no_delta_net)
SINFER_FAMILY_UNRUNNABLE_MTP_LEAVES(no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    // Only the magic is this target's: 'LF2B'.
    family::debug_probe_dump(0x4C463242, tag, tensor, layer_count, stream);
}

} // namespace sinfer::targets::lfm2::detail
