#include "targets/qwen3/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_swiglu.h"

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

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen3::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::qwen3::detail {
namespace {

using family::apply_lora;
using family::apply_lora_qkv;



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
    throw std::invalid_argument("qwen3: invalid weights profile");
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
    // `gate` is deliberately untouched. Qwen3's projection has no output-gate
    // rows, and the family skips the sigmoid multiply for this target
    // (Variant::attention_output_gate == false), so nothing reads the plane.
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
    // The width comes from the weight the SwiGLU reads, so this one function serves whatever
    // size of Qwen3 was bound: gate and up are fused, hence half the rows.
    const bool separate = weights.gate.qdata != nullptr;
    Tensor activation = workspace.alloc(DType::BF16, {separate ? weights.gate.n : weights.gate_up.n / 2, hidden.ne[1]});
    if (separate) {
        auto pair_scope = workspace.scope();
        Tensor gate = workspace.alloc(DType::BF16, {weights.gate.n, hidden.ne[1]});
        Tensor up = workspace.alloc(DType::BF16, {weights.up.n, hidden.ne[1]});
        ops::linear(hidden, weights.gate, gate, kTextPolicy, workspace, stream);
        ops::linear(hidden, weights.up, up, kTextPolicy, workspace, stream);
        apply_lora(weights.gate, family::kGatePort, hidden, gate, stream);
        apply_lora(weights.up, family::kUpPort, hidden, up, stream);
        ops::silu_mul(gate, up, activation, stream);
    } else {
        family::swiglu_mlp(hidden, weights.gate_up, activation, kTextPolicy, workspace, stream);
    }
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
    return std::max(
        post_mixer_workspace_bytes(geometry, qtype, qtype, kTextPolicy, first, last),
        post_mixer_workspace_bytes(geometry, QType::Q4_K, QType::Q4_K,
                                   ops::LinearPolicy::A16Only, first, last));
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Qwen3 topology, and each says so
// rather than returning quietly: a silent no-op here would be a layer that
// contributed nothing to the residual, which reads as a model that merely
// answers badly.

// Every leaf this target cannot run, defined once in the family: see
// family/impl/runtime/unrunnable_leaves.h. The two arguments are this
// target's own refusal messages.
SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear_layers, no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    // Only the magic is this target's: 'Q3PB'. Everything else -- which rounds
    // are captured, how the occurrence is counted, the header layout -- is the
    // family's, and lived in nine byte-identical copies before it moved there.
    family::debug_probe_dump(0x51335042, tag, tensor, layer_count, stream);
}

} // namespace sinfer::targets::qwen3::detail
