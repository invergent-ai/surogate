#include "targets/qwen3_moe/impl/variant.h"

#include "api/ops/attn_input_proj.h"
#include "api/ops/linear_add.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::qwen3_moe::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS qwen3_moe_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::qwen3_moe::detail {
namespace {

using family::apply_lora;
using family::apply_lora_qkv;



/// One export profile, and it is W8. The ungated `attn_input_proj` overload admits A16 only,
/// and keeping the attention leaves on the same policy is what makes the workspace figures
/// below exact rather than optimistic.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

/// The mixture, over one arena of its own. The op sizes its scratch from the geometry and the
/// two routed codecs, which a GGUF source may vary between layers.
void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights, Tensor& residual,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
        kMoeGeometry, weights.routed_gate_up.qtype, weights.routed_down.qtype, hidden.ne[1],
        hidden.ne[1]));
    WorkspaceArena leaf_workspace(storage);
    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, residual,
                    leaf_workspace, stream);
}

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3_moe: ") + leaf +
        " was called, but every layer of this architecture is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the linear branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("qwen3_moe: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("qwen3_moe: invalid weights profile");
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
    // The larger of the two routes this artifact may carry: the profile's row-split format, or
    // the K-quants a GGUF served natively keeps. The layout is planned before the weights are
    // read, so it must hold either.
    return std::max(
        ops::linear_add_workspace_capacity_bytes(profile_qtype(weights_profile), geometry.hidden,
                                                 geometry.query_size(), kTextPolicy, first, last),
        ops::linear_add_workspace_capacity_bytes(QType::Q4_K, geometry.hidden,
                                                 geometry.query_size(), ops::LinearPolicy::A16Only,
                                                 first, last));
}

// ---- Post-mixer (the routed mixture) ---------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    run_sparse_moe(hidden, weights.op, residual, workspace, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry&,
                                                         WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    // The layout is planned before the weights are read, so it must hold either route this
    // artifact may carry: the profile's group-wise int8, or the K-quants a GGUF served natively
    // keeps -- and a GGUF may hold a different down type from its gate/up, layer by layer.
    const QType qtype = profile_qtype(weights_profile);
    return std::max({
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, qtype, qtype, first, last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q4_K, first,
                                                 last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q6_K, first,
                                                 last),
    });
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has to exist. None
// can be reached by this topology, and each says so rather than returning quietly: a silent
// no-op would be a layer that contributed nothing to the residual, which reads as a model that
// merely answers badly.

// Every leaf this target cannot run, defined once in the family: see
// family/impl/runtime/unrunnable_leaves.h. The two arguments are this target's own refusals.
SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear_layers, no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    // Only the magic is this target's: 'Q3ME'.
    family::debug_probe_dump(0x51334D45, tag, tensor, TextConfig::layers, stream);
}

} // namespace sinfer::targets::qwen3_moe::detail
