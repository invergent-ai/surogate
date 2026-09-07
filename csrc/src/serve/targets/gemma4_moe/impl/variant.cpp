#include "targets/gemma4_moe/impl/variant.h"

#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/linear.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scale.h"
#include "api/ops/sparse_moe.h"

#include "core/device.h"
#include "core/layout.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::gemma4_moe::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS gemma4_moe_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::gemma4_moe::detail {
namespace {

using family::apply_lora;

/// Adapter ports. A bank is keyed by (base weight pointer, port), and Gemma 3
/// fuses nothing -- every logical projection is its own matrix, so the pointer
/// alone would already be a sufficient identity. The ports are kept distinct and
/// numbered the way the family's fused targets number them (q/k/v/o/down = 0..4)
/// so that a reader comparing two targets sees the same module in the same slot;
/// gate and up continue the sequence because this target, unlike the fused ones,
/// can actually serve adapters for them.
constexpr std::int32_t kQueryPort  = 0;
constexpr std::int32_t kKeyPort    = 1;
constexpr std::int32_t kValuePort  = 2;
constexpr std::int32_t kOutputPort = 3;
constexpr std::int32_t kDownPort   = 4;
constexpr std::int32_t kGatePort   = 5;
constexpr std::int32_t kUpPort     = 6;



/// This target has one export profile and it is W8. Keeping every linear leaf on
/// one policy is what makes the workspace figures below exact rather than
/// optimistic: a capacity sized for A16 does not hold the quantized activation
/// an A8 route would ask for.
constexpr ops::LinearPolicy kTextPolicy = ops::LinearPolicy::A16Only;

/// `hidden_activation: gelu_pytorch_tanh` in every released Gemma 3 config. The
/// mode is not cosmetic -- the tanh approximation and the erf form differ by more
/// than rounding -- which is why `ops::gelu_mul` takes it as a parameter.
constexpr ops::GeluMode kMlpActivation = ops::GeluMode::Tanh;

[[noreturn]] void no_linear_layers(const char* leaf) {
    throw std::logic_error(
        std::string("gemma4_moe: ") + leaf +
        " was called, but every one of this target's layers is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the GDN branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("gemma4_moe: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("gemma4_moe: invalid weights profile");
}

/// Transient bytes of one `ops::linear`, measured in its own scope so that a
/// sequence of them costs the largest rather than the sum. Exact for this
/// target's one profile: the W8 route takes no transient storage at all, so
/// every figure below is zero and the scoping only documents the intent. It is
/// also the call that *validates* the physical problem -- a shape the W8 kernel
/// registry does not admit throws here, at plan time, rather than at the first
/// forward.
void account_linear(WorkspaceLayoutBuilder& layout, QType qtype, std::int32_t output_rows,
                    std::int32_t input_rows, std::int32_t first, std::int32_t last) {
    auto scope = layout.scope();
    (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(qtype, output_rows, input_rows,
                                                                  kTextPolicy, first, last));
}

std::size_t attention_projection_workspace_bytes(const family::TextGeometry& g, QType qtype, std::int32_t first,
                                                 std::int32_t last) {
    // Three GEMMs out of the same hidden state, straight into the family's own
    // query/key/value planes. Nothing is materialized between them, so the whole
    // cost is whichever of the three asks for the most transient storage.
    WorkspaceLayoutBuilder layout;
    // The raw value plane the norm reads. Only a layer with its own value projection
    // allocates it, but the layout is planned once for every layer, so it is counted here --
    // and at the *wider* of the two geometries, because one plan serves both kinds of layer.
    (void)layout.alloc(DType::BF16, {g.maximum_kv_size(), last});
    account_linear(layout, qtype, g.maximum_query_size(), g.hidden, first, last);
    account_linear(layout, qtype, g.maximum_kv_size(), g.hidden, first, last);
    account_linear(layout, qtype, g.maximum_kv_size(), g.hidden, first, last);
    return layout.peak_bytes(1);
}

std::size_t attention_output_workspace_bytes(const family::TextGeometry& g, QType qtype,
                                             std::int32_t first, std::int32_t last) {
    // o_proj into one plane, then `post_attention_norm` accumulating from it onto
    // the residual. One plane, not two: the norm writes where the residual add
    // used to, so nothing holds the normalised copy.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    account_linear(layout, qtype, g.hidden, g.maximum_query_size(), first, last);
    return layout.peak_bytes(1);
}

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& g, QType qtype,
                                       std::int32_t first, std::int32_t last) {
    // The dense branch is the dense target's: gate, up, the gated activation and the down
    // projection, four planes because `gelu_mul` requires its output not to overlap its
    // inputs. The mixture adds four hidden-wide planes beside them -- the two normalised
    // inputs the router and the experts read, the experts' own output, and the sum the block's
    // last norm reads -- and then the mixture op's own arena, which is sized separately in
    // `post_mixer_workspace_capacity_bytes` because it depends on the routed codecs.
    const std::int32_t dense = dense_intermediate(g);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {dense, last});
    (void)layout.alloc(DType::BF16, {dense, last});
    (void)layout.alloc(DType::BF16, {dense, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    account_linear(layout, qtype, dense, g.hidden, first, last);
    account_linear(layout, qtype, dense, g.hidden, first, last);
    account_linear(layout, qtype, g.hidden, dense, first, last);
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; the ranges follow the family's measured
    // split-policy transitions until the producer grid reaches its fixed cap. This
    // model's native context stops at 32,768, so the last end is exactly its
    // frontier and the tail beyond it is never reached.
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
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // `gate` is deliberately untouched. Gemma 4's attention has no output gate, and the
    // family skips the sigmoid multiply for this target
    // (Variant::attention_output_gate == false), so nothing reads the plane.
    (void)gate;
    auto scope = workspace.scope();
    ops::linear(hidden, weights.query, query, kTextPolicy, workspace, stream);
    apply_lora(weights.query, kQueryPort, hidden, query, stream);
    ops::linear(hidden, weights.key, key, kTextPolicy, workspace, stream);
    apply_lora(weights.key, kKeyPort, hidden, key, stream);

    // The value, which is where Gemma 4 differs from every other target here in two ways at
    // once.
    //
    // First, it is **RMS-normalised per head with no weight at all** -- `v_norm =
    // RMSNorm(head_dim, with_scale=False)` -- and that happens *here*, before the family
    // applies the query/key norms and rope, because those touch q and k only. The value is
    // never roped.
    //
    // Second, on a `k_eq_v` layer there is no value projection: the value is the key
    // projection's **raw** output, taken before the key norm and before rope. `key` still
    // holds exactly that at this point, which is the whole reason this is done in the
    // projection leaf rather than after it.
    //
    // The norm is per head, so both planes are viewed as `[head_dim, heads * columns]`: the
    // storage is `[heads * head_dim, columns]` contiguous, so element (d, h, t) sits at
    // `t * kv_size + h * head_dim + d` either way and the view is free.
    const std::int32_t columns  = key.ne[1];
    const std::int32_t kv_rows  = key.ne[0];
    const std::int32_t head_dim = weights.head_dim;
    if (head_dim <= 0 || kv_rows % head_dim != 0) {
        throw std::logic_error("gemma4_moe: the key plane is not a whole number of heads wide");
    }
    const std::int32_t heads     = kv_rows / head_dim;
    Tensor value_per_head        = value.view({head_dim, heads * columns});
    if (weights.value_is_key) {
        ops::rmsnorm_unweighted(key.view({head_dim, heads * columns}), TextConfig::rms_epsilon,
                                value_per_head, stream);
    } else {
        // Its own projection, then the same norm. The raw output needs a plane of its own:
        // `rmsnorm_unweighted` forbids aliasing its input with its output, and the raw value
        // is not wanted afterwards.
        Tensor raw = workspace.alloc(DType::BF16, {kv_rows, columns});
        ops::linear(hidden, weights.value, raw, kTextPolicy, workspace, stream);
        apply_lora(weights.value, kValuePort, hidden, raw, stream);
        ops::rmsnorm_unweighted(raw.view({head_dim, heads * columns}), TextConfig::rms_epsilon,
                                value_per_head, stream);
    }
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          const FullAttentionProjectionWeights& weights,
                                          Tensor& residual, family::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                 = workspace.scope();
    const std::int32_t columns = attention.ne[1];
    Tensor projected = workspace.alloc(DType::BF16, {residual.ne[0], columns});
    ops::linear(attention, weight, projected, kTextPolicy, workspace, stream);
    apply_lora(weight, kOutputPort, attention, projected, stream);
    // **Not** zero-centred, unlike every Gemma 3 norm: Gemma 4 stores the full scale, so the
    // kernel applies `w` and not `1 + w`. Same convention as `Variant::norm_unit_offset`,
    // which this target sets to false rather than inheriting.
    //
    // The norm accumulates straight onto the residual. The plane it used to write and the
    // residual add that read it back existed only because `ops::rmsnorm` forbids aliasing
    // its output with its inputs; the fused form is bit-identical to that pair, which
    // sinfer_rmsnorm_test pins.
    ops::rmsnorm_add(projected, weights.post_attention_norm, TextConfig::rms_epsilon,
                     /*unit_offset*/ false, residual, stream);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                                   family::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    family::validate_token_interval(first, last);
    // The larger of the two routes this artifact may carry: the profile's row-split format,
    // or the K-quants a GGUF served natively keeps. The layout is planned before the weights
    // are read, so it must hold either.
    return std::max(
        attention_projection_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        attention_projection_workspace_bytes(geometry, QType::Q4_K, first, last));
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile, family::TextPhase, std::int32_t first, std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(
        attention_output_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        attention_output_workspace_bytes(geometry, QType::Q4_K, first, last));
}

// ---- Post-mixer (gated-GELU MLP, between the other two sandwich norms) ------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         family::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope                      = workspace.scope();
    const std::int32_t columns      = hidden.ne[1];
    const std::int32_t width        = residual.ne[0];
    const std::int32_t intermediate = weights.gate.n;

    // ---- the dense branch, which is the dense target's feed-forward exactly ----------
    //
    // `hidden` is already `pre_feedforward_layernorm(residual)`; the family normalised it on
    // the way in. Gate and up are separate matrices here, where Llama and the Qwen families
    // fuse them: that costs a launch and buys two adaptable modules.
    Tensor gate       = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor up         = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor dense      = workspace.alloc(DType::BF16, {width, columns});
    ops::linear(hidden, weights.gate, gate, kTextPolicy, workspace, stream);
    apply_lora(weights.gate, kGatePort, hidden, gate, stream);
    ops::linear(hidden, weights.up, up, kTextPolicy, workspace, stream);
    apply_lora(weights.up, kUpPort, hidden, up, stream);
    Variant::debug_probe("ffn_gate", gate, stream);
    Variant::debug_probe("ffn_up", up, stream);
    ops::gelu_mul(gate, up, kMlpActivation, activation, stream);
    Variant::debug_probe("ffn_act", activation, stream);
    ops::linear(activation, weights.down, dense, kTextPolicy, workspace, stream);
    apply_lora(weights.down, kDownPort, activation, dense, stream);
    // Four probes the dense target has no use for. The family already brackets the whole
    // feed-forward (`post_attention_norm` in, `post_mlp_residual` out), which localises a
    // mismatch to "the mixture" and no further; these say which branch.
    Variant::debug_probe("ffn_dense_out", dense, stream);

    // ---- the routed branch -----------------------------------------------------------
    //
    // Both of its inputs are projections of the **residual**, not of `hidden`. The reference
    // is explicit about it (`hidden_states_flat = residual.reshape(...)` before the router and
    // before `pre_feedforward_layernorm_2`), and it matters: `hidden` has already been through
    // a different norm with a different weight.
    Tensor routed_input = workspace.alloc(DType::BF16, {width, columns});
    Tensor router_input = workspace.alloc(DType::BF16, {width, columns});
    Tensor routed       = workspace.alloc(DType::BF16, {width, columns});
    ops::rmsnorm(residual, weights.pre_feedforward_norm_routed, TextConfig::rms_epsilon,
                 /*unit_offset*/ false, routed_input, stream);
    // The router's own view of the token: `Gemma4RMSNorm(with_scale=False)` times a learned
    // per-channel vector times `hidden ** -0.5`. The first two are one weighted RMSNorm --
    // Gemma 4 applies `normed * w` with no unit offset, so a weightless norm followed by a
    // per-channel multiply *is* a weighted norm with that vector as the weight. Only the
    // scalar is left over, and it is a function of the width rather than a stored tensor.
    ops::rmsnorm(residual, weights.router_scale, TextConfig::rms_epsilon,
                 /*unit_offset*/ false, router_input, stream);
    ops::scale(router_input, router_input_scale(width), stream);
    Variant::debug_probe("ffn_router_in", router_input, stream);

    // `AddResidual` is the op's only epilogue, and the routed output has to reach its own norm
    // before it reaches anything else -- so it accumulates onto zero rather than onto the
    // residual. The memset is one hidden-wide plane per layer; teaching the op a second
    // epilogue would touch four kernel families to save it.
    CUDA_CHECK(cudaMemsetAsync(routed.data, 0, routed.bytes(), stream));
    {
        auto moe_scope           = workspace.scope();
        const DeviceSpan storage = workspace.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
            kMoeGeometry, weights.op.routed_gate_up.qtype, weights.op.routed_down.qtype, columns,
            columns));
        WorkspaceArena moe_workspace(storage);
        ops::sparse_moe(routed_input, router_input, weights.op,
                        ops::SparseMoeEpilogue::AddResidual, routed, moe_workspace, stream,
                        ops::SparseMoeRoundHook{});
    }

    Variant::debug_probe("ffn_routed_out", routed, stream);

    // ---- the two branches meet -------------------------------------------------------
    //
    // `post_feedforward_layernorm_1` over the dense output, `post_feedforward_layernorm_2`
    // over the routed one, summed; then the block's own `post_feedforward_layernorm` over the
    // sum, onto the residual. `rmsnorm_add` accumulates, so the second call adds the routed
    // half onto the first's result rather than through a fifth plane.
    Tensor combined = workspace.alloc(DType::BF16, {width, columns});
    ops::rmsnorm(dense, weights.post_feedforward_norm_dense, TextConfig::rms_epsilon,
                 /*unit_offset*/ false, combined, stream);
    ops::rmsnorm_add(routed, weights.post_feedforward_norm_routed, TextConfig::rms_epsilon,
                     /*unit_offset*/ false, combined, stream);
    Variant::debug_probe("ffn_combined", combined, stream);
    ops::rmsnorm_add(combined, weights.post_feedforward_norm, TextConfig::rms_epsilon,
                     /*unit_offset*/ false, residual, stream);
    // And the last thing a Gemma 4 block does: scale its whole output. The reference
    // multiplies the residual *after* the feed-forward's add, so this is a scale on
    // everything the block has accumulated, not on the feed-forward's contribution.
    ops::scale(residual, weights.layer_scalar_value, stream);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                         WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    const QType qtype = profile_qtype(weights_profile);
    // The dense half's planes and matmul scratch, plus the mixture op's own arena, which is
    // allocated *inside* this leaf's scope and so adds rather than overlaps. Both are sized
    // for either route this artifact may carry: the profile's group-wise int8, or the
    // K-quants a GGUF served natively keeps -- and a GGUF may hold a different down type from
    // its gate/up, layer by layer.
    const std::size_t planes =
        std::max(post_mixer_workspace_bytes(geometry, qtype, first, last),
                 post_mixer_workspace_bytes(geometry, QType::Q4_K, first, last));
    const std::size_t mixture = std::max({
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, qtype, qtype, first, last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q4_K, first,
                                                 last),
        ops::sparse_moe_workspace_capacity_bytes(kMoeGeometry, QType::Q4_K, QType::Q6_K, first,
                                                 last),
    });
    return planes + mixture;
}

// ---- Leaves this target cannot run -----------------------------------------
//
// The family runtime is a template over this interface, so every leaf below has
// to exist. None of them can be reached by a Gemma 3 topology, and each says so
// rather than returning quietly: a silent no-op here would be a layer that
// contributed nothing to the residual, which reads as a model that merely
// answers badly.

// Every leaf this target cannot run, defined once in the family: see
// family/impl/runtime/unrunnable_leaves.h. The two arguments are this
// target's own refusal messages.
SINFER_FAMILY_UNRUNNABLE_LEAVES(no_linear_layers, no_speculation)

void Variant::debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    // Only the magic is this target's: 'G4MX'. Everything else -- which rounds are
    // captured, how the occurrence is counted, the header layout -- is the family's.
    family::debug_probe_dump(0x47344D58, tag, tensor, TextConfig::layers, stream);
}

} // namespace sinfer::targets::gemma4_moe::detail
