#include "targets/gemma4_e/impl/variant.h"

#include "api/ops/gelu.h"
#include "api/ops/gelu_mul.h"
#include "api/ops/embedding.h"
#include "api/ops/linear.h"
#include "api/ops/residual_add.h"
#include "api/ops/scale.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scale.h"

#include "core/device.h"
#include "core/layout.h"
#include "family/impl/lora_hook.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SINFER_FAMILY_VARIANT    ::sinfer::targets::gemma4_e::detail::Variant
#define SINFER_FAMILY_RUNTIME_NS gemma4_e_runtime
#include "family/impl/runtime/instantiate.h"
#include "family/impl/runtime/target_support.h"
#include "family/impl/runtime/unrunnable_leaves.h"

namespace sinfer::targets::gemma4_e::detail {
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
/// The per-layer input's two matrices. Adaptable like any other projection: they are weights
/// of their own, so an adapter has a base to add a delta to.
constexpr std::int32_t kPerLayerGatePort       = 7;
constexpr std::int32_t kPerLayerProjectionPort = 8;



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
        std::string("gemma4_e: ") + leaf +
        " was called, but every one of this target's layers is full attention; it has no "
        "linear-attention mixer, no convolution state and no gating projection. Reaching here "
        "means the family runtime resolved a layer to the GDN branch, which its topology "
        "(full_attention_interval == 1, gdn_layers() == 0) cannot produce.");
}

[[noreturn]] void no_speculation(const char* leaf) {
    throw std::logic_error(
        std::string("gemma4_e: ") + leaf +
        " was called, but this target has no MTP block and no DFlash tower; --spec is refused when "
        "the artifact is bound. Reaching here means a speculative round started without one.");
}

QType profile_qtype(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
        return QType::W8G32_F16S;
    }
    throw std::invalid_argument("gemma4_e: invalid weights profile");
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

std::size_t post_mixer_workspace_bytes(const family::TextGeometry& g, QType qtype, std::int32_t first, std::int32_t last) {
    // gate, up, the gated activation and the down projection. Four planes where
    // Llama's SwiGLU MLP needs one: `gelu_mul` requires its output not to overlap
    // its inputs, so those three stand apart. The normalised copy is gone --
    // the norm accumulates onto the residual instead of through a fifth plane.
    // At the widest feed-forward any layer holds: `use_double_wide_mlp` doubles it on the
    // layers that share their key/value planes, and one plan serves every layer.
    WorkspaceLayoutBuilder layout;
    const std::int32_t ffn = g.maximum_intermediate();
    (void)layout.alloc(DType::BF16, {ffn, last});
    (void)layout.alloc(DType::BF16, {ffn, last});
    (void)layout.alloc(DType::BF16, {ffn, last});
    (void)layout.alloc(DType::BF16, {g.hidden, last});
    account_linear(layout, qtype, ffn, g.hidden, first, last);
    account_linear(layout, qtype, ffn, g.hidden, first, last);
    account_linear(layout, qtype, g.hidden, ffn, first, last);
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
    // A layer that shares an earlier layer's keys and values has no key or value projection
    // at all -- it holds a query and an output, and attends over planes it did not write. The
    // family reads neither `key` nor `value` for such a layer: it takes the cached attention
    // path, which appends nothing.
    if (!weights.owns_kv) { return; }
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
        throw std::logic_error("gemma4_e: the key plane is not a whole number of heads wide");
    }
    const std::int32_t heads     = kv_rows / head_dim;
    Tensor value_per_head        = value.view({head_dim, heads * columns});
    if (weights.value_is_key) {
        ops::rmsnorm_unweighted(key.view({head_dim, heads * columns}), weights.rms_epsilon,
                                value_per_head, stream);
    } else {
        // Its own projection, then the same norm. The raw output needs a plane of its own:
        // `rmsnorm_unweighted` forbids aliasing its input with its output, and the raw value
        // is not wanted afterwards.
        Tensor raw = workspace.alloc(DType::BF16, {kv_rows, columns});
        ops::linear(hidden, weights.value, raw, kTextPolicy, workspace, stream);
        apply_lora(weights.value, kValuePort, hidden, raw, stream);
        ops::rmsnorm_unweighted(raw.view({head_dim, heads * columns}), weights.rms_epsilon,
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
    ops::rmsnorm_add(projected, weights.post_attention_norm, weights.rms_epsilon,
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
    auto scope                 = workspace.scope();
    const std::int32_t columns = hidden.ne[1];
    const std::int32_t intermediate = weights.gate.n;
    Tensor gate       = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor up         = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    Tensor projected  = workspace.alloc(DType::BF16, {residual.ne[0], columns});
    
    // Gate and up are separate matrices here, where Llama and the Qwen families
    // fuse them and feed one `linear_swiglu`. That costs a launch and buys two
    // adaptable modules: `gate_proj` and `up_proj` have their own weights, so the
    // hook can add a delta to each instead of the fused targets' refusal.
    ops::linear(hidden, weights.gate, gate, kTextPolicy, workspace, stream);
    apply_lora(weights.gate, kGatePort, hidden, gate, stream);
    ops::linear(hidden, weights.up, up, kTextPolicy, workspace, stream);
    apply_lora(weights.up, kUpPort, hidden, up, stream);
    ops::gelu_mul(gate, up, kMlpActivation, activation, stream);
    ops::linear(activation, weights.down, projected, kTextPolicy, workspace, stream);
    // down reads the gated activation, which is exactly the input its adapter was
    // trained against.
    apply_lora(weights.down, kDownPort, activation, projected, stream);
    // The second half of the FFN sandwich: `post_feedforward_layernorm` over the MLP's
    // output, before it reaches the residual. No unit offset, for the reason
    // `attention_output_projection` gives.
    ops::rmsnorm_add(projected, weights.post_feedforward_norm, weights.rms_epsilon,
                     /*unit_offset*/ false, residual, stream);
    // The block's scalar is **not** applied here, unlike the dense target's. This family folds
    // its per-layer input in after the feed-forward and scales only then:
    //
    //     hidden = residual + post_feedforward_norm(mlp)
    //     hidden = hidden + post_per_layer_input_norm(...)     <- the epilogue
    //     hidden *= layer_scalar
    //
    // Scaling here would put the per-layer input outside the scale, which is a different
    // model: with a factor of 0.05 at layer 0 the fold would arrive twenty times too large.
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(const family::TextGeometry& geometry, WeightsProfile weights_profile,
                                                         family::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    family::validate_token_interval(first, last);
    return std::max(
        post_mixer_workspace_bytes(geometry, profile_qtype(weights_profile), first, last),
        post_mixer_workspace_bytes(geometry, QType::Q4_K, first, last));
}

// ---- The per-layer input ---------------------------------------------------
//
// Gemma 4's E-series mixes a per-token, per-layer vector into every block, between the
// feed-forward's residual add and the layer scalar:
//
//     residual += post_per_layer_input_norm(
//         per_layer_projection( gelu(per_layer_input_gate(residual)) * per_layer_input ))
//
// and the per-layer input itself is
//
//     ( post_norm(per_layer_model_projection(residual) * hidden^-0.5)
//       + embed_tokens_per_layer(ids) * sqrt(pli_dim) ) * 2^-0.5
//
// Every step runs on a contiguous plane, which is why the artifact stores the two model-wide
// tensors cut per layer: `embedding` and `input_projection` below are this layer's slices, so
// nothing here is strided and no new kernel was needed for any of it.

void Variant::layer_epilogue(const ModelView& model, int layer, const Tensor& ids,
                             const Tensor& embedded, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    const auto& weights = model.full_layers.at(static_cast<std::size_t>(layer))
                              .post_mixer.per_layer_input;
    if (weights.width <= 0 || weights.projection_norm == nullptr) {
        throw std::logic_error("gemma4_e: a layer holds no per-layer input");
    }
    const auto empty = [&](const Weight& w) { return w.n == 0 || w.k == 0; };
    if (empty(weights.input_projection) || empty(weights.gate) || empty(weights.projection) ||
        empty(weights.embedding)) {
        throw std::logic_error(
            "gemma4_e: layer " + std::to_string(layer) + " per-layer input is unbound (" +
            "embedding " + std::to_string(weights.embedding.n) + "x" +
            std::to_string(weights.embedding.k) + ", input_projection " +
            std::to_string(weights.input_projection.n) + "x" +
            std::to_string(weights.input_projection.k) + ", gate " +
            std::to_string(weights.gate.n) + "x" + std::to_string(weights.gate.k) +
            ", projection " + std::to_string(weights.projection.n) + "x" +
            std::to_string(weights.projection.k) + ")");
    }
    auto scope                 = workspace.scope();
    const std::int32_t columns = residual.ne[1];
    const std::int32_t width   = weights.width;
    const std::int32_t hidden  = residual.ne[0];

    // The projection half: the **embedded input** mixed into this layer's slice, scaled and
    // normed. `embedded`, not the running residual: the reference projects `inputs_embeds`
    // once for the whole model (`project_per_layer_inputs`), so every layer sees the same
    // source and only its own slice of the projection differs.
    Tensor projected = workspace.alloc(DType::BF16, {width, columns});
    ops::linear(embedded, weights.input_projection, projected, kTextPolicy, workspace, stream);
    ops::scale(projected, 1.0F / std::sqrt(static_cast<float>(hidden)), stream);
    Tensor normed = workspace.alloc(DType::BF16, {width, columns});
    ops::rmsnorm(projected, *weights.projection_norm, weights.rms_epsilon,
                 /*unit_offset*/ false, normed, stream);

    // The embedding half: this layer's slice of the second table, scaled by sqrt of its width.
    Tensor slice = workspace.alloc(DType::BF16, {width, columns});
    ops::embedding(ids, weights.embedding, slice, stream);
    ops::scale(slice, std::sqrt(static_cast<float>(width)), stream);

    // Their mean, in the reference's own form: add, then scale by 2^-0.5.
    ops::residual_add(normed, slice, stream);
    ops::scale(slice, 0.70710678118654752F, stream);

    // The gate. `gelu_mul` is exactly `gelu(a) * b`, so the activation and the multiply by the
    // per-layer input are one launch.
    Tensor gate = workspace.alloc(DType::BF16, {width, columns});
    ops::linear(residual, weights.gate, gate, kTextPolicy, workspace, stream);
    apply_lora(weights.gate, kPerLayerGatePort, residual, gate, stream);
    Tensor gated = workspace.alloc(DType::BF16, {width, columns});
    ops::gelu_mul(gate, slice, kMlpActivation, gated, stream);

    // Back to the model width, normalised onto the residual. No unit offset, as everywhere.
    Tensor projected_out = workspace.alloc(DType::BF16, {hidden, columns});
    ops::linear(gated, weights.projection, projected_out, kTextPolicy, workspace, stream);
    apply_lora(weights.projection, kPerLayerProjectionPort, gated, projected_out, stream);
    ops::rmsnorm_add(projected_out, weights.norm, weights.rms_epsilon,
                     /*unit_offset*/ false, residual, stream);

    // And now the block's scalar, over everything it has accumulated including the fold above.
    // Measured on the real 12B the factor runs from 0.0053 to 0.918 and is never 1.
    ops::scale(residual, model.full_layers.at(static_cast<std::size_t>(layer))
                             .post_mixer.layer_scalar_value,
               stream);
}

std::size_t layer_epilogue_workspace_bytes(const family::TextGeometry& geometry, QType qtype,
                                           std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    const std::int32_t width = geometry.per_layer_input_dim;
    // Five planes: the projection, its normed copy, the embedded slice, the gate and its
    // gated product, plus the output at the model width.
    (void)layout.alloc(DType::BF16, {width, last});
    (void)layout.alloc(DType::BF16, {width, last});
    (void)layout.alloc(DType::BF16, {width, last});
    (void)layout.alloc(DType::BF16, {width, last});
    (void)layout.alloc(DType::BF16, {width, last});
    (void)layout.alloc(DType::BF16, {geometry.hidden, last});
    account_linear(layout, qtype, width, geometry.hidden, first, last);
    account_linear(layout, qtype, geometry.hidden, width, first, last);
    return layout.peak_bytes(1);
}

std::size_t Variant::layer_epilogue_workspace_capacity_bytes(const family::TextGeometry& geometry,
                                                             std::int32_t first,
                                                             std::int32_t last) {
    family::validate_token_interval(first, last);
    // The same two routes the other three capacity functions hold room for. This one named
    // the W8 route alone, whose transient storage is *zero*, so a GGUF-served E-series booked
    // nothing at all for the epilogue's three GEMMs and then ran them on the GGML route --
    // the arena came up short by a constant, at startup, before the first token. A BF16
    // artifact hid it: its linears also use none of what the layout books for them, so the
    // slack that pays for it was always there.
    //
    // The family hands this leaf no `WeightsProfile`, unlike its siblings, which is why the
    // profile's own qtype is spelled out here rather than asked for.
    return std::max(layer_epilogue_workspace_bytes(geometry, QType::W8G32_F16S, first, last),
                    layer_epilogue_workspace_bytes(geometry, QType::Q4_K, first, last));
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

void Variant::debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    // Only the magic is this target's: 'G4PB'. Everything else -- which rounds are
    // captured, how the occurrence is counted, the header layout -- is the family's.
    family::debug_probe_dump(0x47345042, tag, tensor, layer_count, stream);
}

} // namespace sinfer::targets::gemma4_e_e::detail
