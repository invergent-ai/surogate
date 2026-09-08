#pragma once

// The residual stream's width and the hooks that read or write it outside the Variant's
// projections. A single-stream residual (every Qwen3.5/3.6 target) uses the defaults below,
// which are the family's original ops; a Variant with several residual streams (hyper
// connections) supplies the members the `requires` clauses probe for.

#include <api/family/text_geometry.h>

#include "api/ops/embedding.h"
#include "api/ops/gated_delta_net.h"
#include "api/ops/gated_rmsnorm.h"
#include "api/ops/kimi_delta_net.h"
#include "api/ops/linear.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scale.h"
#include "api/ops/sigmoid_mul.h"
#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/ngram_ple_state.h"
#include "core/tensor.h"
#include "family/impl/runtime/prologue_columns.h"

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

namespace sinfer::family::detail {



template <class Variant>
[[nodiscard]] constexpr bool has_layer_prologue() {
    if constexpr (requires { Variant::has_layer_prologue; }) {
        return Variant::has_layer_prologue;
    } else {
        return false;
    }
}

/// Per-head query/key normalisation. The family normalises each attention head's
/// queries and keys before rope, and its checkpoints carry the weights for it.
/// Llama has no such weights at all, so a target whose artifact does not bind
/// them declares false and the step is skipped rather than applied to a plane
/// nothing wrote.
/// Whether the target mixes a per-token, per-layer input into every block.
template <class Variant>
[[nodiscard]] constexpr bool has_per_layer_inputs() {
    if constexpr (requires { Variant::has_per_layer_inputs; }) {
        return Variant::has_per_layer_inputs;
    } else {
        return false;
    }
}

template <class Variant>
[[nodiscard]] constexpr bool attention_qk_norm() {
    if constexpr (requires { Variant::attention_qk_norm; }) {
        return Variant::attention_qk_norm;
    } else {
        return true;
    }
}

/// Whether the target's draft head projects its attended heads to the model width itself.
///
/// The family's fixed draft tail applies one output weight to the attention output. A head
/// whose attention is absorbed -- the attended vector is a latent that has to be unfolded per
/// head before the output projection can read it -- supplies the leaf, with the capacity it
/// needs, and the tail hands it the payload and the output weight together. A target that says
/// nothing keeps the one linear, which is what every Qwen3.5-shaped head is.
template <class Variant>
[[nodiscard]] constexpr bool mtp_attention_output_is_leaf() {
    return requires(const Tensor& attention,
                    const typename Variant::MtpAttentionProjectionWeights& weights,
                    const Weight& output, Tensor& out, WorkspaceArena& workspace,
                    cudaStream_t stream) {
        Variant::mtp_attention_output_projection(attention, weights, output, out, workspace,
                                                 stream);
        {
            Variant::mtp_attention_output_projection_workspace_capacity_bytes(
                std::declval<const family::TextGeometry&>(), std::int32_t{1}, std::int32_t{1})
        } -> std::same_as<std::size_t>;
    };
}

/// The draft tail's output projection, through the target's leaf or the family's one linear.
/// A template over the Variant so the branch not taken is discarded: the runtime's own
/// members see a concrete Variant, where `if constexpr` would still have to compile both.
template <class Variant>
inline void mtp_attention_output_dispatch(
    const Tensor& attention, const typename Variant::MtpAttentionProjectionWeights& weights,
    const Weight& output, Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    if constexpr (mtp_attention_output_is_leaf<Variant>()) {
        Variant::mtp_attention_output_projection(attention, weights, output, out, workspace,
                                                 stream);
    } else {
        (void)weights;
        (void)workspace;
        ops::linear(attention, output, out, stream);
    }
}

/// The scratch that projection needs: the leaf's own, or none for the one linear.
template <class Variant>
[[nodiscard]] inline std::size_t
mtp_attention_output_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                    std::int32_t last) {
    if constexpr (mtp_attention_output_is_leaf<Variant>()) {
        return Variant::mtp_attention_output_projection_workspace_capacity_bytes(geometry, first,
                                                                                 last);
    } else {
        (void)geometry;
        (void)first;
        (void)last;
        return 0;
    }
}

/// Whether the target's NextN draft head is a trunk block.
///
/// A head whose block is an ordinary layer is run by the trunk's own mixer -- one program, not
/// two that drift -- and the target supplies only the fold that seeds the residual and the
/// collapse that reads it back. A target that says nothing keeps the fixed draft tail, which
/// is what every family with a Qwen3.5-shaped head uses.
template <class Variant>
[[nodiscard]] constexpr bool mtp_block_is_trunk_layer() {
    if constexpr (requires { Variant::mtp_block_is_trunk_layer; }) {
        return Variant::mtp_block_is_trunk_layer;
    } else {
        return false;
    }
}

/// The value a bf16 buffer would hold for `x`, round-to-nearest-even.
constexpr float as_bf16(float x) {
    const std::uint32_t bits    = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t rounded = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    return std::bit_cast<float>(rounded);
}

/// RoPE base and window width follow the checkpoint's explicit layer schedule.
[[nodiscard]] constexpr float layer_rope_theta(int layer, const family::TextGeometry& geometry) {
    return geometry.layer_is_windowed(layer) && geometry.sliding_rope_theta > 0.0F
               ? geometry.sliding_rope_theta : geometry.rope_theta;
}

[[nodiscard]] constexpr std::int32_t layer_sliding_window(
    int layer, const family::TextGeometry& geometry) {
    return geometry.layer_is_windowed(layer) ? geometry.sliding_window : 0;
}

/// RMSNorm weight convention. This family's checkpoints store zero-centred norm
/// weights, so the scale a kernel must apply is `1 + w`. A variant whose
/// checkpoint stores the scale directly -- classic Qwen3 does -- declares false,
/// and the difference is not subtle: applying the offset to a plain weight
/// multiplies every norm by roughly `1 + w` instead of `w`, which leaves a model
/// that still emits fluent text and means none of it.
template <class Variant>
[[nodiscard]] constexpr bool norm_unit_offset() {
    if constexpr (requires { Variant::norm_unit_offset; }) {
        return Variant::norm_unit_offset;
    } else {
        return true;
    }
}

/// Whether the attention rotates its queries and keys.
///
/// Every model the family had served does, so nothing declared it. GLM-5.3's latent attention
/// is the NoPE variant -- its checkpoint states `rope.dimension_count` 0 and carries no rotary
/// split at all -- and rotating it by a zero-width rotation is not what "no rotary" means; the
/// step is skipped. A target that says nothing keeps the rotation it always had, and a target
/// with an unset `rotary_dim` still reaches `ops::rope` and is refused there, which is the
/// difference between declaring this and testing the width.
template <class Variant>
[[nodiscard]] constexpr bool applies_rotary() {
    if constexpr (requires { Variant::applies_rotary; }) {
        return Variant::applies_rotary;
    } else {
        return true;
    }
}

/// Debug probe for parity work: a variant may observe intermediate tensors of the family's
/// layer loop by tag (the default is a no-op that compiles away).
template <class Variant>
inline void debug_probe(const char* tag, const Tensor& tensor, std::int32_t layer_count, cudaStream_t stream) {
    if constexpr (requires { Variant::debug_probe(tag, tensor, layer_count, stream); }) {
        Variant::debug_probe(tag, tensor, layer_count, stream);
    }
}

/// Device memory reserved per decode lane for the ordinary (non-speculative) CUDA graphs:
/// the family's 12 MiB unless the variant declares a larger footprint -- as a constant, or as
/// a function of the weights profile, since the graphs of one route (the native GGUF
/// K-quants) are not the graphs of another and the allowance is KV the other would have had.
template <class Variant>
[[nodiscard]] constexpr std::size_t
ordinary_graph_allowance_per_lane_bytes(typename Variant::WeightsProfile profile) {
    if constexpr (requires { Variant::ordinary_graph_allowance_per_lane_bytes(profile); }) {
        return Variant::ordinary_graph_allowance_per_lane_bytes(profile);
    } else if constexpr (requires { Variant::ordinary_graph_allowance_per_lane_bytes; }) {
        return Variant::ordinary_graph_allowance_per_lane_bytes;
    } else {
        return 12ULL * 1024ULL * 1024ULL;
    }
}

/// Whether the attention output is gated. Every hybrid target in the family fuses
/// an output gate beside the query rows of its attention projection and
/// multiplies the attention result by its logistic sigmoid, so that is the
/// default and no existing target declares anything. A plain GQA stack (Qwen3,
/// Llama, Gemma) writes no gate rows: it declares `attention_output_gate =
/// false` and the family skips the multiply rather than applying it to whatever
/// the unwritten gate plane happened to hold.
template <class Variant>
[[nodiscard]] constexpr bool attention_output_gate() {
    if constexpr (requires { Variant::attention_output_gate; }) {
        return Variant::attention_output_gate;
    } else {
        return true;
    }
}

template <class Variant>
void apply_attention_gate(const Tensor& gate, Tensor& attention, cudaStream_t stream) {
    if constexpr (requires { Variant::apply_attention_gate(gate, attention, stream); }) {
        Variant::apply_attention_gate(gate, attention, stream);
    } else {
        ops::sigmoid_mul(gate, attention, stream);
    }
}

/// Activation of the GDN output gate (`z`): SiLU unless the variant declares otherwise
/// (Qwen3.8-Flash-Next gates with the logistic sigmoid).
template <class Variant>
[[nodiscard]] constexpr ops::GatedRmsGate gdn_output_gate() {
    if constexpr (requires { Variant::gdn_output_gate; }) {
        return Variant::gdn_output_gate;
    } else {
        return ops::GatedRmsGate::Silu;
    }
}

/// Which mixer the non-attending layers run. The delta net unless the variant says otherwise,
/// which is what every target that predates a second mixer means by saying nothing.
template <class Variant>
[[nodiscard]] constexpr family::LinearMixer linear_mixer() {
    if constexpr (requires { Variant::linear_mixer; }) {
        return Variant::linear_mixer;
    } else {
        return family::LinearMixer::GatedDelta;
    }
}

/// The forget gate as the mixer's recurrence reads it: one value per value head, or one per key
/// channel of every head. It is the same buffer either way -- the control projection wrote as
/// many rows as the mixer asks for -- so only the view differs.
template <class Variant>
[[nodiscard]] inline Tensor linear_gate_view(const Tensor& gate, std::int32_t value_head_dim,
                                             std::int32_t value_heads, std::int32_t width) {
    if constexpr (family::linear_mixer_gate_is_per_channel(linear_mixer<Variant>())) {
        return gate.view({value_head_dim, value_heads, width});
    } else {
        (void)value_head_dim;
        return gate.view({value_heads, width});
    }
}

template <class Variant>
[[nodiscard]] inline Tensor linear_gate_view(const Tensor& gate, std::int32_t value_head_dim,
                                             std::int32_t value_heads, std::int32_t width,
                                             std::int32_t batch) {
    if constexpr (family::linear_mixer_gate_is_per_channel(linear_mixer<Variant>())) {
        return gate.view({value_head_dim, value_heads, width, batch});
    } else {
        (void)value_head_dim;
        return gate.view({value_heads, width, batch});
    }
}

/// The mixer's recurrence over one sequence, reading and writing its own state.
template <class Variant>
inline void linear_recurrence(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, WorkspaceArena& work,
                              Tensor& state, Tensor& out, cudaStream_t stream) {
    if constexpr (linear_mixer<Variant>() == family::LinearMixer::KimiDelta) {
        (void)work;
        ops::kimi_delta_net(q, k, v, g, beta, scale, /*normalize_qk=*/true, state, out, stream);
    } else {
        ops::gated_delta_net(q, k, v, g, beta, scale, /*normalize_qk=*/true, work, state, out,
                             stream);
    }
}

/// The mixer's recurrence over B independent lanes, checkpointing after every valid column.
/// This is the shape an ordinary decode round has, so a mixer without it cannot decode.
template <class Variant>
inline void linear_recurrence_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& states, const Tensor& valid_columns,
                                       const Tensor& initial_state_slots,
                                       const Tensor& snapshot_base_slots, Tensor& out,
                                       cudaStream_t stream) {
    if constexpr (linear_mixer<Variant>() == family::LinearMixer::KimiDelta) {
        ops::kimi_delta_net_snapshot(q, k, v, g, beta, scale, /*normalize_qk=*/true, states,
                                     valid_columns, initial_state_slots, snapshot_base_slots, out,
                                     stream);
    } else {
        ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, /*normalize_qk=*/true, states,
                                      valid_columns, initial_state_slots, snapshot_base_slots, out,
                                      stream);
    }
}

/// The mixer's recurrence over B lanes in its replay-record form: the state is read and never
/// written, and the raw key, value and gate of every valid column are recorded so a later fold
/// can re-derive the state from whichever prefix the round accepted. The two delta rules have
/// the form and differ in the gate they record; a short convolution keeps no recurrent state
/// to replay into, so a target running one reaches here only by having been given a draft head
/// it cannot verify, and saying so beats recording the wrong recurrence.
template <class Variant>
inline void linear_recurrence_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& g, const Tensor& beta, float scale,
                                     const Tensor& states, const Tensor& valid_columns,
                                     const Tensor& initial_state_slots,
                                     GdnReplayRecordLayer& records, Tensor& out,
                                     cudaStream_t stream) {
    if constexpr (linear_mixer<Variant>() == family::LinearMixer::KimiDelta) {
        ops::kimi_delta_net_replay_record(q, k, v, g, beta, scale, states, valid_columns,
                                          initial_state_slots, records.key, records.value,
                                          records.gate, records.beta, out, stream);
    } else if constexpr (linear_mixer<Variant>() == family::LinearMixer::GatedDelta) {
        ops::gated_delta_net_replay_record(q, k, v, g, beta, scale, states, valid_columns,
                                           initial_state_slots, records.key, records.value,
                                           records.gate, out, stream);
    } else {
        throw std::logic_error(
            "this target's linear mixer has no replay-record form, so it cannot verify a "
            "speculative round");
    }
}

template <class Variant>
struct ResidualHooks {
    using Model = typename Variant::ModelView;

    /// Whether the Variant runs something on the residual before a layer (an n-gram memory).
    static constexpr bool prologue = has_layer_prologue<Variant>();

    static void layer_prologue(const Model& model, int layer, Tensor& residual,
                               const PrologueColumns& columns, NgramPleStatePool* ple_state,
                               WorkspaceArena& work, cudaStream_t stream) {
        if constexpr (prologue) {
            Variant::layer_prologue(model, layer, residual, columns, ple_state, work, stream);
        } else {
            (void)model; (void)layer; (void)residual; (void)columns; (void)ple_state; (void)work;
            (void)stream;
        }
    }

    /// The prologue's per-slot state pool, planned next to the linear-attention pool.
    [[nodiscard]] static std::optional<NgramPleStatePoolSpec> ple_state_spec(const family::TextGeometry& geometry, std::int32_t slots) {
        if constexpr (prologue) {
            if (!geometry.ple_ngram) { return std::nullopt; }
            return Variant::ple_state_spec(geometry, slots);
        } else {
            (void)slots;
            return std::nullopt;
        }
    }

    [[nodiscard]] static std::size_t layer_prologue_workspace_capacity_bytes(const family::TextGeometry& geometry, std::int32_t first,
                                                                             std::int32_t last) {
        if constexpr (prologue) {
            return Variant::layer_prologue_workspace_capacity_bytes(geometry, first, last);
        } else {
            (void)first; (void)last;
            return 0;
        }
    }

    /// Whether the model mixes a per-token, per-layer input into every block.
    ///
    /// Gemma 4's E-series does, and it is the whole of what the "E" buys: a second embedding
    /// table, `vocab x layers * per_layer_dim`, whose slice for a layer is folded into that
    /// layer after its feed-forward. Nothing else here has one, so nothing else declares it
    /// and neither hook below emits any code.
    static constexpr bool per_layer_inputs = has_per_layer_inputs<Variant>();

    /// What a block does after its feed-forward has landed on the residual.
    ///
    /// Gemma 4's E-series folds this layer's per-layer input in here: a gate over the
    /// residual, gated by this token's slice of a second embedding table, projected back to
    /// the model width and normalised onto the residual. It needs the token ids, because the
    /// slice is an embedding lookup, and it needs nothing else the layer does not already
    /// hold -- the two stacked tensors are stored cut per layer, so this layer's objects are
    /// planes it can read directly.
    static void layer_epilogue(const Model& model, int layer, const Tensor& ids,
                               const Tensor& embedded, Tensor& residual, WorkspaceArena& work,
                               cudaStream_t stream) {
        if constexpr (per_layer_inputs) {
            Variant::layer_epilogue(model, layer, ids, embedded, residual, work, stream);
        } else {
            (void)model; (void)layer; (void)ids; (void)embedded; (void)residual; (void)work;
            (void)stream;
        }
    }

    [[nodiscard]] static std::size_t layer_epilogue_workspace_capacity_bytes(
        const family::TextGeometry& geometry, std::int32_t first, std::int32_t last) {
        if constexpr (per_layer_inputs) {
            return Variant::layer_epilogue_workspace_capacity_bytes(geometry, first, last);
        } else {
            (void)geometry; (void)first; (void)last;
            return 0;
        }
    }

    /// Token ids into the residual planes.
    static void embed(const Model& model, const Tensor& ids, Tensor& residual,
                      WorkspaceArena& work, cudaStream_t stream) {
        if constexpr (requires { Variant::embed_residual(model, ids, residual, work, stream); }) {
            Variant::embed_residual(model, ids, residual, work, stream);
        } else {
            (void)work;
            ops::embedding(ids, model.token_embedding, residual, stream);
            // Gemma casts its configured embedding factor to BF16 before multiplying.
            const float scale = as_bf16(model.geometry.embedding_scale);
            if (scale != 0.0F) { ops::scale(residual, scale, stream); }
        }
    }

    /// Residual planes into the final hidden state the heads consume.
    static void finish(const Model& model, const Tensor& residual, float eps, Tensor& hidden,
                       WorkspaceArena& work, cudaStream_t stream) {
        if constexpr (requires {
                          Variant::final_residual_mix(model, residual, hidden, work, stream);
                      }) {
            Variant::final_residual_mix(model, residual, hidden, work, stream);
        } else {
            (void)work;
            ops::rmsnorm(residual, model.final_norm, eps, norm_unit_offset<Variant>(), hidden,
                         stream);
        }
    }

    /// Residual planes into the attention block's input.
    /// The attention output back into the residual.
    ///
    /// Most models project and add. Gemma 3 normalises in between -- it carries
    /// four sandwich norms per layer where this family has slots for two, and
    /// `post_attention_layernorm` is the one with nowhere to go. A variant that
    /// needs the projection weights (to reach that norm) declares the longer
    /// form and gets it; everything else keeps the projection-and-add it had.
    static void attention_output(const Tensor& attention, const Weight& o_proj,
                                 const typename Variant::FullAttentionProjectionWeights& weights,
                                 Tensor& residual, TextPhase phase, WorkspaceArena& work,
                                 cudaStream_t stream) {
        if constexpr (requires {
                          Variant::attention_output_projection(attention, o_proj, weights, residual,
                                                               phase, work, stream);
                      }) {
            Variant::attention_output_projection(attention, o_proj, weights, residual, phase, work,
                                                 stream);
        } else {
            (void)weights;
            Variant::attention_output_projection(attention, o_proj, residual, phase, work, stream);
        }
    }

    static void attention_norm(const Tensor& residual, const Tensor& norm, float eps,
                               const typename Variant::FullAttentionProjectionWeights& weights,
                               Tensor& hidden, WorkspaceArena& work, cudaStream_t stream) {
        if constexpr (requires {
                          Variant::attention_norm(residual, weights, hidden, work, stream);
                      }) {
            Variant::attention_norm(residual, weights, hidden, work, stream);
        } else {
            (void)weights;
            (void)work;
            ops::rmsnorm(residual, norm, eps, norm_unit_offset<Variant>(), hidden, stream);
        }
    }

    /// Residual planes into the post-mixer (MLP / MoE) input.
    static void post_mixer_norm(const Tensor& residual, const Tensor& norm, float eps,
                                const typename Variant::PostMixerWeights& weights, Tensor& hidden,
                                WorkspaceArena& work, cudaStream_t stream) {
        if constexpr (requires {
                          Variant::post_mixer_norm(residual, weights, hidden, work, stream);
                      }) {
            Variant::post_mixer_norm(residual, weights, hidden, work, stream);
        } else {
            (void)weights;
            (void)work;
            ops::rmsnorm(residual, norm, eps, norm_unit_offset<Variant>(), hidden, stream);
        }
    }
};

} // namespace sinfer::family::detail
