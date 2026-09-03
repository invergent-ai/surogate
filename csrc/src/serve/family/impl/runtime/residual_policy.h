#pragma once

// The residual stream's width and the hooks that read or write it outside the Variant's
// projections. A single-stream residual (every Qwen3.5/3.6 target) uses the defaults below,
// which are the family's original ops; a Variant with several residual streams (hyper
// connections) supplies the members the `requires` clauses probe for.

#include <api/family/text_geometry.h>

#include "api/ops/embedding.h"
#include "api/ops/gated_rmsnorm.h"
#include "api/ops/rmsnorm.h"
#include "api/ops/scale.h"
#include "core/arena.h"
#include "core/ngram_ple_state.h"
#include "core/tensor.h"
#include "family/impl/runtime/prologue_columns.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

namespace sinfer::family::detail {

/// Width of the residual planes: `Config::residual` when the target declares one, else hidden.
template <class Config>
[[nodiscard]] constexpr int residual_width() {
    if constexpr (requires { Config::residual; }) {
        return Config::residual;
    } else {
        return Config::hidden;
    }
}

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
template <class Variant>
[[nodiscard]] constexpr bool attention_qk_norm() {
    if constexpr (requires { Variant::attention_qk_norm; }) {
        return Variant::attention_qk_norm;
    } else {
        return true;
    }
}

/// The value a bf16 buffer would hold for `x`, round-to-nearest-even.
constexpr float as_bf16(float x) {
    const std::uint32_t bits    = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t rounded = (bits + 0x7fffU + ((bits >> 16U) & 1U)) & 0xffff0000U;
    return std::bit_cast<float>(rounded);
}

/// The factor a model applies to its embedding lookup, zero for none.
///
/// Rounded to bf16 deliberately. Gemma downcasts the scalar to the weight dtype
/// before multiplying -- `embed_scale.to(self.weight.dtype)` in transformers'
/// modeling_gemma3.py, and vLLM does the same -- so applying the fp32 value
/// would be a different model: sqrt(640) is 25.2982 while bf16 holds 25.25, a
/// 0.19% difference that lands on every token of every prompt.
template <class Variant>
[[nodiscard]] constexpr float embedding_scale() {
    if constexpr (requires { Variant::TextConfig::embedding_scale; }) {
        return as_bf16(Variant::TextConfig::embedding_scale);
    } else {
        return 0.0F;
    }
}

/// The rope base a layer rotates at.
///
/// Gemma 3 rotates its windowed layers at a base 100x smaller than its global
/// ones, so a single `rope_theta` cannot describe the model. A config that says
/// nothing keeps the one base it always had, which is every target but that one.
template <class TextConfig>
[[nodiscard]] constexpr float layer_rope_theta(int layer, const family::TextGeometry& geometry) {
    if constexpr (requires { TextConfig::layer_rope_theta(layer); }) {
        // Two bases keyed on the schedule: compiled, because which layers are windowed is
        // the family's pattern. Their values are still the checkpoint's to state, which is
        // what a per-layer geometry would carry.
        (void)geometry;
        return TextConfig::layer_rope_theta(layer);
    } else {
        (void)layer;
        return geometry.rope_theta;
    }
}

/// The sliding window this layer attends over, in keys, or 0 for a layer that
/// sees its whole context.
///
/// The window is a property of a *layer*, not of a round: Gemma 3 alternates five
/// windowed layers to one global one, so no single value describes a forward pass.
/// A config that says nothing is unwindowed everywhere, which is every target but
/// that one. `layer` is the absolute layer index, the same index
/// `layer_rope_theta` takes -- the two must agree, because a windowed layer is
/// exactly the layer that rotates at the local base.
template <class TextConfig>
[[nodiscard]] constexpr std::int32_t layer_sliding_window(int layer,
                                                          const family::TextGeometry& geometry) {
    if constexpr (requires {
                      TextConfig::sliding_window;
                      TextConfig::is_windowed_attention(layer);
                  }) {
        // Which layers are windowed is the family's pattern and stays compiled; how wide the
        // window is belongs to the checkpoint, so it comes from the geometry.
        return TextConfig::is_windowed_attention(layer) ? geometry.sliding_window : 0;
    } else {
        (void)layer;
        (void)geometry;
        return 0;
    }
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

/// Debug probe for parity work: a variant may observe intermediate tensors of the family's
/// layer loop by tag (the default is a no-op that compiles away).
template <class Variant>
inline void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    if constexpr (requires { Variant::debug_probe(tag, tensor, stream); }) {
        Variant::debug_probe(tag, tensor, stream);
    }
}

/// Device memory reserved per decode lane for the ordinary (non-speculative) CUDA graphs:
/// the family's 12 MiB unless the variant declares a larger footprint.
template <class Variant>
[[nodiscard]] constexpr std::size_t ordinary_graph_allowance_per_lane_bytes() {
    if constexpr (requires { Variant::ordinary_graph_allowance_per_lane_bytes; }) {
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
    [[nodiscard]] static std::optional<NgramPleStatePoolSpec> ple_state_spec(std::int32_t slots) {
        if constexpr (prologue) {
            return Variant::ple_state_spec(slots);
        } else {
            (void)slots;
            return std::nullopt;
        }
    }

    [[nodiscard]] static std::size_t layer_prologue_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last) {
        if constexpr (prologue) {
            return Variant::layer_prologue_workspace_capacity_bytes(first, last);
        } else {
            (void)first; (void)last;
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
            if constexpr (embedding_scale<Variant>() != 0.0F) {
                ops::scale(residual, embedding_scale<Variant>(), stream);
            }
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
