#pragma once

// The residual stream's width and the hooks that read or write it outside the Variant's
// projections. A single-stream residual (every Qwen3.5/3.6 target) uses the defaults below,
// which are the family's original ops; a Variant with several residual streams (hyper
// connections) supplies the members the `requires` clauses probe for.

#include "api/ops/embedding.h"
#include "api/ops/gated_rmsnorm.h"
#include "api/ops/rmsnorm.h"
#include "core/arena.h"
#include "core/ngram_ple_state.h"
#include "core/tensor.h"
#include "targets/qwen3_6/impl/runtime/prologue_columns.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_6::detail {

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

/// Debug probe for parity work: a variant may observe intermediate tensors of the family's
/// layer loop by tag (the default is a no-op that compiles away).
template <class Variant>
inline void debug_probe(const char* tag, const Tensor& tensor, cudaStream_t stream) {
    if constexpr (requires { Variant::debug_probe(tag, tensor, stream); }) {
        Variant::debug_probe(tag, tensor, stream);
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
            ops::rmsnorm(residual, model.final_norm, eps, true, hidden, stream);
        }
    }

    /// Residual planes into the attention block's input.
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
            ops::rmsnorm(residual, norm, eps, true, hidden, stream);
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
            ops::rmsnorm(residual, norm, eps, true, hidden, stream);
        }
    }
};

} // namespace ninfer::targets::qwen3_6::detail
