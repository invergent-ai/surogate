#pragma once

// The residual stream's width and the hooks that read or write it outside the Variant's
// projections. A single-stream residual (every Qwen3.5/3.6 target) uses the defaults below,
// which are the family's original ops; a Variant with several residual streams (hyper
// connections) supplies the members the `requires` clauses probe for.

#include "api/ops/embedding.h"
#include "api/ops/rmsnorm.h"
#include "core/arena.h"
#include "core/tensor.h"

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
struct ResidualHooks {
    using Model = typename Variant::ModelView;

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
