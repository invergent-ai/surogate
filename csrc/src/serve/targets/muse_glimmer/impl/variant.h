#pragma once
#include "targets/gemma3/impl/variant.h"

namespace sinfer::targets::muse_glimmer::detail {
struct Variant : gemma3_270m::detail::Variant {
    struct DFlashConfig { static constexpr bool supported = true; };
    static constexpr bool supports_dflash = true;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = 15;
    static std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
        std::uint32_t draft_window, std::uint32_t batch_size = 1);
    static constexpr bool attention_output_gate    = true;
    static constexpr bool norm_unit_offset         = false;
    static constexpr bool one_dimensional_rope     = true;
    static constexpr std::uint32_t maximum_context = 131072;

    static void embed_residual(const ModelView& model, const Tensor& ids, Tensor& residual,
                               WorkspaceArena& workspace, cudaStream_t stream);
    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     family::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream);
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            const FullAttentionProjectionWeights& weights,
                                            Tensor& residual, family::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream);
    static void post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                           family::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream);
};
} // namespace sinfer::targets::muse_glimmer::detail
