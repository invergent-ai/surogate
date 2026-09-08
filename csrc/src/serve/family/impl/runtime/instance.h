#pragma once

#ifndef SINFER_FAMILY_VARIANT
#    error "SINFER_FAMILY_VARIANT must name the complete exact Variant"
#endif
#ifndef SINFER_FAMILY_RUNTIME_NS
#    error "SINFER_FAMILY_RUNTIME_NS must be a unique identifier for this instantiation"
#endif

#include <api/family/runtime.h>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

using Variant                        = SINFER_FAMILY_VARIANT;
using WeightsProfile                 = typename Variant::WeightsProfile;
using TextConfig                     = typename Variant::TextConfig;
using VisionConfig                   = typename Variant::VisionConfig;
using DFlashConfig                   = typename Variant::DFlashConfig;
using LoadedModelData                = typename Variant::ModelView;
using FullAttentionWeights           = typename LoadedModelData::FullLayer;
using GdnWeights                     = typename LoadedModelData::GdnLayer;
using MlpWeights                     = typename Variant::PostMixerWeights;
using MtpWeights                     = typename LoadedModelData::MtpLayer;
using DFlashWeights                  = typename LoadedModelData::DFlash;
using FullAttentionProjectionWeights = typename Variant::FullAttentionProjectionWeights;
using GdnProjectionWeights           = typename Variant::GdnProjectionWeights;
using VisionWeights                  = typename Variant::VisionWeights;
using GraphExecutionProfile          = typename Variant::GraphExecutionProfile;

using SequencePlan    = family::SequencePlan<Variant>;
using SequencePlanner = family::SequencePlanner<Variant>;
using RequestBasePlan = family::RequestBasePlan<Variant>;
using RequestPlan     = family::RequestPlan<Variant>;
using Program         = family::Program<Variant>;

inline constexpr std::uint32_t kPrefillChunkAlignment    = Variant::prefill_chunk_alignment;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = Variant::maximum_mtp_draft_tokens;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = Variant::maximum_dflash_draft_tokens;

inline std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    return Variant::ordinary_graph_profiles(capacity);
}

inline std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                             std::uint32_t draft_window) {
    return Variant::mtp_graph_profiles(capacity, draft_window);
}

inline std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                                std::uint32_t draft_window,
                                                                std::uint32_t batch_size) {
    return Variant::dflash_graph_profiles(capacity, draft_window, batch_size);
}

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS
