#pragma once
#include "api/targets/gemma3/package.h"

namespace sinfer::targets::muse_glimmer {
namespace detail {
struct Variant;
}

// Muse shares the sandwich-normalized dense weight binding with Gemma 3.
// Its execution policy supplies gated attention, SwiGLU, and embedding RMSNorm.
struct Package : gemma3_270m::Package {
    static constexpr std::array<std::string_view, 1> model_ids{"muse-glimmer"};
    static constexpr std::string_view model_id   = model_ids[0];
    static constexpr std::string_view target_key = "muse_glimmer";
    using SequencePlanner                        = family::SequencePlanner<detail::Variant>;
    using SequencePlan                           = family::SequencePlan<detail::Variant>;
    using RequestBasePlan                        = family::RequestBasePlan<detail::Variant>;
    using RequestPlan                            = family::RequestPlan<detail::Variant>;
    using Program                                = family::Program<detail::Variant>;

    static std::uint32_t maximum_context() noexcept { return 131072; }

    static ModelSamplingDefaults sampling_defaults(std::string_view model);
    static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    static SequencePlanner
    make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                          WeightsProfile profile, const family::TextGeometry& geometry,
                          const family::VisionGeometry& vision_geometry = {});
    static std::unique_ptr<Program> create_program(const LoadedModel& model, SequencePlan&& plan,
                                                   DeviceContext& device);
};
} // namespace sinfer::targets::muse_glimmer
