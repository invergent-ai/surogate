#pragma once

#include "api/types.h"
#include "api/ops/linear.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <api/family/frontend.h>
#include <api/family/text_geometry.h>
#include <api/family/target_package.h>
#include <api/family/runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string_view>

namespace sinfer {

struct DeviceContext;

namespace artifact {
class Reader;
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::glm5_next {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    /// Group-wise integer throughout, which is what this artifact's materialised objects use.
    /// Most of the weights are not materialised at all -- they are the GGUF's own K-quants,
    /// read where they lie -- and each carries the format the file stored it in.
    GroupwiseInt,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(glm5_next::Package);

} // namespace detail

struct Package {
    /// Every checkpoint this architecture serves. The stack is the same graph at every size --
    /// Kimi Delta Attention at three layers in four and multi-head latent attention at the
    /// rest, over a mixture, with a four-stream hyper-connected residual -- and the artifact
    /// states both its dimensions and which layers attend, so one target serves them all.
    static constexpr std::array<std::string_view, 1> model_ids{"glm5-next"};
    static constexpr std::string_view model_id   = model_ids[0];
    static constexpr std::string_view target_key = "glm5_next";
    /// Longest context the weights were trained for; `max_context = 0` asks the engine to fit
    /// the largest the device's free memory allows, up to this. A function, not a constant:
    /// `detail::Variant` is only forward-declared here.
    [[nodiscard]] static std::uint32_t maximum_context() noexcept;
    /// Every linear of this target runs an A16 profile: nothing is derived from its W8 weights,
    /// and the registry's reservation projection may take that at its word.
    static constexpr ops::LinearPolicy linear_policy = ops::LinearPolicy::A16Only;

    using WeightsProfile  = detail::WeightsProfile;
    using LoadPlan        = detail::LoadPlan;
    using LoadedModel     = detail::LoadedModel;
    using Frontend        = detail::Frontend;
    using PreparedPrompt  = detail::PreparedPrompt;
    using OutputSession   = detail::OutputSession;
    using SequencePlanner = family::SequencePlanner<detail::Variant>;
    using SequencePlan    = family::SequencePlan<detail::Variant>;
    using RequestBasePlan = family::RequestBasePlan<detail::Variant>;
    using RequestPlan     = family::RequestPlan<detail::Variant>;
    using Program         = family::Program<detail::Variant>;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile,
                                                               const family::TextGeometry& geometry);
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::glm5_next
} // namespace sinfer
