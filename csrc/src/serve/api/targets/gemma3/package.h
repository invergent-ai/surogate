#pragma once

#include "api/types.h"
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

/// The namespace is the *size*, not the family: `config.h` is generated as
/// `sinfer::targets::gemma3_270m` because its constants describe
/// google/gemma-3-270m-it and nothing else. The directory and the target key
/// stay `gemma3`, which is the architecture a second size would share.
namespace targets::gemma3_270m {

struct Package;

namespace detail {

struct Variant;

/// The one export profile a Gemma 3 artifact carries today. The family's other
/// targets also list NVFP4 and FP8 profiles; nothing quantises a Gemma 3
/// checkpoint that way yet, and a profile with no converter behind it is a
/// promise the loader cannot keep.
enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(gemma3_270m::Package);

} // namespace detail

struct Package {
    /// Both strings are the converter's, verbatim:
    /// `surogate/serve/convert/gemma3/inventory.py` declares `MODEL_ID`
    /// and `TARGET_KEY`, and the engine matches an artifact to a package by
    /// comparing them character for character.
    /// The checkpoints this target serves. One entry here, because this architecture
    /// ships as one model; the registry asks every package the same question.
    static constexpr std::array<std::string_view, 1> model_ids{"gemma3-270m"};
    static constexpr std::string_view model_id = model_ids[0];
    static constexpr std::string_view target_key = "gemma3";
    /// Longest context the weights were trained for; `max_context = 0` asks the engine to
    /// fit the largest context the device's free memory allows, up to this. A function, not a
    /// constant: `detail::Variant` is only forward-declared here.
    [[nodiscard]] static std::uint32_t maximum_context() noexcept;

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
                                                               const family::TextGeometry& geometry,
                                                               const family::VisionGeometry& vision_geometry = {});
    /// The dimensions this artifact declares, over this target's compiled config.
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::gemma3_270m
} // namespace sinfer
