#pragma once

#include "api/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <api/family/frontend.h>
#include <api/family/target_package.h>
#include <api/family/runtime.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace sinfer {

struct DeviceContext;

namespace artifact {
class Binder;
class MaterializedArtifact;
struct ArtifactIdentity;
struct MaterializationPlan;
} // namespace artifact

namespace targets::qwen3_6_35b_a3b {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
    // Routed experts NVFP4, everything else as GroupwiseInt has it. The routed weights are the
    // only ones the 35B still reads from a GGUF-derived groupwise export, and the format is worth
    // more on this hardware than any scheduling lever the board has measured.
    RoutedNvfp4,
    // A compressed-tensors export converted from its own directory: every text-core weight in
    // the format the export's config gave its module (read from the artifact, not asserted),
    // fused parents split per HF Linear, routed experts NVFP4 as RoutedNvfp4 has them. An
    // interim value: the profile still sizes workspaces; when formats reach the planner from
    // the load plan this enumerator goes with the rest.
    CompressedTensors,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(qwen3_6_35b_a3b::Package);

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.6-35b-a3b";
    static constexpr std::string_view target_key = "qwen3_6_35b_a3b";
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
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_6_35b_a3b
} // namespace sinfer
