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

namespace targets::qwen3_moe {

struct Package;

namespace detail {

struct Variant;

/// The one export profile a Qwen3-MoE artifact carries today. A profile with no converter
/// behind it is a promise the loader cannot keep, so the NVFP4 and FP8 profiles the hybrid MoE
/// target lists are absent here until something writes one.
enum class WeightsProfile : std::uint8_t {
    /// Group-wise integer throughout, which a converted checkpoint stores; a GGUF served
    /// natively keeps its own K-quants, and the kernels dispatch on the weight's qtype.
    GroupwiseInt,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(qwen3_moe::Package);

} // namespace detail

struct Package {
    /// Every checkpoint this architecture serves. One graph at every size -- classic Qwen3
    /// attention over a routed mixture with no always-on expert -- and the artifact states its
    /// dimensions, so the target is the architecture rather than one of its sizes.
    static constexpr std::array<std::string_view, 1> model_ids{"qwen3-moe"};
    static constexpr std::string_view model_id = model_ids[0];
    static constexpr std::string_view target_key = "qwen3_moe";
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
                                                               const family::TextGeometry& geometry);
    /// The dimensions this artifact declares, over this target's compiled config.
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::qwen3_moe
} // namespace sinfer
