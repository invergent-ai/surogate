#pragma once

// Inkling-2-Mini (`inkling2_loop`): the looped target (ENGINE-INKLING lane e1). The engine serves it as a
// 48-"layer" model -- one layer per VISIT (prelude 0-5, the core 6-17 three times, coda 18-23) -- whose KV
// planes, routed-expert capture rows and pipeline split are per visit, over 24 physical weight sets.
// Design: rl/engine-inkling-2026-11-21/lanes/e1/DESIGN-target.md.

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

namespace targets::inkling2 {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    /// The artifact's BF16 attention projections: the faithful oracle path (DESIGN.md 9.0 / 10).
    Bf16Attention,
    /// One FP8 variant of the attention projections (owner ruling 06:58: the default once DESIGN 10.1
    /// has chosen the scheme); experts NVFP4 / MXFP8 either way.
    Fp8Attention,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(inkling2::Package);

} // namespace detail

struct Package {
    static constexpr std::array<std::string_view, 1> model_ids{"inkling-2-mini"};
    static constexpr std::string_view model_id   = model_ids[0];
    static constexpr std::string_view target_key = "inkling2";
    /// BF16 attention projections (A16) only until the FP8 route lands; the experts are lane d's op.
    static constexpr ops::LinearPolicy linear_policy = ops::LinearPolicy::A16Only;

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
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model, const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile,
                                                               const family::TextGeometry& geometry,
                                                               const family::VisionGeometry& vision_geometry = {});
    /// The SERVED geometry: 48 visits over the artifact's 24 physical layers (DESIGN-target.md 0.2).
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::inkling2
} // namespace sinfer
