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

/// The routed Gemma 4: `google/gemma-4-26B-A4B` and its instruction tune.
///
/// The attention is the dense target's, geometry for geometry. What separates them is that
/// **every** layer here runs a dense feed-forward *and* 128 routed experts over the same
/// input and sums them under three norms a dense layer does not have -- so the layer's
/// weights, its objects and its post-mixer leaf are a different set, which is what a target
/// is. The dense sizes and the E-series have their own; the converter refuses to build an
/// artifact for them against this one.
namespace targets::gemma4_moe {

struct Package;

namespace detail {

struct Variant;

/// The export profiles a Gemma 4 mixture artifact carries.
///
/// `GroupwiseInt` is the converted BF16 checkpoint: every matrix W8 (or as a GGUF stores it).
/// `RoutedNvfp4` is an NVFP4 export's: the routed experts in NVFP4 with their per-expert
/// second-level and activation scales (`surogate/serve/convert/gemma4_moe/exports/
/// routed_nvfp4.py`), and every other matrix as the BF16 checkpoint converts it. The experts
/// run on the vendored TensorRT-LLM runner (W4A4) from two tokens up and on the NVFP4 decode
/// kernels (W4A16) for one.
enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
    RoutedNvfp4,
};

/// Do these weights need the sm_120 block-scaled FP4 MMA? The routed-NVFP4 experts do: the
/// runner is pinned to 120a. See the qwen3_5_moe declaration for why this is not left to the
/// archive to refuse.
[[nodiscard]] constexpr bool weights_profile_needs_sm120(WeightsProfile profile) noexcept {
    switch (profile) {
    case WeightsProfile::GroupwiseInt:
        return false;
    case WeightsProfile::RoutedNvfp4:
        return true;
    }
    return false;
}

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(gemma4_moe::Package);

} // namespace detail

struct Package {
    /// Both strings are the converter's, verbatim:
    /// `surogate/serve/convert/gemma4_moe/inventory.py` declares `MODEL_ID` and `TARGET_KEY`,
    /// and the engine matches an artifact to a package by comparing them character for
    /// character.
    static constexpr std::array<std::string_view, 1> model_ids{"gemma4_moe"};
    static constexpr std::string_view model_id = model_ids[0];
    static constexpr std::string_view target_key = "gemma4_moe";
    /// Longest context the weights were trained for; `max_context = 0` asks the engine to
    /// fit the largest context the device's free memory allows, up to this. A function, not
    /// a constant: `detail::Variant` is only forward-declared here.
    [[nodiscard]] static std::uint32_t maximum_context() noexcept;

    /// The compute profile this target's linear leaves admit, and the reservation that
    /// follows from it.
    ///
    /// Every leaf here runs `LinearPolicy::A16Only` (`kTextPolicy` in `variant.cpp`), so no
    /// FP8 or Marlin plane is ever derived from the W8 weights. Left undeclared the registry
    /// assumes `AllowA4` and reserves **1.5x the resident W8 bytes** against planes that are
    /// never created -- which does not corrupt anything, it simply refuses to load on a card
    /// where the model fits. The 26B-A4B is 24 GB of W8: the phantom reservation is 36 GB, so
    /// the budget saturated to zero on an empty 32 GB card.
    ///
    /// Safe to state here precisely because this target has **no GGUF bridge**: the targets
    /// that do can meet K-quants and derive Marlin tiles for them, and their conservative
    /// default is left alone.
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
                                                               const family::TextGeometry& geometry,
                                                               const family::VisionGeometry& vision_geometry = {});
    /// The dimensions this artifact declares, over this target's compiled config.
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::gemma4_moe
} // namespace sinfer
