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

/// The namespace is the *architecture*, not a size, unlike `gemma3_270m`: this target
/// serves both dense Gemma 4 checkpoints -- the 12B and the 31B -- from one compiled
/// config and each artifact's declared geometry.
///
/// The E-series (per-layer input embeddings, cross-layer KV sharing) and the 26B-A4B
/// mixture are different architectures. They get their own targets; the converter refuses
/// to build an artifact for them against this one.
namespace targets::gemma4 {

struct Package;

namespace detail {

struct Variant;

/// The one export profile a Gemma 4 artifact carries today. Nothing quantises these
/// checkpoints any other way yet, and a profile with no converter behind it is a promise
/// the loader cannot keep.
enum class WeightsProfile : std::uint8_t {
    GroupwiseInt,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(gemma4::Package);

} // namespace detail

struct Package {
    /// Both strings are the converter's, verbatim:
    /// `surogate/serve/convert/gemma4/inventory.py` declares `MODEL_ID` and `TARGET_KEY`,
    /// and the engine matches an artifact to a package by comparing them character for
    /// character.
    ///
    /// One id for two checkpoints, which is the point of this target: the 12B and the 31B
    /// differ in their declared geometry, not in their architecture.
    static constexpr std::array<std::string_view, 1> model_ids{"gemma4"};
    static constexpr std::string_view model_id = model_ids[0];
    static constexpr std::string_view target_key = "gemma4";
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
                                                               const family::TextGeometry& geometry);
    /// The dimensions this artifact declares, over this target's compiled config.
    [[nodiscard]] static family::TextGeometry declared_geometry(const artifact::Reader& reader);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device);
};

} // namespace targets::gemma4
} // namespace sinfer
