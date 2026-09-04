#pragma once

#include "api/types.h"
#include "runtime/contract/types.h"
#include "runtime/contract/transient_region.h"
#include <api/family/frontend.h>
#include <api/family/text_geometry.h>
#include <api/family/target_package.h>
#include <api/family/runtime.h>

#include <cstdint>
#include <memory>
#include <array>
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

namespace targets::qwen3_5 {

struct Package;

namespace detail {

struct Variant;

enum class WeightsProfile : std::uint8_t {
    /// Group-wise integer throughout. The endpoints carry whatever the artifact says, so a
    /// K-quant GGUF and a Q6/W8 export both land here.
    GroupwiseInt,
    /// NVFP4 projections with a handful of layers left BF16, as the export wrote them.
    Nvfp4MixedBf16,
    /// Every projection NVFP4, no BF16 exceptions: what a ModelOpt export of this family is.
    Nvfp4Uniform,
    /// Only the MLP is NVFP4; the attention and GDN projections stay FP8, because that is
    /// what the export quantised.
    Nvfp4MlpOnly,
    /// Every language linear NVFP4, from the export that quantises them all.
    Nvfp4All,
};

using Frontend       = family::Frontend;
using PreparedPrompt = family::PreparedPrompt;
using OutputSession  = family::OutputSession;

SINFER_TARGET_LOAD_TYPES(qwen3_5::Package);

} // namespace detail

struct Package {
    /// Every checkpoint this architecture serves. The interleaved gated-delta / full-attention
    /// decoder is the same graph at every one of these sizes and across the model generations
    /// that share it, so the target is the architecture and the artifact states its dimensions.
    static constexpr std::array<std::string_view, 5> model_ids{
        "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.6-27b", "qwen3.8-27b"};
    static constexpr std::string_view model_id   = model_ids[0];
    static constexpr std::string_view target_key = "qwen3_5";
    /// The 3.8 export quantises differently from the 3.6 one at the same dimensions, so the
    /// two are told apart by model id where the weights profile is chosen.
    static constexpr std::string_view qwen3_8_model_id = "qwen3.8-27b";
    /// What a run reports itself as. One target serves the family; a reader still wants to see
    /// which model ran, so the label follows the checkpoint rather than the folder.
    [[nodiscard]] static std::string_view target_key_for(std::string_view model) noexcept;
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

} // namespace targets::qwen3_5
} // namespace sinfer
