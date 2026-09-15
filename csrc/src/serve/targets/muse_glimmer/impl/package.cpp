#include "api/targets/muse_glimmer/package.h"
#include "targets/muse_glimmer/impl/variant.h"
#include "family/impl/runtime/target_support.h"

namespace sinfer::targets::muse_glimmer {
family::TextGeometry Package::declared_geometry(const artifact::Reader& reader) {
    auto g = family::TextGeometry::resolved_gemma3(reader.geometry(), reader.layer_types());
    if (!reader.dflash_geometry().empty()) {
        g.dflash = family::DFlashGeometry::resolved(reader.dflash_geometry(), reader.dflash_target_layers(),
            g.hidden, g.layers, g.output_rows);
    }
    return g;
}
ModelSamplingDefaults Package::sampling_defaults(std::string_view) {
    ModelSamplingDefaults out;
    out.thinking     = {.temperature = 1.0F, .top_k = 64, .top_p = 0.95F};
    out.non_thinking = out.thinking;
    return out;
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.architecture != target_key || identity.weights_id != "groupwise-int") {
        throw std::invalid_argument("unsupported Muse-Glimmer artifact identity");
    }
    return WeightsProfile::GroupwiseInt;
}

Package::SequencePlanner
Package::make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                               WeightsProfile profile, const family::TextGeometry& geometry,
                               const family::VisionGeometry& vision_geometry) {
    return family::make_sequence_planner<detail::Variant>(device, options, profile, geometry,
                                                          vision_geometry);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    return family::create_program<detail::Variant>(
        loaded_data(model).runtime, WeightsProfile::GroupwiseInt, std::move(plan), device);
}
} // namespace sinfer::targets::muse_glimmer
