#pragma once

#include "api/family/text_geometry.h"
#include "api/family/vision.h"
#include "artifact/typed_binding.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::family {

struct Qwen3VlVisionPlan {
    struct DeepstackPlan {
        std::int32_t layer;
        artifact::LinearBinding fc1, fc2;
        artifact::ObjectHandle fc1_bias, fc2_bias, norm_weight, norm_bias;
    };
    family::VisionGeometry vision_geometry;
    family::VisionBackbonePlan vision_backbone;
    family::VisionMergerInputPlan vision_merger_input;
    family::VisionMergerNormPlan vision_merger_norm;
    artifact::LinearBinding vision_merger_output;
    artifact::ObjectHandle vision_merger_output_bias;
    std::vector<DeepstackPlan> deepstack;
};

inline void bind_qwen3_vl_vision(artifact::Binder& binder, Qwen3VlVisionPlan& out,
                                const TextGeometry& geometry, bool enabled, bool multimodal) {
    using artifact::NumericFormat;
    const bool has_vision = binder.has("vision/patch_embedding");
    const bool vl = multimodal;
    if (has_vision != vl || (enabled && !has_vision)) {
        throw std::runtime_error("Qwen3-VL requires its vision encoder; Qwen3 is text-only");
    }
    if (vl) {
        out.vision_geometry = family::VisionGeometry::resolved(binder.reader().vision_geometry());
        const auto& v = out.vision_geometry;
        if (v.siglip2 || v.output_hidden != geometry.hidden ||
            v.deepstack_layers > geometry.layers || !geometry.mrope_temporal) {
            throw std::runtime_error("Qwen3-VL vision or MRoPE geometry disagrees with the text model");
        }
        const auto placement = enabled ? artifact::TensorPlacement::Device
                                              : artifact::TensorPlacement::ValidateOnly;
        out.vision_backbone = family::bind_vision_backbone(binder, placement, v);
        out.vision_merger_input = family::bind_vision_merger_input(binder, placement, v);
        out.vision_merger_norm = family::bind_vision_merger_norm(binder, placement, v);
        out.vision_merger_output = artifact::bind_linear(
            binder, "vision/merger/fc2", v.output_hidden, v.merger_hidden(), placement);
        const auto tensor = [&](const std::string& name, int width) {
            return artifact::bind_tensor(binder, name, NumericFormat::BF16, {width}, placement);
        };
        out.vision_merger_output_bias = tensor("vision/merger/fc2_bias", v.output_hidden);
        for (int layer = 0; layer < v.layers; ++layer) {
            const auto prefix = "vision/layers/" + std::to_string(layer) + "/deepstack/";
            if (!binder.has(prefix + "fc1")) { continue; }
            out.deepstack.push_back(Qwen3VlVisionPlan::DeepstackPlan{
                .layer = layer,
                .fc1 = artifact::bind_linear(binder, prefix + "fc1", v.merger_hidden(), v.merger_hidden(), placement),
                .fc2 = artifact::bind_linear(binder, prefix + "fc2", v.output_hidden, v.merger_hidden(), placement),
                .fc1_bias = tensor(prefix + "fc1_bias", v.merger_hidden()),
                .fc2_bias = tensor(prefix + "fc2_bias", v.output_hidden),
                .norm_weight = tensor(prefix + "norm/weight", v.merger_hidden()),
                .norm_bias = tensor(prefix + "norm/bias", v.merger_hidden()),
            });
        }
        if (out.deepstack.size() != static_cast<std::size_t>(v.deepstack_layers)) {
            throw std::runtime_error("Qwen3-VL deepstack objects disagree with the declared count");
        }
    }
}

inline VisionWeights materialize_qwen3_vl_vision(const artifact::MaterializedArtifact& backing,
                                                 const Qwen3VlVisionPlan& plan) {
    using artifact::NumericFormat;
    const auto& v = plan.vision_geometry;
    family::VisionWeights vision;
    vision.common = family::materialize_vision_common(backing, plan.vision_backbone,
        plan.vision_merger_input, plan.vision_merger_norm, v);
    const auto tensor = [&](artifact::ObjectHandle handle, int width) {
        return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, {width});
    };
    vision.merger_fc2 = artifact::materialized_linear(backing, plan.vision_merger_output,
                                                     v.output_hidden, v.merger_hidden());
    vision.merger_fc2_bias = tensor(plan.vision_merger_output_bias, v.output_hidden);
    for (const auto& merger : plan.deepstack) {
        vision.deepstack.push_back(family::VisionWeights::DeepstackMerger{
            .layer = merger.layer,
            .fc1 = artifact::materialized_linear(backing, merger.fc1, v.merger_hidden(), v.merger_hidden()),
            .fc2 = artifact::materialized_linear(backing, merger.fc2, v.output_hidden, v.merger_hidden()),
            .fc1_bias = tensor(merger.fc1_bias, v.merger_hidden()),
            .fc2_bias = tensor(merger.fc2_bias, v.output_hidden),
            .norm_weight = tensor(merger.norm_weight, v.merger_hidden()),
            .norm_bias = tensor(merger.norm_bias, v.merger_hidden()),
        });
    }
    return vision;
}

} // namespace sinfer::family
