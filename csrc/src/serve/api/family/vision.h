#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "core/tensor.h"

#include <api/family/vision_geometry.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::family {

// The vision tower's geometry is per-model, not per-family: the 0.8B ships 12
// layers of 768, the 2B and 4B 24 of 1024, the 27B/35B and Flash-Next 27 of 1152.
// These values are the last of those, kept as the default so every target that
// inherits them today is unaffected; a target with its own tower overrides them,
// and either way the numbers reach the binder as a `VisionGeometry` value, which
// an artifact may then declare its way out of.
struct VisionBackboneConfig {
    static constexpr int layers              = 27;
    static constexpr int hidden              = 1152;
    static constexpr int intermediate        = 4304;
    static constexpr int heads               = 16;
    static constexpr int head_dim            = hidden / heads;
    static constexpr int patch_dim           = 3 * 2 * 16 * 16;
    static constexpr int merge               = 2;
    static constexpr int merge_unit          = merge * merge;
    static constexpr int merger_hidden       = hidden * merge_unit;
    static constexpr int position_embeddings = 48 * 48;
    static constexpr int rotary_dim          = head_dim;
    static constexpr float rope_theta        = 10'000.0F;
    static constexpr float norm_epsilon      = 1.0e-6F;
};

struct VisionLayerPlan {
    artifact::LinearBinding qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::LinearBinding output;
    artifact::ObjectHandle output_bias;
    artifact::LinearBinding fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::LinearBinding fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

// Sized when the plan is bound, not by the type: a plan whose layer array is 27
// long cannot describe a 24-layer tower, and the count is the geometry's to give.
struct VisionBackbonePlan {
    artifact::LinearBinding patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::vector<VisionLayerPlan> layers;
};

// The plan and the weights no longer vary with the tower config, only with the
// geometry bound into them. The aliases stay so a target that names either type
// through its own `VisionConfig` still compiles.
template <class Config>
using VisionBackbonePlanFor = VisionBackbonePlan;

struct VisionMergerInputPlan {
    artifact::LinearBinding fc1;
    artifact::ObjectHandle fc1_bias;
};

struct VisionMergerNormPlan {
    artifact::ObjectHandle weight;
    artifact::ObjectHandle bias;
};

struct VisionLayerWeights {
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
};

struct VisionCommonWeights {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::vector<VisionLayerWeights> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
    Tensor post_norm_weight;
    Tensor post_norm_bias;
};

template <class Config>
using VisionCommonWeightsFor = VisionCommonWeights;

struct VisionWeights {
    struct DeepstackMerger {
        std::int32_t layer;
        Weight fc1, fc2;
        Tensor fc1_bias, fc2_bias, norm_weight, norm_bias;
    };
    VisionCommonWeights common;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
    std::vector<DeepstackMerger> deepstack;
};

template <class Config>
using VisionWeightsFor = VisionWeights;

// Defined here rather than in a translation unit so a target can bind its own
// tower without a second copy of the shape contract. Every shape below comes
// from `geometry`.
inline VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder,
                                               artifact::TensorPlacement placement,
                                               const VisionGeometry& geometry) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlan out;
    out.patch_embedding = artifact::bind_linear(binder, "vision/patch_embedding", geometry.hidden, geometry.patch_dim, placement);
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {geometry.hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {geometry.position_embeddings, geometry.hidden});

    out.layers.resize(static_cast<std::size_t>(geometry.layers));
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = artifact::bind_linear(binder, prefix + "attention/qkv", 3 * geometry.hidden, geometry.hidden, placement);
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * geometry.hidden});
        target.output            = artifact::bind_linear(binder, prefix + "attention/output", geometry.hidden, geometry.hidden, placement);
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {geometry.hidden});
        target.fc1               = artifact::bind_linear(binder, prefix + "mlp/fc1", geometry.intermediate, geometry.hidden, placement);
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {geometry.intermediate});
        target.fc2               = artifact::bind_linear(binder, prefix + "mlp/fc2", geometry.hidden, geometry.intermediate, placement);
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {geometry.hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {geometry.hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {geometry.hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {geometry.hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {geometry.hidden});
    }
    return out;
}

inline VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                                      artifact::TensorPlacement placement,
                                                      const VisionGeometry& geometry) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    // The merger's input is one visual token's worth of patch vectors laid end to
    // end, so both extents are `merger_hidden` -- which equals the tower's MLP
    // width on a 1024-wide tower and has nothing to do with it.
    return VisionMergerInputPlan{
        .fc1      = artifact::bind_linear(binder, "vision/merger/fc1", geometry.merger_hidden(), geometry.merger_hidden(), placement),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {geometry.merger_hidden()}),
    };
}

inline VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                                    artifact::TensorPlacement placement,
                                                    const VisionGeometry& geometry) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    // The norm runs per patch, before the merge reshapes four of them into a token,
    // so it is the tower's hidden width and not the merger's.
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {geometry.hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {geometry.hidden}),
    };
}

inline VisionCommonWeights materialize_vision_backbone(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionGeometry& geometry) {
    using artifact::NumericFormat;

    VisionCommonWeights out;
    out.patch_embedding = artifact::materialized_linear(
        materialized, backbone.patch_embedding, geometry.hidden,
        geometry.patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {geometry.hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {geometry.hidden, geometry.position_embeddings});

    // The bound plan is the authority on how many layers there are: taking the count
    // from the geometry a second time is one more way for the two to disagree.
    out.layers.resize(backbone.layers.size());
    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv                    = artifact::materialized_linear(
        materialized, source.qkv, 3 * geometry.hidden,
            geometry.hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * geometry.hidden});
        target.output = artifact::materialized_linear(
        materialized, source.output, geometry.hidden,
            geometry.hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {geometry.hidden});
        target.fc1 = artifact::materialized_linear(
        materialized, source.fc1, geometry.intermediate,
            geometry.hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {geometry.intermediate});
        target.fc2 = artifact::materialized_linear(
        materialized, source.fc2, geometry.hidden,
            geometry.intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {geometry.hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {geometry.hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {geometry.hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {geometry.hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {geometry.hidden});
    }

    return out;
}

inline VisionCommonWeights materialize_vision_common(
    const artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm,
    const VisionGeometry& geometry) {
    using artifact::NumericFormat;
    auto out = materialize_vision_backbone(materialized, backbone, geometry);
    out.merger_fc1 = artifact::materialized_linear(
        materialized, merger_input.fc1, geometry.merger_hidden(),
        geometry.merger_hidden());
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {geometry.merger_hidden()});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {geometry.hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {geometry.hidden});
    return out;
}

} // namespace sinfer::family
