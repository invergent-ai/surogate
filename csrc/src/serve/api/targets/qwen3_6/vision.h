#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {

// The vision tower's geometry is per-model, not per-family: the 0.8B ships 12
// layers of 768, the 2B and 4B 24 of 1024, the 27B/35B and Flash-Next 27 of 1152.
// These values are the last of those, kept as the default so every target that
// inherits them today is unaffected; a target with its own tower overrides them
// and instantiates the plan and weight types below on its own config.
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
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

// Sized by the config rather than by the family, because a plan whose layer array
// is 27 long cannot describe a 24-layer tower. `VisionBackbonePlan` keeps naming
// the default instantiation so existing targets are untouched.
template <class Config>
struct VisionBackbonePlanFor {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::array<VisionLayerPlan, Config::layers> layers;
};

using VisionBackbonePlan = VisionBackbonePlanFor<VisionBackboneConfig>;

struct VisionMergerInputPlan {
    artifact::ObjectHandle fc1;
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

template <class Config>
struct VisionCommonWeightsFor {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::array<VisionLayerWeights, Config::layers> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
};

using VisionCommonWeights = VisionCommonWeightsFor<VisionBackboneConfig>;

template <class Config>
struct VisionWeightsFor {
    VisionCommonWeightsFor<Config> common;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
};

using VisionWeights = VisionWeightsFor<VisionBackboneConfig>;

// Defined here rather than in a translation unit so a target can instantiate them
// on its own tower's geometry. Every shape below comes from `Config`.
template <class Config>
inline VisionBackbonePlanFor<Config> bind_vision_backbone(artifact::Binder& binder,
                                        artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };

    VisionBackbonePlanFor<Config> out;
    out.patch_embedding = bind("vision/patch_embedding", NumericFormat::Q6G64_F16S,
                               {Config::hidden, Config::patch_dim});
    out.patch_embedding_bias =
        bind("vision/patch_embedding_bias", NumericFormat::BF16, {Config::hidden});
    out.position_embedding =
        bind("vision/position_embedding", NumericFormat::BF16,
             {Config::position_embeddings, Config::hidden});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.qkv               = bind(prefix + "attention/qkv", NumericFormat::Q4G64_F16S,
                                        {3 * Config::hidden, Config::hidden});
        target.qkv_bias          = bind(prefix + "attention/qkv_bias", NumericFormat::BF16,
                                        {3 * Config::hidden});
        target.output            = bind(prefix + "attention/output", NumericFormat::Q5G64_F16S,
                                        {Config::hidden, Config::hidden});
        target.output_bias       = bind(prefix + "attention/output_bias", NumericFormat::BF16,
                                        {Config::hidden});
        target.fc1               = bind(prefix + "mlp/fc1", NumericFormat::Q4G64_F16S,
                                        {Config::intermediate, Config::hidden});
        target.fc1_bias          = bind(prefix + "mlp/fc1_bias", NumericFormat::BF16,
                                        {Config::intermediate});
        target.fc2               = bind(prefix + "mlp/fc2", NumericFormat::Q5G64_F16S,
                                        {Config::hidden, Config::intermediate});
        target.fc2_bias =
            bind(prefix + "mlp/fc2_bias", NumericFormat::BF16, {Config::hidden});
        target.norm1_weight =
            bind(prefix + "norm1/weight", NumericFormat::BF16, {Config::hidden});
        target.norm1_bias =
            bind(prefix + "norm1/bias", NumericFormat::BF16, {Config::hidden});
        target.norm2_weight =
            bind(prefix + "norm2/weight", NumericFormat::BF16, {Config::hidden});
        target.norm2_bias =
            bind(prefix + "norm2/bias", NumericFormat::BF16, {Config::hidden});
    }
    return out;
}

template <class Config>
inline VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder,
                                               artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerInputPlan{
        .fc1      = bind("vision/merger/fc1", NumericFormat::W8G32_F16S,
                         {Config::merger_hidden, Config::merger_hidden}),
        .fc1_bias = bind("vision/merger/fc1_bias", NumericFormat::BF16,
                         {Config::merger_hidden}),
    };
}

template <class Config>
inline VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder,
                                             artifact::TensorPlacement placement) {
    using artifact::NumericFormat;
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return VisionMergerNormPlan{
        .weight =
            bind("vision/merger/norm/weight", NumericFormat::BF16, {Config::hidden}),
        .bias =
            bind("vision/merger/norm/bias", NumericFormat::BF16, {Config::hidden}),
    };
}

template <class Config>
inline VisionCommonWeightsFor<Config> materialize_vision_common(
    const artifact::MaterializedArtifact& materialized,
    const VisionBackbonePlanFor<Config>& backbone,
    const VisionMergerInputPlan& merger_input,
    const VisionMergerNormPlan& merger_norm) {
    using artifact::NumericFormat;

    VisionCommonWeightsFor<Config> out;
    out.patch_embedding = artifact::materialized_weight(
        materialized, backbone.patch_embedding, NumericFormat::Q6G64_F16S,
        Config::hidden, Config::patch_dim);
    out.patch_embedding_bias =
        artifact::materialized_tensor(materialized, backbone.patch_embedding_bias,
                                      NumericFormat::BF16, {Config::hidden});
    out.position_embedding = artifact::materialized_tensor(
        materialized, backbone.position_embedding, NumericFormat::BF16,
        {Config::hidden, Config::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv                    = artifact::materialized_weight(
            materialized, source.qkv, NumericFormat::Q4G64_F16S, 3 * Config::hidden,
            Config::hidden);
        target.qkv_bias = artifact::materialized_tensor(
            materialized, source.qkv_bias, NumericFormat::BF16, {3 * Config::hidden});
        target.output = artifact::materialized_weight(
            materialized, source.output, NumericFormat::Q5G64_F16S, Config::hidden,
            Config::hidden);
        target.output_bias = artifact::materialized_tensor(
            materialized, source.output_bias, NumericFormat::BF16, {Config::hidden});
        target.fc1 = artifact::materialized_weight(
            materialized, source.fc1, NumericFormat::Q4G64_F16S, Config::intermediate,
            Config::hidden);
        target.fc1_bias =
            artifact::materialized_tensor(materialized, source.fc1_bias, NumericFormat::BF16,
                                          {Config::intermediate});
        target.fc2 = artifact::materialized_weight(
            materialized, source.fc2, NumericFormat::Q5G64_F16S, Config::hidden,
            Config::intermediate);
        target.fc2_bias = artifact::materialized_tensor(
            materialized, source.fc2_bias, NumericFormat::BF16, {Config::hidden});
        target.norm1_weight = artifact::materialized_tensor(
            materialized, source.norm1_weight, NumericFormat::BF16, {Config::hidden});
        target.norm1_bias = artifact::materialized_tensor(
            materialized, source.norm1_bias, NumericFormat::BF16, {Config::hidden});
        target.norm2_weight = artifact::materialized_tensor(
            materialized, source.norm2_weight, NumericFormat::BF16, {Config::hidden});
        target.norm2_bias = artifact::materialized_tensor(
            materialized, source.norm2_bias, NumericFormat::BF16, {Config::hidden});
    }

    out.merger_fc1 = artifact::materialized_weight(
        materialized, merger_input.fc1, NumericFormat::W8G32_F16S,
        Config::merger_hidden, Config::merger_hidden);
    out.merger_fc1_bias =
        artifact::materialized_tensor(materialized, merger_input.fc1_bias, NumericFormat::BF16,
                                      {Config::merger_hidden});
    out.merger_norm_weight = artifact::materialized_tensor(
        materialized, merger_norm.weight, NumericFormat::BF16, {Config::hidden});
    out.merger_norm_bias = artifact::materialized_tensor(
        materialized, merger_norm.bias, NumericFormat::BF16, {Config::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
