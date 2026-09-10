#pragma once

#include "api/family/model_view.h"
#include "api/ops/linear.h"
#include "api/family/startup_features.h"
#include "api/family/text_geometry.h"
#include "artifact/typed_binding.h"

namespace sinfer::family {

struct DFlashLayerPlan {
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle attention_output;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct DFlashPlan {
    artifact::ObjectHandle feature_projection;
    artifact::ObjectHandle context_norm;
    std::vector<DFlashLayerPlan> layers;
    artifact::ObjectHandle final_norm;
};

inline DFlashPlan bind_dflash(artifact::Binder& binder, const TextGeometry& g,
                              StartupFeatures features) {
    using artifact::NumericFormat;
    DFlashPlan out;
    // The DFlash drafter is a separate checkpoint the converter may not have
    // had; such artifacts omit every dflash/* object. Probe the family once
    // and turn a missing drafter into a startup error only when DFlash was
    // actually requested, rather than a missing-object failure on load.
    const bool has_dflash = g.dflash.layers > 0;
    if (has_dflash != binder.has("dflash/feature_projection")) {
        throw artifact::ArtifactError("DFlash metadata disagrees with stored objects");
    }
    if (!has_dflash && features.dflash()) {
        throw std::runtime_error("artifact has no DFlash drafter (converted without "
                                 "--dflash-model); run without --spec dflash");
    }
    const artifact::TensorPlacement dflash_placement =
        features.dflash() ? artifact::TensorPlacement::Device
                          : artifact::TensorPlacement::ValidateOnly;
    const auto bind_dflash = [&](std::string_view name, NumericFormat format,
                                 std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, dflash_placement);
    };
    if (has_dflash) {
        const auto& d = g.dflash;
        out.layers.resize(d.layers);
        out.feature_projection = bind_dflash("dflash/feature_projection", NumericFormat::W8G32_F16S,
                                             {d.hidden, d.feature_rows});
        out.context_norm = bind_dflash("dflash/context_norm", NumericFormat::BF16, {g.hidden});
        for (std::size_t layer = 0; layer < d.layers; ++layer) {
            DFlashLayerPlan& target  = out.layers[layer];
            const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
            target.input_norm = bind_dflash(prefix + "input_norm", NumericFormat::BF16, {g.hidden});
            target.query_key_value =
                bind_dflash(prefix + "attention/query_key_value", NumericFormat::W8G32_F16S,
                            {d.query_size() + 2 * d.kv_size(), g.hidden});
            target.query_norm =
                bind_dflash(prefix + "attention/query_norm", NumericFormat::BF16, {d.head_dim});
            target.key_norm =
                bind_dflash(prefix + "attention/key_norm", NumericFormat::BF16, {d.head_dim});
            target.attention_output = bind_dflash(
                prefix + "attention/output", NumericFormat::W8G32_F16S, {g.hidden, d.query_size()});
            target.post_attention_norm =
                bind_dflash(prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
            target.gate_up = bind_dflash(prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                         {2 * d.intermediate, d.hidden});
            target.down    = bind_dflash(prefix + "mlp/down", NumericFormat::W8G32_F16S,
                                         {d.hidden, d.intermediate});
        }
        out.final_norm = bind_dflash("dflash/final_norm", NumericFormat::BF16, {d.hidden});
    }

    return out;
}

inline DFlashWeights materialize_dflash(const DFlashPlan& plan,
                                        const artifact::MaterializedArtifact& backing,
                                        const TextGeometry& g) {
    using artifact::NumericFormat;
    const auto& d = g.dflash;
    DFlashWeights target;
    target.layers.resize(d.layers);
    target.feature_projection = artifact::materialized_weight(
        backing, plan.feature_projection, NumericFormat::W8G32_F16S, d.hidden, d.feature_rows);
    target.context_norm =
        artifact::materialized_tensor(backing, plan.context_norm, NumericFormat::BF16, {g.hidden});
    for (std::size_t layer = 0; layer < d.layers; ++layer) {
        const DFlashLayerPlan& source = plan.layers[layer];
        DFlashLayerWeights& weights   = target.layers[layer];
        weights.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                      NumericFormat::BF16, {g.hidden});
        weights.query_key_value       = artifact::materialized_weight(
            backing, source.query_key_value, NumericFormat::W8G32_F16S,
            d.query_size() + 2 * d.kv_size(), d.hidden);
        weights.context_key =
            ops::weight_rows(weights.query_key_value, d.query_size(), d.kv_size());
        weights.context_value =
            ops::weight_rows(weights.query_key_value, d.query_size() + d.kv_size(), d.kv_size());
        weights.query_norm       = artifact::materialized_tensor(backing, source.query_norm,
                                                                 NumericFormat::BF16, {d.head_dim});
        weights.key_norm         = artifact::materialized_tensor(backing, source.key_norm,
                                                                 NumericFormat::BF16, {d.head_dim});
        weights.attention_output = artifact::materialized_weight(
            backing, source.attention_output, NumericFormat::W8G32_F16S, d.hidden, d.query_size());
        weights.post_attention_norm = artifact::materialized_tensor(
            backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
        weights.gate_up = artifact::materialized_weight(
            backing, source.gate_up, NumericFormat::W8G32_F16S, 2 * d.intermediate, d.hidden);
        weights.down = artifact::materialized_weight(
            backing, source.down, NumericFormat::W8G32_F16S, d.hidden, d.intermediate);
    }
    target.final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {g.hidden});
    return target;
}

} // namespace sinfer::family
