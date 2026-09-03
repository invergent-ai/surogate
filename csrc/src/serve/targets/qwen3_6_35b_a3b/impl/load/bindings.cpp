#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

/// Routed experts per MoE layer; the second-level NVFP4 scales are one array of this length.
constexpr std::uint64_t kRoutedExperts = 256;

/// The routed formats of one text layer. A groupwise-int artifact mixes Q4 gate/up with Q5 or Q6
/// down per layer (the export's own measured ranking); a routed-NVFP4 one is NVFP4 throughout,
/// which is why the per-layer choice disappears.
struct RoutedFormats {
    NumericFormat gate_up;
    NumericFormat down;
};

RoutedFormats routed_formats(WeightsProfile weights, std::size_t layer);

NumericFormat routed_down_format(std::size_t layer) {
    return layer == 34 || layer == 38 || layer == 39 ? NumericFormat::Q6G64_F16S
                                                     : NumericFormat::Q5G64_F16S;
}

RoutedFormats routed_formats(WeightsProfile weights, std::size_t layer) {
    if (weights == WeightsProfile::RoutedNvfp4 || weights == WeightsProfile::CompressedTensors) {
        return {NumericFormat::NVFP4, NumericFormat::NVFP4};
    }
    return {NumericFormat::Q4G64_F16S, routed_down_format(layer)};
}

/// One routed NVFP4 matrix as the MoE kernels read it.
///
/// `artifact::materialized_weight` refuses NVFP4 on purpose: the dense path pairs every NVFP4
/// weight with an activation divisor, and building one without that pairing would silently drop
/// a scale. The routed experts are w4a16 — the activations stay BF16 — and the format's second
/// level rides in a separate per-expert array, so this builds the Weight directly rather than
/// borrowing a contract that does not apply.
Weight routed_nvfp4_weight(const artifact::MaterializedArtifact& materialized,
                           artifact::ObjectHandle handle, std::int32_t rows,
                           std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = QType::NVFP4;
    out.group_size      = 16;
    out.ndim            = 2;
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = 16;
    out.layout          = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype     = DType::FP8_E4M3FN;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    // Neither divisor applies here, and the kernels do not read them; the converter writes 1.0
    // into the payload's word, which `bind_moe` checks so a checkpoint that means something by
    // it fails loudly instead of being ignored.
    out.weight_scale_divisor = 1.0F;
    out.input_scale_divisor  = 1.0F;
    return out;
}

/// The payload's per-tensor divisor word must be the identity: a routed artifact carries its
/// second level per expert, and a per-tensor value here would be dropped on the floor.
void require_identity_divisor(const artifact::Binder& binder, artifact::ObjectHandle handle,
                              std::string_view name, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const artifact::PayloadSpan payload = binder.payload(handle);
    if (payload.data.size() < geometry.weight_divisor_offset + sizeof(std::uint32_t)) {
        throw artifact::ArtifactError(std::string(name) + ": NVFP4 payload is short of its divisor");
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, payload.data.data() + geometry.weight_divisor_offset, sizeof(bits));
    if (std::bit_cast<float>(bits) != 1.0F) {
        throw artifact::ArtifactError(
            std::string(name) +
            ": routed NVFP4 must carry a per-tensor divisor of 1.0; the second level belongs in "
            "the per-expert scale object");
    }
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.qtype != QType::W8G32_F16S || block.layout != QuantLayout::RowSplit ||
        block.group != 32) {
        throw std::logic_error("invalid DFlash W8 row view");
    }
    const std::uint64_t code_row = static_cast<std::uint64_t>(block.padded_shape[1]);
    const std::uint64_t scale_row =
        static_cast<std::uint64_t>(block.padded_shape[1] / block.group) * sizeof(std::uint16_t);
    Weight out = block;
    out.qdata  = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, NumericFormat routed_gate_up,
                 NumericFormat routed_down, artifact::TensorPlacement placement) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    const auto bind_stored = [&](std::string_view name, std::int32_t rows, std::int32_t columns) {
        return artifact::bind_linear(binder, name, rows, columns, placement).object;
    };
    MoePlan plan{
        .router_shared_gate = bind(prefix + "router_shared_gate", NumericFormat::BF16, {257, 2048}),
        // The routed experts' format comes from the artifact: a GGUF serves them as its own
        // K-quant superblocks, a converted checkpoint as the groupwise-int or NVFP4 profile.
        // `routed_gate_up`/`routed_down` remain the profile's expectation for the NVFP4 extras
        // below, which only that profile carries.
        .routed_gate_up     = bind_stored(prefix + "routed_gate_up", 262144, 2048),
        .routed_down        = bind_stored(prefix + "routed_down", 524288, 512),
        .shared_gate_up = bind(prefix + "shared_gate_up", NumericFormat::W8G32_F16S, {1024, 2048}),
        .shared_down    = bind(prefix + "shared_down", NumericFormat::W8G32_F16S, {2048, 512}),
    };
    // NVFP4 carries a second-level scale per expert per projection; the row block holds gate and
    // up stacked, so gate/up needs two entries per expert and down one.
    if (routed_gate_up == NumericFormat::NVFP4) {
        plan.routed_gate_up_scale =
            bind(prefix + "routed_gate_up_scale", NumericFormat::FP32, {2 * kRoutedExperts});
        plan.routed_gate_up_act_scale =
            bind(prefix + "routed_gate_up_act_scale", NumericFormat::FP32, {kRoutedExperts});
        plan.routed_gate_up_alpha =
            bind(prefix + "routed_gate_up_alpha", NumericFormat::FP32, {kRoutedExperts});
        require_identity_divisor(binder, plan.routed_gate_up, prefix + "routed_gate_up", 262144,
                                 2048);
    }
    if (routed_down == NumericFormat::NVFP4) {
        plan.routed_down_scale =
            bind(prefix + "routed_down_scale", NumericFormat::FP32, {kRoutedExperts});
        plan.routed_down_act_scale =
            bind(prefix + "routed_down_act_scale", NumericFormat::FP32, {kRoutedExperts});
        plan.routed_down_alpha =
            bind(prefix + "routed_down_alpha", NumericFormat::FP32, {kRoutedExperts});
        require_identity_divisor(binder, plan.routed_down, prefix + "routed_down", 524288, 512);
    }
    return plan;
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          NumericFormat routed_gate_up, NumericFormat routed_down) {
    SparseMoePayload payload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router_shared_gate, NumericFormat::BF16, 257, 2048),
            .routed_gate_up =
                routed_gate_up == NumericFormat::NVFP4
                    ? routed_nvfp4_weight(materialized, plan.routed_gate_up, 262144, 2048)
                    : artifact::materialized_weight(materialized, plan.routed_gate_up,
                                                    routed_gate_up, 262144, 2048),
            .routed_down =
                routed_down == NumericFormat::NVFP4
                    ? routed_nvfp4_weight(materialized, plan.routed_down, 524288, 512)
                    : artifact::materialized_weight(materialized, plan.routed_down, routed_down,
                                                    524288, 512),
            .shared_gate_up = artifact::materialized_weight(materialized, plan.shared_gate_up,
                                                            NumericFormat::W8G32_F16S, 1024, 2048),
            .shared_down    = artifact::materialized_weight(materialized, plan.shared_down,
                                                            NumericFormat::W8G32_F16S, 2048, 512),
            .experts_per_token = ops::kSparseMoeQwen36Geometry.experts_per_token,
        }};
    if (plan.routed_gate_up_scale.has_value()) {
        payload.op.routed_gate_up_scale = static_cast<const float*>(
            artifact::materialized_tensor(materialized, *plan.routed_gate_up_scale,
                                          NumericFormat::FP32, {2 * kRoutedExperts})
                .data);
    }
    if (plan.routed_down_scale.has_value()) {
        payload.op.routed_down_scale =
            static_cast<const float*>(artifact::materialized_tensor(materialized,
                                                                    *plan.routed_down_scale,
                                                                    NumericFormat::FP32,
                                                                    {kRoutedExperts})
                                          .data);
    }
    const auto per_expert = [&](const std::optional<artifact::ObjectHandle>& handle) {
        return handle.has_value()
                   ? static_cast<const float*>(
                         artifact::materialized_tensor(materialized, *handle, NumericFormat::FP32,
                                                       {kRoutedExperts})
                             .data)
                   : nullptr;
    };
    payload.op.routed_gate_up_act_scale = per_expert(plan.routed_gate_up_act_scale);
    payload.op.routed_gate_up_alpha     = per_expert(plan.routed_gate_up_alpha);
    payload.op.routed_down_act_scale    = per_expert(plan.routed_down_act_scale);
    payload.op.routed_down_alpha        = per_expert(plan.routed_down_alpha);
    return payload;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab || seen[id]) {
            throw artifact::ArtifactError("invalid optimized draft-head token ids");
        }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, family::StartupFeatures features,
                               WeightsProfile weights) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    out.frontend        = family::bind_frontend_resources(binder);
    out.features        = features;
    out.weights         = weights;
    out.token_embedding = artifact::bind_linear(binder, "text/token_embedding", 248320, 2048);

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {2048});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            // One object per HF Linear when the artifact stores them so (a quantized export,
            // each constituent with its own global scale), the fused parent otherwise. The
            // format of each is read from the artifact, not asserted.
            if (binder.has(prefix + "attention/query")) {
                target.attention.split = SplitAttentionPlan{
                    .query = artifact::bind_linear(binder, prefix + "attention/query", 4096, 2048),
                    .key   = artifact::bind_linear(binder, prefix + "attention/key", 512, 2048),
                    .gate  = artifact::bind_linear(binder, prefix + "attention/gate", 4096, 2048),
                    .value = artifact::bind_linear(binder, prefix + "attention/value", 512, 2048),
                };
            } else {
                target.attention.query_key_gate_value = artifact::bind_linear(
                    binder, prefix + "attention/query_key_gate_value", 9216, 2048);
            }
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output =
                artifact::bind_linear(binder, prefix + "attention/output", 2048, 4096);
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {32});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {32});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_b_projection", NumericFormat::BF16, {64, 2048});
            if (binder.has(prefix + "gdn/query_key_value")) {
                target.gdn.split = SplitGdnInputPlan{
                    .query_key_value =
                        artifact::bind_linear(binder, prefix + "gdn/query_key_value", 8192, 2048),
                    .z = artifact::bind_linear(binder, prefix + "gdn/z", 4096, 2048),
                };
            } else {
                target.gdn.query_key_value_z =
                    artifact::bind_linear(binder, prefix + "gdn/query_key_value_z", 12288, 2048);
            }
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = artifact::bind_linear(binder, prefix + "gdn/output", 2048, 4096);
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {2048});
        const RoutedFormats routed = routed_formats(weights, layer);
        target.moe = bind_moe(binder, prefix + "moe/", routed.gate_up, routed.down,
                              artifact::TensorPlacement::Device);
    }

    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {2048});
    out.output_head = artifact::bind_linear(binder, "text/output_head", 248320, 2048);
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 2048}, proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {2048});
    out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {2048});
    out.mtp.input_norm     = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {2048});
    out.mtp.attention.query_key_gate_value = artifact::bind_linear(
        binder, "mtp/layer/attention/query_key_gate_value", 9216, 2048, mtp_placement);
    out.mtp.attention.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.attention.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.attention.output =
        artifact::bind_linear(binder, "mtp/layer/attention/output", 2048, 4096, mtp_placement);
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {2048});
    out.mtp.moe        = bind_moe(binder, "mtp/layer/moe/", NumericFormat::W8G32_F16S,
                                  NumericFormat::W8G32_F16S, mtp_placement);
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {2048});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = family::bind_vision_backbone<family::VisionBackboneConfig>(binder, vision_placement);
    out.vision_merger_input = family::bind_vision_merger_input<family::VisionBackboneConfig>(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {2048, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {2048}, vision_placement);
    out.vision_merger_norm = family::bind_vision_merger_norm<family::VisionBackboneConfig>(binder, vision_placement);

    // The DFlash drafter is a separate checkpoint the converter may not have
    // had; such artifacts omit every dflash/* object. Probe the family once
    // and turn a missing drafter into a startup error only when DFlash was
    // actually requested, rather than a missing-object failure on load.
    out.has_dflash = binder.has("dflash/feature_projection");
    if (!out.has_dflash && features.dflash()) {
        throw std::runtime_error(
            "qwen3.6-35b-a3b artifact has no DFlash drafter (converted without "
            "--dflash-model); run without --spec dflash");
    }
    const artifact::TensorPlacement dflash_placement =
        features.dflash() ? artifact::TensorPlacement::Device
                          : artifact::TensorPlacement::ValidateOnly;
    const auto bind_dflash = [&](std::string_view name, NumericFormat format,
                                 std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, dflash_placement);
    };
    if (out.has_dflash) {
    out.dflash.feature_projection =
        bind_dflash("dflash/feature_projection", NumericFormat::W8G32_F16S, {2048, 16384});
    out.dflash.context_norm = bind_dflash("dflash/context_norm", NumericFormat::BF16, {2048});
    for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
        DFlashLayerPlan& target  = out.dflash.layers[layer];
        const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_dflash(prefix + "input_norm", NumericFormat::BF16, {2048});
        target.query_key_value   = bind_dflash(prefix + "attention/query_key_value",
                                               NumericFormat::W8G32_F16S, {6144, 2048});
        target.query_norm =
            bind_dflash(prefix + "attention/query_norm", NumericFormat::BF16, {128});
        target.key_norm = bind_dflash(prefix + "attention/key_norm", NumericFormat::BF16, {128});
        target.attention_output =
            bind_dflash(prefix + "attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
        target.post_attention_norm =
            bind_dflash(prefix + "post_attention_norm", NumericFormat::BF16, {2048});
        target.gate_up =
            bind_dflash(prefix + "mlp/gate_up", NumericFormat::W8G32_F16S, {12288, 2048});
        target.down = bind_dflash(prefix + "mlp/down", NumericFormat::W8G32_F16S, {2048, 6144});
    }
    out.dflash.final_norm = bind_dflash("dflash/final_norm", NumericFormat::BF16, {2048});
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = family::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = artifact::materialized_linear(backing, plan.token_embedding, 248320, 2048);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {2048});
            if (source.attention.split) {
                const SplitAttentionPlan& split = *source.attention.split;
                target.projection.split         = SplitAttentionWeights{
                            .query = artifact::materialized_linear(backing, split.query, 4096, 2048),
                            .key   = artifact::materialized_linear(backing, split.key, 512, 2048),
                            .gate  = artifact::materialized_linear(backing, split.gate, 4096, 2048),
                            .value = artifact::materialized_linear(backing, split.value, 512, 2048),
                };
            } else {
                target.projection.query_key_gate_value = artifact::materialized_linear(
                    backing, *source.attention.query_key_gate_value, 9216, 2048);
            }
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = artifact::materialized_linear(backing, source.attention.output, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            const RoutedFormats routed = routed_formats(plan.weights, layer);
            target.post_mixer = load_moe(source.moe, backing, routed.gate_up, routed.down);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {2048});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {32});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {32});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {8192, 4});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16, 64, 2048);
            if (source.gdn.split) {
                const SplitGdnInputPlan& split = *source.gdn.split;
                target.projection.split        = SplitGdnInputWeights{
                           .query_key_value =
                               artifact::materialized_linear(backing, split.query_key_value, 8192, 2048),
                           .z = artifact::materialized_linear(backing, split.z, 4096, 2048),
                };
            } else {
                target.projection.query_key_value_z = artifact::materialized_linear(
                    backing, *source.gdn.query_key_value_z, 12288, 2048);
            }
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output              = artifact::materialized_linear(backing, source.gdn.output, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            const RoutedFormats routed = routed_formats(plan.weights, layer);
            target.post_mixer = load_moe(source.moe, backing, routed.gate_up, routed.down);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("35B Text topology binding is incomplete");
    }

    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {2048});
    output_head = artifact::materialized_linear(backing, plan.output_head, 248320, 2048);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 2048);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(backing, plan.mtp.input_projection,
                                                             NumericFormat::W8G32_F16S, 2048, 4096);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.attention.query_key_gate_value = artifact::materialized_linear(
            backing, *plan.mtp.attention.query_key_gate_value, 9216, 2048);
        mtp.query_norm = artifact::materialized_tensor(backing, plan.mtp.attention.query_norm,
                                                       NumericFormat::BF16, {256});
        mtp.key_norm   = artifact::materialized_tensor(backing, plan.mtp.attention.key_norm,
                                                       NumericFormat::BF16, {256});
        mtp.output     = artifact::materialized_linear(backing, plan.mtp.attention.output, 2048, 4096);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {2048});
        mtp.post_mixer =
            load_moe(plan.mtp.moe, backing, NumericFormat::W8G32_F16S, NumericFormat::W8G32_F16S);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {2048});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = family::materialize_vision_common<family::VisionBackboneConfig>(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 2048, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {2048});
    }

    if (plan.features.dflash()) {
        DFlashWeights& target     = runtime.dflash.emplace();
        target.feature_projection = artifact::materialized_weight(
            backing, plan.dflash.feature_projection, NumericFormat::W8G32_F16S, 2048, 16384);
        target.context_norm = artifact::materialized_tensor(backing, plan.dflash.context_norm,
                                                            NumericFormat::BF16, {2048});
        for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
            const DFlashLayerPlan& source = plan.dflash.layers[layer];
            DFlashLayerWeights& weights   = target.layers[layer];
            weights.input_norm      = artifact::materialized_tensor(backing, source.input_norm,
                                                                    NumericFormat::BF16, {2048});
            weights.query_key_value = artifact::materialized_weight(
                backing, source.query_key_value, NumericFormat::W8G32_F16S, 6144, 2048);
            weights.context_key   = row_view(weights.query_key_value, 4096, 1024);
            weights.context_value = row_view(weights.query_key_value, 5120, 1024);
            weights.query_norm    = artifact::materialized_tensor(backing, source.query_norm,
                                                                  NumericFormat::BF16, {128});
            weights.key_norm =
                artifact::materialized_tensor(backing, source.key_norm, NumericFormat::BF16, {128});
            weights.attention_output = artifact::materialized_weight(
                backing, source.attention_output, NumericFormat::W8G32_F16S, 2048, 4096);
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            weights.gate_up = artifact::materialized_weight(backing, source.gate_up,
                                                            NumericFormat::W8G32_F16S, 12288, 2048);
            weights.down    = artifact::materialized_weight(backing, source.down,
                                                            NumericFormat::W8G32_F16S, 2048, 6144);
        }
        target.final_norm = artifact::materialized_tensor(backing, plan.dflash.final_norm,
                                                          NumericFormat::BF16, {2048});
    }
}

} // namespace sinfer::targets::qwen3_6_35b_a3b::detail
