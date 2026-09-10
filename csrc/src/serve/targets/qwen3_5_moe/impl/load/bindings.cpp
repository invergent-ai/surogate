#include "artifact/linear_storage.h"
#include "targets/qwen3_5_moe/impl/load/bindings.h"

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

namespace sinfer::targets::qwen3_5_moe::detail {
namespace {

using artifact::NumericFormat;

/// The routed block's counts. `intermediate` in this target's config is one expert's, so the
/// stored parents multiply by the expert count; the shared expert has its own width.
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

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, artifact::TensorPlacement placement,
                 const family::TextGeometry& g,
                 artifact::TensorPlacement routed_placement = artifact::TensorPlacement::Device) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    // The routed experts may live elsewhere than the rest of the mixture: they are almost all
    // of its bytes and a token reaches a handful of them, which is what makes the host bank
    // worth the PCIe crossing here and nowhere else in the layer.
    const auto routed_of = [&](artifact::TensorPlacement wanted) {
        return placement == artifact::TensorPlacement::Device ? wanted : placement;
    };
    const auto bind_stored = [&](std::string_view name, std::int32_t rows, std::int32_t columns) {
        const auto* descriptor = std::get_if<artifact::TensorDescriptor>(binder.reader().find(name));
        if (descriptor && descriptor->format == NumericFormat::NVFP4) {
            // Routed NVFP4 has per-expert calibration, not the dense input-divisor object.
            return artifact::LinearBinding{
                artifact::bind_tensor(binder, name, NumericFormat::NVFP4,
                                      {static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)},
                                      routed_of(routed_placement)), NumericFormat::NVFP4};
        }
        return artifact::bind_linear(binder, name, rows, columns, routed_of(routed_placement));
    };
    MoePlan plan{
        .router_shared_gate = bind(prefix + "router_shared_gate", NumericFormat::BF16,
                                  {g.experts + 1, g.hidden}),
        // The routed experts' format comes from the artifact: a GGUF serves them as its own
        // K-quant superblocks, a converted checkpoint as the groupwise-int or NVFP4 profile.
        .routed_gate_up     = bind_stored(prefix + "routed_gate_up", g.experts * 2 * g.intermediate, g.hidden),
        .routed_down        = bind_stored(prefix + "routed_down", g.experts * g.hidden, g.intermediate),
        .shared_gate_up = bind(prefix + "shared_gate_up", NumericFormat::W8G32_F16S,
                              {2 * g.shared_intermediate, g.hidden}),
        .shared_down    = bind(prefix + "shared_down", NumericFormat::W8G32_F16S,
                              {g.hidden, g.shared_intermediate}),
    };
    // NVFP4 carries a second-level scale per expert per projection; the row block holds gate and
    // up stacked, so gate/up needs two entries per expert and down one.
    if (plan.routed_gate_up.format == NumericFormat::NVFP4) {
        plan.routed_gate_up_scale =
            bind(prefix + "routed_gate_up_scale", NumericFormat::FP32, {2 * g.experts});
        plan.routed_gate_up_act_scale =
            bind(prefix + "routed_gate_up_act_scale", NumericFormat::FP32, {g.experts});
        plan.routed_gate_up_alpha =
            bind(prefix + "routed_gate_up_alpha", NumericFormat::FP32, {g.experts});
        require_identity_divisor(binder, plan.routed_gate_up.object,
                                 prefix + "routed_gate_up", g.experts * 2 * g.intermediate, g.hidden);
    }
    if (plan.routed_down.format == NumericFormat::NVFP4) {
        plan.routed_down_scale =
            bind(prefix + "routed_down_scale", NumericFormat::FP32, {g.experts});
        plan.routed_down_act_scale =
            bind(prefix + "routed_down_act_scale", NumericFormat::FP32, {g.experts});
        plan.routed_down_alpha =
            bind(prefix + "routed_down_alpha", NumericFormat::FP32, {g.experts});
        require_identity_divisor(binder, plan.routed_down.object, prefix + "routed_down",
                                 g.experts * g.hidden, g.intermediate);
    }
    return plan;
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          const family::TextGeometry& g) {
    SparseMoePayload payload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router_shared_gate, NumericFormat::BF16, g.experts + 1, g.hidden),
            // The stored format decides how these bytes are read. Passing the profile's
            // expectation instead decoded a GGUF's Q4_K/Q5_K superblocks as the groupwise-int
            // row-split codec: same byte count, entirely different meaning, and every routed
            // expert silently wrong.
            .routed_gate_up =
                plan.routed_gate_up.format == NumericFormat::NVFP4
                    ? routed_nvfp4_weight(materialized, plan.routed_gate_up.object,
                                            g.experts * 2 * g.intermediate, g.hidden)
                    : artifact::materialized_linear(materialized, plan.routed_gate_up,
                                                        g.experts * 2 * g.intermediate, g.hidden),
            .routed_down =
                plan.routed_down.format == NumericFormat::NVFP4
                    ? routed_nvfp4_weight(materialized, plan.routed_down.object,
                                            g.experts * g.hidden, g.intermediate)
                    : artifact::materialized_linear(materialized, plan.routed_down,
                                                        g.experts * g.hidden, g.intermediate),
            .shared_gate_up = artifact::materialized_weight(materialized, plan.shared_gate_up,
                                                            NumericFormat::W8G32_F16S, 2 * g.shared_intermediate, g.hidden),
            .shared_down    = artifact::materialized_weight(materialized, plan.shared_down,
                                                            NumericFormat::W8G32_F16S, g.hidden, g.shared_intermediate),
            .experts_per_token = g.experts_per_token,
        }};
    if (plan.routed_gate_up_scale.has_value()) {
        payload.op.routed_gate_up_scale = static_cast<const float*>(
            artifact::materialized_tensor(materialized, *plan.routed_gate_up_scale,
                                          NumericFormat::FP32, {2 * g.experts})
                .data);
    }
    if (plan.routed_down_scale.has_value()) {
        payload.op.routed_down_scale =
            static_cast<const float*>(artifact::materialized_tensor(materialized,
                                                                    *plan.routed_down_scale,
                                                                    NumericFormat::FP32,
                                                                    {g.experts})
                                          .data);
    }
    const auto per_expert = [&](const std::optional<artifact::ObjectHandle>& handle) {
        return handle.has_value()
                   ? static_cast<const float*>(
                         artifact::materialized_tensor(materialized, *handle, NumericFormat::FP32,
                                                       {g.experts})
                             .data)
                   : nullptr;
    };
    payload.op.routed_gate_up_act_scale = per_expert(plan.routed_gate_up_act_scale);
    payload.op.routed_gate_up_alpha     = per_expert(plan.routed_gate_up_alpha);
    payload.op.routed_down_act_scale    = per_expert(plan.routed_down_act_scale);
    payload.op.routed_down_alpha        = per_expert(plan.routed_down_alpha);
    return payload;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle, const family::TextGeometry& g) {
    const std::size_t kTokenizerVocab = g.token_domain;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < g.draft_vocab; ++i) {
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

family::TextGeometry resolved_geometry(const artifact::Reader& reader) {
    auto g = family::TextGeometry::resolved_hybrid(reader.geometry(), reader.layer_types(), true);
    if (!reader.dflash_geometry().empty()) {
        g.dflash = family::DFlashGeometry::resolved(reader.dflash_geometry(), reader.dflash_target_layers(),
                                                    g.hidden, g.layers, g.output_rows);
    }
    artifact::resolve_linear_storage(reader, g);
    return g;
}

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, family::StartupFeatures features,
                               WeightsProfile weights, std::uint32_t host_moe_layers,
                               std::uint32_t gpu_layers, LoadProgress progress) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    // Required checkpoint metadata is shared by binding and memory planning.
    out.geometry = resolved_geometry(binder.reader());
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(g.layers);
    out.frontend        = family::bind_frontend_resources(binder);
    out.features        = features;
    out.weights         = weights;
    out.token_embedding = artifact::bind_linear(binder, "text/token_embedding", g.output_rows, g.hidden);

    for (std::size_t layer = 0; layer < g.layers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        target.resident = binder.contains_layer(static_cast<int>(layer));
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {g.hidden});
        target.is_full_attention = g.layer_attends(layer);
        if (target.is_full_attention) {
            // One object per HF Linear when the artifact stores them so (a quantized export,
            // each constituent with its own global scale), the fused parent otherwise. The
            // format of each is read from the artifact, not asserted.
            if (binder.has(prefix + "attention/query")) {
                target.attention.split = SplitAttentionPlan{
                    .query = artifact::bind_linear(binder, prefix + "attention/query", g.query_size(), g.hidden),
                    .key   = artifact::bind_linear(binder, prefix + "attention/key", g.kv_size(), g.hidden),
                    .gate  = artifact::bind_linear(binder, prefix + "attention/gate", g.query_size(), g.hidden),
                    .value = artifact::bind_linear(binder, prefix + "attention/value", g.kv_size(), g.hidden),
                };
            } else {
                target.attention.query_key_gate_value = artifact::bind_linear(
                    binder, prefix + "attention/query_key_gate_value",
                                              2 * g.query_size() + 2 * g.kv_size(), g.hidden);
            }
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.output =
                artifact::bind_linear(binder, prefix + "attention/output", g.hidden, g.query_size());
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {g.gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {g.gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {g.gdn_conv_kernel, g.convolution_dim()});
            target.gdn.a_b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_b_projection", NumericFormat::BF16, {2 * g.gdn_value_heads, g.hidden});
            if (binder.has(prefix + "gdn/query_key_value")) {
                target.gdn.split = SplitGdnInputPlan{
                    .query_key_value =
                        artifact::bind_linear(binder, prefix + "gdn/query_key_value", g.convolution_dim(), g.hidden),
                    .z = artifact::bind_linear(binder, prefix + "gdn/z", g.value_dim(), g.hidden),
                };
            } else {
                target.gdn.query_key_value_z =
                    artifact::bind_linear(binder, prefix + "gdn/query_key_value_z",
                                              g.convolution_dim() + g.value_dim(), g.hidden);
            }
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {g.gdn_value_head_dim});
            target.gdn.output = artifact::bind_linear(binder, prefix + "gdn/output", g.hidden, g.value_dim());
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.moe = bind_moe(binder, prefix + "moe/",
                              artifact::ScopedPlacement::current(), g,
                              artifact::TensorPlacement::Device);
    }

    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {g.hidden});
    out.output_head = artifact::bind_linear(binder, "text/output_head", g.output_rows, g.hidden);
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    // A GGUF serves the draft head as its own K-quant; a converted checkpoint as Q4.
    out.draft_head = artifact::bind_linear(binder, "text/draft_head", g.draft_vocab, g.hidden,
                                           proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {g.draft_vocab},
        proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids, g);

    // A community GGUF export usually strips the nextn block; the artifact then omits mtp/*
    // and only `--spec mtp` against such a model is the error.
    out.has_mtp = g.mtp_layers > 0;
    if (out.has_mtp != binder.has("mtp/input_projection")) {
        throw artifact::ArtifactError("MTP metadata disagrees with stored objects");
    }
    if (!out.has_mtp && features.mtp()) {
        throw artifact::ArtifactError(
            "qwen3_5_moe: --spec mtp was requested but this artifact carries no MTP block");
    }
    out.resident_mtp = features.mtp() && binder.contains_layer(g.layers - 1);
    const artifact::TensorPlacement mtp_placement = out.resident_mtp
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    if (out.has_mtp) {
        out.mtp.input_projection =
            bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {g.hidden, g.mtp_input_rows()});
        out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {g.hidden});
        out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {g.hidden});
        out.mtp.input_norm     = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {g.hidden});
        out.mtp.attention.query_key_gate_value = artifact::bind_linear(
            binder, "mtp/layer/attention/query_key_gate_value", g.mtp_attention_input_rows(),
            g.hidden, mtp_placement);
        out.mtp.attention.query_norm =
            bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {g.head_dim});
        out.mtp.attention.key_norm =
            bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {g.head_dim});
        out.mtp.attention.output =
            artifact::bind_linear(binder, "mtp/layer/attention/output", g.hidden, g.query_size(),
                                  mtp_placement);
        out.mtp.post_attention_norm =
            bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {g.hidden});
        out.mtp.moe        = bind_moe(binder, "mtp/layer/moe/", mtp_placement, g);
        out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {g.hidden});
    }
    // Community GGUF exports of this family are text-only; the artifact then omits vision/*
    // and only `--vision` against such a model is the error.
    out.has_vision = binder.has("vision/patch_embedding");
    if (!out.has_vision && features.vision) {
        throw artifact::ArtifactError(
            "qwen3_5_moe: --vision was requested but this artifact carries no vision tower");
    }
    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    if (out.has_vision) {
    out.vision_geometry = family::VisionGeometry::resolved(binder.reader().vision_geometry());
    const auto& vg = out.vision_geometry;
    if (vg.output_hidden != g.hidden) { throw artifact::ArtifactError("vision output width disagrees with text"); }
    out.vision_backbone     = family::bind_vision_backbone(binder, vision_placement, vg);
    out.vision_merger_input = family::bind_vision_merger_input(binder, vision_placement, vg);
    out.vision_merger_fc2   = artifact::bind_linear(binder, "vision/merger/fc2", vg.output_hidden, vg.merger_hidden(), vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {vg.output_hidden},
            vision_placement);
    out.vision_merger_norm = family::bind_vision_merger_norm(binder, vision_placement, vg);
    }

    // The DFlash drafter is a separate checkpoint the converter may not have
    // had; such artifacts omit every dflash/* object. Probe the family once
    // and turn a missing drafter into a startup error only when DFlash was
    // actually requested, rather than a missing-object failure on load.
    out.has_dflash = g.dflash.layers > 0;
    if (out.has_dflash != binder.has("dflash/feature_projection")) {
        throw artifact::ArtifactError("DFlash metadata disagrees with stored objects");
    }
    if (!out.has_dflash && features.dflash()) {
        throw std::runtime_error(
            "qwen3_5_moe artifact has no DFlash drafter (converted without "
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
    const auto& d = g.dflash;
    out.dflash.layers.resize(d.layers);
    out.dflash.feature_projection =
        bind_dflash("dflash/feature_projection", NumericFormat::W8G32_F16S,
                    {d.hidden, d.feature_rows});
    out.dflash.context_norm = bind_dflash("dflash/context_norm", NumericFormat::BF16, {g.hidden});
    for (std::size_t layer = 0; layer < d.layers; ++layer) {
        DFlashLayerPlan& target  = out.dflash.layers[layer];
        const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_dflash(prefix + "input_norm", NumericFormat::BF16, {g.hidden});
        target.query_key_value   = bind_dflash(prefix + "attention/query_key_value",
                                               NumericFormat::W8G32_F16S, {d.query_size() + 2 * d.kv_size(), g.hidden});
        target.query_norm =
            bind_dflash(prefix + "attention/query_norm", NumericFormat::BF16,
                        {d.head_dim});
        target.key_norm = bind_dflash(prefix + "attention/key_norm", NumericFormat::BF16,
                                      {d.head_dim});
        target.attention_output =
            bind_dflash(prefix + "attention/output", NumericFormat::W8G32_F16S, {g.hidden, d.query_size()});
        target.post_attention_norm =
            bind_dflash(prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.gate_up =
            bind_dflash(prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                        {2 * d.intermediate, d.hidden});
        target.down = bind_dflash(prefix + "mlp/down", NumericFormat::W8G32_F16S,
                                  {d.hidden, d.intermediate});
    }
    out.dflash.final_norm = bind_dflash("dflash/final_norm", NumericFormat::BF16, {d.hidden});
    }

    load_plan.materialization = binder.finish();
    out.host_bank = family::collect_host_bank(binder, load_plan.materialization,
                                              std::move(progress));
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)),
      host_bank(plan.host_bank.objects.empty() ? nullptr
                                               : family::HostBank::shared(plan.host_bank)) {
    if (host_bank) { host_bank->attach(backing); }
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    const auto full_count = std::count_if(plan.text_layers.begin(), plan.text_layers.end(),
                                         [](const auto& layer) { return layer.is_full_attention; });
    runtime.full_layers.resize(full_count);
    runtime.gdn_layers.resize(g.layers - full_count);
    frontend = family::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = artifact::materialized_linear(backing, plan.token_embedding, g.output_rows, g.hidden);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < g.layers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (!source.resident) {
            if (source.is_full_attention) { ++full_index; } else { ++gdn_index; }
            continue;
        }
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {g.hidden});
            target.projection.head_dim = g.head_dim;
            if (source.attention.split) {
                const SplitAttentionPlan& split = *source.attention.split;
                target.projection.split         = SplitAttentionWeights{
                            .query = artifact::materialized_linear(backing, split.query, g.query_size(), g.hidden),
                            .key   = artifact::materialized_linear(backing, split.key, g.kv_size(), g.hidden),
                            .gate  = artifact::materialized_linear(backing, split.gate, g.query_size(), g.hidden),
                            .value = artifact::materialized_linear(backing, split.value, g.kv_size(), g.hidden),
                };
            } else {
                target.projection.query_key_gate_value = artifact::materialized_linear(
                    backing, *source.attention.query_key_gate_value,
            2 * g.query_size() + 2 * g.kv_size(), g.hidden);
            }
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.output     = artifact::materialized_linear(backing, source.attention.output, g.hidden, g.query_size());
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
            target.post_mixer = load_moe(source.moe, backing, g);
            target.post_mixer.banked = family::bind_banked_experts(
                target.post_mixer.op, host_bank.get(), source.moe.routed_gate_up.object,
                source.moe.routed_down.object, static_cast<std::int32_t>(layer), g.layers + g.mtp_layers);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {g.hidden});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {g.gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {g.gdn_value_heads});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {g.convolution_dim(), g.gdn_conv_kernel});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16, 2 * g.gdn_value_heads,
                g.hidden);
            if (source.gdn.split) {
                const SplitGdnInputPlan& split = *source.gdn.split;
                target.projection.split        = SplitGdnInputWeights{
                           .query_key_value =
                               artifact::materialized_linear(backing, split.query_key_value, g.convolution_dim(), g.hidden),
                           .z = artifact::materialized_linear(backing, split.z, g.value_dim(), g.hidden),
                };
            } else {
                target.projection.query_key_value_z = artifact::materialized_linear(
                    backing, *source.gdn.query_key_value_z, g.convolution_dim() + g.value_dim(), g.hidden);
            }
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {g.gdn_value_head_dim});
            target.output              = artifact::materialized_linear(backing, source.gdn.output, g.hidden, g.value_dim());
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
            target.post_mixer = load_moe(source.moe, backing, g);
            target.post_mixer.banked = family::bind_banked_experts(
                target.post_mixer.op, host_bank.get(), source.moe.routed_gate_up.object,
                source.moe.routed_down.object, static_cast<std::int32_t>(layer), g.layers + g.mtp_layers);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("hybrid MoE text topology binding is incomplete");
    }

    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {g.hidden});
    output_head = artifact::materialized_linear(backing, plan.output_head, g.output_rows, g.hidden);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_linear(backing, plan.draft_head, g.draft_vocab, g.hidden);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {g.draft_vocab});
    }

    if (plan.resident_mtp) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(backing, plan.mtp.input_projection,
                                                             NumericFormat::W8G32_F16S, g.hidden, g.mtp_input_rows());
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.attention.head_dim = g.head_dim;
        mtp.attention.query_key_gate_value = artifact::materialized_linear(
            backing, *plan.mtp.attention.query_key_gate_value,
            2 * g.query_size() + 2 * g.kv_size(), g.hidden);
        mtp.query_norm = artifact::materialized_tensor(backing, plan.mtp.attention.query_norm,
                                                       NumericFormat::BF16, {g.head_dim});
        mtp.key_norm   = artifact::materialized_tensor(backing, plan.mtp.attention.key_norm,
                                                       NumericFormat::BF16, {g.head_dim});
        mtp.output     = artifact::materialized_linear(backing, plan.mtp.attention.output, g.hidden, g.query_size());
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {g.hidden});
        mtp.post_mixer =
            load_moe(plan.mtp.moe, backing, g);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {g.hidden});
    }

    if (plan.features.vision) {
        const auto& vg = plan.vision_geometry;
        runtime.vision_geometry = vg;
        auto& vision  = runtime.vision.emplace();
        vision.common = family::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm, vg);
        vision.merger_fc2      = artifact::materialized_linear(backing, plan.vision_merger_fc2, vg.output_hidden,
            vg.merger_hidden());
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {g.hidden});
    }

    if (plan.features.dflash()) {
        const auto& d = g.dflash;
        DFlashWeights& target     = runtime.dflash.emplace();
        target.layers.resize(d.layers);
        target.feature_projection = artifact::materialized_weight(
            backing, plan.dflash.feature_projection, NumericFormat::W8G32_F16S, d.hidden,
        d.feature_rows);
        target.context_norm = artifact::materialized_tensor(backing, plan.dflash.context_norm,
                                                            NumericFormat::BF16, {g.hidden});
        for (std::size_t layer = 0; layer < d.layers; ++layer) {
            const DFlashLayerPlan& source = plan.dflash.layers[layer];
            DFlashLayerWeights& weights   = target.layers[layer];
            weights.input_norm      = artifact::materialized_tensor(backing, source.input_norm,
                                                                    NumericFormat::BF16, {g.hidden});
            weights.query_key_value = artifact::materialized_weight(
                backing, source.query_key_value, NumericFormat::W8G32_F16S,
            d.query_size() + 2 * d.kv_size(), d.hidden);
            weights.context_key   = row_view(weights.query_key_value, d.query_size(), d.kv_size());
            weights.context_value = row_view(weights.query_key_value, d.query_size() + d.kv_size(),
                     d.kv_size());
            weights.query_norm    = artifact::materialized_tensor(backing, source.query_norm,
                                                                  NumericFormat::BF16, {d.head_dim});
            weights.key_norm =
                artifact::materialized_tensor(backing, source.key_norm, NumericFormat::BF16, {d.head_dim});
            weights.attention_output = artifact::materialized_weight(
                backing, source.attention_output, NumericFormat::W8G32_F16S, d.hidden,
            d.query_size());
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
            weights.gate_up = artifact::materialized_weight(backing, source.gate_up,
                                                            NumericFormat::W8G32_F16S, 2 * d.intermediate,
                                                        d.hidden);
            weights.down    = artifact::materialized_weight(backing, source.down,
                                                            NumericFormat::W8G32_F16S, d.hidden,
                                                     d.intermediate);
        }
        target.final_norm = artifact::materialized_tensor(backing, plan.dflash.final_norm,
                                                          NumericFormat::BF16, {g.hidden});
    }
}

} // namespace sinfer::targets::qwen3_5_moe::detail
