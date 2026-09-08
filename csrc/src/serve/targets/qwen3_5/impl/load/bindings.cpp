#include "artifact/linear_storage.h"
#include "targets/qwen3_5/impl/load/bindings.h"
#include "api/ops/linear.h"

#include "targets/qwen3_5/impl/config.h"

#include "artifact/typed_binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sinfer::targets::qwen3_5::detail {

family::TextGeometry resolved_geometry(const artifact::Reader& reader) {
    auto geometry = family::TextGeometry::resolved_hybrid(reader.geometry(), reader.layer_types());
    artifact::resolve_linear_storage(reader, geometry);
    return geometry;
}
namespace {

using artifact::NumericFormat;

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - static_cast<std::size_t>(offset) < 4) {
        throw artifact::ArtifactError(std::string(label) + ": FP32 word is outside payload");
    }
    const std::byte* value = bytes.data() + static_cast<std::size_t>(offset);
    return std::to_integer<std::uint32_t>(value[0]) |
           (std::to_integer<std::uint32_t>(value[1]) << 8U) |
           (std::to_integer<std::uint32_t>(value[2]) << 16U) |
           (std::to_integer<std::uint32_t>(value[3]) << 24U);
}

void require_positive_finite(std::uint32_t bits, std::string_view label) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw artifact::ArtifactError(std::string(label) + ": divisor must be finite and positive");
    }
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape) {
    if (format == NumericFormat::NVFP4) {
        throw std::logic_error("NVFP4 weight requires a paired input divisor");
    }
    return WeightPlan{.object = artifact::bind_device_tensor(binder, name, format, shape),
                      .format = format};
}

// A linear or table object bound by shape alone: the stored format is read from the
// artifact (W8 from a BF16 conversion, a GGML K-quant from a GGUF served natively) and
// materialized_weight builds the Weight the ops dispatch on.
/// A matrix at whatever format the artifact declares, with an explicit placement: the draft
/// block is bound but not materialised when speculation is off.
WeightPlan bind_linear_weight_at(artifact::Binder& binder, std::string_view name,
                                 std::int32_t rows, std::int32_t columns,
                                 artifact::TensorPlacement placement) {
    const artifact::LinearBinding binding =
        artifact::bind_linear(binder, name, rows, columns, placement);
    return WeightPlan{.object = binding.object, .format = binding.format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name);

WeightPlan bind_linear_weight(artifact::Binder& binder, std::string_view name,
                              std::initializer_list<std::uint64_t> shape) {
    if (shape.size() != 2) { throw std::logic_error("bind_linear_weight: rank-two shape"); }
    const auto dims = std::vector<std::uint64_t>(shape);
    const auto* object = binder.reader().find(name);
    const auto* tensor = object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
    if (tensor && tensor->format == NumericFormat::NVFP4) {
        static const std::pair<std::string_view, std::string_view> sites[] = {
            {"attention/query_key_gate_value", "attention/input_projection"},
            {"attention/query_key", "attention/input_projection"},
            {"attention/output", "attention/output_projection"},
            {"gdn/query_key_value_z", "gdn/input_projection"},
            {"gdn/query_key_value", "gdn/input_projection"},
            {"gdn/z", "gdn/z_projection"}, {"gdn/output", "gdn/output_projection"},
            {"mlp/gate_up", "mlp/gate_up_projection"}, {"mlp/down", "mlp/down_projection"},
        };
        for (const auto& [role, site] : sites) {
            if (name.ends_with(role)) {
                const std::string divisor = std::string(name.substr(0, name.size() - role.size())) +
                                            std::string(site) + "/input_scale_divisor";
                return bind_nvfp4_weight(binder, name, dims[0], dims[1], divisor);
            }
        }
        throw artifact::ArtifactError(std::string(name) + ": unsupported NVFP4 object role");
    }
    const artifact::LinearBinding binding =
        artifact::bind_linear(binder, name, static_cast<std::int32_t>(dims[0]),
                              static_cast<std::int32_t>(dims[1]));
    return WeightPlan{.object = binding.object, .format = binding.format};
}

WeightPlan bind_nvfp4_weight(artifact::Binder& binder, std::string_view name, std::int32_t rows,
                             std::int32_t columns, std::string_view input_divisor_name) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::ObjectHandle parent      = binder.require_tensor(
        name, NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1, shape);
    binder.materialize_on_device(parent);

    const artifact::ObjectHandle input_divisor =
        artifact::bind_tensor(binder, input_divisor_name, NumericFormat::FP32, {},
                              artifact::TensorPlacement::ValidateOnly);
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const std::uint32_t weight_bits =
        read_u32_le(binder.payload(parent).data, geometry.weight_divisor_offset, name);
    const std::uint32_t input_bits =
        read_u32_le(binder.payload(input_divisor).data, 0, input_divisor_name);
    require_positive_finite(weight_bits, name);
    require_positive_finite(input_bits, input_divisor_name);
    return WeightPlan{.object                    = parent,
                      .format                    = NumericFormat::NVFP4,
                      .weight_scale_divisor_bits = weight_bits,
                      .input_scale_divisor_bits  = input_bits};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
    }

    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));

    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = rows;
    out.k                    = columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = rows;
    out.shape[1]             = columns;
    out.padded_shape[0]      = rows;
    out.padded_shape[1]      = columns;
    out.weight_scale_divisor = std::bit_cast<float>(plan.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(plan.input_scale_divisor_bits);
    return out;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    // A native GGUF parent: its rows are the file's blocks, typed per segment when the file
    // quantised the components differently; the public view knows both.
    if (block.layout == QuantLayout::GgmlBlocks) { return ops::weight_rows(block, row_begin, row_count); }
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized, const family::TextGeometry& g) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 2 * g.intermediate, g.hidden);
    out.down    = materialized_weight(materialized, plan.down, g.hidden, g.intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized, const family::TextGeometry& g) {
    // The 27B-class groupwise export stores the parent as two halves, query|key and
    // gate|value, each (query_size + kv_size) rows: the query and gate halves are one head
    // width each, the key and value halves one KV width each. The extents come from this
    // target's geometry -- the branch that once lived here carried a sibling target's
    // constants, which is why it was cut, and why the 27B could not load until it was put
    // back in its own terms.
    if (const auto* split = std::get_if<SplitAttentionProjectionPlan>(&plan.projection)) {
        const std::int32_t half_rows = g.query_size() + g.kv_size();
        return SplitAttentionProjectionPayload{
            .query_key  = materialized_weight(materialized, split->query_key, half_rows, g.hidden),
            .gate_value = materialized_weight(materialized, split->gate_value, half_rows, g.hidden),
        };
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, g.mtp_attention_input_rows(), g.hidden),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized, const family::TextGeometry& g) {
    // A GGUF whose qkv and z halves carry different K-quant types is stored as two objects.
    if (const auto* split = std::get_if<QkvPlusZGdnInputProjectionPlan>(&plan.input_projection)) {
        return QkvPlusZGdnInputProjectionPayload{
            .query_key_value = materialized_weight(materialized, split->query_key_value,
                                                   g.convolution_dim(), g.hidden),
            .z = materialized_weight(materialized, split->z, g.value_dim(),
                                     g.hidden),
        };
    }
    // The 27B-class groupwise export splits one component earlier: query|key, then value|z.
    if (const auto* split = std::get_if<QkPlusVzGdnInputProjectionPlan>(&plan.input_projection)) {
        return QkPlusVzGdnInputProjectionPayload{
            .query_key =
                materialized_weight(materialized, split->query_key, 2 * g.key_dim(), g.hidden),
            .value_z =
                materialized_weight(materialized, split->value_z, 2 * g.value_dim(), g.hidden),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, g.convolution_dim() + g.value_dim(), g.hidden),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized, const family::TextGeometry& g) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, g.gdn_value_heads, g.hidden),
            .b_projection = materialized_weight(materialized, split->b_projection, g.gdn_value_heads, g.hidden),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, 2 * g.gdn_value_heads, g.hidden),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    const family::TextGeometry& g = out.geometry;
    out.text_layers.resize(static_cast<std::size_t>(g.layers));
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {g.hidden});
        target.is_full_attention = g.layer_attends(layer);
        if (target.is_full_attention) {
            // Which objects exist decides the shape of the plan: the 27B-class export stores
            // query|key and gate|value as two objects so each can carry its own group-wise
            // type, and every other export stores the one fused parent.
            if (binder.has(prefix + "attention/query_key")) {
                target.attention.projection = SplitAttentionProjectionPlan{
                    .query_key = bind_linear_weight(binder, prefix + "attention/query_key",
                                                    {g.query_size() + g.kv_size(), g.hidden}),
                    .gate_value = bind_linear_weight(binder, prefix + "attention/gate_value",
                                                     {g.query_size() + g.kv_size(), g.hidden}),
                };
            } else {
                target.attention.projection = FusedAttentionProjectionPlan{
                    .query_key_gate_value = bind_linear_weight(
                        binder, prefix + "attention/query_key_gate_value",
                        {g.mtp_attention_input_rows(), g.hidden}),
                };
            }
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {g.head_dim});
            target.attention.output = bind_linear_weight(binder, prefix + "attention/output", {g.hidden, g.query_size()});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {g.gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {g.gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {g.gdn_conv_kernel, g.convolution_dim()});
            if (binder.has(prefix + "gdn/a_b_projection")) {
                target.gdn.control_projection = FusedGdnControlProjectionPlan{
                    .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                                  NumericFormat::BF16, {2 * g.gdn_value_heads, g.hidden}),
                };
            } else {
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {g.gdn_value_heads, g.hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {g.gdn_value_heads, g.hidden}),
            };
            }
            if (binder.has(prefix + "gdn/query_key")) {
                target.gdn.input_projection = QkPlusVzGdnInputProjectionPlan{
                    .query_key = bind_linear_weight(binder, prefix + "gdn/query_key",
                                                    {2 * g.key_dim(), g.hidden}),
                    .value_z   = bind_linear_weight(binder, prefix + "gdn/value_z",
                                                    {2 * g.value_dim(), g.hidden}),
                };
            } else if (binder.has(prefix + "gdn/query_key_value")) {
                target.gdn.input_projection = QkvPlusZGdnInputProjectionPlan{
                    .query_key_value = bind_linear_weight(
                        binder, prefix + "gdn/query_key_value",
                        {g.convolution_dim(), g.hidden}),
                    .z = bind_linear_weight(binder, prefix + "gdn/z",
                                            {g.value_dim(), g.hidden}),
                };
            } else {
                target.gdn.input_projection = FusedGdnInputProjectionPlan{
                    .query_key_value_z = bind_linear_weight(
                        binder, prefix + "gdn/query_key_value_z",
                        {g.convolution_dim() + g.value_dim(), g.hidden}),
                };
            }
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {g.gdn_value_head_dim});
            target.gdn.output =
                bind_linear_weight(binder, prefix + "gdn/output", {g.hidden, g.value_dim()});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {g.hidden});
        target.mlp.gate_up =
            bind_linear_weight(binder, prefix + "mlp/gate_up", {2 * g.intermediate, g.hidden});
        target.mlp.down =
            bind_linear_weight(binder, prefix + "mlp/down", {g.hidden, g.intermediate});
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle,
                        const family::TextGeometry& g) {
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(g.token_domain, false);
    for (std::size_t i = 0; i < g.draft_vocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= g.token_domain) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               family::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.geometry = resolved_geometry(binder.reader());
    const family::TextGeometry& g = out.geometry;
    out.frontend     = family::bind_frontend_resources(binder);
    out.features     = features;

    out.token_embedding =
        bind_linear_weight(binder, "text/token_embedding", {g.output_rows, g.hidden});
    (void)weights_profile; // The package validates the profile ID; objects declare their formats.
    bind_text_layers(binder, out);
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {g.hidden});
    // A tied head and the token embedding are the same vocabulary table. A converter
    // that noticed the tie stored it once; both plans then read that one object.
    // Artifacts written before that carry a second copy, and still bind it.
    out.output_head = binder.has("text/output_head")
                          ? bind_linear_weight(binder, "text/output_head", {g.output_rows, g.hidden})
                          : out.token_embedding;
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_linear(binder, "text/draft_head", g.draft_vocab, g.hidden,
                                           proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {g.draft_vocab},
        proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids, g);

    // Community GGUF exports often strip the MTP (nextn) block, so those artifacts carry no
    // mtp/* objects at all. Binding follows the artifact: the runtime is already optional, and
    // asking for MTP speculation without the block is reported rather than guessed at.
    out.has_mtp = g.mtp_layers > 0;
    if (!out.has_mtp && features.mtp()) {
        throw std::runtime_error(
            "artifact has no MTP block (the source checkpoint was "
            "exported without nextn); run without --spec mtp");
    }
    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    if (out.has_mtp) {
    // The matrices take whatever format the artifact declares; the norms are BF16 in every
    // export. `bind_linear` is the same call the text layers use, so a K-quant draft block
    // binds exactly like a K-quant layer.
    out.mtp.input_projection_w =
        bind_linear_weight_at(binder, "mtp/input_projection", g.hidden,
                              g.mtp_input_rows(),
                              mtp_placement);
    out.mtp.input_projection = out.mtp.input_projection_w.object;
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {g.hidden});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {g.hidden});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {g.hidden});
    out.mtp.query_key_gate_value_w =
        bind_linear_weight_at(binder, "mtp/layer/attention/query_key_gate_value",
                              g.mtp_attention_input_rows(), g.hidden, mtp_placement);
    out.mtp.query_key_gate_value = out.mtp.query_key_gate_value_w.object;
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {g.head_dim});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {g.head_dim});
    out.mtp.output_w = bind_linear_weight_at(binder, "mtp/layer/attention/output", g.hidden,
                                            g.query_size(), mtp_placement);
    out.mtp.output   = out.mtp.output_w.object;
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {g.hidden});
    out.mtp.mlp.gate_up = bind_linear_weight_at(binder, "mtp/layer/mlp/gate_up",
                                               2 * g.intermediate, g.hidden, mtp_placement);
    out.mtp.mlp.down    = bind_linear_weight_at(binder, "mtp/layer/mlp/down", g.hidden,
                                               g.intermediate, mtp_placement);
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {g.hidden});
    }

    // Some checkpoints of this family ship a vision tower and some do not -- the 0.8B has
    // none, and the community GGUF exports drop it everywhere. Whether an artifact carries
    // it is a property of its source, so probe once and bind only what is there. The loader
    // rejects any object no binder claims, which is why the probe has to happen at all.
    // Placement is what `--vision` decides: ValidateOnly checks the shapes without spending
    // device memory, which is what a text-only serve of a multimodal checkpoint wants.
    out.has_vision = binder.has("vision/patch_embedding");
    if (!out.has_vision && features.vision) {
        throw std::runtime_error(
            "qwen3.5: --vision was requested but this artifact carries no vision tower "
            "(it was converted from a source that has none)");
    }
    if (out.has_vision) {
        const artifact::TensorPlacement vision_placement =
            features.vision ? artifact::TensorPlacement::Device
                            : artifact::TensorPlacement::ValidateOnly;
        // The tower is per-checkpoint the way the text stack is: the 0.8B ships 12 layers
        // of 768, the 2B and 4B 24 of 1024, the 27B 27 of 1152. The compiled config is one
        // of those; the artifact's `vision_geometry` member states the checkpoint's own,
        // and the merger always projects into the text width bound above.
        out.vision_geometry = family::VisionGeometry::resolved(
            binder.reader().vision_geometry());
        if (out.vision_geometry.output_hidden != g.hidden) {
            throw artifact::ArtifactError("vision output width disagrees with text geometry");
        }
        const family::VisionGeometry& vg  = out.vision_geometry;
        out.vision_backbone     = family::bind_vision_backbone(binder, vision_placement, vg);
        out.vision_merger_input = family::bind_vision_merger_input(binder, vision_placement, vg);
        out.vision_merger_fc2   = artifact::bind_linear(binder, "vision/merger/fc2", vg.output_hidden, vg.merger_hidden(), vision_placement);
        out.vision_merger_fc2_bias = artifact::bind_tensor(
            binder, "vision/merger/fc2_bias", NumericFormat::BF16,
            {static_cast<std::uint64_t>(vg.output_hidden)}, vision_placement);
        out.vision_merger_norm = family::bind_vision_merger_norm(binder, vision_placement, vg);
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    // The layer storage is sized here, not by the type: the counts come from the
    // geometry these weights were bound against.
    runtime.geometry              = plan.geometry;
    const family::TextGeometry& g = runtime.geometry;
    // Count layer kinds from the checkpoint's explicit schedule.
    std::size_t full_count = 0;
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        full_count += g.layer_attends(layer) ? 1 : 0;
    }
    runtime.full_layers.resize(full_count);
    runtime.gdn_layers.resize(static_cast<std::size_t>(g.layers) - full_count);
    frontend = family::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding        = materialized_weight(backing, plan.token_embedding, g.output_rows, g.hidden);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(g.layers); ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {g.hidden});
            target.projection            = load_attention_projection(source.attention, backing, g);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {g.head_dim});
            target.output     = materialized_weight(backing, source.attention.output, g.hidden, g.query_size());
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
            target.post_mixer = load_mlp(source.mlp, backing, g);
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
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing, g);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing, g);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {g.gdn_value_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, g.hidden, g.value_dim());
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {g.hidden});
            target.post_mixer = load_mlp(source.mlp, backing, g);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {g.hidden});
    output_head = materialized_weight(backing, plan.output_head, g.output_rows, g.hidden);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_linear(backing, plan.draft_head, g.draft_vocab, g.hidden);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {g.draft_vocab});
    }

    if (plan.features.mtp() && plan.has_mtp) {
        auto& mtp            = runtime.mtp.emplace();
        // The draft block projects [embedding ; hidden] -- two hidden widths -- not a query
        // plane. The binder above already says `mtp_input_rows()`; this said `query_size()`,
        // and the two are the same number at the size this target compiles (1024 hidden, 8
        // query heads of 256), so every model where they differ bound the weight with the
        // wrong K and failed at the first draft round.
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, plan.mtp.input_projection_w.format, g.hidden,
            g.mtp_input_rows());
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {g.hidden});
        mtp.attention.head_dim = g.head_dim;
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, plan.mtp.query_key_gate_value_w.format,
            g.mtp_attention_input_rows(), g.hidden);
        // The fused qkgv parent is q | k | gate | v in that order, so each component's rows
        // are found by walking those extents rather than by any one model's numbers.
        mtp.attention.query       = row_view(mtp.attention.packed, 0, g.query_size());
        mtp.attention.key         = row_view(mtp.attention.packed, g.query_size(), g.kv_size());
        mtp.attention.output_gate = row_view(mtp.attention.packed, g.query_size() + g.kv_size(), g.query_size());
        mtp.attention.value       = row_view(mtp.attention.packed, 2 * g.query_size() + g.kv_size(), g.kv_size());
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {g.head_dim});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {g.head_dim});
        mtp.output              = artifact::materialized_weight(
            backing, plan.mtp.output, plan.mtp.output_w.format, g.hidden, g.query_size());
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {g.hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing, g);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {g.hidden});
    }

    if (plan.features.vision && plan.has_vision) {
        const family::VisionGeometry& vg = plan.vision_geometry;
        runtime.vision_geometry          = vg;
        auto& vision                     = runtime.vision.emplace();
        vision.common = family::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm, vg);
        vision.merger_fc2 = artifact::materialized_linear(
            backing, plan.vision_merger_fc2, vg.output_hidden,
            vg.merger_hidden());
        vision.merger_fc2_bias = artifact::materialized_tensor(
            backing, plan.vision_merger_fc2_bias, NumericFormat::BF16, {vg.output_hidden});
    }
}

} // namespace sinfer::targets::qwen3_5::detail
