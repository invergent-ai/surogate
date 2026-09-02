#include "targets/qwen3_5_2b/impl/load/bindings.h"

#include "targets/qwen3_5_2b/impl/config.h"

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

namespace sinfer::targets::qwen3_5_2b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

bool is_early_attention_input(std::size_t layer) {
    return layer == 3 || layer == 7 || layer == 11 || layer == 15 || layer == 19 || layer == 23;
}

bool is_bf16_attention_output(std::size_t layer) { return layer == 3 || layer == 7; }

bool is_bf16_gdn_output(std::size_t layer) { return layer == 4; }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
        return NumericFormat::W8G32_F16S;  // W8 profile (Q8_0-faithful)
    case WeightsProfile::Qwen38GroupwiseInt:
    case WeightsProfile::Qwen36Nvfp4:
        return NumericFormat::W8G32_F16S;
    case WeightsProfile::Qwen38Nvfp4:
        return NumericFormat::FP8_E4M3FN_ROW_BF16S;
    }
    throw std::invalid_argument("qwen3_5_2b: invalid weights profile");
}

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
WeightPlan bind_linear_weight(artifact::Binder& binder, std::string_view name,
                              std::initializer_list<std::uint64_t> shape) {
    if (shape.size() != 2) { throw std::logic_error("bind_linear_weight: rank-two shape"); }
    const auto dims = std::vector<std::uint64_t>(shape);
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
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, 2 * TextConfig::intermediate, TextConfig::hidden);
    out.down    = materialized_weight(materialized, plan.down, TextConfig::hidden, TextConfig::intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    // The split attention plan is never constructed for this target — only the
    // 27B produces one. The branch that used to live here carried shape
    // constants copied from a sibling target (a hidden size that was not this
    // target's), which nothing caught because the code was unreachable. If a
    // split plan ever does appear here, that is a binding bug and it should
    // stop the load rather than silently materialise wrong extents.
    if (std::holds_alternative<SplitAttentionProjectionPlan>(plan.projection)) {
        throw std::invalid_argument("attention projection: split plans are not produced for this target");
    }
    const auto& fused = std::get<FusedAttentionProjectionPlan>(plan.projection);
    return FusedAttentionProjectionPayload{
        .query_key_gate_value =
            materialized_weight(materialized, fused.query_key_gate_value, TextConfig::mtp_attention_input_rows, TextConfig::hidden),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    // A GGUF whose qkv and z halves carry different K-quant types is stored as two objects.
    if (const auto* split = std::get_if<SplitGdnInputProjectionPlan>(&plan.input_projection)) {
        return SplitGdnInputProjectionPayload{
            .query_key_value = materialized_weight(materialized, split->query_key_value,
                                                   TextConfig::convolution_dim, TextConfig::hidden),
            .z = materialized_weight(materialized, split->z, TextConfig::value_dim,
                                     TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnInputProjectionPlan>(plan.input_projection);
    return FusedGdnInputProjectionPayload{
        .query_key_value_z =
            materialized_weight(materialized, fused.query_key_value_z, TextConfig::convolution_dim + TextConfig::value_dim, TextConfig::hidden),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    if (const auto* split = std::get_if<SplitGdnControlProjectionPlan>(&plan.control_projection)) {
        return SplitGdnControlProjectionPayload{
            .a_projection = materialized_weight(materialized, split->a_projection, TextConfig::gdn_value_heads, TextConfig::hidden),
            .b_projection = materialized_weight(materialized, split->b_projection, TextConfig::gdn_value_heads, TextConfig::hidden),
        };
    }
    const auto& fused = std::get<FusedGdnControlProjectionPlan>(plan.control_projection);
    return FusedGdnControlProjectionPayload{
        .a_b_projection = materialized_weight(materialized, fused.a_b_projection, 2 * TextConfig::gdn_value_heads, TextConfig::hidden),
    };
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value =
                    bind_linear_weight(binder, prefix + "attention/query_key_gate_value", {TextConfig::mtp_attention_input_rows, TextConfig::hidden}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output = bind_linear_weight(binder, prefix + "attention/output", {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            if (binder.has(prefix + "gdn/query_key_value")) {
                target.gdn.input_projection = SplitGdnInputProjectionPlan{
                    .query_key_value = bind_linear_weight(
                        binder, prefix + "gdn/query_key_value",
                        {TextConfig::convolution_dim, TextConfig::hidden}),
                    .z = bind_linear_weight(binder, prefix + "gdn/z",
                                            {TextConfig::value_dim, TextConfig::hidden}),
                };
            } else {
                target.gdn.input_projection = FusedGdnInputProjectionPlan{
                    .query_key_value_z = bind_linear_weight(
                        binder, prefix + "gdn/query_key_value_z",
                        {TextConfig::convolution_dim + TextConfig::value_dim, TextConfig::hidden}),
                };
            }
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {TextConfig::gdn_key_head_dim});
            target.gdn.output =
                bind_linear_weight(binder, prefix + "gdn/output", {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up =
            bind_linear_weight(binder, prefix + "mlp/gate_up", {2 * TextConfig::intermediate, TextConfig::hidden});
        target.mlp.down =
            bind_linear_weight(binder, prefix + "mlp/down", {TextConfig::hidden, TextConfig::intermediate});
    }
}

void bind_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            WeightPlan input;
            if (is_early_attention_input(layer)) {
                input = bind_weight(binder, prefix + "attention/query_key_gate_value",
                                    NumericFormat::BF16, {TextConfig::mtp_attention_input_rows, TextConfig::hidden});
            } else {
                input = bind_nvfp4_weight(
                    binder, prefix + "attention/query_key_gate_value", TextConfig::mtp_attention_input_rows, TextConfig::hidden,
                    prefix + "attention/input_projection/input_scale_divisor");
            }
            target.attention.projection =
                FusedAttentionProjectionPlan{.query_key_gate_value = input};
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            if (is_bf16_attention_output(layer)) {
                target.attention.output = bind_weight(binder, prefix + "attention/output",
                                                      NumericFormat::BF16, {TextConfig::hidden, TextConfig::query_size});
            } else {
                target.attention.output =
                    bind_nvfp4_weight(binder, prefix + "attention/output", TextConfig::hidden, TextConfig::query_size,
                                      prefix + "attention/output_projection/input_scale_divisor");
            }
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = SplitGdnControlProjectionPlan{
                .a_projection = bind_weight(binder, prefix + "gdn/a_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
                .b_projection = bind_weight(binder, prefix + "gdn/b_projection",
                                            NumericFormat::BF16, {TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_nvfp4_weight(binder, prefix + "gdn/query_key_value_z", TextConfig::convolution_dim + TextConfig::value_dim, TextConfig::hidden,
                                      prefix + "gdn/input_projection/input_scale_divisor"),
            };
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16, {TextConfig::gdn_key_head_dim});
            if (is_bf16_gdn_output(layer)) {
                target.gdn.output =
                    bind_weight(binder, prefix + "gdn/output", NumericFormat::BF16, {TextConfig::hidden, TextConfig::value_dim});
            } else {
                target.gdn.output =
                    bind_nvfp4_weight(binder, prefix + "gdn/output", TextConfig::hidden, TextConfig::value_dim,
                                      prefix + "gdn/output_projection/input_scale_divisor");
            }
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        target.mlp.gate_up =
            bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 2 * TextConfig::intermediate, TextConfig::hidden,
                              prefix + "mlp/gate_up_projection/input_scale_divisor");
        target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", TextConfig::hidden, TextConfig::intermediate,
                                            prefix + "mlp/down_projection/input_scale_divisor");
    }
}

void bind_qwen38_nvfp4_text_layers(artifact::Binder& binder, BindingPlan& out) {
    constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {TextConfig::hidden});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.projection = FusedAttentionProjectionPlan{
                .query_key_gate_value = bind_weight(
                    binder, prefix + "attention/query_key_gate_value", kFp8, {TextConfig::mtp_attention_input_rows, TextConfig::hidden}),
            };
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", kFp8, {TextConfig::hidden, TextConfig::query_size});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {TextConfig::gdn_conv_kernel, TextConfig::convolution_dim});
            target.gdn.control_projection = FusedGdnControlProjectionPlan{
                .a_b_projection = bind_weight(binder, prefix + "gdn/a_b_projection",
                                              NumericFormat::BF16, {2 * TextConfig::gdn_value_heads, TextConfig::hidden}),
            };
            target.gdn.input_projection = FusedGdnInputProjectionPlan{
                .query_key_value_z =
                    bind_weight(binder, prefix + "gdn/query_key_value_z", kFp8, {TextConfig::convolution_dim + TextConfig::value_dim, TextConfig::hidden}),
            };
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {TextConfig::gdn_key_head_dim});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", kFp8, {TextConfig::hidden, TextConfig::value_dim});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
        if (layer < 56) {
            target.mlp.gate_up =
                bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 2 * TextConfig::intermediate, TextConfig::hidden,
                                  prefix + "mlp/gate_up_projection/input_scale_divisor");
            target.mlp.down = bind_nvfp4_weight(binder, prefix + "mlp/down", TextConfig::hidden, TextConfig::intermediate,
                                                prefix + "mlp/down_projection/input_scale_divisor");
        } else {
            target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kFp8, {2 * TextConfig::intermediate, TextConfig::hidden});
            target.mlp.down    = bind_weight(binder, prefix + "mlp/down", kFp8, {TextConfig::hidden, TextConfig::intermediate});
        }
    }
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
        if (id >= kTokenizerVocab) {
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
    out.frontend     = family::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding =
        bind_linear_weight(binder, "text/token_embedding", {TextConfig::output_rows, TextConfig::hidden});
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen36Nvfp4:
        bind_nvfp4_text_layers(binder, out);
        break;
    case WeightsProfile::Qwen38Nvfp4:
        bind_qwen38_nvfp4_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_5_2b: invalid weights profile");
    }
    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {TextConfig::hidden});
    // A tied head and the token embedding are the same vocabulary table. A converter
    // that noticed the tie stored it once; both plans then read that one object.
    // Artifacts written before that carry a second copy, and still bind it.
    out.output_head = binder.has("text/output_head")
                          ? bind_linear_weight(binder, "text/output_head", {TextConfig::output_rows, TextConfig::hidden})
                          : out.token_embedding;
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_linear(binder, "text/draft_head", 131072, 2048,
                                           proposal_placement);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    // surogate vendor patch (PATCHES.md #15): community GGUF exports often
    // strip the MTP (nextn) block; such artifacts omit the mtp/* objects
    // entirely. The runtime is already optional (materialization and the
    // speculative executor key off features.mtp()); binding follows the
    // artifact, and requesting MTP speculation without the block is a clear
    // startup error instead of a missing-object failure.
    out.has_mtp = binder.has("mtp/input_projection");
    if (!out.has_mtp && features.mtp()) {
        throw std::runtime_error(
            "qwen3.5-2b artifact has no MTP block (the source checkpoint was "
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
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {2048, TextConfig::mtp_input_rows});
    out.mtp.embedding_norm       = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.hidden_norm          = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.input_norm           = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                            NumericFormat::W8G32_F16S, {TextConfig::mtp_attention_input_rows, TextConfig::hidden});
    out.mtp.query_norm = bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.key_norm   = bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {TextConfig::head_dim});
    out.mtp.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {TextConfig::hidden, TextConfig::query_size});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {TextConfig::hidden});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S, {2 * TextConfig::intermediate, TextConfig::hidden}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S, {TextConfig::hidden, TextConfig::intermediate}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {TextConfig::hidden});
    }

    // This checkpoint ships a vision tower, so the artifact may carry it and the
    // binder must consume it — the loader rejects any object no binder claims.
    // Placement is what `--vision` decides: ValidateOnly checks the shapes without
    // spending device memory, which is what a text-only serve of a multimodal
    // checkpoint wants.
    // Whether an artifact carries the tower is a property of its source, not of the
    // model: the community GGUF exports of this family drop vision entirely. Probe
    // once and bind only what is there, so a text-only artifact loads; asking for
    // --vision without one is the error, not the artifact's existence.
    out.has_vision = binder.has("vision/patch_embedding");
    if (!out.has_vision && features.vision) {
        throw std::runtime_error(
            "qwen3.5-2b: --vision was requested but this artifact carries no vision tower "
            "(it was converted from a source that has none)");
    }
    if (out.has_vision) {
        const artifact::TensorPlacement vision_placement =
            features.vision ? artifact::TensorPlacement::Device
                            : artifact::TensorPlacement::ValidateOnly;
        out.vision_backbone =
            family::bind_vision_backbone<VisionConfig>(binder, vision_placement);
        out.vision_merger_input =
            family::bind_vision_merger_input<VisionConfig>(binder, vision_placement);
        out.vision_merger_fc2 =
            artifact::bind_tensor(binder, "vision/merger/fc2", NumericFormat::W8G32_F16S,
                                  {VisionConfig::output_hidden,
                                   VisionConfig::merger_hidden},
                                  vision_placement);
        out.vision_merger_fc2_bias =
            artifact::bind_tensor(binder, "vision/merger/fc2_bias", NumericFormat::BF16,
                                  {VisionConfig::output_hidden}, vision_placement);
        out.vision_merger_norm =
            family::bind_vision_merger_norm<VisionConfig>(binder, vision_placement);
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

    token_embedding        = materialized_weight(backing, plan.token_embedding, TextConfig::output_rows, TextConfig::hidden);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {TextConfig::hidden});
            target.projection            = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {TextConfig::head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {TextConfig::head_dim});
            target.output     = materialized_weight(backing, source.attention.output, TextConfig::hidden, TextConfig::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {TextConfig::hidden});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {TextConfig::gdn_value_heads});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {TextConfig::convolution_dim, TextConfig::gdn_conv_kernel});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {TextConfig::gdn_key_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, TextConfig::hidden, TextConfig::value_dim);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {TextConfig::hidden});
    output_head = materialized_weight(backing, plan.output_head, TextConfig::output_rows, TextConfig::hidden);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_linear(backing, plan.draft_head, 131072, TextConfig::hidden);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp() && plan.has_mtp) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, 2048, TextConfig::mtp_input_rows);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {TextConfig::hidden});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S, TextConfig::mtp_attention_input_rows, TextConfig::hidden);
        // surogate vendor patch (PATCHES.md #13): qwen3.5-2b fused qkgv rows
        // (q 2048 | k 512 | gate 2048 | v 512), not the 27B extents.
        mtp.attention.query       = row_view(mtp.attention.packed, 0, TextConfig::query_size);
        mtp.attention.key         = row_view(mtp.attention.packed, TextConfig::query_size, TextConfig::kv_size);
        mtp.attention.output_gate = row_view(mtp.attention.packed, TextConfig::query_size + TextConfig::kv_size, TextConfig::query_size);
        mtp.attention.value       = row_view(mtp.attention.packed, 2 * TextConfig::query_size + TextConfig::kv_size, TextConfig::kv_size);
        mtp.query_norm =
            artifact::materialized_tensor(backing, plan.mtp.query_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.key_norm =
            artifact::materialized_tensor(backing, plan.mtp.key_norm, NumericFormat::BF16, {TextConfig::head_dim});
        mtp.output              = artifact::materialized_weight(backing, plan.mtp.output,
                                                                NumericFormat::W8G32_F16S, 2048, TextConfig::hidden);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {TextConfig::hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {TextConfig::hidden});
    }

    if (plan.features.vision && plan.has_vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = family::materialize_vision_common<VisionConfig>(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2 = artifact::materialized_weight(
            backing, plan.vision_merger_fc2, NumericFormat::W8G32_F16S,
            VisionConfig::output_hidden, VisionConfig::merger_hidden);
        vision.merger_fc2_bias = artifact::materialized_tensor(
            backing, plan.vision_merger_fc2_bias, NumericFormat::BF16,
            {VisionConfig::output_hidden});
    }
}

} // namespace sinfer::targets::qwen3_5_2b::detail
