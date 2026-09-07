#include "artifact/reader.h"

#include <limits>

namespace sinfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, or I32");
    }
}

} // namespace

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::FP8_E4M3FN_BLK128_F32S:
        return "FP8_E4M3FN_BLK128_F32S";
    case NumericFormat::FP8_E4M3FN_ROW_F32S:
        return "FP8_E4M3FN_ROW_F32S";
    case NumericFormat::Q2_K:
        return "Q2_K";
    case NumericFormat::Q3_K:
        return "Q3_K";
    case NumericFormat::Q4_K:
        return "Q4_K";
    case NumericFormat::Q5_K:
        return "Q5_K";
    case NumericFormat::Q6_K:
        return "Q6_K";
    case NumericFormat::Q8_0:
        return "Q8_0";
    case NumericFormat::Q4_1:
        return "Q4_1";
    case NumericFormat::Q5_1:
        return "Q5_1";
    case NumericFormat::IQ4_NL:
        return "IQ4_NL";
    case NumericFormat::Q4_0:
        return "Q4_0";
    case NumericFormat::Q5_0:
        return "Q5_0";
    case NumericFormat::IQ2_XXS:
        return "IQ2_XXS";
    case NumericFormat::IQ2_XS:
        return "IQ2_XS";
    case NumericFormat::IQ2_S:
        return "IQ2_S";
    case NumericFormat::IQ3_XXS:
        return "IQ3_XXS";
    case NumericFormat::IQ3_S:
        return "IQ3_S";
    case NumericFormat::IQ1_S:
        return "IQ1_S";
    case NumericFormat::IQ1_M:
        return "IQ1_M";
    case NumericFormat::IQ4_XS:
        return "IQ4_XS";
    case NumericFormat::TQ1_0:
        return "TQ1_0";
    case NumericFormat::TQ2_0:
        return "TQ2_0";
    case NumericFormat::MXFP4:
        return "MXFP4";
    case NumericFormat::NVFP4_GGML:
        return "NVFP4_GGML";
    case NumericFormat::Q1_0:
        return "Q1_0";
    case NumericFormat::Q2_0:
        return "Q2_0";
    case NumericFormat::F16:
        return "F16";
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1:
        return "row-scale-v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

/// Values per stored block: a K-quant or IQ superblock is 256, the plain block types are 32 --
/// except NVFP4 (64 under four sub-scales) and Q1_0 (128 under one scale).
std::uint64_t ggml_block_values(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q8_0:
    case NumericFormat::Q4_1:
    case NumericFormat::Q5_1:
    case NumericFormat::IQ4_NL:
    case NumericFormat::Q4_0:
    case NumericFormat::Q5_0:
    case NumericFormat::MXFP4: return 32;
    case NumericFormat::NVFP4_GGML: return 64;
    case NumericFormat::Q1_0: return 128;
    case NumericFormat::Q2_0: return 64;
    case NumericFormat::F16: return 32;
    default: return 256;
    }
}

std::uint64_t ggml_block_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q2_K: return 84;
    case NumericFormat::Q3_K: return 110;
    case NumericFormat::Q4_K: return 144;
    case NumericFormat::Q5_K: return 176;
    case NumericFormat::Q6_K: return 210;
    case NumericFormat::Q8_0: return 34;
    case NumericFormat::Q4_1: return 20;
    case NumericFormat::Q5_1: return 24;
    case NumericFormat::IQ4_NL: return 18;
    case NumericFormat::Q4_0: return 18;
    case NumericFormat::Q5_0: return 22;
    case NumericFormat::IQ2_XXS: return 66;
    case NumericFormat::IQ2_XS: return 74;
    case NumericFormat::IQ2_S: return 82;
    case NumericFormat::IQ3_XXS: return 98;
    case NumericFormat::IQ3_S: return 110;
    case NumericFormat::IQ1_S: return 50;
    case NumericFormat::IQ1_M: return 56;
    case NumericFormat::IQ4_XS: return 136;
    case NumericFormat::TQ1_0: return 54;
    case NumericFormat::TQ2_0: return 66;
    case NumericFormat::MXFP4: return 17;
    case NumericFormat::NVFP4_GGML: return 36;
    case NumericFormat::Q1_0: return 18;
    case NumericFormat::Q2_0: return 18;
    case NumericFormat::F16: return 64;
    default: break;
    }
    throw ArtifactError("format is not a GGML superblock format");
}

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape) {
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        return block_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleV1) {
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScale128Fp8V1) {
        return block_scale128_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleF32V1) {
        return row_scale_f32_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::GgmlBlocksV1) {
        const auto values = ggml_block_values(format);
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0 || (shape[1] % values) != 0) {
            throw ArtifactError(
                "ggml-blocks-v1 requires a rank-two shape with k a whole number of blocks");
        }
        const auto blocks = checked_mul(shape[0], shape[1] / values, "ggml block count");
        return checked_mul(blocks, ggml_block_bytes(format), "ggml encoded size");
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

BlockScale128Geometry block_scale128_geometry(NumericFormat format,
                                              std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_BLK128_F32S) {
        throw ArtifactError("block-scale-128-fp8-v1 requires FP8_E4M3FN_BLK128_F32S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0 || (shape[0] % 128) != 0 ||
        (shape[1] % 128) != 0) {
        throw ArtifactError("block-scale-128-fp8-v1 requires a rank-two shape of whole 128-blocks");
    }
    BlockScale128Geometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 block scale plane offset");
    out.scale_plane_bytes = checked_mul(checked_mul(out.rows / 128, out.columns / 128, "FP8 block count"),
                                        4, "FP8 block scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 block tensor encoded size");
    return out;
}

BlockScale128Geometry row_scale_f32_geometry(NumericFormat format,
                                             std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_ROW_F32S) {
        throw ArtifactError("row-scale-f32-v1 requires FP8_E4M3FN_ROW_F32S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0 || (shape[1] % 128) != 0) {
        throw ArtifactError("row-scale-f32-v1 requires a rank-two shape with k a whole number of 128s");
    }
    BlockScale128Geometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset = align_up(out.code_plane_bytes, kTensorAlignment, "FP8 row scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, 4, "FP8 row scale plane bytes");
    out.encoded_bytes = checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 row tensor encoded size");
    return out;
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
        throw ArtifactError("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-scale-v1 requires a positive rank-two shape");
    }

    RowScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, 2, "FP8 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 tensor encoded size");
    return out;
}

} // namespace sinfer::artifact
