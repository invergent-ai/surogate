#include "family/impl/load/host_decode.h"
#include "ops/linear/ggml/ggml_host_decode.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sinfer::family {
namespace {
using artifact::NumericFormat;

template <class T> T read(const std::byte* data) {
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}
float half(const std::byte* data) { return __half2float(std::bit_cast<__half>(read<std::uint16_t>(data))); }
float bf16(const std::byte* data) { return std::bit_cast<float>(std::uint32_t{read<std::uint16_t>(data)} << 16); }
float e4m3(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int fraction = bits & 7;
    const float value = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -9)
                                     : std::ldexp(1.F + fraction / 8.F, exponent - 7);
    return bits & 128 ? -value : value;
}
const std::byte* span_at(const HostObjectPlan& source, std::uint64_t offset, std::size_t count) {
    if (source.parts.empty()) {
        if (offset + count > source.payload.size()) { throw std::logic_error("host row exceeds its tensor"); }
        return source.payload.data() + offset;
    }
    for (const auto part : source.parts) {
        if (offset >= part.size()) { offset -= part.size(); continue; }
        if (offset + count > part.size()) { throw std::logic_error("host row crosses payload runs"); }
        return part.data() + offset;
    }
    throw std::logic_error("host row exceeds payload runs");
}
} // namespace

void decode_host_row(const HostObjectPlan& source, std::int64_t row, float* output) {
    const auto& tensor = source.source_tensor;
    const int k = source.decode_k;
    if (source.swap_half_rows != 0) {
        const auto half_rows = source.swap_half_rows;
        row = row / (2 * half_rows) * (2 * half_rows) + (row + half_rows) % (2 * half_rows);
    }
    const auto original_row = row;
    NumericFormat format = tensor.format;
    std::uint64_t offset = 0;
    for (const auto& segment : tensor.segments) {
        if (row < static_cast<std::int64_t>(segment.rows)) { format = segment.format; break; }
        const std::array<std::uint64_t, 2> shape = {segment.rows, static_cast<std::uint64_t>(k)};
        offset += artifact::tensor_encoded_size(tensor.layout, segment.format, shape);
        row -= segment.rows;
    }
    const QType type = tensor.transform == artifact::PayloadTransform::Q8ToW8RowSplit
                          ? QType::Q8_0 : artifact::qtype_for(format);
    const auto row_bytes = ops::ggml_row_bytes(type, k);
    if (row_bytes > 0) {
        if (!ops::ggml_decode_row_float(type, span_at(source, offset + row * row_bytes, row_bytes), k, output)) {
            throw std::logic_error("host row decoder does not support " + std::string(artifact::format_name(format)));
        }
    } else if (tensor.layout == artifact::StorageLayout::RowSplitK128V1) {
        const auto layout = artifact::row_split_geometry(format, tensor.shape);
        const auto* bytes = span_at(source, 0, layout.encoded_bytes);
        const int bits = format == NumericFormat::Q4G64_F16S ? 4 :
                         format == NumericFormat::Q5G64_F16S ? 5 :
                         format == NumericFormat::Q6G64_F16S ? 6 : 8;
        for (int c = 0; c < k; ++c) {
            const auto group = row * layout.groups_per_row + c / layout.group_size;
            const auto within = c % layout.group_size;
            int code;
            if (bits == 8) { code = read<std::int8_t>(bytes + group * 32 + within); }
            else {
                code = (std::to_integer<unsigned>(bytes[group * 32 + within / 2]) >> ((within % 2) * 4)) & 15;
                if (bits > 4) {
                    const auto bit = within * (bits - 4);
                    code |= ((std::to_integer<unsigned>(bytes[layout.high_plane_offset +
                              group * layout.high_bytes_per_group + bit / 8]) >> (bit % 8)) & ((1 << (bits - 4)) - 1)) << 4;
                }
                if (code & (1 << (bits - 1))) { code -= 1 << bits; }
            }
            output[c] = code * half(bytes + layout.scale_plane_offset + group * 2);
        }
    } else if (format == NumericFormat::NVFP4) {
        const auto layout = artifact::block_scale_geometry(format, tensor.shape);
        const auto* bytes = span_at(source, 0, layout.encoded_bytes);
        constexpr float levels[8] = {0.F, .5F, 1.F, 1.5F, 2.F, 3.F, 4.F, 6.F};
        const float divisor = read<float>(bytes + layout.weight_divisor_offset);
        for (int c = 0; c < k; ++c) {
            const unsigned code = (std::to_integer<unsigned>(bytes[row * (k / 2) + c / 2]) >> ((c % 2) * 4)) & 15;
            const auto group = c / 16;
            const auto scale = (row / 128 * layout.k_tiles + group / 4) * 512 +
                               (row % 32) * 16 + (row % 128 / 32) * 4 + group % 4;
            output[c] = levels[code & 7] * (code & 8 ? -1.F : 1.F) *
                         e4m3(std::to_integer<std::uint8_t>(bytes[layout.scale_plane_offset + scale])) / divisor;
        }
    } else if (format == NumericFormat::FP32) {
        std::memcpy(output, span_at(source, row * k * 4, k * 4), k * 4);
    } else if (format == NumericFormat::FP8_E4M3FN_ROW_BF16S ||
               format == NumericFormat::FP8_E4M3FN_ROW_F32S ||
               format == NumericFormat::FP8_E4M3FN_BLK128_F32S) {
        const bool block = format == NumericFormat::FP8_E4M3FN_BLK128_F32S;
        const bool bf = format == NumericFormat::FP8_E4M3FN_ROW_BF16S;
        const auto scales_offset = bf ? artifact::row_scale_geometry(format, tensor.shape).scale_plane_offset
            : block ? artifact::block_scale128_geometry(format, tensor.shape).scale_plane_offset
                    : artifact::row_scale_f32_geometry(format, tensor.shape).scale_plane_offset;
        const auto* codes = span_at(source, row * k, k);
        for (int c = 0; c < k; ++c) {
            const auto scale_index = block ? row / 128 * (k / 128) + c / 128 : row;
            const auto* scale = span_at(source, scales_offset + scale_index * (bf ? 2 : 4), bf ? 2 : 4);
            output[c] = e4m3(std::to_integer<std::uint8_t>(codes[c])) * (bf ? bf16(scale) : read<float>(scale));
        }
    } else {
        throw std::invalid_argument("cannot decode host expert format " + std::string(artifact::format_name(format)));
    }
    if (!source.row_scales.empty()) {
        const float scale = source.row_scales.at(original_row / source.scale_rows);
        for (int c = 0; c < k; ++c) { output[c] *= scale; }
    }
    if (!tensor.group_map.empty()) {
        const std::vector<float> stored(output, output + k);
        for (int group = 0; group < k / 32; ++group) {
            std::copy_n(stored.data() + tensor.group_map.at(group) * 32, 32, output + group * 32);
        }
    }
}

void quantize_host_row(const float* values, std::int32_t columns,
                       std::int8_t* codes, std::uint16_t* scales) {
    for (int group = 0; group < columns / 32; ++group) {
        float maximum = 0.F;
        for (int i = 0; i < 32; ++i) { maximum = std::max(maximum, std::fabs(values[group * 32 + i])); }
        const float scale = maximum / 127.F;
        const float inverse = scale > 0.F ? 1.F / scale : 0.F;
        scales[group] = std::bit_cast<std::uint16_t>(__float2half_rn(scale));
        for (int i = 0; i < 32; ++i) {
            codes[group * 32 + i] = static_cast<std::int8_t>(std::nearbyint(values[group * 32 + i] * inverse));
        }
    }
}
} // namespace sinfer::family
