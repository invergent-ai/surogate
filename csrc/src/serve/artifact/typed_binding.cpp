#include "artifact/typed_binding.h"

#include "artifact/materializer.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <string>
#include <variant>

namespace sinfer::artifact {
namespace {

StorageLayout storage_layout_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return StorageLayout::ContiguousLeV1;
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
        return StorageLayout::RowSplitK128V1;
    case NumericFormat::Q2_K:
    case NumericFormat::Q3_K:
    case NumericFormat::Q4_K:
    case NumericFormat::Q5_K:
    case NumericFormat::Q6_K:
    case NumericFormat::Q8_0:
        return StorageLayout::GgmlBlocksV1;
    case NumericFormat::NVFP4:
        return StorageLayout::BlockScaleK16M128x4V1;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return StorageLayout::RowScaleV1;
    }
    throw std::logic_error("unhandled numeric format");
}

QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return QType::BF16_CTRL;
    case NumericFormat::FP32:
        return QType::FP32_CTRL;
    case NumericFormat::I32:
        return QType::I32_CTRL;
    case NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    case NumericFormat::Q2_K:
        return QType::Q2_K;
    case NumericFormat::Q3_K:
        return QType::Q3_K;
    case NumericFormat::Q4_K:
        return QType::Q4_K;
    case NumericFormat::Q5_K:
        return QType::Q5_K;
    case NumericFormat::Q6_K:
        return QType::Q6_K;
    case NumericFormat::Q8_0:
        return QType::Q8_0;
    case NumericFormat::NVFP4:
        return QType::NVFP4;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return QType::FP8_E4M3FN_ROW_BF16S;
    }
    throw std::logic_error("unhandled numeric format");
}

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return DType::BF16;
    case NumericFormat::FP32:
        return DType::FP32;
    case NumericFormat::I32:
        return DType::I32;
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

Weight contiguous_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload       = materialized.device_data(handle);
    out.qdata         = out.payload;
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

Weight row_split_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry          = row_split_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh       = geometry.high_plane_bytes == 0 ? nullptr : bytes + geometry.high_plane_offset;
    out.scales      = bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

Weight row_scale_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowScaleGeometry geometry          = row_scale_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = qtype_for(format);
    out.layout          = QuantLayout::RowScale;
    out.group_size      = static_cast<std::uint32_t>(geometry.columns);
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = columns;
    out.scale_dtype     = DType::BF16;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 2;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 2;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

} // namespace

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    const ObjectHandle handle =
        binder.require_tensor(name, format, storage_layout_for(format),
                              std::span<const std::uint64_t>(shape.begin(), shape.size()));
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    const ObjectHandle handle = binder.require_resource(name, ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format,
                           std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), dtype_for(format), internal_shape);
}

namespace {

// A K-quant weight is the GGUF's superblock bytes, [rows][columns/256] blocks; nothing is
// derived from them at load. Scale planes do not exist: the sub-scales live inside each block.
Weight ggml_blocks_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                          NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const std::uint64_t bytes = tensor_encoded_size(StorageLayout::GgmlBlocksV1, format, shape);
    const auto* data          = static_cast<const std::byte*>(materialized.device_data(handle));
    Weight out{};
    out.payload         = data;
    out.payload_bytes   = bytes;
    const auto values   = static_cast<std::int32_t>(ggml_block_values(format));
    out.qtype           = qtype_for(format);
    out.group_size      = static_cast<std::uint32_t>(values);
    out.ndim            = 2;
    out.qdata           = data;
    out.qhigh           = nullptr;
    out.scales          = nullptr;
    out.n               = rows;
    out.k               = columns;
    out.group           = values;
    out.layout          = QuantLayout::GgmlBlocks;
    out.scale_dtype     = DType::FP16;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

} // namespace

Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns) {
    if (format == NumericFormat::NVFP4) {
        throw std::invalid_argument(
            "materialized_weight: NVFP4 requires target-validated weight and input divisors");
    }
    if (storage_layout_for(format) == StorageLayout::ContiguousLeV1) {
        return contiguous_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::RowScaleV1) {
        return row_scale_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::GgmlBlocksV1) {
        return ggml_blocks_weight(materialized, handle, format, rows, columns);
    }
    return row_split_weight(materialized, handle, format, rows, columns);
}

namespace {

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::uint64_t offset,
                          std::string_view what) {
    if (offset + 4 > bytes.size()) {
        throw ArtifactError("payload too short for a divisor word: " + std::string(what));
    }
    std::uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
}

void require_positive_finite(std::uint32_t bits, std::string_view what) {
    const float value = std::bit_cast<float>(bits);
    if (!std::isfinite(value) || value <= 0.0F) {
        throw ArtifactError("divisor is not finite and positive: " + std::string(what));
    }
}

} // namespace

bool is_linear_format(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
    case NumericFormat::NVFP4:
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
    case NumericFormat::Q2_K:
    case NumericFormat::Q3_K:
    case NumericFormat::Q4_K:
    case NumericFormat::Q5_K:
    case NumericFormat::Q6_K:
    case NumericFormat::Q8_0:
        return true;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return false;
    }
    return false;
}

LinearBinding bind_linear(Binder& binder, std::string_view name, std::int32_t rows,
                          std::int32_t columns, TensorPlacement placement) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const ObjectHandle handle = binder.require_tensor_shaped(name, shape);
    const auto* tensor        = std::get_if<TensorDescriptor>(&binder.descriptor(handle));
    if (!is_linear_format(tensor->format)) {
        throw ArtifactError("stored format is not one a linear op can run: " +
                            std::string(format_name(tensor->format)) + " for " + std::string(name));
    }
    if (tensor->layout != storage_layout_for(tensor->format)) {
        throw ArtifactError("stored layout does not belong to its format: " + std::string(name));
    }
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    LinearBinding binding{handle, tensor->format};
    if (tensor->format == NumericFormat::NVFP4) {
        const std::string divisor_name = std::string(name) + "/input_scale_divisor";
        const ObjectHandle divisor =
            bind_tensor(binder, divisor_name, NumericFormat::FP32, {}, TensorPlacement::ValidateOnly);
        const BlockScaleGeometry geometry = block_scale_geometry(NumericFormat::NVFP4, shape);
        binding.weight_scale_divisor_bits =
            read_u32_le(binder.payload(handle).data, geometry.weight_divisor_offset, name);
        binding.input_scale_divisor_bits = read_u32_le(binder.payload(divisor).data, 0, divisor_name);
        require_positive_finite(binding.weight_scale_divisor_bits, name);
        require_positive_finite(binding.input_scale_divisor_bits, divisor_name);
    }
    return binding;
}

Weight materialized_linear(const MaterializedArtifact& materialized, const LinearBinding& binding,
                           std::int32_t rows, std::int32_t columns) {
    if (binding.format != NumericFormat::NVFP4) {
        return materialized_weight(materialized, binding.object, binding.format, rows, columns);
    }
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const BlockScaleGeometry geometry = block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(binding.object));

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
    out.weight_scale_divisor = std::bit_cast<float>(binding.weight_scale_divisor_bits);
    out.input_scale_divisor  = std::bit_cast<float>(binding.input_scale_divisor_bits);
    return out;
}

} // namespace sinfer::artifact
