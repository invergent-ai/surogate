#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace sinfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

/// A linear weight bound in whatever format the artifact stores it.
///
/// The functions above assert the format a target was compiled for; this one
/// reads it. Structure -- that the object exists and has the shape the
/// geometry implies -- is still the target's to assert; format is the
/// checkpoint's to declare, and a quantized export declares it per module.
/// For NVFP4 the two global scales come along: the weight's from the tail of
/// its own payload, the activation's from the sibling object
/// `<name>/input_scale_divisor`, both validated finite and positive here so
/// no kernel ever sees a divisor it cannot use.
struct LinearBinding {
    ObjectHandle object;
    NumericFormat format;
    std::uint32_t weight_scale_divisor_bits = 0;
    std::uint32_t input_scale_divisor_bits  = 0;
};

/// The runtime quantisation type a stored format is read as.
[[nodiscard]] QType qtype_for(NumericFormat format);

[[nodiscard]] LinearBinding bind_linear(Binder& binder, std::string_view name, std::int32_t rows,
                                        std::int32_t columns,
                                        TensorPlacement placement = TensorPlacement::Device);

/// The Weight for a LinearBinding: the existing constructors for every format
/// but NVFP4, and the NVFP4 one -- previously re-implemented in each target
/// that served the format -- here.
[[nodiscard]] Weight materialized_linear(const MaterializedArtifact& materialized,
                                         const LinearBinding& binding, std::int32_t rows,
                                         std::int32_t columns);

/// Every stored format `materialized_linear` can produce a Weight for.
[[nodiscard]] bool is_linear_format(NumericFormat format) noexcept;

} // namespace sinfer::artifact
