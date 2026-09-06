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

/// The placement `bind_device_tensor` and a default-placement `bind_linear` use, for the scope
/// of this object. Device unless something says otherwise, which is what makes host residency
/// available to every target that binds weights the ordinary way -- including one written after
/// this -- rather than a feature each has to grow.
///
/// Thread-scoped, because a pipeline builds its stages on one thread each.
class ScopedPlacement {
public:
    explicit ScopedPlacement(TensorPlacement placement) noexcept;
    ~ScopedPlacement();
    ScopedPlacement(const ScopedPlacement&)            = delete;
    ScopedPlacement& operator=(const ScopedPlacement&) = delete;
    ScopedPlacement(ScopedPlacement&&)                 = delete;
    ScopedPlacement& operator=(ScopedPlacement&&)      = delete;

    /// What is in force now.
    [[nodiscard]] static TensorPlacement current() noexcept;

private:
    TensorPlacement previous_;
};

/// Bind at the placement in force (`ScopedPlacement`), which is Device unless a scope changed
/// it. The name is the common case, not a promise: a target that offloads a layer wraps its
/// binding and every call below follows.
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

/// Omitting the placement takes the one in force (`ScopedPlacement`), which is Device unless a
/// scope changed it -- so a target that offloads a layer needs no change at the call site.
[[nodiscard]] LinearBinding bind_linear(Binder& binder, std::string_view name, std::int32_t rows,
                                        std::int32_t columns);
[[nodiscard]] LinearBinding bind_linear(Binder& binder, std::string_view name, std::int32_t rows,
                                        std::int32_t columns, TensorPlacement placement);

/// The Weight for a LinearBinding: the existing constructors for every format
/// but NVFP4, and the NVFP4 one -- previously re-implemented in each target
/// that served the format -- here.
[[nodiscard]] Weight materialized_linear(const MaterializedArtifact& materialized,
                                         const LinearBinding& binding, std::int32_t rows,
                                         std::int32_t columns);

/// Every stored format `materialized_linear` can produce a Weight for.
[[nodiscard]] bool is_linear_format(NumericFormat format) noexcept;

} // namespace sinfer::artifact
