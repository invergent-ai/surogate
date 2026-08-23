#include "ninfer/ops/scalar.h"

#include "ops/launcher/scalar.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_scalar(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype || tensor.ne[0] != 1 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be a non-null contiguous scalar");
    }
}

} // namespace

void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    require_scalar(destination, DType::I32, "set_i32_scalar destination");
    detail::set_i32_scalar_launch(destination, value, stream);
}

void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    require_scalar(source, DType::I32, "assign_i32_scalar source");
    require_scalar(destination, DType::I32, "assign_i32_scalar destination");
    if (source.data == destination.data) {
        throw std::invalid_argument("assign_i32_scalar: source and destination must not alias");
    }
    detail::assign_i32_scalar_launch(source, destination, stream);
}

void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     cudaStream_t stream) {
    require_scalar(lhs, DType::I32, "add_i32_scalars lhs");
    require_scalar(rhs, DType::I32, "add_i32_scalars rhs");
    require_scalar(destination, DType::I32, "add_i32_scalars destination");
    if (destination.data == lhs.data || destination.data == rhs.data) {
        throw std::invalid_argument("add_i32_scalars: inputs must not alias destination");
    }
    detail::add_i32_scalars_launch(lhs, rhs, destination, stream);
}

void increment_i32_scalar(Tensor& scalar, cudaStream_t stream) {
    require_scalar(scalar, DType::I32, "increment_i32_scalar scalar");
    detail::increment_i32_scalar_launch(scalar, stream);
}

void increment_i64_scalar(Tensor& scalar, cudaStream_t stream) {
    require_scalar(scalar, DType::I64, "increment_i64_scalar scalar");
    detail::increment_i64_scalar_launch(scalar, stream);
}

} // namespace ninfer::ops
