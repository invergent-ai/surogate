#include "ops/launcher/scalar.h"

#include "core/device.h"
#include "ops/kernel/scalar.cuh"

namespace ninfer::ops::detail {

void set_i32_scalar_launch(Tensor& destination, std::int32_t value, cudaStream_t stream) {
    set_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(destination.data), value);
    CUDA_CHECK(cudaGetLastError());
}

void assign_i32_scalar_launch(const Tensor& source, Tensor& destination, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void add_i32_scalars_launch(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                            cudaStream_t stream) {
    add_i32_scalars_kernel<<<1, 1, 0, stream>>>(static_cast<const std::int32_t*>(lhs.data),
                                                static_cast<const std::int32_t*>(rhs.data),
                                                static_cast<std::int32_t*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

void increment_i32_scalar_launch(Tensor& scalar, cudaStream_t stream) {
    increment_i32_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(scalar.data));
    CUDA_CHECK(cudaGetLastError());
}

void increment_i64_scalar_launch(Tensor& scalar, cudaStream_t stream) {
    increment_i64_scalar_kernel<<<1, 1, 0, stream>>>(static_cast<std::int64_t*>(scalar.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
