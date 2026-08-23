#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void set_i32_scalar_kernel(std::int32_t* destination, std::int32_t value) {
    destination[0] = value;
}

__global__ void add_i32_scalars_kernel(const std::int32_t* lhs, const std::int32_t* rhs,
                                       std::int32_t* destination) {
    destination[0] = lhs[0] + rhs[0];
}

__global__ void increment_i32_scalar_kernel(std::int32_t* scalar) { ++scalar[0]; }

__global__ void increment_i64_scalar_kernel(std::int64_t* scalar) { ++scalar[0]; }

} // namespace ninfer::ops
