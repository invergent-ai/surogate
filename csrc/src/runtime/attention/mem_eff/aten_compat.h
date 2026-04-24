#pragma once

// Compatibility shim for porting PyTorch's mem_eff_attention kernels
// into the surogate runtime. The cutlass kernel code was authored against
// PyTorch's internals; this header provides the narrow subset of `at::` /
// `c10::` / `TORCH_CHECK` symbols the kernels reference, stubbed for the
// features we don't use (dropout) and forwarded to surogate's own
// diagnostics for the features we do use (compile-time asserts).

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace at {

enum class ScalarType : int {
    Float = 0,
    Half = 1,
    BFloat16 = 2
};

// Minimal stand-in for `at::PhiloxCudaState`. The kernel only consults this
// via `at::cuda::philox::unpack(state)` when dropout is active. surogate's
// training attention never enables dropout, so we compile the dropout
// branch out (kSupportsDropout=false template parameter on every
// kernel instantiation we emit). The struct must still exist because
// `Params::rng_engine_inputs` is an unconditional member.
struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    constexpr PhiloxCudaState() = default;
    constexpr PhiloxCudaState(uint64_t s, uint64_t o)
        : seed_(s),
          offset_(o) {
    }
};

namespace cuda {
namespace philox {

__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(const PhiloxCudaState& s) {
    return std::make_tuple(s.seed_, s.offset_);
}

}  // namespace philox
}  // namespace cuda

}  // namespace at

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::ostringstream _torch_check_oss;                               \
            _torch_check_oss << "TORCH_CHECK failed: " #cond " (";             \
            ::at::detail::torch_check_append(_torch_check_oss, ##__VA_ARGS__); \
            _torch_check_oss << ")";                                           \
            throw std::runtime_error(_torch_check_oss.str());                  \
        }                                                                      \
    } while (0)
#endif

namespace at {
namespace detail {

inline void torch_check_append(std::ostringstream&) {
}

template <typename T, typename... Rest>
inline void torch_check_append(std::ostringstream& oss, T&& first, Rest&&... rest) {
    oss << first;
    torch_check_append(oss, std::forward<Rest>(rest)...);
}

}  // namespace detail
}  // namespace at
