// Minimal shim replacing PyTorch's ATen/cuda/CUDAGeneratorImpl.h
// for Flash Attention integration. Only at::PhiloxCudaState is needed
// (used for dropout RNG which we disable with p_dropout=0).
#pragma once
#include <cstdint>
#include <utility>

namespace at {

// flash.h references at::PhiloxCudaState (not at::cuda::PhiloxCudaState)
struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
};

namespace cuda {
namespace philox {

#ifdef __CUDACC__
__host__ __device__
#endif
inline std::pair<uint64_t, uint64_t> unpack(const PhiloxCudaState& s) {
    return {s.seed_, s.offset_};
}

}  // namespace philox
}  // namespace cuda
}  // namespace at
