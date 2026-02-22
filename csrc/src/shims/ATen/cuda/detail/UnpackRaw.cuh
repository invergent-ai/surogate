// Minimal shim replacing PyTorch's ATen/cuda/detail/UnpackRaw.cuh
// for Flash Attention integration. Provides at::cuda::philox::unpack()
// which is used by flash_fwd_kernel.h for dropout RNG (disabled: p_dropout=0).
#pragma once
#include <ATen/cuda/CUDAGeneratorImpl.h>
