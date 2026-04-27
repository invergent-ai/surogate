// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_KERNELS_MOE_MOE_FP8_WGRAD_CUTLASS_H
#define SUROGATE_SRC_KERNELS_MOE_MOE_FP8_WGRAD_CUTLASS_H

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

void moe_grouped_gemm_weight_grad_fp8_cutlass(nv_bfloat16* d_weight,
                                              const __nv_fp8_e5m2* grad_output,
                                              const __nv_fp8_e4m3* input,
                                              int num_experts,
                                              int M,
                                              int N,
                                              cudaStream_t stream,
                                              const int* host_offsets,
                                              float alpha,
                                              float beta,
                                              const int* active_expert_indices,
                                              bool weight_is_compact,
                                              int num_active_experts);

#endif  // SUROGATE_SRC_KERNELS_MOE_MOE_FP8_WGRAD_CUTLASS_H
