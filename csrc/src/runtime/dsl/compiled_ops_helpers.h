// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Helper functions for compiled operation dispatch (logging, debugging, etc).

#ifndef SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H
#define SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H

#include <cstddef>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "utilities/tensor.h"

namespace dsl {

// Global state for QKV gradient tracking (shared across split op files)
extern std::vector<std::byte*> g_qkv_dA_ptr_by_layer;
extern std::vector<int> g_qkv_dA_micro_by_layer;

// MoE compact weight information
struct MoeCompactInfo {
    std::vector<int> host_offsets;
    std::vector<int> active_experts;
    int num_active = 0;
    bool weight_is_compact = false;
};

// Build MoE compact info from expert offsets (device memory)
MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream,
                                      int layer_idx,
                                      const char* tag);

// Build MoE compact info from expert offsets (host memory)
MoeCompactInfo build_moe_compact_info_from_host(const int* host_offsets,
                                                int num_experts,
                                                int weight_experts,
                                                int layer_idx,
                                                const char* tag);

int env_int(const char* name, int fallback);
float env_float(const char* name, float fallback);

// Build a Tensor wrapping a raw GPU pointer with proper Rank/Device set.
// IMPORTANT: Manual `Tensor{}` leaves Rank=0, Device=-1 which makes .bytes()
// and .nelem() return wrong values. Always use this helper instead.
inline Tensor make_raw_tensor(void* ptr, ETensorDType dtype,
                              const std::vector<long>& shape, int device) {
    Tensor t{};
    t.Data = static_cast<std::byte*>(ptr);
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    t.Device = device;
    for (int i = 0; i < t.Rank; ++i) t.Sizes[i] = shape[i];
    for (int i = t.Rank; i < MAX_TENSOR_DIM; ++i) t.Sizes[i] = 1;
    return t;
}

bool refresh_moe_experts_if_needed(int layer_idx,
                                   const int* host_offsets,
                                   int num_experts,
                                   DslParamStore& weights,
                                   cudaStream_t stream);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_COMPILED_OPS_HELPERS_H
