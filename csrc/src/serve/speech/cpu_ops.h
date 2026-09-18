// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>

namespace sinfer::speech::cpu {
// One packed shape per immutable weight matrix. Recurring shapes can replace it;
// occasional misses use ordinary GEMM without discarding the retained layout.
class PackedLinear {
    at::Tensor packed_, original_;
    int64_t rows_ = 0;
    int threads_  = 0;
    at::Tensor pending_original_;
    int64_t pending_rows_ = 0;
    int pending_threads_ = 0, observations_ = 0;
public:
    at::Tensor run(const at::Tensor& input, const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias);
};

at::Tensor relative_shift(const at::Tensor& scores, int64_t keys);
} // namespace sinfer::speech::cpu
