#pragma once
// The Inkling-2 vision tower's CUDA leaves (inkling_vision_ops.cu). Kept apart from inkling_vision.h so the .cu does
// not include the vision API headers, whose brace-initialised widths nvcc refuses as narrowing.
#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace sinfer::family {
// positions I32 [T, 2]: axis 0 = x + 1 (the first head_dim / 4 pairs), axis 1 = y + 1 (the rest); the CLS row 0, 0.
// Interleaved pairs (2j, 2j + 1), inv_freq_j = fp32(10000^(-4j / head_dim)), fp32 cos / sin, one rounding to BF16.
void inkling_vision_rope(const Tensor& positions, Tensor& q, Tensor& k, cudaStream_t stream);
// x = bf16(x + bf16(update * gamma)) per column; gamma BF16 [D].
void inkling_vision_scaled_residual(const Tensor& update, const Tensor& gamma, Tensor& x, cudaStream_t stream);
// out [D] = bf16(class_embedding + cls_position_embedding) in fp32.
void inkling_vision_cls(const Tensor& class_embedding, const Tensor& position, Tensor& out, cudaStream_t stream);
// dst column indices[v], elements [0, D) = src column v, where dst's columns step by `column_elements` (Inkling-2's
// AttnRes stack: slot 0 of a 7 x D column). src BF16 [D, V] contiguous, indices I32 [V].
void inkling_scatter_columns(const Tensor& src, const Tensor& indices, void* dst, std::int64_t column_elements,
                             cudaStream_t stream);
} // namespace sinfer::family
