#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail {

/// True when the split qkv|z GDN projection can run the fused decode route: both halves are
/// GGML K-quants of the same K, one token, one row of the batch.
bool ggml_gdn_input_decode_admits(const Weight& query_key_value, const Weight& z,
                                  std::int32_t batch, std::int32_t width) noexcept;

/// Activation-plane bytes the fused route needs (the int8 planes, quantised once for both).
std::size_t ggml_gdn_input_decode_workspace_bytes(std::int32_t input_rows) noexcept;

/// The split projection and the causal convolution in one pass over each weight: the qkv half's
/// rows go straight into query/key/value through the shared convolution epilogue, so no
/// projected plane is ever materialised, and the z half writes its own destination.
void ggml_gdn_input_conv_snapshot_decode_launch(
    const Tensor& x, const Weight& query_key_value, const Weight& z_weight,
    const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
    Tensor& value, Tensor& z, void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

/// The record twin: the convolution reads the same state and publishes into `conv_record`.
void ggml_gdn_input_conv_record_decode_launch(
    const Tensor& x, const Weight& query_key_value, const Weight& z_weight,
    const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
    const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
    Tensor& z, void* scratch, std::size_t scratch_bytes, cudaStream_t stream);

} // namespace sinfer::ops::detail
