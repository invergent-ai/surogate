#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace sinfer {

struct LinearAttentionStatePoolSpec {
    std::uint32_t layers        = 0;
    std::int32_t conv_channels  = 0;
    std::int32_t conv_width     = 0;
    std::int32_t value_heads    = 0;
    std::int32_t value_head_dim = 0;
    std::int32_t key_head_dim   = 0;
    std::int32_t slot_count     = 1;
    DType conv_dtype            = DType::BF16;

    /// Whether the mixer carries a recurrent state beside its convolution. A gated delta net
    /// does -- that matrix is the mixer. A short convolution does not: its whole memory is the
    /// K-1 columns behind the round, and the three head dimensions are left at zero to say so,
    /// which costs the pool nothing rather than a zero-sized region per layer.
    [[nodiscard]] constexpr bool has_recurrent() const noexcept { return value_heads > 0; }
};

struct LinearAttentionStatePoolLayout {
    LinearAttentionStatePoolSpec spec;

    std::vector<LayoutRegion> conv;
    std::vector<LayoutRegion> recurrent;
};

struct LinearAttentionStateAllLayersView {
    Tensor conv_layer0;
    Tensor recurrent_layer0;
    std::int64_t conv_layer_stride_bytes      = 0;
    std::int64_t recurrent_layer_stride_bytes = 0;
    LinearAttentionStatePoolSpec spec;
};

[[nodiscard]] LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec);

/**
 * Fixed-capacity physical storage for model-level Linear Attention state images.
 *
 * One logical slot selects the same frontier across every layer's convolution and recurrent
 * component. The pool owns no slot roles, validity, request metadata, allocation policy, or CUDA
 * stream. Construction binds caller-owned backing without mutating it.
 */
struct LinearAttentionStatePool {
    std::vector<Tensor> conv;
    std::vector<Tensor> recurrent;
    LinearAttentionStatePoolSpec spec;
    // Optional single-slot views outside the resident pool. Capture/restore uses
    // these host accessors; batched kernels only address the resident slots.
    std::vector<const LinearAttentionStatePool*> checkpoint_slots;

    LinearAttentionStatePool() = default;
    LinearAttentionStatePool(DeviceSpan backing, const LinearAttentionStatePoolLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept;
    [[nodiscard]] std::int32_t slot_count() const noexcept;
    [[nodiscard]] std::int64_t conv_slot_stride_elements() const noexcept;
    [[nodiscard]] std::int64_t recurrent_slot_stride_elements() const noexcept;
    [[nodiscard]] LinearAttentionStateAllLayersView all_layers_view() const;
    [[nodiscard]] Tensor conv_slot(std::uint32_t layer, std::int32_t slot) const;
    [[nodiscard]] Tensor recurrent_slot(std::uint32_t layer, std::int32_t slot) const;

    void copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream = nullptr);
    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
};

} // namespace sinfer
