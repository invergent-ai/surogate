// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL QLoRA weight provider interface (runtime weight resolution).

#ifndef SUROGATE_SRC_DSL_QLORA_PROVIDER_H
#define SUROGATE_SRC_DSL_QLORA_PROVIDER_H

#include <cstddef>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "runtime/moe/moe_types.h"

#include "utilities/tensor.h"

class NCCLCommunicator;

namespace qlora {
struct QuantizedTensor;
class IQuantizer;
}  // namespace qlora

namespace dsl {

/**
 * @brief Abstract interface for resolving QLoRA-managed weights in DSL runtime.
 *
 * Implementations provide dequantized (or native) weights on-demand
 * for named DSL parameters and manage quantized storage internally.
 */
class QLoRAWeightProvider {
public:
    virtual ~QLoRAWeightProvider() = default;

    /// Return true if the provider can supply this parameter name.
    virtual bool handles_param(std::string_view name) const = 0;

    /// Resolve a parameter tensor (dequantize if needed).
    virtual Tensor& resolve_param(std::string_view name, cudaStream_t stream) = 0;

    /// Import weights from checkpoint and quantize base weights.
    virtual void import_and_quantize(const std::string& file_name, NCCLCommunicator& comm,
                                     cudaStream_t stream) = 0;

    /// Invalidate any cached dequantized weights (call per-step).
    virtual void invalidate_cache() = 0;

    /// Refresh MoE expert buffers for selective dequant (returns true if handled).
    virtual bool refresh_moe_experts(int layer_idx,
                                     const modules::SelectiveExpertInfo& selection,
                                     cudaStream_t stream) {
        (void)layer_idx;
        (void)selection;
        (void)stream;
        return false;
    }

    /// Returns true if provider supports selective MoE expert refresh.
    [[nodiscard]] virtual bool supports_selective_moe() const { return false; }

    /// Prefetch offloaded weights for a given layer to GPU.
    ///
    /// Called by the graph executor ahead of layer computation to overlap
    /// weight transfers with compute on the current layer. Implementations
    /// that use group-based offloading should map the layer index to offload
    /// group IDs and initiate async prefetching.
    ///
    /// @param layer_idx  Layer index to prefetch weights for
    /// @param stream     CUDA stream for async prefetch transfers
    virtual void prefetch_for_layer(int layer_idx, cudaStream_t stream) {
        (void)layer_idx;
        (void)stream;
    }

    /// Returns true if this provider uses weight offloading (CPU↔GPU).
    /// Used by the graph executor to enable prefetch scheduling.
    [[nodiscard]] virtual bool has_offloading() const { return false; }

    /// Total bytes used by quantized weights (for memory stats).
    virtual std::size_t quantized_weights_bytes() const = 0;

    /// Memory savings ratio vs BF16 base weights (for stats).
    virtual float memory_savings_ratio() const = 0;

    /// Get quantized tensor for a parameter (no dequantization).
    /// Returns nullptr if the parameter is not quantized or not found.
    virtual const qlora::QuantizedTensor* try_get_quantized(std::string_view name) const {
        (void)name;
        return nullptr;
    }

    /// Get the quantizer used by this provider (for remote dequantization).
    /// Returns nullptr if no quantizer is available.
    virtual qlora::IQuantizer* get_quantizer() const { return nullptr; }

    /// Auto-tune offloading parameters based on current GPU memory state.
    /// Called after import_and_quantize() to maximize resident groups.
    virtual void auto_tune_offloading() {}
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_QLORA_PROVIDER_H
