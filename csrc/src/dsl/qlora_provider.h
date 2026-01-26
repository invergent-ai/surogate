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

#include "utilities/tensor.h"

class NCCLCommunicator;

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

    /// Total bytes used by quantized weights (for memory stats).
    virtual std::size_t quantized_weights_bytes() const = 0;

    /// Memory savings ratio vs BF16 base weights (for stats).
    virtual float memory_savings_ratio() const = 0;
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_QLORA_PROVIDER_H
