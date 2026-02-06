// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Factory function for creating IQuantizer instances.

#include "runtime/qlora/generic_quantizer.h"

#include <stdexcept>

#include "runtime/qlora/bnb_quantizer.h"
#include "runtime/qlora/fp8_quantizer.h"
#include "runtime/qlora/fp4_quantizer.h"
#include "kernels/kernels.h"

namespace qlora {

std::unique_ptr<IQuantizer> create_quantizer(const QuantizerConfig& config) {
    switch (config.format) {
        case QuantFormat::BNB_NF4:
            return std::make_unique<BnBQuantizer>(config);

        case QuantFormat::FP8_PER_BLOCK: {
            // FP8 requires SM89+ (Ada Lovelace / RTX 40xx) for native FP8 operations.
            // The per-block quantization kernels work on any GPU, but performance
            // is best on SM89+ where hardware FP8 instructions are available.
            if (config.sm_version > 0 && config.sm_version < 89) {
                throw std::runtime_error(
                    "FP8 per-block quantization requires SM89+ (Ada Lovelace or newer). "
                    "Detected SM" + std::to_string(config.sm_version) + ".");
            }
            return std::make_unique<FP8Quantizer>(config);
        }

        case QuantFormat::FP4_BLOCK_2D: {
            // FP4 requires SM100+ (Blackwell) for native PTX fp4 instructions.
            if (config.sm_version > 0 && config.sm_version < 100) {
                throw std::runtime_error(
                    "FP4 block quantization requires SM100+ (Blackwell or newer). "
                    "Detected SM" + std::to_string(config.sm_version) + ".");
            }
            if (!device_supports_fp4()) {
                throw std::runtime_error(
                    "FP4 block quantization is not supported on this device. "
                    "Requires Blackwell (SM100+) GPU with CUDA 12.8+.");
            }
            return std::make_unique<FP4Quantizer>(config);
        }

        case QuantFormat::NONE:
            return nullptr;

        default:
            throw std::runtime_error(
                "Unknown quantization format: " + std::to_string(static_cast<int>(config.format)));
    }
}

}  // namespace qlora
