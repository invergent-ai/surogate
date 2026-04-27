// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H
#define SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "runtime/executor/op_descriptor_types.h"

namespace recipes {

inline bool recipe_capability_fallback_log_enabled() {
    const char* env = std::getenv("SUROGATE_RECIPE_CAPABILITY_LOG");
    return env && std::string_view(env) != "0";
}

inline bool descriptor_allows_capability(dsl::OpCapabilities caps, std::uint32_t flag) {
    return caps.flags == dsl::OpCapabilityNone || caps.has(flag);
}

inline bool descriptor_allows_capability(dsl::OpCapabilities caps,
                                         std::uint32_t flag,
                                         const char* context,
                                         const char* capability) {
    if (caps.flags == dsl::OpCapabilityNone) {
        if (recipe_capability_fallback_log_enabled()) {
            std::fprintf(stderr,
                         "[recipe capability] %s legacy allow: descriptor has no capability metadata for %s\n",
                         context,
                         capability);
        }
        return true;
    }
    const bool allowed = caps.has(flag);
    if (!allowed && recipe_capability_fallback_log_enabled()) {
        std::fprintf(stderr, "[recipe capability] %s disabled: descriptor lacks %s\n", context, capability);
    }
    return allowed;
}

inline bool descriptor_allows_moe_capability(dsl::MoECapabilities caps, std::uint32_t flag) {
    return caps.flags == dsl::MoECapabilityNone || caps.has(flag);
}

inline bool descriptor_allows_matmul_capability(dsl::MatmulCapabilities caps, std::uint32_t flag) {
    return caps.flags == dsl::MatmulCapabilityNone || caps.has(flag);
}

inline bool descriptor_allows_moe_capability(dsl::MoECapabilities caps,
                                             std::uint32_t flag,
                                             const char* context,
                                             const char* capability) {
    if (caps.flags == dsl::MoECapabilityNone) {
        if (recipe_capability_fallback_log_enabled()) {
            std::fprintf(stderr,
                         "[recipe capability] %s legacy allow: descriptor has no MoE capability metadata for %s\n",
                         context,
                         capability);
        }
        return true;
    }
    const bool allowed = caps.has(flag);
    if (!allowed && recipe_capability_fallback_log_enabled()) {
        std::fprintf(stderr, "[recipe capability] %s disabled: MoE descriptor lacks %s\n", context, capability);
    }
    return allowed;
}

inline bool descriptor_allows_matmul_capability(dsl::MatmulCapabilities caps,
                                                std::uint32_t flag,
                                                const char* context,
                                                const char* capability) {
    if (caps.flags == dsl::MatmulCapabilityNone) {
        if (recipe_capability_fallback_log_enabled()) {
            std::fprintf(stderr,
                         "[recipe capability] %s legacy allow: descriptor has no matmul capability metadata for %s\n",
                         context,
                         capability);
        }
        return true;
    }
    const bool allowed = caps.has(flag);
    if (!allowed && recipe_capability_fallback_log_enabled()) {
        std::fprintf(stderr, "[recipe capability] %s disabled: matmul descriptor lacks %s\n", context, capability);
    }
    return allowed;
}

inline bool descriptor_allows_fp8(dsl::OpCapabilities caps) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp8Eligible);
}

inline bool descriptor_allows_fp8(dsl::OpCapabilities caps, const char* context) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp8Eligible, context, "FP8Eligible");
}

inline bool descriptor_allows_fp4(dsl::OpCapabilities caps) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp4Eligible);
}

inline bool descriptor_allows_fp4(dsl::OpCapabilities caps, const char* context) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp4Eligible, context, "FP4Eligible");
}

inline bool descriptor_allows_moe_fp8_grouped(dsl::MoECapabilities caps) {
    return descriptor_allows_moe_capability(caps, dsl::MoECapabilityFp8GroupedEligible);
}

inline bool descriptor_allows_moe_fp8_grouped(dsl::MoECapabilities caps, const char* context) {
    return descriptor_allows_moe_capability(caps, dsl::MoECapabilityFp8GroupedEligible, context, "FP8GroupedEligible");
}

inline bool descriptor_allows_moe_fp4_grouped(dsl::MoECapabilities caps) {
    return descriptor_allows_moe_capability(caps, dsl::MoECapabilityFp4GroupedEligible);
}

inline bool descriptor_allows_moe_fp4_grouped(dsl::MoECapabilities caps, const char* context) {
    return descriptor_allows_moe_capability(caps, dsl::MoECapabilityFp4GroupedEligible, context, "FP4GroupedEligible");
}

inline bool descriptor_allows_matmul_fp8_forward(dsl::MatmulCapabilities caps, const char* context) {
    return descriptor_allows_matmul_capability(caps,
                                               dsl::MatmulCapabilityFp8ForwardEligible,
                                               context,
                                               "FP8ForwardEligible");
}

inline bool descriptor_allows_matmul_fp8_colocated_forward(dsl::MatmulCapabilities caps,
                                                           const dsl::TensorRole* input_role,
                                                           const char* context) {
    if (!descriptor_allows_matmul_fp8_forward(caps, context)) {
        return false;
    }
    if (caps.colocate_input == dsl::QuantColocation::None) {
        return true;
    }
    if (!input_role) {
        if (recipe_capability_fallback_log_enabled()) {
            std::fprintf(stderr,
                         "[recipe capability] %s disabled: matmul descriptor requires %s but input role is missing\n",
                         context,
                         dsl::quant_colocation_name(caps.colocate_input));
        }
        return false;
    }
    const bool ready = input_role->quant_state == dsl::QuantState::FP8Ready;
    if (!ready && recipe_capability_fallback_log_enabled()) {
        std::fprintf(stderr,
                     "[recipe capability] %s disabled: matmul descriptor requires %s but input quant_state=%s\n",
                     context,
                     dsl::quant_colocation_name(caps.colocate_input),
                     dsl::quant_state_name(input_role->quant_state));
    }
    return ready;
}

inline bool descriptor_allows_matmul_fp8_backward(dsl::MatmulCapabilities caps, const char* context) {
    return descriptor_allows_matmul_capability(caps,
                                               dsl::MatmulCapabilityFp8BackwardEligible,
                                               context,
                                               "FP8BackwardEligible");
}

inline bool descriptor_allows_matmul_fp4_forward(dsl::MatmulCapabilities caps, const char* context) {
    return descriptor_allows_matmul_capability(caps,
                                               dsl::MatmulCapabilityFp4ForwardEligible,
                                               context,
                                               "FP4ForwardEligible");
}

inline bool descriptor_allows_matmul_fp4_backward(dsl::MatmulCapabilities caps, const char* context) {
    return descriptor_allows_matmul_capability(caps,
                                               dsl::MatmulCapabilityFp4BackwardEligible,
                                               context,
                                               "FP4BackwardEligible");
}

inline bool descriptor_has_moe_fp8_backward(dsl::MoECapabilities caps, const char* context) {
    return descriptor_allows_moe_capability(caps,
                                            dsl::MoECapabilityFp8BackwardImplemented,
                                            context,
                                            "FP8BackwardImplemented");
}

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H
