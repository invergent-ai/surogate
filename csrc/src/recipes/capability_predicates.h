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

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H
