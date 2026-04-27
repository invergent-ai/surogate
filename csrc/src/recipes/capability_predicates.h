// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H
#define SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H

#include <cstdint>

#include "runtime/executor/op_descriptor_types.h"

namespace recipes {

inline bool descriptor_allows_capability(dsl::OpCapabilities caps, std::uint32_t flag) {
    return caps.flags == dsl::OpCapabilityNone || caps.has(flag);
}

inline bool descriptor_allows_fp8(dsl::OpCapabilities caps) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp8Eligible);
}

inline bool descriptor_allows_fp4(dsl::OpCapabilities caps) {
    return descriptor_allows_capability(caps, dsl::OpCapabilityFp4Eligible);
}

}  // namespace recipes

#endif  // SUROGATE_SRC_RECIPES_CAPABILITY_PREDICATES_H
