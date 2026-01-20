// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Thread-local weight mapping override for DSL-driven loading.

#ifndef SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_OVERRIDE_H
#define SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_OVERRIDE_H

#include "modules/weights/weight_mapping.h"

namespace modules {

// Returns the current thread-local mapping override (nullptr if none).
BaseWeightMapping* get_weight_mapping_override();

// Sets the thread-local mapping override.
void set_weight_mapping_override(BaseWeightMapping* mapping);

// RAII guard for temporarily overriding weight mappings.
class WeightMappingOverrideGuard {
public:
    explicit WeightMappingOverrideGuard(BaseWeightMapping* mapping);
    ~WeightMappingOverrideGuard();

    WeightMappingOverrideGuard(const WeightMappingOverrideGuard&) = delete;
    WeightMappingOverrideGuard& operator=(const WeightMappingOverrideGuard&) = delete;

private:
    BaseWeightMapping* mPrev = nullptr;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_MAPPING_OVERRIDE_H
