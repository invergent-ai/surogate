// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Thread-local weight mapping override implementation.

#include "modules/weights/weight_mapping_override.h"

namespace modules {
namespace {
thread_local BaseWeightMapping* gWeightMappingOverride = nullptr;
} // namespace

BaseWeightMapping* get_weight_mapping_override() {
    return gWeightMappingOverride;
}

void set_weight_mapping_override(BaseWeightMapping* mapping) {
    gWeightMappingOverride = mapping;
}

WeightMappingOverrideGuard::WeightMappingOverrideGuard(BaseWeightMapping* mapping)
    : mPrev(get_weight_mapping_override()) {
    set_weight_mapping_override(mapping);
}

WeightMappingOverrideGuard::~WeightMappingOverrideGuard() {
    set_weight_mapping_override(mPrev);
}

} // namespace modules
