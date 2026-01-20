// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL-driven weight mapping utilities.

#ifndef SUROGATE_SRC_DSL_WEIGHT_MAPPING_H
#define SUROGATE_SRC_DSL_WEIGHT_MAPPING_H

#include <memory>

#include "dsl/ir.h"
#include "modules/weights/weight_mapping.h"

namespace dsl {

// Build a dynamic weight mapping from a DSL IR module's hf_mapping section.
// Returns nullptr if no mapping is defined.
std::unique_ptr<modules::BaseWeightMapping> build_weight_mapping(const Module& module);

} // namespace dsl

#endif // SUROGATE_SRC_DSL_WEIGHT_MAPPING_H
