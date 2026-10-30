// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// The dispatch-PP stage plan, as the trainers publish it in the environment (surogate/train/dispatch_pp.py).

#pragma once

#include <vector>

namespace dsl {

/// Stage index of every layer. SUROGATE_DISPATCH_STAGE_PLAN (comma-separated stage sizes that sum to
/// num_layers) when set, else uniform stages of SUROGATE_DISPATCH_STAGE_BLOCKS layers; empty when neither is
/// set. Stage s runs on GPU s % gpus.
std::vector<int> dispatch_pp_layer_stages(int num_layers);

}  // namespace dsl
