// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// A dispatch-PP stage boundary: the named tensors one stage's GPU hands to the next stage's (block hi's residual and
// MLP output forward, their gradients backward), staged in pinned host memory so both copies are plain DMAs.

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "utilities/pinned_pool.h"

namespace dsl {

using StageBoundary = std::vector<std::pair<std::string, surogate::PinnedBytes>>;

}  // namespace dsl
