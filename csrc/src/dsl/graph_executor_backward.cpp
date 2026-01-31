// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Segment-based backward recomputation support for DSL Graph executor.
//
// This file implements a clean, segment-based recomputation system that:
// 1. Organizes recomputation into atomic segments (attention path, FFN path)
// 2. Ensures correct dependency ordering within each segment
// 3. Guarantees numerical consistency with the forward pass
//
// Recompute Levels:
//   - None: All activations saved, no recomputation
//   - Standard: Recompute attention and FFN intermediates from checkpoints
//   - Aggressive: Recompute everything except residuals and LSE

#include "dsl/graph_executor.h"

#include <string>

#include "dsl/dsl_run_state.h"
#include "dsl/dsl_runtime.h"
#include "dsl/forward_plan.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "dsl/recompute_plan.h"
#include "kernels/kernels.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_run_state.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "training/runtime_options.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

namespace dsl {

void GraphExecutor::recompute_block(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;
    if (!mRecomputePlan || mRecomputePlan->empty()) {
        throw std::runtime_error("DSL recompute plan missing; hardcoded path removed");
    }
    mRecomputePlan->execute_layer(*this, layer_idx, B, T,
                                  mRunState.is_lora_only_mode(),
                                  mRunState.MainStream);
}

}  // namespace dsl
