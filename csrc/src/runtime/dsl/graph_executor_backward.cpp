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

#include "runtime/dsl/graph_executor.h"

#include <string>

#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/forward_plan.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "runtime/dsl/recompute_plan.h"
#include "kernels/kernels.h"
#include "runtime/lora/lora_model_utils.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

namespace dsl {

void GraphExecutor::recompute_block(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;
    if (!mRecomputePlan || mRecomputePlan->empty()) {
        if (mRunState.is_lora_only_mode()) {
            static bool warned = false;
            if (!warned) {
                fprintf(stderr,
                        "[WARN] DSL recompute plan missing; skipping recompute in LoRA mode.\n");
                warned = true;
            }
            return;
        }
        throw std::runtime_error("DSL recompute plan missing; hardcoded path removed");
    }
    mRecomputePlan->execute_layer(*this, layer_idx, B, T,
                                  mRunState.is_lora_only_mode(),
                                  mRunState.MainStream);
}

}  // namespace dsl
