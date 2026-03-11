// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Backward recomputation support for DSL Graph executor.
//
// Forward replay (gradient checkpointing) is now handled entirely by
// CompiledExecutor::replay_layer_forward(). The old RecomputePlan-based
// system has been removed.

#include "runtime/dsl/graph_executor.h"

namespace dsl {

// recompute_block was removed — replay_layer_forward in compiled_ops.cpp
// handles torch-style gradient checkpointing directly.

}  // namespace dsl
