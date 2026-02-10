// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>

namespace dsl {

void CompiledExecutor::dispatch_layernorm(const CompiledOp& op) {
    throw std::runtime_error("LayerNorm forward dispatch not yet implemented");
}

void CompiledExecutor::dispatch_layernorm_backward(const CompiledOp& op) {
    throw std::runtime_error("LayerNorm backward dispatch not yet implemented");
}

}  // namespace dsl
