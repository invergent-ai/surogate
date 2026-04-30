// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Embedding-only runtime profile.

#ifndef SUROGATE_SRC_RUNTIME_EXECUTOR_EMBEDDING_EXECUTION_PROFILE_H
#define SUROGATE_SRC_RUNTIME_EXECUTOR_EMBEDDING_EXECUTION_PROFILE_H

#include <utility>
#include <vector>

#include "runtime/executor/execution_request.h"

namespace dsl {

class EmbeddingExecutionProfile {
public:
    [[nodiscard]] ExecutionRequest forward_request(Tensor input_ids, std::vector<std::string> outputs) const {
        ExecutionRequest request;
        request.batch = input_ids.Rank > 0 ? input_ids.Sizes[0] : 0;
        request.sequence = input_ids.Rank > 1 ? input_ids.Sizes[1] : 0;
        request.mode = ExecutionMode::Forward;
        request.bindings.push_back(RuntimeBinding{"input_ids", input_ids});
        request.requested_outputs = std::move(outputs);
        return request;
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_EXECUTOR_EMBEDDING_EXECUTION_PROFILE_H
