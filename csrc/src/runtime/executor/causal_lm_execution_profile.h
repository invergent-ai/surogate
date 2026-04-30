// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Causal language-model runtime profile.

#ifndef SUROGATE_SRC_RUNTIME_EXECUTOR_CAUSAL_LM_EXECUTION_PROFILE_H
#define SUROGATE_SRC_RUNTIME_EXECUTOR_CAUSAL_LM_EXECUTION_PROFILE_H

#include "runtime/executor/execution_request.h"
#include "runtime/core/run_state_requirements.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct RuntimeOptions;

namespace modules {
struct ModelConfig;
}  // namespace modules

namespace dsl {

class DslRunState;
class IGraphExecutor;

struct CausalLMDocMaskingInfo {
    std::vector<std::int32_t> cu_seqlens;
    int num_docs = 0;
    int max_seqlen = 0;
    int total_q = 0;
};

class CausalLMExecutionProfile {
public:
    [[nodiscard]] std::optional<CausalLMDocMaskingInfo>
    compute_doc_masking(const std::int32_t* position_ids, int B, int T, bool mrope = false) const;

    [[nodiscard]] bool apply_doc_masking(IGraphExecutor& executor,
                                         const RuntimeOptions& options,
                                         const modules::ModelConfig& config,
                                         Tensor inputs,
                                         Tensor position_ids,
                                         int micro_step = 0) const;

    [[nodiscard]] ExecutionRequest make_forward_request(DslRunState& rs,
                                                        const modules::ModelConfig& config,
                                                        const RuntimeOptions& options,
                                                        Tensor inputs,
                                                        Tensor position_ids,
                                                        int micro_step) const;

    [[nodiscard]] ExecutionRequest make_eval_request(DslRunState& rs,
                                                     const modules::ModelConfig& config,
                                                     const RuntimeOptions& options,
                                                     Tensor inputs,
                                                     Tensor position_ids,
                                                     Tensor targets,
                                                     int micro_step) const;

    [[nodiscard]] ExecutionRequest make_backward_request(DslRunState& rs,
                                                         const modules::ModelConfig& config,
                                                         const RuntimeOptions& options,
                                                         Tensor inputs,
                                                         Tensor targets,
                                                         int grad_accum_steps,
                                                         int micro_step) const;

    [[nodiscard]] std::vector<std::string> backward_save_exclusions() const;

    [[nodiscard]] RuntimeRunStateRequirements run_state_requirements() const {
        return RuntimeRunStateRequirements::causal_lm();
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_EXECUTOR_CAUSAL_LM_EXECUTION_PROFILE_H
