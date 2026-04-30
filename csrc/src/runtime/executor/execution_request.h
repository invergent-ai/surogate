// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Generic runtime execution requests for DSL graph execution.

#ifndef SUROGATE_SRC_RUNTIME_EXECUTOR_EXECUTION_REQUEST_H
#define SUROGATE_SRC_RUNTIME_EXECUTOR_EXECUTION_REQUEST_H

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"
#include "utilities/tensor.h"

namespace dsl {

enum class ExecutionMode {
    Forward,
    Eval,
    Backward
};

struct RuntimeBinding {
    std::string name;
    Tensor tensor{};
};

enum class RuntimeCopyTransform {
    None,
    ReplicateSinglePlaneToThree
};

struct RuntimeCopy {
    std::string name;
    Tensor source{};
    Tensor destination{};
    cudaMemcpyKind kind = cudaMemcpyHostToDevice;
    RuntimeCopyTransform transform = RuntimeCopyTransform::None;
};

struct ExecutionResult {
    std::optional<float> loss;
    std::optional<float> accuracy;
};

struct ExecutionRequest {
    long batch = 0;
    long sequence = 0;
    ExecutionMode mode = ExecutionMode::Forward;
    int micro_step = 0;
    int grad_accum_steps = 1;
    bool full_forward = false;
    bool initialize_loss_buffers = false;
    bool reduce_loss_on_completion = false;
    bool disable_forward_saves = false;

    std::vector<RuntimeBinding> bindings;
    std::vector<RuntimeCopy> input_copies;
    std::vector<std::string> requested_outputs;
    std::vector<std::string> skipped_backward_tensors;
    std::vector<std::string> gather_before_forward_weight_groups;
    std::vector<std::string> release_after_forward_weight_groups;
    std::vector<std::string> gather_before_backward_weight_groups;
    std::vector<std::string> release_after_backward_weight_groups;
    std::vector<std::string> fp8_forward_cache_weight_groups;
    std::vector<std::string> fp8_backward_cache_weight_groups;

    Tensor last_inputs_cpu{};
    bool has_last_inputs_cpu = false;

    const modules::ForwardHook* forward_hook = nullptr;
    const modules::BackwardHook* backward_hook = nullptr;

    float* logprobs_gpu = nullptr;
    float* custom_dloss_gpu = nullptr;
    const float* inv_temperature_gpu = nullptr;

    [[nodiscard]] const RuntimeBinding* find_binding(const std::string& name) const {
        for (const auto& binding : bindings) {
            if (binding.name == name) return &binding;
        }
        return nullptr;
    }
};

inline void validate_execution_request(const ExecutionRequest& request) {
    if (request.batch <= 0) {
        throw std::runtime_error("execution request batch must be positive");
    }
    if (request.sequence <= 0) {
        throw std::runtime_error("execution request sequence must be positive");
    }
    std::unordered_set<std::string> names;
    for (const auto& binding : request.bindings) {
        if (binding.name.empty()) {
            throw std::runtime_error("execution request contains an empty runtime binding name");
        }
        if (!names.insert(binding.name).second) {
            throw std::runtime_error("duplicate runtime binding '" + binding.name + "'");
        }
    }
    for (const auto& copy : request.input_copies) {
        if (copy.name.empty()) {
            throw std::runtime_error("execution request contains an empty runtime copy name");
        }
        if (!names.insert(copy.name).second) {
            throw std::runtime_error("duplicate runtime binding '" + copy.name + "'");
        }
    }
}

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_EXECUTOR_EXECUTION_REQUEST_H
