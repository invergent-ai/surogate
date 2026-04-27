// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Phase 5 schema hook registry scaffold.
//
// Hooks are declared against structural block-schema slots instead of legacy
// enum hook points. Dispatch remains opt-in while execution paths migrate from
// imperative call sites to hook callbacks.

#ifndef SUROGATE_SRC_RUNTIME_DSL_HOOK_REGISTRY_H
#define SUROGATE_SRC_RUNTIME_DSL_HOOK_REGISTRY_H

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/dsl/buffer_plan.h"

class NCCLCommunicator;
struct Tensor;

namespace dsl {

class DslGradStore;
class DslWeightManager;

enum class HookEventKind {
    Unknown,
    AfterProduce,
    BeforeConsume,
    AfterConsume,
    AfterCommunication,
    AfterAllReduce,
    AfterAllToAll,
    AfterReduceScatter,
};

[[nodiscard]] const char* hook_event_name(HookEventKind event);

struct HookTarget {
    std::string schema_id;
    std::string slot_name;

    [[nodiscard]] bool valid() const {
        return !schema_id.empty() && !slot_name.empty();
    }
};

struct HookContext {
    int layer_idx = -1;
    HookTarget target;
    HookEventKind event = HookEventKind::Unknown;
    cudaStream_t stream = nullptr;
    void* payload = nullptr;
};

struct AfterProduceHookPayload {
    void* action_context = nullptr;
    void (*apply_lora)(void*) = nullptr;
    bool lora_applied = false;
};

struct BeforeConsumeHookPayload {
    DslWeightManager* weight_manager = nullptr;
    NCCLCommunicator* comm = nullptr;
    cudaStream_t prefetch_stream = nullptr;
    cudaStream_t wait_stream = nullptr;
    bool capturing = false;
    bool current_layer_handled = false;
};

struct AfterConsumeHookPayload {
    DslWeightManager* weight_manager = nullptr;
    cudaStream_t release_stream = nullptr;
    bool capturing = false;
    bool current_layer_released = false;
};

struct GradientOffloadHookPayload {
    DslGradStore* grads = nullptr;
    NCCLCommunicator* comm = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t sync_event = nullptr;
    bool capturing = false;
    bool offloaded = false;
};

struct CommunicationHookPayload {
    const Tensor* send_tensor = nullptr;
    const Tensor* recv_tensor = nullptr;
    int ep_size = 1;
    int total_send = 0;
    int total_recv = 0;
    bool token_all_to_all_completed = false;
    bool after_all_to_all_observed = false;
    bool after_communication_observed = false;
};

using HookCallback = std::function<void(HookContext&)>;

struct HookRegistration {
    HookEventKind event = HookEventKind::Unknown;
    HookTarget target;
    std::string name;
    int priority = 0;
    bool distribution_aware = false;
    HookCallback callback;
};

class HookRegistry {
public:
    void register_hook(HookRegistration registration);

    void on_after_produce(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_before_consume(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_after_consume(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_after_communication(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_after_all_reduce(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_after_all_to_all(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);
    void on_after_reduce_scatter(HookTarget target, std::string name, HookCallback callback = {}, int priority = 0);

    [[nodiscard]] std::vector<const HookRegistration*>
    find(HookEventKind event, std::string_view schema_id, std::string_view slot_name) const;

    [[nodiscard]] int dispatch(HookContext& context) const;

    [[nodiscard]] const std::vector<HookRegistration>& registrations() const {
        return mRegistrations;
    }

    [[nodiscard]] int size() const {
        return static_cast<int>(mRegistrations.size());
    }

private:
    std::vector<HookRegistration> mRegistrations;
};

[[nodiscard]] std::string schema_id_for_hook_target(const BlockSchemaPlanRecord& record);
[[nodiscard]] std::string schema_id_for_hook_target(const BlockSchemaLayerSummary& layer);

/// Collect structural hook targets from schema metadata for a future migration.
/// The result is diagnostics/scaffolding only; it does not alter execution.
[[nodiscard]] std::vector<HookTarget> collect_schema_hook_targets(const std::vector<BlockSchemaPlanRecord>& records,
                                                                  HookEventKind event);

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_HOOK_REGISTRY_H
