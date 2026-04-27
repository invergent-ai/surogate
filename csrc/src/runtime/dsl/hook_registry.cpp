// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/hook_registry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace dsl {

namespace {

[[nodiscard]] bool is_distribution_event(HookEventKind event) {
    return event == HookEventKind::AfterCommunication || event == HookEventKind::AfterAllReduce ||
           event == HookEventKind::AfterAllToAll || event == HookEventKind::AfterReduceScatter;
}

[[nodiscard]] bool is_param_slot(const BlockSchemaSlotSummary& slot) {
    return slot.kind == "param";
}

[[nodiscard]] bool is_streamable_param_slot(const BlockSchemaSlotSummary& slot) {
    if (!is_param_slot(slot)) return false;
    if (slot.streaming_prefetch_distance >= 0) return true;
    return slot.residency == "auto" || slot.residency == "cpu_pinned_stream" || slot.residency == "cpu_pageable" ||
           slot.residency == "nvme_offload";
}

[[nodiscard]] bool is_expert_parallel_activation(const BlockSchemaSlotSummary& slot) {
    return !is_param_slot(slot) && slot.distribution_kind == "expert_parallel";
}

[[nodiscard]] bool is_sharded_param_slot(const BlockSchemaSlotSummary& slot) {
    return is_param_slot(slot) &&
           (slot.distribution_kind == "sharded_dim" || slot.distribution_kind == "expert_parallel");
}

[[nodiscard]] bool is_replicated_param_slot(const BlockSchemaSlotSummary& slot) {
    return is_param_slot(slot) && (slot.distribution_kind.empty() || slot.distribution_kind == "replicated" ||
                                   slot.distribution_kind == "router_replicated");
}

[[nodiscard]] bool is_lora_after_produce_slot(const BlockSchemaSlotSummary& slot) {
    if (is_param_slot(slot)) return false;
    return slot.name == "qkv" || slot.name == "att_out" || slot.name == "mlp_up" || slot.name == "mlp_down" ||
           slot.name == "router_logits" || slot.name == "expert_gate_up" || slot.name == "expert_up" ||
           slot.name == "expert_down";
}

[[nodiscard]] bool registration_less(const HookRegistration& a, const HookRegistration& b) {
    if (a.event != b.event) return static_cast<int>(a.event) < static_cast<int>(b.event);
    if (a.target.schema_id != b.target.schema_id) return a.target.schema_id < b.target.schema_id;
    if (a.target.slot_name != b.target.slot_name) return a.target.slot_name < b.target.slot_name;
    if (a.priority != b.priority) return a.priority > b.priority;
    return a.name < b.name;
}

void validate_registration(const HookRegistration& registration) {
    if (registration.event == HookEventKind::Unknown) {
        throw std::invalid_argument("hook registration event is unknown");
    }
    if (!registration.target.valid()) {
        throw std::invalid_argument("hook registration target must include schema_id and slot_name");
    }
    if (registration.name.empty()) {
        throw std::invalid_argument("hook registration name must be non-empty");
    }
}

void add(HookRegistry& registry,
         HookEventKind event,
         HookTarget target,
         std::string name,
         HookCallback callback,
         int priority) {
    HookRegistration registration;
    registration.event = event;
    registration.target = std::move(target);
    registration.name = std::move(name);
    registration.priority = priority;
    registration.distribution_aware = is_distribution_event(event);
    registration.callback = std::move(callback);
    registry.register_hook(std::move(registration));
}

}  // namespace

const char* hook_event_name(HookEventKind event) {
    switch (event) {
        case HookEventKind::AfterProduce: return "after_produce";
        case HookEventKind::BeforeConsume: return "before_consume";
        case HookEventKind::AfterConsume: return "after_consume";
        case HookEventKind::AfterCommunication: return "after_communication";
        case HookEventKind::AfterAllReduce: return "after_all_reduce";
        case HookEventKind::AfterAllToAll: return "after_all_to_all";
        case HookEventKind::AfterReduceScatter: return "after_reduce_scatter";
        case HookEventKind::Unknown: return "unknown";
    }
    return "unknown";
}

void HookRegistry::register_hook(HookRegistration registration) {
    validate_registration(registration);
    registration.distribution_aware = registration.distribution_aware || is_distribution_event(registration.event);
    mRegistrations.push_back(std::move(registration));
    std::stable_sort(mRegistrations.begin(), mRegistrations.end(), registration_less);
}

void HookRegistry::on_after_produce(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterProduce, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_before_consume(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::BeforeConsume, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_after_consume(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterConsume, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_after_communication(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterCommunication, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_after_all_reduce(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterAllReduce, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_after_all_to_all(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterAllToAll, std::move(target), std::move(name), std::move(callback), priority);
}

void HookRegistry::on_after_reduce_scatter(HookTarget target, std::string name, HookCallback callback, int priority) {
    add(*this, HookEventKind::AfterReduceScatter, std::move(target), std::move(name), std::move(callback), priority);
}

std::vector<const HookRegistration*>
HookRegistry::find(HookEventKind event, std::string_view schema_id, std::string_view slot_name) const {
    std::vector<const HookRegistration*> matches;
    for (const HookRegistration& registration : mRegistrations) {
        if (registration.event != event) continue;
        if (std::string_view(registration.target.schema_id) != schema_id) continue;
        if (std::string_view(registration.target.slot_name) != slot_name) continue;
        matches.push_back(&registration);
    }
    return matches;
}

int HookRegistry::dispatch(HookContext& context) const {
    int dispatched = 0;
    const auto matches = find(context.event, context.target.schema_id, context.target.slot_name);
    for (const HookRegistration* registration : matches) {
        if (registration->callback) {
            registration->callback(context);
        }
        ++dispatched;
    }
    return dispatched;
}

std::string schema_id_for_hook_target(const BlockSchemaPlanRecord& record) {
    if (!record.block_family.empty()) return record.block_family;
    if (!record.block_name.empty()) return record.block_name;
    return record.block_type;
}

std::string schema_id_for_hook_target(const BlockSchemaLayerSummary& layer) {
    return layer.block_family;
}

std::vector<HookTarget> collect_schema_hook_targets(const std::vector<BlockSchemaPlanRecord>& records,
                                                    HookEventKind event) {
    std::vector<HookTarget> targets;
    for (const BlockSchemaPlanRecord& record : records) {
        const std::string schema_id = schema_id_for_hook_target(record);
        if (schema_id.empty()) continue;
        for (const BlockSchemaSlotSummary& slot : record.slots) {
            bool include = false;
            switch (event) {
                case HookEventKind::AfterProduce: include = is_lora_after_produce_slot(slot); break;
                case HookEventKind::BeforeConsume: include = is_streamable_param_slot(slot); break;
                case HookEventKind::AfterConsume: include = is_streamable_param_slot(slot); break;
                case HookEventKind::AfterCommunication:
                case HookEventKind::AfterAllToAll: include = is_expert_parallel_activation(slot); break;
                case HookEventKind::AfterAllReduce: include = is_replicated_param_slot(slot); break;
                case HookEventKind::AfterReduceScatter: include = is_sharded_param_slot(slot); break;
                case HookEventKind::Unknown: include = false; break;
            }
            if (include) {
                targets.push_back(HookTarget{schema_id, slot.name});
            }
        }
    }
    std::stable_sort(targets.begin(), targets.end(), [](const HookTarget& a, const HookTarget& b) {
        if (a.schema_id != b.schema_id) return a.schema_id < b.schema_id;
        return a.slot_name < b.slot_name;
    });
    targets.erase(std::unique(targets.begin(),
                              targets.end(),
                              [](const HookTarget& a, const HookTarget& b) {
                                  return a.schema_id == b.schema_id && a.slot_name == b.slot_name;
                              }),
                  targets.end());
    return targets;
}

}  // namespace dsl
