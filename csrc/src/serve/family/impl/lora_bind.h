#pragma once

// The load-side half of adapter application for the family's hybrid targets:
// register where every adaptable module of every layer lives, preallocate the
// banks, then apply whatever adapters the deployment named.
//
// One definition serves every hybrid (GDN + periodic full attention) target;
// each target instantiates it with its own config and payload types. The
// directory this registers is what makes loading target-agnostic afterwards --
// an adapter names "layers.7.down_proj" and the store resolves the rest,
// identically at startup and from the runtime endpoints.

#include "api/ops/lora_store.h"
#include "family/impl/mlp_swiglu.h"
#include "family/impl/lora_gdn.h"
#include "family/impl/lora_globals.h"
#include "api/types.h"
#include "api/ops/sparse_moe.h"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include <algorithm>
#include <cstdint>
#include <variant>

namespace sinfer::family {

inline void configure_lora_store(ops::LoraStore& store, const EngineOptions& options) {
    if (!store.empty()) { return; }
    const auto window = options.speculative.backend == SpeculativeBackend::None
                            ? 1U : options.speculative.draft_tokens + 1U;
    store.configure(std::max(options.lora_slots, 1U), std::max(options.lora_max_rank, 1U),
                    std::max({options.prefill_chunk, options.max_concurrency * window, 1U}));
}

inline void finish_lora_bind(ops::LoraStore& store, const EngineOptions& options) {
    store.validate_payloads(options.lora_payloads);
    store.ensure_banks();
    for (const auto& p : options.lora_payloads) {
        if (!store.covers_layer(p.layer)) { continue; }
        store.set_payload(p.slot, p);
    }
    store.set_active(true);
}

// The checkpoint's q_proj produces query and gate interleaved by head. The
// serving representation may keep those rows in separate weights or tensors.
template<class Projection, class Geometry>
void bind_lora_gated_attention(ops::LoraStore& store, int layer,
                                const Projection& projection, const Geometry& g) {
    using Binding = ops::LoraStore::ModuleBinding;
    if constexpr (requires { std::variant_size<Projection>::value; }) {
        std::visit([&](const auto& p) { bind_lora_gated_attention(store, layer, p, g); }, projection);
    } else if constexpr (requires { projection.query_gate; }) {
        store.register_module(layer, "q_proj", {projection.query_gate.qdata, kQueryPort,
                                                 g.hidden, 2 * g.query_size()});
        store.register_module(layer, "k_proj", {projection.key.qdata, kKeyPort, g.hidden, g.kv_size()});
        store.register_module(layer, "v_proj", {projection.value.qdata, kValuePort, g.hidden, g.kv_size()});
    } else {
        const void *q, *k, *v, *gate;
        if constexpr (requires { projection.query_key; }) {
            q = k = projection.query_key.qdata;
            v = gate = projection.gate_value.qdata;
        } else if constexpr (requires { projection.split; }) {
            if (projection.split) {
                q = projection.split->query.qdata;
                k = projection.split->key.qdata;
                v = projection.split->value.qdata;
                gate = projection.split->gate.qdata;
            } else {
                q = k = v = gate = projection.query_key_gate_value.qdata;
            }
        } else {
            q = k = v = gate = projection.query_key_gate_value.qdata;
        }
        store.register_module(layer, "q_proj", Binding{q, kQueryPort, g.hidden, g.query_size(),
            2 * g.query_size(), 0, g.head_dim, 2 * g.head_dim});
        store.register_module(layer, "q_proj", Binding{gate, kAttentionGatePort, g.hidden, g.query_size(),
            2 * g.query_size(), g.head_dim, g.head_dim, 2 * g.head_dim});
        store.register_module(layer, "k_proj", Binding{k, kKeyPort, g.hidden, g.kv_size()});
        store.register_module(layer, "v_proj", Binding{v, kValuePort, g.hidden, g.kv_size()});
    }
}

template<class Mlp>
void bind_lora_dense_mlp(ops::LoraStore& store, int layer, const Mlp& mlp,
                         int hidden, int intermediate, const std::string& prefix = "") {
    const void* gate = nullptr;
    const void* up = nullptr;
    if constexpr (requires { mlp.gate_up; }) { gate = up = mlp.gate_up.qdata; }
    if constexpr (requires { mlp.gate; mlp.up; }) {
        if (mlp.gate.qdata) { gate = mlp.gate.qdata; up = mlp.up.qdata; }
    }
    store.register_module(layer, prefix + "gate_proj", {gate, kGatePort, hidden, intermediate});
    store.register_module(layer, prefix + "up_proj", {up, kUpPort, hidden, intermediate});
    store.register_module(layer, prefix + "down_proj", {mlp.down.qdata, kDownPort, intermediate, hidden});
    // Surogate's fused gate_up adapters use [up; gate] checkpoint rows.
    store.register_module(layer, prefix + "gate_up_proj", {up, kUpPort, hidden, intermediate,
        2 * intermediate, 0});
    store.register_module(layer, prefix + "gate_up_proj", {gate, kGatePort, hidden, intermediate,
        2 * intermediate, intermediate});
}

// Device-selected expert projections retain their complete checkpoint names.
// Tables are [router, shared gate, (gate, up, down) for each expert, shared expert].
inline void bind_lora_moe(ops::LoraStore& store, int layer, const ops::SparseMoeWeights& weights) {
    if (store.max_rank() > ops::detail::kMoeLoraMaxRank) {
        throw std::invalid_argument("expert LoRA supports --max-lora-rank up to 256");
    }
    const auto g = ops::sparse_moe_geometry(weights);
    const auto* key = weights.router_shared_gate.qdata;
    using Binding = ops::LoraStore::ModuleBinding;
    std::vector<Binding> table;
    const Binding router{key, 32, g.hidden, g.experts};
    store.register_module(layer, "mlp.gate", router);
    store.register_module(layer, "router.proj", router);
    store.register_module(layer, "feed_forward.router", router);
    store.register_module(layer, "feed_forward.gate", router);
    table.push_back(router);
    store.register_base(key, 32, {{weights.router_shared_gate, 0, 0, g.experts}});
    if (g.has_shared() && g.shared_gated) {
        const Binding gate{key, 33, g.hidden, 1};
        store.register_module(layer, "mlp.shared_expert_gate", gate);
        table.push_back(gate);
        store.register_base(key, 33, {{weights.router_shared_gate, g.experts, 0, 1}});
    } else if (weights.router_bias) {
        const Binding correction{key, 34, 1, g.experts};
        for (const std::string name : {"feed_forward.expert_bias.bias", "mlp.gate.e_score_correction_bias.bias",
                                       "mlp.experts.e_score_correction_bias.bias"}) {
            store.register_module(layer, name, correction);
        }
        store.register_base(key, 34, {}, Tensor(const_cast<float*>(weights.router_bias), DType::FP32, {g.experts}));
        table.push_back(correction);
    } else { table.push_back({}); }
    for (int expert = 0; expert < g.experts + (g.has_shared() ? 1 : 0); ++expert) {
        const int width = expert == g.experts ? g.shared_intermediate : g.intermediate;
        const std::string prefix = expert == g.experts ? "mlp.shared_expert."
                                                      : "mlp.experts." + std::to_string(expert) + ".";
        const Binding gate{key, 100 + expert * 3, g.hidden, width};
        const Binding up{key, 101 + expert * 3, g.hidden, width};
        const Binding down{key, 102 + expert * 3, width, g.hidden};
        store.register_module(layer, prefix + "gate_proj", gate);
        store.register_module(layer, prefix + "up_proj", up);
        store.register_module(layer, prefix + "down_proj", down);
        auto gate_part = gate;
        gate_part.source_out = 2 * width;
        gate_part.row_offset = width;
        auto up_part = up;
        up_part.source_out = 2 * width;
        store.register_module(layer, prefix + "gate_up_proj", up_part);
        store.register_module(layer, prefix + "gate_up_proj", gate_part);
        const std::string alternate = expert == g.experts ? "mlp.shared_experts."
                                                          : "experts." + std::to_string(expert) + ".";
        store.register_module(layer, alternate + "gate_proj", gate);
        store.register_module(layer, alternate + "up_proj", up);
        store.register_module(layer, alternate + "down_proj", down);
        store.register_module(layer, alternate + "gate_up_proj", up_part);
        store.register_module(layer, alternate + "gate_up_proj", gate_part);
        if (expert < g.experts) {
            const std::string ffn = "feed_forward.experts." + std::to_string(expert) + ".";
            store.register_module(layer, ffn + "w1", gate);
            store.register_module(layer, ffn + "w3", up);
            store.register_module(layer, ffn + "w2", down);
        }
        const bool shared = expert == g.experts;
        const auto& gu = shared ? weights.shared_gate_up : weights.routed_gate_up;
        const auto& dw = shared ? weights.shared_down : weights.routed_down;
        const bool up_first = gu.qtype == QType::NVFP4;
        const int first = shared ? 0 : expert * 2 * width;
        const auto multiplier = [&](int half) -> const float* {
            return shared || !weights.routed_gate_up_scale ? nullptr : weights.routed_gate_up_scale + expert * 2 + half;
        };
        store.register_base(key, gate.port, {{gu, first + (up_first ? width : 0), 0, width, false, multiplier(up_first ? 1 : 0)}});
        store.register_base(key, up.port, {{gu, first + (up_first ? 0 : width), 0, width, false, multiplier(up_first ? 0 : 1)}});
        store.register_base(key, down.port, {{dw, shared ? 0 : expert * g.hidden, 0, g.hidden, false,
            shared || !weights.routed_down_scale ? nullptr : weights.routed_down_scale + expert}});
        table.insert(table.end(), {gate, up, down});
    }
    store.register_bank_table(key, table);
}

template <class FusedPayload, class Runtime, class IsFull>
void bind_lora_hybrid(const Runtime& runtime, const EngineOptions& options, IsFull&& is_full) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    auto& store = ops::lora_store_for_current_device();
    configure_lora_store(store, options);
    bind_lora_globals(store, runtime);
    const auto& g = runtime.geometry;
    std::size_t full_index = 0, linear_index = 0;
    for (int layer = 0; layer < g.layers; ++layer) {
        if (g.attention_schedule_declared ? g.layer_attends(layer) : is_full(layer)) {
            const auto& full = runtime.full_layers.at(full_index++);
            bind_lora_gated_attention(store, layer, full.projection, g);
            store.register_module(layer, "o_proj", {full.output.qdata, kOutputPort, g.query_size(), g.hidden});
            bind_lora_dense_mlp(store, layer, full.post_mixer, g.hidden, g.intermediate);
        } else {
            const auto& linear = runtime.gdn_layers.at(linear_index++);
            bind_lora_gdn(store, layer, linear, g);
            bind_lora_dense_mlp(store, layer, linear.post_mixer, g.hidden, g.intermediate);
        }
    }
    finish_lora_bind(store, options);
}

/// The hybrid MoE directory includes full/linear attention, routers and experts.
template <class Runtime, class IsFull>
void bind_lora_moe_hybrid(const Runtime& runtime, const EngineOptions& options, IsFull&& is_full) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    configure_lora_store(store, options);
    bind_lora_globals(store, runtime);

    const auto& g = runtime.geometry;
    std::size_t full_index = 0, linear_index = 0;
    for (int layer = 0; layer < g.layers; ++layer) {
        if (g.attention_schedule_declared ? g.layer_attends(layer) : is_full(layer)) {
            const auto& full = runtime.full_layers.at(full_index++);
            bind_lora_gated_attention(store, layer, full.projection, g);
            store.register_module(layer, "o_proj", {full.output.qdata, kOutputPort, g.query_size(), g.hidden});
            bind_lora_moe(store, layer, [&]() -> const ops::SparseMoeWeights& {
                if constexpr (requires { full.post_mixer.op; }) { return full.post_mixer.op; }
                else { return full.post_mixer.moe; }
            }());
        } else {
            const auto& linear = runtime.gdn_layers.at(linear_index++);
            bind_lora_gdn(store, layer, linear, g);
            bind_lora_moe(store, layer, [&]() -> const ops::SparseMoeWeights& {
                if constexpr (requires { linear.post_mixer.op; }) { return linear.post_mixer.op; }
                else { return linear.post_mixer.moe; }
            }());
        }
    }
    finish_lora_bind(store, options);
}

} // namespace sinfer::family
