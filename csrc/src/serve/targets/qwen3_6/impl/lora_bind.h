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
#include "api/types.h"

#include <algorithm>
#include <cstdint>
#include <variant>

namespace sinfer::targets::qwen3_6 {

template <class TextConfig, class FusedPayload, class Runtime, class IsFull>
void bind_lora_hybrid(const Runtime& runtime, const EngineOptions& options, IsFull&& is_full) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        // The widest round an adapter can see: a prefill chunk, or every lane
        // times the verify window when a draft is in flight. Sizing this for
        // lanes alone left every prompt without a scratch, so the prompt was
        // built on the base model while the generated tokens carried the delta --
        // an adapter that looked weak rather than unapplied.
        const std::uint32_t window = options.speculative.backend == SpeculativeBackend::None
                                         ? 1U
                                         : options.speculative.draft_tokens + 1U;
        const std::uint32_t decode_columns =
            std::max<std::uint32_t>(options.max_concurrency, 1) * window;
        const std::uint32_t widest = std::max<std::uint32_t>(
            decode_columns, std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    // q, k and v leave one fused projection as separate contiguous tensors, so
    // they share a bank key and are told apart by the port; o and down have a
    // weight each. Attention modules exist only on the full-attention layers;
    // the MLP is on every layer, and a real PEFT adapter carries down_proj on
    // all of them.
    using Binding          = ops::LoraStore::ModuleBinding;
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        const auto index      = static_cast<std::int32_t>(layer);
        const void* mlp_down  = nullptr;
        if (is_full(layer)) {
            const auto& full = runtime.full_layers.at(full_index++);
            mlp_down         = full.post_mixer.down.qdata;
            const auto* fused = std::get_if<FusedPayload>(&full.projection);
            if (fused != nullptr) {
                const void* qkv = fused->query_key_gate_value.qdata;
                store.register_module(index, "q_proj",
                                      Binding{qkv, 0, TextConfig::hidden,
                                              TextConfig::query_heads * TextConfig::head_dim});
                store.register_module(index, "k_proj",
                                      Binding{qkv, 1, TextConfig::hidden,
                                              TextConfig::kv_heads * TextConfig::head_dim});
                store.register_module(index, "v_proj",
                                      Binding{qkv, 2, TextConfig::hidden,
                                              TextConfig::kv_heads * TextConfig::head_dim});
            } else {
                store.register_layer_refusal(
                    index, "q_proj",
                    "this artifact splits its attention projection, which the attention adapter "
                    "path does not bind");
            }
            store.register_module(index, "o_proj",
                                  Binding{full.output.qdata, 3,
                                          TextConfig::query_heads * TextConfig::head_dim,
                                          TextConfig::hidden});
        } else {
            mlp_down = runtime.gdn_layers.at(gdn_index++).post_mixer.down.qdata;
            for (const char* module : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer is linear attention, which has no self_attn projections; an "
                    "adapter naming them here was trained against a different architecture");
            }
        }
        store.register_module(index, "down_proj",
                              Binding{mlp_down, 4, TextConfig::intermediate, TextConfig::hidden});
    }
    for (const char* module : {"gate_proj", "up_proj"}) {
        store.register_refusal(module,
                               "gate and up are fused and consumed by SwiGLU inside the "
                               "projection, so there is no intermediate tensor to add a delta to");
    }
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        store.set_module_slot(payload.layer, payload.module, payload.slot, payload.a, payload.b,
                              payload.rank, payload.in_dim, payload.out_dim, payload.scale);
    }
    ops::lora_set_active(true);
}

/// The MoE flavor: attention adapters bind; the MLP is routed experts, so its
/// modules are refused with the reason rather than half-applied. The attention
/// projection is a plain struct here (always fused), not a variant.
template <class TextConfig, class Runtime, class IsFull>
void bind_lora_moe_hybrid(const Runtime& runtime, const EngineOptions& options, IsFull&& is_full) {
    if (!options.lora_enable && options.lora_payloads.empty()) { return; }
    ops::LoraStore& store = ops::lora_store_for_current_device();
    if (store.empty()) {
        const std::uint32_t window = options.speculative.backend == SpeculativeBackend::None
                                         ? 1U
                                         : options.speculative.draft_tokens + 1U;
        const std::uint32_t decode_columns =
            std::max<std::uint32_t>(options.max_concurrency, 1) * window;
        const std::uint32_t widest = std::max<std::uint32_t>(
            decode_columns, std::max<std::uint32_t>(options.prefill_chunk, 1));
        store.configure(static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_slots, 1)),
                        static_cast<std::int32_t>(std::max<std::uint32_t>(options.lora_max_rank, 1)),
                        static_cast<std::int32_t>(widest));
    }

    using Binding          = ops::LoraStore::ModuleBinding;
    std::size_t full_index = 0;
    for (std::size_t layer = 0; layer < TextConfig::layers; ++layer) {
        const auto index = static_cast<std::int32_t>(layer);
        if (is_full(layer)) {
            const auto& full = runtime.full_layers.at(full_index++);
            const void* qkv  = full.projection.query_key_gate_value.qdata;
            store.register_module(index, "q_proj",
                                  Binding{qkv, 0, TextConfig::hidden,
                                          TextConfig::query_heads * TextConfig::head_dim});
            store.register_module(index, "k_proj",
                                  Binding{qkv, 1, TextConfig::hidden,
                                          TextConfig::kv_heads * TextConfig::head_dim});
            store.register_module(index, "v_proj",
                                  Binding{qkv, 2, TextConfig::hidden,
                                          TextConfig::kv_heads * TextConfig::head_dim});
            store.register_module(index, "o_proj",
                                  Binding{full.output.qdata, 3,
                                          TextConfig::query_heads * TextConfig::head_dim,
                                          TextConfig::hidden});
        } else {
            for (const char* module : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                store.register_layer_refusal(
                    index, module,
                    "this layer is linear attention, which has no self_attn projections; an "
                    "adapter naming them here was trained against a different architecture");
            }
        }
    }
    for (const char* module : {"down_proj", "gate_proj", "up_proj"}) {
        store.register_refusal(
            module,
            "this model's MLP is routed experts, and per-expert adapters are not applied; "
            "merge the adapter into the checkpoint before conversion (`surogate merge`) to "
            "serve its MLP weights here");
    }
    store.ensure_banks();

    for (const auto& payload : options.lora_payloads) {
        store.set_module_slot(payload.layer, payload.module, payload.slot, payload.a, payload.b,
                              payload.rank, payload.in_dim, payload.out_dim, payload.scale);
    }
    ops::lora_set_active(true);
}

} // namespace sinfer::targets::qwen3_6
