#pragma once
#include "api/ops/lora_store.h"
#include "api/types.h"
#include <algorithm>
#include "api/family/vision.h"
#include <type_traits>
#include <variant>

namespace sinfer::family {
inline std::uint32_t lora_prefill_columns(const VisionGeometry& vision,
                                          const EngineOptions& options) {
    const auto image =
        options.enable_vision && vision.attention_mode
            ? ((static_cast<std::uint32_t>(vision.max_image_tokens) + 127U) / 128U) * 128U
            : 0U;
    return std::min(options.max_context, std::max(options.prefill_chunk, image));
}

// Collect represented weights before adapters load, including offloaded views.
template <class T>
void collect_lora_bases(ops::LoraStore& store, const T& value) {
    if constexpr (std::is_same_v<T, Weight>) {
        store.register_base_weight(value);
    } else if constexpr (std::is_same_v<T, Tensor>) {
        return;
    } else if constexpr (requires { std::variant_size<T>::value; }) {
        std::visit([&](const auto& child) { collect_lora_bases(store, child); }, value);
    } else if constexpr (requires {
                             value.has_value();
                             *value;
                         }) {
        if (value) { collect_lora_bases(store, *value); }
    } else if constexpr (requires {
                             value.begin();
                             value.end();
                         }) {
        for (const auto& child : value) { collect_lora_bases(store, child); }
    } else {
#define COLLECT(member)                                                                            \
    if constexpr (requires { value.member; }) { collect_lora_bases(store, value.member); }
        COLLECT(extra_linears)
        COLLECT(second) COLLECT(full_layers) COLLECT(gdn_layers) COLLECT(token_embedding)
            COLLECT(output_head) COLLECT(projection) COLLECT(output) COLLECT(post_mixer) COLLECT(
                per_layer_input) COLLECT(query) COLLECT(key) COLLECT(value) COLLECT(output_gate)
                COLLECT(query_gate) COLLECT(query_key_value) COLLECT(query_key_gate_value)
                    COLLECT(query_key) COLLECT(gate_value) COLLECT(gate) COLLECT(up) COLLECT(down)
                        COLLECT(gate_up) COLLECT(in_projection) COLLECT(input_projection)
                            COLLECT(control_projection) COLLECT(query_key_value_z) COLLECT(value_z)
                                COLLECT(z) COLLECT(a_projection) COLLECT(b_projection)
                                    COLLECT(a_b_projection) COLLECT(split) COLLECT(op) COLLECT(moe)
                                        COLLECT(router_shared_gate) COLLECT(routed_gate_up)
                                            COLLECT(routed_down) COLLECT(shared_gate_up)
                                                COLLECT(shared_down) COLLECT(vision) COLLECT(common)
                                                    COLLECT(layers) COLLECT(patch_embedding)
                                                        COLLECT(merger_fc1) COLLECT(merger_fc2)
                                                            COLLECT(fc1) COLLECT(fc2) COLLECT(qkv)
                                                                COLLECT(deepstack)
#undef COLLECT
    }
}

inline void bind_lora_bias(ops::LoraStore& store, int layer, const std::string& name,
                           const Tensor& bias, int offset = 0, int rows = 0,
                           int port = ops::kLoraBiasPort) {
    if (!bias.data) { return; }
    if (!rows) { rows = bias.ne[0]; }
    store.register_module(layer, name + ".bias", {bias.data, port, 1, rows});
    Tensor original(static_cast<std::byte*>(bias.data) +
                        offset * (bias.dtype == DType::FP32 ? 4 : 2),
                    bias.dtype, {rows});
    store.register_base(bias.data, port, {}, original);
    store.register_auto(bias.data, {port, offset, rows});
}

inline void bind_lora_auto(ops::LoraStore& store, int layer, const std::string& name,
                           const Weight& weight, const Tensor& bias = {}, int offset = 0,
                           int rows = 0, int port = ops::kLoraAutomaticPort, int source_out = 0,
                           int source_offset = 0) {
    if (!rows) { rows = weight.n; }
    store.register_module(layer, name,
                          {weight.qdata, port, weight.k, rows, source_out, source_offset});
    Tensor original =
        bias.data ? Tensor(static_cast<std::byte*>(bias.data) + offset * 2, DType::BF16, {rows})
                  : Tensor{};
    store.register_base(weight.qdata, port, {{weight, offset, 0, rows}}, original);
    store.register_auto(weight.qdata, {port, offset, rows});
    // Standalone saved biases use the same source names, without touching base requests.
    if (!source_out) {
        bind_lora_bias(store, layer, name, bias, offset, rows,
                       ops::kLoraBiasPort + port - ops::kLoraAutomaticPort);
    }
}

inline void bind_lora_vision(ops::LoraStore& store, const VisionWeights& vision) {
    if (!vision.extra_linears.empty()) {
        const auto bias = [&](const std::string& name) {
            const auto found = vision.extra_tensors.find(name);
            return found == vision.extra_tensors.end() ? Tensor{} : found->second;
        };
        const bool gemma3  = vision.extra_tensors.contains("projection_norm");
        const bool unified = vision.extra_tensors.contains("patch_norm1/weight");
        for (const auto& [name, weight] : vision.extra_linears) {
            if (name == "patch_embedding") {
                bind_lora_auto(store, -2,
                               gemma3    ? "vision.embeddings.patch_embedding"
                               : unified ? "vision.embed_vision.patch_dense"
                                         : "vision.patch_embedder.input_proj",
                               weight, bias("patch_embedding_bias"));
                continue;
            }
            if (name == "projection") {
                if (!gemma3) {
                    bind_lora_auto(
                        store, -2,
                        unified ? "vision.embed_vision.multimodal_embedder.embedding_projection"
                                : "vision.embed_vision.embedding_projection",
                        weight);
                }
                continue;
            }
            const auto slash  = name.find('/', 7);
            const auto layer  = name.substr(7, slash - 7);
            const auto role   = name.substr(slash + 1);
            const auto prefix = "vision.encoder.layers." + layer + ".";
            if (role == "attention/qkv") {
                for (int part = 0; part < 3; ++part) {
                    bind_lora_auto(store, -2,
                                   prefix + "self_attn." +
                                       (part == 0   ? "q_proj"
                                        : part == 1 ? "k_proj"
                                                    : "v_proj"),
                                   weight, bias(name + "_bias"), part * weight.n / 3, weight.n / 3,
                                   ops::kLoraAutomaticPort + part);
                }
                continue;
            }
            std::string module;
            if (role == "attention/query") {
                module = "self_attn.q_proj.linear";
            } else if (role == "attention/key") {
                module = "self_attn.k_proj.linear";
            } else if (role == "attention/value") {
                module = "self_attn.v_proj.linear";
            } else if (role == "attention/output") {
                module = gemma3 ? "self_attn.out_proj" : "self_attn.o_proj.linear";
            } else if (role == "mlp/gate") {
                module = "mlp.gate_proj.linear";
            } else if (role == "mlp/up") {
                module = "mlp.up_proj.linear";
            } else if (role == "mlp/down") {
                module = "mlp.down_proj.linear";
            } else if (role == "mlp/fc1") {
                module = "mlp.fc1";
            } else if (role == "mlp/fc2") {
                module = "mlp.fc2";
            } else {
                throw std::logic_error("unknown Gemma vision adapter projection");
            }
            bind_lora_auto(store, -2, prefix + module, weight, bias(name + "_bias"));
        }
        return;
    }
    const auto& common = vision.common;
    for (const std::string name :
         {"vision.patch_embed.proj", "vision.embeddings.patch_embedding"}) {
        bind_lora_auto(store, -2, name, common.patch_embedding, common.patch_embedding_bias);
    }
    for (std::size_t layer = 0; layer < common.layers.size(); ++layer) {
        const auto& block        = common.layers[layer];
        const std::string qwen   = "vision.blocks." + std::to_string(layer) + ".";
        const std::string siglip = "vision.encoder.layers." + std::to_string(layer) + ".";
        const int hidden         = block.qkv.n / 3;
        bind_lora_bias(store, -2, qwen + "attn.qkv", block.qkv_bias, 0, 0, ops::kLoraBiasPort + 10);
        for (int part = 0; part < 3; ++part) {
            const auto name = part == 0 ? "q_proj" : part == 1 ? "k_proj" : "v_proj";
            bind_lora_auto(store, -2, siglip + "self_attn." + name, block.qkv, block.qkv_bias,
                           part * hidden, hidden, ops::kLoraAutomaticPort + part);
            bind_lora_auto(store, -2, qwen + "attn.qkv", block.qkv, block.qkv_bias, part * hidden,
                           hidden, ops::kLoraAutomaticPort + part, 3 * hidden, part * hidden);
        }
        bind_lora_auto(store, -2, qwen + "attn.proj", block.output, block.output_bias);
        bind_lora_auto(store, -2, siglip + "self_attn.out_proj", block.output, block.output_bias);
        for (const auto& prefix : {qwen, siglip}) {
            bind_lora_auto(store, -2, prefix + "mlp.fc1", block.fc1, block.fc1_bias);
            bind_lora_auto(store, -2, prefix + "mlp.fc2", block.fc2, block.fc2_bias);
        }
        bind_lora_auto(store, -2, qwen + "mlp.linear_fc1", block.fc1, block.fc1_bias);
        bind_lora_auto(store, -2, qwen + "mlp.linear_fc2", block.fc2, block.fc2_bias);
        bind_lora_bias(store, -2, qwen + "norm1", block.norm1_bias);
        bind_lora_bias(store, -2, qwen + "norm2", block.norm2_bias);
        bind_lora_bias(store, -2, siglip + "layer_norm1", block.norm1_bias);
        bind_lora_bias(store, -2, siglip + "layer_norm2", block.norm2_bias);
    }
    for (const std::string name :
         {"vision.merger.linear_fc1", "vision.merger.mlp.0", "vision.projector.linear_1"}) {
        bind_lora_auto(store, -2, name, common.merger_fc1, common.merger_fc1_bias);
    }
    for (const std::string name :
         {"vision.merger.linear_fc2", "vision.merger.mlp.2", "vision.projector.linear_2"}) {
        bind_lora_auto(store, -2, name, vision.merger_fc2, vision.merger_fc2_bias);
    }
    bind_lora_bias(store, -2, "vision.post_layernorm", common.post_norm_bias);
    bind_lora_bias(store, -2, "vision.merger.norm", common.merger_norm_bias);
    bind_lora_bias(store, -2, "vision.merger.ln_q", common.merger_norm_bias);
    bind_lora_bias(store, -2, "vision.projector.layer_norm", common.merger_norm_bias);
    for (std::size_t i = 0; i < vision.deepstack.size(); ++i) {
        const auto& merger = vision.deepstack[i];
        const auto prefix  = "vision.deepstack_merger_list." + std::to_string(i) + ".";
        bind_lora_auto(store, -2, prefix + "linear_fc1", merger.fc1, merger.fc1_bias);
        bind_lora_auto(store, -2, prefix + "linear_fc2", merger.fc2, merger.fc2_bias);
        bind_lora_bias(store, -2, prefix + "norm", merger.norm_bias);
    }
}

template <class Runtime>
void bind_lora_globals(ops::LoraStore& store, const Runtime& runtime) {
    collect_lora_bases(store, runtime);
    const auto& embedding = runtime.token_embedding;
    if (embedding.qdata) {
        store.register_replacement(embedding.qdata, ops::kLoraEmbeddingPort);
        for (const std::string name :
             {"embed_tokens", "embedding", "tok_embeddings", "word_embeddings"}) {
            store.register_module(
                -1, name, {embedding.qdata, ops::kLoraEmbeddingPort, embedding.n, embedding.k});
        }
        store.register_base(embedding.qdata, ops::kLoraEmbeddingPort,
                            {{embedding, 0, 0, embedding.k, true}});
    }
    if (runtime.output_head.qdata) {
        store.register_replacement(runtime.output_head.qdata, ops::kLoraAutomaticPort);
        bind_lora_auto(store, -1, "lm_head", runtime.output_head);
    }
    if (runtime.vision) { bind_lora_vision(store, *runtime.vision); }
}
} // namespace sinfer::family
