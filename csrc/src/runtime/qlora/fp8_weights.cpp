// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "fp8_weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <cuda_bf16.h>

#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace modules {

namespace {

// Helper to find an entry by name without throwing
const SafeTensorEntry* find_entry_opt(const SafeTensorsReader& reader, std::string_view name) {
    for (const auto& entry : reader.entries()) {
        if (entry.name() == name) {
            return &entry;
        }
    }
    return nullptr;
}

void scan_embedding_for_nan_once(const Tensor& t, cudaStream_t stream, long hidden, std::string_view tag) {
    static bool scanned = false;
    if (scanned || !t.Data || t.nelem() == 0) {
        return;
    }
    scanned = true;
    fprintf(stderr, "[EMB_WT_SCAN] tag=%.*s dtype=%s nelem=%zu hidden=%ld\n",
            static_cast<int>(tag.size()), tag.data(),
            dtype_to_str(t.DType), t.nelem(), hidden);

    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t chunk = 1 << 20; // 1M elements per chunk
    if (t.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> host(chunk);
        for (std::size_t off = 0; off < total; off += chunk) {
            const std::size_t n = std::min(chunk, total - off);
            cudaMemcpy(host.data(),
                       t.Data + off * sizeof(nv_bfloat16),
                       n * sizeof(nv_bfloat16),
                       cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < n; ++i) {
                const float v = static_cast<float>(host[i]);
                if (std::isnan(v) || std::isinf(v)) {
                    const std::size_t idx = off + i;
                    const std::size_t row = hidden > 0 ? idx / static_cast<std::size_t>(hidden) : 0;
                    const std::size_t col = hidden > 0 ? idx % static_cast<std::size_t>(hidden) : idx;
                    fprintf(stderr, "[EMB_WT_NAN] tag=%.*s idx=%zu row=%zu col=%zu\n",
                            static_cast<int>(tag.size()), tag.data(), idx, row, col);
                    throw std::runtime_error("Embedding weights contain NaNs/Inf");
                }
            }
        }
    } else if (t.DType == ETensorDType::FP32) {
        std::vector<float> host(chunk);
        for (std::size_t off = 0; off < total; off += chunk) {
            const std::size_t n = std::min(chunk, total - off);
            cudaMemcpy(host.data(),
                       t.Data + off * sizeof(float),
                       n * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (std::size_t i = 0; i < n; ++i) {
                const float v = host[i];
                if (std::isnan(v) || std::isinf(v)) {
                    const std::size_t idx = off + i;
                    const std::size_t row = hidden > 0 ? idx / static_cast<std::size_t>(hidden) : 0;
                    const std::size_t col = hidden > 0 ? idx % static_cast<std::size_t>(hidden) : idx;
                    fprintf(stderr, "[EMB_WT_NAN] tag=%.*s idx=%zu row=%zu col=%zu\n",
                            static_cast<int>(tag.size()), tag.data(), idx, row, col);
                    throw std::runtime_error("Embedding weights contain NaNs/Inf");
                }
            }
        }
    } else {
        fprintf(stderr, "[EMB_WT_SCAN] tag=%.*s skipped unsupported dtype=%s\n",
                static_cast<int>(tag.size()), tag.data(), dtype_to_str(t.DType));
    }
}

const HfMappingSpec* resolve_hf_spec(const HfMapping* mapping,
                                     const std::string& internal_name,
                                     int& layer_idx,
                                     std::string& resolved_name) {
    if (!mapping) {
        return nullptr;
    }
    const HfMappingSpec* spec = mapping->find(internal_name, layer_idx);
    if (!spec) {
        return nullptr;
    }
    resolved_name = internal_name;
    if (spec->kind == HfMappingSpec::Kind::TiedTo && !spec->target.empty()) {
        int tied_layer = -1;
        const HfMappingSpec* tied = mapping->find(spec->target, tied_layer);
        if (tied) {
            resolved_name = spec->target;
            if (tied_layer >= 0) {
                layer_idx = tied_layer;
            }
            return tied;
        }
    }
    return spec;
}

bool load_tensor_from_spec(const SafeTensorsReader& reader,
                           const HfMappingSpec& spec,
                           std::string_view internal_name,
                           int layer_idx,
                           int expert_idx,
                           Tensor& target,
                           TensorAllocator* allocator,
                           cudaStream_t stream,
                           bool allow_cast,
                           std::string_view warn_tag) {
    auto warn_missing = [&](const std::string& name) {
        if (!spec.optional) {
            std::cerr << "[" << warn_tag << " WARN] weight not found: " << name << "\n";
        }
    };

    switch (spec.kind) {
        case HfMappingSpec::Kind::Direct: {
            const std::string hf_name = HfMapping::format_name(
                spec.source.empty() ? std::string(internal_name) : spec.source, layer_idx, expert_idx);
            if (const auto* entry = find_entry_opt(reader, hf_name)) {
                entry->read_tensor(target, allow_cast);
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        case HfMappingSpec::Kind::Fuse: {
            if (spec.dim != 0 || spec.sources.empty()) {
                return false;
            }
            const std::size_t elem_size = get_dtype_size(target.DType);
            long offset = 0;
            bool any_loaded = false;
            for (const auto& src : spec.sources) {
                const std::string hf_name = HfMapping::format_name(src, layer_idx, expert_idx);
                if (const auto* entry = find_entry_opt(reader, hf_name)) {
                    if (!entry->shape().empty()) {
                        const long rows = entry->shape().at(0);
                        Tensor slice = target;
                        slice.Sizes[0] = rows;
                        slice.Data = target.Data + static_cast<std::size_t>(offset) *
                                     static_cast<std::size_t>(target.Sizes[1]) * elem_size;
                        entry->read_tensor(slice, allow_cast);
                        offset += rows;
                        any_loaded = true;
                    }
                } else {
                    warn_missing(hf_name);
                }
            }
            if (any_loaded && offset != target.Sizes[0]) {
                std::cerr << "[" << warn_tag << " WARN] fuse size mismatch for "
                          << internal_name << " (loaded " << offset
                          << " rows, expected " << target.Sizes[0] << ")\n";
            }
            return any_loaded;
        }
        case HfMappingSpec::Kind::Split: {
            if (spec.dim != 0 || spec.ranges.empty()) {
                return false;
            }
            const auto [start, end] = spec.ranges.front();
            if (start < 0 || end <= start) {
                return false;
            }
            const std::string hf_name = HfMapping::format_name(spec.source, layer_idx, expert_idx);
            if (const auto* entry = find_entry_opt(reader, hf_name)) {
                long stride = 1;
                for (std::size_t i = 1; i < entry->shape().size(); ++i) {
                    stride *= entry->shape()[i];
                }
                const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(start) * stride;
                entry->read_raw(target, offset, target.nelem(), allow_cast);
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        case HfMappingSpec::Kind::Transform: {
            if (spec.fn != "transpose") {
                return false;
            }
            if (!allocator) {
                return false;
            }
            const std::string hf_name = HfMapping::format_name(spec.source, layer_idx, expert_idx);
            if (const auto* entry = find_entry_opt(reader, hf_name)) {
                if (entry->shape().size() != 2 || target.Rank != 2) {
                    return false;
                }
                Tensor tmp = allocator->allocate(target.DType, ("hf_tmp_" + std::string(internal_name)).c_str(),
                                                 EAllocationType::ON_DEVICE,
                                                 {entry->shape().at(0), entry->shape().at(1)});
                entry->read_tensor(tmp, allow_cast);
                transpose(target, tmp, static_cast<int>(entry->shape().at(0)),
                          static_cast<int>(entry->shape().at(1)), stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                return true;
            }
            warn_missing(hf_name);
            return false;
        }
        default:
            return false;
    }
}

} // anonymous namespace

FP8WeightsManager::FP8WeightsManager(const Config& config, TensorAllocator& allocator,
                                         const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(&device_props)
    , mHfMapping(config.hf_mapping)
{
    if (!config.qlora_config.is_quantized()) {
        throw std::runtime_error("FP8WeightsManager: QLoRA must be enabled");
    }

    // Pre-size the vector but don't allocate GPU memory yet.
    // Each layer's storage will be allocated lazily in load_and_quantize_block()
    // to reduce peak memory during initialization.
    if (is_moe()) {
        mMoEBlocks.resize(mConfig.num_layers);
    } else {
        mQuantizedBlocks.resize(mConfig.num_layers);
    }
}

void FP8WeightsManager::allocate_single_block(int layer_idx) {
    auto ctx = mAllocator->with_context("FP8_Weights");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int mlp_M = mConfig.mlp_up_factor * intermediate;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int block_size = mConfig.qlora_config.block_size();

    auto& block = mQuantizedBlocks[layer_idx];

    // QKV projection: (hidden, (num_q + 2*num_kv) * head_size)
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    {
        auto& w = block.qkv_proj;
        w.M = qkv_out;
        w.K = hidden;
        w.block_size = block_size;
        auto [scale_rows, scale_cols] = mConfig.qlora_config.scale_config.num_blocks(w.M, w.K);

        // Allocate FP8 data
        w.data = mAllocator->allocate(ETensorDType::FP8_E4M3, "qkv_fp8",
                                      EAllocationType::ON_DEVICE, {(long)w.M, (long)w.K});

        // Allocate block scales
        w.block_scales = mAllocator->allocate(ETensorDType::FP32, "qkv_scales",
                                               EAllocationType::ON_DEVICE, {(long)scale_rows, (long)scale_cols});
    }

    // Output projection: (hidden, num_q * head_size)
    {
        auto& w = block.out_proj;
        w.M = hidden;
        w.K = num_q_heads * head_size;
        w.block_size = block_size;
        auto [scale_rows, scale_cols] = mConfig.qlora_config.scale_config.num_blocks(w.M, w.K);

        w.data = mAllocator->allocate(ETensorDType::FP8_E4M3, "out_fp8",
                                      EAllocationType::ON_DEVICE, {(long)w.M, (long)w.K});
        w.block_scales = mAllocator->allocate(ETensorDType::FP32, "out_scales",
                                               EAllocationType::ON_DEVICE, {(long)scale_rows, (long)scale_cols});
    }

    // Gate+Up projection: (mlp_up_factor * intermediate, hidden)
    {
        auto& w = block.gate_up_proj;
        w.M = mlp_M;
        w.K = hidden;
        w.block_size = block_size;
        auto [scale_rows, scale_cols] = mConfig.qlora_config.scale_config.num_blocks(w.M, w.K);

        w.data = mAllocator->allocate(ETensorDType::FP8_E4M3, "gate_up_fp8",
                                      EAllocationType::ON_DEVICE, {(long)w.M, (long)w.K});
        w.block_scales = mAllocator->allocate(ETensorDType::FP32, "gate_up_scales",
                                               EAllocationType::ON_DEVICE, {(long)scale_rows, (long)scale_cols});
    }

    // Down projection: (hidden, intermediate)
    {
        auto& w = block.down_proj;
        w.M = hidden;
        w.K = intermediate;
        w.block_size = block_size;
        auto [scale_rows, scale_cols] = mConfig.qlora_config.scale_config.num_blocks(w.M, w.K);

        w.data = mAllocator->allocate(ETensorDType::FP8_E4M3, "down_fp8",
                                      EAllocationType::ON_DEVICE, {(long)w.M, (long)w.K});
        w.block_scales = mAllocator->allocate(ETensorDType::FP32, "down_scales",
                                               EAllocationType::ON_DEVICE, {(long)scale_rows, (long)scale_cols});
    }

    // Layer norm weights (BF16, not quantized)
    block.ln1_weight = mAllocator->allocate(ETensorDType::BF16, "ln1",
                                             EAllocationType::ON_DEVICE, {(long)hidden});
    block.ln2_weight = mAllocator->allocate(ETensorDType::BF16, "ln2",
                                             EAllocationType::ON_DEVICE, {(long)hidden});

    // QK-norm weights (BF16, not quantized) - only for models like Qwen3
    if (mConfig.use_qk_norm) {
        block.q_norm_weight = mAllocator->allocate(ETensorDType::BF16, "q_norm",
                                                    EAllocationType::ON_DEVICE, {(long)head_size});
        block.k_norm_weight = mAllocator->allocate(ETensorDType::BF16, "k_norm",
                                                    EAllocationType::ON_DEVICE, {(long)head_size});
    }
}

void FP8WeightsManager::import_and_quantize(const std::string& file_name,
                                               NCCLCommunicator& comm,
                                               cudaStream_t stream) {
    std::cerr << "[FP8] importing and quantizing weights from " << file_name << "\n";
    std::cerr << "[FP8] block_size=" << mConfig.qlora_config.block_size() << "\n";

    if (is_moe()) {
        std::cerr << "[FP8] MoE mode: " << num_experts() << " experts per layer, "
                  << mConfig.qlora_config.num_experts_per_tok << " active per token\n";
    }

    SafeTensorsReader reader(file_name);

    // Allocate temporary load buffer for BF16 weights
    // Size it for the largest weight we need to load
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int mlp_M = mConfig.mlp_up_factor * intermediate;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // For MoE models, use the per-expert intermediate size
    const int moe_intermediate = mConfig.qlora_config.moe_intermediate_size > 0
        ? mConfig.qlora_config.moe_intermediate_size
        : intermediate;
    const int moe_M = mConfig.mlp_up_factor * moe_intermediate;

    std::size_t max_weight_elems = std::max({
        static_cast<std::size_t>(qkv_out) * hidden,
        static_cast<std::size_t>(hidden) * num_q_heads * head_size,
        static_cast<std::size_t>(mlp_M) * hidden,
        static_cast<std::size_t>(hidden) * intermediate,
        static_cast<std::size_t>(mConfig.vocab_size) * hidden,  // Embedding
        // For MoE: expert gate_up projection
        static_cast<std::size_t>(moe_M) * hidden
    });

    // Allocate temporary load buffer directly (not through allocator) so we can free it after use
    {
        void* ptr = nullptr;
        const std::size_t load_buf_bytes = max_weight_elems * sizeof(nv_bfloat16);
        CUDA_CHECK(cudaMalloc(&ptr, load_buf_bytes));
        mLoadBuffer.Data = static_cast<std::byte*>(ptr);
        mLoadBuffer.DType = ETensorDType::BF16;
        mLoadBuffer.Rank = 1;
        mLoadBuffer.Sizes[0] = static_cast<long>(max_weight_elems);
        std::fill(mLoadBuffer.Sizes.begin() + 1, mLoadBuffer.Sizes.end(), 1);
        mLoadBufferBytes = load_buf_bytes;  // Remember original size for freeing
    }

    // Load embeddings (not quantized)
    load_embeddings(reader, stream);

    // Load and quantize each transformer block
    for (int layer = 0; layer < mConfig.num_layers; ++layer) {
        show_progress_bar(layer, mConfig.num_layers, "[FP8] Quantizing");
        if (is_moe()) {
            load_and_quantize_moe_block(layer, reader, stream);
        } else {
            load_and_quantize_block(layer, reader, stream);
        }
    }

    // Sync before freeing load buffer
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free load buffer (allocated directly, not via allocator)
    if (mLoadBuffer.Data && mLoadBufferBytes > 0) {
        CUDA_CHECK(cudaFree(mLoadBuffer.Data));
        mLoadBuffer = Tensor{};
        mLoadBufferBytes = 0;
    }

    std::cerr << "[QLoRA] import and quantization complete (" << mConfig.num_layers << " layers)\n";
}

void FP8WeightsManager::load_embeddings(SafeTensorsReader& reader, cudaStream_t stream) {
    auto ctx = mAllocator->with_context("FP8_Embeddings");

    const int vocab = mConfig.vocab_size;
    const int hidden = mConfig.hidden_size;

    // Allocate embedding (BF16, not quantized)
    mEmbeddings.embedding = mAllocator->allocate(ETensorDType::BF16, "embedding",
                                                  EAllocationType::ON_DEVICE,
                                                  {(long)vocab, (long)hidden});

    bool embedding_loaded = false;
    if (mHfMapping) {
        int map_layer = -1;
        std::string resolved_name;
        if (const auto* spec = resolve_hf_spec(mHfMapping, "embedding", map_layer, resolved_name)) {
            embedding_loaded = load_tensor_from_spec(reader, *spec, resolved_name, map_layer, -1,
                                                     mEmbeddings.embedding, mAllocator, stream,
                                                     /*allow_cast=*/true, "QLoRA");
        }
    }

    if (!embedding_loaded) {
        // Try common embedding weight names
        const std::vector<std::string> embed_names = {
            "model.embed_tokens.weight",
            "model.embeddings.weight",
            "backbone.embed_tokens.weight",
            "backbone.embeddings.weight",
            "transformer.wte.weight",
            "embeddings.word_embeddings.weight"
        };

        for (const auto& name : embed_names) {
            if (const auto* entry = find_entry_opt(reader, name)) {
                entry->read_tensor(mEmbeddings.embedding, /*allow_cast=*/true);
                embedding_loaded = true;
                break;
            }
        }
    }

    if (embedding_loaded) {
        scan_embedding_for_nan_once(mEmbeddings.embedding, stream, mConfig.hidden_size, "embedding");
    }

    // LM head - respect model config's tied_embeddings setting
    if (mConfig.tied_embeddings) {
        // Tied embeddings: lm_head shares memory with embedding
        mEmbeddings.lm_head = mEmbeddings.embedding;
        mEmbeddings.tied_weights = true;
    } else {
        bool found_lm_head = false;
        if (mHfMapping) {
            int map_layer = -1;
            std::string resolved_name;
            if (const auto* spec = resolve_hf_spec(mHfMapping, "lm_head", map_layer, resolved_name)) {
                if (spec->kind == HfMappingSpec::Kind::TiedTo) {
                    mEmbeddings.lm_head = mEmbeddings.embedding;
                    mEmbeddings.tied_weights = true;
                    found_lm_head = true;
                } else {
                    mEmbeddings.lm_head = mAllocator->allocate(ETensorDType::BF16, "lm_head",
                                                                EAllocationType::ON_DEVICE,
                                                                {(long)vocab, (long)hidden});
                    found_lm_head = load_tensor_from_spec(reader, *spec, resolved_name, map_layer, -1,
                                                          mEmbeddings.lm_head, mAllocator, stream,
                                                          /*allow_cast=*/true, "QLoRA");
                    mEmbeddings.tied_weights = !found_lm_head;
                }
            }
        }

        if (!found_lm_head) {
            // Separate lm_head: allocate and load
            const std::vector<std::string> lm_head_names = {
                "lm_head.weight",
                "transformer.lm_head.weight",
                "cls.predictions.decoder.weight"
            };

            for (const auto& name : lm_head_names) {
                if (const auto* entry = find_entry_opt(reader, name)) {
                    mEmbeddings.lm_head = mAllocator->allocate(ETensorDType::BF16, "lm_head",
                                                                EAllocationType::ON_DEVICE,
                                                                {(long)vocab, (long)hidden});
                    entry->read_tensor(mEmbeddings.lm_head, /*allow_cast=*/true);
                    mEmbeddings.tied_weights = false;
                    found_lm_head = true;
                    break;
                }
            }
        }

        // Fallback to tied if separate lm_head not found (shouldn't happen if config is correct)
        if (!found_lm_head) {
            std::cerr << "[QLoRA WARN] tied_embeddings=false but lm_head.weight not found, using tied\n";
            mEmbeddings.lm_head = mEmbeddings.embedding;
            mEmbeddings.tied_weights = true;
        }
    }
}

void FP8WeightsManager::load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                                   cudaStream_t stream) {
    // Lazily allocate this layer's FP8 storage just before we need it.
    // This reduces peak memory during initialization by avoiding upfront allocation
    // of all layers at once.
    allocate_single_block(layer_idx);

    auto& block = mQuantizedBlocks[layer_idx];
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    auto load_quantized_from_mapping = [&](const std::string& internal_name,
                                           BlockQuantizedWeight& dest,
                                           int M, int K) -> bool {
        if (!mHfMapping) {
            return false;
        }
        int map_layer = -1;
        std::string resolved_name;
        const HfMappingSpec* spec = resolve_hf_spec(mHfMapping, internal_name, map_layer, resolved_name);
        if (!spec) {
            return false;
        }
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;
        if (load_tensor_from_spec(reader, *spec, resolved_name, map_layer, -1,
                                  mLoadBuffer, mAllocator, stream,
                                  /*allow_cast=*/true, "QLoRA")) {
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
            return true;
        }
        return false;
    };

    auto load_tensor_from_mapping = [&](const std::string& internal_name, Tensor& dest) -> bool {
        if (!mHfMapping) {
            return false;
        }
        int map_layer = -1;
        std::string resolved_name;
        const HfMappingSpec* spec = resolve_hf_spec(mHfMapping, internal_name, map_layer, resolved_name);
        if (!spec) {
            return false;
        }
        return load_tensor_from_spec(reader, *spec, resolved_name, map_layer, -1,
                                     dest, mAllocator, stream, /*allow_cast=*/true, "QLoRA");
    };

    // Common prefix for model architectures (LLaMA, Qwen, Nemotron, etc.)
    auto pick_layer_prefix = [&]() {
        const std::string model_prefix = fmt::format("model.layers.{}", layer_idx);
        const std::string backbone_prefix = fmt::format("backbone.layers.{}", layer_idx);
        const bool model_has = find_entry_opt(reader, model_prefix + ".self_attn.q_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".self_attn.qkv_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".mixer.q_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".norm.weight")
                               || find_entry_opt(reader, model_prefix + ".input_layernorm.weight");
        return model_has ? model_prefix : backbone_prefix;
    };
    const std::string prefix = pick_layer_prefix();

    auto pick_attn_name = [&](const std::string& self_attn, const std::string& mixer) -> std::string {
        if (find_entry_opt(reader, self_attn)) return self_attn;
        if (find_entry_opt(reader, mixer)) return mixer;
        return self_attn;
    };

    // Helper to load and quantize a weight
    auto load_and_quantize = [&](const std::string& name, BlockQuantizedWeight& dest, int M, int K) {
        // Set up load buffer shape
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;

        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mLoadBuffer, /*allow_cast=*/true);
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
        } else {
            std::cerr << "[QLoRA WARN] layer " << layer_idx << " weight not found: " << name << "\n";
        }
    };

    // Load Q, K, V projections (need to handle both fused and separate)
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    const std::string qkv_internal = fmt::format("blocks[{}].qkv_weight", layer_idx);
    bool qkv_loaded = load_quantized_from_mapping(qkv_internal, block.qkv_proj, qkv_out, hidden);
    if (!qkv_loaded) {
        const std::string qkv_name = pick_attn_name(prefix + ".self_attn.qkv_proj.weight",
                                                    prefix + ".mixer.qkv_proj.weight");
        if (find_entry_opt(reader, qkv_name)) {
            // Fused QKV
            load_and_quantize(qkv_name, block.qkv_proj, qkv_out, hidden);
        } else {
            // Separate Q, K, V - load into parts of mLoadBuffer and then quantize
            const int q_out = num_q_heads * head_size;
            const int kv_out = num_kv_heads * head_size;

            // Set up load buffer for fused QKV
            mLoadBuffer.Sizes[0] = qkv_out;
            mLoadBuffer.Sizes[1] = hidden;
            mLoadBuffer.Rank = 2;

            // Load Q into the first part
            Tensor q_view = mLoadBuffer;
            q_view.Sizes[0] = q_out;
            const std::string q_name = pick_attn_name(prefix + ".self_attn.q_proj.weight",
                                                      prefix + ".mixer.q_proj.weight");
            if (const auto* entry = find_entry_opt(reader, q_name)) {
                entry->read_tensor(q_view, true);
            }

            // Load K into the middle part
            Tensor k_view = mLoadBuffer;
            k_view.Data = mLoadBuffer.Data + q_out * hidden * sizeof(nv_bfloat16);
            k_view.Sizes[0] = kv_out;
            const std::string k_name = pick_attn_name(prefix + ".self_attn.k_proj.weight",
                                                      prefix + ".mixer.k_proj.weight");
            if (const auto* entry = find_entry_opt(reader, k_name)) {
                entry->read_tensor(k_view, true);
            }

            // Load V into the last part
            Tensor v_view = mLoadBuffer;
            v_view.Data = mLoadBuffer.Data + (q_out + kv_out) * hidden * sizeof(nv_bfloat16);
            v_view.Sizes[0] = kv_out;
            const std::string v_name = pick_attn_name(prefix + ".self_attn.v_proj.weight",
                                                      prefix + ".mixer.v_proj.weight");
            if (const auto* entry = find_entry_opt(reader, v_name)) {
                entry->read_tensor(v_view, true);
            }

            // Restore full shape and quantize
            mLoadBuffer.Sizes[0] = qkv_out;
            quantize_and_store(block.qkv_proj, mLoadBuffer, qkv_out, hidden, stream);
        }
    }

    // Output projection
    const std::string out_internal = fmt::format("blocks[{}].out_weight", layer_idx);
    if (!load_quantized_from_mapping(out_internal, block.out_proj, hidden, num_q_heads * head_size)) {
        const std::string out_name = pick_attn_name(prefix + ".self_attn.o_proj.weight",
                                                    prefix + ".mixer.o_proj.weight");
        load_and_quantize(out_name, block.out_proj, hidden, num_q_heads * head_size);
    }

    const int mlp_M = mConfig.mlp_up_factor * intermediate;

    // MLP projections (handle both fused and separate gate/up)
    auto pick_mlp_name = [&](const std::string& mlp_name, const std::string& mixer_name) -> std::string {
        if (find_entry_opt(reader, mlp_name)) return mlp_name;
        if (find_entry_opt(reader, mixer_name)) return mixer_name;
        return mlp_name;
    };

    const std::string gate_up_internal = fmt::format("blocks[{}].mlp_up_weight", layer_idx);
    if (!load_quantized_from_mapping(gate_up_internal, block.gate_up_proj, mlp_M, hidden)) {
        const std::string gate_up_name = pick_mlp_name(prefix + ".mlp.gate_up_proj.weight",
                                                       prefix + ".mixer.gate_up_proj.weight");
        if (find_entry_opt(reader, gate_up_name)) {
            load_and_quantize(gate_up_name, block.gate_up_proj, mlp_M, hidden);
        } else {
            if (mConfig.mlp_up_factor == 1) {
                // Non-gated MLP: only up projection
                mLoadBuffer.Sizes[0] = intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;
                const std::string up_name = pick_mlp_name(prefix + ".mlp.up_proj.weight",
                                                          prefix + ".mixer.up_proj.weight");
                if (const auto* entry = find_entry_opt(reader, up_name)) {
                    entry->read_tensor(mLoadBuffer, true);
                }
                quantize_and_store(block.gate_up_proj, mLoadBuffer, mlp_M, hidden, stream);
            } else {
                // Separate gate and up - load into parts of mLoadBuffer
                // Layout: [up; gate] - up in first half, gate in second half
                mLoadBuffer.Sizes[0] = 2 * intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;

                // Load up into the first part
                Tensor up_view = mLoadBuffer;
                up_view.Sizes[0] = intermediate;
                const std::string up_name = pick_mlp_name(prefix + ".mlp.up_proj.weight",
                                                          prefix + ".mixer.up_proj.weight");
                if (const auto* entry = find_entry_opt(reader, up_name)) {
                    entry->read_tensor(up_view, true);
                }

                // Load gate into the second part
                Tensor gate_view = mLoadBuffer;
                gate_view.Data = mLoadBuffer.Data + intermediate * hidden * sizeof(nv_bfloat16);
                gate_view.Sizes[0] = intermediate;
                const std::string gate_name = pick_mlp_name(prefix + ".mlp.gate_proj.weight",
                                                            prefix + ".mixer.gate_proj.weight");
                if (const auto* entry = find_entry_opt(reader, gate_name)) {
                    entry->read_tensor(gate_view, true);
                }

                // Restore full shape and quantize
                mLoadBuffer.Sizes[0] = 2 * intermediate;
                quantize_and_store(block.gate_up_proj, mLoadBuffer, mlp_M, hidden, stream);
            }
        }
    }

    // Down projection
    const std::string down_internal = fmt::format("blocks[{}].mlp_down_weight", layer_idx);
    if (!load_quantized_from_mapping(down_internal, block.down_proj, hidden, intermediate)) {
        const std::string down_name = pick_mlp_name(prefix + ".mlp.down_proj.weight",
                                                    prefix + ".mixer.down_proj.weight");
        load_and_quantize(down_name, block.down_proj, hidden, intermediate);
    }

    // Layer norms (not quantized, just copy)
    bool ln1_loaded = false;
    bool ln2_loaded = false;
    const std::string ln1_internal = fmt::format("blocks[{}].ln1_weight", layer_idx);
    const std::string ln2_internal = fmt::format("blocks[{}].ln2_weight", layer_idx);
    ln1_loaded = load_tensor_from_mapping(ln1_internal, block.ln1_weight);
    ln2_loaded = load_tensor_from_mapping(ln2_internal, block.ln2_weight);
    if (!ln1_loaded || !ln2_loaded) {
        if (const auto* entry = find_entry_opt(reader, prefix + ".input_layernorm.weight")) {
            entry->read_tensor(block.ln1_weight, true);
            ln1_loaded = true;
        }
        if (const auto* entry = find_entry_opt(reader, prefix + ".post_attention_layernorm.weight")) {
            entry->read_tensor(block.ln2_weight, true);
            ln2_loaded = true;
        }
        if (!ln1_loaded || !ln2_loaded) {
            if (const auto* entry = find_entry_opt(reader, prefix + ".norm.weight")) {
                if (!ln1_loaded) entry->read_tensor(block.ln1_weight, true);
                if (!ln2_loaded) entry->read_tensor(block.ln2_weight, true);
            }
        }
    }

    // QK-norm weights (for models like Qwen3)
    if (mConfig.use_qk_norm && block.q_norm_weight.has_value() && block.k_norm_weight.has_value()) {
        const std::string qn_internal = fmt::format("blocks[{}].q_norm_weight", layer_idx);
        const std::string kn_internal = fmt::format("blocks[{}].k_norm_weight", layer_idx);
        bool qn_loaded = load_tensor_from_mapping(qn_internal, block.q_norm_weight.value());
        bool kn_loaded = load_tensor_from_mapping(kn_internal, block.k_norm_weight.value());
        if (!qn_loaded || !kn_loaded) {
            const std::string qn_name = pick_attn_name(prefix + ".self_attn.q_norm.weight",
                                                       prefix + ".mixer.q_norm.weight");
            const std::string kn_name = pick_attn_name(prefix + ".self_attn.k_norm.weight",
                                                       prefix + ".mixer.k_norm.weight");
            if (!qn_loaded) {
                if (const auto* entry = find_entry_opt(reader, qn_name)) {
                    entry->read_tensor(block.q_norm_weight.value(), true);
                }
            }
            if (!kn_loaded) {
                if (const auto* entry = find_entry_opt(reader, kn_name)) {
                    entry->read_tensor(block.k_norm_weight.value(), true);
                }
            }
        }
    }
}

void FP8WeightsManager::quantize_and_store(BlockQuantizedWeight& dest, const Tensor& src,
                                              int M, int K, cudaStream_t stream) {
    // Perform per-block quantization using our kernel
    quantize_per_block(
        dest.data.get<__nv_fp8_e4m3>(),
        dest.block_scales.get<float>(),
        src.get<nv_bfloat16>(),
        M, K,
        dest.block_size,
        *mDeviceProps,
        stream);
}

std::size_t FP8WeightsManager::quantized_weights_bytes() const {
    std::size_t total = 0;
    if (is_moe()) {
        for (const auto& block : mMoEBlocks) {
            total += block.bytes();
        }
    } else {
        for (const auto& block : mQuantizedBlocks) {
            total += block.bytes();
        }
    }
    total += mEmbeddings.bytes();
    return total;
}

float FP8WeightsManager::memory_savings_ratio() const {
    // Calculate what BF16 storage would have been
    std::size_t bf16_bytes = 0;

    if (is_moe()) {
        for (const auto& block : mMoEBlocks) {
            bf16_bytes += static_cast<std::size_t>(block.qkv_proj.M) * block.qkv_proj.K * 2;
            bf16_bytes += static_cast<std::size_t>(block.out_proj.M) * block.out_proj.K * 2;
            bf16_bytes += block.ln1_weight.bytes();
            bf16_bytes += block.ln2_weight.bytes();
            // Router gate is BF16 already
            bf16_bytes += block.router_gate.bytes();
            // Experts
            for (const auto& expert : block.experts) {
                bf16_bytes += static_cast<std::size_t>(expert.gate_up_proj.M) * expert.gate_up_proj.K * 2;
                bf16_bytes += static_cast<std::size_t>(expert.down_proj.M) * expert.down_proj.K * 2;
            }
            if (block.shared_expert.has_value()) {
                bf16_bytes += static_cast<std::size_t>(block.shared_expert->gate_up_proj.M) * block.shared_expert->gate_up_proj.K * 2;
                bf16_bytes += static_cast<std::size_t>(block.shared_expert->down_proj.M) * block.shared_expert->down_proj.K * 2;
            }
        }
    } else {
        for (const auto& block : mQuantizedBlocks) {
            bf16_bytes += static_cast<std::size_t>(block.qkv_proj.M) * block.qkv_proj.K * 2;
            bf16_bytes += static_cast<std::size_t>(block.out_proj.M) * block.out_proj.K * 2;
            bf16_bytes += static_cast<std::size_t>(block.gate_up_proj.M) * block.gate_up_proj.K * 2;
            bf16_bytes += static_cast<std::size_t>(block.down_proj.M) * block.down_proj.K * 2;
            bf16_bytes += block.ln1_weight.bytes();
            bf16_bytes += block.ln2_weight.bytes();
        }
    }
    bf16_bytes += mEmbeddings.bytes();

    std::size_t actual_bytes = quantized_weights_bytes();
    return 1.0f - static_cast<float>(actual_bytes) / static_cast<float>(bf16_bytes);
}

// ============================================================================
// MoE-specific helpers
// ============================================================================

void FP8WeightsManager::allocate_fp8_weight(BlockQuantizedWeight& weight, int M, int K,
                                             const std::string& name_prefix) {
    weight.M = M;
    weight.K = K;
    weight.block_size = mConfig.qlora_config.block_size();
    auto [scale_rows, scale_cols] = mConfig.qlora_config.scale_config.num_blocks(M, K);

    weight.data = mAllocator->allocate(ETensorDType::FP8_E4M3, (name_prefix + "_fp8").c_str(),
                                       EAllocationType::ON_DEVICE, {(long)M, (long)K});
    weight.block_scales = mAllocator->allocate(ETensorDType::FP32, (name_prefix + "_scales").c_str(),
                                                EAllocationType::ON_DEVICE,
                                                {(long)scale_rows, (long)scale_cols});
}

void FP8WeightsManager::allocate_moe_block(int layer_idx) {
    auto ctx = mAllocator->with_context("FP8_MoE_Weights");

    const int hidden = mConfig.hidden_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int n_experts = num_experts();
    const int moe_intermediate = mConfig.qlora_config.moe_intermediate_size > 0
        ? mConfig.qlora_config.moe_intermediate_size
        : mConfig.intermediate_size;
    const int shared_intermediate = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_intermediate;
    const int shared_M = mConfig.mlp_up_factor * shared_intermediate;
    const int moe_M = mConfig.mlp_up_factor * moe_intermediate;
    const bool use_shared = mConfig.qlora_config.num_shared_experts > 0;

    auto& block = mMoEBlocks[layer_idx];

    // QKV projection: (hidden, (num_q + 2*num_kv) * head_size)
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    allocate_fp8_weight(block.qkv_proj, qkv_out, hidden,
                        fmt::format("l{}_qkv", layer_idx));

    // Output projection: (hidden, num_q * head_size)
    allocate_fp8_weight(block.out_proj, hidden, num_q_heads * head_size,
                        fmt::format("l{}_out", layer_idx));

    // Layer norm weights (BF16, not quantized)
    block.ln1_weight = mAllocator->allocate(ETensorDType::BF16, "ln1",
                                             EAllocationType::ON_DEVICE, {(long)hidden});
    block.ln2_weight = mAllocator->allocate(ETensorDType::BF16, "ln2",
                                             EAllocationType::ON_DEVICE, {(long)hidden});

    // QK-norm weights (for models like Qwen3)
    if (mConfig.use_qk_norm) {
        block.q_norm_weight = mAllocator->allocate(ETensorDType::BF16, "q_norm",
                                                    EAllocationType::ON_DEVICE, {(long)head_size});
        block.k_norm_weight = mAllocator->allocate(ETensorDType::BF16, "k_norm",
                                                    EAllocationType::ON_DEVICE, {(long)head_size});
    }

    // Router gate (BF16, not quantized - small tensor)
    // NOTE: Shape matches HuggingFace (num_experts, hidden). `model_forward.hpp` uses matmul(TN)
    //       and expects router_gate as (num_experts, hidden) like other linear weights.
    block.router_gate = mAllocator->allocate(ETensorDType::BF16, "router_gate",
                                             EAllocationType::ON_DEVICE,
                                             {(long)n_experts, (long)hidden});

    // Expert weights
    block.experts.resize(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        auto& expert = block.experts[e];

        // Gate+Up projection: (mlp_up_factor * moe_intermediate, hidden)
        allocate_fp8_weight(expert.gate_up_proj, moe_M, hidden,
                            fmt::format("l{}_e{}_gate_up", layer_idx, e));

        // Down projection: (hidden, moe_intermediate)
        allocate_fp8_weight(expert.down_proj, hidden, moe_intermediate,
                            fmt::format("l{}_e{}_down", layer_idx, e));
    }

    // Shared expert weights (optional)
    if (use_shared) {
        block.shared_expert.emplace();
        allocate_fp8_weight(block.shared_expert->gate_up_proj, shared_M, hidden,
                            fmt::format("l{}_shared_gate_up", layer_idx));
        allocate_fp8_weight(block.shared_expert->down_proj, hidden, shared_intermediate,
                            fmt::format("l{}_shared_down", layer_idx));
    }
}

void FP8WeightsManager::load_and_quantize_moe_block(int layer_idx, SafeTensorsReader& reader,
                                                     cudaStream_t stream) {
    // Lazily allocate this layer's storage
    allocate_moe_block(layer_idx);

    auto& block = mMoEBlocks[layer_idx];
    const int hidden = mConfig.hidden_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int n_experts = num_experts();
    const int moe_intermediate = mConfig.qlora_config.moe_intermediate_size > 0
        ? mConfig.qlora_config.moe_intermediate_size
        : mConfig.intermediate_size;
    const int shared_intermediate = (mConfig.qlora_config.moe_shared_expert_intermediate_size > 0)
        ? mConfig.qlora_config.moe_shared_expert_intermediate_size
        : moe_intermediate;
    const int shared_M = mConfig.mlp_up_factor * shared_intermediate;

    auto load_quantized_from_mapping = [&](const std::string& internal_name,
                                           BlockQuantizedWeight& dest,
                                           int M, int K,
                                           int expert_idx = -1) -> bool {
        if (!mHfMapping) {
            return false;
        }
        int map_layer = -1;
        std::string resolved_name;
        const HfMappingSpec* spec = resolve_hf_spec(mHfMapping, internal_name, map_layer, resolved_name);
        if (!spec) {
            return false;
        }
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;
        if (load_tensor_from_spec(reader, *spec, resolved_name, map_layer, expert_idx,
                                  mLoadBuffer, mAllocator, stream,
                                  /*allow_cast=*/true, "FP8")) {
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
            return true;
        }
        return false;
    };

    auto load_tensor_from_mapping = [&](const std::string& internal_name, Tensor& dest,
                                        int expert_idx = -1) -> bool {
        if (!mHfMapping) {
            return false;
        }
        int map_layer = -1;
        std::string resolved_name;
        const HfMappingSpec* spec = resolve_hf_spec(mHfMapping, internal_name, map_layer, resolved_name);
        if (!spec) {
            return false;
        }
        return load_tensor_from_spec(reader, *spec, resolved_name, map_layer, expert_idx,
                                     dest, mAllocator, stream,
                                     /*allow_cast=*/true, "FP8");
    };

    auto pick_layer_prefix = [&]() {
        const std::string model_prefix = fmt::format("model.layers.{}", layer_idx);
        const std::string backbone_prefix = fmt::format("backbone.layers.{}", layer_idx);
        const bool model_has = find_entry_opt(reader, model_prefix + ".self_attn.q_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".self_attn.qkv_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".mixer.q_proj.weight")
                               || find_entry_opt(reader, model_prefix + ".norm.weight")
                               || find_entry_opt(reader, model_prefix + ".input_layernorm.weight");
        return model_has ? model_prefix : backbone_prefix;
    };
    const std::string prefix = pick_layer_prefix();

    auto pick_attn_name = [&](const std::string& self_attn, const std::string& mixer) -> std::string {
        if (find_entry_opt(reader, self_attn)) return self_attn;
        if (find_entry_opt(reader, mixer)) return mixer;
        return self_attn;
    };

    // Helper to load and quantize a weight
    auto load_and_quantize = [&](const std::string& name, BlockQuantizedWeight& dest, int M, int K) {
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;

        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mLoadBuffer, /*allow_cast=*/true);
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
        } else {
            std::cerr << "[FP8 WARN] layer " << layer_idx << " weight not found: " << name << "\n";
        }
    };

    // =========================================================================
    // Attention weights (same as dense)
    // =========================================================================
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    const std::string qkv_internal = fmt::format("blocks[{}].qkv_weight", layer_idx);
    bool qkv_loaded = load_quantized_from_mapping(qkv_internal, block.qkv_proj, qkv_out, hidden);
    if (!qkv_loaded) {
        const std::string qkv_name = pick_attn_name(prefix + ".self_attn.qkv_proj.weight",
                                                    prefix + ".mixer.qkv_proj.weight");
        if (find_entry_opt(reader, qkv_name)) {
            load_and_quantize(qkv_name, block.qkv_proj, qkv_out, hidden);
        } else {
            // Separate Q, K, V - load into parts of mLoadBuffer
            const int q_out = num_q_heads * head_size;
            const int kv_out = num_kv_heads * head_size;

            mLoadBuffer.Sizes[0] = qkv_out;
            mLoadBuffer.Sizes[1] = hidden;
            mLoadBuffer.Rank = 2;

            Tensor q_view = mLoadBuffer;
            q_view.Sizes[0] = q_out;
            const std::string q_name = pick_attn_name(prefix + ".self_attn.q_proj.weight",
                                                      prefix + ".mixer.q_proj.weight");
            if (const auto* entry = find_entry_opt(reader, q_name)) {
                entry->read_tensor(q_view, true);
            }

            Tensor k_view = mLoadBuffer;
            k_view.Data = mLoadBuffer.Data + q_out * hidden * sizeof(nv_bfloat16);
            k_view.Sizes[0] = kv_out;
            const std::string k_name = pick_attn_name(prefix + ".self_attn.k_proj.weight",
                                                      prefix + ".mixer.k_proj.weight");
            if (const auto* entry = find_entry_opt(reader, k_name)) {
                entry->read_tensor(k_view, true);
            }

            Tensor v_view = mLoadBuffer;
            v_view.Data = mLoadBuffer.Data + (q_out + kv_out) * hidden * sizeof(nv_bfloat16);
            v_view.Sizes[0] = kv_out;
            const std::string v_name = pick_attn_name(prefix + ".self_attn.v_proj.weight",
                                                      prefix + ".mixer.v_proj.weight");
            if (const auto* entry = find_entry_opt(reader, v_name)) {
                entry->read_tensor(v_view, true);
            }

            mLoadBuffer.Sizes[0] = qkv_out;
            quantize_and_store(block.qkv_proj, mLoadBuffer, qkv_out, hidden, stream);
        }
    }

    // Output projection
    const std::string out_internal = fmt::format("blocks[{}].out_weight", layer_idx);
    if (!load_quantized_from_mapping(out_internal, block.out_proj, hidden, num_q_heads * head_size)) {
        const std::string out_name = pick_attn_name(prefix + ".self_attn.o_proj.weight",
                                                    prefix + ".mixer.o_proj.weight");
        load_and_quantize(out_name, block.out_proj, hidden, num_q_heads * head_size);
    }

    // Layer norms
    bool ln1_loaded = false;
    bool ln2_loaded = false;
    const std::string ln1_internal = fmt::format("blocks[{}].ln1_weight", layer_idx);
    const std::string ln2_internal = fmt::format("blocks[{}].ln2_weight", layer_idx);
    ln1_loaded = load_tensor_from_mapping(ln1_internal, block.ln1_weight);
    ln2_loaded = load_tensor_from_mapping(ln2_internal, block.ln2_weight);
    if (!ln1_loaded || !ln2_loaded) {
        if (const auto* entry = find_entry_opt(reader, prefix + ".input_layernorm.weight")) {
            entry->read_tensor(block.ln1_weight, true);
            ln1_loaded = true;
        }
        if (const auto* entry = find_entry_opt(reader, prefix + ".post_attention_layernorm.weight")) {
            entry->read_tensor(block.ln2_weight, true);
            ln2_loaded = true;
        }
        if (!ln1_loaded || !ln2_loaded) {
            if (const auto* entry = find_entry_opt(reader, prefix + ".norm.weight")) {
                if (!ln1_loaded) entry->read_tensor(block.ln1_weight, true);
                if (!ln2_loaded) entry->read_tensor(block.ln2_weight, true);
            }
        }
    }

    // QK-norm weights
    if (mConfig.use_qk_norm && block.q_norm_weight.has_value() && block.k_norm_weight.has_value()) {
        const std::string qn_internal = fmt::format("blocks[{}].q_norm_weight", layer_idx);
        const std::string kn_internal = fmt::format("blocks[{}].k_norm_weight", layer_idx);
        bool qn_loaded = load_tensor_from_mapping(qn_internal, block.q_norm_weight.value());
        bool kn_loaded = load_tensor_from_mapping(kn_internal, block.k_norm_weight.value());
        if (!qn_loaded || !kn_loaded) {
            const std::string qn_name = pick_attn_name(prefix + ".self_attn.q_norm.weight",
                                                       prefix + ".mixer.q_norm.weight");
            const std::string kn_name = pick_attn_name(prefix + ".self_attn.k_norm.weight",
                                                       prefix + ".mixer.k_norm.weight");
            if (!qn_loaded) {
                if (const auto* entry = find_entry_opt(reader, qn_name)) {
                    entry->read_tensor(block.q_norm_weight.value(), true);
                }
            }
            if (!kn_loaded) {
                if (const auto* entry = find_entry_opt(reader, kn_name)) {
                    entry->read_tensor(block.k_norm_weight.value(), true);
                }
            }
        }
    }

    // =========================================================================
    // Router gate (BF16, not quantized)
    // =========================================================================
    // Model stores as (num_experts, hidden) and our matmul(TN) expects the same layout.
    {
        const std::string router_internal = fmt::format("blocks[{}].router_weight", layer_idx);
        bool found = load_tensor_from_mapping(router_internal, block.router_gate);
        if (!found) {
            const std::array<std::string, 6> router_names = {
                prefix + ".mlp.gate.weight",
                prefix + ".mlp.router.weight",
                prefix + ".mlp.router.gate.weight",
                prefix + ".mixer.gate.weight",
                prefix + ".mixer.router.weight",
                prefix + ".mixer.router.gate.weight"
            };
            for (const auto& name : router_names) {
                if (const auto* entry = find_entry_opt(reader, name)) {
                    entry->read_tensor(block.router_gate, /*allow_cast=*/true);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "[FP8 WARN] layer " << layer_idx
                          << " router gate not found (tried mlp/mixer variants) - this will cause NaN!\n";
            }
        }
    }

    // =========================================================================
    const int moe_M = mConfig.mlp_up_factor * moe_intermediate;

    // Expert weights
    // =========================================================================
    bool experts_loaded = false;
    if (mHfMapping) {
        const std::string gate_internal = fmt::format("blocks[{}].experts_gate_up", layer_idx);
        const std::string down_internal = fmt::format("blocks[{}].experts_down", layer_idx);
        int gate_layer = -1;
        int down_layer = -1;
        std::string gate_resolved;
        std::string down_resolved;
        const HfMappingSpec* gate_spec = resolve_hf_spec(mHfMapping, gate_internal, gate_layer, gate_resolved);
        const HfMappingSpec* down_spec = resolve_hf_spec(mHfMapping, down_internal, down_layer, down_resolved);
        if (gate_spec && down_spec &&
            gate_spec->kind == HfMappingSpec::Kind::StackExperts &&
            down_spec->kind == HfMappingSpec::Kind::StackExperts &&
            !gate_spec->source.empty() && !down_spec->source.empty()) {
            const int gate_layer_idx = gate_layer >= 0 ? gate_layer : layer_idx;
            const int down_layer_idx = down_layer >= 0 ? down_layer : layer_idx;
            if (gate_spec->num_experts > 0 && gate_spec->num_experts != n_experts) {
                std::cerr << "[FP8 WARN] experts_gate_up num_experts mismatch: mapping="
                          << gate_spec->num_experts << " config=" << n_experts << "\n";
            }
            if (down_spec->num_experts > 0 && down_spec->num_experts != n_experts) {
                std::cerr << "[FP8 WARN] experts_down num_experts mismatch: mapping="
                          << down_spec->num_experts << " config=" << n_experts << "\n";
            }

            std::string gate_pattern = gate_spec->source;
            std::string up_pattern = gate_pattern;
            const std::size_t pos = up_pattern.find("gate_proj");
            if (pos != std::string::npos) {
                up_pattern.replace(pos, 9, "up_proj");
            }

            for (int e = 0; e < n_experts; ++e) {
                auto& expert = block.experts[e];

                // Gate+Up projection
                if (gate_spec->fuse_gate_up && mConfig.mlp_up_factor != 1) {
                    mLoadBuffer.Sizes[0] = 2 * moe_intermediate;
                    mLoadBuffer.Sizes[1] = hidden;
                    mLoadBuffer.Rank = 2;

                    // Up in first half
                    Tensor up_view = mLoadBuffer;
                    up_view.Sizes[0] = moe_intermediate;
                    const std::string up_name = HfMapping::format_name(up_pattern, gate_layer_idx, e);
                    if (const auto* entry = find_entry_opt(reader, up_name)) {
                        entry->read_tensor(up_view, true);
                    } else {
                        std::cerr << "[FP8 WARN] expert up_proj not found: " << up_name << "\n";
                    }

                    // Gate in second half
                    Tensor gate_view = mLoadBuffer;
                    gate_view.Data = mLoadBuffer.Data + moe_intermediate * hidden * sizeof(nv_bfloat16);
                    gate_view.Sizes[0] = moe_intermediate;
                    const std::string gate_name = HfMapping::format_name(gate_pattern, gate_layer_idx, e);
                    if (const auto* entry = find_entry_opt(reader, gate_name)) {
                        entry->read_tensor(gate_view, true);
                    } else {
                        std::cerr << "[FP8 WARN] expert gate_proj not found: " << gate_name << "\n";
                    }

                    mLoadBuffer.Sizes[0] = 2 * moe_intermediate;
                    quantize_and_store(expert.gate_up_proj, mLoadBuffer, moe_M, hidden, stream);
                } else {
                    mLoadBuffer.Sizes[0] = moe_M;
                    mLoadBuffer.Sizes[1] = hidden;
                    mLoadBuffer.Rank = 2;
                    const std::string gate_up_name = HfMapping::format_name(
                        gate_spec->fuse_gate_up ? up_pattern : gate_pattern, gate_layer_idx, e);
                    if (const auto* entry = find_entry_opt(reader, gate_up_name)) {
                        entry->read_tensor(mLoadBuffer, true);
                    } else {
                        std::cerr << "[FP8 WARN] expert gate_up not found: " << gate_up_name << "\n";
                    }
                    quantize_and_store(expert.gate_up_proj, mLoadBuffer, moe_M, hidden, stream);
                }

                // Down projection
                mLoadBuffer.Sizes[0] = hidden;
                mLoadBuffer.Sizes[1] = moe_intermediate;
                mLoadBuffer.Rank = 2;
                const std::string down_name = HfMapping::format_name(down_spec->source, down_layer_idx, e);
                if (const auto* entry = find_entry_opt(reader, down_name)) {
                    entry->read_tensor(mLoadBuffer, true);
                } else {
                    std::cerr << "[FP8 WARN] expert down_proj not found: " << down_name << "\n";
                }
                quantize_and_store(expert.down_proj, mLoadBuffer, hidden, moe_intermediate, stream);
            }
            experts_loaded = true;
        }
    }

    if (!experts_loaded) {
        for (int e = 0; e < n_experts; ++e) {
            auto& expert = block.experts[e];
            const std::string mlp_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);
            const std::string mixer_prefix = fmt::format("{}.mixer.experts.{}", prefix, e);
            const bool use_mixer = !find_entry_opt(reader, mlp_prefix + ".gate_up_proj.weight")
                                   && !find_entry_opt(reader, mlp_prefix + ".up_proj.weight")
                                   && !find_entry_opt(reader, mlp_prefix + ".down_proj.weight")
                                   && (find_entry_opt(reader, mixer_prefix + ".gate_up_proj.weight")
                                       || find_entry_opt(reader, mixer_prefix + ".up_proj.weight")
                                       || find_entry_opt(reader, mixer_prefix + ".down_proj.weight"));
            const std::string exp_prefix = use_mixer ? mixer_prefix : mlp_prefix;

            // Gate+Up projection (may be fused or separate)
            const std::string gate_up_name = exp_prefix + ".gate_up_proj.weight";
            if (find_entry_opt(reader, gate_up_name)) {
                load_and_quantize(gate_up_name, expert.gate_up_proj, moe_M, hidden);
            } else if (mConfig.mlp_up_factor == 1) {
                // Non-gated experts: only up projection
                mLoadBuffer.Sizes[0] = moe_intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;
                if (const auto* entry = find_entry_opt(reader, exp_prefix + ".up_proj.weight")) {
                    entry->read_tensor(mLoadBuffer, true);
                }
                quantize_and_store(expert.gate_up_proj, mLoadBuffer, moe_M, hidden, stream);
            } else {
                // Separate gate and up - fuse them
                mLoadBuffer.Sizes[0] = 2 * moe_intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;

                // Up in first half
                Tensor up_view = mLoadBuffer;
                up_view.Sizes[0] = moe_intermediate;
                if (const auto* entry = find_entry_opt(reader, exp_prefix + ".up_proj.weight")) {
                    entry->read_tensor(up_view, true);
                }

                // Gate in second half
                Tensor gate_view = mLoadBuffer;
                gate_view.Data = mLoadBuffer.Data + moe_intermediate * hidden * sizeof(nv_bfloat16);
                gate_view.Sizes[0] = moe_intermediate;
                if (const auto* entry = find_entry_opt(reader, exp_prefix + ".gate_proj.weight")) {
                    entry->read_tensor(gate_view, true);
                }

                mLoadBuffer.Sizes[0] = 2 * moe_intermediate;
                quantize_and_store(expert.gate_up_proj, mLoadBuffer, moe_M, hidden, stream);
            }

            // Down projection
            load_and_quantize(exp_prefix + ".down_proj.weight",
                              expert.down_proj, hidden, moe_intermediate);
        }
    }

    // Shared expert weights (optional)
    if (block.shared_expert.has_value()) {
        bool gate_up_loaded = false;
        bool down_loaded = false;
        if (mHfMapping) {
            const std::string gate_internal = fmt::format("blocks[{}].shared_expert_gate", layer_idx);
            const std::string up_internal = fmt::format("blocks[{}].shared_expert_up", layer_idx);
            const std::string down_internal = fmt::format("blocks[{}].shared_expert_down", layer_idx);
            int gate_layer = -1;
            int up_layer = -1;
            int down_layer = -1;
            std::string gate_resolved;
            std::string up_resolved;
            std::string down_resolved;
            const HfMappingSpec* gate_spec = resolve_hf_spec(mHfMapping, gate_internal, gate_layer, gate_resolved);
            const HfMappingSpec* up_spec = resolve_hf_spec(mHfMapping, up_internal, up_layer, up_resolved);
            const HfMappingSpec* down_spec = resolve_hf_spec(mHfMapping, down_internal, down_layer, down_resolved);

            if (mConfig.mlp_up_factor == 1) {
                // Non-gated shared expert: only up projection
                mLoadBuffer.Sizes[0] = shared_M;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;
                if (up_spec && load_tensor_from_spec(reader, *up_spec, up_resolved,
                                                     up_layer, -1, mLoadBuffer,
                                                     mAllocator, stream, true, "FP8")) {
                    gate_up_loaded = true;
                } else if (gate_spec && load_tensor_from_spec(reader, *gate_spec, gate_resolved,
                                                             gate_layer, -1, mLoadBuffer,
                                                             mAllocator, stream, true, "FP8")) {
                    gate_up_loaded = true;
                }
                if (gate_up_loaded) {
                    quantize_and_store(block.shared_expert->gate_up_proj, mLoadBuffer, shared_M, hidden, stream);
                }
            } else if (gate_spec && up_spec) {
                // Separate gate and up - fuse them
                mLoadBuffer.Sizes[0] = 2 * shared_intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;

                Tensor up_view = mLoadBuffer;
                up_view.Sizes[0] = shared_intermediate;
                if (load_tensor_from_spec(reader, *up_spec, up_resolved, up_layer, -1,
                                          up_view, mAllocator, stream, true, "FP8")) {
                    // ok
                }

                Tensor gate_view = mLoadBuffer;
                gate_view.Data = mLoadBuffer.Data + shared_intermediate * hidden * sizeof(nv_bfloat16);
                gate_view.Sizes[0] = shared_intermediate;
                if (load_tensor_from_spec(reader, *gate_spec, gate_resolved, gate_layer, -1,
                                          gate_view, mAllocator, stream, true, "FP8")) {
                    // ok
                }

                mLoadBuffer.Sizes[0] = 2 * shared_intermediate;
                quantize_and_store(block.shared_expert->gate_up_proj, mLoadBuffer, shared_M, hidden, stream);
                gate_up_loaded = true;
            } else if (gate_spec) {
                // Fused gate_up mapping
                mLoadBuffer.Sizes[0] = shared_M;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;
                if (load_tensor_from_spec(reader, *gate_spec, gate_resolved,
                                          gate_layer, -1, mLoadBuffer,
                                          mAllocator, stream, true, "FP8")) {
                    quantize_and_store(block.shared_expert->gate_up_proj, mLoadBuffer, shared_M, hidden, stream);
                    gate_up_loaded = true;
                }
            }

            if (down_spec) {
                mLoadBuffer.Sizes[0] = hidden;
                mLoadBuffer.Sizes[1] = shared_intermediate;
                mLoadBuffer.Rank = 2;
                if (load_tensor_from_spec(reader, *down_spec, down_resolved,
                                          down_layer, -1, mLoadBuffer,
                                          mAllocator, stream, true, "FP8")) {
                    quantize_and_store(block.shared_expert->down_proj, mLoadBuffer,
                                       hidden, shared_intermediate, stream);
                    down_loaded = true;
                }
            }
        }

        const std::string mlp_prefix = prefix + ".mlp.shared_expert";
        const std::string mixer_prefix = prefix + ".mixer.shared_expert";
        const bool use_mixer = !find_entry_opt(reader, mlp_prefix + ".gate_up_proj.weight")
                               && !find_entry_opt(reader, mlp_prefix + ".up_proj.weight")
                               && !find_entry_opt(reader, mlp_prefix + ".down_proj.weight")
                               && (find_entry_opt(reader, mixer_prefix + ".gate_up_proj.weight")
                                   || find_entry_opt(reader, mixer_prefix + ".up_proj.weight")
                                   || find_entry_opt(reader, mixer_prefix + ".down_proj.weight"));
        const std::string shared_prefix = use_mixer ? mixer_prefix : mlp_prefix;

        if (!gate_up_loaded) {
            const std::string gate_up_name = shared_prefix + ".gate_up_proj.weight";
            if (find_entry_opt(reader, gate_up_name)) {
                load_and_quantize(gate_up_name, block.shared_expert->gate_up_proj, shared_M, hidden);
                gate_up_loaded = true;
            } else if (mConfig.mlp_up_factor == 1) {
                mLoadBuffer.Sizes[0] = shared_intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;
                if (const auto* entry = find_entry_opt(reader, shared_prefix + ".up_proj.weight")) {
                    entry->read_tensor(mLoadBuffer, true);
                }
                quantize_and_store(block.shared_expert->gate_up_proj, mLoadBuffer, shared_M, hidden, stream);
                gate_up_loaded = true;
            } else {
                mLoadBuffer.Sizes[0] = 2 * shared_intermediate;
                mLoadBuffer.Sizes[1] = hidden;
                mLoadBuffer.Rank = 2;

                Tensor up_view = mLoadBuffer;
                up_view.Sizes[0] = shared_intermediate;
                if (const auto* entry = find_entry_opt(reader, shared_prefix + ".up_proj.weight")) {
                    entry->read_tensor(up_view, true);
                }

                Tensor gate_view = mLoadBuffer;
                gate_view.Data = mLoadBuffer.Data + shared_intermediate * hidden * sizeof(nv_bfloat16);
                gate_view.Sizes[0] = shared_intermediate;
                if (const auto* entry = find_entry_opt(reader, shared_prefix + ".gate_proj.weight")) {
                    entry->read_tensor(gate_view, true);
                }

                mLoadBuffer.Sizes[0] = 2 * shared_intermediate;
                quantize_and_store(block.shared_expert->gate_up_proj, mLoadBuffer, shared_M, hidden, stream);
                gate_up_loaded = true;
            }
        }

        if (!down_loaded) {
            load_and_quantize(shared_prefix + ".down_proj.weight",
                              block.shared_expert->down_proj, hidden, shared_intermediate);
        }
    }
}

} // namespace modules
