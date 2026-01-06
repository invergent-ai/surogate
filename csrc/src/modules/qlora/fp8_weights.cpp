// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "fp8_weights.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <fmt/format.h>

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

} // anonymous namespace

FP8WeightsManager::FP8WeightsManager(const Config& config, TensorAllocator& allocator,
                                         const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(&device_props)
{
    if (!config.qlora_config.is_quantized()) {
        throw std::runtime_error("FP8WeightsManager: QLoRA must be enabled");
    }

    // Pre-size the vector but don't allocate GPU memory yet.
    // Each layer's storage will be allocated lazily in load_and_quantize_block()
    // to reduce peak memory during initialization.
    mQuantizedBlocks.resize(mConfig.num_layers);
}

void FP8WeightsManager::allocate_single_block(int layer_idx) {
    auto ctx = mAllocator->with_context("FP8_Weights");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
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

    // Gate+Up projection: (2 * intermediate, hidden)
    {
        auto& w = block.gate_up_proj;
        w.M = 2 * intermediate;
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

    SafeTensorsReader reader(file_name);

    // Allocate temporary load buffer for BF16 weights
    // Size it for the largest weight we need to load
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    std::size_t max_weight_elems = std::max({
        static_cast<std::size_t>(qkv_out) * hidden,
        static_cast<std::size_t>(hidden) * num_q_heads * head_size,
        static_cast<std::size_t>(2 * intermediate) * hidden,
        static_cast<std::size_t>(hidden) * intermediate,
        static_cast<std::size_t>(mConfig.vocab_size) * hidden  // Embedding
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
        load_and_quantize_block(layer, reader, stream);
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

    // Try common embedding weight names
    const std::vector<std::string> embed_names = {
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embeddings.word_embeddings.weight"
    };

    for (const auto& name : embed_names) {
        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mEmbeddings.embedding, /*allow_cast=*/true);
            break;
        }
    }

    // LM head - respect model config's tied_embeddings setting
    if (mConfig.tied_embeddings) {
        // Tied embeddings: lm_head shares memory with embedding
        mEmbeddings.lm_head = mEmbeddings.embedding;
        mEmbeddings.tied_weights = true;
    } else {
        // Separate lm_head: allocate and load
        const std::vector<std::string> lm_head_names = {
            "lm_head.weight",
            "transformer.lm_head.weight",
            "cls.predictions.decoder.weight"
        };

        bool found_lm_head = false;
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

    // Common prefix for model architectures (LLaMA, Qwen, etc.)
    const std::string prefix = fmt::format("model.layers.{}", layer_idx);

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
    const std::string qkv_name = prefix + ".self_attn.qkv_proj.weight";
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
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.q_proj.weight")) {
            entry->read_tensor(q_view, true);
        }

        // Load K into the middle part
        Tensor k_view = mLoadBuffer;
        k_view.Data = mLoadBuffer.Data + q_out * hidden * sizeof(nv_bfloat16);
        k_view.Sizes[0] = kv_out;
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.k_proj.weight")) {
            entry->read_tensor(k_view, true);
        }

        // Load V into the last part
        Tensor v_view = mLoadBuffer;
        v_view.Data = mLoadBuffer.Data + (q_out + kv_out) * hidden * sizeof(nv_bfloat16);
        v_view.Sizes[0] = kv_out;
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.v_proj.weight")) {
            entry->read_tensor(v_view, true);
        }

        // Restore full shape and quantize
        mLoadBuffer.Sizes[0] = qkv_out;
        quantize_and_store(block.qkv_proj, mLoadBuffer, qkv_out, hidden, stream);
    }

    // Output projection
    load_and_quantize(prefix + ".self_attn.o_proj.weight",
                      block.out_proj, hidden, num_q_heads * head_size);

    // MLP projections (handle both fused and separate gate/up)
    const std::string gate_up_name = prefix + ".mlp.gate_up_proj.weight";
    if (find_entry_opt(reader, gate_up_name)) {
        load_and_quantize(gate_up_name, block.gate_up_proj, 2 * intermediate, hidden);
    } else {
        // Separate gate and up - load into parts of mLoadBuffer
        // Layout: [up; gate] - up in first half, gate in second half
        mLoadBuffer.Sizes[0] = 2 * intermediate;
        mLoadBuffer.Sizes[1] = hidden;
        mLoadBuffer.Rank = 2;

        // Load up into the first part
        Tensor up_view = mLoadBuffer;
        up_view.Sizes[0] = intermediate;
        if (const auto* entry = find_entry_opt(reader, prefix + ".mlp.up_proj.weight")) {
            entry->read_tensor(up_view, true);
        }

        // Load gate into the second part
        Tensor gate_view = mLoadBuffer;
        gate_view.Data = mLoadBuffer.Data + intermediate * hidden * sizeof(nv_bfloat16);
        gate_view.Sizes[0] = intermediate;
        if (const auto* entry = find_entry_opt(reader, prefix + ".mlp.gate_proj.weight")) {
            entry->read_tensor(gate_view, true);
        }

        // Restore full shape and quantize
        mLoadBuffer.Sizes[0] = 2 * intermediate;
        quantize_and_store(block.gate_up_proj, mLoadBuffer, 2 * intermediate, hidden, stream);
    }

    // Down projection
    load_and_quantize(prefix + ".mlp.down_proj.weight",
                      block.down_proj, hidden, intermediate);

    // Layer norms (not quantized, just copy)
    if (const auto* entry = find_entry_opt(reader, prefix + ".input_layernorm.weight")) {
        entry->read_tensor(block.ln1_weight, true);
    }
    if (const auto* entry = find_entry_opt(reader, prefix + ".post_attention_layernorm.weight")) {
        entry->read_tensor(block.ln2_weight, true);
    }

    // QK-norm weights (for models like Qwen3)
    if (mConfig.use_qk_norm && block.q_norm_weight.has_value() && block.k_norm_weight.has_value()) {
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.q_norm.weight")) {
            entry->read_tensor(block.q_norm_weight.value(), true);
        }
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.k_norm.weight")) {
            entry->read_tensor(block.k_norm_weight.value(), true);
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
    for (const auto& block : mQuantizedBlocks) {
        total += block.bytes();
    }
    total += mEmbeddings.bytes();
    return total;
}

float FP8WeightsManager::memory_savings_ratio() const {
    // Calculate what BF16 storage would have been
    std::size_t bf16_bytes = 0;
    for (const auto& block : mQuantizedBlocks) {
        bf16_bytes += static_cast<std::size_t>(block.qkv_proj.M) * block.qkv_proj.K * 2;
        bf16_bytes += static_cast<std::size_t>(block.out_proj.M) * block.out_proj.K * 2;
        bf16_bytes += static_cast<std::size_t>(block.gate_up_proj.M) * block.gate_up_proj.K * 2;
        bf16_bytes += static_cast<std::size_t>(block.down_proj.M) * block.down_proj.K * 2;
        bf16_bytes += block.ln1_weight.bytes();
        bf16_bytes += block.ln2_weight.bytes();
    }
    bf16_bytes += mEmbeddings.bytes();

    std::size_t actual_bytes = quantized_weights_bytes();
    return 1.0f - static_cast<float>(actual_bytes) / static_cast<float>(bf16_bytes);
}

} // namespace modules
