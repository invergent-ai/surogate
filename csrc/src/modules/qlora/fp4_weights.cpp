// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "fp4_weights.h"

#include <algorithm>
#include <cmath>
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

FP4WeightsManager::FP4WeightsManager(const Config& config, TensorAllocator& allocator,
                                     const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(&device_props)
{
    if (!config.qlora_config.is_fp4()) {
        throw std::runtime_error("FP4WeightsManager: NVFP4 QLoRA must be enabled");
    }

    // Allocate FP4 weight storage
    allocate_fp4_blocks();
}

FP4WeightsManager::~FP4WeightsManager() {
    // Free global amax device memory
    for (float* ptr : mGlobalAmaxPtrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    mGlobalAmaxPtrs.clear();
}

float* FP4WeightsManager::allocate_global_amax() {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(float)));
    CUDA_CHECK(cudaMemset(ptr, 0, sizeof(float)));
    mGlobalAmaxPtrs.push_back(ptr);
    return ptr;
}

void FP4WeightsManager::allocate_fp4_blocks() {
    auto ctx = mAllocator->with_context("FP4_Weights");

    const int num_layers = mConfig.num_layers;
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    mFP4Blocks.resize(num_layers);

    for (int layer = 0; layer < num_layers; ++layer) {
        auto& block = mFP4Blocks[layer];

        // QKV projection: (qkv_out, hidden)
        const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
        {
            auto& w = block.qkv_proj;
            w.M = qkv_out;
            w.K = hidden;

            // FP4 packed data: M * K / 2 bytes
            std::size_t packed_bytes = FP4BlockScaleConfig::packed_data_bytes(w.M, w.K);
            w.data = mAllocator->allocate(ETensorDType::BYTE, "qkv_fp4",
                                          EAllocationType::ON_DEVICE, {(long)packed_bytes});

            // FP8 block scales with F8_128x4 alignment
            auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(w.M, w.K);
            w.block_scales_rowwise = mAllocator->allocate(ETensorDType::FP8_E4M3, "qkv_scales",
                                                           EAllocationType::ON_DEVICE,
                                                           {(long)scale_rows, (long)scale_cols});

            // Global amax for this weight
            w.global_amax_rowwise = allocate_global_amax();
        }

        // Output projection: (hidden, num_q * head_size)
        {
            auto& w = block.out_proj;
            w.M = hidden;
            w.K = num_q_heads * head_size;

            std::size_t packed_bytes = FP4BlockScaleConfig::packed_data_bytes(w.M, w.K);
            w.data = mAllocator->allocate(ETensorDType::BYTE, "out_fp4",
                                          EAllocationType::ON_DEVICE, {(long)packed_bytes});

            auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(w.M, w.K);
            w.block_scales_rowwise = mAllocator->allocate(ETensorDType::FP8_E4M3, "out_scales",
                                                           EAllocationType::ON_DEVICE,
                                                           {(long)scale_rows, (long)scale_cols});
            w.global_amax_rowwise = allocate_global_amax();
        }

        // Gate+Up projection: (2 * intermediate, hidden)
        {
            auto& w = block.gate_up_proj;
            w.M = 2 * intermediate;
            w.K = hidden;

            std::size_t packed_bytes = FP4BlockScaleConfig::packed_data_bytes(w.M, w.K);
            w.data = mAllocator->allocate(ETensorDType::BYTE, "gate_up_fp4",
                                          EAllocationType::ON_DEVICE, {(long)packed_bytes});

            auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(w.M, w.K);
            w.block_scales_rowwise = mAllocator->allocate(ETensorDType::FP8_E4M3, "gate_up_scales",
                                                           EAllocationType::ON_DEVICE,
                                                           {(long)scale_rows, (long)scale_cols});
            w.global_amax_rowwise = allocate_global_amax();
        }

        // Down projection: (hidden, intermediate)
        {
            auto& w = block.down_proj;
            w.M = hidden;
            w.K = intermediate;

            std::size_t packed_bytes = FP4BlockScaleConfig::packed_data_bytes(w.M, w.K);
            w.data = mAllocator->allocate(ETensorDType::BYTE, "down_fp4",
                                          EAllocationType::ON_DEVICE, {(long)packed_bytes});

            auto [scale_rows, scale_cols] = FP4BlockScaleConfig::scale_dims(w.M, w.K);
            w.block_scales_rowwise = mAllocator->allocate(ETensorDType::FP8_E4M3, "down_scales",
                                                           EAllocationType::ON_DEVICE,
                                                           {(long)scale_rows, (long)scale_cols});
            w.global_amax_rowwise = allocate_global_amax();
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

    std::cerr << "[FP4-QLoRA] allocated " << num_layers << " FP4 quantized blocks\n";
}

void FP4WeightsManager::import_and_quantize(const std::string& file_name,
                                             NCCLCommunicator& comm,
                                             cudaStream_t stream) {
    std::cerr << "[FP4-QLoRA] importing and quantizing weights from " << file_name << "\n";

    SafeTensorsReader reader(file_name);

    // Allocate temporary load buffer for BF16 weights
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
        static_cast<std::size_t>(mConfig.vocab_size) * hidden
    });

    // Allocate temporary load buffer directly
    {
        void* ptr = nullptr;
        const std::size_t load_buf_bytes = max_weight_elems * sizeof(nv_bfloat16);
        CUDA_CHECK(cudaMalloc(&ptr, load_buf_bytes));
        mLoadBuffer.Data = static_cast<std::byte*>(ptr);
        mLoadBuffer.DType = ETensorDType::BF16;
        mLoadBuffer.Rank = 1;
        mLoadBuffer.Sizes[0] = static_cast<long>(max_weight_elems);
        std::fill(mLoadBuffer.Sizes.begin() + 1, mLoadBuffer.Sizes.end(), 1);
        mLoadBufferBytes = load_buf_bytes;
    }

    // Load embeddings (not quantized)
    load_embeddings(reader, stream);

    // Load and quantize each transformer block
    for (int layer = 0; layer < mConfig.num_layers; ++layer) {
        load_and_quantize_block(layer, reader, stream);
    }

    // Sync before freeing load buffer
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free load buffer
    if (mLoadBuffer.Data && mLoadBufferBytes > 0) {
        std::cerr << "[FP4-QLoRA] freeing load buffer ("
                  << (mLoadBufferBytes / (1024.0 * 1024.0)) << " MB)\n";
        CUDA_CHECK(cudaFree(mLoadBuffer.Data));
        mLoadBuffer = Tensor{};
        mLoadBufferBytes = 0;
    }

    std::cerr << "[FP4-QLoRA] import and FP4 quantization complete\n";
}

void FP4WeightsManager::load_embeddings(SafeTensorsReader& reader, cudaStream_t stream) {
    auto ctx = mAllocator->with_context("FP4_Embeddings");

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

    // LM head
    if (mConfig.tied_embeddings) {
        mEmbeddings.lm_head = mEmbeddings.embedding;
        mEmbeddings.tied_weights = true;
    } else {
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

        if (!found_lm_head) {
            std::cerr << "[FP4-QLoRA WARN] tied_embeddings=false but lm_head.weight not found, using tied\n";
            mEmbeddings.lm_head = mEmbeddings.embedding;
            mEmbeddings.tied_weights = true;
        }
    }
}

void FP4WeightsManager::load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                                 cudaStream_t stream) {
    auto& block = mFP4Blocks[layer_idx];
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    const std::string prefix = fmt::format("model.layers.{}", layer_idx);

    // Helper to load and quantize a weight to FP4
    auto load_and_quantize = [&](const std::string& name, FP4BlockQuantizedWeight& dest, int M, int K) {
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;

        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mLoadBuffer, /*allow_cast=*/true);
            quantize_fp4_and_store(dest, mLoadBuffer, M, K, stream);
        } else {
            std::cerr << "[FP4-QLoRA WARN] layer " << layer_idx << " weight not found: " << name << "\n";
        }
    };

    // Load Q, K, V projections
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    const std::string qkv_name = prefix + ".self_attn.qkv_proj.weight";
    if (find_entry_opt(reader, qkv_name)) {
        load_and_quantize(qkv_name, block.qkv_proj, qkv_out, hidden);
    } else {
        // Separate Q, K, V
        const int q_out = num_q_heads * head_size;
        const int kv_out = num_kv_heads * head_size;

        mLoadBuffer.Sizes[0] = qkv_out;
        mLoadBuffer.Sizes[1] = hidden;
        mLoadBuffer.Rank = 2;

        Tensor q_view = mLoadBuffer;
        q_view.Sizes[0] = q_out;
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.q_proj.weight")) {
            entry->read_tensor(q_view, true);
        }

        Tensor k_view = mLoadBuffer;
        k_view.Data = mLoadBuffer.Data + q_out * hidden * sizeof(nv_bfloat16);
        k_view.Sizes[0] = kv_out;
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.k_proj.weight")) {
            entry->read_tensor(k_view, true);
        }

        Tensor v_view = mLoadBuffer;
        v_view.Data = mLoadBuffer.Data + (q_out + kv_out) * hidden * sizeof(nv_bfloat16);
        v_view.Sizes[0] = kv_out;
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.v_proj.weight")) {
            entry->read_tensor(v_view, true);
        }

        mLoadBuffer.Sizes[0] = qkv_out;
        quantize_fp4_and_store(block.qkv_proj, mLoadBuffer, qkv_out, hidden, stream);
    }

    // Output projection
    load_and_quantize(prefix + ".self_attn.o_proj.weight",
                      block.out_proj, hidden, num_q_heads * head_size);

    // MLP projections
    const std::string gate_up_name = prefix + ".mlp.gate_up_proj.weight";
    if (find_entry_opt(reader, gate_up_name)) {
        load_and_quantize(gate_up_name, block.gate_up_proj, 2 * intermediate, hidden);
    } else {
        mLoadBuffer.Sizes[0] = 2 * intermediate;
        mLoadBuffer.Sizes[1] = hidden;
        mLoadBuffer.Rank = 2;

        Tensor up_view = mLoadBuffer;
        up_view.Sizes[0] = intermediate;
        if (const auto* entry = find_entry_opt(reader, prefix + ".mlp.up_proj.weight")) {
            entry->read_tensor(up_view, true);
        }

        Tensor gate_view = mLoadBuffer;
        gate_view.Data = mLoadBuffer.Data + intermediate * hidden * sizeof(nv_bfloat16);
        gate_view.Sizes[0] = intermediate;
        if (const auto* entry = find_entry_opt(reader, prefix + ".mlp.gate_proj.weight")) {
            entry->read_tensor(gate_view, true);
        }

        mLoadBuffer.Sizes[0] = 2 * intermediate;
        quantize_fp4_and_store(block.gate_up_proj, mLoadBuffer, 2 * intermediate, hidden, stream);
    }

    // Down projection
    load_and_quantize(prefix + ".mlp.down_proj.weight",
                      block.down_proj, hidden, intermediate);

    // Layer norms (not quantized)
    if (const auto* entry = find_entry_opt(reader, prefix + ".input_layernorm.weight")) {
        entry->read_tensor(block.ln1_weight, true);
    }
    if (const auto* entry = find_entry_opt(reader, prefix + ".post_attention_layernorm.weight")) {
        entry->read_tensor(block.ln2_weight, true);
    }

    // QK-norm weights
    if (mConfig.use_qk_norm && block.q_norm_weight.has_value() && block.k_norm_weight.has_value()) {
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.q_norm.weight")) {
            entry->read_tensor(block.q_norm_weight.value(), true);
        }
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.k_norm.weight")) {
            entry->read_tensor(block.k_norm_weight.value(), true);
        }
    }
}

void FP4WeightsManager::quantize_fp4_and_store(FP4BlockQuantizedWeight& dest, const Tensor& src,
                                                int M, int K, cudaStream_t stream) {
    const auto& qlora_cfg = mConfig.qlora_config;

    if (qlora_cfg.enable_four_over_six) {
        // Four Over Six (4/6) adaptive block scaling quantization.
        // Uses tensor scale 1536 instead of 2688.
        int metric = static_cast<int>(qlora_cfg.four_over_six_metric);
        quantize_fp4_block_4o6_auto_scale(
            dest.data.get<uint8_t>(),
            dest.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            dest.global_amax_rowwise,
            src.get<nv_bfloat16>(),
            M, K,
            metric,
            *mDeviceProps,
            stream);
    } else {
        // Standard NVFP4 quantization with tensor scale 2688.
        quantize_fp4_block_auto_scale(
            dest.data.get<uint8_t>(),
            dest.block_scales_rowwise.get<__nv_fp8_e4m3>(),
            dest.global_amax_rowwise,
            src.get<nv_bfloat16>(),
            M, K,
            *mDeviceProps,
            stream);
    }

    // Cache a host copy of amax and the corresponding global decode scale for dequantization.
    // This runs at import time (outside CUDA graph capture) and avoids syncs during training.
    float amax_h = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&amax_h, dest.global_amax_rowwise, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dest.global_amax_rowwise_host = amax_h;
    if (amax_h > 0.0f && std::isfinite(amax_h)) {
        if (qlora_cfg.enable_four_over_six) {
            // 4/6 tensor scale = 1536 (= 384 * 4)
            constexpr float FP4_4O6_TENSOR_SCALE = 384.0f * 4.0f;
            dest.global_decode_scale_rowwise_host = amax_h / FP4_4O6_TENSOR_SCALE;
        } else {
            // Standard tensor scale = 2688 (= 6 * 448)
            dest.global_decode_scale_rowwise_host =
                amax_h / (FP4_E2M1_MAX * fp8_max_v<__nv_fp8_e4m3>);
        }
    } else {
        dest.global_decode_scale_rowwise_host = 1.0f;
    }
}

std::size_t FP4WeightsManager::quantized_weights_bytes() const {
    std::size_t total = 0;
    for (const auto& block : mFP4Blocks) {
        total += block.bytes();
    }
    total += mEmbeddings.bytes();
    return total;
}

float FP4WeightsManager::memory_savings_ratio() const {
    // Calculate what BF16 storage would have been
    std::size_t bf16_bytes = 0;
    for (const auto& block : mFP4Blocks) {
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
