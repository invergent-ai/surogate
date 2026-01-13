// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "bnb_weights.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <mutex>

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

BnBWeightsManager::BnBWeightsManager(const Config& config, TensorAllocator& allocator,
                                     const cudaDeviceProp& device_props)
    : mConfig(config)
    , mAllocator(&allocator)
    , mDeviceProps(&device_props)
{
    if (!config.qlora_config.is_bnb()) {
        throw std::runtime_error("BnBWeightsManager: BitsAndBytes QLoRA must be enabled");
    }

    // Initialize scale config from qlora_config
    mScaleConfig.block_size = mConfig.qlora_config.block_size();
    mScaleConfig.double_quant = mConfig.qlora_config.bnb_double_quant;
    mScaleConfig.double_quant_group_size = mConfig.qlora_config.bnb_double_quant_group_size;

    // Pre-size the vector but don't allocate GPU memory yet.
    // Each layer's storage will be allocated lazily in load_and_quantize_block()
    // to reduce peak memory during initialization.
    if (config.qlora_config.is_moe()) {
        mMoEBlocks.resize(mConfig.num_layers);
    } else {
        mQuantizedBlocks.resize(mConfig.num_layers);
    }
}

void BnBWeightsManager::allocate_bnb_weight(BnBBlockQuantizedWeight& weight, int M, int K,
                                            const std::string& name_prefix) {
    // Default to ON_DEVICE allocation
    allocate_bnb_weight(weight, M, K, name_prefix, EAllocationType::ON_DEVICE);
}

void BnBWeightsManager::allocate_bnb_weight(BnBBlockQuantizedWeight& weight, int M, int K,
                                            const std::string& name_prefix, EAllocationType alloc_type) {
    const int block_size = mScaleConfig.block_size;
    const bool double_quant = mScaleConfig.double_quant;
    const int dq_group_size = mScaleConfig.double_quant_group_size;

    weight.M = M;
    weight.K = K;
    weight.block_size = block_size;
    weight.double_quant = double_quant;
    weight.double_quant_group_size = dq_group_size;

    // Calculate packed size (2 values per byte for 4-bit)
    const long num_elements = static_cast<long>(M) * K;
    const long packed_bytes = (num_elements + 1) / 2;  // Round up for odd counts

    // Allocate packed NF4 data (BYTE storage for packed 4-bit values)
    std::string data_name = name_prefix + "_nf4";
    weight.data = mAllocator->allocate(ETensorDType::BYTE, data_name.c_str(),
                                       alloc_type, {packed_bytes});

    // Calculate number of absmax blocks
    const long num_blocks = (num_elements + block_size - 1) / block_size;

    if (double_quant) {
        // Double quantization: absmax stored as BYTE (uint8) with per-group scale/offset
        std::string absmax_name = name_prefix + "_absmax_q";
        weight.absmax = mAllocator->allocate(ETensorDType::BYTE, absmax_name.c_str(),
                                             alloc_type, {num_blocks});

        // Number of groups for double quantization
        const long num_groups = (num_blocks + dq_group_size - 1) / dq_group_size;
        std::string scale_name = name_prefix + "_absmax_scale";
        weight.absmax_scale = mAllocator->allocate(ETensorDType::FP32, scale_name.c_str(),
                                                   alloc_type, {num_groups});
        std::string offset_name = name_prefix + "_absmax_offset";
        weight.absmax_offset = mAllocator->allocate(ETensorDType::FP32, offset_name.c_str(),
                                                    alloc_type, {num_groups});
    } else {
        // No double quantization: absmax stored as FP32
        std::string absmax_name = name_prefix + "_absmax";
        weight.absmax = mAllocator->allocate(ETensorDType::FP32, absmax_name.c_str(),
                                             alloc_type, {num_blocks});
    }
}

void BnBWeightsManager::allocate_single_block(int layer_idx) {
    auto ctx = mAllocator->with_context("BnB_Weights");

    // Show progress bar
    show_progress_bar(layer_idx, mConfig.num_layers, "[BnB] Quantizing layers");

    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;

    auto& block = mQuantizedBlocks[layer_idx];

    // QKV projection: (hidden, (num_q + 2*num_kv) * head_size)
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    allocate_bnb_weight(block.qkv_proj, qkv_out, hidden, "qkv");

    // Output projection: (hidden, num_q * head_size)
    allocate_bnb_weight(block.out_proj, hidden, num_q_heads * head_size, "out");

    // Gate+Up projection: (2 * intermediate, hidden)
    allocate_bnb_weight(block.gate_up_proj, 2 * intermediate, hidden, "gate_up");

    // Down projection: (hidden, intermediate)
    allocate_bnb_weight(block.down_proj, hidden, intermediate, "down");

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

void BnBWeightsManager::import_and_quantize(const std::string& file_name,
                                            NCCLCommunicator& comm,
                                            cudaStream_t stream) {
    std::cerr << "[BnB] importing and quantizing weights from " << file_name << "\n";
    std::cerr << "[BnB] block_size=" << mScaleConfig.block_size
              << ", double_quant=" << (mScaleConfig.double_quant ? "true" : "false") << "\n";

    if (is_moe()) {
        const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                              mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;
        std::cerr << "[BnB] MoE model detected: " << mConfig.qlora_config.num_experts << " experts, "
                  << "top_k=" << mConfig.qlora_config.num_experts_per_tok
                  << ", moe_intermediate_size=" << mConfig.qlora_config.moe_intermediate_size
                  << ", using moe_inter=" << moe_inter
                  << ", hidden=" << mConfig.hidden_size << "\n";
    }

    SafeTensorsReader reader(file_name);

    // Allocate temporary load buffer for BF16 weights
    // Size it for the largest weight we need to load
    const int hidden = mConfig.hidden_size;
    const int intermediate = mConfig.intermediate_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;

    // For MoE models, use moe_intermediate_size for expert projections
    const int moe_inter = is_moe() ?
        (mConfig.qlora_config.moe_intermediate_size > 0 ?
         mConfig.qlora_config.moe_intermediate_size : intermediate) : intermediate;

    std::size_t max_weight_elems = std::max({
        static_cast<std::size_t>(qkv_out) * hidden,
        static_cast<std::size_t>(hidden) * num_q_heads * head_size,
        static_cast<std::size_t>(2 * intermediate) * hidden,  // Dense MLP (if any)
        static_cast<std::size_t>(2 * moe_inter) * hidden,     // Expert gate+up
        static_cast<std::size_t>(hidden) * moe_inter,         // Expert down
        static_cast<std::size_t>(mConfig.vocab_size) * hidden // Embedding
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
        mLoadBufferBytes = load_buf_bytes;
    }

    // Allocate temporary absmax buffer for double quantization
    // Sized for the largest weight matrix
    if (mScaleConfig.double_quant) {
        const long max_num_blocks = (static_cast<long>(max_weight_elems) + mScaleConfig.block_size - 1)
                                    / mScaleConfig.block_size;
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, max_num_blocks * sizeof(float)));
        mAbsmaxBuffer.Data = static_cast<std::byte*>(ptr);
        mAbsmaxBuffer.DType = ETensorDType::FP32;
        mAbsmaxBuffer.Rank = 1;
        mAbsmaxBuffer.Sizes[0] = max_num_blocks;
    }

    // Load embeddings (not quantized)
    load_embeddings(reader, stream);

    // Load and quantize each transformer block using double-buffering for I/O overlap
    // Allocate a second load buffer for ping-pong to allow CPU file I/O while GPU quantizes
    Tensor load_buffer_2;
    Tensor absmax_buffer_2;
    {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, mLoadBufferBytes));
        load_buffer_2.Data = static_cast<std::byte*>(ptr);
        load_buffer_2.DType = ETensorDType::BF16;
        load_buffer_2.Rank = 1;
        load_buffer_2.Sizes[0] = mLoadBuffer.Sizes[0];
        std::fill(load_buffer_2.Sizes.begin() + 1, load_buffer_2.Sizes.end(), 1);

        if (mScaleConfig.double_quant) {
            void* absmax_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&absmax_ptr, mAbsmaxBuffer.Sizes[0] * sizeof(float)));
            absmax_buffer_2.Data = static_cast<std::byte*>(absmax_ptr);
            absmax_buffer_2.DType = ETensorDType::FP32;
            absmax_buffer_2.Rank = 1;
            absmax_buffer_2.Sizes[0] = mAbsmaxBuffer.Sizes[0];
        }
    }

    // Create two non-blocking streams for double-buffering
    cudaStream_t stream_0, stream_1;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_0, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking));

    if (is_moe()) {
        // MoE path: load experts + router with double-buffering
        for (int layer = 0; layer < mConfig.num_layers; ++layer) {
            const bool use_buffer_0 = (layer % 2 == 0);
            cudaStream_t curr_stream = use_buffer_0 ? stream_0 : stream_1;

            // Swap buffers
            Tensor saved_load = mLoadBuffer;
            Tensor saved_absmax = mAbsmaxBuffer;

            if (!use_buffer_0) {
                mLoadBuffer = load_buffer_2;
                if (mScaleConfig.double_quant) {
                    mAbsmaxBuffer = absmax_buffer_2;
                }
            }

            // Load and quantize on current stream (GPU work is async)
            load_and_quantize_moe_block(layer, reader, curr_stream);

            // Restore buffers
            mLoadBuffer = saved_load;
            mAbsmaxBuffer = saved_absmax;
        }
    } else {
        // Dense path with double-buffering
        for (int layer = 0; layer < mConfig.num_layers; ++layer) {
            const bool use_buffer_0 = (layer % 2 == 0);
            cudaStream_t curr_stream = use_buffer_0 ? stream_0 : stream_1;

            // Swap buffers
            Tensor saved_load = mLoadBuffer;
            Tensor saved_absmax = mAbsmaxBuffer;

            if (!use_buffer_0) {
                mLoadBuffer = load_buffer_2;
                if (mScaleConfig.double_quant) {
                    mAbsmaxBuffer = absmax_buffer_2;
                }
            }

            // Load and quantize on current stream (GPU work is async)
            load_and_quantize_block(layer, reader, curr_stream);

            // Restore buffers
            mLoadBuffer = saved_load;
            mAbsmaxBuffer = saved_absmax;
        }
    }

    // Sync both streams
    CUDA_CHECK(cudaStreamSynchronize(stream_0));
    CUDA_CHECK(cudaStreamSynchronize(stream_1));
    CUDA_CHECK(cudaStreamDestroy(stream_0));
    CUDA_CHECK(cudaStreamDestroy(stream_1));

    // Free second buffer
    if (load_buffer_2.Data) {
        CUDA_CHECK(cudaFree(load_buffer_2.Data));
    }
    if (absmax_buffer_2.Data) {
        CUDA_CHECK(cudaFree(absmax_buffer_2.Data));
    }

    // Sync original stream
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Free load buffer (allocated directly, not via allocator)
    if (mLoadBuffer.Data && mLoadBufferBytes > 0) {
        CUDA_CHECK(cudaFree(mLoadBuffer.Data));
        mLoadBuffer = Tensor{};
        mLoadBufferBytes = 0;
    }

    // Free absmax buffer
    if (mAbsmaxBuffer.Data) {
        CUDA_CHECK(cudaFree(mAbsmaxBuffer.Data));
        mAbsmaxBuffer = Tensor{};
    }

    std::cerr << "[BnB] import and quantization complete (" << mConfig.num_layers << " layers)\n";
    std::cerr << "[BnB] memory usage: " << (quantized_weights_bytes() / (1024*1024)) << " MB\n";
    std::cerr << "[BnB] memory savings: " << (memory_savings_ratio() * 100.0f) << "%\n";
}

void BnBWeightsManager::load_embeddings(SafeTensorsReader& reader, cudaStream_t stream) {
    auto ctx = mAllocator->with_context("BnB_Embeddings");

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

    // Final layer norm
    mEmbeddings.final_norm = mAllocator->allocate(ETensorDType::BF16, "final_norm",
                                                  EAllocationType::ON_DEVICE, {(long)hidden});

    const std::vector<std::string> norm_names = {
        "model.norm.weight",
        "transformer.ln_f.weight",
        "encoder.final_layernorm.weight"
    };

    for (const auto& name : norm_names) {
        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mEmbeddings.final_norm, /*allow_cast=*/true);
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
            std::cerr << "[BnB WARN] tied_embeddings=false but lm_head.weight not found, using tied\n";
            mEmbeddings.lm_head = mEmbeddings.embedding;
            mEmbeddings.tied_weights = true;
        }
    }
}

void BnBWeightsManager::load_and_quantize_block(int layer_idx, SafeTensorsReader& reader,
                                                cudaStream_t stream) {
    // Lazily allocate this layer's NF4 storage just before we need it.
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
    auto load_and_quantize = [&](const std::string& name, BnBBlockQuantizedWeight& dest, int M, int K) {
        // Set up load buffer shape
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;

        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mLoadBuffer, /*allow_cast=*/true);
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
        } else {
            std::cerr << "[BnB WARN] layer " << layer_idx << " weight not found: " << name << "\n";
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

void BnBWeightsManager::quantize_and_store(BnBBlockQuantizedWeight& dest, const Tensor& src,
                                           int M, int K, cudaStream_t stream) {
    const long num_elements = static_cast<long>(M) * K;
    const long num_blocks = (num_elements + dest.block_size - 1) / dest.block_size;

    if (dest.double_quant) {
        // Two-step quantization:
        // 1. Quantize weights to NF4, storing FP32 absmax in temporary buffer
        // 2. Double-quantize absmax values to INT8

        // Step 1: Quantize to NF4 with FP32 absmax in temp buffer
        quantize_bnb_nf4(
            dest.data.get<unsigned char>(),
            mAbsmaxBuffer.get<float>(),
            src.get<nv_bfloat16>(),
            M, K,
            dest.block_size,
            *mDeviceProps,
            stream);

        // Step 2: Apply double quantization to absmax values
        apply_double_quantization(dest, stream);        
    } else {
        // Single-step quantization: NF4 with FP32 absmax directly
        quantize_bnb_nf4(
            dest.data.get<unsigned char>(),
            dest.absmax.get<float>(),
            src.get<nv_bfloat16>(),
            M, K,
            dest.block_size,
            *mDeviceProps,
            stream);
    }
}

void BnBWeightsManager::apply_double_quantization(BnBBlockQuantizedWeight& weight, cudaStream_t stream) {
    const long num_elements = static_cast<long>(weight.M) * weight.K;
    const long num_absmax = (num_elements + weight.block_size - 1) / weight.block_size;

    // Quantize the FP32 absmax values to INT8 with per-group scale/offset
    quantize_absmax_double(
        weight.absmax.get<unsigned char>(),
        weight.absmax_scale.get<float>(),
        weight.absmax_offset.get<float>(),
        mAbsmaxBuffer.get<float>(),
        static_cast<int>(num_absmax),
        weight.double_quant_group_size,
        *mDeviceProps,
        stream);
}

std::size_t BnBWeightsManager::quantized_weights_bytes() const {
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

float BnBWeightsManager::memory_savings_ratio() const {
    // Calculate what BF16 storage would have been
    std::size_t bf16_bytes = 0;

    if (is_moe()) {
        for (const auto& block : mMoEBlocks) {
            bf16_bytes += static_cast<std::size_t>(block.qkv_proj.M) * block.qkv_proj.K * 2;
            bf16_bytes += static_cast<std::size_t>(block.out_proj.M) * block.out_proj.K * 2;
            bf16_bytes += block.ln1_weight.bytes();
            bf16_bytes += block.ln2_weight.bytes();
            bf16_bytes += block.router_gate.bytes();
            for (const auto& expert : block.experts) {
                bf16_bytes += static_cast<std::size_t>(expert.gate_up_proj.M) * expert.gate_up_proj.K * 2;
                bf16_bytes += static_cast<std::size_t>(expert.down_proj.M) * expert.down_proj.K * 2;
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

void BnBWeightsManager::allocate_moe_block(int layer_idx) {
    auto ctx = mAllocator->with_context("BnB_MoE_Weights");

    // Show progress bar
    show_progress_bar(layer_idx, mConfig.num_layers, "[BnB] Quantizing layers");

    const int hidden = mConfig.hidden_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int n_experts = mConfig.qlora_config.num_experts;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;

    // Determine allocation type for expert weights
    // When offload_experts is enabled, store experts in pinned CPU memory
    const EAllocationType exp_alloc = expert_alloc_type();

    auto& block = mMoEBlocks[layer_idx];

    // QKV projection (same as dense) - always on GPU
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    allocate_bnb_weight(block.qkv_proj, qkv_out, hidden, "qkv");

    // Output projection (same as dense) - always on GPU
    allocate_bnb_weight(block.out_proj, hidden, num_q_heads * head_size, "out");

    // Layer norm weights (BF16, not quantized) - always on GPU
    block.ln1_weight = mAllocator->allocate(ETensorDType::BF16, "ln1",
                                            EAllocationType::ON_DEVICE, {(long)hidden});
    block.ln2_weight = mAllocator->allocate(ETensorDType::BF16, "ln2",
                                            EAllocationType::ON_DEVICE, {(long)hidden});

    // QK-norm weights (for Qwen3) - always on GPU
    if (mConfig.use_qk_norm) {
        block.q_norm_weight = mAllocator->allocate(ETensorDType::BF16, "q_norm",
                                                   EAllocationType::ON_DEVICE, {(long)head_size});
        block.k_norm_weight = mAllocator->allocate(ETensorDType::BF16, "k_norm",
                                                   EAllocationType::ON_DEVICE, {(long)head_size});
    }

    // Router gate (BF16, not quantized - small tensor) - always on GPU
    // NOTE: Shape matches HuggingFace (num_experts, hidden). `model_forward.hpp` uses matmul(TN)
    //       and expects router_gate as (num_experts, hidden) like other linear weights.
    block.router_gate = mAllocator->allocate(ETensorDType::BF16, "router_gate",
                                             EAllocationType::ON_DEVICE,
                                             {(long)n_experts, (long)hidden});

    // Allocate expert weights - on GPU or CPU depending on offload setting
    block.experts.resize(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        auto& expert = block.experts[e];
        std::string prefix = fmt::format("expert_{}", e);
        allocate_bnb_weight(expert.gate_up_proj, 2 * moe_inter, hidden, prefix + "_gate_up", exp_alloc);
        allocate_bnb_weight(expert.down_proj, hidden, moe_inter, prefix + "_down", exp_alloc);
    }
}

void BnBWeightsManager::load_and_quantize_moe_block(int layer_idx, SafeTensorsReader& reader,
                                                     cudaStream_t stream) {
    // Lazily allocate this layer's NF4 storage
    allocate_moe_block(layer_idx);

    auto& block = mMoEBlocks[layer_idx];
    const int hidden = mConfig.hidden_size;
    const int num_q_heads = mConfig.num_query_heads;
    const int num_kv_heads = mConfig.num_kv_heads;
    const int head_size = mConfig.head_size;
    const int n_experts = mConfig.qlora_config.num_experts;
    const int moe_inter = mConfig.qlora_config.moe_intermediate_size > 0 ?
                          mConfig.qlora_config.moe_intermediate_size : mConfig.intermediate_size;

    const std::string prefix = fmt::format("model.layers.{}", layer_idx);

    // Helper to load and quantize a weight
    auto load_and_quantize = [&](const std::string& name, BnBBlockQuantizedWeight& dest, int M, int K) {
        mLoadBuffer.Sizes[0] = M;
        mLoadBuffer.Sizes[1] = K;
        mLoadBuffer.Rank = 2;

        if (const auto* entry = find_entry_opt(reader, name)) {
            entry->read_tensor(mLoadBuffer, /*allow_cast=*/true);
            quantize_and_store(dest, mLoadBuffer, M, K, stream);
        } else {
            std::cerr << "[BnB WARN] layer " << layer_idx << " weight not found: " << name << "\n";
        }
    };

    // Load Q, K, V projections (handle both fused and separate)
    const int qkv_out = (num_q_heads + 2 * num_kv_heads) * head_size;
    const std::string qkv_name = prefix + ".self_attn.qkv_proj.weight";
    if (find_entry_opt(reader, qkv_name)) {
        load_and_quantize(qkv_name, block.qkv_proj, qkv_out, hidden);
    } else {
        // Separate Q, K, V - load into parts of mLoadBuffer and then quantize
        const int q_out = num_q_heads * head_size;
        const int kv_out = num_kv_heads * head_size;

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

        mLoadBuffer.Sizes[0] = qkv_out;
        quantize_and_store(block.qkv_proj, mLoadBuffer, qkv_out, hidden, stream);
    }

    // Output projection
    load_and_quantize(prefix + ".self_attn.o_proj.weight",
                      block.out_proj, hidden, num_q_heads * head_size);

    // Layer norms
    if (const auto* entry = find_entry_opt(reader, prefix + ".input_layernorm.weight")) {
        entry->read_tensor(block.ln1_weight, true);
    }
    if (const auto* entry = find_entry_opt(reader, prefix + ".post_attention_layernorm.weight")) {
        entry->read_tensor(block.ln2_weight, true);
    }

    // QK-norm weights (for Qwen3)
    if (mConfig.use_qk_norm && block.q_norm_weight.has_value() && block.k_norm_weight.has_value()) {
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.q_norm.weight")) {
            entry->read_tensor(block.q_norm_weight.value(), true);
        }
        if (const auto* entry = find_entry_opt(reader, prefix + ".self_attn.k_norm.weight")) {
            entry->read_tensor(block.k_norm_weight.value(), true);
        }
    }

    // Router gate (BF16, not quantized)
    // Model stores as (num_experts, hidden) and our matmul(TN) expects the same layout.
    if (const auto* entry = find_entry_opt(reader, prefix + ".mlp.gate.weight")) {
        entry->read_tensor(block.router_gate, /*allow_cast=*/true);
    } else {
        std::cerr << "[BnB WARN] layer " << layer_idx << " router gate not found: "
                  << prefix << ".mlp.gate.weight - this will cause NaN!\n";
    }

    // Load and quantize each expert
    for (int e = 0; e < n_experts; ++e) {
        auto& expert = block.experts[e];
        const std::string exp_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);

        // Expert gate+up projection (handle both fused and separate)
        const std::string gate_up_name = exp_prefix + ".gate_up_proj.weight";
        if (find_entry_opt(reader, gate_up_name)) {
            load_and_quantize(gate_up_name, expert.gate_up_proj, 2 * moe_inter, hidden);
        } else {
            // Separate gate and up - fuse them
            mLoadBuffer.Sizes[0] = 2 * moe_inter;
            mLoadBuffer.Sizes[1] = hidden;
            mLoadBuffer.Rank = 2;

            // Load up into first part (Qwen3 uses [up; gate] layout)
            Tensor up_view = mLoadBuffer;
            up_view.Sizes[0] = moe_inter;
            if (const auto* entry = find_entry_opt(reader, exp_prefix + ".up_proj.weight")) {
                entry->read_tensor(up_view, true);
            } else if (e == 0 && layer_idx == 0) {
                std::cerr << "[BnB WARN] expert 0 up_proj not found: " << exp_prefix << ".up_proj.weight\n";
            }

            // Load gate into second part
            Tensor gate_view = mLoadBuffer;
            gate_view.Data = mLoadBuffer.Data + moe_inter * hidden * sizeof(nv_bfloat16);
            gate_view.Sizes[0] = moe_inter;
            if (const auto* entry = find_entry_opt(reader, exp_prefix + ".gate_proj.weight")) {
                entry->read_tensor(gate_view, true);
            } else if (e == 0 && layer_idx == 0) {
                std::cerr << "[BnB WARN] expert 0 gate_proj not found: " << exp_prefix << ".gate_proj.weight\n";
            }

            mLoadBuffer.Sizes[0] = 2 * moe_inter;
            quantize_and_store(expert.gate_up_proj, mLoadBuffer, 2 * moe_inter, hidden, stream);
        }

        // Expert down projection
        load_and_quantize(exp_prefix + ".down_proj.weight",
                          expert.down_proj, hidden, moe_inter);
    }
}

} // namespace modules
