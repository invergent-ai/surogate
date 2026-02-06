// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DslQLoRAPipeline: Unified weight loading + quantization pipeline.

#include "runtime/qlora/dsl_qlora_pipeline.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <cuda_bf16.h>
#include <fmt/format.h>

#include "config/pretrained_config.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "utilities/allocator.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace qlora {

namespace {

/// Find the maximum weight size (in elements) across all quantizable specs.
/// Used to allocate reusable BF16 load buffers.
size_t max_weight_elements(const std::vector<WeightLoadSpec>& specs) {
    size_t max_elem = 0;
    for (const auto& spec : specs) {
        size_t elem = static_cast<size_t>(spec.M) * std::max(spec.K, 1);
        if (elem > max_elem) {
            max_elem = elem;
        }
    }
    return max_elem;
}

/// Count quantizable specs (for progress reporting).
int count_quantizable(const std::vector<WeightLoadSpec>& specs) {
    int count = 0;
    for (const auto& s : specs) {
        if (s.quantize) count++;
    }
    return count;
}

/// Show a simple progress bar on stderr.
void show_progress(int current, int total, const char* label) {
    if (total <= 0) return;
    const int bar_width = 40;
    const float progress = static_cast<float>(current + 1) / static_cast<float>(total);
    const int filled = static_cast<int>(progress * bar_width);

    fprintf(stderr, "\r[%s] [", label);
    for (int i = 0; i < bar_width; ++i) {
        fprintf(stderr, "%c", i < filled ? '#' : '.');
    }
    fprintf(stderr, "] %d/%d", current + 1, total);
    if (current + 1 == total) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);
}

/// Create a shaped view of a flat load buffer.
Tensor make_buffer_view(const Tensor& buffer, int M, int K) {
    if (K > 0) {
        return Tensor::from_pointer(
            buffer.Data, buffer.Device,
            ETensorDType::BF16,
            std::vector<long>{static_cast<long>(M), static_cast<long>(K)});
    }
    return Tensor::from_pointer(
        buffer.Data, buffer.Device,
        ETensorDType::BF16,
        std::vector<long>{static_cast<long>(M)});
}

/// RAII guard for a CUDA stream.
struct ScopedStream {
    cudaStream_t stream = nullptr;
    ScopedStream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~ScopedStream() {
        if (stream) cudaStreamDestroy(stream);
    }
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;
};

/// RAII guard for a CUDA event.
struct ScopedEvent {
    cudaEvent_t event = nullptr;
    ScopedEvent() {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~ScopedEvent() {
        if (event) cudaEventDestroy(event);
    }
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
};

}  // anonymous namespace

// =============================================================================
// Main pipeline function
// =============================================================================

std::unique_ptr<GenericWeightManager> import_and_quantize_weights(
    const std::string& file_name,
    const DslQLoRAPipelineConfig& config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator,
    cudaStream_t stream) {

    auto t_start = std::chrono::steady_clock::now();

    // ---- Step 1: Create quantizer and weight manager ----

    auto quantizer = create_quantizer(config.quantizer_config);
    if (!quantizer && config.quantizer_config.format != QuantFormat::NONE) {
        throw std::runtime_error("Failed to create quantizer for the specified format");
    }

    auto weight_mgr = std::make_unique<GenericWeightManager>(
        config.weight_manager_config,
        std::move(quantizer),
        allocator);

    // ---- Step 2: Register all quantizable weights ----

    for (const auto& spec : config.weight_specs) {
        if (spec.quantize && weight_mgr->quantizer()) {
            weight_mgr->register_weight(spec.name, spec.M, spec.K, spec.offload_group);
        }
        // Full-precision weights will be registered during load (we need the tensor first)
    }

    // ---- Step 3: Open SafeTensors and create weight loader ----

    SafeTensorsReader reader(file_name);

    dsl::ShardConfig shard_config;
    shard_config.shard_idx = config.shard_idx;
    shard_config.num_shards = config.num_shards;

    dsl::MoEWeightConfig moe_config;
    moe_config.num_experts = config.num_experts;
    moe_config.moe_intermediate_size = config.moe_intermediate_size;

    dsl::DslWeightLoader loader(
        reader,
        config.mapping,
        pt_config,
        *allocator,
        shard_config,
        moe_config);

    // ---- Step 4: Allocate double-buffered load buffers ----
    //
    // Double buffering allows overlapping:
    //   - CPU disk I/O for weight N+1  with  GPU quantization of weight N
    //
    // Buffer A and B alternate: while GPU quantizes from buffer A,
    // CPU reads the next weight into buffer B (and vice versa).

    const size_t max_elem = max_weight_elements(config.weight_specs);
    const int num_quantizable = count_quantizable(config.weight_specs);
    const bool use_double_buffer = (num_quantizable >= 2) && (max_elem > 0);

    Tensor load_buffers[2];
    if (max_elem > 0) {
        load_buffers[0] = allocator->allocate(
            ETensorDType::BF16,
            "qlora_pipeline.load_buf_A",
            EAllocationType::ON_DEVICE,
            {static_cast<long>(max_elem)});
        if (use_double_buffer) {
            load_buffers[1] = allocator->allocate(
                ETensorDType::BF16,
                "qlora_pipeline.load_buf_B",
                EAllocationType::ON_DEVICE,
                {static_cast<long>(max_elem)});
        }
    }

    // Create quantization stream and per-buffer completion events
    ScopedStream quant_stream_guard;
    ScopedEvent buf_ready[2];
    cudaStream_t quant_stream = use_double_buffer ? quant_stream_guard.stream : stream;

    // ---- Step 5: Load and quantize each weight (pipelined) ----
    //
    // For quantizable weights with double buffering:
    //   1. Wait for previous quant on this buffer slot to complete
    //   2. Load BF16 from SafeTensors into buffer[slot]
    //   3. Issue quantization on quant_stream (async GPU kernel)
    //   4. Record completion event for this buffer slot
    //   5. CPU immediately continues to next weight (disk I/O overlaps with quant)
    //
    // For full-precision weights: load directly (no buffering needed)

    const int total = static_cast<int>(config.weight_specs.size());
    int loaded = 0;
    int quant_slot = 0;  // Alternates 0/1 for double buffering

    for (int i = 0; i < total; ++i) {
        const auto& spec = config.weight_specs[i];
        show_progress(i, total, "QLoRA Import");

        if (spec.quantize && weight_mgr->quantizer()) {
            // Quantizable weight: double-buffered load → quantize pipeline
            const int slot = use_double_buffer ? (quant_slot % 2) : 0;

            // Wait for previous quantization that used this buffer slot
            if (use_double_buffer && quant_slot >= 2) {
                CUDA_CHECK(cudaEventSynchronize(buf_ready[slot].event));
            }

            // Create a view of the load buffer with the correct shape
            Tensor target = make_buffer_view(load_buffers[slot], spec.M, spec.K);

            // Load BF16 weight from SafeTensors (CPU disk I/O + GPU copy on main stream)
            bool success = loader.load_param(spec.name, target, true, spec.sharded, nullptr, stream);
            if (success) {
                // Ensure GPU copy is complete before quantizing
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Issue quantization on quant stream (async GPU kernel)
                weight_mgr->quantize_and_store(spec.name, target, quant_stream);

                // Record event: buffer slot is free when this quant completes
                if (use_double_buffer) {
                    CUDA_CHECK(cudaEventRecord(buf_ready[slot].event, quant_stream));
                }

                loaded++;
            }

            quant_slot++;
        } else {
            // Full-precision weight: allocate storage and load directly
            std::vector<long> shape;
            if (spec.K > 0) {
                shape = {static_cast<long>(spec.M), static_cast<long>(spec.K)};
            } else {
                shape = {static_cast<long>(spec.M)};
            }

            Tensor tensor = allocator->allocate(
                ETensorDType::BF16,
                spec.name.c_str(),
                EAllocationType::ON_DEVICE,
                shape);

            bool success = loader.load_param(spec.name, tensor, true, spec.sharded, nullptr, stream);
            if (success) {
                weight_mgr->register_full_precision(spec.name, tensor);
                loaded++;
            }
        }
    }

    // ---- Step 6: Resolve tied parameters ----

    // Wait for all quantizations to finish first
    if (use_double_buffer) {
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
    }

    loader.resolve_tied_params([&](const std::string& name) -> Tensor& {
        return weight_mgr->get(name, stream);
    });

    // ---- Step 7: Synchronize to ensure all operations complete ----

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // ---- Step 8: Report statistics ----

    const size_t quant_bytes = weight_mgr->quantized_bytes();
    const size_t dequant_bytes = weight_mgr->dequant_buffer_bytes();
    const size_t fp_bytes = weight_mgr->full_precision_bytes();
    const size_t total_gpu_bytes = quant_bytes + dequant_bytes + fp_bytes;

    fprintf(stderr, "[QLoRA Import] Loaded %d/%d weights (%d quantized, %d full-precision) "
                    "in %.1f ms%s\n",
            loaded, total,
            weight_mgr->num_quantized(),
            weight_mgr->num_full_precision(),
            elapsed_ms,
            use_double_buffer ? " [double-buffered]" : "");

    fprintf(stderr, "[QLoRA Import] Memory: quantized=%.1f MB, dequant_buf=%.1f MB, "
                    "full_precision=%.1f MB, total=%.1f MB\n",
            static_cast<double>(quant_bytes) / (1024.0 * 1024.0),
            static_cast<double>(dequant_bytes) / (1024.0 * 1024.0),
            static_cast<double>(fp_bytes) / (1024.0 * 1024.0),
            static_cast<double>(total_gpu_bytes) / (1024.0 * 1024.0));

    if (weight_mgr->is_pooled()) {
        fprintf(stderr, "[QLoRA Import] Dequant buffer pool: max_cache=%d "
                        "(saves %.1f MB vs pre-allocated)\n",
                config.weight_manager_config.max_dequant_cache_size,
                static_cast<double>(
                    static_cast<size_t>(weight_mgr->num_quantized()) *
                    max_elem * 2  // BF16 = 2 bytes per element
                    - dequant_bytes
                ) / (1024.0 * 1024.0));
    }

    if (config.weight_manager_config.enable_offloading) {
        const auto* om = weight_mgr->offload_manager();
        if (om) {
            fprintf(stderr, "[QLoRA Import] Offloading: %d groups, %d resident, "
                            "gpu=%.1f MB, cpu=%.1f MB\n",
                    om->num_groups(), om->num_resident(),
                    static_cast<double>(om->gpu_memory_used()) / (1024.0 * 1024.0),
                    static_cast<double>(om->cpu_memory_used()) / (1024.0 * 1024.0));
        }
    }

    return weight_mgr;
}

// =============================================================================
// Config builder
// =============================================================================

DslQLoRAPipelineConfig build_pipeline_config(
    const dsl::MappingTable& mapping,
    const std::vector<WeightLoadSpec>& weight_specs,
    const QuantizerConfig& quantizer_config,
    int shard_idx,
    int num_shards) {

    DslQLoRAPipelineConfig config;
    config.mapping = mapping;
    config.weight_specs = weight_specs;
    config.quantizer_config = quantizer_config;
    config.shard_idx = shard_idx;
    config.num_shards = num_shards;

    // Default weight manager config
    config.weight_manager_config.device_id = quantizer_config.device_id;

    return config;
}

}  // namespace qlora
