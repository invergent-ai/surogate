// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DslQLoRAPipeline: Unified weight loading + quantization pipeline.
//
// Connects DslWeightLoader (BF16 weight loading from HF SafeTensors) with
// GenericWeightManager (quantized storage and lazy dequantization). This is
// the entry point for loading a model's weights into the new generic QLoRA
// system, replacing the architecture-specific BnBWeightsManager/FP8/FP4.
//
// Usage:
//   1. Create pipeline config from IR and runtime options
//   2. Call import_and_quantize_weights() with safetensors path
//   3. Use the returned GenericWeightManager for weight access

#ifndef SUROGATE_SRC_RUNTIME_QLORA_DSL_QLORA_PIPELINE_H
#define SUROGATE_SRC_RUNTIME_QLORA_DSL_QLORA_PIPELINE_H

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/dsl/mapping_spec.h"
#include "runtime/qlora/generic_weight_manager.h"
#include "runtime/qlora/generic_quantizer.h"

class SafeTensorsReader;
class TensorAllocator;
class PretrainedConfig;
class NCCLCommunicator;

namespace dsl {
struct ShardConfig;
struct MoEWeightConfig;
}

namespace qlora {

/// Describes a weight parameter to be loaded and (optionally) quantized.
struct WeightLoadSpec {
    /// Internal parameter name (e.g., "blocks[0].qkv_weight", "embedding").
    std::string name;

    /// Flattened 2D dimensions for quantizer: M = total rows, K = columns.
    /// For 2D weights [R, C]: M=R, K=C.
    /// For 3D weights [E, R, C]: M=E*R, K=C (flattened for quantizer).
    int M = 0;
    int K = 0;

    /// Full tensor shape for loading and dequant buffer allocation.
    /// For 2D: {M, K}. For 3D experts: {E, per_expert_M, K}.
    /// Empty means use {M, K} as the shape.
    std::vector<long> shape;

    /// Whether this weight should be quantized (true) or kept full precision.
    /// Norms, biases, embeddings, and lm_head are typically not quantized.
    bool quantize = true;

    /// Offload group ID (-1 = no offloading, always on GPU).
    int offload_group = -1;

    /// Whether this parameter is sharded across GPUs.
    bool sharded = false;

    /// Target dtype for full-precision weights (e.g., FP32 for SSM params).
    /// BF16 by default; only relevant for non-quantized weights.
    ETensorDType target_dtype = ETensorDType::BF16;
};

/// Configuration for the weight loading + quantization pipeline.
struct DslQLoRAPipelineConfig {
    /// Quantizer configuration (format, block size, double quant, etc.).
    QuantizerConfig quantizer_config;

    /// GenericWeightManager configuration.
    GenericWeightManagerConfig weight_manager_config;

    /// Sharding configuration for multi-GPU.
    int shard_idx = 0;
    int num_shards = 1;

    /// MoE configuration.
    int num_experts = 0;
    int moe_intermediate_size = 0;

    /// List of weight parameters to load and their properties.
    std::vector<WeightLoadSpec> weight_specs;

    /// The HF mapping table (internal names -> HF SafeTensors paths).
    dsl::MappingTable mapping;
};

/// Import weights from HuggingFace SafeTensors and quantize into a GenericWeightManager.
///
/// This is the main entry point for the unified pipeline. It:
///   1. Creates a GenericWeightManager with the specified quantizer
///   2. Registers all weights (quantized or full-precision)
///   3. Uses DslWeightLoader to load BF16 weights from SafeTensors
///   4. Quantizes each weight via the GenericWeightManager
///   5. Resolves tied parameters
///
/// @param file_name   Path to HuggingFace SafeTensors checkpoint directory or file.
/// @param config      Pipeline configuration (quantizer, mapping, weight specs).
/// @param pt_config   Pretrained model configuration (for fuse slice inference).
/// @param allocator   Tensor allocator for all GPU/CPU memory.
/// @param stream      CUDA stream for async operations.
///
/// @return Fully initialized GenericWeightManager with all weights loaded and quantized.
std::unique_ptr<GenericWeightManager> import_and_quantize_weights(
    const std::string& file_name,
    const DslQLoRAPipelineConfig& config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator,
    cudaStream_t stream);

/// Build a DslQLoRAPipelineConfig from the DSL IR.
///
/// Extracts weight specs and mapping from the compiled IR, combining
/// with runtime options (quantization format, offloading, sharding).
///
/// @param mapping           HF weight mapping table from IR.
/// @param weight_specs      Weight load specifications.
/// @param quantizer_config  Quantizer configuration.
/// @param shard_idx         This GPU's shard index.
/// @param num_shards        Total number of GPUs.
///
/// @return Pipeline configuration ready for import_and_quantize_weights().
DslQLoRAPipelineConfig build_pipeline_config(
    const dsl::MappingTable& mapping,
    const std::vector<WeightLoadSpec>& weight_specs,
    const QuantizerConfig& quantizer_config,
    int shard_idx = 0,
    int num_shards = 1);

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_DSL_QLORA_PIPELINE_H
