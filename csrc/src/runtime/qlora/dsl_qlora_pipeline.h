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

    /// Expert Parallelism: each GPU loads only local experts.
    int ep_rank = 0;   ///< This GPU's rank within the EP group
    int ep_size = 1;   ///< Number of GPUs in the EP group (1 = no EP)

    /// List of weight parameters to load and their properties.
    std::vector<WeightLoadSpec> weight_specs;

    /// The HF mapping table (internal names -> HF SafeTensors paths).
    dsl::MappingTable mapping;

    // =========================================================================
    // Pre-quantized model loading
    // =========================================================================

    /// Whether this is a pre-quantized HF model (skip online quantization).
    /// When true, quantized data and scales are loaded directly from safetensors.
    bool prequantized = false;

    /// Suffix appended to HF weight name to get the data tensor name.
    /// Empty for FP8/NVFP4 (data tensor uses the same name as the weight).
    /// "_blocks" for MXFP4 (HF stores packed data with _blocks suffix).
    std::string data_suffix;

    /// Suffix appended to HF weight name to get the scale tensor name.
    /// "_scale_inv" for FP8, "_scale" for NVFP4, "_scales" for MXFP4.
    std::string scale_suffix;

    /// Suffix for second-level scale tensor (NVFP4 only: "_scale_2" for global scale).
    /// Empty for formats with single-level scaling.
    std::string scale2_suffix;

    /// HF module paths that should NOT be quantized (loaded full-precision).
    /// Populated from HF quantization_config "ignore" / "modules_to_not_convert".
    std::vector<std::string> modules_to_not_convert;

    // =========================================================================
    // Adapter merging (stacked LoRA)
    // =========================================================================

    /// Path to a PEFT adapter directory to merge into base weights before
    /// quantization. Empty string means no adapter merge.
    std::string adapter_path;
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

/// Import pre-quantized weights from HuggingFace SafeTensors into a GenericWeightManager.
///
/// Unlike import_and_quantize_weights(), this loads already-quantized data
/// (FP8/NVFP4/MXFP4) directly from safetensors without online quantization.
/// Scale tensors are loaded using the configured suffixes.
///
/// @param file_name   Path to HuggingFace SafeTensors checkpoint directory or file.
/// @param config      Pipeline configuration (must have prequantized=true).
/// @param pt_config   Pretrained model configuration.
/// @param allocator   Tensor allocator for all GPU/CPU memory.
/// @param stream      CUDA stream for async operations.
///
/// @return Fully initialized GenericWeightManager with all weights loaded.
std::unique_ptr<GenericWeightManager> import_prequantized_weights(
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
