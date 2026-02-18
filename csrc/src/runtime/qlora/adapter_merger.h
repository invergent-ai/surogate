// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// AdapterMerger: Merges a PEFT LoRA adapter into BF16 base weights in-place.
//
// This class is used during weight import to merge a previously-trained
// adapter into the base model weights before quantization (QLoRA) or
// storage (BF16 LoRA). This enables iterative fine-tuning: train adapter A,
// merge A into base, then train adapter B on the merged weights.
//
// The merger operates on the BF16 weight buffers produced by DslWeightLoader,
// applying the LoRA delta (scaling * B @ A) via cuBLAS GEMM directly into the
// buffer — no temporary delta tensor is needed.
//
// This class has zero coupling to training infrastructure and can be reused
// for inference pipelines.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_ADAPTER_MERGER_H
#define SUROGATE_SRC_RUNTIME_QLORA_ADAPTER_MERGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "runtime/dsl/mapping_spec.h"
#include "runtime/dsl/dsl_weight_loader.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace qlora {

/// Merges a PEFT LoRA adapter into base model weights during import.
///
/// Constructed once per import operation with the adapter directory.
/// For each weight parameter, call apply() after loading BF16 data and
/// before quantization. The merger resolves DSL param names to HF names
/// via the mapping table, then loads and applies matching LoRA deltas.
class AdapterMerger {
public:
    /// Construct and load adapter metadata.
    ///
    /// Opens adapter_config.json for rank/alpha/target_modules, and
    /// adapter_model.safetensors for weight data. Pre-builds the set
    /// of HF weight names that have adapter deltas.
    ///
    /// @param adapter_dir   Path to PEFT adapter directory.
    /// @param hf_mapping    DSL → HF weight mapping table.
    /// @param base_reader   Base model SafeTensors reader (for component sizes in fused weights).
    /// @param shard         Sharding config for multi-GPU weight loading.
    AdapterMerger(const std::string& adapter_dir,
                  const dsl::MappingTable& hf_mapping,
                  SafeTensorsReader& base_reader,
                  dsl::ShardConfig shard = {});

    ~AdapterMerger();

    // Non-copyable, non-movable (owns cuBLAS handle + GPU buffers).
    AdapterMerger(const AdapterMerger&) = delete;
    AdapterMerger& operator=(const AdapterMerger&) = delete;

    /// Merge adapter deltas into a loaded BF16 parameter.
    ///
    /// Resolves the DSL parameter name to HF component names via the mapping
    /// table, and for each component that has an adapter, loads lora_A/B and
    /// applies the delta: W += scaling * B @ A.
    ///
    /// For Fuse mappings (e.g., fused QKV), each component's delta is applied
    /// at the correct row offset within the fused buffer.
    ///
    /// @param dsl_param_name  Internal param name (e.g., "blocks[0].qkv_weight").
    /// @param bf16_weight     BF16 tensor on GPU (modified in-place).
    /// @param stream          CUDA stream for the merge operations.
    void apply(const std::string& dsl_param_name,
               Tensor& bf16_weight,
               cudaStream_t stream);

    /// Merge adapter delta for a single expert in per-expert streaming.
    ///
    /// Used in Pass 2 of the QLoRA pipeline where expert weights are loaded
    /// one at a time into a small [per_M, K] buffer.
    ///
    /// @param dsl_param_name  Internal param name (e.g., "moe_blocks[0].experts_gate_up").
    /// @param expert_idx      Global expert index (0-based).
    /// @param bf16_expert     BF16 tensor [per_M, K] on GPU (modified in-place).
    /// @param stream          CUDA stream.
    void apply_expert(const std::string& dsl_param_name,
                      int expert_idx,
                      Tensor& bf16_expert,
                      cudaStream_t stream);

    /// Number of adapter targets that were found in the adapter.
    [[nodiscard]] int num_targets() const { return static_cast<int>(mAdapterTargets.size()); }

private:
    /// Info about one LoRA adapter pair (A and B matrices).
    struct LoRAPair {
        std::string lora_a_key;  ///< Full safetensors key for lora_A
        std::string lora_b_key;  ///< Full safetensors key for lora_B
    };

    /// Resolve a DSL param name to its MappingSpec and layer index.
    const dsl::MappingSpec* find_spec(const std::string& dsl_name, int& layer_idx) const;

    /// Strip ".weight" suffix from an HF name to get the base name.
    static std::string strip_weight_suffix(const std::string& hf_name);

    /// Merge a single component's adapter delta into the weight buffer.
    ///
    /// @param hf_base_name  HF name without ".weight" (e.g., "model.layers.0.self_attn.q_proj").
    /// @param bf16_weight   The full BF16 weight tensor.
    /// @param row_offset    Row offset within the weight tensor for this component.
    /// @param component_rows Number of rows for this component (0 = use lora_B rows).
    /// @param stream        CUDA stream.
    void merge_component(const std::string& hf_base_name,
                         Tensor& bf16_weight,
                         long row_offset,
                         long component_rows,
                         cudaStream_t stream);

    /// Ensure GPU buffer is at least `bytes` large, reallocating if needed.
    static void ensure_buf(void*& buf, size_t& current_size, size_t needed);

    // ---- State ----

    SafeTensorsReader mAdapterReader;
    const dsl::MappingTable& mMapping;
    SafeTensorsReader& mBaseReader;
    dsl::ShardConfig mShard;

    float mScaling = 0.0f;   ///< alpha / rank (or alpha / sqrt(rank) for RS-LoRA)
    int mRank = 0;

    /// Map: HF weight base name (without ".weight") → LoRA pair keys.
    /// e.g., "model.layers.0.self_attn.q_proj" → {lora_a_key, lora_b_key}
    std::unordered_map<std::string, LoRAPair> mAdapterTargets;

    /// cuBLAS handle for GEMM operations.
    cublasHandle_t mCublasHandle = nullptr;

    /// Reusable GPU scratch buffers for loading lora_A and lora_B.
    void* mLoraABuf = nullptr;
    void* mLoraBBuf = nullptr;
    size_t mLoraABufBytes = 0;
    size_t mLoraBBufBytes = 0;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_ADAPTER_MERGER_H
