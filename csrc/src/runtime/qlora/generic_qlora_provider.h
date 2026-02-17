// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GenericQLoRAProvider: QLoRAWeightProvider backed by GenericWeightManager.

#ifndef SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QLORA_PROVIDER_H
#define SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QLORA_PROVIDER_H

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "runtime/core/qlora_provider.h"
#include "runtime/qlora/generic_weight_manager.h"
#include "runtime/qlora/dsl_qlora_pipeline.h"

class PretrainedConfig;

namespace qlora {

/// QLoRAWeightProvider implementation that delegates to GenericWeightManager.
///
/// This adapter allows the existing DSL runtime to use the new generic
/// quantization system without changes to the DslModel graph executor.
class GenericQLoRAProvider final : public dsl::QLoRAWeightProvider {
public:
    /// Construct with a pre-built GenericWeightManager.
    ///
    /// @param weight_mgr  Initialized weight manager (with weights loaded)
    explicit GenericQLoRAProvider(std::unique_ptr<GenericWeightManager> weight_mgr);

    /// Construct with pipeline configuration (deferred import).
    ///
    /// Weights are loaded during import_and_quantize().
    ///
    /// @param config      Pipeline configuration
    /// @param pt_config   Pretrained model configuration
    /// @param allocator   Tensor allocator
    GenericQLoRAProvider(
        DslQLoRAPipelineConfig config,
        const PretrainedConfig& pt_config,
        std::shared_ptr<TensorAllocator> allocator);

    ~GenericQLoRAProvider() override;

    // =========================================================================
    // QLoRAWeightProvider interface
    // =========================================================================

    bool handles_param(std::string_view name) const override;

    Tensor& resolve_param(std::string_view name, cudaStream_t stream) override;

    void import_and_quantize(const std::string& file_name,
                             NCCLCommunicator& comm,
                             cudaStream_t stream) override;

    void invalidate_cache() override;

    bool refresh_moe_experts(int layer_idx,
                             const modules::SelectiveExpertInfo& selection,
                             cudaStream_t stream) override;

    void prefetch_for_layer(int layer_idx, cudaStream_t stream) override;

    [[nodiscard]] bool has_offloading() const override;

    std::size_t quantized_weights_bytes() const override;

    float memory_savings_ratio() const override;

    // =========================================================================
    // Quantized data access (for EP quantized weight transfer)
    // =========================================================================

    const qlora::QuantizedTensor* try_get_quantized(std::string_view name) const override;
    qlora::IQuantizer* get_quantizer() const override;
    void auto_tune_offloading() override;

    // =========================================================================
    // Direct access to the underlying weight manager
    // =========================================================================

    GenericWeightManager* weight_manager() { return mWeightMgr.get(); }
    const GenericWeightManager* weight_manager() const { return mWeightMgr.get(); }

private:
    /// Build the layer → offload groups mapping from weight names.
    void build_layer_offload_map();

    std::unique_ptr<GenericWeightManager> mWeightMgr;

    // Deferred auto-tune: run after first training step when all lazy allocs are settled.
    int mStepCount = 0;
    bool mAutoTunePending = false;

    // Deferred construction state
    std::unique_ptr<DslQLoRAPipelineConfig> mDeferredConfig;
    const PretrainedConfig* mPtConfig = nullptr;
    std::shared_ptr<TensorAllocator> mAllocator;

    /// Total BF16 bytes (for memory savings calculation).
    size_t mTotalBF16Bytes = 0;

    /// Map from layer index to the set of offload group IDs that
    /// have weights in that layer. Used for prefetch scheduling.
    std::unordered_map<int, std::unordered_set<int>> mLayerOffloadGroups;

    /// Whether any weights use offloading.
    bool mHasOffloading = false;

    /// EP size (> 1 means LLEP may use extra GPU memory for foreign weight transfers).
    int mEPSize = 1;
};

}  // namespace qlora

#endif  // SUROGATE_SRC_RUNTIME_QLORA_GENERIC_QLORA_PROVIDER_H
