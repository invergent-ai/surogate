// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H
#define SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H

#include <functional>
#include <string>

#include "lora_types.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"
#include "utilities/dtype.h"
#include "utilities/allocator.h"

class NCCLCommunicator;

namespace modules {

/**
 * @brief Modular LoRA optimizer state manager
 *
 * Manages Adam optimizer state (m and v) for LoRA parameters.
 */
class ModularLoRAOptimizerState {
public:
    struct Config {
        int num_layers;
        int hidden_size;
        int intermediate_size;
        int num_query_heads;
        int num_kv_heads;
        int head_size;
        ModularLoRAConfig lora_config;
        ETensorDType m_dtype;
        ETensorDType v_dtype;
        int shard_idx = 0;
        int num_shards = 1;
        bool offload_m = false;
        bool offload_v = false;
        bool use_zero_copy = false;
        EAllocationType offload_alloc = EAllocationType::PINNED;
    };

    ModularLoRAOptimizerState(const Config& config, cudaStream_t stream,
                               NCCLCommunicator& comm, TensorAllocator& allocator);
    ~ModularLoRAOptimizerState();

    /**
     * @brief Get block momentum for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_m(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get block variance for optimizer
     */
    LoRABlockWeights<TensorShard>& get_block_v(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for block momentum (m)
     */
    LoRABlockWeights<TensorShard>& get_block_m_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get FP8 scales for block variance (v)
     */
    LoRABlockWeights<TensorShard>& get_block_v_scales(int layer_idx, cudaStream_t stream);

    /**
     * @brief Tensor containers for checkpointing (names match PEFT adapter tensors)
     */
    ITensorContainer& full_m();
    ITensorContainer& full_v();
    ITensorContainer& full_m_scales();
    ITensorContainer& full_v_scales();

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] Tensor& staging_m() { return mStagingM; }
    [[nodiscard]] Tensor& staging_v() { return mStagingV; }
    [[nodiscard]] Tensor& staging_m_scales() { return mStagingMScales; }
    [[nodiscard]] Tensor& staging_v_scales() { return mStagingVScales; }

private:
    Config mConfig;
    TensorAllocator* mAllocator;

    LoRAWeightsSet<TensorShard> mMomentum;   // First moment (m)
    LoRAWeightsSet<TensorShard> mVariance;   // Second moment (v)
    LoRAWeightsSet<TensorShard> mMomentumScales;   // FP8 scales for m (FP32)
    LoRAWeightsSet<TensorShard> mVarianceScales;   // FP8 scales for v (FP32)

    class StateContainer final : public ITensorContainer {
    public:
        explicit StateContainer(LoRAWeightsSet<TensorShard>* set) : mSet(set) {}
        void set(LoRAWeightsSet<TensorShard>* set) { mSet = set; }
        void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;
    private:
        LoRAWeightsSet<TensorShard>* mSet = nullptr;
    };

    StateContainer mMomentumContainer{&mMomentum};
    StateContainer mVarianceContainer{&mVariance};
    StateContainer mMomentumScalesContainer{&mMomentumScales};
    StateContainer mVarianceScalesContainer{&mVarianceScales};

    // Device staging buffers used when optimizer state is offloaded to host.
    // These are reused across all tensors and rely on stream ordering for correctness.
    Tensor mStagingM;
    Tensor mStagingV;
    Tensor mStagingMScales;
    Tensor mStagingVScales;

    void allocate_state();
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H
