// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H
#define SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H

#include <functional>
#include <string>
#include <vector>
#include <cublas_v2.h>

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

// 8-bit AdamW optimizer state for LoRA weights
struct LoRAAdamW8BitState {
    bool initialized = false;
    bool values_restored = false;  // Set when state values loaded from checkpoint
    bool grad_ptrs_initialized = false;  // Set after grad pointer array is populated
    size_t total_params = 0;
    size_t num_blocks = 0;
    int num_tensors = 0;

    // Offloading configuration
    bool offload_state = false;  // If true, state tensors are in pinned host memory
    bool use_zero_copy = false;  // If true, use zero-copy access instead of transfers

    Tensor quantiles1;   // 256 entries - quantization map for m
    Tensor quantiles2;   // 256 entries - quantization map for v
    Tensor state1;       // uint8 quantized m, size = total_params
    Tensor state2;       // uint8 quantized v, size = total_params
    Tensor absmax1;      // per-block absmax for m
    Tensor absmax2;      // per-block absmax for v

    // Multi-tensor optimizer buffers (device memory)
    // Pre-allocated arrays of pointers/sizes to avoid per-step CPU work
    Tensor param_ptrs;      // float** or nv_bfloat16** - array of param pointers
    Tensor grad_ptrs;       // float** or nv_bfloat16** - array of grad pointers
    Tensor tensor_sizes;    // int* - array of tensor sizes
    Tensor state_offsets;   // int* - element offset for each tensor in state buffers
};

// NorMuon optimizer state for LoRA weights
// Uses 8-bit quantized momentum + FP32 variance buffers
struct LoRANorMuonState {
    bool initialized = false;
    bool values_restored = false;  // Set when state values loaded from checkpoint
    size_t total_params = 0;
    size_t state_elems = 0;
    size_t num_blocks = 0;

    // 8-bit quantized momentum buffer (combined for all LoRA weights)
    Tensor momentum_quantiles;  // float[256] - signed quantization map
    Tensor momentum_state;      // uint8[state_elems]
    Tensor momentum_absmax;     // float[num_blocks]

    // Variance buffers - stored per LoRA tensor as FP32
    // For LoRA, each A/B matrix is a 2D weight
    std::vector<Tensor> variance_buffers;
    std::vector<std::pair<int, int>> variance_shapes;  // (M, N) for each buffer

    // Polar Express workspace (reused across layers)
    Tensor polar_workspace;
    size_t max_weight_M = 0;  // Max weight rows seen
    size_t max_weight_N = 0;  // Max weight cols seen

    // Temporary buffer for dequantized momentum (reused per weight)
    Tensor momentum_temp;  // BF16[max_weight_size]

    // cuBLAS handle for Polar Express matrix multiplications
    cublasHandle_t cublas_handle = nullptr;

    ~LoRANorMuonState() {
        if (cublas_handle) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_OPTIMIZER_STATE_H
