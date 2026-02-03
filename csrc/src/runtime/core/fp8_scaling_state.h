// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FP8_SCALING_STATE_H
#define SUROGATE_SRC_MODULES_FP8_SCALING_STATE_H

#include <memory>

#include "fp8_scaling_config.h"
#include "utilities/allocator.h"
#include "utilities/tensor.h"

namespace modules {

/**
 * @brief State manager for FP8 delayed scaling
 *
 * Manages the amax history buffer and current scales for all quantizers.
 * The delayed scaling approach uses scales computed from previous iterations'
 * abs_max values, providing more stable FP8 training.
 *
 * Memory layout:
 * - amax_history: [amax_history_len, NUM_FP8_QUANTIZERS] - rolling history buffer
 * - scales: [NUM_FP8_QUANTIZERS] - current scale factors to use for quantization
 * - recorded_amaxes: [NUM_FP8_QUANTIZERS] - abs_max values from current iteration
 *
 * Usage:
 * 1. During forward/backward, call get_scale() to get the delayed scale
 * 2. After quantization, call record_amax() to store the current abs_max
 * 3. At end of optimizer step, call update_scales() to roll history and compute new scales
 */
class FP8ScalingState {
public:
    /**
     * @brief Construct and allocate FP8 scaling state
     *
     * @param config Delayed scaling configuration
     * @param allocator Tensor allocator for GPU memory
     * @param device_id GPU device ID
     * @param num_layers Number of transformer layers (for per-layer quantizers)
     */
    FP8ScalingState(const FP8ScalingConfig& config,
                    const std::shared_ptr<TensorAllocator>& allocator,
                    int device_id,
                    int num_layers);

    // Non-copyable
    FP8ScalingState(const FP8ScalingState&) = delete;
    FP8ScalingState& operator=(const FP8ScalingState&) = delete;

    // Movable
    FP8ScalingState(FP8ScalingState&&) noexcept = default;
    FP8ScalingState& operator=(FP8ScalingState&&) noexcept = default;

    /**
     * @brief Get the scale factor for a specific quantizer
     *
     * Returns a device pointer to the scale factor. This scale was computed
     * from previous iteration(s) amax values.
     *
     * @param idx Quantizer index
     * @return Device pointer to scale factor (single float)
     */
    [[nodiscard]] float* get_scale(QuantizerIndex idx) {
        return mScales.get<float>() + static_cast<int>(idx);
    }

    /**
     * @brief Get the pointer where abs_max should be recorded for a quantizer
     *
     * The quantization kernel should write the computed abs_max to this location.
     * It will be used in update_scales() to update the history and compute new scales.
     *
     * @param idx Quantizer index
     * @return Device pointer for recording abs_max
     */
    [[nodiscard]] float* get_recorded_amax_ptr(QuantizerIndex idx) {
        return mRecordedAmaxes.get<float>() + static_cast<int>(idx);
    }

    /**
     * @brief Get all recorded amaxes as a single tensor
     *
     * Used for all-reduce across ranks in distributed training.
     */
    [[nodiscard]] Tensor& recorded_amaxes() { return mRecordedAmaxes; }

    /**
     * @brief Get all scales as a single tensor
     */
    [[nodiscard]] Tensor& scales() { return mScales; }

    /**
     * @brief Get the amax history buffer
     */
    [[nodiscard]] Tensor& amax_history() { return mAmaxHistory; }

    /**
     * @brief Get the configuration
     */
    [[nodiscard]] const FP8ScalingConfig& config() const { return mConfig; }

    /**
     * @brief Get the number of layers
     */
    [[nodiscard]] int num_layers() const { return mNumLayers; }

    /**
     * @brief Get the total number of quantizers
     */
    [[nodiscard]] int num_quantizers() const { return mNumQuantizers; }

    /**
     * @brief Reset all state to initial values
     *
     * Sets scales to 1.0, history to 0, recorded amaxes to 0.
     * Called at initialization and can be called to reset state.
     */
    void reset(cudaStream_t stream);

    /**
     * @brief Zero out recorded amaxes before a new forward/backward pass
     *
     * Should be called at the start of each training step to ensure
     * clean accumulation of abs_max values.
     */
    void zero_recorded_amaxes(cudaStream_t stream);

private:
    FP8ScalingConfig mConfig;
    int mDeviceId;
    int mNumLayers;       ///< Number of transformer layers
    int mNumQuantizers;   ///< Total number of quantizers (num_layers * NUM_QUANTIZERS_PER_LAYER)

    /// Amax history buffer: [amax_history_len, num_quantizers]
    /// Row 0 = most recent, row N-1 = oldest
    Tensor mAmaxHistory;

    /// Current scale factors: [num_quantizers]
    /// These are the delayed scales to use for quantization
    Tensor mScales;

    /// Recorded abs_max values from current iteration: [num_quantizers]
    /// Updated by quantization kernels, consumed by update_scales()
    Tensor mRecordedAmaxes;
};

/**
 * @brief Update amax history and compute new scales (CUDA kernel launch)
 *
 * This function should be called once per optimizer step (not per layer).
 * It rolls the history buffer, inserts current amaxes, and computes new scales.
 *
 * @param state FP8 scaling state to update
 * @param stream CUDA stream for async execution
 */
void delayed_scaling_update(FP8ScalingState& state, cudaStream_t stream);

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FP8_SCALING_STATE_H
