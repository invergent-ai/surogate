// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RESIDUAL_MANAGER_H
#define SUROGATE_SRC_MODULES_RESIDUAL_MANAGER_H

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include "utilities/tensor.h"
#include "utilities/allocator.h"
#include "utilities/utils.h"
#include "run_state_types.h"

namespace modules {

/**
 * @brief Manages residual buffers with optional CPU offloading
 */
class ResidualManager {
public:
    /**
     * @brief Construct a new Residual Manager
     *
     * @param allocator Tensor allocator
     * @param num_layers Total number of layers
     * @param batch_size Batch size
     * @param seq_length Sequence length
     * @param hidden_size Hidden dimension
     * @param dtype Data type for residual tensors
     * @param offload_residuals Whether to offload residuals to host memory
     * @param num_residual_buffers Number of device buffers for double-buffering (when offloading)
     * @param main_stream Stream to use for initial event synchronization
     */
    ResidualManager(
        std::shared_ptr<TensorAllocator> allocator,
        int num_layers,
        int batch_size,
        int seq_length,
        int hidden_size,
        ETensorDType dtype,
        bool offload_residuals,
        int num_residual_buffers,
        cudaStream_t main_stream)
        : mAllocator(allocator),
          mNumLayers(num_layers),
          mOffloadResiduals(offload_residuals),
          mNumResidualBuffers(num_residual_buffers)
    {
        long B = batch_size;
        long T = seq_length;
        long C = hidden_size;

        // Final residual is always needed and must not alias any per-layer residual buffer.
        mFinalResidual = mAllocator->allocate(
            dtype, "final_residual", EAllocationType::ON_DEVICE, {B, T, C});

        if (mOffloadResiduals) {
            // With offloading: double-buffered device + per-layer host
            mDeviceResiduals.resize(mNumResidualBuffers);
            for (int i = 0; i < mNumResidualBuffers; ++i) {
                mDeviceResiduals[i] = mAllocator->allocate(
                    dtype, ("device_residual_" + std::to_string(i)).c_str(),
                    EAllocationType::ON_DEVICE, {B, T, C});
            }

            // Host buffers (one per layer)
            mOffloadedResiduals.resize(mNumLayers);
            for (int i = 0; i < mNumLayers; ++i) {
                mOffloadedResiduals[i] = mAllocator->allocate(
                    dtype, ("offloaded_residual_" + std::to_string(i)).c_str(),
                    EAllocationType::PINNED, {B, T, C});
            }

            // State tracking for double-buffering
            mResidualState.resize(mNumResidualBuffers);
            for (int i = 0; i < mNumResidualBuffers; ++i) {
                CUDA_CHECK(cudaEventCreate(&mResidualState[i].event));
                CUDA_CHECK(cudaEventCreate(&mResidualState[i].ready_event));
                mResidualState[i].layer_idx = -1;
                mResidualState[i].is_ready = false;
                // Mark the buffer event as "done" initially so first wait doesn't stall.
                CUDA_CHECK(cudaEventRecord(mResidualState[i].event, main_stream));
                CUDA_CHECK(cudaEventRecord(mResidualState[i].ready_event, main_stream));
            }
        } else {
            // Without offloading: one device buffer per layer
            mDeviceResiduals.resize(mNumLayers);
            for (int i = 0; i < mNumLayers; ++i) {
                mDeviceResiduals[i] = mAllocator->allocate(
                    dtype, ("res_ffn_" + std::to_string(i)).c_str(),
                    EAllocationType::ON_DEVICE, {B, T, C});
            }
        }
    }

    ~ResidualManager() {
        if (mNumResidualBuffers > 0) {
            for (auto& state : mResidualState) {
                if (state.event) cudaEventDestroy(state.event);
                if (state.ready_event) cudaEventDestroy(state.ready_event);
            }
        }
    }

    // Disable copy/assignment
    ResidualManager(const ResidualManager&) = delete;
    ResidualManager& operator=(const ResidualManager&) = delete;

    /**
     * @brief Initiate prefetch of residual from host (CPU) to device (GPU)
     */
    void fetch_residual(int layer_idx, cudaStream_t fetch_stream) {
        if (!mOffloadResiduals) return;

        const int buf_idx = layer_idx % mNumResidualBuffers;
        auto& status = mResidualState.at((std::size_t)buf_idx);

        CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.event, 0));
        status.layer_idx = layer_idx;
        status.is_ready = false;

        const size_t size = mOffloadedResiduals.at(layer_idx).bytes();
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceResiduals.at((std::size_t)buf_idx).Data,
            mOffloadedResiduals.at(layer_idx).Data,
            size,
            cudaMemcpyHostToDevice,
            fetch_stream));

        CUDA_CHECK(cudaEventRecord(status.event, fetch_stream));
    }

    /**
     * @brief Store residual from device (GPU) back to host (CPU)
     */
    void put_residual(int layer_idx, cudaStream_t put_stream) {
        if (!mOffloadResiduals) return;

        const int buf_idx = layer_idx % mNumResidualBuffers;
        auto& status = mResidualState.at((std::size_t)buf_idx);
        status.is_ready = false;
        if (status.layer_idx != layer_idx) {
            throw std::logic_error("ResidualManager::put_residual: mismatched layer index");
        }

        CUDA_CHECK(cudaStreamWaitEvent(put_stream, status.ready_event, 0));

        const size_t size = mDeviceResiduals.at((std::size_t)buf_idx).bytes();
        CUDA_CHECK(cudaMemcpyAsync(
            mOffloadedResiduals.at(layer_idx).Data,
            mDeviceResiduals.at((std::size_t)buf_idx).Data,
            size,
            cudaMemcpyDeviceToHost,
            put_stream));

        CUDA_CHECK(cudaEventRecord(status.event, put_stream));
    }

    /**
     * @brief Get residual tensor for a layer, ensuring it is ready on the given stream
     */
    Tensor& get_residual(int layer_idx, cudaStream_t stream) {
        if (!mOffloadResiduals) {
            return mDeviceResiduals.at((std::size_t)layer_idx);
        }

        const int buf_idx = layer_idx % mNumResidualBuffers;
        auto& status = mResidualState.at((std::size_t)buf_idx);
        if (!status.is_ready) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, status.event, 0));
            status.is_ready = true;
        }
        return mDeviceResiduals.at((std::size_t)buf_idx);
    }

    /**
     * @brief Mark residual as ready on the device after production
     */
    void mark_residual_ready(int layer_idx, cudaStream_t stream) {
        if (!mOffloadResiduals) return;

        const int buf_idx = layer_idx % mNumResidualBuffers;
        auto& status = mResidualState.at((std::size_t)buf_idx);
        status.layer_idx = layer_idx;
        CUDA_CHECK(cudaEventRecord(status.ready_event, stream));
    }

    /**
     * @brief Release residual (allow buffer reuse) after consumption
     */
    void release_residual(int layer_idx, cudaStream_t stream) {
        if (!mOffloadResiduals) return;

        const int buf_idx = layer_idx % mNumResidualBuffers;
        auto& status = mResidualState.at((std::size_t)buf_idx);
        status.is_ready = false;
        CUDA_CHECK(cudaEventRecord(status.event, stream));
    }

    /**
     * @brief Get the final residual buffer (output of the model block stack)
     */
    Tensor& get_final_residual() { return mFinalResidual; }

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    int mNumLayers;
    bool mOffloadResiduals;
    int mNumResidualBuffers;

    std::vector<Tensor> mDeviceResiduals;
    std::vector<Tensor> mOffloadedResiduals;
    std::vector<ResidualState> mResidualState;
    Tensor mFinalResidual;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_RESIDUAL_MANAGER_H