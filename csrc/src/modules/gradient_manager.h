// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_GRADIENT_MANAGER_H
#define SUROGATE_SRC_MODULES_GRADIENT_MANAGER_H

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "kernels/kernels.h"
#include "module_concept.h"
#include "utilities/allocator.h"
#include "utilities/tensor.h"
#include "utilities/philox.h"
#include "utilities/comm.h"
#include "modules/mlp_utils.h"

namespace modules {

/**
 * @brief Modular gradient manager template
 *
 * Manages gradient tensors for a model composed of modular blocks. Handles:
 * - Gradient accumulation across micro-steps
 * - All-reduce across data-parallel ranks
 * - Scatter-reduce for memory efficiency
 *
 * The gradient manager provides two views of gradients:
 * 1. Full gradients: Used during backward pass computation
 * 2. Sharded gradients: Used by optimizer (after all-reduce)
 *
 * @tparam Block The transformer block module type
 */
template<typename Block>
class ModularGradientManager {
public:
    using BlockGradients = typename Block::Gradients;

    /**
     * @brief Configuration for gradient manager
     */
    struct Config {
        int num_layers;
        typename Block::Config block_config;

        ETensorDType grad_dtype;        ///< Gradient dtype (typically BF16)
        int shard_idx;
        int num_shards;

        // Memory optimization
        bool shard_gradients = false;   ///< Use scatter-reduce instead of all-reduce
        bool use_all_to_all_reduce = false; ///< ZeRO-2: all-to-all reducer (legacy parity)
        bool offload_grads = false;     ///< ZeRO-2: store persistent shards on host (pinned / WC)
        EAllocationType offload_alloc = EAllocationType::PINNED; ///< Host alloc type for offloaded grads

        // LoRA mode: skip per-layer gradient allocation (use single shared buffer)
        // The gradients are computed but immediately discarded (not accumulated or reduced)
        bool skip_allocation = false;

        // Train MoE router gate during LoRA fine-tuning
        // When enabled, router gradient is allocated and reduced even in skip_allocation mode
        bool train_router = false;

        // Non-block config
        int vocab_size;
        int hidden_size;
        bool tied_embeddings = false;

        // Hybrid layer markers (optional)
        bool has_mamba = false;
        std::vector<std::uint8_t> layer_is_mamba;
        bool has_moe = false;
        std::vector<std::uint8_t> layer_is_moe;
    };

    /**
     * @brief Non-block gradients
     */
    struct NonBlockGradients {
        Tensor d_embeddings;
        Tensor d_lm_head;
        Tensor d_final_norm;
    };

    ModularGradientManager(std::uint64_t seed, int step, const Config& config,
                           const std::shared_ptr<TensorAllocator>& allocator);
    ~ModularGradientManager() = default;

    // ========================================================================
    // Micro-step management
    // ========================================================================

    /**
     * @brief Start a new micro-step
     *
     * @param stream CUDA stream
     * @param micro_step Current micro-step index (0-based)
     * @param total_steps Total number of micro-steps in gradient accumulation
     */
    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);

    /**
     * @brief End current micro-step
     *
     * Triggers all-reduce if this is the last micro-step.
     */
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    // ========================================================================
    // Full gradient access (for backward pass)
    // ========================================================================

    /**
     * @brief Get full gradient buffer for embeddings
     *
     * @param stream CUDA stream
     * @param comm NCCL communicator
     * @param[out] accumulate Set to true if gradients should be accumulated
     * @return Reference to gradient tensor
     */
    Tensor& get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate);
    Tensor& get_lm_head_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate);
    Tensor& get_final_norm_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate);
    BlockGradients& get_block_full(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate);

    // ========================================================================
    // Sharded gradient access (for optimizer)
    // ========================================================================

    TensorShard& get_embeddings_shard(cudaStream_t stream);
    TensorShard& get_lm_head_shard(cudaStream_t stream);
    TensorShard& get_final_norm_shard(cudaStream_t stream);
    BlockGradients& get_block_shard(int layer_idx, cudaStream_t stream);

    // ========================================================================
    // Notification (gradient computation complete)
    // ========================================================================

    void notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm);
    void notify_lm_head(cudaStream_t stream, NCCLCommunicator& comm);
    void notify_final_norm(cudaStream_t stream, NCCLCommunicator& comm);
    void notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm);

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] bool is_first_micro_step() const { return mIsFirstMicroStep; }
    [[nodiscard]] bool is_last_micro_step() const { return mIsLastMicroStep; }

private:
    Config mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;

    // ZeRO-1: full gradient buffers (one per layer).
    std::vector<BlockGradients> mFullGradients;

    // ZeRO-2: double-buffered full block gradients for overlapped reduce-scatter.
    std::array<BlockGradients, 2> mBlockBuffers{};
    struct BufferState {
        cudaEvent_t Event = nullptr;
        int LayerIdx = -1;
        bool NeedsAccumulation = false;
    };
    std::array<BufferState, 2> mBlockStates{};

    // Sharded gradients for optimizer update:
    // - ZeRO-1: views into mFullGradients (point at the local reduce-scatter shard region)
    // - ZeRO-2: persistent per-layer shard storage
    std::vector<BlockGradients> mShardGradients;
    NonBlockGradients mFullNonBlock;

    // Shard views for optimizer access.
    TensorShard mEmbeddingsShardView;
    TensorShard mLMHeadShardView;
    TensorShard mFinalNormShardView;

    // Micro-step state
    Philox4x32 mRng;
    int mStepCounter = -1;
    bool mIsFirstMicroStep = true;
    bool mIsLastMicroStep = false;

    // Events for synchronization (ZeRO-1: per-layer reduce-scatter completion; ZeRO-2 uses per-buffer events).
    std::vector<cudaEvent_t> mBlockReduceEvents;
    cudaEvent_t mEmbeddingsReduceEvent = nullptr;
    cudaEvent_t mLMHeadReduceEvent = nullptr;
    cudaEvent_t mFinalNormReduceEvent = nullptr;

    // Internal helpers
    void on_first_micro_step(cudaStream_t stream);
    void scatter_reduce(Tensor& tensor, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
    void scatter_reduce_block(BlockGradients& grads, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
    void sr_accumulate_layer(int layer_idx, BlockGradients& full, BlockGradients& shard, cudaStream_t stream);
    void sr_accumulate_tensor(Tensor& dst, Tensor& src, cudaStream_t stream, unsigned seed);
    void all_to_all_block(int layer_idx, BlockGradients& grads, cudaStream_t stream, cudaEvent_t signal, NCCLCommunicator& comm);
    void sr_accumulate_layer_all_to_all(int layer_idx, BlockGradients& full, BlockGradients& shard, cudaStream_t stream, NCCLCommunicator& comm);
    void sr_accumulate_tensor_shard(Tensor& dst, Tensor& src, cudaStream_t stream, bool first, float scale, int shard, unsigned seed);
    void allocate_block_gradients(BlockGradients& grads, ETensorDType dtype, int layer_idx, bool force_mamba = false);
    void allocate_block_gradients_shard(BlockGradients& grads, ETensorDType dtype, int layer_idx);
    void allocate_non_block_gradients(NonBlockGradients& grads, ETensorDType dtype);
};

// ============================================================================
// Implementation
// ============================================================================

template<typename Block>
ModularGradientManager<Block>::ModularGradientManager(
    std::uint64_t seed, int step, const Config& config,
    const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config), mAllocator(allocator), mRng(seed), mStepCounter(step) {

    if (config.offload_grads && !config.shard_gradients) {
        throw std::logic_error("Offloading gradients is not supported for unsharded gradients");
    }

    if (config.skip_allocation) {
        // LoRA mode: single shared buffer for all layers (gradients computed but discarded)
        // This saves ~8GB for a 4B model by not allocating per-layer gradient storage.
        // Force disable gradient sharding in LoRA mode since base gradients aren't used for
        // optimization anyway - the double-buffering logic would fail with only one buffer.
        mConfig.shard_gradients = false;
        
        allocate_block_gradients(mBlockBuffers[0], config.grad_dtype, /*layer_idx=*/-1);
        CUDA_CHECK(cudaEventCreate(&mBlockStates[0].Event));

        // Point all layers to the shared buffer
        mFullGradients.resize(config.num_layers);
        mShardGradients.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            mFullGradients[i] = mBlockBuffers[0];
            mShardGradients[i] = mBlockBuffers[0];
        }
    } else if (config.shard_gradients) {
        // ZeRO-2: persistent per-layer sharded gradients + 2 full buffers for backward.
        for (int i = 0; i < 2; ++i) {
            allocate_block_gradients(mBlockBuffers[i], config.grad_dtype, /*layer_idx=*/-1);
            CUDA_CHECK(cudaEventCreate(&mBlockStates[i].Event));
        }

        mShardGradients.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_gradients_shard(mShardGradients[i], config.grad_dtype, i);
        }
    } else {
        // ZeRO-1: full gradients per layer; optimizer reads a shard view after reduce-scatter.
        mFullGradients.resize(config.num_layers);
        mShardGradients.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            allocate_block_gradients(mFullGradients[i], config.grad_dtype, i);
            mShardGradients[i] = mFullGradients[i];

            if (config.num_shards > 1) {
                auto shard_tensor = [&](Tensor& t) -> Tensor {
                    return static_cast<Tensor>(shard_view(t, config.shard_idx, config.num_shards));
                };

                auto& full = mFullGradients[i];
                auto& shard = mShardGradients[i];

                if constexpr (requires { full.ln1_grads.d_weight; shard.ln1_grads.d_weight; }) shard.ln1_grads.d_weight = shard_tensor(full.ln1_grads.d_weight);
                if constexpr (requires { full.ln1.d_weight; shard.ln1.d_weight; }) shard.ln1.d_weight = shard_tensor(full.ln1.d_weight);
                if constexpr (requires { full.ln2_grads.d_weight; shard.ln2_grads.d_weight; }) shard.ln2_grads.d_weight = shard_tensor(full.ln2_grads.d_weight);
                if constexpr (requires { full.ln2.d_weight; shard.ln2.d_weight; }) shard.ln2.d_weight = shard_tensor(full.ln2.d_weight);

                if constexpr (requires { full.attention_grads.d_qkv_weight; shard.attention_grads.d_qkv_weight; }) shard.attention_grads.d_qkv_weight = shard_tensor(full.attention_grads.d_qkv_weight);
                if constexpr (requires { full.attention.d_qkv_weight; shard.attention.d_qkv_weight; }) shard.attention.d_qkv_weight = shard_tensor(full.attention.d_qkv_weight);

                if constexpr (requires { full.attention_grads.d_qkv_bias; shard.attention_grads.d_qkv_bias; }) {
                    if (full.attention_grads.d_qkv_bias.has_value()) {
                        shard.attention_grads.d_qkv_bias = shard_tensor(full.attention_grads.d_qkv_bias.value());
                    }
                }
                if constexpr (requires { full.attention.d_qkv_bias; shard.attention.d_qkv_bias; }) {
                    if (full.attention.d_qkv_bias.has_value()) {
                        shard.attention.d_qkv_bias = shard_tensor(full.attention.d_qkv_bias.value());
                    }
                }

                if constexpr (requires { full.attention_grads.d_out_weight; shard.attention_grads.d_out_weight; }) shard.attention_grads.d_out_weight = shard_tensor(full.attention_grads.d_out_weight);
                if constexpr (requires { full.attention.d_out_weight; shard.attention.d_out_weight; }) shard.attention.d_out_weight = shard_tensor(full.attention.d_out_weight);

                if constexpr (requires { full.attention_grads.d_q_norm_weight; shard.attention_grads.d_q_norm_weight; }) {
                    if (full.attention_grads.d_q_norm_weight.has_value()) {
                        shard.attention_grads.d_q_norm_weight = shard_tensor(full.attention_grads.d_q_norm_weight.value());
                    }
                }
                if constexpr (requires { full.attention_grads.d_k_norm_weight; shard.attention_grads.d_k_norm_weight; }) {
                    if (full.attention_grads.d_k_norm_weight.has_value()) {
                        shard.attention_grads.d_k_norm_weight = shard_tensor(full.attention_grads.d_k_norm_weight.value());
                    }
                }

                if constexpr (requires { full.attention.d_q_norm_weight; shard.attention.d_q_norm_weight; }) {
                    if (full.attention.d_q_norm_weight.has_value()) {
                        shard.attention.d_q_norm_weight = shard_tensor(full.attention.d_q_norm_weight.value());
                    }
                }
                if constexpr (requires { full.attention.d_k_norm_weight; shard.attention.d_k_norm_weight; }) {
                    if (full.attention.d_k_norm_weight.has_value()) {
                        shard.attention.d_k_norm_weight = shard_tensor(full.attention.d_k_norm_weight.value());
                    }
                }

                if constexpr (requires { full.router.d_gate; shard.router.d_gate; }) {
                    shard.router.d_gate = shard_tensor(full.router.d_gate);
                }
                if constexpr (requires { full.experts.d_gate_up_proj; shard.experts.d_gate_up_proj; }) {
                    shard.experts.d_gate_up_proj = shard_tensor(full.experts.d_gate_up_proj);
                }
                if constexpr (requires { full.experts.d_down_proj; shard.experts.d_down_proj; }) {
                    shard.experts.d_down_proj = shard_tensor(full.experts.d_down_proj);
                }

                if constexpr (requires { full.d_mlp_up_weight; shard.d_mlp_up_weight; }) shard.d_mlp_up_weight = shard_tensor(full.d_mlp_up_weight);
                if constexpr (requires { full.d_mlp_down_weight; shard.d_mlp_down_weight; }) shard.d_mlp_down_weight = shard_tensor(full.d_mlp_down_weight);

                if constexpr (requires { full.mamba; shard.mamba; }) {
                    if (full.mamba.has_value()) {
                        if (!shard.mamba.has_value()) {
                            shard.mamba.emplace();
                        }
                        auto& fs = *full.mamba;
                        auto& ss = *shard.mamba;
                        ss.d_in_proj_weight = shard_tensor(fs.d_in_proj_weight);
                        if (fs.d_in_proj_bias.has_value()) {
                            ss.d_in_proj_bias = shard_tensor(fs.d_in_proj_bias.value());
                        }
                        ss.d_out_proj_weight = shard_tensor(fs.d_out_proj_weight);
                        if (fs.d_out_proj_bias.has_value()) {
                            ss.d_out_proj_bias = shard_tensor(fs.d_out_proj_bias.value());
                        }
                        ss.d_conv1d_weight = shard_tensor(fs.d_conv1d_weight);
                        if (fs.d_conv1d_bias.has_value()) {
                            ss.d_conv1d_bias = shard_tensor(fs.d_conv1d_bias.value());
                        }
                        ss.d_A_log = shard_tensor(fs.d_A_log);
                        ss.d_D = shard_tensor(fs.d_D);
                        ss.d_dt_bias = shard_tensor(fs.d_dt_bias);
                        ss.d_norm_weight = shard_tensor(fs.d_norm_weight);
                    }
                }
            }
        }
    }

    // Skip non-block gradient allocation in LoRA mode - these are large (vocab_size * hidden_size)
    if (!config.skip_allocation) {
        allocate_non_block_gradients(mFullNonBlock, config.grad_dtype);
    } else {
        // LoRA mode: only allocate d_final_norm (small: hidden_size elements)
        // d_embeddings and d_lm_head are NOT needed because:
        // - encoder_backward is skipped when lora_only=true
        // - lm_head backward is skipped when lora_only=true
        // This saves ~762 MiB for a 4B model (vocab_size * hidden_size * 2 bytes)
        long C = config.hidden_size;
        mFullNonBlock.d_final_norm = mAllocator->allocate(config.grad_dtype, "d_final_norm", EAllocationType::ON_DEVICE, {C});
        // Leave d_embeddings and d_lm_head as empty tensors (Data = nullptr)
        mFullNonBlock.d_embeddings = {};
        mFullNonBlock.d_lm_head = {};
    }

    // Create shard views for non-block gradients (skip embeddings/lm_head in LoRA mode)
    if (config.num_shards > 1) {
        if (!config.skip_allocation) {
            mEmbeddingsShardView = shard_view(mFullNonBlock.d_embeddings, config.shard_idx, config.num_shards);
            mLMHeadShardView = shard_view(mFullNonBlock.d_lm_head, config.shard_idx, config.num_shards);
        }
        mFinalNormShardView = shard_view(mFullNonBlock.d_final_norm, config.shard_idx, config.num_shards);
    } else {
        if (!config.skip_allocation) {
            mEmbeddingsShardView = TensorShard(mFullNonBlock.d_embeddings);
            mLMHeadShardView = TensorShard(mFullNonBlock.d_lm_head);
        }
        mFinalNormShardView = TensorShard(mFullNonBlock.d_final_norm);
    }

    // Create synchronization events (skip in LoRA mode - no gradient sync needed for base model)
    if (!config.skip_allocation && !config.shard_gradients) {
        mBlockReduceEvents.resize(config.num_layers);
        for (int i = 0; i < config.num_layers; ++i) {
            CUDA_CHECK(cudaEventCreate(&mBlockReduceEvents[i]));
        }
    }
    CUDA_CHECK(cudaEventCreate(&mEmbeddingsReduceEvent));
    CUDA_CHECK(cudaEventCreate(&mLMHeadReduceEvent));
    CUDA_CHECK(cudaEventCreate(&mFinalNormReduceEvent));
}

template<typename Block>
void ModularGradientManager<Block>::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mIsFirstMicroStep = (micro_step == 0);
    mIsLastMicroStep = (micro_step == total_steps - 1);
    if (mIsFirstMicroStep) {
        ++mStepCounter;
        on_first_micro_step(stream);
    }
}

template<typename Block>
void ModularGradientManager<Block>::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    (void)comm;

    // ZeRO-2: finish pending accumulation for both double-buffer slots.
    if (mConfig.shard_gradients && mConfig.num_shards > 1) {
        for (auto& state : mBlockStates) {
            if (!state.NeedsAccumulation) continue;
            CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0));
            auto& full = mBlockBuffers.at(state.LayerIdx % 2);
            auto& shard = mShardGradients.at(state.LayerIdx);
            if (mConfig.use_all_to_all_reduce) {
                sr_accumulate_layer_all_to_all(state.LayerIdx, full, shard, stream, comm);
            } else {
                sr_accumulate_layer(state.LayerIdx, full, shard, stream);
            }
            state.NeedsAccumulation = false;
        }
    }

    // If this is the last micro-step, block on any outstanding comms-backed reduce-scatter ops
    // so the optimizer sees fully reduced shards.
    if (mIsLastMicroStep && mConfig.num_shards > 1) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, mEmbeddingsReduceEvent, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, mLMHeadReduceEvent, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, mFinalNormReduceEvent, 0));

        if (!mConfig.shard_gradients) {
            for (auto& ev : mBlockReduceEvents) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, ev, 0));
            }
        }
    }
}

template<typename Block>
    void ModularGradientManager<Block>::on_first_micro_step(cudaStream_t stream) {
    if (!mConfig.shard_gradients) {
        // ZeRO-1: accumulate full gradients across micro-steps in-place.
        for (auto& block : mFullGradients) {
            if constexpr (requires { block.ln1_grads.d_weight; }) fill_zero(block.ln1_grads.d_weight, stream);
            if constexpr (requires { block.ln2_grads.d_weight; }) fill_zero(block.ln2_grads.d_weight, stream);
            if constexpr (requires { block.ln1.d_weight; }) fill_zero(block.ln1.d_weight, stream);
            if constexpr (requires { block.ln2.d_weight; }) fill_zero(block.ln2.d_weight, stream);

            if constexpr (requires { block.attention_grads.d_qkv_weight; }) fill_zero(block.attention_grads.d_qkv_weight, stream);
            if constexpr (requires { block.attention_grads.d_out_weight; }) fill_zero(block.attention_grads.d_out_weight, stream);
            if constexpr (requires { block.attention_grads.d_q_norm_weight; }) {
                if (block.attention_grads.d_q_norm_weight.has_value()) {
                    fill_zero(block.attention_grads.d_q_norm_weight.value(), stream);
                }
            }
            if constexpr (requires { block.attention_grads.d_k_norm_weight; }) {
                if (block.attention_grads.d_k_norm_weight.has_value()) {
                    fill_zero(block.attention_grads.d_k_norm_weight.value(), stream);
                }
            }
            if constexpr (requires { block.attention_grads.d_qkv_bias; }) {
                if (block.attention_grads.d_qkv_bias.has_value()) {
                    fill_zero(block.attention_grads.d_qkv_bias.value(), stream);
                }
            }

            if constexpr (requires { block.attention.d_qkv_weight; }) fill_zero(block.attention.d_qkv_weight, stream);
            if constexpr (requires { block.attention.d_out_weight; }) fill_zero(block.attention.d_out_weight, stream);
            if constexpr (requires { block.attention.d_qkv_bias; }) {
                if (block.attention.d_qkv_bias.has_value()) {
                    fill_zero(block.attention.d_qkv_bias.value(), stream);
                }
            }
            if constexpr (requires { block.attention.d_q_norm_weight; }) {
                if (block.attention.d_q_norm_weight.has_value()) {
                    fill_zero(block.attention.d_q_norm_weight.value(), stream);
                }
            }
            if constexpr (requires { block.attention.d_k_norm_weight; }) {
                if (block.attention.d_k_norm_weight.has_value()) {
                    fill_zero(block.attention.d_k_norm_weight.value(), stream);
                }
            }

            if constexpr (requires { block.d_mlp_up_weight; }) fill_zero(block.d_mlp_up_weight, stream);
            if constexpr (requires { block.d_mlp_down_weight; }) fill_zero(block.d_mlp_down_weight, stream);

            if constexpr (requires { block.mamba; }) {
                if (block.mamba.has_value()) {
                    auto& mg = *block.mamba;
                    fill_zero(mg.d_in_proj_weight, stream);
                    if (mg.d_in_proj_bias.has_value()) fill_zero(mg.d_in_proj_bias.value(), stream);
                    fill_zero(mg.d_out_proj_weight, stream);
                    if (mg.d_out_proj_bias.has_value()) fill_zero(mg.d_out_proj_bias.value(), stream);
                    fill_zero(mg.d_conv1d_weight, stream);
                    if (mg.d_conv1d_bias.has_value()) fill_zero(mg.d_conv1d_bias.value(), stream);
                    fill_zero(mg.d_A_log, stream);
                    fill_zero(mg.d_D, stream);
                    fill_zero(mg.d_dt_bias, stream);
                    fill_zero(mg.d_norm_weight, stream);
                }
            }

            if constexpr (requires { block.router.d_gate; }) fill_zero(block.router.d_gate, stream);
            if constexpr (requires { block.experts.d_gate_up_proj; }) fill_zero(block.experts.d_gate_up_proj, stream);
            if constexpr (requires { block.experts.d_down_proj; }) fill_zero(block.experts.d_down_proj, stream);
        }
    } else {
        // ZeRO-2: reset buffer bookkeeping; shard buffers are overwritten on the first micro-step.
        for (auto& st : mBlockStates) {
            st.LayerIdx = -1;
            st.NeedsAccumulation = false;
        }
    }

    // Non-block gradients are accumulated in-place across micro-steps.
    // Skip embeddings/lm_head in LoRA mode (not allocated)
    if (mFullNonBlock.d_embeddings.Data) {
        fill_zero(mFullNonBlock.d_embeddings, stream);
    }
    if (mFullNonBlock.d_lm_head.Data && mFullNonBlock.d_lm_head.Data != mFullNonBlock.d_embeddings.Data) {
        fill_zero(mFullNonBlock.d_lm_head, stream);
    }
    fill_zero(mFullNonBlock.d_final_norm, stream);
}

template<typename Block>
Tensor& ModularGradientManager<Block>::get_embeddings_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.d_embeddings;
}

template<typename Block>
Tensor& ModularGradientManager<Block>::get_lm_head_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.d_lm_head;
}

template<typename Block>
Tensor& ModularGradientManager<Block>::get_final_norm_full(cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    accumulate = !mIsFirstMicroStep;
    return mFullNonBlock.d_final_norm;
}

template<typename Block>
typename Block::Gradients& ModularGradientManager<Block>::get_block_full(
    int layer_idx, cudaStream_t stream, NCCLCommunicator& comm, bool& accumulate) {
    if (!mConfig.shard_gradients) {
        (void)comm;
        accumulate = !mIsFirstMicroStep;
        return mFullGradients[layer_idx];
    }

    // ZeRO-2: overwrite gradients into a double-buffer slot and accumulate reduced shards
    // into persistent storage.
    accumulate = false;

    auto& state = mBlockStates.at(layer_idx % 2);
    auto& dw = mBlockBuffers.at(layer_idx % 2);

    if (state.Event) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0));
    }

    if (state.NeedsAccumulation) {
        auto& sw = mShardGradients.at(state.LayerIdx);
        if (mConfig.use_all_to_all_reduce) {
            sr_accumulate_layer_all_to_all(state.LayerIdx, dw, sw, stream, comm);
        } else {
            sr_accumulate_layer(state.LayerIdx, dw, sw, stream);
        }
        state.NeedsAccumulation = false;
    }

    state.LayerIdx = layer_idx;

    // Reset additive gradients (LN weights and optional bias) before computing them again.
    if constexpr (requires { dw.ln1_grads.d_weight; }) fill_zero(dw.ln1_grads.d_weight, stream);
    if constexpr (requires { dw.ln1.d_weight; }) fill_zero(dw.ln1.d_weight, stream);
    if constexpr (requires { dw.ln2_grads.d_weight; }) fill_zero(dw.ln2_grads.d_weight, stream);
    if constexpr (requires { dw.ln2.d_weight; }) fill_zero(dw.ln2.d_weight, stream);

    if constexpr (requires { dw.attention_grads.d_qkv_bias; }) {
        if (dw.attention_grads.d_qkv_bias.has_value()) {
            fill_zero(dw.attention_grads.d_qkv_bias.value(), stream);
        }
    }
    if constexpr (requires { dw.attention.d_qkv_bias; }) {
        if (dw.attention.d_qkv_bias.has_value()) {
            fill_zero(dw.attention.d_qkv_bias.value(), stream);
        }
    }

    return dw;
}

template<typename Block>
TensorShard& ModularGradientManager<Block>::get_embeddings_shard(cudaStream_t stream) {
    (void)stream;
    return mEmbeddingsShardView;
}

template<typename Block>
TensorShard& ModularGradientManager<Block>::get_lm_head_shard(cudaStream_t stream) {
    (void)stream;
    return mLMHeadShardView;
}

template<typename Block>
TensorShard& ModularGradientManager<Block>::get_final_norm_shard(cudaStream_t stream) {
    (void)stream;
    return mFinalNormShardView;
}

template<typename Block>
typename Block::Gradients& ModularGradientManager<Block>::get_block_shard(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mShardGradients[layer_idx];
}

template<typename Block>
void ModularGradientManager<Block>::notify_embeddings(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mIsLastMicroStep) return;
    // Skip in LoRA mode (embeddings not allocated)
    if (!mFullNonBlock.d_embeddings.Data) return;
    scatter_reduce(mFullNonBlock.d_embeddings, stream, mEmbeddingsReduceEvent, comm);
}

template<typename Block>
void ModularGradientManager<Block>::notify_lm_head(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mIsLastMicroStep) return;
    // Skip in LoRA mode (lm_head not allocated)
    if (!mFullNonBlock.d_lm_head.Data) return;
    if (mFullNonBlock.d_lm_head.Data == mFullNonBlock.d_embeddings.Data) {
        CUDA_CHECK(cudaEventRecord(mLMHeadReduceEvent, stream));
        return;
    }
    scatter_reduce(mFullNonBlock.d_lm_head, stream, mLMHeadReduceEvent, comm);
}

template<typename Block>
void ModularGradientManager<Block>::notify_final_norm(cudaStream_t stream, NCCLCommunicator& comm) {
    if (!mIsLastMicroStep) return;
    scatter_reduce(mFullNonBlock.d_final_norm, stream, mFinalNormReduceEvent, comm);
}

template<typename Block>
void ModularGradientManager<Block>::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if (mConfig.num_shards == 1) return;
    if (mConfig.skip_allocation) return;  // LoRA mode: base gradients not used, skip reduction

    if (!mConfig.shard_gradients) {
        // ZeRO-1: reduce-scatter once per optimizer step (on the last micro-step).
        if (!mIsLastMicroStep) return;
        scatter_reduce_block(mFullGradients[layer_idx], stream, mBlockReduceEvents[layer_idx], comm);
        return;
    }

    // ZeRO-2: reduce-scatter the current buffer on every micro-step, then accumulate into shard storage.
    auto& state = mBlockStates.at(layer_idx % 2);
    if (state.LayerIdx != layer_idx) {
        throw std::logic_error("ModularGradientManager::notify_block called with wrong layer index");
    }
    if (state.NeedsAccumulation) {
        throw std::logic_error("ModularGradientManager::notify_block called before accumulation has finished");
    }

    auto& dw = mBlockBuffers.at(layer_idx % 2);
    if (mConfig.use_all_to_all_reduce) {
        all_to_all_block(layer_idx, dw, stream, state.Event, comm);
    } else {
        scatter_reduce_block(dw, stream, state.Event, comm);
    }
    state.NeedsAccumulation = true;
}

template<typename Block>
void ModularGradientManager<Block>::scatter_reduce(Tensor& tensor, cudaStream_t stream,
                                                    cudaEvent_t signal, NCCLCommunicator& comm) {
    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(signal, stream));
        return;
    }

    if (tensor.Data && tensor.nelem() > 0) {
        comm.begin_transaction(stream);
        comm.schedule_reduce_scatter(tensor);
        comm.execute_transaction(signal);
        return;
    }

    CUDA_CHECK(cudaEventRecord(signal, stream));
}

template<typename Block>
void ModularGradientManager<Block>::scatter_reduce_block(BlockGradients& grads, cudaStream_t stream,
                                                         cudaEvent_t signal, NCCLCommunicator& comm) {
    if (mConfig.num_shards == 1) {
        CUDA_CHECK(cudaEventRecord(signal, stream));
        return;
    }

    comm.begin_transaction(stream);

    auto maybe_schedule = [&](Tensor& t) {
        if (t.Data && t.nelem() > 0) {
            comm.schedule_reduce_scatter(t);
        }
    };

    if constexpr (requires { grads.ln1_grads.d_weight; }) maybe_schedule(grads.ln1_grads.d_weight);
    if constexpr (requires { grads.ln1.d_weight; }) maybe_schedule(grads.ln1.d_weight);
    if constexpr (requires { grads.ln2_grads.d_weight; }) maybe_schedule(grads.ln2_grads.d_weight);
    if constexpr (requires { grads.ln2.d_weight; }) maybe_schedule(grads.ln2.d_weight);

    if constexpr (requires { grads.attention_grads.d_qkv_weight; }) maybe_schedule(grads.attention_grads.d_qkv_weight);
    if constexpr (requires { grads.attention.d_qkv_weight; }) maybe_schedule(grads.attention.d_qkv_weight);

    if constexpr (requires { grads.attention_grads.d_qkv_bias; }) {
        if (grads.attention_grads.d_qkv_bias.has_value()) {
            maybe_schedule(grads.attention_grads.d_qkv_bias.value());
        }
    }
    if constexpr (requires { grads.attention.d_qkv_bias; }) {
        if (grads.attention.d_qkv_bias.has_value()) {
            maybe_schedule(grads.attention.d_qkv_bias.value());
        }
    }

    if constexpr (requires { grads.attention_grads.d_out_weight; }) maybe_schedule(grads.attention_grads.d_out_weight);
    if constexpr (requires { grads.attention.d_out_weight; }) maybe_schedule(grads.attention.d_out_weight);
    if constexpr (requires { grads.attention_grads.d_q_norm_weight; }) {
        if (grads.attention_grads.d_q_norm_weight.has_value()) {
            maybe_schedule(grads.attention_grads.d_q_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention_grads.d_k_norm_weight; }) {
        if (grads.attention_grads.d_k_norm_weight.has_value()) {
            maybe_schedule(grads.attention_grads.d_k_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention.d_q_norm_weight; }) {
        if (grads.attention.d_q_norm_weight.has_value()) {
            maybe_schedule(grads.attention.d_q_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention.d_k_norm_weight; }) {
        if (grads.attention.d_k_norm_weight.has_value()) {
            maybe_schedule(grads.attention.d_k_norm_weight.value());
        }
    }

    if constexpr (requires { grads.d_mlp_up_weight; }) maybe_schedule(grads.d_mlp_up_weight);
    if constexpr (requires { grads.d_mlp_down_weight; }) maybe_schedule(grads.d_mlp_down_weight);

    if constexpr (requires { grads.mamba; }) {
        if (grads.mamba.has_value()) {
            auto& mg = *grads.mamba;
            maybe_schedule(mg.d_in_proj_weight);
            if (mg.d_in_proj_bias.has_value()) maybe_schedule(mg.d_in_proj_bias.value());
            maybe_schedule(mg.d_out_proj_weight);
            if (mg.d_out_proj_bias.has_value()) maybe_schedule(mg.d_out_proj_bias.value());
            maybe_schedule(mg.d_conv1d_weight);
            if (mg.d_conv1d_bias.has_value()) maybe_schedule(mg.d_conv1d_bias.value());
            maybe_schedule(mg.d_A_log);
            maybe_schedule(mg.d_D);
            maybe_schedule(mg.d_dt_bias);
            maybe_schedule(mg.d_norm_weight);
        }
    }

    if constexpr (requires { grads.router.d_gate; }) maybe_schedule(grads.router.d_gate);
    if constexpr (requires { grads.experts.d_gate_up_proj; }) maybe_schedule(grads.experts.d_gate_up_proj);
    if constexpr (requires { grads.experts.d_down_proj; }) maybe_schedule(grads.experts.d_down_proj);
    if constexpr (requires { grads.shared_expert; }) {
        if (grads.shared_expert.has_value()) {
            maybe_schedule(grads.shared_expert->d_gate_proj);
            maybe_schedule(grads.shared_expert->d_down_proj);
        }
    }

    comm.execute_transaction(signal);
}

template<typename Block>
void ModularGradientManager<Block>::all_to_all_block(int layer_idx, BlockGradients& grads, cudaStream_t stream,
                                                     cudaEvent_t signal, NCCLCommunicator& comm) {
    // Legacy ZeRO-2 all-to-all reducer:
    // - accumulate this rank's shard locally (since the all-to-all overwrites in-place)
    // - perform destructive all-to-all exchange of the full gradient buffer
    // - the remaining shards are reduced into persistent storage in sr_accumulate_layer_all_to_all()
    auto& shard = mShardGradients.at(layer_idx);
    const int rank = comm.rank();

    auto rng_1 = mRng.generate(2 * mStepCounter + 0, layer_idx);
    auto rng_2 = mRng.generate(2 * mStepCounter + 1, layer_idx);

    auto acc_local = [&](Tensor& dst, Tensor& src, unsigned seed) {
        sr_accumulate_tensor_shard(dst, src, stream, /*first=*/mIsFirstMicroStep, /*scale=*/1.f, /*shard=*/rank, seed);
    };

    if constexpr (requires { shard.ln1_grads.d_weight; grads.ln1_grads.d_weight; }) acc_local(shard.ln1_grads.d_weight, grads.ln1_grads.d_weight, rng_1[0]);
    if constexpr (requires { shard.ln1.d_weight; grads.ln1.d_weight; }) acc_local(shard.ln1.d_weight, grads.ln1.d_weight, rng_1[0]);
    if constexpr (requires { shard.ln2_grads.d_weight; grads.ln2_grads.d_weight; }) acc_local(shard.ln2_grads.d_weight, grads.ln2_grads.d_weight, rng_1[1]);
    if constexpr (requires { shard.ln2.d_weight; grads.ln2.d_weight; }) acc_local(shard.ln2.d_weight, grads.ln2.d_weight, rng_1[1]);

    if constexpr (requires { shard.attention_grads.d_qkv_weight; grads.attention_grads.d_qkv_weight; }) acc_local(shard.attention_grads.d_qkv_weight, grads.attention_grads.d_qkv_weight, rng_1[2]);
    if constexpr (requires { shard.attention.d_qkv_weight; grads.attention.d_qkv_weight; }) acc_local(shard.attention.d_qkv_weight, grads.attention.d_qkv_weight, rng_1[2]);
    if constexpr (requires { shard.attention_grads.d_out_weight; grads.attention_grads.d_out_weight; }) acc_local(shard.attention_grads.d_out_weight, grads.attention_grads.d_out_weight, rng_1[3]);
    if constexpr (requires { shard.attention.d_out_weight; grads.attention.d_out_weight; }) acc_local(shard.attention.d_out_weight, grads.attention.d_out_weight, rng_1[3]);
    if constexpr (requires { shard.attention_grads.d_q_norm_weight; grads.attention_grads.d_q_norm_weight; }) {
        if (shard.attention_grads.d_q_norm_weight.has_value() && grads.attention_grads.d_q_norm_weight.has_value()) {
            acc_local(shard.attention_grads.d_q_norm_weight.value(), grads.attention_grads.d_q_norm_weight.value(), rng_2[2]);
        }
    }
    if constexpr (requires { shard.attention_grads.d_k_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
        if (shard.attention_grads.d_k_norm_weight.has_value() && grads.attention_grads.d_k_norm_weight.has_value()) {
            acc_local(shard.attention_grads.d_k_norm_weight.value(), grads.attention_grads.d_k_norm_weight.value(), rng_2[3]);
        }
    }
    if constexpr (requires { shard.attention.d_q_norm_weight; grads.attention.d_q_norm_weight; }) {
        if (shard.attention.d_q_norm_weight.has_value() && grads.attention.d_q_norm_weight.has_value()) {
            acc_local(shard.attention.d_q_norm_weight.value(), grads.attention.d_q_norm_weight.value(), rng_2[2] ^ 0x3c6ef372u);
        }
    }
    if constexpr (requires { shard.attention.d_k_norm_weight; grads.attention.d_k_norm_weight; }) {
        if (shard.attention.d_k_norm_weight.has_value() && grads.attention.d_k_norm_weight.has_value()) {
            acc_local(shard.attention.d_k_norm_weight.value(), grads.attention.d_k_norm_weight.value(), rng_2[3] ^ 0xbb67ae85u);
        }
    }

    if constexpr (requires { shard.d_mlp_up_weight; grads.d_mlp_up_weight; }) acc_local(shard.d_mlp_up_weight, grads.d_mlp_up_weight, rng_2[0]);
    if constexpr (requires { shard.d_mlp_down_weight; grads.d_mlp_down_weight; }) acc_local(shard.d_mlp_down_weight, grads.d_mlp_down_weight, rng_2[1]);

    if constexpr (requires { shard.mamba; grads.mamba; }) {
        if (shard.mamba.has_value() && grads.mamba.has_value()) {
            auto& ss = *shard.mamba;
            auto& gs = *grads.mamba;
            acc_local(ss.d_in_proj_weight, gs.d_in_proj_weight, rng_1[2] ^ 0x1f83d9abu);
            if (ss.d_in_proj_bias.has_value() && gs.d_in_proj_bias.has_value()) {
                acc_local(ss.d_in_proj_bias.value(), gs.d_in_proj_bias.value(), rng_1[2] ^ 0x85ebca6bu);
            }
            acc_local(ss.d_out_proj_weight, gs.d_out_proj_weight, rng_1[3] ^ 0x9e3779b9u);
            if (ss.d_out_proj_bias.has_value() && gs.d_out_proj_bias.has_value()) {
                acc_local(ss.d_out_proj_bias.value(), gs.d_out_proj_bias.value(), rng_1[3] ^ 0xc2b2ae35u);
            }
            acc_local(ss.d_conv1d_weight, gs.d_conv1d_weight, rng_2[1] ^ 0x27d4eb2fu);
            if (ss.d_conv1d_bias.has_value() && gs.d_conv1d_bias.has_value()) {
                acc_local(ss.d_conv1d_bias.value(), gs.d_conv1d_bias.value(), rng_2[1] ^ 0x165667b1u);
            }
            acc_local(ss.d_A_log, gs.d_A_log, rng_2[2] ^ 0xd3a2646cu);
            acc_local(ss.d_D, gs.d_D, rng_2[2] ^ 0xfd7046c5u);
            acc_local(ss.d_dt_bias, gs.d_dt_bias, rng_2[3] ^ 0x5a48dfe7u);
            acc_local(ss.d_norm_weight, gs.d_norm_weight, rng_2[3] ^ 0x6a09e667u);
        }
    }

    if constexpr (requires { shard.router.d_gate; grads.router.d_gate; }) acc_local(shard.router.d_gate, grads.router.d_gate, rng_2[0] ^ 0xa54ff53au);
    if constexpr (requires { shard.experts.d_gate_up_proj; grads.experts.d_gate_up_proj; }) acc_local(shard.experts.d_gate_up_proj, grads.experts.d_gate_up_proj, rng_2[1] ^ 0x510e527fu);
    if constexpr (requires { shard.experts.d_down_proj; grads.experts.d_down_proj; }) acc_local(shard.experts.d_down_proj, grads.experts.d_down_proj, rng_2[2] ^ 0x9b05688cu);
    if constexpr (requires { shard.shared_expert; grads.shared_expert; }) {
        if (shard.shared_expert.has_value() && grads.shared_expert.has_value()) {
            acc_local(shard.shared_expert->d_gate_proj, grads.shared_expert->d_gate_proj, rng_2[2] ^ 0x3c6ef372u);
            acc_local(shard.shared_expert->d_down_proj, grads.shared_expert->d_down_proj, rng_2[3] ^ 0xbb67ae85u);
        }
    }

    if constexpr (requires { shard.attention_grads.d_qkv_bias; grads.attention_grads.d_qkv_bias; }) {
        if (shard.attention_grads.d_qkv_bias.has_value() && grads.attention_grads.d_qkv_bias.has_value()) {
            acc_local(shard.attention_grads.d_qkv_bias.value(), grads.attention_grads.d_qkv_bias.value(), rng_2[2]);
        }
    }
    if constexpr (requires { shard.attention.d_qkv_bias; grads.attention.d_qkv_bias; }) {
        if (shard.attention.d_qkv_bias.has_value() && grads.attention.d_qkv_bias.has_value()) {
            acc_local(shard.attention.d_qkv_bias.value(), grads.attention.d_qkv_bias.value(), rng_2[2]);
        }
    }

    // Ensure local accumulation completes before communication begins.
    CUDA_CHECK(cudaEventRecord(signal, stream));

    comm.begin_transaction(signal);

    auto maybe_schedule = [&](Tensor& t) {
        if (t.Data && t.nelem() > 0) {
            comm.schedule_destructive_all_to_all(t);
        }
    };

    if constexpr (requires { grads.ln1_grads.d_weight; }) maybe_schedule(grads.ln1_grads.d_weight);
    if constexpr (requires { grads.ln1.d_weight; }) maybe_schedule(grads.ln1.d_weight);
    if constexpr (requires { grads.ln2_grads.d_weight; }) maybe_schedule(grads.ln2_grads.d_weight);
    if constexpr (requires { grads.ln2.d_weight; }) maybe_schedule(grads.ln2.d_weight);

    if constexpr (requires { grads.attention_grads.d_qkv_weight; }) maybe_schedule(grads.attention_grads.d_qkv_weight);
    if constexpr (requires { grads.attention.d_qkv_weight; }) maybe_schedule(grads.attention.d_qkv_weight);
    if constexpr (requires { grads.attention_grads.d_out_weight; }) maybe_schedule(grads.attention_grads.d_out_weight);
    if constexpr (requires { grads.attention.d_out_weight; }) maybe_schedule(grads.attention.d_out_weight);
    if constexpr (requires { grads.attention_grads.d_q_norm_weight; }) {
        if (grads.attention_grads.d_q_norm_weight.has_value()) {
            maybe_schedule(grads.attention_grads.d_q_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention_grads.d_k_norm_weight; }) {
        if (grads.attention_grads.d_k_norm_weight.has_value()) {
            maybe_schedule(grads.attention_grads.d_k_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention.d_q_norm_weight; }) {
        if (grads.attention.d_q_norm_weight.has_value()) {
            maybe_schedule(grads.attention.d_q_norm_weight.value());
        }
    }
    if constexpr (requires { grads.attention.d_k_norm_weight; }) {
        if (grads.attention.d_k_norm_weight.has_value()) {
            maybe_schedule(grads.attention.d_k_norm_weight.value());
        }
    }

    if constexpr (requires { grads.attention_grads.d_qkv_bias; }) {
        if (grads.attention_grads.d_qkv_bias.has_value()) {
            maybe_schedule(grads.attention_grads.d_qkv_bias.value());
        }
    }
    if constexpr (requires { grads.attention.d_qkv_bias; }) {
        if (grads.attention.d_qkv_bias.has_value()) {
            maybe_schedule(grads.attention.d_qkv_bias.value());
        }
    }

    if constexpr (requires { grads.d_mlp_up_weight; }) maybe_schedule(grads.d_mlp_up_weight);
    if constexpr (requires { grads.d_mlp_down_weight; }) maybe_schedule(grads.d_mlp_down_weight);

    if constexpr (requires { grads.mamba; }) {
        if (grads.mamba.has_value()) {
            auto& mg = *grads.mamba;
            maybe_schedule(mg.d_in_proj_weight);
            if (mg.d_in_proj_bias.has_value()) maybe_schedule(mg.d_in_proj_bias.value());
            maybe_schedule(mg.d_out_proj_weight);
            if (mg.d_out_proj_bias.has_value()) maybe_schedule(mg.d_out_proj_bias.value());
            maybe_schedule(mg.d_conv1d_weight);
            if (mg.d_conv1d_bias.has_value()) maybe_schedule(mg.d_conv1d_bias.value());
            maybe_schedule(mg.d_A_log);
            maybe_schedule(mg.d_D);
            maybe_schedule(mg.d_dt_bias);
            maybe_schedule(mg.d_norm_weight);
        }
    }

    if constexpr (requires { grads.router.d_gate; }) maybe_schedule(grads.router.d_gate);
    if constexpr (requires { grads.experts.d_gate_up_proj; }) maybe_schedule(grads.experts.d_gate_up_proj);
    if constexpr (requires { grads.experts.d_down_proj; }) maybe_schedule(grads.experts.d_down_proj);
    if constexpr (requires { grads.shared_expert; }) {
        if (grads.shared_expert.has_value()) {
            maybe_schedule(grads.shared_expert->d_gate_proj);
            maybe_schedule(grads.shared_expert->d_down_proj);
        }
    }

    comm.execute_transaction(signal);
}

template<typename Block>
void ModularGradientManager<Block>::sr_accumulate_tensor(Tensor& dst, Tensor& src, cudaStream_t stream, unsigned seed) {
    Tensor local_slice = shard_view(src, mConfig.shard_idx, mConfig.num_shards);
    if (mIsFirstMicroStep) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        vector_add_sr(dst, dst, local_slice, 1.f, (long)local_slice.nelem(), seed, stream);
    }
}

template<typename Block>
void ModularGradientManager<Block>::sr_accumulate_tensor_shard(
    Tensor& dst, Tensor& src, cudaStream_t stream, bool first, float scale, int shard, unsigned seed) {
    Tensor local_slice = shard_view(src, shard, mConfig.num_shards);
    if (first) {
        CUDA_CHECK(cudaMemcpyAsync(dst.Data, local_slice.Data, local_slice.bytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        vector_add_sr(dst, dst, local_slice, scale, (long)local_slice.nelem(), seed, stream);
    }
}

template<typename Block>
void ModularGradientManager<Block>::sr_accumulate_layer(int layer_idx, BlockGradients& full, BlockGradients& shard, cudaStream_t stream) {
    auto rng_1 = mRng.generate(2 * mStepCounter + 0, layer_idx);
    auto rng_2 = mRng.generate(2 * mStepCounter + 1, layer_idx);

    if constexpr (requires { shard.ln1_grads.d_weight; full.ln1_grads.d_weight; }) {
        sr_accumulate_tensor(shard.ln1_grads.d_weight, full.ln1_grads.d_weight, stream, rng_1[0]);
    }
    if constexpr (requires { shard.ln1.d_weight; full.ln1.d_weight; }) {
        sr_accumulate_tensor(shard.ln1.d_weight, full.ln1.d_weight, stream, rng_1[0]);
    }
    if constexpr (requires { shard.ln2_grads.d_weight; full.ln2_grads.d_weight; }) {
        sr_accumulate_tensor(shard.ln2_grads.d_weight, full.ln2_grads.d_weight, stream, rng_1[1]);
    }
    if constexpr (requires { shard.ln2.d_weight; full.ln2.d_weight; }) {
        sr_accumulate_tensor(shard.ln2.d_weight, full.ln2.d_weight, stream, rng_1[1]);
    }

    if constexpr (requires { shard.attention_grads.d_qkv_weight; full.attention_grads.d_qkv_weight; }) {
        sr_accumulate_tensor(shard.attention_grads.d_qkv_weight, full.attention_grads.d_qkv_weight, stream, rng_2[0]);
    }
    if constexpr (requires { shard.attention.d_qkv_weight; full.attention.d_qkv_weight; }) {
        sr_accumulate_tensor(shard.attention.d_qkv_weight, full.attention.d_qkv_weight, stream, rng_2[0]);
    }

    if constexpr (requires { shard.attention_grads.d_qkv_bias; full.attention_grads.d_qkv_bias; }) {
        if (shard.attention_grads.d_qkv_bias.has_value() && full.attention_grads.d_qkv_bias.has_value()) {
            sr_accumulate_tensor(shard.attention_grads.d_qkv_bias.value(), full.attention_grads.d_qkv_bias.value(), stream, rng_2[2]);
        }
    }
    if constexpr (requires { shard.attention.d_qkv_bias; full.attention.d_qkv_bias; }) {
        if (shard.attention.d_qkv_bias.has_value() && full.attention.d_qkv_bias.has_value()) {
            sr_accumulate_tensor(shard.attention.d_qkv_bias.value(), full.attention.d_qkv_bias.value(), stream, rng_2[2]);
        }
    }

    if constexpr (requires { shard.attention_grads.d_out_weight; full.attention_grads.d_out_weight; }) {
        sr_accumulate_tensor(shard.attention_grads.d_out_weight, full.attention_grads.d_out_weight, stream, rng_2[1]);
    }
    if constexpr (requires { shard.attention.d_out_weight; full.attention.d_out_weight; }) {
        sr_accumulate_tensor(shard.attention.d_out_weight, full.attention.d_out_weight, stream, rng_2[1]);
    }
    if constexpr (requires { shard.attention_grads.d_q_norm_weight; full.attention_grads.d_q_norm_weight; }) {
        if (shard.attention_grads.d_q_norm_weight.has_value() && full.attention_grads.d_q_norm_weight.has_value()) {
            sr_accumulate_tensor(shard.attention_grads.d_q_norm_weight.value(), full.attention_grads.d_q_norm_weight.value(), stream, rng_2[3]);
        }
    }
    if constexpr (requires { shard.attention_grads.d_k_norm_weight; full.attention_grads.d_k_norm_weight; }) {
        if (shard.attention_grads.d_k_norm_weight.has_value() && full.attention_grads.d_k_norm_weight.has_value()) {
            sr_accumulate_tensor(shard.attention_grads.d_k_norm_weight.value(), full.attention_grads.d_k_norm_weight.value(), stream, rng_2[3] ^ 0x9e3779b9u);
        }
    }
    if constexpr (requires { shard.attention.d_q_norm_weight; full.attention.d_q_norm_weight; }) {
        if (shard.attention.d_q_norm_weight.has_value() && full.attention.d_q_norm_weight.has_value()) {
            sr_accumulate_tensor(shard.attention.d_q_norm_weight.value(), full.attention.d_q_norm_weight.value(), stream, rng_2[3] ^ 0x3c6ef372u);
        }
    }
    if constexpr (requires { shard.attention.d_k_norm_weight; full.attention.d_k_norm_weight; }) {
        if (shard.attention.d_k_norm_weight.has_value() && full.attention.d_k_norm_weight.has_value()) {
            sr_accumulate_tensor(shard.attention.d_k_norm_weight.value(), full.attention.d_k_norm_weight.value(), stream, rng_2[3] ^ 0xbb67ae85u);
        }
    }

    if constexpr (requires { shard.d_mlp_up_weight; full.d_mlp_up_weight; }) {
        sr_accumulate_tensor(shard.d_mlp_up_weight, full.d_mlp_up_weight, stream, rng_1[2]);
    }
    if constexpr (requires { shard.d_mlp_down_weight; full.d_mlp_down_weight; }) {
        sr_accumulate_tensor(shard.d_mlp_down_weight, full.d_mlp_down_weight, stream, rng_1[3]);
    }

    if constexpr (requires { shard.mamba; full.mamba; }) {
        if (shard.mamba.has_value() && full.mamba.has_value()) {
            auto& ss = *shard.mamba;
            auto& fs = *full.mamba;
            sr_accumulate_tensor(ss.d_in_proj_weight, fs.d_in_proj_weight, stream, rng_1[2] ^ 0x1f83d9abu);
            if (ss.d_in_proj_bias.has_value() && fs.d_in_proj_bias.has_value()) {
                sr_accumulate_tensor(ss.d_in_proj_bias.value(), fs.d_in_proj_bias.value(), stream, rng_1[2] ^ 0x85ebca6bu);
            }
            sr_accumulate_tensor(ss.d_out_proj_weight, fs.d_out_proj_weight, stream, rng_1[3] ^ 0x9e3779b9u);
            if (ss.d_out_proj_bias.has_value() && fs.d_out_proj_bias.has_value()) {
                sr_accumulate_tensor(ss.d_out_proj_bias.value(), fs.d_out_proj_bias.value(), stream, rng_1[3] ^ 0xc2b2ae35u);
            }
            sr_accumulate_tensor(ss.d_conv1d_weight, fs.d_conv1d_weight, stream, rng_2[1] ^ 0x27d4eb2fu);
            if (ss.d_conv1d_bias.has_value() && fs.d_conv1d_bias.has_value()) {
                sr_accumulate_tensor(ss.d_conv1d_bias.value(), fs.d_conv1d_bias.value(), stream, rng_2[1] ^ 0x165667b1u);
            }
            sr_accumulate_tensor(ss.d_A_log, fs.d_A_log, stream, rng_2[2] ^ 0xd3a2646cu);
            sr_accumulate_tensor(ss.d_D, fs.d_D, stream, rng_2[2] ^ 0xfd7046c5u);
            sr_accumulate_tensor(ss.d_dt_bias, fs.d_dt_bias, stream, rng_2[3] ^ 0x5a48dfe7u);
            sr_accumulate_tensor(ss.d_norm_weight, fs.d_norm_weight, stream, rng_2[3] ^ 0x6a09e667u);
        }
    }

    if constexpr (requires { shard.router.d_gate; full.router.d_gate; }) {
        sr_accumulate_tensor(shard.router.d_gate, full.router.d_gate, stream, rng_2[0] ^ 0xa54ff53au);
    }
    if constexpr (requires { shard.experts.d_gate_up_proj; full.experts.d_gate_up_proj; }) {
        sr_accumulate_tensor(shard.experts.d_gate_up_proj, full.experts.d_gate_up_proj, stream, rng_2[1] ^ 0x510e527fu);
    }
    if constexpr (requires { shard.experts.d_down_proj; full.experts.d_down_proj; }) {
        sr_accumulate_tensor(shard.experts.d_down_proj, full.experts.d_down_proj, stream, rng_2[2] ^ 0x9b05688cu);
    }
}

template<typename Block>
void ModularGradientManager<Block>::sr_accumulate_layer_all_to_all(
    int layer_idx, BlockGradients& full, BlockGradients& shard, cudaStream_t stream, NCCLCommunicator& comm) {
    const int rank = comm.rank();
    const int world = comm.world_size();
    if (world != mConfig.num_shards || rank != mConfig.shard_idx) {
        throw std::logic_error("ModularGradientManager: inconsistent communicator rank/world for all-to-all reduction");
    }

    float scale = 1.f;
    if (mIsLastMicroStep) {
        scale = 1.f / static_cast<float>(world);
    }
    const int skip = (rank + world - 1) % world;

    auto rng_1 = mRng.generate(2 * mStepCounter + 0, layer_idx);
    auto rng_2 = mRng.generate(2 * mStepCounter + 1, layer_idx + 12345);

    auto reduce = [&](Tensor& dst, Tensor& src, unsigned seed) {
        vector_reduce_sr(dst, src, scale, world, skip, (long)dst.nelem(), /*accumulate=*/true, seed, stream);
    };

    if constexpr (requires { shard.ln1_grads.d_weight; full.ln1_grads.d_weight; }) reduce(shard.ln1_grads.d_weight, full.ln1_grads.d_weight, rng_1[0]);
    if constexpr (requires { shard.ln1.d_weight; full.ln1.d_weight; }) reduce(shard.ln1.d_weight, full.ln1.d_weight, rng_1[0]);
    if constexpr (requires { shard.ln2_grads.d_weight; full.ln2_grads.d_weight; }) reduce(shard.ln2_grads.d_weight, full.ln2_grads.d_weight, rng_1[1]);
    if constexpr (requires { shard.ln2.d_weight; full.ln2.d_weight; }) reduce(shard.ln2.d_weight, full.ln2.d_weight, rng_1[1]);

    if constexpr (requires { shard.d_mlp_up_weight; full.d_mlp_up_weight; }) reduce(shard.d_mlp_up_weight, full.d_mlp_up_weight, rng_1[2]);
    if constexpr (requires { shard.d_mlp_down_weight; full.d_mlp_down_weight; }) reduce(shard.d_mlp_down_weight, full.d_mlp_down_weight, rng_1[3]);

    if constexpr (requires { shard.mamba; full.mamba; }) {
        if (shard.mamba.has_value() && full.mamba.has_value()) {
            auto& ss = *shard.mamba;
            auto& fs = *full.mamba;
            reduce(ss.d_in_proj_weight, fs.d_in_proj_weight, rng_1[2] ^ 0x1f83d9abu);
            if (ss.d_in_proj_bias.has_value() && fs.d_in_proj_bias.has_value()) {
                reduce(ss.d_in_proj_bias.value(), fs.d_in_proj_bias.value(), rng_1[2] ^ 0x85ebca6bu);
            }
            reduce(ss.d_out_proj_weight, fs.d_out_proj_weight, rng_1[3] ^ 0x9e3779b9u);
            if (ss.d_out_proj_bias.has_value() && fs.d_out_proj_bias.has_value()) {
                reduce(ss.d_out_proj_bias.value(), fs.d_out_proj_bias.value(), rng_1[3] ^ 0xc2b2ae35u);
            }
            reduce(ss.d_conv1d_weight, fs.d_conv1d_weight, rng_2[1] ^ 0x27d4eb2fu);
            if (ss.d_conv1d_bias.has_value() && fs.d_conv1d_bias.has_value()) {
                reduce(ss.d_conv1d_bias.value(), fs.d_conv1d_bias.value(), rng_2[1] ^ 0x165667b1u);
            }
            reduce(ss.d_A_log, fs.d_A_log, rng_2[2] ^ 0xd3a2646cu);
            reduce(ss.d_D, fs.d_D, rng_2[2] ^ 0xfd7046c5u);
            reduce(ss.d_dt_bias, fs.d_dt_bias, rng_2[3] ^ 0x5a48dfe7u);
            reduce(ss.d_norm_weight, fs.d_norm_weight, rng_2[3] ^ 0x6a09e667u);
        }
    }

    if constexpr (requires { shard.attention_grads.d_qkv_weight; full.attention_grads.d_qkv_weight; }) reduce(shard.attention_grads.d_qkv_weight, full.attention_grads.d_qkv_weight, rng_2[0]);
    if constexpr (requires { shard.attention.d_qkv_weight; full.attention.d_qkv_weight; }) reduce(shard.attention.d_qkv_weight, full.attention.d_qkv_weight, rng_2[0]);
    if constexpr (requires { shard.attention_grads.d_out_weight; full.attention_grads.d_out_weight; }) reduce(shard.attention_grads.d_out_weight, full.attention_grads.d_out_weight, rng_2[1]);
    if constexpr (requires { shard.attention.d_out_weight; full.attention.d_out_weight; }) reduce(shard.attention.d_out_weight, full.attention.d_out_weight, rng_2[1]);
    if constexpr (requires { shard.attention_grads.d_q_norm_weight; full.attention_grads.d_q_norm_weight; }) {
        if (shard.attention_grads.d_q_norm_weight.has_value() && full.attention_grads.d_q_norm_weight.has_value()) {
            reduce(shard.attention_grads.d_q_norm_weight.value(), full.attention_grads.d_q_norm_weight.value(), rng_2[3]);
        }
    }
    if constexpr (requires { shard.attention_grads.d_k_norm_weight; full.attention_grads.d_k_norm_weight; }) {
        if (shard.attention_grads.d_k_norm_weight.has_value() && full.attention_grads.d_k_norm_weight.has_value()) {
            reduce(shard.attention_grads.d_k_norm_weight.value(), full.attention_grads.d_k_norm_weight.value(), rng_2[3] ^ 0x9e3779b9u);
        }
    }
    if constexpr (requires { shard.attention.d_q_norm_weight; full.attention.d_q_norm_weight; }) {
        if (shard.attention.d_q_norm_weight.has_value() && full.attention.d_q_norm_weight.has_value()) {
            reduce(shard.attention.d_q_norm_weight.value(), full.attention.d_q_norm_weight.value(), rng_2[3] ^ 0x3c6ef372u);
        }
    }
    if constexpr (requires { shard.attention.d_k_norm_weight; full.attention.d_k_norm_weight; }) {
        if (shard.attention.d_k_norm_weight.has_value() && full.attention.d_k_norm_weight.has_value()) {
            reduce(shard.attention.d_k_norm_weight.value(), full.attention.d_k_norm_weight.value(), rng_2[3] ^ 0xbb67ae85u);
        }
    }

    if constexpr (requires { shard.router.d_gate; full.router.d_gate; }) reduce(shard.router.d_gate, full.router.d_gate, rng_2[0] ^ 0xa54ff53au);
    if constexpr (requires { shard.experts.d_gate_up_proj; full.experts.d_gate_up_proj; }) reduce(shard.experts.d_gate_up_proj, full.experts.d_gate_up_proj, rng_2[1] ^ 0x510e527fu);
    if constexpr (requires { shard.experts.d_down_proj; full.experts.d_down_proj; }) reduce(shard.experts.d_down_proj, full.experts.d_down_proj, rng_2[2] ^ 0x9b05688cu);

    if constexpr (requires { shard.attention_grads.d_qkv_bias; full.attention_grads.d_qkv_bias; }) {
        if (shard.attention_grads.d_qkv_bias.has_value() && full.attention_grads.d_qkv_bias.has_value()) {
            reduce(shard.attention_grads.d_qkv_bias.value(), full.attention_grads.d_qkv_bias.value(), rng_2[2]);
        }
    }
    if constexpr (requires { shard.attention.d_qkv_bias; full.attention.d_qkv_bias; }) {
        if (shard.attention.d_qkv_bias.has_value() && full.attention.d_qkv_bias.has_value()) {
            reduce(shard.attention.d_qkv_bias.value(), full.attention.d_qkv_bias.value(), rng_2[2]);
        }
    }
}

template<typename Block>
void ModularGradientManager<Block>::allocate_block_gradients(BlockGradients& grads, ETensorDType dtype, int layer_idx, bool force_mamba) {
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long M = mlp_up_rows(cfg);
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;
    const bool allocate_mamba = mConfig.has_mamba &&
        (force_mamba || (layer_idx >= 0 &&
                         layer_idx < static_cast<int>(mConfig.layer_is_mamba.size()) &&
                         mConfig.layer_is_mamba[static_cast<std::size_t>(layer_idx)] != 0));
    const bool allocate_moe = mConfig.has_moe &&
        (layer_idx < 0 ||
         (layer_idx < static_cast<int>(mConfig.layer_is_moe.size()) &&
          mConfig.layer_is_moe[static_cast<std::size_t>(layer_idx)] != 0));

    auto kind = EAllocationType::ON_DEVICE;

    if constexpr (requires { grads.ln1_grads.d_weight; }) {
        grads.ln1_grads.d_weight = mAllocator->allocate(dtype, "d_ln1_w", kind, {C});
    }
    if constexpr (requires { grads.ln2_grads.d_weight; }) {
        grads.ln2_grads.d_weight = mAllocator->allocate(dtype, "d_ln2_w", kind, {C});
    }
    // MoE blocks use grads.ln1/ln2.d_weight instead of grads.ln1_grads/ln2_grads.d_weight
    if constexpr (requires { grads.ln1.d_weight; }) {
        grads.ln1.d_weight = mAllocator->allocate(dtype, "d_ln1_w", kind, {C});
    }
    if constexpr (requires { grads.ln2.d_weight; }) {
        grads.ln2.d_weight = mAllocator->allocate(dtype, "d_ln2_w", kind, {C});
    }

    if constexpr (requires { grads.attention_grads.d_qkv_weight; }) {
        grads.attention_grads.d_qkv_weight = mAllocator->allocate(dtype, "d_qkv_w", kind, {qkv_channels, C});
        if constexpr (requires { grads.attention_grads.d_qkv_bias; }) {
            if (cfg.use_qkv_bias) {
                grads.attention_grads.d_qkv_bias = mAllocator->allocate(dtype, "d_qkv_b", kind, {qkv_channels});
            }
        }
        if constexpr (requires { grads.attention_grads.d_out_weight; }) {
            grads.attention_grads.d_out_weight = mAllocator->allocate(dtype, "d_att_out_w", kind, {C, q_rows});
        }
        if constexpr (requires { grads.attention_grads.d_q_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
            if constexpr (requires { cfg.use_qk_norm; }) {
                if (cfg.use_qk_norm) {
                    grads.attention_grads.d_q_norm_weight = mAllocator->allocate(dtype, "d_q_norm_w", kind, {HS});
                    grads.attention_grads.d_k_norm_weight = mAllocator->allocate(dtype, "d_k_norm_w", kind, {HS});
                }
            }
        }
    }

    if constexpr (requires { grads.attention.d_qkv_weight; }) {
        grads.attention.d_qkv_weight = mAllocator->allocate(dtype, "d_qkv_w", kind, {qkv_channels, C});
        if constexpr (requires { grads.attention.d_qkv_bias; }) {
            if (cfg.use_qkv_bias) {
                grads.attention.d_qkv_bias = mAllocator->allocate(dtype, "d_qkv_b", kind, {qkv_channels});
            }
        }
        if constexpr (requires { grads.attention.d_out_weight; }) {
            grads.attention.d_out_weight = mAllocator->allocate(dtype, "d_att_out_w", kind, {C, q_rows});
        }
        if constexpr (requires { grads.attention.d_q_norm_weight; grads.attention.d_k_norm_weight; }) {
            if constexpr (requires { cfg.use_qk_norm; }) {
                if (cfg.use_qk_norm) {
                    grads.attention.d_q_norm_weight = mAllocator->allocate(dtype, "d_q_norm_w", kind, {HS});
                    grads.attention.d_k_norm_weight = mAllocator->allocate(dtype, "d_k_norm_w", kind, {HS});
                }
            }
        }
    }

    if constexpr (requires { grads.d_mlp_up_weight; }) {
        grads.d_mlp_up_weight = mAllocator->allocate(dtype, "d_mlp_up_w", kind, {M, C});
    }
    if constexpr (requires { grads.d_mlp_down_weight; }) {
        grads.d_mlp_down_weight = mAllocator->allocate(dtype, "d_mlp_down_w", kind, {C, D});
    }

    if constexpr (requires { grads.mamba; }) {
        if (allocate_mamba) {
            if (!grads.mamba.has_value()) {
                grads.mamba.emplace();
            }
            auto& mg = *grads.mamba;
            const int mamba_dim = cfg.mamba_dim();
            const int conv_dim = cfg.mamba_conv_dim();
            const int proj_size = cfg.mamba_proj_size();

            mg.d_in_proj_weight = mAllocator->allocate(dtype, "d_mamba_in_w", kind, {proj_size, C});
            if (cfg.mamba_use_bias) {
                mg.d_in_proj_bias = mAllocator->allocate(dtype, "d_mamba_in_b", kind, {proj_size});
            } else {
                mg.d_in_proj_bias.reset();
            }
            mg.d_out_proj_weight = mAllocator->allocate(dtype, "d_mamba_out_w", kind, {C, mamba_dim});
            if (cfg.mamba_use_bias) {
                mg.d_out_proj_bias = mAllocator->allocate(dtype, "d_mamba_out_b", kind, {C});
            } else {
                mg.d_out_proj_bias.reset();
            }
            mg.d_conv1d_weight = mAllocator->allocate(dtype, "d_mamba_conv_w", kind, {conv_dim, cfg.mamba_conv_kernel});
            if (cfg.mamba_use_conv_bias) {
                mg.d_conv1d_bias = mAllocator->allocate(dtype, "d_mamba_conv_b", kind, {conv_dim});
            } else {
                mg.d_conv1d_bias.reset();
            }
            mg.d_A_log = mAllocator->allocate(ETensorDType::FP32, "d_mamba_A_log", kind, {cfg.mamba_num_heads});
            mg.d_D = mAllocator->allocate(ETensorDType::FP32, "d_mamba_D", kind, {cfg.mamba_num_heads});
            mg.d_dt_bias = mAllocator->allocate(ETensorDType::FP32, "d_mamba_dt_bias", kind, {cfg.mamba_num_heads});
            mg.d_norm_weight = mAllocator->allocate(dtype, "d_mamba_norm_w", kind, {mamba_dim});
        } else {
            grads.mamba.reset();
        }
    }

    // MoE-specific gradients can be extremely large; avoid allocating them in LoRA-only mode where base weights are frozen.
    // Exception: router gradient is allocated when train_router is enabled (for LoRA + router training mode).
    if (!mConfig.skip_allocation || mConfig.train_router) {
        if constexpr (requires { grads.router.d_gate; }) {
            if (allocate_moe) {
                grads.router.d_gate = mAllocator->allocate(dtype, "d_router_gate_w", kind, {cfg.num_experts, C});
            } else {
                grads.router.d_gate = Tensor();
            }
        }
    }
    if (!mConfig.skip_allocation) {
        if constexpr (requires { grads.experts.d_gate_up_proj; }) {
            if (allocate_moe) {
                grads.experts.d_gate_up_proj = mAllocator->allocate(dtype, "d_experts_gate_up_w", kind, {cfg.num_experts, M, C});
            } else {
                grads.experts.d_gate_up_proj = Tensor();
            }
        }
        if constexpr (requires { grads.experts.d_down_proj; }) {
            if (allocate_moe) {
                grads.experts.d_down_proj = mAllocator->allocate(dtype, "d_experts_down_w", kind, {cfg.num_experts, C, D});
            } else {
                grads.experts.d_down_proj = Tensor();
            }
        }
        if constexpr (requires { grads.shared_expert; }) {
            if (allocate_moe && cfg.use_shared_expert) {
                int shared_D = cfg.shared_expert_intermediate > 0 ? cfg.shared_expert_intermediate : static_cast<int>(D);
                long shared_M = static_cast<long>(mlp_up_factor(cfg)) * static_cast<long>(shared_D);
                grads.shared_expert.emplace();
                grads.shared_expert->d_gate_proj = mAllocator->allocate(dtype, "d_shared_expert_gate_w", kind, {shared_M, C});
                grads.shared_expert->d_down_proj = mAllocator->allocate(dtype, "d_shared_expert_down_w", kind, {C, shared_D});
            } else {
                grads.shared_expert.reset();
            }
        }
    }
}

template<typename Block>
void ModularGradientManager<Block>::allocate_block_gradients_shard(BlockGradients& grads, ETensorDType dtype, int layer_idx) {
    const auto& cfg = mConfig.block_config;
    long C = cfg.hidden_size;
    long D = cfg.intermediate_size;
    long M = mlp_up_rows(cfg);
    long HS = cfg.head_size;
    long HQ = cfg.num_query_heads;
    long HKV = cfg.num_kv_heads;
    long qkv_channels = HS * (HQ + 2 * HKV);
    long q_rows = HS * HQ;
    const bool allocate_mamba = mConfig.has_mamba &&
        (layer_idx >= 0 &&
         layer_idx < static_cast<int>(mConfig.layer_is_mamba.size()) &&
         mConfig.layer_is_mamba[static_cast<std::size_t>(layer_idx)] != 0);
    const bool allocate_moe = mConfig.has_moe &&
        (layer_idx < 0 ||
         (layer_idx < static_cast<int>(mConfig.layer_is_moe.size()) &&
          mConfig.layer_is_moe[static_cast<std::size_t>(layer_idx)] != 0));

    auto kind = mConfig.offload_grads ? mConfig.offload_alloc : EAllocationType::ON_DEVICE;

    auto alloc_shard_1d = [&](const char* name, long elems) -> Tensor {
        TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, {elems}, kind);
        return static_cast<Tensor>(shard);
    };
    auto alloc_shard_2d = [&](const char* name, long rows, long cols) -> Tensor {
        TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, {rows, cols}, kind);
        return static_cast<Tensor>(shard);
    };
    auto alloc_shard_3d = [&](const char* name, long dim0, long dim1, long dim2) -> Tensor {
        TensorShard shard = mAllocator->allocate_shard(dtype, mConfig.shard_idx, mConfig.num_shards, name, {dim0, dim1, dim2}, kind);
        return static_cast<Tensor>(shard);
    };

    if constexpr (requires { grads.ln1_grads.d_weight; }) {
        grads.ln1_grads.d_weight = alloc_shard_1d("d_ln1_w_shard", C);
    }
    if constexpr (requires { grads.ln2_grads.d_weight; }) {
        grads.ln2_grads.d_weight = alloc_shard_1d("d_ln2_w_shard", C);
    }
    // MoE blocks use grads.ln1/ln2.d_weight instead of grads.ln1_grads/ln2_grads.d_weight
    if constexpr (requires { grads.ln1.d_weight; }) {
        grads.ln1.d_weight = alloc_shard_1d("d_ln1_w_shard", C);
    }
    if constexpr (requires { grads.ln2.d_weight; }) {
        grads.ln2.d_weight = alloc_shard_1d("d_ln2_w_shard", C);
    }

    if constexpr (requires { grads.attention_grads.d_qkv_weight; }) {
        grads.attention_grads.d_qkv_weight = alloc_shard_2d("d_qkv_w_shard", qkv_channels, C);
        if constexpr (requires { grads.attention_grads.d_qkv_bias; }) {
            if (cfg.use_qkv_bias) {
                grads.attention_grads.d_qkv_bias = alloc_shard_1d("d_qkv_b_shard", qkv_channels);
            }
        }
        if constexpr (requires { grads.attention_grads.d_out_weight; }) {
            grads.attention_grads.d_out_weight = alloc_shard_2d("d_att_out_w_shard", C, q_rows);
        }
        if constexpr (requires { grads.attention_grads.d_q_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
            if constexpr (requires { cfg.use_qk_norm; }) {
                if (cfg.use_qk_norm) {
                    grads.attention_grads.d_q_norm_weight = alloc_shard_1d("d_q_norm_w_shard", HS);
                    grads.attention_grads.d_k_norm_weight = alloc_shard_1d("d_k_norm_w_shard", HS);
                }
            }
        }
    }

    if constexpr (requires { grads.attention.d_qkv_weight; }) {
        grads.attention.d_qkv_weight = alloc_shard_2d("d_qkv_w_shard", qkv_channels, C);
        if constexpr (requires { grads.attention.d_qkv_bias; }) {
            if (cfg.use_qkv_bias) {
                grads.attention.d_qkv_bias = alloc_shard_1d("d_qkv_b_shard", qkv_channels);
            }
        }
        if constexpr (requires { grads.attention.d_out_weight; }) {
            grads.attention.d_out_weight = alloc_shard_2d("d_att_out_w_shard", C, q_rows);
        }
        if constexpr (requires { grads.attention.d_q_norm_weight; grads.attention.d_k_norm_weight; }) {
            if constexpr (requires { cfg.use_qk_norm; }) {
                if (cfg.use_qk_norm) {
                    grads.attention.d_q_norm_weight = alloc_shard_1d("d_q_norm_w_shard", HS);
                    grads.attention.d_k_norm_weight = alloc_shard_1d("d_k_norm_w_shard", HS);
                }
            }
        }
    }

    if constexpr (requires { grads.d_mlp_up_weight; }) {
        grads.d_mlp_up_weight = alloc_shard_2d("d_mlp_up_w_shard", M, C);
    }
    if constexpr (requires { grads.d_mlp_down_weight; }) {
        grads.d_mlp_down_weight = alloc_shard_2d("d_mlp_down_w_shard", C, D);
    }

    if constexpr (requires { grads.mamba; }) {
        if (allocate_mamba) {
            if (!grads.mamba.has_value()) {
                grads.mamba.emplace();
            }
            auto& mg = *grads.mamba;
            const int mamba_dim = cfg.mamba_dim();
            const int conv_dim = cfg.mamba_conv_dim();
            const int proj_size = cfg.mamba_proj_size();

            mg.d_in_proj_weight = alloc_shard_2d("d_mamba_in_w_shard", proj_size, C);
            if (cfg.mamba_use_bias) {
                mg.d_in_proj_bias = alloc_shard_1d("d_mamba_in_b_shard", proj_size);
            } else {
                mg.d_in_proj_bias.reset();
            }
            mg.d_out_proj_weight = alloc_shard_2d("d_mamba_out_w_shard", C, mamba_dim);
            if (cfg.mamba_use_bias) {
                mg.d_out_proj_bias = alloc_shard_1d("d_mamba_out_b_shard", C);
            } else {
                mg.d_out_proj_bias.reset();
            }
            mg.d_conv1d_weight = alloc_shard_2d("d_mamba_conv_w_shard", conv_dim, cfg.mamba_conv_kernel);
            if (cfg.mamba_use_conv_bias) {
                mg.d_conv1d_bias = alloc_shard_1d("d_mamba_conv_b_shard", conv_dim);
            } else {
                mg.d_conv1d_bias.reset();
            }
            {
                TensorShard shard = mAllocator->allocate_shard(ETensorDType::FP32, mConfig.shard_idx, mConfig.num_shards,
                                                               "d_mamba_A_log_shard", {cfg.mamba_num_heads}, kind);
                mg.d_A_log = static_cast<Tensor>(shard);
            }
            {
                TensorShard shard = mAllocator->allocate_shard(ETensorDType::FP32, mConfig.shard_idx, mConfig.num_shards,
                                                               "d_mamba_D_shard", {cfg.mamba_num_heads}, kind);
                mg.d_D = static_cast<Tensor>(shard);
            }
            {
                TensorShard shard = mAllocator->allocate_shard(ETensorDType::FP32, mConfig.shard_idx, mConfig.num_shards,
                                                               "d_mamba_dt_bias_shard", {cfg.mamba_num_heads}, kind);
                mg.d_dt_bias = static_cast<Tensor>(shard);
            }
            mg.d_norm_weight = alloc_shard_1d("d_mamba_norm_w_shard", mamba_dim);
        } else {
            grads.mamba.reset();
        }
    }

    // MoE-specific gradients
    if constexpr (requires { grads.router.d_gate; }) {
        if (allocate_moe) {
            grads.router.d_gate = alloc_shard_2d("d_router_gate_w_shard", cfg.num_experts, C);
        } else {
            grads.router.d_gate = Tensor();
        }
    }
    if constexpr (requires { grads.experts.d_gate_up_proj; }) {
        if (allocate_moe) {
            grads.experts.d_gate_up_proj = alloc_shard_3d("d_experts_gate_up_w_shard", cfg.num_experts, M, C);
        } else {
            grads.experts.d_gate_up_proj = Tensor();
        }
    }
    if constexpr (requires { grads.experts.d_down_proj; }) {
        if (allocate_moe) {
            grads.experts.d_down_proj = alloc_shard_3d("d_experts_down_w_shard", cfg.num_experts, C, D);
        } else {
            grads.experts.d_down_proj = Tensor();
        }
    }
    if constexpr (requires { grads.shared_expert; }) {
        if (allocate_moe && cfg.use_shared_expert) {
            int shared_D = cfg.shared_expert_intermediate > 0 ? cfg.shared_expert_intermediate : static_cast<int>(D);
            long shared_M = static_cast<long>(mlp_up_factor(cfg)) * static_cast<long>(shared_D);
            grads.shared_expert.emplace();
            grads.shared_expert->d_gate_proj = alloc_shard_2d("d_shared_expert_gate_w_shard", shared_M, C);
            grads.shared_expert->d_down_proj = alloc_shard_2d("d_shared_expert_down_w_shard", C, shared_D);
        } else {
            grads.shared_expert.reset();
        }
    }
}

template<typename Block>
void ModularGradientManager<Block>::allocate_non_block_gradients(NonBlockGradients& grads, ETensorDType dtype) {
    long V = mConfig.vocab_size;
    long C = mConfig.hidden_size;

    grads.d_embeddings = mAllocator->allocate(dtype, "d_embeddings", EAllocationType::ON_DEVICE, {V, C});
    if (mConfig.tied_embeddings) {
        grads.d_lm_head = grads.d_embeddings;
    } else {
        grads.d_lm_head = mAllocator->allocate(dtype, "d_lm_head", EAllocationType::ON_DEVICE, {V, C});
    }
    grads.d_final_norm = mAllocator->allocate(dtype, "d_final_norm", EAllocationType::ON_DEVICE, {C});
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_GRADIENT_MANAGER_H
