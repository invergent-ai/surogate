// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_OPTIMIZER_H
#define LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_OPTIMIZER_H

#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "modules/weights/weight_manager_helpers.h"
#include "utilities/comm.h"

namespace modules {

template<typename Block>
void ModularWeightManager<Block>::begin_optimizer(DeviceMemoryStack& memory, cudaStream_t stream) {
    (void)memory;
    mOptimizerActive = true;
    mOptimizerStream = stream;
    if (mAbsMaxes.Data) {
        fill_zero(mAbsMaxes, stream);
    }

    if (mConfig.offload_master && !mConfig.use_zero_copy) {
        auto copy_h2d = [&](const Tensor& src, Tensor& dst) {
            if (!src.Data || !dst.Data || src.nelem() == 0) return;
            dst.Stats = src.Stats;
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, dst.bytes(), cudaMemcpyHostToDevice, stream));
        };

        copy_h2d(mMasterNonBlock.embeddings, mMasterNonBlockDevice.embeddings);
        copy_h2d(mMasterNonBlock.final_norm_weight, mMasterNonBlockDevice.final_norm_weight);
        if (!mConfig.tied_embeddings) {
            copy_h2d(mMasterNonBlock.lm_head, mMasterNonBlockDevice.lm_head);
        } else {
            mMasterNonBlockDevice.lm_head = mMasterNonBlockDevice.embeddings;
        }

        // Point master shard views at the device staging buffers for the duration of the optimizer pass.
        if (mConfig.num_shards > 1) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlockDevice.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                    std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlockDevice.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.hidden_size});
            if (mConfig.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlockDevice.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                     std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlockDevice.embeddings);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlockDevice.final_norm_weight);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlockDevice.lm_head);
        }
    }
}

template<typename Block>
void ModularWeightManager<Block>::end_optimizer(DeviceMemoryStack& memory) {
    (void)memory;
    if (mConfig.offload_master && !mConfig.use_zero_copy) {
        if (!mOptimizerStream) {
            throw std::logic_error("ModularWeightManager::end_optimizer called without a prior begin_optimizer()");
        }
        auto copy_d2h = [&](Tensor& dst, const Tensor& src) {
            if (!dst.Data || !src.Data || src.nelem() == 0) return;
            CUDA_CHECK(cudaMemcpyAsync(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToHost, mOptimizerStream));
        };

        copy_d2h(mMasterNonBlock.embeddings, mMasterNonBlockDevice.embeddings);
        copy_d2h(mMasterNonBlock.final_norm_weight, mMasterNonBlockDevice.final_norm_weight);
        if (!mConfig.tied_embeddings) {
            copy_d2h(mMasterNonBlock.lm_head, mMasterNonBlockDevice.lm_head);
        }

        // Restore master shard views to the (host) master tensors.
        if (mConfig.num_shards > 1) {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings, mConfig.shard_idx, mConfig.num_shards,
                                                    std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight, mConfig.shard_idx, mConfig.num_shards,
                                                   std::vector<long>{mConfig.hidden_size});
            if (mConfig.tied_embeddings) {
                mMasterLMHeadShardView = mMasterEmbeddingsShardView;
            } else {
                mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head, mConfig.shard_idx, mConfig.num_shards,
                                                     std::vector<long>{mConfig.vocab_size, mConfig.hidden_size});
            }
        } else {
            mMasterEmbeddingsShardView = TensorShard(mMasterNonBlock.embeddings);
            mMasterFinalNormShardView = TensorShard(mMasterNonBlock.final_norm_weight);
            mMasterLMHeadShardView = TensorShard(mMasterNonBlock.lm_head);
        }
    }

    mOptimizerActive = false;
    mOptimizerStream = nullptr;
}

template<typename Block>
void ModularWeightManager<Block>::fetch_master_block(int layer_idx, cudaStream_t fetch_stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) return;

    int buf_idx = layer_idx % 2;
    auto& status = mMasterStatus[buf_idx];

    CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, status.done_event));

    if (status.layer_idx == layer_idx && status.version == mVersion) {
        return;
    }

    status.layer_idx = layer_idx;
    status.fetch_pending = true;
    status.is_ready = false;
    status.version = mVersion;

    auto& src = mMasterBlocks.at(layer_idx);
    auto& dst = mMasterBuffer.at(buf_idx);

    auto copy_h2d = [&](const Tensor& s, Tensor& d) {
        if (!s.Data || !d.Data || s.nelem() == 0) return;
        d.Stats = s.Stats;
        CUDA_CHECK(cudaMemcpyAsync(d.Data, s.Data, d.bytes(), cudaMemcpyHostToDevice, fetch_stream));
    };

    copy_h2d(src.ln1.weight, dst.ln1.weight);
    copy_h2d(src.attention.qkv_weight, dst.attention.qkv_weight);
    if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        copy_h2d(src.attention.qkv_bias.value(), dst.attention.qkv_bias.value());
    }
    copy_h2d(src.attention.out_weight, dst.attention.out_weight);
    using AttentionWeightsType = std::decay_t<decltype(src.attention)>;
    if constexpr (has_qk_norm_weights<AttentionWeightsType>::value) {
        if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            copy_h2d(src.attention.q_norm_weight.value(), dst.attention.q_norm_weight.value());
        }
        if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            copy_h2d(src.attention.k_norm_weight.value(), dst.attention.k_norm_weight.value());
        }
    }
	    copy_h2d(src.ln2.weight, dst.ln2.weight);
	    if constexpr (has_mlp_weights<BlockWeights>::value) {
	        copy_h2d(src.mlp_up_weight, dst.mlp_up_weight);
	        copy_h2d(src.mlp_down_weight, dst.mlp_down_weight);
	    }
	    if constexpr (has_moe_weights<BlockWeights>::value) {
	        copy_h2d(src.router.gate, dst.router.gate);
	        if (src.router.bias.has_value() && dst.router.bias.has_value()) {
	            copy_h2d(src.router.bias.value(), dst.router.bias.value());
	        }
	        if (src.experts.use_batched && dst.experts.use_batched) {
	            copy_h2d(src.experts.gate_up_proj, dst.experts.gate_up_proj);
	            copy_h2d(src.experts.down_proj, dst.experts.down_proj);
	        }
	        if (src.shared_expert.has_value() && dst.shared_expert.has_value()) {
	            copy_h2d(src.shared_expert->gate_proj, dst.shared_expert->gate_proj);
	            copy_h2d(src.shared_expert->up_proj, dst.shared_expert->up_proj);
	            copy_h2d(src.shared_expert->down_proj, dst.shared_expert->down_proj);
	        }
	    }

    CUDA_CHECK(cudaEventRecord(status.done_event, fetch_stream));
}

template<typename Block>
typename Block::Weights& ModularWeightManager<Block>::get_master_block(int layer_idx, cudaStream_t stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) {
        return mMasterBlocks.at(layer_idx);
    }

    int buf_idx = layer_idx % 2;
    auto& st = mMasterStatus.at(buf_idx);
    if (st.layer_idx != layer_idx) {
        throw std::runtime_error("Master weights not fetched: layer " + std::to_string(layer_idx));
    }
    wait_for_buffer(st, stream);
    return mMasterBuffer.at(buf_idx);
}

template<typename Block>
void ModularWeightManager<Block>::release_master_block(int layer_idx, cudaStream_t compute_stream, cudaStream_t put_stream) {
    if (!mConfig.offload_master || mConfig.use_zero_copy) return;

    int buf_idx = layer_idx % 2;
    auto& st = mMasterStatus.at(buf_idx);
    if (st.layer_idx != layer_idx) return;

    auto& src = mMasterBuffer.at(buf_idx);
    auto& dst = mMasterBlocks.at(layer_idx);

    // Ensure all update kernels that touched `src` are enqueued before starting D2H.
    CUDA_CHECK(cudaEventRecord(st.done_event, compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(put_stream, st.done_event, 0));

    auto copy_d2h = [&](const Tensor& s, Tensor& d) {
        if (!s.Data || !d.Data || s.nelem() == 0) return;
        CUDA_CHECK(cudaMemcpyAsync(d.Data, s.Data, s.bytes(), cudaMemcpyDeviceToHost, put_stream));
    };

    copy_d2h(src.ln1.weight, dst.ln1.weight);
    copy_d2h(src.attention.qkv_weight, dst.attention.qkv_weight);
    if (src.attention.qkv_bias.has_value() && dst.attention.qkv_bias.has_value()) {
        copy_d2h(src.attention.qkv_bias.value(), dst.attention.qkv_bias.value());
    }
    copy_d2h(src.attention.out_weight, dst.attention.out_weight);
    using ReleaseAttentionWeightsType = std::decay_t<decltype(src.attention)>;
    if constexpr (has_qk_norm_weights<ReleaseAttentionWeightsType>::value) {
        if (src.attention.q_norm_weight.has_value() && dst.attention.q_norm_weight.has_value()) {
            copy_d2h(src.attention.q_norm_weight.value(), dst.attention.q_norm_weight.value());
        }
        if (src.attention.k_norm_weight.has_value() && dst.attention.k_norm_weight.has_value()) {
            copy_d2h(src.attention.k_norm_weight.value(), dst.attention.k_norm_weight.value());
        }
    }
	    copy_d2h(src.ln2.weight, dst.ln2.weight);
	    if constexpr (has_mlp_weights<BlockWeights>::value) {
	        copy_d2h(src.mlp_up_weight, dst.mlp_up_weight);
	        copy_d2h(src.mlp_down_weight, dst.mlp_down_weight);
	    }
	    if constexpr (has_moe_weights<BlockWeights>::value) {
	        copy_d2h(src.router.gate, dst.router.gate);
	        if (src.router.bias.has_value() && dst.router.bias.has_value()) {
	            copy_d2h(src.router.bias.value(), dst.router.bias.value());
	        }
	        if (src.experts.use_batched && dst.experts.use_batched) {
	            copy_d2h(src.experts.gate_up_proj, dst.experts.gate_up_proj);
	            copy_d2h(src.experts.down_proj, dst.experts.down_proj);
	        }
	        if (src.shared_expert.has_value() && dst.shared_expert.has_value()) {
	            copy_d2h(src.shared_expert->gate_proj, dst.shared_expert->gate_proj);
	            copy_d2h(src.shared_expert->up_proj, dst.shared_expert->up_proj);
	            copy_d2h(src.shared_expert->down_proj, dst.shared_expert->down_proj);
	        }
	    }

    CUDA_CHECK(cudaEventRecord(st.done_event, put_stream));
    st.is_ready = true;
    st.fetch_pending = false;
    st.version = mVersion;

    // Make the compute stream wait so the caller's "optimizer done" event includes the D2H.
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, st.done_event, 0));
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_embeddings() {
    return mMasterEmbeddingsShardView;
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_lm_head() {
    return mMasterLMHeadShardView;
}

template<typename Block>
TensorShard& ModularWeightManager<Block>::get_master_final_norm() {
    return mMasterFinalNormShardView;
}

template<typename Block>
void ModularWeightManager<Block>::synchronize_absmax(NCCLCommunicator& comm) {
    if (!mAbsMaxes.Data) return;
    if (mConfig.offload_master && mMasterNonBlock.embeddings.Device == -1 && !mConfig.use_zero_copy) {
        throw std::runtime_error("ModularWeightManager::synchronize_absmax: --offload-master requires --use-zero-copy (abs_max needs device-accessible weights)");
    }

    cudaStream_t stream = comm.stream();

    auto compute = [&](Tensor& t) {
        if (!t.Data || t.nelem() == 0 || !t.abs_max()) return;
        // abs_max() is only implemented for FP32/BF16.
        if (t.DType != ETensorDType::FP32 && t.DType != ETensorDType::BF16) return;

        abs_max(t.abs_max(), t, (long)t.nelem(), mDeviceProp, stream);
        if (comm.world_size() > 1) {
            comm.reduce_max(t.abs_max(), /*n=*/1, stream);
        }
    };

    compute(mMasterNonBlock.embeddings);
    compute(mMasterNonBlock.final_norm_weight);
    if (!mConfig.tied_embeddings) {
        compute(mMasterNonBlock.lm_head);
    }

    for (int i = 0; i < mConfig.num_layers; ++i) {
        auto& b = mMasterBlocks[i];
        compute(b.ln1.weight);
        compute(b.ln2.weight);
        compute(b.attention.qkv_weight);
        if (b.attention.qkv_bias.has_value()) {
            compute(b.attention.qkv_bias.value());
        }
	        compute(b.attention.out_weight);
	        if constexpr (has_mlp_weights<BlockWeights>::value) {
	            compute(b.mlp_up_weight);
	            compute(b.mlp_down_weight);
	        }
	        if constexpr (has_moe_weights<BlockWeights>::value) {
	            compute(b.router.gate);
	            if (b.router.bias.has_value()) compute(b.router.bias.value());
	            if (b.experts.use_batched) {
	                compute(b.experts.gate_up_proj);
	                compute(b.experts.down_proj);
	            }
	            if (b.shared_expert.has_value()) {
	                compute(b.shared_expert->gate_proj);
	                compute(b.shared_expert->up_proj);
	                compute(b.shared_expert->down_proj);
	            }
	        }
	        comm.wait_on_comms(stream);
	    }

    comm.barrier();
}

} // namespace modules

#endif // LLMQ_SRC_MODULES_WEIGHTS_WEIGHT_MANAGER_OPTIMIZER_H
