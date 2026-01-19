#pragma once

// Per-block ops + loss/grad-norm

template<typename Block>
void ModularTransformerModel<Block>::forward_block(int layer_idx, BlockWeights& weights,
                                                    BlockActivations& acts, Tensor& residual) {
    // Delegate to block module's forward
    // This is a stub - actual implementation chains through sub-modules
}

template<typename Block>
void ModularTransformerModel<Block>::recompute_block(int layer_idx, BlockWeights& weights,
                                                      BlockActivations& acts, Tensor& residual) {
    (void)acts;
    if (!mOptions.recompute_rmsnorm &&
        !mOptions.recompute_qkv &&
        !mOptions.recompute_attention &&
        !mOptions.recompute_ffn &&
        !mOptions.recompute_swiglu &&
        !mOptions.recompute_block) {
        return;
    }

    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    auto& a = rs.simplified_acts(layer_idx);
    auto& q = rs.simplified_quant_acts(layer_idx);

    // Determine if this layer should use quantization (FP4/FP8)
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_quant_layer = !in_skip_range;

    static_assert(requires { &Block::template recompute_block_modular<Block>; },
                  "Block must implement recompute_block_modular");

    Block::template recompute_block_modular<Block>(
        *mRecipe, rs, weights, a, q, residual, layer_idx,
        mConfig, mOptions, stream,
        rs.has_fp8_forward() ? &rs.fp8_forward_quants() : nullptr,
        rs.has_fp4_forward() ? &rs.fp4_forward_quants() : nullptr,
        mWeights.get(), allow_quant_layer);
}

template<typename Block>
void ModularTransformerModel<Block>::backward_block(int layer_idx, bool accumulate, BlockWeights& weights,
                                                     BlockGradients& grads, BlockActivations& acts,
                                                     typename ModularRunState<Block>::BlockGradients& d_acts,
                                                     const BackwardBlockHook* hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    // Determine if this layer should use quantization (FP4/FP8)
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_quant_layer = !in_skip_range;

    (void)acts;
    (void)d_acts;

    // Use the simplified activation/gradient buffers
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);
    auto& qa = rs.simplified_quant_acts(layer_idx);
    auto& qg = rs.simplified_quant_grads();

    static_assert(requires { &Block::template backward_block_modular<Block>; },
                  "Block must implement backward_block_modular");

    Block::template backward_block_modular<Block>(
        *mRecipe, rs, weights, grads, a, da, qa, qg,
        layer_idx, mConfig, mOptions, accumulate,
        stream, allow_quant_layer, hook);
}

template<typename Block>
void ModularTransformerModel<Block>::backward_lmhead(long B, long T, int micro_step, int grad_accum_steps,
                                                      NCCLCommunicator& comm) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    const size_t C = mConfig.HiddenSize;
    const size_t V = mConfig.VocabSize;
    const size_t Vp = mConfig.VocabSize;  // Padded vocab size (same for now)
    cudaStream_t main_stream = rs.MainStream;

    // LoRA-only mode: skip computing LM head weight gradients
    const bool lora_only = rs.is_lora_only_mode();

    long nano_batches = mOptions.lmhead_chunks;
    int nano_batch_size = static_cast<int>((B * T) / nano_batches);

    // Loss normalization factor
    const float d_loss = 1.0f / static_cast<float>(B * T * grad_accum_steps);

    NvtxRange classifier_range("lm-head");
    mWeights->gather_lm_head(comm, rs.side_stream());

    // Note: Losses and ValidTokenCount are zeroed in backward_with_hook on micro_step==0
    // The fused_classifier kernel accumulates: losses[idx] -= logf(prob)
    rs.temp_acquire(rs.non_block_activations().output);

    for (int nano_step = 0; nano_step < nano_batches; nano_step++) {
        if (nano_step == 0) {
            // Ensure targets have been copied to device before fused_classifier reads them.
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.TransferDone, 0));
            // On the first micro-step, gradients are zeroed on the side stream; ensure that's done
            // before we start writing into any grad buffers on the main stream.
            if (micro_step == 0) {
                CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.side_stream_event(), 0));
            }
        }
        // Slice tensors for this nano-batch
        // Note: Update both Data pointer AND Sizes for correct matmul dimensions
        Tensor lnf_slice = rs.non_block_activations().ln_final;
        lnf_slice.Data = static_cast<std::byte*>(lnf_slice.Data) +
                         nano_step * nano_batch_size * C * get_dtype_size(lnf_slice.DType);
        lnf_slice.Sizes[0] = nano_batch_size;
        lnf_slice.Sizes[1] = C;
        lnf_slice.Rank = 2;

        Tensor tgt = rs.Targets;
        tgt.Data = static_cast<std::byte*>(tgt.Data) +
                   nano_step * nano_batch_size * get_dtype_size(tgt.DType);
        tgt.Sizes[0] = nano_batch_size;
        tgt.Rank = 1;

        Tensor losses = rs.Losses;
        losses.Data = static_cast<std::byte*>(losses.Data) +
                      nano_step * nano_batch_size * get_dtype_size(losses.DType);
        losses.Sizes[0] = nano_batch_size;
        losses.Rank = 1;

        Tensor dlnf_slice = rs.non_block_gradients().d_ln_final;
        dlnf_slice.Data = static_cast<std::byte*>(dlnf_slice.Data) +
                          nano_step * nano_batch_size * C * get_dtype_size(dlnf_slice.DType);
        dlnf_slice.Sizes[0] = nano_batch_size;
        dlnf_slice.Sizes[1] = C;
        dlnf_slice.Rank = 2;

        // Forward: logits = lnf @ lm_head.T
        matmul(rs.non_block_activations().output, mWeights->get_lm_head(main_stream), lnf_slice,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

        // Wait for targets on first nano-step
        if (nano_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.TransferDone, 0));
        }

        // Fused classifier: softmax + cross-entropy + sets logits to dlogits
        fused_classifier(rs.non_block_activations().output, losses, d_loss, tgt,
                         &rs.ValidTokenCount, nano_batch_size, V, Vp, true, main_stream);

        // Wait for grad zero on first micro/nano step
        if (micro_step == 0 && nano_step == 0) {
            CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.side_stream_event(), 0));
        }

        // Backward: d_lm_head += lnf.T @ dlogits (skip in LoRA-only mode)
        if (!lora_only) {
            bool accumulate;
            auto& d_lmhead = mGrads->get_lm_head_full(main_stream, comm, accumulate);
            accumulate |= nano_step != 0;
            matmul(d_lmhead, lnf_slice, rs.non_block_activations().output,
                   std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
                   C, V, nano_batch_size, EMMTranspose::NT, accumulate, main_stream);
            if (nano_step == nano_batches - 1) {
                mGrads->notify_lm_head(main_stream, comm);
            }
        }

        // Backward: d_lnf = dlogits @ lm_head
        matmul(dlnf_slice, mWeights->get_lm_head(main_stream), rs.non_block_activations().output,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               C, nano_batch_size, V, EMMTranspose::NN, false, main_stream);
    }

    rs.temp_free(rs.non_block_activations().output);
    mWeights->release_lm_head(main_stream);
}

template<typename Block>
void ModularTransformerModel<Block>::reduce_loss(long B, long T, NCCLCommunicator& comm) {
    NVTX_RANGE_FN();
    auto& rs = *mRunState;

    // Reduce all losses within the current GPU (across all nano-batches)
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);

    // Reduce loss across GPUs to a single, final float
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);

    // Copy loss to host
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
}

template<typename Block>
void ModularTransformerModel<Block>::calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
    NVTX_RANGE_FN();
    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Ensure backward has completed before reading gradients.
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, rs.BackwardDone));

    detail::trace_or_execute_cuda_graph([&]() {
        calculate_gradient_norm_impl(comm, grad_clip, main_stream);
    }, main_stream, rs.global_norm_graph(), mOptions.use_cuda_graphs);

    CUDA_CHECK(cudaEventRecord(rs.NormDone, main_stream));
}

template<typename Block>
void ModularTransformerModel<Block>::calculate_gradient_norm_impl(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream) {
    auto& rs = *mRunState;

    fill_zero(rs.scratch().norm_buffer, stream);

    auto norm_squared = [&](const TensorShard& grad) {
        global_norm_squared(rs.scratch().norm_buffer, grad, grad.nelem(), rs.DeviceProp, stream);
    };

    auto& emb_grad = mGrads->get_embeddings_shard(stream);
    norm_squared(emb_grad);
    auto& head_grad = mGrads->get_lm_head_shard(stream);
    if (head_grad.Data != emb_grad.Data) {
        norm_squared(head_grad);
    }
    norm_squared(mGrads->get_final_norm_shard(stream));

    for (int i = 0; i < mConfig.NumLayers; ++i) {
        auto& g = mGrads->get_block_shard(i, stream);
        if constexpr (requires { g.ln1_grads.d_weight; }) norm_squared(TensorShard(g.ln1_grads.d_weight));
        if constexpr (requires { g.ln2_grads.d_weight; }) norm_squared(TensorShard(g.ln2_grads.d_weight));
        if constexpr (requires { g.ln1.d_weight; }) norm_squared(TensorShard(g.ln1.d_weight));
        if constexpr (requires { g.ln2.d_weight; }) norm_squared(TensorShard(g.ln2.d_weight));

        if constexpr (requires { g.attention_grads.d_qkv_weight; }) norm_squared(TensorShard(g.attention_grads.d_qkv_weight));
        if constexpr (requires { g.attention_grads.d_qkv_bias; }) {
            if (g.attention_grads.d_qkv_bias.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_qkv_bias.value()));
            }
        }
        if constexpr (requires { g.attention_grads.d_out_weight; }) norm_squared(TensorShard(g.attention_grads.d_out_weight));
        if constexpr (requires { g.attention_grads.d_q_norm_weight; }) {
            if (g.attention_grads.d_q_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_q_norm_weight.value()));
            }
        }
        if constexpr (requires { g.attention_grads.d_k_norm_weight; }) {
            if (g.attention_grads.d_k_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention_grads.d_k_norm_weight.value()));
            }
        }

        if constexpr (requires { g.attention.d_qkv_weight; }) norm_squared(TensorShard(g.attention.d_qkv_weight));
        if constexpr (requires { g.attention.d_qkv_bias; }) {
            if (g.attention.d_qkv_bias.has_value()) {
                norm_squared(TensorShard(g.attention.d_qkv_bias.value()));
            }
        }
        if constexpr (requires { g.attention.d_out_weight; }) norm_squared(TensorShard(g.attention.d_out_weight));
        if constexpr (requires { g.attention.d_q_norm_weight; }) {
            if (g.attention.d_q_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention.d_q_norm_weight.value()));
            }
        }
        if constexpr (requires { g.attention.d_k_norm_weight; }) {
            if (g.attention.d_k_norm_weight.has_value()) {
                norm_squared(TensorShard(g.attention.d_k_norm_weight.value()));
            }
        }

        if constexpr (requires { g.d_mlp_up_weight; }) norm_squared(TensorShard(g.d_mlp_up_weight));
        if constexpr (requires { g.d_mlp_down_weight; }) norm_squared(TensorShard(g.d_mlp_down_weight));

        if constexpr (requires { g.router.d_gate; }) norm_squared(TensorShard(g.router.d_gate));
        if constexpr (requires { g.experts.d_gate_up_proj; }) norm_squared(TensorShard(g.experts.d_gate_up_proj));
        if constexpr (requires { g.experts.d_down_proj; }) norm_squared(TensorShard(g.experts.d_down_proj));
    }

    // Reduce partial sums to a single scalar on-device.
    deterministic_sum(rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.nelem(),
                      stream);

    // Cross-rank reduction.
    comm.reduce_norm(rs.scratch().norm_buffer.template get<float>(), stream);

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));
    global_norm_sqrt(rs.scratch().norm_buffer.template get<float>(), rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
}
