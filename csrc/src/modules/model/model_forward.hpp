#pragma once

// Forward + validation

template<typename Block>
void ModularTransformerModel<Block>::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    forward_with_hook(inputs, position_ids, comm, micro_step, {});
}

template<typename Block>
void ModularTransformerModel<Block>::forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                                                       const ForwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;  // From IRunState base class
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    long V = mConfig.VocabSize;

    // Invalidate weight cache on first micro-step
    if (micro_step == 0) {
        // Ensure no weight prefetch starts before the previous optimizer update finished.
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        mWeights->invalidate();

        // Zero recorded amaxes for FP8 delayed scaling at start of each training step
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            // Initialize on first use (scales to 1.0, history to 0)
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(main_stream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(main_stream);
        }

        // Reset MoE stats for new training step
        rs.reset_moe_stats();
    }

    // Copy inputs to device
    assert(inputs.Device == -1);  // Inputs should be on host
    {
        NvtxRange r{"copy-input"};
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, main_stream));

        // Copy position IDs to GPU for RoPE
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, main_stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, main_stream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, main_stream));
    }

    // Embedding lookup
    {
        NvtxRange emb_range("embedding");
        mWeights->gather_embeddings(comm, rs.side_stream());
        auto& emb_weights = mWeights->get_embeddings(main_stream);
        auto& encoded = rs.non_block_activations().encoded;
        encoder_forward(encoded, rs.Inputs, emb_weights, std::nullopt, B, T, C, V, main_stream);
        mWeights->release_embeddings(main_stream);
    }

    // Process transformer blocks
    // Hooks can be captured safely as long as their topology is stable for the run (e.g. LoRA hooks).
    const bool use_cuda_graphs = mOptions.use_cuda_graphs;
    if (use_cuda_graphs) {
        rs.configure_forward_graphs(/*hooked=*/static_cast<bool>(hook));
    }
    mWeights->gather_block(0, comm, rs.side_stream());
    for (int l = 0; l < mConfig.NumLayers; l++) {
        NvtxRange layer_range("Layer", l);

        // Determine if this layer should use quantization (FP4/FP8)
        const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
        const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
        const bool in_skip_range = (l < skip_first) || (l >= mConfig.NumLayers - skip_last);
        const bool allow_quant_layer = !in_skip_range;

        // Prefetch next block
        if (l != mConfig.NumLayers - 1) {
            mWeights->gather_block(l + 1, comm, rs.side_stream());
        }

        auto& weights = mWeights->get_block(l, main_stream);

        auto& acts = rs.simplified_acts(l);
        auto& q = rs.simplified_quant_acts(l);

        // Get the residual buffer for this layer
        // For layer 0: residual is the encoded input
        // For layer L>0: residual is get_residual(l-1) which gets populated by the LN1 fused op
        Tensor& residual = l == 0 ? rs.non_block_activations().encoded :
                                    rs.get_residual(l - 1, main_stream);

        static_assert(requires { &Block::template forward_block_modular<Block>; },
                      "Block must implement forward_block_modular");
        const ForwardBlockHook* hook_ptr = hook ? &hook : nullptr;
        detail::trace_or_execute_cuda_graph_with_stack([&]() {
            Block::template forward_block_modular<Block>(
                *mRecipe, rs, weights, acts, q, residual, l,
                mConfig, mOptions, main_stream,
                rs.has_fp8_forward() ? &rs.fp8_forward_quants() : nullptr,
                rs.has_fp4_forward() ? &rs.fp4_forward_quants() : nullptr,
                mWeights.get(), allow_quant_layer, hook_ptr);
        }, main_stream, rs.forward_block_graph(l), use_cuda_graphs,
           rs.Stack, rs.forward_block_stack_checkpoint(l));

        mWeights->release_block(l, main_stream);
        if (l > 0 && mOptions.offload_residuals) {
            rs.put_residual(l - 1, rs.side_stream());
        }
    }

    // Final layer norm
    {
        NvtxRange lnf_range("LNF");
        auto& last_acts = rs.simplified_acts(mConfig.NumLayers - 1);
        mWeights->gather_final_norm(comm, rs.side_stream());
        auto& lnf_weight = mWeights->get_final_norm(main_stream);

        // Final residual: last_residual_att + last_mlp_down
        auto& final_res = rs.get_final_residual();
        fused_residual_rmsnorm_forward(final_res, rs.non_block_activations().ln_final,
                                       rs.non_block_activations().ln_final_rstd,
                                       last_acts.residual_att, last_acts.mlp_down, lnf_weight,
                                       nullptr, mConfig.RmsNormEps, B * T, C, main_stream);
        mWeights->release_final_norm(main_stream);
    }

    // Wait for input transfer to complete before returning
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, main_stream));
}

template<typename Block>
float ModularTransformerModel<Block>::validate(Tensor inputs, Tensor position_ids, Tensor targets,
                                                NCCLCommunicator& comm, int micro_step) {
    return validate_with_hook(inputs, position_ids, targets, comm, micro_step, {});
}

template<typename Block>
float ModularTransformerModel<Block>::validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets,
                                                         NCCLCommunicator& comm, int micro_step,
                                                         const ForwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    const size_t V = mConfig.VocabSize;
    const size_t Vp = mConfig.VocabSize;  // Padded vocab size (same as V for now)
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    cudaStream_t main_stream = rs.MainStream;

    // Run forward pass
    forward_with_hook(inputs, position_ids, comm, micro_step, hook);

    NvtxRange classifier_and_loss_range("classifier_and_loss");

    // Initialize losses and counts
    fill_zero(rs.Losses, main_stream);
    fill_zero(rs.ValidTokenCount, main_stream);
    fill_zero(rs.CorrectCount, main_stream);

    // Copy targets to device
    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    // LM head: project ln_final output to vocab logits
    // Do this in chunks for memory efficiency
    long nano_batches = mOptions.lmhead_chunks;
    int nano_batch_size = static_cast<int>((B * T) / nano_batches);

    mWeights->gather_lm_head(comm, rs.side_stream());

    // Use the output tensor from non_block_activations as scratch
    rs.temp_acquire(rs.non_block_activations().output);

    for (int nano_step = 0; nano_step < nano_batches; nano_step++) {
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

        // LM head matmul: logits = lnf @ lm_head.T
        matmul(rs.non_block_activations().output, mWeights->get_lm_head(main_stream), lnf_slice,
               std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
               V, nano_batch_size, C, EMMTranspose::TN, false, main_stream);

        // Fused classifier: softmax + cross-entropy loss + accuracy
        const float d_loss = 1.0f;
        fused_classifier(rs.non_block_activations().output, losses, d_loss, tgt,
                         &rs.ValidTokenCount, &rs.CorrectCount, nano_batch_size, V, Vp, true, main_stream);
    }

    rs.temp_free(rs.non_block_activations().output);
    mWeights->release_lm_head(main_stream);

    // Reduce loss
    reduce_loss(B, T, comm);

    // Get valid token count and correct count, then normalize loss and compute accuracy
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, main_stream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, main_stream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, main_stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);  // Reusing NormHost for valid token count
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);

    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        // Store accuracy as a percentage
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    return *rs.LossHost;
}
