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

    // Configuration shortcuts
    const long D = mConfig.IntermediateSize;
    const int Hq = mConfig.NumQueryHeads;
    const int Hkv = mConfig.NumKeyValHeads;
    const int Hs = mConfig.head_size();
    const long AttC = static_cast<long>(Hq) * static_cast<long>(Hs);
    const long qkv_channels = mConfig.qkv_channels();

    // RoPE frequencies and position IDs
    auto& freq_cis = rs.non_block_activations().freq_cis;
    int* pos_ids_ptr = rs.PositionIDs.template get<int>();

    // Process transformer blocks
    // Hooks can be captured safely as long as their topology is stable for the run (e.g. LoRA hooks).
    const bool use_cuda_graphs = mOptions.use_cuda_graphs;
    if (use_cuda_graphs) {
        rs.configure_forward_graphs(/*hooked=*/static_cast<bool>(hook));
    }
    mWeights->gather_block(0, comm, rs.side_stream());
    for (int l = 0; l < mConfig.NumLayers; l++) {
        NvtxRange layer_range("Layer", l);

        // Determine if this layer should use FP4 quantization
        const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
        const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
        const bool in_skip_range = (l < skip_first) || (l >= mConfig.NumLayers - skip_last);
        const bool allow_fp4_layer = !in_skip_range;

        // Prefetch next block
        if (l != mConfig.NumLayers - 1) {
            mWeights->gather_block(l + 1, comm, rs.side_stream());
        }

        auto& weights = mWeights->get_block(l, main_stream);

        auto& acts = rs.simplified_acts(l);
        auto& q = rs.simplified_quant_acts(l);

        // Get the residual buffer for this layer (like legacy DeviceResiduals)
        // For layer 0: residual is the encoded input
        // For layer L>0: residual is get_residual(l-1) which gets populated by the LN1 fused op
        Tensor& residual = l == 0 ? rs.non_block_activations().encoded :
                                    rs.get_residual(l - 1, main_stream);

        // 1) First layer norm
        // The fused op writes: output_residual = inp1 + inp2, output_norm = rmsnorm(output_residual)
        // For L=0: just normalize encoded (no residual accumulation)
        // For L>0: accumulate prev.residual_att + prev.mlp_down into get_residual(l-1)
        // Determine abs_max pointer for RMSNorm: for FP8 forward, we need abs_max computed
        // so the recipe can reuse it for input quantization (FP8/FP4).
        float* ln1_abs_max_ptr = nullptr;
        if (rs.has_fp8_forward()) {
            ln1_abs_max_ptr = rs.fp8_forward_quants().ln1.abs_max();
        } else if (rs.has_fp4_forward()) {
            ln1_abs_max_ptr = rs.fp4_forward_quants().ln1_global_amax;
        } else if (rs.has_activation_quants()) {
            ln1_abs_max_ptr = q.ln1.abs_max();
        }

        if (l == 0) {
            // First layer: just normalize (no residual addition)
            rmsnorm_forward(acts.ln1, acts.ln1_rstd, residual, weights.ln1.weight,
                            ln1_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, main_stream);
        } else {
            // Subsequent layers: compute accumulated residual and normalize
            // residual buffer = prev.residual_att + prev.mlp_down
            auto& prev = rs.simplified_acts(l - 1);
            // Write to residual buffer (get_residual(l-1)) which is separate from prev.residual_att
            fused_residual_rmsnorm_forward(residual, acts.ln1, acts.ln1_rstd,
                                           prev.residual_att, prev.mlp_down, weights.ln1.weight,
                                           ln1_abs_max_ptr,
                                           mConfig.RmsNormEps, B * T, C, main_stream);
            // If residual offload is enabled, mark the produced residual (for layer l-1) ready for D2H.
            if (mOptions.offload_residuals) {
                rs.mark_residual_ready(l - 1, main_stream);
            }
        }

        detail::trace_or_execute_cuda_graph_with_stack([&]() {
            // 2) QKV projection - recipe handles all format decisions
            {
                Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln1 : nullptr;
                const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().qkv_weight : nullptr;
                const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN1) : -1;

                // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                const Tensor* fp4_data = nullptr;
                const Tensor* fp4_scales = nullptr;
                const float* fp4_amax = nullptr;
                if (mWeights->has_fp4_forward_cache()) {
                    auto& fp4_cache = mWeights->fp4_weight_cache();
                    fp4_data = &fp4_cache.qkv_weight.data;
                    fp4_scales = &fp4_cache.qkv_weight.scales;
                    fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 0;  // qkv at offset 0
                }

                detail::recipe_forward_matmul(
                    *mRecipe, acts.qkv, acts.ln1, weights.attention.qkv_weight,
                    weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
                    rs, (int)B, (int)T, (int)C, (int)qkv_channels,
                    l, modules::MatmulOp::QKV,
                    inp_quant, cached_weight, qidx, main_stream,
                    fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
            }

            if (hook) {
                hook(l, main_stream, ForwardHookPoint::AfterQKVProjection);
            }

            const bool use_qk_norm =
                weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value();

            // 2.5) Optional Q/K head RMSNorm (Qwen3-style).
            if (use_qk_norm) {
                const int q_rows = Hq * Hs;
                qkv_head_rmsnorm_forward(
                    acts.qkv, acts.q_rstd, weights.attention.q_norm_weight.value(),
                    mConfig.RmsNormEps,
                    (int)B, (int)T, (int)qkv_channels,
                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                    main_stream
                );
                qkv_head_rmsnorm_forward(
                    acts.qkv, acts.k_rstd, weights.attention.k_norm_weight.value(),
                    mConfig.RmsNormEps,
                    (int)B, (int)T, (int)qkv_channels,
                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                    main_stream
                );
            }

            // 3) Apply RoPE
            Tensor& qkv_for_attn = (use_qk_norm && acts.qkv_rope.Data != nullptr) ? acts.qkv_rope : acts.qkv;
            if (use_qk_norm && acts.qkv_rope.Data != nullptr) {
                // Keep pre-RoPE QKV (after QK-norm) for backward, and store RoPE outputs separately.
                if (mOptions.use_fused_rope) {
                    rope_fused_forward(qkv_for_attn, acts.qkv, pos_ids_ptr, nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, main_stream);
                } else {
                    rope_forward(qkv_for_attn, acts.qkv, freq_cis, pos_ids_ptr, nullptr, B, T, Hq, Hkv, Hs, main_stream);
                }
            } else {
                if (mOptions.use_fused_rope) {
                    rope_fused_forward(acts.qkv, acts.qkv, pos_ids_ptr, nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, main_stream);
                } else {
                    rope_forward(acts.qkv, acts.qkv, freq_cis, pos_ids_ptr, nullptr, B, T, Hq, Hkv, Hs, main_stream);
                }
            }

            // 4) Attention (FlashAttention via cuDNN)
            // Match legacy behavior: use the shared CuBLAS workspace for cuDNN forward.
            attention_forward_cudnn(acts.att, acts.lse, qkv_for_attn, rs.CuBlasWorkspace,
                                    rs.CudnnHandle, B, T, Hq, Hkv, Hs, main_stream);

            // Compute abs_max for attention output (used by FP8/FP4 recipes for output projection quantization)
            if (rs.has_fp8_forward()) {
                abs_max(rs.fp8_forward_quants().att.abs_max(), acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            } else if (rs.has_fp4_forward()) {
                abs_max(rs.fp4_forward_quants().att_global_amax, acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            } else if (rs.has_activation_quants()) {
                abs_max(q.att.abs_max(), acts.att, (long)acts.att.nelem(), rs.DeviceProp, main_stream);
            }

            // 5) Output projection - recipe handles all format decisions
            {
                Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().att : nullptr;
                const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().o_weight : nullptr;
                const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_ATT) : -1;

                // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                const Tensor* fp4_data = nullptr;
                const Tensor* fp4_scales = nullptr;
                const float* fp4_amax = nullptr;
                if (mWeights->has_fp4_forward_cache()) {
                    auto& fp4_cache = mWeights->fp4_weight_cache();
                    fp4_data = &fp4_cache.o_weight.data;
                    fp4_scales = &fp4_cache.o_weight.scales;
                    fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 1;  // o at offset 1
                }

                detail::recipe_forward_matmul(
                    *mRecipe, acts.att_out, acts.att, weights.attention.out_weight,
                    nullptr,  // no bias
                    rs, (int)B, (int)T, (int)AttC, (int)C,
                    l, modules::MatmulOp::AttnOut,
                    inp_quant, cached_weight, qidx, main_stream,
                    fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
            }

            if (hook) {
                hook(l, main_stream, ForwardHookPoint::AfterAttnOutProjection);
            }

            // 6) Residual + LN2 (fused)
            // LN2 computes: acts.residual_att = residual + att_out, acts.ln2 = rmsnorm(acts.residual_att)
            // Determine abs_max pointer for LN2 (same logic as LN1)
            float* ln2_abs_max_ptr = nullptr;
            if (rs.has_fp8_forward()) {
                ln2_abs_max_ptr = rs.fp8_forward_quants().ln2.abs_max();
            } else if (rs.has_fp4_forward()) {
                ln2_abs_max_ptr = rs.fp4_forward_quants().ln2_global_amax;
            } else if (rs.has_activation_quants()) {
                ln2_abs_max_ptr = q.ln2.abs_max();
            }
            fused_residual_rmsnorm_forward(acts.residual_att, acts.ln2, acts.ln2_rstd,
                                           residual, acts.att_out, weights.ln2.weight,
                                           ln2_abs_max_ptr,
                                           mConfig.RmsNormEps, B * T, C, main_stream);

            // 7) MLP up projection (gate + up fused) - Dense blocks only
            if constexpr (has_mlp_weights<BlockWeights>::value) {
                // In full-block recompute mode, don't persist the large FFN intermediates
                // (mlp_up/swiglu). Keep them stack-backed so they can overlap with other temporaries.
                bool free_ffn_temporaries = false;
                if (mOptions.recompute_block) {
                    // Use the per-layer descriptors, but back them with stack memory for this forward.
                    // This is required for LoRA forward hooks, which mutate `acts.mlp_up` and consume `acts.swiglu`.
                    if (acts.mlp_up.Data == nullptr) rs.temp_acquire(acts.mlp_up);
                    if (acts.swiglu.Data == nullptr) rs.temp_acquire(acts.swiglu);
                    free_ffn_temporaries = true;

                    // MLPUp projection (recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN2) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_up_weight.data;
                            fp4_scales = &fp4_cache.mlp_up_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;  // mlp_up at offset 2
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_up, acts.ln2, weights.mlp_up_weight,
                            nullptr, rs, (int)B, (int)T, (int)C, (int)(2 * D),
                            l, modules::MatmulOp::MLPUp,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
                    }

                    if (hook) {
                        hook(l, main_stream, ForwardHookPoint::AfterMLPUpProjection);
                    }

                    // 8) SwiGLU activation
                    // Determine abs_max pointer for swiglu output (used by FP8/FP4 recipes for MLP down quantization)
                    float* swiglu_abs_max_ptr = nullptr;
                    if (rs.has_fp8_forward()) {
                        swiglu_abs_max_ptr = rs.fp8_forward_quants().swiglu.abs_max();
                    } else if (rs.has_fp4_forward()) {
                        swiglu_abs_max_ptr = rs.fp4_forward_quants().swiglu_global_amax;
                    } else if (rs.has_activation_quants()) {
                        swiglu_abs_max_ptr = q.swiglu.abs_max();
                    }
                    swiglu_forward(acts.swiglu, acts.mlp_up, swiglu_abs_max_ptr, B, T, D, main_stream);

                    // 9) MLP down projection (recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().swiglu : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_down_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_SWIGLU) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_down_weight.data;
                            fp4_scales = &fp4_cache.mlp_down_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 3;  // mlp_down at offset 3
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_down, acts.swiglu, weights.mlp_down_weight,
                            nullptr, rs, (int)B, (int)T, (int)D, (int)C,
                            l, modules::MatmulOp::MLPDown,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
                    }
                } else {
                    // MLPUp projection (non-recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_LN2) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_up_weight.data;
                            fp4_scales = &fp4_cache.mlp_up_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;  // mlp_up at offset 2
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_up, acts.ln2, weights.mlp_up_weight,
                            nullptr, rs, (int)B, (int)T, (int)C, (int)(2 * D),
                            l, modules::MatmulOp::MLPUp,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
                    }

                    if (hook) {
                        hook(l, main_stream, ForwardHookPoint::AfterMLPUpProjection);
                    }

                    // 8) SwiGLU activation
                    // Determine abs_max pointer for swiglu output (used by FP8/FP4 recipes for MLP down quantization)
                    float* swiglu_abs_max_ptr2 = nullptr;
                    if (rs.has_fp8_forward()) {
                        swiglu_abs_max_ptr2 = rs.fp8_forward_quants().swiglu.abs_max();
                    } else if (rs.has_fp4_forward()) {
                        swiglu_abs_max_ptr2 = rs.fp4_forward_quants().swiglu_global_amax;
                    } else if (rs.has_activation_quants()) {
                        swiglu_abs_max_ptr2 = q.swiglu.abs_max();
                    }
                    swiglu_forward(acts.swiglu, acts.mlp_up, swiglu_abs_max_ptr2, B, T, D, main_stream);

                    // 9) MLP down projection (non-recompute path) - recipe handles all format decisions
                    {
                        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().swiglu : nullptr;
                        const Tensor* cached_weight = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_down_weight : nullptr;
                        const int qidx = rs.has_fp8_delayed_scaling() ? get_quantizer_index(l, modules::QuantizerIndex::FWD_SWIGLU) : -1;

                        // FP4 cached weight (CUTLASS layout) for NVFP4Recipe
                        const Tensor* fp4_data = nullptr;
                        const Tensor* fp4_scales = nullptr;
                        const float* fp4_amax = nullptr;
                        if (mWeights->has_fp4_forward_cache()) {
                            auto& fp4_cache = mWeights->fp4_weight_cache();
                            fp4_data = &fp4_cache.mlp_down_weight.data;
                            fp4_scales = &fp4_cache.mlp_down_weight.scales;
                            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 3;  // mlp_down at offset 3
                        }

                        detail::recipe_forward_matmul(
                            *mRecipe, acts.mlp_down, acts.swiglu, weights.mlp_down_weight,
                            nullptr, rs, (int)B, (int)T, (int)D, (int)C,
                            l, modules::MatmulOp::MLPDown,
                            inp_quant, cached_weight, qidx, main_stream,
                            fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
                    }
                }

                if (hook) {
                    hook(l, main_stream, ForwardHookPoint::AfterMLPDownProjection);
                }

                // LoRA's `AfterMLPDownProjection` hook consumes `acts.swiglu`.
                // Keep recompute-block intermediates alive until after the hook runs.
                if (free_ffn_temporaries) {
                    rs.temp_free(acts.swiglu);
                    rs.temp_free(acts.mlp_up);
                }
            } else if constexpr (has_moe_weights<BlockWeights>::value) {
                // MoE block forward pass
                // Full implementation using the tested MoE kernels (moe_kernels.cu)

                const int BT = B * T;
                // MoE config is required for MoE blocks
                assert(mConfig.moe_config.has_value() && "MoE config must be set for MoE blocks");
                const auto& moe_cfg = *mConfig.moe_config;
                const int num_experts = moe_cfg.num_experts;
                const int top_k = moe_cfg.top_k;
                const int expert_D = mConfig.IntermediateSize; // Per-expert intermediate size
                const int total_expert_tokens = BT * top_k;
                const int dev = rs.DeviceId;

                // Create flat view of ln2 for routing: (B, T, C) -> (BT, C)
                Tensor flat_ln2;
                flat_ln2.Data = acts.ln2.Data;
                flat_ln2.DType = acts.ln2.DType;
                flat_ln2.Sizes[0] = BT;
                flat_ln2.Sizes[1] = C;
                flat_ln2.Rank = 2;
                flat_ln2.Device = dev;

                // Allocate temporary tensors for router outputs from stack
                // Router computation always in FP32 for numerical stability
                Tensor router_logits{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                Tensor router_probs{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
                Tensor routing_weights_fp32{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
                Tensor expert_indices{ETensorDType::INT32, {BT, top_k}, nullptr, nullptr, 2, dev};
                Tensor expert_counts{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};

                rs.temp_acquire(router_logits);
                rs.temp_acquire(router_probs);
                rs.temp_acquire(routing_weights_fp32);
                rs.temp_acquire(expert_indices);
                rs.temp_acquire(expert_counts);

                // Zero expert counts before atomic adds
                fill_zero(expert_counts, main_stream);

                // Step 1: Compute routing logits via matmul
                // router_logits = flat_ln2 @ router.gate^T -> (BT, num_experts)
                matmul(
                    router_logits, weights.router.gate, flat_ln2, std::nullopt,
                    nullptr, nullptr,
                    rs.CublasLtHandle, rs.CuBlasWorkspace,
                    num_experts, BT, C, EMMTranspose::TN, false,
                    main_stream
                );

                // Step 2: Softmax over experts
                moe_softmax_forward(
                    router_probs.get<float>(),
                    router_logits.get<float>(),
                    BT, num_experts, main_stream
                );

                // Step 3: Top-K selection with weight normalization
                moe_topk_forward(
                    expert_indices.get<int>(),
                    routing_weights_fp32.get<float>(),
                    router_probs.get<float>(),
                    BT, num_experts, top_k, true, main_stream
                );

                // Step 4: Compute expert token counts for load balancing
                moe_compute_expert_counts(
                    expert_counts.get<int>(),
                    expert_indices.get<int>(),
                    BT, top_k, num_experts, main_stream
                );

                // Step 5: Build gather/scatter indices for token permutation
                // gather_indices: maps permuted position -> original token index
                // scatter_indices: maps original token assignment -> permuted position
                Tensor gather_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};
                Tensor scatter_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};
                rs.temp_acquire(gather_indices);
                rs.temp_acquire(scatter_indices);

                // Build indices using atomic position tracking
                Tensor expert_offsets{ETensorDType::INT32, {num_experts + 1}, nullptr, nullptr, 1, dev};
                Tensor expert_positions{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
                rs.temp_acquire(expert_offsets);
                rs.temp_acquire(expert_positions);
                fill_zero(expert_positions, main_stream);

                // Compute expert offsets (cumsum of counts) - simple kernel or thrust scan
                // For now, use exclusive scan pattern
                moe_compute_expert_offsets(
                    expert_offsets.get<int>(),
                    expert_counts.get<int>(),
                    num_experts, main_stream
                );

                // Build gather/scatter indices
                moe_build_indices(
                    gather_indices.get<int>(),
                    scatter_indices.get<int>(),
                    expert_indices.get<int>(),
                    expert_offsets.get<int>(),
                    expert_positions.get<int>(),
                    BT, top_k, num_experts, main_stream
                );

                // Step 6: Permute tokens to expert-grouped order
                Tensor permuted_input{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(permuted_input);

                if (acts.ln2.DType == ETensorDType::BF16) {
                    moe_permute_tokens(
                        permuted_input.get<nv_bfloat16>(),
                        flat_ln2.get<nv_bfloat16>(),
                        gather_indices.get<int>(),
                        total_expert_tokens, BT, C, top_k, main_stream
                    );
                } else {
                    moe_permute_tokens(
                        permuted_input.get<float>(),
                        flat_ln2.get<float>(),
                        gather_indices.get<int>(),
                        total_expert_tokens, BT, C, top_k, main_stream
                    );
                }

                // Step 7: Expert computation (sequential for now, TODO: grouped GEMM)
                // Each expert processes its assigned tokens with gate+up -> SwiGLU -> down
                Tensor expert_outputs{acts.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_outputs);
                fill_zero(expert_outputs, main_stream);

                // Copy expert_offsets to host for sequential expert dispatch
                // TODO: Replace with grouped GEMM for efficiency
                std::vector<int> h_expert_offsets(num_experts + 1);
                CUDA_CHECK(cudaMemcpyAsync(h_expert_offsets.data(), expert_offsets.get<int>(),
                                           (num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, main_stream));
                CUDA_CHECK(cudaStreamSynchronize(main_stream));

                // Allocate per-expert intermediate buffers
                Tensor expert_gate_up{acts.ln2.DType, {total_expert_tokens, 2 * expert_D}, nullptr, nullptr, 2, dev};
                Tensor expert_swiglu{acts.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_gate_up);
                rs.temp_acquire(expert_swiglu);

                // Sequential expert execution
                for (int e = 0; e < num_experts; ++e) {
                    int start = h_expert_offsets[e];
                    int end = h_expert_offsets[e + 1];
                    int expert_tokens = end - start;
                    if (expert_tokens == 0) continue;

                    // Slice tensors for this expert's tokens
                    Tensor exp_inp = slice(permuted_input, 0, start, end);
                    Tensor exp_gate_up = slice(expert_gate_up, 0, start, end);
                    Tensor exp_swiglu = slice(expert_swiglu, 0, start, end);
                    Tensor exp_out = slice(expert_outputs, 0, start, end);

                    // Get this expert's weights from batched layout
                    // gate_up_proj: (num_experts, 2*D, C) -> slice expert e: (2*D, C)
                    // down_proj: (num_experts, C, D) -> slice expert e: (C, D)
                    Tensor exp_gate_up_w = slice(weights.experts.gate_up_proj, 0, e, e + 1);
                    exp_gate_up_w.Sizes[0] = 2 * expert_D;
                    exp_gate_up_w.Sizes[1] = C;
                    exp_gate_up_w.Rank = 2;

                    Tensor exp_down_w = slice(weights.experts.down_proj, 0, e, e + 1);
                    exp_down_w.Sizes[0] = C;
                    exp_down_w.Sizes[1] = expert_D;
                    exp_down_w.Rank = 2;

                    // Gate+Up projection: exp_gate_up = exp_inp @ exp_gate_up_w^T
                    matmul(
                        exp_gate_up, exp_gate_up_w, exp_inp, std::nullopt,
                        nullptr, nullptr,
                        rs.CublasLtHandle, rs.CuBlasWorkspace,
                        2 * expert_D, expert_tokens, C, EMMTranspose::TN, false,
                        main_stream
                    );

                    // SwiGLU activation
                    swiglu_forward(exp_swiglu, exp_gate_up, nullptr, 1, expert_tokens, expert_D, main_stream);

                    // Down projection: exp_out = exp_swiglu @ exp_down_w^T
                    matmul(
                        exp_out, exp_down_w, exp_swiglu, std::nullopt,
                        nullptr, nullptr,
                        rs.CublasLtHandle, rs.CuBlasWorkspace,
                        C, expert_tokens, expert_D, EMMTranspose::TN, false,
                        main_stream
                    );
                }

                // Step 8: Unpermute and combine expert outputs
                // Output goes directly to acts.mlp_down
                Tensor& mlp_down_out = acts.mlp_down;
                if (acts.ln2.DType == ETensorDType::BF16) {
                    // Convert routing weights to BF16 for the combine kernel
                    Tensor routing_weights_bf16{ETensorDType::BF16, {BT, top_k}, nullptr, nullptr, 2, dev};
                    rs.temp_acquire(routing_weights_bf16);
                    convert_dtype(routing_weights_bf16.get<nv_bfloat16>(),
                                  routing_weights_fp32.get<float>(),
                                  BT * top_k, main_stream);
                    moe_unpermute_and_combine(
                        mlp_down_out.get<nv_bfloat16>(),
                        expert_outputs.get<nv_bfloat16>(),
                        routing_weights_bf16.get<nv_bfloat16>(),
                        scatter_indices.get<int>(),
                        BT, total_expert_tokens, C, top_k, main_stream
                    );
                    rs.temp_free(routing_weights_bf16);
                } else {
                    moe_unpermute_and_combine(
                        mlp_down_out.get<float>(),
                        expert_outputs.get<float>(),
                        routing_weights_fp32.get<float>(),
                        scatter_indices.get<int>(),
                        BT, total_expert_tokens, C, top_k, main_stream
                    );
                }

                // Free temporaries in reverse order
                rs.temp_free(expert_swiglu);
                rs.temp_free(expert_gate_up);
                rs.temp_free(expert_outputs);
                rs.temp_free(permuted_input);
                rs.temp_free(expert_positions);
                rs.temp_free(expert_offsets);
                rs.temp_free(scatter_indices);
                rs.temp_free(gather_indices);
                rs.temp_free(expert_counts);
                rs.temp_free(expert_indices);
                rs.temp_free(routing_weights_fp32);
                rs.temp_free(router_probs);
                rs.temp_free(router_logits);

            } else {
                // Unknown block type - zero output
                fill_zero(acts.mlp_down, main_stream);
            }
        }, main_stream, rs.forward_block_graph(l), use_cuda_graphs,
           rs.Stack, rs.forward_block_stack_checkpoint(l));

        mWeights->release_block(l, main_stream);
        // Offload the residual stream for the previous layer (l-1) once it's been produced.
        // Safe to overlap with subsequent reads of `residual` on the main stream because both are reads.
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

