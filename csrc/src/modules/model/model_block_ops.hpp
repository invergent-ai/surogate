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

    const int B = (int)rs.B;
    const int T = (int)rs.T;
    const int C = (int)mConfig.HiddenSize;
    const int D = (int)mConfig.IntermediateSize;
    const int Hq = (int)mConfig.NumQueryHeads;
    const int Hkv = (int)mConfig.NumKeyValHeads;
    const int Hs = (int)mConfig.head_size();
    const int qkv_channels = (int)mConfig.qkv_channels();

    auto& a = rs.simplified_acts(layer_idx);
    auto& q = rs.simplified_quant_acts(layer_idx);

    // Determine if this layer should use quantization (FP4/FP8)
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_quant_layer = !in_skip_range;

    // recompute dependency rules
    const bool recompute_ln1 = mOptions.recompute_rmsnorm || mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_ln2 = mOptions.recompute_rmsnorm || mOptions.recompute_ffn || mOptions.recompute_block;
    const bool recompute_qkv = mOptions.recompute_qkv || mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_att = mOptions.recompute_attention || mOptions.recompute_block;
    const bool recompute_mlp_up = mOptions.recompute_ffn || mOptions.recompute_block;
    const bool recompute_swiglu = mOptions.recompute_swiglu || mOptions.recompute_ffn || mOptions.recompute_block;

    // Recompute LN1
    if (recompute_ln1) {
        if (rs.has_activation_quants() && q.ln1.DType == ETensorDType::FP8_E4M3) {
            // Use fused kernel: RMSNorm + quantization in one pass
            rmsnorm_forward_quant(q.ln1, q.ln1.scale(), a.ln1_rstd,
                                  residual, weights.ln1.weight, q.ln1.abs_max(),
                                  mConfig.RmsNormEps, B, T, C, stream);
            // Note: a.ln1 not needed in recompute mode, only quantized buffer q.ln1
        } else {
            float* ln1_abs_max_ptr = nullptr;
            if (rs.has_fp4_forward()) {
                ln1_abs_max_ptr = rs.fp4_forward_quants().ln1_global_amax;
            } else if (rs.has_activation_quants()) {
                ln1_abs_max_ptr = q.ln1.abs_max();
            }
            rmsnorm_forward(a.ln1, a.ln1_rstd, residual, weights.ln1.weight,
                            ln1_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, stream);
        }
    }

    // QKV projection + RoPE (needed for attention backward)
    if (recompute_qkv) {
        // Note: During recomputation we DON'T use delayed scaling quantizer indices
        // because we're recomputing for backward pass, not updating scale history
        Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln1 : nullptr;
        const Tensor* cached_qkv = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().qkv_weight : nullptr;

        // FP4 cached weight for recomputation
        const Tensor* fp4_data = nullptr;
        const Tensor* fp4_scales = nullptr;
        const float* fp4_amax = nullptr;
        if (mWeights->has_fp4_forward_cache()) {
            auto& fp4_cache = mWeights->fp4_weight_cache();
            fp4_data = &fp4_cache.qkv_weight.data;
            fp4_scales = &fp4_cache.qkv_weight.scales;
            fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 0;
        }

        detail::recipe_forward_matmul(
            *mRecipe, a.qkv, a.ln1, weights.attention.qkv_weight,
            weights.attention.qkv_bias.has_value() ? &weights.attention.qkv_bias.value() : nullptr,
            rs, B, T, C, qkv_channels,
            layer_idx, modules::MatmulOp::QKV,
            inp_quant, cached_qkv, /*delayed_quantizer_idx=*/-1, stream,
            fp4_data, fp4_scales, fp4_amax, allow_quant_layer);

        using AttentionWeightsType = std::decay_t<decltype(weights.attention)>;
        if constexpr (has_qk_norm_weights<AttentionWeightsType>::value) {
            if (weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value()) {
                const int q_rows = Hq * Hs;
                qkv_head_rmsnorm_forward(
                    a.qkv, a.q_rstd, weights.attention.q_norm_weight.value(),
                    mConfig.RmsNormEps,
                    B, T, qkv_channels,
                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                    stream
                );
                qkv_head_rmsnorm_forward(
                    a.qkv, a.k_rstd, weights.attention.k_norm_weight.value(),
                    mConfig.RmsNormEps,
                    B, T, qkv_channels,
                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                    stream
                );
            }
        }

        // RoPE: operates in-place on a.qkv (can't fuse with quantization here)
        if (mOptions.use_fused_rope) {
            rope_fused_forward(a.qkv, a.qkv, rs.PositionIDs.template get<int>(),
                               nullptr, mConfig.RopeTheta, B, T, Hq, Hkv, Hs, stream);
        } else {
            rope_forward(a.qkv, a.qkv, rs.non_block_activations().freq_cis,
                         rs.PositionIDs.template get<int>(),
                         nullptr, B, T, Hq, Hkv, Hs, stream);
        }
    }

    // Attention forward (FlashAttention): recompute att + lse for backward.
    if (recompute_att) {
        attention_forward_cudnn(a.att, a.lse, a.qkv, rs.CuBlasWorkspace,
                                rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);

        // Compute abs_max for attention output (used by FP8/FP4 recipes for output projection quantization)
        if (rs.has_fp8_forward()) {
            abs_max(rs.fp8_forward_quants().att.abs_max(), a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        } else if (rs.has_fp4_forward()) {
            abs_max(rs.fp4_forward_quants().att_global_amax, a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        } else if (rs.has_activation_quants()) {
            abs_max(q.att.abs_max(), a.att, (long)a.att.nelem(), rs.DeviceProp, stream);
        }

        // only recompute att_out when recomputing the whole block (needed to rebuild residual_att/LN2).
        if (mOptions.recompute_block) {
            // Note: During recomputation we DON'T use delayed scaling quantizer indices
            Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().att : nullptr;
            const Tensor* cached_o = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().o_weight : nullptr;

            // FP4 cached weight for recomputation
            const Tensor* fp4_data = nullptr;
            const Tensor* fp4_scales = nullptr;
            const float* fp4_amax = nullptr;
            if (mWeights->has_fp4_forward_cache()) {
                auto& fp4_cache = mWeights->fp4_weight_cache();
                fp4_data = &fp4_cache.o_weight.data;
                fp4_scales = &fp4_cache.o_weight.scales;
                fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 1;
            }

            detail::recipe_forward_matmul(
                *mRecipe, a.att_out, a.att, weights.attention.out_weight,
                nullptr,
                rs, B, T, Hq * Hs, C,
                layer_idx, modules::MatmulOp::AttnOut,
                inp_quant, cached_o, /*delayed_quantizer_idx=*/-1, stream,
                fp4_data, fp4_scales, fp4_amax, allow_quant_layer);
        }
    }

    // Recompute LN2
    if (recompute_ln2) {
        float* ln2_abs_max_ptr = nullptr;
        if (rs.has_fp4_forward()) {
            ln2_abs_max_ptr = rs.fp4_forward_quants().ln2_global_amax;
        } else if (rs.has_activation_quants()) {
            ln2_abs_max_ptr = q.ln2.abs_max();
        }
        if (mOptions.recompute_block) {
            fused_residual_rmsnorm_forward(
                a.residual_att,
                a.ln2,
                a.ln2_rstd,
                residual,
                a.att_out,
                weights.ln2.weight,
                ln2_abs_max_ptr,
                mConfig.RmsNormEps,
                B * T,
                C,
                stream
            );
        } else {
            rmsnorm_forward(a.ln2, a.ln2_rstd, a.residual_att, weights.ln2.weight,
                            ln2_abs_max_ptr,
                            mConfig.RmsNormEps, B, T, C, stream);
        }
    }

    // Recompute MLP-up (gate+up) if needed.
    if (recompute_mlp_up) {
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (rs.ffn_temps_on_stack()) {
                if (a.mlp_up.Data == nullptr) rs.temp_acquire(a.mlp_up);
                if (a.swiglu.Data == nullptr) rs.temp_acquire(a.swiglu);
            }
            // Note: During recomputation we DON'T use delayed scaling quantizer indices
            Tensor* inp_quant = rs.has_fp8_forward() ? &rs.fp8_forward_quants().ln2 : nullptr;
            const Tensor* cached_mlp_up = mWeights->has_fp8_forward_cache() ? &mWeights->fp8_weight_cache().mlp_up_weight : nullptr;

            // FP4 cached weight for recomputation
            const Tensor* fp4_data = nullptr;
            const Tensor* fp4_scales = nullptr;
            const float* fp4_amax = nullptr;
            if (mWeights->has_fp4_forward_cache()) {
                auto& fp4_cache = mWeights->fp4_weight_cache();
                fp4_data = &fp4_cache.mlp_up_weight.data;
                fp4_scales = &fp4_cache.mlp_up_weight.scales;
                fp4_amax = mWeights->fp4_weight_amax().template get<float>() + 2;
            }

            detail::recipe_forward_matmul(
                *mRecipe, a.mlp_up, a.ln2, weights.mlp_up_weight,
                nullptr,
                rs, B, T, C, 2 * D,
                layer_idx, modules::MatmulOp::MLPUp,
                inp_quant, cached_mlp_up, /*delayed_quantizer_idx=*/-1, stream,
                fp4_data, fp4_scales, fp4_amax, allow_quant_layer);
        }
    }

    // Recompute SwiGLU if needed.
    if (recompute_swiglu) {
        if constexpr (has_mlp_weights<BlockWeights>::value) {
            if (rs.ffn_temps_on_stack()) {
                if (a.swiglu.Data == nullptr) rs.temp_acquire(a.swiglu);
            }
            // Determine abs_max pointer for swiglu output (for FP8 quantization in MLP down)
            float* swiglu_abs_max_ptr = nullptr;
            if (rs.has_fp8_forward()) {
                swiglu_abs_max_ptr = rs.fp8_forward_quants().swiglu.abs_max();
            } else if (rs.has_fp4_forward()) {
                swiglu_abs_max_ptr = rs.fp4_forward_quants().swiglu_global_amax;
            } else if (rs.has_activation_quants()) {
                swiglu_abs_max_ptr = q.swiglu.abs_max();
            }
            swiglu_forward(a.swiglu, a.mlp_up, swiglu_abs_max_ptr, B, T, D, stream);
        }
    }
}

template<typename Block>
void ModularTransformerModel<Block>::backward_block(int layer_idx, bool accumulate, BlockWeights& weights,
                                                     BlockGradients& grads, BlockActivations& acts,
                                                     typename ModularRunState<Block>::BlockGradients& d_acts,
                                                     const BackwardBlockHook* hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    // Dimensions
    const int B = (int)rs.B;
    const int T = (int)rs.T;
    const int C = (int)mConfig.HiddenSize;
    const int D = (int)mConfig.IntermediateSize;
    const int Hq = (int)mConfig.NumQueryHeads;
    const int Hkv = (int)mConfig.NumKeyValHeads;
    const int Hs = (int)mConfig.head_size();
    const int qkv_channels = (int)mConfig.qkv_channels();
    // Determine if this layer should use quantization (FP4/FP8)
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_quant_layer = !in_skip_range;

    // Use the simplified activation/gradient buffers (the full modular per-module
    // activation capture is still being wired up).
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);
    auto& qa = rs.simplified_quant_acts(layer_idx);
    auto& qg = rs.simplified_quant_grads();

    // Keep compilation valid for non-dense block types (MoE/hybrid), even if the
    // simplified backward path isn't implemented for them yet.
    constexpr bool kDenseLike =
        requires(BlockWeights w) {
            w.ln2.weight;
            w.attention.qkv_weight;
            w.attention.out_weight;
            w.mlp_up_weight;
            w.mlp_down_weight;
        } &&
        requires(BlockGradients g) {
            g.ln2_grads.d_weight;
            g.attention_grads.d_qkv_weight;
            g.attention_grads.d_out_weight;
            g.d_mlp_up_weight;
            g.d_mlp_down_weight;
        };

    // Check if this is an MoE block
    constexpr bool kMoELike =
        has_moe_weights<BlockWeights>::value &&
        requires(BlockWeights w) {
            w.ln2.weight;
            w.attention.qkv_weight;
            w.attention.out_weight;
            w.router.gate;
            w.experts.gate_up_proj;
            w.experts.down_proj;
        };

    if constexpr (!kDenseLike && !kMoELike) {
        (void)weights;
        (void)grads;
        (void)acts;
        (void)d_acts;
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward, nullptr);
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
        throw std::runtime_error("ModularTransformerModel::backward_block: simplified backward is not implemented for this block type");
    } else if constexpr (kMoELike) {
        // MoE backward path - LoRA-only mode (no base weight gradients)
        // This path only computes gradient flow for LoRA training, not base model gradients
        backward_block_moe(layer_idx, accumulate, weights, grads, acts, d_acts, hook);
    } else {
        auto with_ctx = [&](const char* stage, const auto& fn) {
            try {
                fn();
            } catch (const std::exception& e) {
                throw std::runtime_error(
                    "ModularTransformerModel::backward_block layer " + std::to_string(layer_idx) + " (" + stage + "): " + e.what());
            }
        };

        // LoRA-only mode: skip computing base weight gradients (only compute dinp for gradient flow)
        const bool lora_only = rs.is_lora_only_mode();

        // Hooks: layer start
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward, nullptr);

        // -------------------- MLP backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward, nullptr);

        // In full-block recompute mode, keep the large FFN backward intermediates stack-backed.
        const bool stack_large_bwd_temps = rs.large_bwd_temps_on_stack();
        Tensor saved_d_mlp_up{};
        if (stack_large_bwd_temps) {
            if (da.d_swiglu.Data == nullptr) rs.temp_acquire(da.d_swiglu);
            // Reuse the (recomputed) mlp_up buffer in-place for d_mlp_up.
            saved_d_mlp_up = da.d_mlp_up;
            da.d_mlp_up = a.mlp_up;
        }

        // MLP down: swiglu -> mlp_down_weight -> d_res_ffn
        with_ctx("mlp_down:qmm", [&]() {
            modules::MatmulContext ctx;
            ctx.dinp = &da.d_swiglu;
            ctx.dweight = &grads.d_mlp_down_weight;
            ctx.dbias = nullptr;
            ctx.dout = &da.d_res_ffn;
            ctx.inp = &a.swiglu;
            ctx.weight = &weights.mlp_down_weight;
            ctx.inp_quant = &qa.swiglu;
            ctx.dout_quant = &qg.d_res_ffn;
            ctx.bias_buffer = nullptr;
            ctx.B = (int)B;
            ctx.T = (int)T;
            ctx.C_in = (int)D;
            ctx.C_out = (int)C;
            ctx.run_state = &rs;
            ctx.stream = stream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::MLPDown;
            ctx.accumulate = accumulate;
            ctx.skip_weight_grad = lora_only;
            ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
            ctx.allow_fp4 = allow_quant_layer;
            ctx.allow_fp8 = allow_quant_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.mlp_down_weight.data;
                ctx.cached_fp4_scales = &fp4_t.mlp_down_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 3;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward, nullptr);

        // SwiGLU backward: d_mlp_up from d_swiglu and saved pre-activation (mlp_up)
        with_ctx("swiglu", [&]() {
            swiglu_backward(da.d_mlp_up, da.d_swiglu, a.mlp_up,
                            rs.has_grad_quants() ? qg.d_mlp_up.abs_max() : nullptr,
                            B, T, D, stream);
        });

        if (stack_large_bwd_temps) {
            rs.temp_free(da.d_swiglu);
            // We no longer need swiglu activations after swiglu_backward.
            if (rs.ffn_temps_on_stack()) {
                rs.temp_free(a.swiglu);
            }
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPUpBackward, nullptr);

        // MLP up: ln2 -> mlp_up_weight -> d_mlp_up
        with_ctx("mlp_up:qmm", [&]() {
            modules::MatmulContext ctx;
            ctx.dinp = &da.d_ln2;
            ctx.dweight = &grads.d_mlp_up_weight;
            ctx.dbias = nullptr;
            ctx.dout = &da.d_mlp_up;
            ctx.inp = &a.ln2;
            ctx.weight = &weights.mlp_up_weight;
            ctx.inp_quant = &qa.ln2;
            ctx.dout_quant = &qg.d_mlp_up;
            ctx.bias_buffer = nullptr;
            ctx.B = (int)B;
            ctx.T = (int)T;
            ctx.C_in = (int)C;
            ctx.C_out = (int)(2 * D);
            ctx.run_state = &rs;
            ctx.stream = stream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::MLPUp;
            ctx.accumulate = accumulate;
            ctx.skip_weight_grad = lora_only;
            ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
            ctx.allow_fp4 = allow_quant_layer;
            ctx.allow_fp8 = allow_quant_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.mlp_up_weight.data;
                ctx.cached_fp4_scales = &fp4_t.mlp_up_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 2;
            }

            mRecipe->backward_matmul(ctx);
        });

        // LoRA's `AfterMLPUpBackward` hook consumes `da.d_mlp_up`. In recompute-block mode
        // we may be temporarily reusing `a.mlp_up` as the gradient buffer, so run the hook
        // before restoring pointers and freeing the activation buffer.
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward, nullptr);

        if (stack_large_bwd_temps) {
            da.d_mlp_up = saved_d_mlp_up;
            if (rs.ffn_temps_on_stack()) {
                rs.temp_free(a.mlp_up);
            }
        }

        // LN2 backward: accumulates residual gradient (d_res_ffn) into d_res_att
        with_ctx("ln2", [&]() {
            rmsnorm_backward(da.d_res_att, grads.ln2_grads.d_weight, rs.scratch().rmsnorm_scratch,
                             da.d_res_ffn, da.d_ln2,
                             a.residual_att, weights.ln2.weight, a.ln2_rstd,
                             rs.has_grad_quants() ? qg.d_res_att.abs_max() : nullptr,
                             B, T, C, rs.DeviceProp, stream,
                             /*skip_weight_grad=*/lora_only);
        });

        // -------------------- Attention backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);

        // Output projection backward
        with_ctx("att_out:qmm", [&]() {
            modules::MatmulContext ctx;
            ctx.dinp = &da.d_att;
            ctx.dweight = &grads.attention_grads.d_out_weight;
            ctx.dbias = nullptr;
            ctx.dout = &da.d_res_att;
            ctx.inp = &a.att;
            ctx.weight = &weights.attention.out_weight;
            ctx.inp_quant = &qa.att;
            ctx.dout_quant = &qg.d_res_att;
            ctx.bias_buffer = nullptr;
            ctx.B = (int)B;
            ctx.T = (int)T;
            ctx.C_in = (int)(Hq * Hs);
            ctx.C_out = (int)C;
            ctx.run_state = &rs;
            ctx.stream = stream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::AttnOut;
            ctx.accumulate = accumulate;
            ctx.skip_weight_grad = lora_only;
            ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
            ctx.allow_fp4 = allow_quant_layer;
            ctx.allow_fp8 = allow_quant_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.o_weight.data;
                ctx.cached_fp4_scales = &fp4_t.o_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 1;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);


	        // FlashAttention backward: produces d_qkv (post-RoPE space)
	        with_ctx("attention_backward", [&]() {
	            if (stack_large_bwd_temps && da.d_qkv.Data == nullptr) {
	                rs.temp_acquire(da.d_qkv);
	            }
	            rs.temp_acquire(rs.scratch().cudnn_workspace);
	            const Tensor& qkv_for_attn = (a.qkv_rope.Data != nullptr) ? a.qkv_rope : a.qkv;
	            const int chunks = mOptions.attention_bwd_chunks;
	            if (chunks < 1) {
	                throw std::runtime_error("attention_bwd_chunks must be >= 1");
	            }
	            if (chunks > 1 && B % chunks != 0) {
	                throw std::runtime_error(fmt::format(
	                    "attn_bwd_chunks ({}) must evenly divide per_device_train_batch_size ({}). "
	                    "Either increase batch size to a multiple of {} or reduce attn_bwd_chunks.",
	                    chunks, B, chunks));
	            }
	            if (chunks == 1) {
	                attention_backward_cudnn(
	                    da.d_qkv,
	                    a.lse,
	                    a.att,
	                    da.d_att,
	                    qkv_for_attn,
	                    rs.scratch().cudnn_workspace,
	                    rs.CudnnHandle,
	                    B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            } else {
	                const long chunk_batch_size = div_exact(static_cast<long>(B), static_cast<long>(chunks));
	                for (int i = 0; i < chunks; ++i) {
	                    Tensor d_qkv = shard_view(da.d_qkv, i, chunks);
	                    Tensor lse = shard_view(a.lse, i, chunks);
	                    Tensor att = shard_view(a.att, i, chunks);
	                    Tensor d_att = shard_view(da.d_att, i, chunks);
	                    Tensor qkv = shard_view(qkv_for_attn, i, chunks);
	                    attention_backward_cudnn(
	                        d_qkv,
	                        lse,
	                        att,
	                        d_att,
                        qkv,
                        rs.scratch().cudnn_workspace,
                        rs.CudnnHandle,
                        chunk_batch_size, T, Hq, Hkv, Hs,
                        stream
                    );
                }
            }
            rs.temp_free(rs.scratch().cudnn_workspace);
        });

	        // RoPE backward
	        with_ctx("rope_backward", [&]() {
	            if (mOptions.use_fused_rope) {
	                rope_fused_backward(
	                    da.d_qkv, da.d_qkv,
	                    rs.PositionIDs.template get<int>(),
	                    rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
	                    mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            } else {
	                rope_backward(
	                    da.d_qkv, da.d_qkv,
	                    rs.non_block_activations().freq_cis,
	                    rs.PositionIDs.template get<int>(),
	                    rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
	                    B, T, Hq, Hkv, Hs,
	                    stream
	                );
	            }
	        });

	        // Optional Q/K head RMSNorm backward (Qwen3-style).
	        using BwdAttentionWeightsType = std::decay_t<decltype(weights.attention)>;
	        if constexpr (has_qk_norm_weights<BwdAttentionWeightsType>::value) {
	            if (weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value()) {
	                with_ctx("qk_norm_backward", [&]() {
	                    // If we didn't keep pre-RoPE QKV, convert activations back to pre-RoPE space.
	                    if (a.qkv_rope.Data == nullptr) {
	                        if (mOptions.use_fused_rope) {
	                            rope_fused_backward(
	                                a.qkv, a.qkv,
	                                rs.PositionIDs.template get<int>(),
	                                nullptr,
	                                mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
	                                stream
	                            );
	                        } else {
	                            rope_backward(
	                                a.qkv, a.qkv,
	                                rs.non_block_activations().freq_cis,
	                                rs.PositionIDs.template get<int>(),
	                                nullptr,
	                                B, T, Hq, Hkv, Hs,
	                                stream
	                            );
	                        }
	                    }

	                    const int q_rows = Hq * Hs;

	                    // Weight gradients must use dy (post-attention backward, pre-qk_norm_backward_dx).
	                    // Skip in LoRA-only mode since QK-norm weights are frozen.
	                    if constexpr (requires { grads.attention_grads.d_q_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
	                        if (!lora_only) {
	                            if (grads.attention_grads.d_q_norm_weight.has_value()) {
	                                qkv_head_rmsnorm_backward_dweight(
	                                    grads.attention_grads.d_q_norm_weight.value(),
	                                    da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(),
	                                    B, T, qkv_channels,
	                                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
	                                    /*accumulate=*/accumulate,
	                                    stream
	                                );
	                            }
	                            if (grads.attention_grads.d_k_norm_weight.has_value()) {
	                                qkv_head_rmsnorm_backward_dweight(
	                                    grads.attention_grads.d_k_norm_weight.value(),
	                                    da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(),
	                                    B, T, qkv_channels,
	                                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
	                                    /*accumulate=*/accumulate,
	                                    stream
	                                );
	                            }
	                        }
	                    }

	                    // Transform dy -> dx in-place.
	                    qkv_head_rmsnorm_backward_dx(
	                        da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(), a.q_rstd,
	                        B, T, qkv_channels,
	                        /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
	                        stream
	                    );
	                    qkv_head_rmsnorm_backward_dx(
	                        da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(), a.k_rstd,
	                        B, T, qkv_channels,
	                        /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
	                        stream
	                    );
	                });
	            }
	        }

        // -------------------- QKV backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward, nullptr);

        with_ctx("qkv:qmm", [&]() {
            std::optional<Tensor> dbias = std::nullopt;
            if (!lora_only && weights.attention.qkv_bias.has_value() && grads.attention_grads.d_qkv_bias.has_value()) {
                dbias = grads.attention_grads.d_qkv_bias.value();
            }
            modules::MatmulContext ctx;
            ctx.dinp = &da.d_ln1;
            ctx.dweight = &grads.attention_grads.d_qkv_weight;
            ctx.dbias = dbias.has_value() ? &dbias.value() : nullptr;
            ctx.dout = &da.d_qkv;
            ctx.inp = &a.ln1;
            ctx.weight = &weights.attention.qkv_weight;
            ctx.inp_quant = &qa.ln1;
            ctx.dout_quant = &qg.d_qkv;
            ctx.bias_buffer = &rs.scratch().matmul_bias_scratch;
            ctx.B = (int)B;
            ctx.T = (int)T;
            ctx.C_in = (int)C;
            ctx.C_out = (int)qkv_channels;
            ctx.run_state = &rs;
            ctx.stream = stream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::QKV;
            ctx.accumulate = accumulate;
            ctx.skip_weight_grad = lora_only;
            ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
            ctx.allow_fp4 = allow_quant_layer;
            ctx.allow_fp8 = allow_quant_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.qkv_weight.data;
                ctx.cached_fp4_scales = &fp4_t.qkv_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 0;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward, nullptr);

        if (stack_large_bwd_temps) {
            rs.temp_free(da.d_qkv);
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
    }
}

template<typename Block>
void ModularTransformerModel<Block>::backward_block_moe(int layer_idx, bool accumulate, BlockWeights& weights,
                                                         BlockGradients& grads, BlockActivations& acts,
                                                         typename ModularRunState<Block>::BlockGradients& d_acts,
                                                         const BackwardBlockHook* hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;

    // Dimensions
    const int B = (int)rs.B;
    const int T = (int)rs.T;
    const int C = (int)mConfig.HiddenSize;
    const int Hq = (int)mConfig.NumQueryHeads;
    const int Hkv = (int)mConfig.NumKeyValHeads;
    const int Hs = (int)mConfig.head_size();
    const int qkv_channels = (int)mConfig.qkv_channels();
    const int BT = B * T;

    // Determine if this layer should use FP4/FP8 quantization (skip ranges match dense path).
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_quant_layer = !in_skip_range;

    // MoE config
    assert(mConfig.moe_config.has_value() && "MoE config must be set for MoE blocks");
    const auto& moe_cfg = *mConfig.moe_config;
    const int num_experts = moe_cfg.num_experts;
    const int top_k = moe_cfg.top_k;
    const int expert_D = moe_cfg.moe_intermediate_size > 0 ? moe_cfg.moe_intermediate_size : (int)mConfig.IntermediateSize;
    const int total_expert_tokens = BT * top_k;
    const int dev = rs.DeviceId;

    // Use the simplified activation/gradient buffers
    auto& a = rs.simplified_acts(layer_idx);
    auto& da = rs.simplified_grads(layer_idx);
    auto& qa = rs.simplified_quant_acts(layer_idx);
    auto& qg = rs.simplified_quant_grads();

    // LoRA-only mode: skip computing base weight gradients (only compute dinp for gradient flow)
    const bool lora_only = rs.is_lora_only_mode();

    // In full-block recompute mode, keep the large backward intermediates stack-backed.
    const bool stack_large_bwd_temps = rs.large_bwd_temps_on_stack();

    auto with_ctx = [&](const char* stage, const auto& fn) {
        try {
            fn();
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "ModularTransformerModel::backward_block_moe layer " + std::to_string(layer_idx) + " (" + stage + "): " + e.what());
        }
    };

    // Hooks: layer start
    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward, nullptr);

    // ================================================================================
    // MLP (MoE) Backward Pass
    // ================================================================================
    // For LoRA-only mode, we need to:
    // 1. Backward through expert down projections -> d_swiglu
    // 2. Backward through SwiGLU -> d_mlp_up (gate_up)
    // 3. Backward through expert up projections -> d_ln2
    // 4. Backward through LN2 -> d_residual_att

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward, nullptr);

    // For MoE backward, we need the expert offsets and indices from forward pass
    // These should be stored in activations (a.moe_*) but for LoRA-only mode,
    // we recompute routing to get the same expert assignments

    // Allocate temporary tensors for MoE backward
    Tensor router_logits{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
    Tensor router_probs{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
    Tensor routing_weights_fp32{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
    Tensor expert_indices{ETensorDType::INT32, {BT, top_k}, nullptr, nullptr, 2, dev};
    Tensor expert_counts{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
    Tensor expert_offsets{ETensorDType::INT32, {num_experts + 1}, nullptr, nullptr, 1, dev};
    Tensor expert_positions{ETensorDType::INT32, {num_experts}, nullptr, nullptr, 1, dev};
    Tensor gather_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};
    Tensor scatter_indices{ETensorDType::INT32, {total_expert_tokens}, nullptr, nullptr, 1, dev};

    rs.temp_acquire(router_logits);
    rs.temp_acquire(router_probs);
    rs.temp_acquire(routing_weights_fp32);
    rs.temp_acquire(expert_indices);
    rs.temp_acquire(expert_counts);
    rs.temp_acquire(expert_offsets);
    rs.temp_acquire(expert_positions);
    rs.temp_acquire(gather_indices);
    rs.temp_acquire(scatter_indices);

    fill_zero(expert_counts, stream);
    fill_zero(expert_positions, stream);

    // Create flat view of ln2
    Tensor flat_ln2;
    flat_ln2.Data = a.ln2.Data;
    flat_ln2.DType = a.ln2.DType;
    flat_ln2.Sizes[0] = BT;
    flat_ln2.Sizes[1] = C;
    flat_ln2.Rank = 2;
    flat_ln2.Device = dev;

    // Recompute routing (same as forward)
    with_ctx("moe_router_recompute", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            matmul(
                router_logits, weights.router.gate, flat_ln2, std::nullopt,
                nullptr, nullptr,
                rs.CublasLtHandle, rs.CuBlasWorkspace,
                num_experts, BT, C, EMMTranspose::TN, false,
                stream
            );

            moe_softmax_forward(
                router_probs.get<float>(),
                router_logits.get<float>(),
                BT, num_experts, stream
            );

            moe_topk_forward(
                expert_indices.get<int>(),
                routing_weights_fp32.get<float>(),
                router_probs.get<float>(),
                BT, num_experts, top_k, true, stream
            );

            moe_compute_expert_counts(
                expert_counts.get<int>(),
                expert_indices.get<int>(),
                BT, top_k, num_experts, stream
            );

            moe_compute_expert_offsets(
                expert_offsets.get<int>(),
                expert_counts.get<int>(),
                num_experts, stream
            );

            moe_build_indices(
                gather_indices.get<int>(),
                scatter_indices.get<int>(),
                expert_indices.get<int>(),
                expert_offsets.get<int>(),
                expert_positions.get<int>(),
                BT, top_k, num_experts, stream
            );
        }
    });

    // Cache expert offsets on host (per layer) to avoid repeated D2H sync in grouped GEMM calls.
    // IMPORTANT: rs.MoeHostExpertOffsets is a single reusable buffer; in multi-layer backward it must be
    // refreshed for each layer, otherwise grouped GEMMs will use stale offsets from another layer and
    // produce incorrect/uninitialized gradients (often manifesting as extreme norm spikes).
    with_ctx("moe_cache_host_offsets", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            rs.MoeHostExpertOffsets.resize(num_experts + 1);
            CUDA_CHECK(cudaMemcpyAsync(rs.MoeHostExpertOffsets.data(), expert_offsets.get<int>(),
                                       (num_experts + 1) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            rs.MoeHostOffsetsValid = true;
        }
    });

    // Allocate expert backward temporaries
    Tensor permuted_input{a.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
    Tensor expert_gate_up{a.ln2.DType, {total_expert_tokens, 2 * expert_D}, nullptr, nullptr, 2, dev};
    Tensor expert_outputs{a.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
    Tensor d_expert_outputs{a.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};
    Tensor d_expert_gate_up{a.ln2.DType, {total_expert_tokens, 2 * expert_D}, nullptr, nullptr, 2, dev};
    Tensor d_permuted_input{a.ln2.DType, {total_expert_tokens, C}, nullptr, nullptr, 2, dev};

    rs.temp_acquire(permuted_input);
    rs.temp_acquire(expert_gate_up);
    rs.temp_acquire(expert_outputs);
    rs.temp_acquire(d_expert_outputs);
    rs.temp_acquire(d_expert_gate_up);
    rs.temp_acquire(d_permuted_input);

    fill_zero(d_permuted_input, stream);

    // Recompute forward pass for expert activations (needed for backward)
    with_ctx("moe_expert_recompute", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            // Permute tokens to expert-grouped order
            if (a.ln2.DType == ETensorDType::BF16) {
                moe_permute_tokens(
                    permuted_input.get<nv_bfloat16>(),
                    flat_ln2.get<nv_bfloat16>(),
                    gather_indices.get<int>(),
                    total_expert_tokens, BT, C, top_k, stream
                );
            } else {
                moe_permute_tokens(
                    permuted_input.get<float>(),
                    flat_ln2.get<float>(),
                    gather_indices.get<int>(),
                    total_expert_tokens, BT, C, top_k, stream
                );
            }

            // Gate+Up projection across all experts
            const int* host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;
            if (a.ln2.DType == ETensorDType::BF16) {
                moe_grouped_gemm_gate_up(
                    expert_gate_up.get<nv_bfloat16>(),
                    permuted_input.get<nv_bfloat16>(),
                    weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                    expert_offsets.get<int>(),
                    num_experts, C, expert_D,
                    rs.CublasHandle, stream, host_offsets
                );
            } else {
                moe_grouped_gemm_gate_up(
                    expert_gate_up.get<float>(),
                    permuted_input.get<float>(),
                    weights.experts.gate_up_proj.template get<float>(),
                    expert_offsets.get<int>(),
                    num_experts, C, expert_D,
                    rs.CublasHandle, stream, host_offsets
                );
            }

            // SwiGLU activation on tokens + down projection
            {
                Tensor expert_swiglu{a.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_swiglu);
                swiglu_forward(expert_swiglu, expert_gate_up, nullptr, 1, total_expert_tokens, expert_D, stream);

                // Down projection for all experts
                if (a.ln2.DType == ETensorDType::BF16) {
                    moe_grouped_gemm_down(
                        expert_outputs.template get<nv_bfloat16>(),
                        expert_swiglu.template get<nv_bfloat16>(),
                        weights.experts.down_proj.template get<nv_bfloat16>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                } else {
                    moe_grouped_gemm_down(
                        expert_outputs.template get<float>(),
                        expert_swiglu.template get<float>(),
                        weights.experts.down_proj.template get<float>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                }
                rs.temp_free(expert_swiglu);
            }
        }
    });

    // Backward through combine (unpermute + weight)
    // Keep routing weight gradients in FP32 for router backward (routing is computed in FP32).
    Tensor d_routing_weights_fp32{ETensorDType::FP32, {BT, top_k}, nullptr, nullptr, 2, dev};
    rs.temp_acquire(d_routing_weights_fp32);
    with_ctx("moe_combine_backward", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            // d_res_ffn is the gradient from the residual connection
            // We need to backward through the combine operation

            if (da.d_res_ffn.DType == ETensorDType::BF16) {
                // For BF16, routing weight gradients are also BF16
                Tensor d_routing_weights{ETensorDType::BF16, {BT, top_k}, nullptr, nullptr, 2, dev};
                Tensor routing_weights_bf16{ETensorDType::BF16, {BT, top_k}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(d_routing_weights);
                rs.temp_acquire(routing_weights_bf16);

                // Convert routing weights to BF16
                convert_dtype(routing_weights_bf16.get<nv_bfloat16>(),
                              routing_weights_fp32.get<float>(),
                              BT * top_k, stream);

                Tensor d_res_ffn_bf16 = da.d_res_ffn;
                d_res_ffn_bf16.Sizes[0] = BT;
                d_res_ffn_bf16.Sizes[1] = C;
                d_res_ffn_bf16.Rank = 2;

                moe_combine_backward(
                    d_expert_outputs.get<nv_bfloat16>(),
                    d_routing_weights.get<nv_bfloat16>(),
                    d_res_ffn_bf16.get<nv_bfloat16>(),
                    expert_outputs.get<nv_bfloat16>(),
                    routing_weights_bf16.get<nv_bfloat16>(),
                    scatter_indices.get<int>(),
                    BT, total_expert_tokens, C, top_k, stream
                );

                // Convert d_routing_weights to FP32 for router backward
                convert_dtype(
                    d_routing_weights_fp32.get<float>(),
                    d_routing_weights.get<nv_bfloat16>(),
                    BT * top_k,
                    stream
                );

                rs.temp_free(routing_weights_bf16);
                rs.temp_free(d_routing_weights);
            } else {
                // For FP32
                Tensor d_res_ffn_fp32 = da.d_res_ffn;
                d_res_ffn_fp32.Sizes[0] = BT;
                d_res_ffn_fp32.Sizes[1] = C;
                d_res_ffn_fp32.Rank = 2;

                moe_combine_backward(
                    d_expert_outputs.get<float>(),
                    d_routing_weights_fp32.get<float>(),
                    d_res_ffn_fp32.get<float>(),
                    expert_outputs.get<float>(),
                    routing_weights_fp32.get<float>(),
                    scatter_indices.get<int>(),
                    BT, total_expert_tokens, C, top_k, stream
                );
            }
        }
    });

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward, nullptr);

    // Backward through each expert using grouped GEMM
    bool hook_handled = false;
    if (hook) {
        MoEGroupedContext moe_ctx;
        moe_ctx.expert_offsets = &expert_offsets;
        moe_ctx.permuted_input = &permuted_input;
        moe_ctx.expert_gate_up = &expert_gate_up;
        moe_ctx.expert_outputs = &expert_outputs;
        moe_ctx.d_expert_outputs = &d_expert_outputs;
        moe_ctx.d_expert_gate_up = &d_expert_gate_up;
        moe_ctx.d_permuted_input = &d_permuted_input;
        moe_ctx.host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;
        moe_ctx.num_experts = num_experts;
        moe_ctx.top_k = top_k;
        moe_ctx.total_tokens = total_expert_tokens;

        (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::MoEExpertGroupManual, &moe_ctx);
        hook_handled = moe_ctx.handled;
    }

    if (!hook_handled) {
        with_ctx("moe_expert_backward", [&]() {
            if constexpr (has_moe_weights<BlockWeights>::value) {
                // Use cached host offsets if available
                const int* host_offsets = rs.MoeHostOffsetsValid ? rs.MoeHostExpertOffsets.data() : nullptr;

                // Expert down_proj weight gradients require the SwiGLU activations.
                // Keep this out of the main backward path in LoRA-only mode.
                if (!lora_only) {
                    if constexpr (requires { grads.experts.d_down_proj; }) {
                        if (grads.experts.d_down_proj.Data && grads.experts.d_down_proj.nelem() > 0) {
                            Tensor expert_swiglu{a.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                            rs.temp_acquire(expert_swiglu);
                            swiglu_forward(expert_swiglu, expert_gate_up, nullptr, 1, total_expert_tokens, expert_D, stream);

                            const float beta = accumulate ? 1.0f : 0.0f;
                            if (a.ln2.DType == ETensorDType::BF16) {
                                moe_grouped_gemm_weight_grad(
                                    grads.experts.d_down_proj.template get<nv_bfloat16>(),
                                    d_expert_outputs.get<nv_bfloat16>(),
                                    expert_swiglu.get<nv_bfloat16>(),
                                    expert_offsets.get<int>(),
                                    num_experts,
                                    /*M=*/C,
                                    /*N=*/expert_D,
                                    rs.CublasHandle,
                                    stream,
                                    host_offsets,
                                    /*alpha=*/1.0f,
                                    beta
                                );
                            } else {
                                moe_grouped_gemm_weight_grad(
                                    grads.experts.d_down_proj.template get<float>(),
                                    d_expert_outputs.get<float>(),
                                    expert_swiglu.get<float>(),
                                    expert_offsets.get<int>(),
                                    num_experts,
                                    /*M=*/C,
                                    /*N=*/expert_D,
                                    rs.CublasHandle,
                                    stream,
                                    host_offsets,
                                    /*alpha=*/1.0f,
                                    beta
                                );
                            }

                            rs.temp_free(expert_swiglu);
                        }
                    }
                }

                Tensor d_expert_swiglu{a.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(d_expert_swiglu);

                // Backward through down projection for all experts
                // d_expert_swiglu = d_expert_outputs @ down_proj (no transpose)
                if (a.ln2.DType == ETensorDType::BF16) {
                    moe_grouped_gemm_down_backward(
                        d_expert_swiglu.template get<nv_bfloat16>(),
                        d_expert_outputs.template get<nv_bfloat16>(),
                        weights.experts.down_proj.template get<nv_bfloat16>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                } else {
                    moe_grouped_gemm_down_backward(
                        d_expert_swiglu.template get<float>(),
                        d_expert_outputs.template get<float>(),
                        weights.experts.down_proj.template get<float>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                }

                // Backward through SwiGLU activation
                // BnB weight loading stores MoE expert weights as [up | gate] per row:
                //   - up_proj in first half (rows 0 to D-1)
                //   - gate_proj in second half (rows D to 2D-1)
                // Forward was: h = silu(gate) * up
                // split_gate_up: first output gets cols [0,D), second gets cols [D,2D)
                // So: expert_up = first half = up, expert_gate = second half = gate
                Tensor expert_up{a.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                Tensor expert_gate{a.ln2.DType, {total_expert_tokens, expert_D}, nullptr, nullptr, 2, dev};
                rs.temp_acquire(expert_up);
                rs.temp_acquire(expert_gate);

                // expert_up = first half = up weights, expert_gate = second half = gate weights
                split_gate_up(expert_gate_up, expert_up, expert_gate, total_expert_tokens, expert_D, stream);

                // silu_mul_backward_inplace: forward was h = silu(gate) * up
                // silu_mul_backward_inplace(e, g, dh, h) expects e=what silu was applied to (gate), g=what was multiplied (up)
                // After this: expert_gate = d_gate, expert_up = d_up
                Tensor expert_h;
                expert_h.Data = d_expert_gate_up.Data;  // Reuse d_expert_gate_up buffer for h
                expert_h.DType = a.ln2.DType;
                expert_h.Sizes = {total_expert_tokens, expert_D};
                silu_mul_backward_inplace(expert_gate, expert_up, d_expert_swiglu, &expert_h, total_expert_tokens, expert_D, stream);

                // Concatenate gradients: need [d_up | d_gate] to match BnB layout [up | gate]
                // concat_d_gate_up puts first arg at cols [0,D), second at cols [D,2D)
                // So: first = d_up (expert_up), second = d_gate (expert_gate)
                concat_d_gate_up(expert_up, expert_gate, d_expert_gate_up, total_expert_tokens, expert_D, stream);

                rs.temp_free(expert_gate);
                rs.temp_free(expert_up);
                rs.temp_free(d_expert_swiglu);

                // Expert gate_up_proj weight gradient (before computing d_permuted_input).
                if (!lora_only) {
                    if constexpr (requires { grads.experts.d_gate_up_proj; }) {
                        if (grads.experts.d_gate_up_proj.Data && grads.experts.d_gate_up_proj.nelem() > 0) {
                            const float beta = accumulate ? 1.0f : 0.0f;
                            if (a.ln2.DType == ETensorDType::BF16) {
                                moe_grouped_gemm_weight_grad(
                                    grads.experts.d_gate_up_proj.template get<nv_bfloat16>(),
                                    d_expert_gate_up.get<nv_bfloat16>(),
                                    permuted_input.get<nv_bfloat16>(),
                                    expert_offsets.get<int>(),
                                    num_experts,
                                    /*M=*/2 * expert_D,
                                    /*N=*/C,
                                    rs.CublasHandle,
                                    stream,
                                    host_offsets,
                                    /*alpha=*/1.0f,
                                    beta
                                );
                            } else {
                                moe_grouped_gemm_weight_grad(
                                    grads.experts.d_gate_up_proj.template get<float>(),
                                    d_expert_gate_up.get<float>(),
                                    permuted_input.get<float>(),
                                    expert_offsets.get<int>(),
                                    num_experts,
                                    /*M=*/2 * expert_D,
                                    /*N=*/C,
                                    rs.CublasHandle,
                                    stream,
                                    host_offsets,
                                    /*alpha=*/1.0f,
                                    beta
                                );
                            }
                        }
                    }
                }

                // Backward through gate+up projection for all experts
                // d_permuted_input = d_expert_gate_up @ gate_up_proj (no transpose)
                if (a.ln2.DType == ETensorDType::BF16) {
                    moe_grouped_gemm_gate_up_backward(
                        d_permuted_input.template get<nv_bfloat16>(),
                        d_expert_gate_up.template get<nv_bfloat16>(),
                        weights.experts.gate_up_proj.template get<nv_bfloat16>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                } else {
                    moe_grouped_gemm_gate_up_backward(
                        d_permuted_input.template get<float>(),
                        d_expert_gate_up.template get<float>(),
                        weights.experts.gate_up_proj.template get<float>(),
                        expert_offsets.template get<int>(),
                        num_experts, C, expert_D,
                        rs.CublasHandle, stream, host_offsets
                    );
                }
            }
        });
    }

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward, nullptr);

    // Backward through permutation: scatter d_permuted_input back to d_ln2
    with_ctx("moe_permute_backward", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            // moe_permute_backward uses atomicAdd into d_ln2, so we must start from a zero buffer.
            // Dense blocks overwrite d_ln2 via matmul backward, but MoE uses scatter-add for top-k.
            fill_zero(da.d_ln2, stream);

            if (da.d_ln2.DType == ETensorDType::BF16) {
                Tensor d_ln2_flat = da.d_ln2;
                d_ln2_flat.Sizes[0] = BT;
                d_ln2_flat.Sizes[1] = C;
                d_ln2_flat.Rank = 2;

                moe_permute_backward(
                    d_ln2_flat.get<nv_bfloat16>(),
                    d_permuted_input.get<nv_bfloat16>(),
                    gather_indices.get<int>(),
                    total_expert_tokens, BT, C, top_k, stream
                );
            } else {
                Tensor d_ln2_flat = da.d_ln2;
                d_ln2_flat.Sizes[0] = BT;
                d_ln2_flat.Sizes[1] = C;
                d_ln2_flat.Rank = 2;

                moe_permute_backward(
                    d_ln2_flat.get<float>(),
                    d_permuted_input.get<float>(),
                    gather_indices.get<int>(),
                    total_expert_tokens, BT, C, top_k, stream
                );
            }
        }
    });

    // Router backward: propagate gradients through top-k normalization and softmax back into LN2,
    // and (if training base weights) compute router gate weight gradients.
    with_ctx("moe_router_backward", [&]() {
        if constexpr (has_moe_weights<BlockWeights>::value) {
            // d_probs is sparse (only selected experts have non-zero entries), but represented as a dense (BT, E) buffer.
            Tensor d_probs{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
            Tensor d_logits{ETensorDType::FP32, {BT, num_experts}, nullptr, nullptr, 2, dev};
            rs.temp_acquire(d_probs);
            rs.temp_acquire(d_logits);

            fill_zero(d_probs, stream);

            // Top-k backward (treat indices as constants): d_probs from d_routing_weights.
            moe_topk_backward(
                d_probs.get<float>(),
                d_routing_weights_fp32.get<float>(),
                router_probs.get<float>(),
                expert_indices.get<int>(),
                BT, num_experts, top_k,
                /*normalize_weights=*/true,
                stream
            );

            // Softmax backward: d_logits from d_probs.
            moe_softmax_backward(
                d_logits.get<float>(),
                d_probs.get<float>(),
                router_probs.get<float>(),
                BT, num_experts,
                stream
            );

            // Convert d_logits to BF16 for matmuls against BF16 weights/activations.
            Tensor d_logits_bf16{ETensorDType::BF16, {BT, num_experts}, nullptr, nullptr, 2, dev};
            rs.temp_acquire(d_logits_bf16);
            convert_dtype(
                d_logits_bf16.get<nv_bfloat16>(),
                d_logits.get<float>(),
                BT * num_experts,
                stream
            );

            // d_ln2 += d_logits @ router.gate
            // In column-major view: d_ln2^T(C, BT) = router.gate^T(C, E) @ d_logits^T(E, BT)
            Tensor d_ln2_flat = da.d_ln2;
            d_ln2_flat.Sizes[0] = BT;
            d_ln2_flat.Sizes[1] = C;
            d_ln2_flat.Rank = 2;
            matmul(
                d_ln2_flat,
                weights.router.gate,
                d_logits_bf16,
                std::nullopt,
                nullptr, nullptr,
                rs.CublasLtHandle, rs.CuBlasWorkspace,
                C, BT, num_experts,
                EMMTranspose::NN,
                /*accumulate=*/true,
                stream
            );

            // d_router_gate (weight grad): dW = d_logits^T @ ln2  (E, C)
            // In lora_only mode, skip base weight gradients - but if train_router is enabled,
            // we still need to compute the router gradient even in lora_only mode.
            if (!lora_only || rs.is_train_router()) {
                if constexpr (requires { grads.router.d_gate; }) {
                    if (grads.router.d_gate.Data && grads.router.d_gate.nelem() > 0) {
                        matmul(
                            grads.router.d_gate,
                            flat_ln2,
                            d_logits_bf16,
                            std::nullopt,
                            nullptr, nullptr,
                            rs.CublasLtHandle, rs.CuBlasWorkspace,
                            C, num_experts, BT,
                            EMMTranspose::NT,
                            /*accumulate=*/accumulate,
                            stream
                        );
                    }
                }
            }

            rs.temp_free(d_logits_bf16);
            rs.temp_free(d_logits);
            rs.temp_free(d_probs);
        }
    });

    // Free expert temporaries
    rs.temp_free(d_routing_weights_fp32);
    rs.temp_free(d_permuted_input);
    rs.temp_free(d_expert_gate_up);
    rs.temp_free(d_expert_outputs);
    rs.temp_free(expert_outputs);
    rs.temp_free(expert_gate_up);
    rs.temp_free(permuted_input);
    rs.temp_free(scatter_indices);
    rs.temp_free(gather_indices);
    rs.temp_free(expert_positions);
    rs.temp_free(expert_offsets);
    rs.temp_free(expert_counts);
    rs.temp_free(expert_indices);
    rs.temp_free(routing_weights_fp32);
    rs.temp_free(router_probs);
    rs.temp_free(router_logits);

    // LN2 backward: accumulates residual gradient (d_res_ffn) into d_res_att
    with_ctx("ln2", [&]() {
        // MoE blocks use grads.ln2.d_weight, not grads.ln2_grads.d_weight
        rmsnorm_backward(da.d_res_att, grads.ln2.d_weight, rs.scratch().rmsnorm_scratch,
                         da.d_res_ffn, da.d_ln2,
                         a.residual_att, weights.ln2.weight, a.ln2_rstd,
                         nullptr,  // No quant for MoE path yet
                         B, T, C, rs.DeviceProp, stream,
                         /*skip_weight_grad=*/lora_only);
    });

    // ================================================================================
    // Attention Backward Pass (same as dense)
    // ================================================================================

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward, nullptr);

    // Output projection backward
    with_ctx("att_out:qmm", [&]() {
        Tensor* d_out_weight = nullptr;
        if (!lora_only) {
            if constexpr (requires { grads.attention_grads.d_out_weight; }) {
                d_out_weight = &grads.attention_grads.d_out_weight;
            } else if constexpr (requires { grads.attention.d_out_weight; }) {
                d_out_weight = &grads.attention.d_out_weight;
            }
        }

        modules::MatmulContext ctx;
        ctx.dinp = &da.d_att;
        ctx.dweight = d_out_weight;
        ctx.dbias = nullptr;
        ctx.dout = &da.d_res_att;
        ctx.inp = &a.att;
        ctx.weight = &weights.attention.out_weight;
        ctx.inp_quant = &qa.att;
        ctx.dout_quant = &qg.d_res_att;
        ctx.bias_buffer = nullptr;
        ctx.B = (int)B;
        ctx.T = (int)T;
        ctx.C_in = (int)(Hq * Hs);
        ctx.C_out = (int)C;
        ctx.run_state = &rs;
        ctx.stream = stream;
        ctx.layer_idx = layer_idx;
        ctx.op = modules::MatmulOp::AttnOut;
        ctx.accumulate = accumulate;
        ctx.skip_weight_grad = lora_only;
        ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
        ctx.allow_fp4 = allow_quant_layer;
        ctx.allow_fp8 = allow_quant_layer;

        // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
        if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
            auto& fp4_t = mWeights->fp4_weight_cache_transposed();
            ctx.cached_fp4_data = &fp4_t.o_weight.data;
            ctx.cached_fp4_scales = &fp4_t.o_weight.scales;
            ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 1;
        }

        mRecipe->backward_matmul(ctx);
    });

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward, nullptr);

    // FlashAttention backward
    with_ctx("attention_backward", [&]() {
        // Acquire d_qkv from stack if needed (when large_bwd_temps_on_stack is true)
        if (stack_large_bwd_temps && da.d_qkv.Data == nullptr) {
            rs.temp_acquire(da.d_qkv);
        }
        rs.temp_acquire(rs.scratch().cudnn_workspace);
        const Tensor& qkv_for_attn = (a.qkv_rope.Data != nullptr) ? a.qkv_rope : a.qkv;

        const int chunks = mOptions.attention_bwd_chunks;
        if (chunks < 1) {
            throw std::runtime_error("attention_bwd_chunks must be >= 1");
        }
        if (chunks > 1 && B % chunks != 0) {
            throw std::runtime_error(fmt::format(
                "attn_bwd_chunks ({}) must evenly divide per_device_train_batch_size ({}). "
                "Either increase batch size to a multiple of {} or reduce attn_bwd_chunks.",
                chunks, B, chunks));
        }

        if (chunks == 1) {
            attention_backward_cudnn(
                da.d_qkv,
                a.lse,
                a.att,
                da.d_att,
                qkv_for_attn,
                rs.scratch().cudnn_workspace,
                rs.CudnnHandle,
                B, T, Hq, Hkv, Hs,
                stream
            );
        } else {
            const long chunk_batch_size = div_exact(static_cast<long>(B), static_cast<long>(chunks));
            for (int i = 0; i < chunks; ++i) {
                Tensor d_qkv = shard_view(da.d_qkv, i, chunks);
                Tensor lse = shard_view(a.lse, i, chunks);
                Tensor att = shard_view(a.att, i, chunks);
                Tensor d_att = shard_view(da.d_att, i, chunks);
                Tensor qkv = shard_view(qkv_for_attn, i, chunks);
                attention_backward_cudnn(
                    d_qkv,
                    lse,
                    att,
                    d_att,
                    qkv,
                    rs.scratch().cudnn_workspace,
                    rs.CudnnHandle,
                    chunk_batch_size, T, Hq, Hkv, Hs,
                    stream
                );
            }
        }
        rs.temp_free(rs.scratch().cudnn_workspace);
    });

    // RoPE backward: produces d_qkv in pre-RoPE space.
    with_ctx("rope_backward", [&]() {
        if (mOptions.use_fused_rope) {
            rope_fused_backward(
                da.d_qkv, da.d_qkv,
                rs.PositionIDs.template get<int>(),
                rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
                mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
                stream
            );
        } else {
            rope_backward(
                da.d_qkv, da.d_qkv,
                rs.non_block_activations().freq_cis,
                rs.PositionIDs.template get<int>(),
                rs.has_grad_quants() ? qg.d_qkv.abs_max() : nullptr,
                B, T, Hq, Hkv, Hs,
                stream
            );
        }
    });

    // Optional Q/K head RMSNorm backward (Qwen3-style).
    using BwdAttentionWeightsType = std::decay_t<decltype(weights.attention)>;
    if constexpr (has_qk_norm_weights<BwdAttentionWeightsType>::value) {
        if (weights.attention.q_norm_weight.has_value() && weights.attention.k_norm_weight.has_value()) {
            with_ctx("qk_norm_backward", [&]() {
                // If we didn't keep pre-RoPE QKV, convert activations back to pre-RoPE space.
                if (a.qkv_rope.Data == nullptr) {
                    if (mOptions.use_fused_rope) {
                        rope_fused_backward(
                            a.qkv, a.qkv,
                            rs.PositionIDs.template get<int>(),
                            nullptr,
                            mConfig.RopeTheta, B, T, Hq, Hkv, Hs,
                            stream
                        );
                    } else {
                        rope_backward(
                            a.qkv, a.qkv,
                            rs.non_block_activations().freq_cis,
                            rs.PositionIDs.template get<int>(),
                            nullptr,
                            B, T, Hq, Hkv, Hs,
                            stream
                        );
                    }
                }

                const int q_rows = Hq * Hs;

                // Weight gradients must use dy (post-attention backward, pre-qk_norm_backward_dx).
                // Skip in LoRA-only mode since QK-norm weights are frozen.
                if (!lora_only) {
                    if constexpr (requires { grads.attention_grads.d_q_norm_weight; grads.attention_grads.d_k_norm_weight; }) {
                        if (grads.attention_grads.d_q_norm_weight.has_value()) {
                            qkv_head_rmsnorm_backward_dweight(
                                grads.attention_grads.d_q_norm_weight.value(),
                                da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(),
                                B, T, qkv_channels,
                                /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                                /*accumulate=*/accumulate,
                                stream
                            );
                        }
                        if (grads.attention_grads.d_k_norm_weight.has_value()) {
                            qkv_head_rmsnorm_backward_dweight(
                                grads.attention_grads.d_k_norm_weight.value(),
                                da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(),
                                B, T, qkv_channels,
                                /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                                /*accumulate=*/accumulate,
                                stream
                            );
                        }
                    } else if constexpr (requires { grads.attention.d_q_norm_weight; grads.attention.d_k_norm_weight; }) {
                        if (grads.attention.d_q_norm_weight.has_value()) {
                            qkv_head_rmsnorm_backward_dweight(
                                grads.attention.d_q_norm_weight.value(),
                                da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(),
                                B, T, qkv_channels,
                                /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                                /*accumulate=*/accumulate,
                                stream
                            );
                        }
                        if (grads.attention.d_k_norm_weight.has_value()) {
                            qkv_head_rmsnorm_backward_dweight(
                                grads.attention.d_k_norm_weight.value(),
                                da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(),
                                B, T, qkv_channels,
                                /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                                /*accumulate=*/accumulate,
                                stream
                            );
                        }
                    }
                }

                // Transform dy -> dx in-place.
                qkv_head_rmsnorm_backward_dx(
                    da.d_qkv, a.qkv, weights.attention.q_norm_weight.value(), a.q_rstd,
                    B, T, qkv_channels,
                    /*num_heads=*/Hq, /*head_size=*/Hs, /*channel_offset=*/0,
                    stream
                );
                qkv_head_rmsnorm_backward_dx(
                    da.d_qkv, a.qkv, weights.attention.k_norm_weight.value(), a.k_rstd,
                    B, T, qkv_channels,
                    /*num_heads=*/Hkv, /*head_size=*/Hs, /*channel_offset=*/q_rows,
                    stream
                );
            });
        }
    }

    // QKV projection backward
    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward, nullptr);

    with_ctx("qkv:qmm", [&]() {
        Tensor* d_qkv_weight = nullptr;
        Tensor* d_qkv_bias = nullptr;
        std::optional<Tensor> dbias_tmp = std::nullopt;

        if (!lora_only) {
            if constexpr (requires { grads.attention_grads.d_qkv_weight; }) {
                d_qkv_weight = &grads.attention_grads.d_qkv_weight;
                if constexpr (requires { grads.attention_grads.d_qkv_bias; }) {
                    if (weights.attention.qkv_bias.has_value() && grads.attention_grads.d_qkv_bias.has_value()) {
                        dbias_tmp = grads.attention_grads.d_qkv_bias.value();
                        d_qkv_bias = &dbias_tmp.value();
                    }
                }
            } else if constexpr (requires { grads.attention.d_qkv_weight; }) {
                d_qkv_weight = &grads.attention.d_qkv_weight;
                if constexpr (requires { grads.attention.d_qkv_bias; }) {
                    if (weights.attention.qkv_bias.has_value() && grads.attention.d_qkv_bias.has_value()) {
                        dbias_tmp = grads.attention.d_qkv_bias.value();
                        d_qkv_bias = &dbias_tmp.value();
                    }
                }
            }
        }

        modules::MatmulContext ctx;
        ctx.dinp = &da.d_ln1;
        ctx.dweight = d_qkv_weight;
        ctx.dbias = d_qkv_bias;
        ctx.dout = &da.d_qkv;
        ctx.inp = &a.ln1;
        ctx.weight = &weights.attention.qkv_weight;
        ctx.inp_quant = &qa.ln1;
        ctx.dout_quant = &qg.d_qkv;
        ctx.bias_buffer = &rs.scratch().matmul_bias_scratch;
        ctx.B = (int)B;
        ctx.T = (int)T;
        ctx.C_in = (int)C;
        ctx.C_out = (int)qkv_channels;
        ctx.run_state = &rs;
        ctx.stream = stream;
        ctx.layer_idx = layer_idx;
        ctx.op = modules::MatmulOp::QKV;
        ctx.accumulate = accumulate;
        ctx.skip_weight_grad = lora_only;
        ctx.seed = static_cast<unsigned int>(mOptimizerRNG());
        ctx.allow_fp4 = allow_quant_layer;
        ctx.allow_fp8 = allow_quant_layer;

        // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
        if (mWeights->has_fp4_dgrad_cache() && allow_quant_layer) {
            auto& fp4_t = mWeights->fp4_weight_cache_transposed();
            ctx.cached_fp4_data = &fp4_t.qkv_weight.data;
            ctx.cached_fp4_scales = &fp4_t.qkv_weight.scales;
            ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 0;
        }

        mRecipe->backward_matmul(ctx);
    });

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward, nullptr);

    // Free d_qkv from stack if it was acquired
    if (stack_large_bwd_temps) {
        rs.temp_free(da.d_qkv);
    }

    // Note: LN1 backward is handled in the main backward loop (backward_with_hook)
    // because it needs access to prev_da.d_res_ffn (previous layer's gradient buffer)

    if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward, nullptr);
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
