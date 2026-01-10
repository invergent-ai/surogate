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

    // Determine if this layer should use FP4 quantization
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_fp4_layer = !in_skip_range;

    // Match legacy recompute dependency rules (LLamaModel::_recompute_block).
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
            fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);

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

        // Match legacy: only recompute att_out when recomputing the whole block (needed to rebuild residual_att/LN2).
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
                fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
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
                fp4_data, fp4_scales, fp4_amax, allow_fp4_layer);
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
    // Determine if this layer should use FP4 quantization
    const int skip_first = std::max(0, mOptions.skip_quant_first_layers);
    const int skip_last = std::max(0, mOptions.skip_quant_last_layers);
    const bool in_skip_range = (layer_idx < skip_first) || (layer_idx >= mConfig.NumLayers - skip_last);
    const bool allow_fp4_layer = !in_skip_range;

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

    if constexpr (!kDenseLike) {
        (void)weights;
        (void)grads;
        (void)acts;
        (void)d_acts;
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward);
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward);
        throw std::runtime_error("ModularTransformerModel::backward_block: simplified backward is not implemented for this block type");
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
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeLayerBackward);

        // -------------------- MLP backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPDownBackward);

        // In full-block recompute mode, keep the large FFN backward intermediates stack-backed.
        const bool stack_large_bwd_temps = rs.large_bwd_temps_on_stack();
        Tensor saved_d_mlp_up{};
        if (stack_large_bwd_temps) {
            if (da.d_swiglu.Data == nullptr) rs.temp_acquire(da.d_swiglu);
            // Reuse the (recomputed) mlp_up buffer in-place for d_mlp_up (matches legacy).
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
            ctx.allow_fp4 = allow_fp4_layer;
            ctx.allow_fp8 = allow_fp4_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_fp4_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.mlp_down_weight.data;
                ctx.cached_fp4_scales = &fp4_t.mlp_down_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 3;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPDownBackward);

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

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeMLPUpBackward);

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
            ctx.allow_fp4 = allow_fp4_layer;
            ctx.allow_fp8 = allow_fp4_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_fp4_layer) {
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
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterMLPUpBackward);

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
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeAttnOutBackward);

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
            ctx.allow_fp4 = allow_fp4_layer;
            ctx.allow_fp8 = allow_fp4_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_fp4_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.o_weight.data;
                ctx.cached_fp4_scales = &fp4_t.o_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 1;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterAttnOutBackward);

	        // FlashAttention backward: produces d_qkv (post-RoPE space)
	        with_ctx("attention_backward", [&]() {
	            if (stack_large_bwd_temps && da.d_qkv.Data == nullptr) {
	                rs.temp_acquire(da.d_qkv);
	            }
	            rs.temp_acquire(rs.scratch().cudnn_workspace);
	            const Tensor& qkv_for_attn = (a.qkv_rope.Data != nullptr) ? a.qkv_rope : a.qkv;
	            const int chunks = mOptions.attention_bwd_chunks;
	            if (chunks < 1) {
	                throw std::invalid_argument("attention_bwd_chunks must be >= 1");
	            }
	            if (chunks > 1 && B % chunks != 0) {
	                throw std::invalid_argument(fmt::format(
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

        // -------------------- QKV backward --------------------
        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::BeforeQKVBackward);

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
            ctx.allow_fp4 = allow_fp4_layer;
            ctx.allow_fp8 = allow_fp4_layer;

            // FP4 dgrad optimization: reuse cached FP4 W^T for dinp = dout @ W.
            if (mWeights->has_fp4_dgrad_cache() && allow_fp4_layer) {
                auto& fp4_t = mWeights->fp4_weight_cache_transposed();
                ctx.cached_fp4_data = &fp4_t.qkv_weight.data;
                ctx.cached_fp4_scales = &fp4_t.qkv_weight.scales;
                ctx.cached_fp4_amax = mWeights->fp4_weight_amax_transposed().template get<float>() + 0;
            }

            mRecipe->backward_matmul(ctx);
        });

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterQKVBackward);

        if (stack_large_bwd_temps) {
            rs.temp_free(da.d_qkv);
        }

        if (hook) (*hook)(layer_idx, accumulate, stream, BackwardHookPoint::AfterLayerBackward);
    }
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

        if constexpr (requires { g.d_mlp_up_weight; }) norm_squared(TensorShard(g.d_mlp_up_weight));
        if constexpr (requires { g.d_mlp_down_weight; }) norm_squared(TensorShard(g.d_mlp_down_weight));
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

