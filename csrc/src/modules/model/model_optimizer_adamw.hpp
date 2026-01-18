#pragma once

// AdamW optimizer

template<typename Block>
void ModularTransformerModel<Block>::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                             int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();

    // Only 8-bit AdamW optimizer is supported
    if (!mAdamW8BitState) {
        throw std::logic_error("ModularTransformerModel::update() but no 8-bit optimizer state available");
    }
    update_adamw_8bit(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
}

template<typename Block>
void ModularTransformerModel<Block>::update_with_config(NCCLCommunicator& comm,
                                                         const optimizers::OptimizerConfig& config, int step) {
    NVTX_RANGE_FN();

    switch (config.type) {
        case optimizers::OptimizerType::ADAMW_8BIT:
            // Require state to be allocated by allocate_run_state()
            if (!mAdamW8BitState) {
                throw std::logic_error("ModularTransformerModel::update_with_config(ADAMW_8BIT): "
                                       "optimizer state not allocated (call allocate_run_state first)");
            }
            update_adamw_8bit(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                              step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
            break;

        case optimizers::OptimizerType::NORMUON:
            if (step == 1) {
                fmt::print("NorMuon optimizer selected (step {})\n", step);
            }
            update_normuon(comm, config, step);
            break;

        default:
            throw std::logic_error("ModularTransformerModel::update_with_config(): unsupported optimizer type");
    }
}

template<typename Block>
void ModularTransformerModel<Block>::update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                                                        int t, float epsilon, float weight_decay, float grad_clip) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    auto& state = *mAdamW8BitState;

    mWeights->begin_optimizer(rs.Stack, main_stream);

    // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
    calculate_gradient_norm(comm, grad_clip);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    // Lazy initialization of 8-bit state tensors
    // We need to count total parameters first
    if (!state.initialized) {
        constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
        size_t total_params = 0;
        size_t state_elems = 0;

        auto add_tensor = [&](size_t n) {
            total_params += n;
            // Ensure each tensor starts at a block boundary in the combined state buffers.
            state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            state_elems += n;
        };
        
        // Count embedding parameters
        add_tensor(mWeights->get_master_embeddings().nelem());

        // Count final norm parameters
        add_tensor(mWeights->get_master_final_norm().nelem());

        // Count block parameters (must match order in update phase)
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            mWeights->fetch_master_block(i, main_stream);
            auto& bw = mWeights->get_master_block(i, main_stream);
            
            // Count block parameters based on block type
            if constexpr (requires { bw.ln1.weight; }) {
                add_tensor(bw.ln1.weight.nelem());
            }
            if constexpr (requires { bw.ln2.weight; }) {
                add_tensor(bw.ln2.weight.nelem());
            }
            if constexpr (requires { bw.attention.qkv_weight; }) {
                add_tensor(bw.attention.qkv_weight.nelem());
                if constexpr (requires { bw.attention.out_weight; }) {
                    add_tensor(bw.attention.out_weight.nelem());
                }
                if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                    if (bw.attention.q_norm_weight.has_value()) {
                        add_tensor(bw.attention.q_norm_weight->nelem());
                    }
                    if (bw.attention.k_norm_weight.has_value()) {
                        add_tensor(bw.attention.k_norm_weight->nelem());
                    }
                }
            }
            if constexpr (has_mlp_weights<typename Block::Weights>::value) {
                add_tensor(bw.mlp_up_weight.nelem());
                add_tensor(bw.mlp_down_weight.nelem());
            }

            if constexpr (requires { bw.mamba; }) {
                if (bw.mamba.has_value()) {
                    auto& mw = *bw.mamba;
                    add_tensor(mw.in_proj_weight.nelem());
                    if (mw.in_proj_bias.has_value()) {
                        add_tensor(mw.in_proj_bias->nelem());
                    }
                    add_tensor(mw.out_proj_weight.nelem());
                    if (mw.out_proj_bias.has_value()) {
                        add_tensor(mw.out_proj_bias->nelem());
                    }
                    add_tensor(mw.conv1d_weight.nelem());
                    if (mw.conv1d_bias.has_value()) {
                        add_tensor(mw.conv1d_bias->nelem());
                    }
                    add_tensor(mw.A_log.nelem());
                    add_tensor(mw.D.nelem());
                    add_tensor(mw.dt_bias.nelem());
                    add_tensor(mw.norm_weight.nelem());
                }
            }

            if constexpr (has_moe_weights<typename Block::Weights>::value) {
                add_tensor(bw.router.gate.nelem());
                if (bw.experts.use_batched) {
                    add_tensor(bw.experts.gate_up_proj.nelem());
                    add_tensor(bw.experts.down_proj.nelem());
                }
            }
            
            mWeights->release_master_block(i, main_stream, rs.side_stream());
        }

        // Count LM head parameters (if not tied) - must be after blocks to match update order
        if (!mConfig.TiedWordEmbeddings) {
            add_tensor(mWeights->get_master_lm_head().nelem());
        }

        state.total_params = total_params;
        state.total_state_elems = state_elems;
        state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Determine allocation location based on offload options
        // For adamw8bit, we only support zero-copy offloading (pinned memory accessed directly by GPU)
        // because the 8-bit kernel needs direct GPU access to the state tensors
        state.offload_state = mOptions.offload_optimizer;
        state.use_zero_copy = mOptions.use_zero_copy;

        EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
        if (state.offload_state) {
            if (state.use_zero_copy) {
                // Zero-copy: allocate in pinned host memory, GPU accesses directly
                alloc_kind = mOptions.get_offload_alloc();
            } else {
                // Non-zero-copy offload not supported for adamw8bit (would need double-buffering)
                // Fall back to device allocation with a warning
                // Note: This could be implemented with double-buffering in the future
                alloc_kind = EAllocationType::ON_DEVICE;
            }
        }

        // Allocate 8-bit state tensors
        state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", alloc_kind, {(long)state.total_state_elems});
        state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", alloc_kind, {(long)state.total_state_elems});
        state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", alloc_kind, {(long)state.num_blocks});
        state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", alloc_kind, {(long)state.num_blocks});

        // Initialize state tensors
        init_adamw8bit_state(
            reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()),
            reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
            state.absmax1.template get<float>(),
            state.absmax2.template get<float>(),
            state.total_state_elems, main_stream
        );

        state.initialized = true;
    }
    
    // Track offset into combined state tensors
    size_t state_offset = 0;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    
    // Helper lambda for 8-bit update of a single tensor
    auto run_8bit_update = [&](const char* name, Tensor& val, const Tensor& grad, float wd) {
        // Align each tensor start to a block boundary. The single-tensor kernel assumes block 0
        // corresponds to the first element of the passed-in pointers.
        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        size_t n = val.nelem();
        size_t block_offset = state_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()) + state_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + state_offset;
        float* am1 = state.absmax1.template get<float>() + block_offset;
        float* am2 = state.absmax2.template get<float>() + block_offset;
        float* q1 = state.quantiles1.template get<float>();
        float* q2 = state.quantiles2.template get<float>();

        // Call the appropriate 8-bit update based on tensor dtype
        // grad_scale is passed as device pointer for CUDA graph compatibility
        if (val.DType == ETensorDType::FP32) {
            if (grad.DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<float>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, main_stream
                );
            } else if (grad.DType == ETensorDType::BF16) {
                // Mixed precision: FP32 master weights with BF16 gradients.
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<nv_bfloat16>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, main_stream
                );
            } else {
                throw std::runtime_error(std::string("adamw8bit: unsupported grad dtype for ") + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad.DType != ETensorDType::BF16) {
                throw std::runtime_error(std::string("adamw8bit: unsupported grad dtype for ") + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad.template get<nv_bfloat16>(),
                s1, s2, n,
                learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                q1, q2, am1, am2, main_stream
            );
        } else {
            throw std::runtime_error(std::string("adamw8bit: unsupported dtype for ") + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("adamw8bit: state buffer overflow (layout mismatch).");
        }
    };

    // Update embeddings
    run_8bit_update("embeddings", mWeights->get_master_embeddings(), 
                    mGrads->get_embeddings_shard(main_stream), weight_decay);

    // Update final norm
    run_8bit_update("final_norm", mWeights->get_master_final_norm(),
                    mGrads->get_final_norm_shard(main_stream), 0.f);

    // Update blocks
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        mWeights->fetch_master_block(i, comm.stream());
        auto& bw = mWeights->get_master_block(i, main_stream);
        auto& bg = mGrads->get_block_shard(i, main_stream);

        // Norm weights (no weight decay)
        if constexpr (requires { bw.ln1.weight; }) {
            if constexpr (requires { bg.ln1_grads.d_weight; }) {
                run_8bit_update("ln1.weight", bw.ln1.weight, bg.ln1_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln1.d_weight; }) {
                run_8bit_update("ln1.weight", bw.ln1.weight, bg.ln1.d_weight, 0.f);
            }
        }
        if constexpr (requires { bw.ln2.weight; }) {
            if constexpr (requires { bg.ln2_grads.d_weight; }) {
                run_8bit_update("ln2.weight", bw.ln2.weight, bg.ln2_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln2.d_weight; }) {
                run_8bit_update("ln2.weight", bw.ln2.weight, bg.ln2.d_weight, 0.f);
            }
        }

        // Attention weights
        if constexpr (requires { bw.attention.qkv_weight; }) {
            if constexpr (requires { bg.attention_grads.d_qkv_weight; }) {
                run_8bit_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention_grads.d_qkv_weight, weight_decay);
            } else if constexpr (requires { bg.attention.d_qkv_weight; }) {
                run_8bit_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention.d_qkv_weight, weight_decay);
            }
            if constexpr (requires { bw.attention.out_weight; }) {
                if constexpr (requires { bg.attention_grads.d_out_weight; }) {
                    run_8bit_update("attn.out_weight", bw.attention.out_weight, bg.attention_grads.d_out_weight, weight_decay);
                } else if constexpr (requires { bg.attention.d_out_weight; }) {
                    run_8bit_update("attn.out_weight", bw.attention.out_weight, bg.attention.d_out_weight, weight_decay);
                }
            }
            if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                if (bw.attention.q_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_q_norm_weight; }) {
                        if (bg.attention_grads.d_q_norm_weight.has_value()) {
                            run_8bit_update("attn.q_norm_weight",
                                            bw.attention.q_norm_weight.value(),
                                            bg.attention_grads.d_q_norm_weight.value(),
                                            0.f);
                        }
                    }
                    if constexpr (requires { bg.attention.d_q_norm_weight; }) {
                        if (bg.attention.d_q_norm_weight.has_value()) {
                            run_8bit_update("attn.q_norm_weight",
                                            bw.attention.q_norm_weight.value(),
                                            bg.attention.d_q_norm_weight.value(),
                                            0.f);
                        }
                    }
                }
                if (bw.attention.k_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_k_norm_weight; }) {
                        if (bg.attention_grads.d_k_norm_weight.has_value()) {
                            run_8bit_update("attn.k_norm_weight",
                                            bw.attention.k_norm_weight.value(),
                                            bg.attention_grads.d_k_norm_weight.value(),
                                            0.f);
                        }
                    }
                    if constexpr (requires { bg.attention.d_k_norm_weight; }) {
                        if (bg.attention.d_k_norm_weight.has_value()) {
                            run_8bit_update("attn.k_norm_weight",
                                            bw.attention.k_norm_weight.value(),
                                            bg.attention.d_k_norm_weight.value(),
                                            0.f);
                        }
                    }
                }
            }
        }

        // MLP weights (dense blocks only)
        if constexpr (has_mlp_weights<typename Block::Weights>::value) {
            run_8bit_update("mlp_up_weight", bw.mlp_up_weight, bg.d_mlp_up_weight, weight_decay);
            run_8bit_update("mlp_down_weight", bw.mlp_down_weight, bg.d_mlp_down_weight, weight_decay);
        }

        if constexpr (requires { bw.mamba; bg.mamba; }) {
            if (bw.mamba.has_value() && bg.mamba.has_value()) {
                auto& mw = *bw.mamba;
                auto& mg = *bg.mamba;
                run_8bit_update("mamba.in_proj_weight", mw.in_proj_weight, mg.d_in_proj_weight, weight_decay);
                if (mw.in_proj_bias.has_value() && mg.d_in_proj_bias.has_value()) {
                    run_8bit_update("mamba.in_proj_bias", mw.in_proj_bias.value(), mg.d_in_proj_bias.value(), 0.f);
                }
                run_8bit_update("mamba.out_proj_weight", mw.out_proj_weight, mg.d_out_proj_weight, weight_decay);
                if (mw.out_proj_bias.has_value() && mg.d_out_proj_bias.has_value()) {
                    run_8bit_update("mamba.out_proj_bias", mw.out_proj_bias.value(), mg.d_out_proj_bias.value(), 0.f);
                }
                run_8bit_update("mamba.conv1d_weight", mw.conv1d_weight, mg.d_conv1d_weight, weight_decay);
                if (mw.conv1d_bias.has_value() && mg.d_conv1d_bias.has_value()) {
                    run_8bit_update("mamba.conv1d_bias", mw.conv1d_bias.value(), mg.d_conv1d_bias.value(), 0.f);
                }
                run_8bit_update("mamba.A_log", mw.A_log, mg.d_A_log, 0.f);
                run_8bit_update("mamba.D", mw.D, mg.d_D, 0.f);
                run_8bit_update("mamba.dt_bias", mw.dt_bias, mg.d_dt_bias, 0.f);
                run_8bit_update("mamba.norm_weight", mw.norm_weight, mg.d_norm_weight, 0.f);
            }
        }

        if constexpr (has_moe_weights<typename Block::Weights>::value) {
            run_8bit_update("router.gate", bw.router.gate, bg.router.d_gate, weight_decay);
            if (bw.experts.use_batched) {
                run_8bit_update("experts.gate_up_proj", bw.experts.gate_up_proj, bg.experts.d_gate_up_proj, weight_decay);
                run_8bit_update("experts.down_proj", bw.experts.down_proj, bg.experts.d_down_proj, weight_decay);
            }
        }

        mWeights->release_master_block(i, main_stream, rs.side_stream());
        CUDA_CHECK(cudaEventRecord(rs.layer_update_done(i), main_stream));
    }

    // Update LM head if not tied
    if (!mConfig.TiedWordEmbeddings) {
        run_8bit_update("lm_head", mWeights->get_master_lm_head(),
                        mGrads->get_lm_head_shard(main_stream), weight_decay);
    }

    // Update FP8 delayed scaling state: roll amax history and compute new scales
    // This must happen after the forward pass (amaxes recorded) but before next forward pass uses them
    if (rs.has_fp8_delayed_scaling()) {
        delayed_scaling_update(rs.fp8_scaling_state(), main_stream);
    }

    comm.wait_on_comms(main_stream);
    mWeights->end_optimizer(rs.Stack);
    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
}
