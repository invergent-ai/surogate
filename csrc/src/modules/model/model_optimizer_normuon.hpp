#pragma once

// NorMuon optimizer

template<typename Block>
void ModularTransformerModel<Block>::update_normuon(NCCLCommunicator& comm,
                                                     const optimizers::OptimizerConfig& config, int t) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    // Initialize NorMuon state if needed
    if (!mNorMuonState) {
        mNorMuonState = std::make_unique<NorMuonState>();
    }
    auto& state = *mNorMuonState;

    mWeights->begin_optimizer(rs.Stack, main_stream);

    // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
    calculate_gradient_norm(comm, config.grad_clip);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    // Lazy initialization of state tensors
    constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

    if (!state.initialized) {
        fmt::print("NorMuon: Initializing optimizer state...\n");
        // Phase 1: Count parameters for AdamW (embeddings, norms, lm_head) and NorMuon (2D weights)
        size_t adamw_total = 0;
        size_t adamw_elems = 0;
        size_t normuon_total = 0;
        size_t normuon_elems = 0;
        size_t max_M = 0, max_N = 0;

        auto add_adamw = [&](size_t n) {
            adamw_total += n;
            adamw_elems = (adamw_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            adamw_elems += n;
        };

        auto add_normuon = [&](size_t M, size_t N) {
            size_t n = M * N;
            normuon_total += n;
            normuon_elems = (normuon_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            normuon_elems += n;
            max_M = std::max(max_M, M);
            max_N = std::max(max_N, N);
            state.variance_shapes.push_back({(int)M, (int)N});
        };

        // Embeddings -> AdamW
        add_adamw(mWeights->get_master_embeddings().nelem());

        // Final norm -> AdamW
        add_adamw(mWeights->get_master_final_norm().nelem());

        // Block parameters
        for (int i = 0; i < mConfig.NumLayers; ++i) {
            mWeights->fetch_master_block(i, main_stream);
            auto& bw = mWeights->get_master_block(i, main_stream);

            // Norm weights -> AdamW (no weight decay)
            if constexpr (requires { bw.ln1.weight; }) {
                add_adamw(bw.ln1.weight.nelem());
            }
            if constexpr (requires { bw.ln2.weight; }) {
                add_adamw(bw.ln2.weight.nelem());
            }

            // Attention weights -> NorMuon (2D)
            if constexpr (requires { bw.attention.qkv_weight; }) {
                // QKV is 2D: (3 * num_heads * head_dim, hidden_size) or similar
                auto& qkv = bw.attention.qkv_weight;
                add_normuon(qkv.Sizes[0], qkv.Sizes[1]);

                if constexpr (requires { bw.attention.out_weight; }) {
                    auto& out = bw.attention.out_weight;
                    add_normuon(out.Sizes[0], out.Sizes[1]);
                }

                // Q/K norm weights -> AdamW (1D)
                if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                    if (bw.attention.q_norm_weight.has_value()) {
                        add_adamw(bw.attention.q_norm_weight->nelem());
                    }
                    if (bw.attention.k_norm_weight.has_value()) {
                        add_adamw(bw.attention.k_norm_weight->nelem());
                    }
                }
            }

            // MLP weights -> NorMuon (2D)
            if constexpr (has_mlp_weights<typename Block::Weights>::value) {
                auto& up = bw.mlp_up_weight;
                auto& down = bw.mlp_down_weight;
                add_normuon(up.Sizes[0], up.Sizes[1]);
                add_normuon(down.Sizes[0], down.Sizes[1]);
            }

            // Mamba weights -> AdamW (treat as non-2D params for now)
            if constexpr (requires { bw.mamba; }) {
                if (bw.mamba.has_value()) {
                    auto& mw = *bw.mamba;
                    add_adamw(mw.in_proj_weight.nelem());
                    if (mw.in_proj_bias.has_value()) {
                        add_adamw(mw.in_proj_bias->nelem());
                    }
                    add_adamw(mw.out_proj_weight.nelem());
                    if (mw.out_proj_bias.has_value()) {
                        add_adamw(mw.out_proj_bias->nelem());
                    }
                    add_adamw(mw.conv1d_weight.nelem());
                    if (mw.conv1d_bias.has_value()) {
                        add_adamw(mw.conv1d_bias->nelem());
                    }
                    add_adamw(mw.A_log.nelem());
                    add_adamw(mw.D.nelem());
                    add_adamw(mw.dt_bias.nelem());
                    add_adamw(mw.norm_weight.nelem());
                }
            }

            // MoE weights -> AdamW (treat as non-2D params to avoid mixing experts in NorMuon)
            if constexpr (has_moe_weights<typename Block::Weights>::value) {
                add_adamw(bw.router.gate.nelem());
                if (bw.experts.use_batched) {
                    add_adamw(bw.experts.gate_up_proj.nelem());
                    add_adamw(bw.experts.down_proj.nelem());
                }
            }

            mWeights->release_master_block(i, main_stream, rs.side_stream());
        }

        // LM head -> AdamW (if not tied)
        if (!mConfig.TiedWordEmbeddings) {
            add_adamw(mWeights->get_master_lm_head().nelem());
        }

        // Store counts
        state.adamw_total_params = adamw_total;
        state.adamw_state_elems = adamw_elems;
        state.adamw_num_blocks = (adamw_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        state.normuon_total_params = normuon_total;
        state.normuon_state_elems = normuon_elems;
        state.normuon_num_blocks = (normuon_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

        state.max_weight_M = max_M;
        state.max_weight_N = max_N;

        // Allocate AdamW state tensors
        EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
        state.adamw_state1 = mAllocator->allocate(ETensorDType::BYTE, "normuon_adamw_state1", alloc_kind, {(long)state.adamw_state_elems});
        state.adamw_state2 = mAllocator->allocate(ETensorDType::BYTE, "normuon_adamw_state2", alloc_kind, {(long)state.adamw_state_elems});
        state.adamw_absmax1 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_absmax1", alloc_kind, {(long)state.adamw_num_blocks});
        state.adamw_absmax2 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_absmax2", alloc_kind, {(long)state.adamw_num_blocks});

        // Allocate AdamW quantiles
        state.adamw_quantiles1 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q1", EAllocationType::ON_DEVICE, {256});
        state.adamw_quantiles2 = mAllocator->allocate(ETensorDType::FP32, "normuon_adamw_q2", EAllocationType::ON_DEVICE, {256});

        // Initialize AdamW state
        init_adamw8bit_state(
            reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()),
            reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()),
            state.adamw_absmax1.template get<float>(),
            state.adamw_absmax2.template get<float>(),
            state.adamw_state_elems, main_stream
        );

        // Initialize quantile maps for AdamW (must be done on CPU then copied)
        {
            std::vector<float> h_quantiles1(256), h_quantiles2(256);
            optimizers::create_adamw8bit_quantiles1(h_quantiles1.data());
            optimizers::create_adamw8bit_quantiles2(h_quantiles2.data());
            CUDA_CHECK(cudaMemcpyAsync(state.adamw_quantiles1.Data, h_quantiles1.data(),
                                       256 * sizeof(float), cudaMemcpyHostToDevice, main_stream));
            CUDA_CHECK(cudaMemcpyAsync(state.adamw_quantiles2.Data, h_quantiles2.data(),
                                       256 * sizeof(float), cudaMemcpyHostToDevice, main_stream));
        }

        // Allocate NorMuon momentum state
        state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "normuon_momentum", alloc_kind, {(long)state.normuon_state_elems});
        state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "normuon_absmax", alloc_kind, {(long)state.normuon_num_blocks});
        state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "normuon_quantiles", EAllocationType::ON_DEVICE, {256});

        // Initialize NorMuon momentum state
        optimizers::init_normuon_momentum_state(
            reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()),
            state.momentum_absmax.template get<float>(),
            state.normuon_state_elems, main_stream
        );

        // Initialize signed quantile map for NorMuon momentum (same map as AdamW first moment)
        {
            std::vector<float> h_quantiles(256);
            optimizers::create_normuon_quantiles(h_quantiles.data());
            CUDA_CHECK(cudaMemcpyAsync(state.momentum_quantiles.Data, h_quantiles.data(),
                                       256 * sizeof(float), cudaMemcpyHostToDevice, main_stream));
        }

        // Allocate variance buffers for each 2D weight
        for (auto& [M, N] : state.variance_shapes) {
            // Variance is stored as row means or col means (low-rank)
            // For tall matrices (M >= N): reduce over cols -> shape (M, 1) -> need M floats
            // For wide matrices (M < N): reduce over rows -> shape (1, N) -> need N floats
            int var_size = optimizers::normuon_variance_buffer_size(M, N);
            state.variance_buffers.push_back(
                mAllocator->allocate(ETensorDType::FP32, "normuon_variance", alloc_kind, {var_size})
            );
            // Initialize variance buffer to ones
            fill_constant(state.variance_buffers.back().template get<float>(), 1.0f, var_size, main_stream);
        }

        // Allocate Polar Express workspace
        // normuon_update_2d uses the workspace for:
        // 1. momentum_out buffer (max_M * max_N bf16 elements) at the beginning
        // 2. Polar Express algorithm workspace after that
        // So total size = max_weight_elems + polar_express_workspace_elems
        size_t max_weight_size = max_M * max_N;
        size_t pe_ws_size = optimizers::polar_express_workspace_size(1, (int)max_M, (int)max_N);
        size_t total_ws_elems = max_weight_size + (pe_ws_size / sizeof(nv_bfloat16) + 1);
        state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "polar_workspace", alloc_kind,
                                                      {(long)total_ws_elems});

        // momentum_temp is no longer needed - momentum_out uses polar_workspace
        // Keeping allocation for potential future use or debugging
        state.momentum_temp = mAllocator->allocate(ETensorDType::BF16, "normuon_momentum_temp", alloc_kind,
                                                    {(long)max_weight_size});

        // Create cuBLAS handle for Polar Express matrix multiplications
        CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
        CUBLAS_CHECK(cublasSetStream(state.cublas_handle, main_stream));

        state.initialized = true;
        fmt::print("NorMuon: Initialized with {} AdamW params, {} NorMuon params (max weight {}x{})\n",
                   state.adamw_total_params, state.normuon_total_params, state.max_weight_M, state.max_weight_N);
    }

    // Extract hyperparameters from config
    float lr = config.normuon_lr;
    float beta1 = config.normuon_momentum;
    float beta2 = config.normuon_beta2;
    float weight_decay = config.weight_decay;
    bool cautious_wd = config.normuon_cautious_wd;

    // AdamW hyperparameters (for embeddings, norms, etc.)
    float adamw_lr = config.learning_rate;
    float adamw_beta1 = config.adamw_beta1;
    float adamw_beta2 = config.adamw_beta2;
    float adamw_eps = config.adamw_epsilon;

    // Track offsets into state buffers
    size_t adamw_offset = 0;
    size_t normuon_offset = 0;
    int variance_idx = 0;

    // Helper for AdamW 8-bit update (embeddings, norms, lm_head)
    auto run_adamw_update = [&](const char* name, Tensor& val, const Tensor& grad, float wd) {
        adamw_offset = (adamw_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        size_t n = val.nelem();
        size_t block_offset = adamw_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.adamw_state1.template get<std::byte>()) + adamw_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.adamw_state2.template get<std::byte>()) + adamw_offset;
        float* am1 = state.adamw_absmax1.template get<float>() + block_offset;
        float* am2 = state.adamw_absmax2.template get<float>() + block_offset;
        float* q1 = state.adamw_quantiles1.template get<float>();
        float* q2 = state.adamw_quantiles2.template get<float>();

        if (val.DType == ETensorDType::BF16 && grad.DType == ETensorDType::BF16) {
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad.template get<nv_bfloat16>(),
                s1, s2, n,
                adamw_lr, adamw_beta1, adamw_beta2, t, adamw_eps, wd, grad_scale,
                q1, q2, am1, am2, nullptr, nullptr, main_stream
            );
        } else if (val.DType == ETensorDType::FP32) {
            if (grad.DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<float>(),
                    s1, s2, n,
                    adamw_lr, adamw_beta1, adamw_beta2, t, adamw_eps, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, main_stream
                );
            } else if (grad.DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad.template get<nv_bfloat16>(),
                    s1, s2, n,
                    adamw_lr, adamw_beta1, adamw_beta2, t, adamw_eps, wd, grad_scale,
                    q1, q2, am1, am2, nullptr, nullptr, main_stream
                );
            }
        }

        adamw_offset += n;
    };

    // Helper for NorMuon update (2D weight matrices)
    auto run_normuon_update = [&](const char* name, Tensor& val, const Tensor& grad, float wd) {
        normuon_offset = (normuon_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        int M = val.Sizes[0];
        int N = val.Sizes[1];
        size_t n = (size_t)M * N;
        size_t block_offset = normuon_offset / BLOCK_SIZE;

        // Pointers into combined momentum state
        unsigned char* mom_state = reinterpret_cast<unsigned char*>(state.momentum_state.template get<std::byte>()) + normuon_offset;
        float* mom_absmax = state.momentum_absmax.template get<float>() + block_offset;
        float* quantiles = state.momentum_quantiles.template get<float>();
        float* var_buf = state.variance_buffers[variance_idx].template get<float>();
        nv_bfloat16* workspace = state.polar_workspace.template get<nv_bfloat16>();
        nv_bfloat16* momentum_temp = state.momentum_temp.template get<nv_bfloat16>();

        // Get gradient pointer (apply grad_scale)
        const nv_bfloat16* grad_ptr = grad.template get<nv_bfloat16>();

        // Apply NorMuon update
        // Note: val must be BF16 for NorMuon
        if (val.DType == ETensorDType::BF16 && grad.DType == ETensorDType::BF16) {
            optimizers::normuon_update_2d(
                state.cublas_handle,
                val.template get<nv_bfloat16>(),
                grad_ptr,
                mom_state,
                var_buf,
                workspace,
                M, N,
                lr, beta1, beta2, cautious_wd ? wd : 0.0f,
                quantiles, mom_absmax,
                main_stream
            );

            // Apply non-cautious weight decay if needed
            if (!cautious_wd && wd > 0.0f) {
                // Standard weight decay: p = p - lr * wd * p
                // This is done separately for non-cautious mode
                // (The cautious version is handled inside normuon_update_2d)
            }
        }

        normuon_offset += n;
        variance_idx++;
    };

    // Update embeddings (AdamW)
    run_adamw_update("embeddings", mWeights->get_master_embeddings(),
                     mGrads->get_embeddings_shard(main_stream), weight_decay);

    // Update final norm (AdamW, no weight decay)
    run_adamw_update("final_norm", mWeights->get_master_final_norm(),
                     mGrads->get_final_norm_shard(main_stream), 0.f);

    // Update blocks
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        mWeights->fetch_master_block(i, comm.stream());
        auto& bw = mWeights->get_master_block(i, main_stream);
        auto& bg = mGrads->get_block_shard(i, main_stream);

        // Norm weights (AdamW, no weight decay)
        if constexpr (requires { bw.ln1.weight; }) {
            if constexpr (requires { bg.ln1_grads.d_weight; }) {
                run_adamw_update("ln1.weight", bw.ln1.weight, bg.ln1_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln1.d_weight; }) {
                run_adamw_update("ln1.weight", bw.ln1.weight, bg.ln1.d_weight, 0.f);
            }
        }
        if constexpr (requires { bw.ln2.weight; }) {
            if constexpr (requires { bg.ln2_grads.d_weight; }) {
                run_adamw_update("ln2.weight", bw.ln2.weight, bg.ln2_grads.d_weight, 0.f);
            } else if constexpr (requires { bg.ln2.d_weight; }) {
                run_adamw_update("ln2.weight", bw.ln2.weight, bg.ln2.d_weight, 0.f);
            }
        }

        // Attention weights (NorMuon for 2D, AdamW for Q/K norms)
        if constexpr (requires { bw.attention.qkv_weight; }) {
            if constexpr (requires { bg.attention_grads.d_qkv_weight; }) {
                run_normuon_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention_grads.d_qkv_weight, weight_decay);
            } else if constexpr (requires { bg.attention.d_qkv_weight; }) {
                run_normuon_update("attn.qkv_weight", bw.attention.qkv_weight, bg.attention.d_qkv_weight, weight_decay);
            }
            if constexpr (requires { bw.attention.out_weight; }) {
                if constexpr (requires { bg.attention_grads.d_out_weight; }) {
                    run_normuon_update("attn.out_weight", bw.attention.out_weight, bg.attention_grads.d_out_weight, weight_decay);
                } else if constexpr (requires { bg.attention.d_out_weight; }) {
                    run_normuon_update("attn.out_weight", bw.attention.out_weight, bg.attention.d_out_weight, weight_decay);
                }
            }
            // Q/K norm weights (AdamW)
            if constexpr (requires { bw.attention.q_norm_weight; bw.attention.k_norm_weight; }) {
                if (bw.attention.q_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_q_norm_weight; }) {
                        if (bg.attention_grads.d_q_norm_weight.has_value()) {
                            run_adamw_update("attn.q_norm_weight",
                                             bw.attention.q_norm_weight.value(),
                                             bg.attention_grads.d_q_norm_weight.value(),
                                             0.f);
                        }
                    }
                    if constexpr (requires { bg.attention.d_q_norm_weight; }) {
                        if (bg.attention.d_q_norm_weight.has_value()) {
                            run_adamw_update("attn.q_norm_weight",
                                             bw.attention.q_norm_weight.value(),
                                             bg.attention.d_q_norm_weight.value(),
                                             0.f);
                        }
                    }
                }
                if (bw.attention.k_norm_weight.has_value()) {
                    if constexpr (requires { bg.attention_grads.d_k_norm_weight; }) {
                        if (bg.attention_grads.d_k_norm_weight.has_value()) {
                            run_adamw_update("attn.k_norm_weight",
                                             bw.attention.k_norm_weight.value(),
                                             bg.attention_grads.d_k_norm_weight.value(),
                                             0.f);
                        }
                    }
                    if constexpr (requires { bg.attention.d_k_norm_weight; }) {
                        if (bg.attention.d_k_norm_weight.has_value()) {
                            run_adamw_update("attn.k_norm_weight",
                                             bw.attention.k_norm_weight.value(),
                                             bg.attention.d_k_norm_weight.value(),
                                             0.f);
                        }
                    }
                }
            }
        }

        // MLP weights (NorMuon)
        if constexpr (has_mlp_weights<typename Block::Weights>::value) {
            run_normuon_update("mlp_up_weight", bw.mlp_up_weight, bg.d_mlp_up_weight, weight_decay);
            run_normuon_update("mlp_down_weight", bw.mlp_down_weight, bg.d_mlp_down_weight, weight_decay);
        }

        // Mamba weights (AdamW)
        if constexpr (requires { bw.mamba; bg.mamba; }) {
            if (bw.mamba.has_value() && bg.mamba.has_value()) {
                auto& mw = *bw.mamba;
                auto& mg = *bg.mamba;
                run_adamw_update("mamba.in_proj_weight", mw.in_proj_weight, mg.d_in_proj_weight, weight_decay);
                if (mw.in_proj_bias.has_value() && mg.d_in_proj_bias.has_value()) {
                    run_adamw_update("mamba.in_proj_bias", mw.in_proj_bias.value(), mg.d_in_proj_bias.value(), 0.f);
                }
                run_adamw_update("mamba.out_proj_weight", mw.out_proj_weight, mg.d_out_proj_weight, weight_decay);
                if (mw.out_proj_bias.has_value() && mg.d_out_proj_bias.has_value()) {
                    run_adamw_update("mamba.out_proj_bias", mw.out_proj_bias.value(), mg.d_out_proj_bias.value(), 0.f);
                }
                run_adamw_update("mamba.conv1d_weight", mw.conv1d_weight, mg.d_conv1d_weight, weight_decay);
                if (mw.conv1d_bias.has_value() && mg.d_conv1d_bias.has_value()) {
                    run_adamw_update("mamba.conv1d_bias", mw.conv1d_bias.value(), mg.d_conv1d_bias.value(), 0.f);
                }
                run_adamw_update("mamba.A_log", mw.A_log, mg.d_A_log, 0.f);
                run_adamw_update("mamba.D", mw.D, mg.d_D, 0.f);
                run_adamw_update("mamba.dt_bias", mw.dt_bias, mg.d_dt_bias, 0.f);
                run_adamw_update("mamba.norm_weight", mw.norm_weight, mg.d_norm_weight, 0.f);
            }
        }

        if constexpr (has_moe_weights<typename Block::Weights>::value) {
            run_adamw_update("router.gate", bw.router.gate, bg.router.d_gate, weight_decay);
            if (bw.experts.use_batched) {
                run_adamw_update("experts.gate_up_proj", bw.experts.gate_up_proj, bg.experts.d_gate_up_proj, weight_decay);
                run_adamw_update("experts.down_proj", bw.experts.down_proj, bg.experts.d_down_proj, weight_decay);
            }
        }

        mWeights->release_master_block(i, main_stream, rs.side_stream());
        CUDA_CHECK(cudaEventRecord(rs.layer_update_done(i), main_stream));
    }

    // Update LM head if not tied (AdamW)
    if (!mConfig.TiedWordEmbeddings) {
        run_adamw_update("lm_head", mWeights->get_master_lm_head(),
                         mGrads->get_lm_head_shard(main_stream), weight_decay);
    }

    // Update FP8 delayed scaling state
    if (rs.has_fp8_delayed_scaling()) {
        delayed_scaling_update(rs.fp8_scaling_state(), main_stream);
    }

    comm.wait_on_comms(main_stream);
    mWeights->end_optimizer(rs.Stack);
    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
}
