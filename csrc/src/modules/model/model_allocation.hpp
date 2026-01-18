#pragma once

// Run state allocation

template<typename Block>
void ModularTransformerModel<Block>::allocate_run_state(const ModelOptions& options, NCCLCommunicator& comm,
                                                         int B, int T, bool allocate_optimizer) {
    NVTX_RANGE_FN();
    
    mOptions = options;

    // Synchronize Four Over Six (4/6) setting from recipe to weight manager.
    // This ensures cached FP4 weights use the same quantization method as the recipe.
    if (mRecipe) {
        if (auto* nvfp4_recipe = dynamic_cast<recipes::NVFP4Recipe*>(mRecipe.get())) {
            mWeights->set_four_over_six(
                nvfp4_recipe->uses_four_over_six(),
                nvfp4_recipe->four_over_six_metric()
            );
        }
    }

    // B200/B300 optimization: persist FP4 base weights across steps in LoRA/frozen-base mode.
    // This avoids re-quantizing/transposing BF16 base weights every iteration, which can make
    // NVFP4 slower than FP8 on very fast datacenter GPUs.
    if (options.enable_fp4_forward) {
        mWeights->maybe_enable_fp4_persistent_cache(/*weights_static=*/options.skip_base_gradients);
    }

    // Create run state config
    typename ModularRunState<Block>::Config rs_config;
    rs_config.num_layers = mConfig.NumLayers;
    rs_config.batch_size = B;
    rs_config.seq_length = T;
    rs_config.hidden_size = mConfig.HiddenSize;
    rs_config.vocab_size = mConfig.VocabSize;
    // Activations/gradients are computed and stored in a real compute dtype (BF16/FP32).
    // Some modes (e.g. FP8 QLoRA weight storage) can set the model dtype to an FP8 type;
    // we must not allocate activations as FP8, because many kernels (e.g. abs_max) only
    // support FP32/BF16 and FP8 is handled via separate quant buffers.
    ETensorDType act_dtype = options.model_dtype.value_or(mConfig.DType);
    if (is_fp8_dtype(act_dtype)) act_dtype = ETensorDType::BF16;
    rs_config.activation_dtype = act_dtype;
    // Activation-gradient tensors stay in activation dtype; FP8 training uses separate quant buffers.
    rs_config.grad_dtype = act_dtype;
    rs_config.matmul_dtype = options.get_matmul_dtype();
    rs_config.grad_quant_dtype = options.get_grad_dtype();
    rs_config.enable_fp8_forward = options.enable_fp8_forward;
    rs_config.enable_fp8_hybrid_delayed = options.enable_fp8_hybrid;  // Auto-enable delayed scaling with hybrid mode
    rs_config.fp8_scaling_config = options.fp8_scaling_config;
    rs_config.forward_matmul_dtype = options.get_forward_matmul_dtype();
    rs_config.enable_fp4_forward = options.enable_fp4_forward;
    rs_config.enable_fp4_backward = options.enable_fp4_backward;
    rs_config.offload_residuals = options.offload_residuals;
    rs_config.recompute_rmsnorm = options.recompute_rmsnorm;
    rs_config.recompute_qkv = options.recompute_qkv;
    rs_config.recompute_attention = options.recompute_attention;
    rs_config.recompute_ffn = options.recompute_ffn;
    rs_config.recompute_swiglu = options.recompute_swiglu;
    rs_config.recompute_block = options.recompute_block;
    rs_config.lmhead_chunks = options.lmhead_chunks;
    rs_config.attention_bwd_chunks = options.attention_bwd_chunks;
    rs_config.lora_only_mode = options.lora_only_mode;
    rs_config.recompute_lora = options.recompute_lora;
    rs_config.train_router = options.train_router;
    rs_config.use_fused_rope = options.use_fused_rope;

    // Set block config for run state
    rs_config.block_config.hidden_size = mConfig.HiddenSize;
    // For MoE models, use MoeIntermediateSize for the per-expert intermediate dimension
    rs_config.block_config.intermediate_size = (mConfig.NumExperts > 0 && mConfig.MoeIntermediateSize > 0)
                                               ? mConfig.MoeIntermediateSize
                                               : mConfig.IntermediateSize;
    if constexpr (requires { rs_config.block_config.mlp_up_factor; }) {
        rs_config.block_config.mlp_up_factor = mConfig.mlp_up_factor();
    }

    rs_config.block_config.num_query_heads = mConfig.NumQueryHeads;
    rs_config.block_config.num_kv_heads = mConfig.NumKeyValHeads;
    rs_config.block_config.head_size = mConfig.head_size();
    rs_config.block_config.rms_norm_eps = mConfig.RmsNormEps;
    rs_config.block_config.rope = mConfig.Rope;  // Use full RoPEConfig
    rs_config.block_config.rope.theta = mConfig.RopeTheta;  // Backwards compat
    rs_config.block_config.max_seq_len = mConfig.MaxPositionEmbeddings;
    rs_config.block_config.use_qkv_bias = mConfig.UseQKVBias;
    if constexpr (requires { rs_config.block_config.use_qk_norm; }) {
        rs_config.block_config.use_qk_norm = mConfig.UseQKNorm;
    }
    if constexpr (requires { rs_config.block_config.mamba_num_heads; }) {
        rs_config.block_config.mamba_num_heads = mConfig.MambaNumHeads;
        rs_config.block_config.mamba_head_dim = mConfig.MambaHeadDim;
        rs_config.block_config.mamba_ssm_state_size = mConfig.MambaSsmStateSize;
        rs_config.block_config.mamba_conv_kernel = mConfig.MambaConvKernel;
        rs_config.block_config.mamba_n_groups = mConfig.MambaNGroups;
        rs_config.block_config.mamba_chunk_size = mConfig.MambaChunkSize;
        rs_config.block_config.mamba_use_bias = mConfig.MambaUseBias;
        rs_config.block_config.mamba_use_conv_bias = mConfig.MambaUseConvBias;
        rs_config.block_config.mamba_activation = mConfig.MambaActivation;
    }

    rs_config.layer_is_mamba.resize(mConfig.NumLayers);
    rs_config.has_mamba = false;
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        const bool is_mamba = (mConfig.get_block_type(i) == BlockType::Mamba);
        rs_config.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
        rs_config.has_mamba = rs_config.has_mamba || is_mamba;
    }

    // Set PretrainedConfig for IRunState base class.
    // Use original_config to preserve the full derived type (e.g., Qwen3MoEConfig)
    rs_config.pretrained_config = mConfig.original_config.get();

    // Two-pass stack allocation:
    // 1. First pass with a dummy stack to measure peak temporary usage
    // 2. Second pass with a properly allocated stack
    //
    // Important: the peak must include temporaries allocated during the optimizer pass
    // (e.g., device caches for offloaded Adam moments), not just construction-time
    // stack simulations inside ModularRunState.
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    DeviceMemoryStack dummy_stack(nullptr, 1024LL * 1024 * 1024 * 1024, dev);

    // First pass - construct run state (dummy_stack is moved into mRunState->Stack).
    mRunState = std::make_unique<ModularRunState<Block>>(rs_config, dummy_stack, mAllocator);

    // Allocate gradient manager
    // In LoRA mode (skip_base_gradients), we still need a gradient manager for the backward pass
    // computation flow, but we can use minimal allocation since the gradients are discarded.
    typename ModularGradientManager<Block>::Config gm_config;
    gm_config.num_layers = mConfig.NumLayers;
    gm_config.block_config = rs_config.block_config;
    // Weight gradients are stored in model dtype. The CLI flag --gradient-dtype
    // refers to activation-gradient quantization for FP8 matmuls, not parameter gradients.
    gm_config.grad_dtype = options.model_dtype.value_or(mConfig.DType);
    gm_config.shard_idx = comm.rank();
    gm_config.num_shards = comm.world_size();
    gm_config.shard_gradients = options.shard_gradients;
    gm_config.use_all_to_all_reduce = options.use_all_to_all_reduce;
    gm_config.offload_grads = options.offload_grads;
    gm_config.offload_alloc = options.get_offload_alloc();
    gm_config.vocab_size = mConfig.VocabSize;
    gm_config.hidden_size = mConfig.HiddenSize;
    gm_config.tied_embeddings = mConfig.TiedWordEmbeddings;
    gm_config.skip_allocation = options.skip_base_gradients;  // Skip allocating large gradient buffers
    gm_config.train_router = options.train_router;  // Allocate router gradient even in LoRA mode
    gm_config.layer_is_mamba.resize(mConfig.NumLayers);
    gm_config.has_mamba = false;
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        const bool is_mamba = (mConfig.get_block_type(i) == BlockType::Mamba);
        gm_config.layer_is_mamba[i] = static_cast<std::uint8_t>(is_mamba ? 1 : 0);
        gm_config.has_mamba = gm_config.has_mamba || is_mamba;
    }

    mGrads = std::make_unique<ModularGradientManager<Block>>(42, 0, gm_config, mAllocator);

    // Allocate 8-bit AdamW optimizer state if requested
    if (allocate_optimizer) {
        mAdamW8BitState = std::make_unique<AdamW8BitState>();

        // State tensors will be initialized lazily in the first update call
        mAdamW8BitState->initialized = false;

        // Allocate quantization maps (256 entries each)
        mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
        mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});

        // Initialize quantization maps on host then copy to device
        std::vector<float> h_quantiles1(256), h_quantiles2(256);
        create_adamw8bit_quantiles1(h_quantiles1.data());
        create_adamw8bit_quantiles2(h_quantiles2.data());
        CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_quantiles1.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_quantiles2.data(),
                              256 * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Get measured usage from the stack that's now inside mRunState.
    // MoE models need significant extra stack for batched expert weight transposes during backward.
    // For Qwen3-30B MoE with 128 experts:
    //   - expert gate_up transpose: 128 * 2 * 768 * 2048 * 2 = 768 MB
    //   - expert down transpose: 128 * 768 * 2048 * 2 = 384 MB
    //   - permuted tokens (input + output): 2 * B * T * top_k * C * 2 = 256 MB
    // Peak usage can be all of these live simultaneously during backward pass.
    long base_size = static_cast<long>(mRunState->Stack.max_utilization());
    long moe_extra = 0;
    if (mConfig.NumExperts > 0) {
        // Expert gate_up weight transpose: num_experts * 2 * moe_intermediate * hidden_size * 2 bytes
        long expert_gate_up_tp = static_cast<long>(mConfig.NumExperts) * 2 * mConfig.MoeIntermediateSize * mConfig.HiddenSize * 2;
        // Expert down weight transpose: num_experts * moe_intermediate * hidden_size * 2 bytes
        long expert_down_tp = static_cast<long>(mConfig.NumExperts) * mConfig.MoeIntermediateSize * mConfig.HiddenSize * 2;
        // Permuted tokens: B * T * top_k * hidden_size * 2 bytes (x2 for input and output)
        long permuted_tokens = 2L * B * T * mConfig.NumExpertsPerTok * mConfig.HiddenSize * 2;
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens;
    }
    long required_size = std::max(1024L * 1024, base_size + base_size / 2 + moe_extra);
    auto high_mark = mRunState->Stack.get_high_mark();

    // Allocate real stack and replace the dummy.
    mRunState->Stack = DeviceMemoryStack{
        mAllocator->allocate(ETensorDType::BYTE, "stack", {required_size}).Data,
        static_cast<std::size_t>(required_size),
        dev
    };
    mRunState->Stack.set_high_mark(high_mark);

    comm.barrier();
}
