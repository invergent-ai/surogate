#pragma once

// Lifecycle methods

template<typename Block>
ModularTransformerModel<Block>::ModularTransformerModel(
    const ModelConfig& config, const ModelOptions& options,
    int rank, int world, const std::shared_ptr<TensorAllocator>& alloc)
    : mConfig(config)
    , mOptions(options)
    , mAllocator(alloc ? alloc : std::make_shared<TensorAllocator>())
    , mEmbedding({.vocab_size = config.VocabSize, .hidden_size = config.HiddenSize})
    , mLMHead({.hidden_size = config.HiddenSize, .vocab_size = config.VocabSize,
               .tied_weights = config.TiedWordEmbeddings, .num_chunks = options.lmhead_chunks})
    , mFinalNorm({.hidden_size = config.HiddenSize, .epsilon = config.RmsNormEps})
    , mOptimizerRNG(42) {

    // Create transformer blocks
    BlockConfig block_config;
    block_config.hidden_size = config.HiddenSize;
    block_config.intermediate_size = config.IntermediateSize;
    block_config.num_query_heads = config.NumQueryHeads;
    block_config.num_kv_heads = config.NumKeyValHeads;
    block_config.head_size = config.head_size();
    block_config.rms_norm_eps = config.RmsNormEps;
    block_config.rope = config.Rope;  // Use full RoPEConfig
    block_config.rope.theta = config.RopeTheta;  // Backwards compat: sync theta
    block_config.max_seq_len = config.MaxPositionEmbeddings;
    block_config.use_qkv_bias = config.UseQKVBias;
    if constexpr (requires { block_config.use_qk_norm; }) {
        block_config.use_qk_norm = config.UseQKNorm;
    }

    mBlocks.reserve(config.NumLayers);
    for (int i = 0; i < config.NumLayers; ++i) {
        mBlocks.emplace_back(block_config);
    }

    // Create weight manager
    typename ModularWeightManager<Block>::Config wm_config;
    wm_config.num_layers = config.NumLayers;
    wm_config.block_config = block_config;
    wm_config.model_dtype = options.model_dtype.value_or(config.DType);
    // For FP8 forward-only mode, keep work weights in BF16 (not FP8).
    // Weight quantization to FP8 happens on-the-fly in forward_qmm_fp8.
    // This allows backward to use BF16 weights for stability.
    wm_config.matmul_dtype = options.get_matmul_dtype();
    wm_config.master_dtype = options.master_dtype.value_or(wm_config.model_dtype);
    wm_config.shard_idx = rank;
    wm_config.num_shards = world;
    wm_config.shard_weights = options.shard_weights;
    wm_config.offload_master = options.offload_master;
    wm_config.offload_quants = options.offload_quants;
    wm_config.use_zero_copy = options.use_zero_copy;
    wm_config.offload_alloc = options.get_offload_alloc();
    wm_config.persistent_quants = options.persistent_quants;
    wm_config.init_projections_to_zero = options.init_projections_to_zero;
    wm_config.vocab_size = config.VocabSize;
    wm_config.hidden_size = config.HiddenSize;
    wm_config.tied_embeddings = config.TiedWordEmbeddings;
    wm_config.skip_block_allocation = options.skip_block_allocation;
    wm_config.enable_fp8_forward = options.enable_fp8_forward;
    wm_config.enable_fp4_forward = options.enable_fp4_forward;

    mWeights = std::make_unique<ModularWeightManager<Block>>(wm_config, *mAllocator);
}

template<typename Block>
ModularTransformerModel<Block>::~ModularTransformerModel() = default;

template<typename Block>
void ModularTransformerModel<Block>::init_weights(NCCLCommunicator& comm) {
    mWeights->random_init(42, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    mWeights->import_from_file(file_name, allow_cast, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    mWeights->export_to_file(file_name, comm);
}

template<typename Block>
void ModularTransformerModel<Block>::on_restore_checkpoint(NCCLCommunicator& comm) {
    mWeights->synchronize_absmax(comm);
}

template<typename Block>
ModuleContext ModularTransformerModel<Block>::create_context(int B, int T) {
    auto& rs = *mRunState;
    ModuleContext ctx;
    ctx.stream = rs.non_block_activations().encoded.stream;  // Main stream
    ctx.cublas_handle = nullptr;  // Will be set from IRunState
    ctx.cudnn_handle = nullptr;   // Will be set from IRunState
    ctx.workspace = &rs.scratch().cudnn_workspace;
    ctx.device_prop = nullptr;    // Will be set from IRunState
    ctx.B = B;
    ctx.T = T;
    ctx.position_ids = nullptr;   // Will be set per-call
    ctx.matmul_dtype = mOptions.get_matmul_dtype();
    ctx.use_quantization = mOptions.matmul_dtype.has_value() &&
                           mOptions.matmul_dtype.value() != mOptions.model_dtype.value_or(mConfig.DType);
    return ctx;
}

