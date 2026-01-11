#pragma once

// CUDA graph capture + full-step training

template<typename Block>
void ModularTransformerModel<Block>::initialize_optimizer_state(cudaStream_t stream) {
    NVTX_RANGE_FN();

    if (!mAdamW8BitState) {
        throw std::runtime_error("initialize_optimizer_state: optimizer state not allocated (call allocate_run_state first)");
    }

    auto& state = *mAdamW8BitState;
    if (state.initialized) {
        return;  // Already initialized
    }

    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    // Count total parameters across all weight tensors and compute a padded layout.
    size_t total_params = 0;
    size_t state_elems = 0;

    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    // Count embedding parameters
    add_tensor(mWeights->get_master_embeddings().nelem());

    // Count final norm parameters
    add_tensor(mWeights->get_master_final_norm().nelem());

    // Count LM head parameters (if not tied)
    if (!mConfig.TiedWordEmbeddings) {
        add_tensor(mWeights->get_master_lm_head().nelem());
    }

    // Count block parameters
    for (int i = 0; i < mConfig.NumLayers; ++i) {
        mWeights->fetch_master_block(i, stream);
        auto& bw = mWeights->get_master_block(i, stream);

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
        }
        if constexpr (has_mlp_weights<typename Block::Weights>::value) {
            add_tensor(bw.mlp_up_weight.nelem());
            add_tensor(bw.mlp_down_weight.nelem());
        }

        mWeights->release_master_block(i, stream, stream);
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Determine allocation location based on offload options
    state.offload_state = mOptions.offload_optimizer;
    state.use_zero_copy = mOptions.use_zero_copy;

    EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
    if (state.offload_state) {
        if (state.use_zero_copy) {
            alloc_kind = mOptions.get_offload_alloc();
        } else {
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
        state.total_state_elems, stream
    );

    state.initialized = true;
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template<typename Block>
void ModularTransformerModel<Block>::reset_full_step_graph() {
    if (mFullStepGraph) {
        if (mFullStepGraph->graph_exec) {
            CUDA_CHECK(cudaGraphExecDestroy(mFullStepGraph->graph_exec));
            mFullStepGraph->graph_exec = nullptr;
        }
        mFullStepGraph->captured = false;
    }
}

template<typename Block>
void ModularTransformerModel<Block>::train_step_graphed(
    NCCLCommunicator& comm, DataLoader& loader,
    int grad_accum_steps,
    float learning_rate, float beta_1, float beta_2, int step,
    float epsilon, float weight_decay, float grad_clip,
    bool use_graph
) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    int B = rs.batch_size();
    int T = rs.seq_length();

    // Ensure optimizer state is initialized
    if (!mAdamW8BitState || !mAdamW8BitState->initialized) {
        throw std::runtime_error("train_step_graphed: optimizer state not initialized. "
                                 "Call initialize_optimizer_state() before using graphed training.");
    }

    // Initialize full-step graph state if needed
    if (!mFullStepGraph) {
        mFullStepGraph = std::make_unique<FullStepGraphState>();
        // Allocate device buffers for optimizer parameters
        mFullStepGraph->opt_params = mAllocator->allocate(ETensorDType::FP32, "graph_opt_params", {5});
        mFullStepGraph->opt_step = mAllocator->allocate(ETensorDType::INT32, "graph_opt_step", {1});
    }

    auto& graph_state = *mFullStepGraph;

    // Check if we need to re-capture the graph due to shape change
    if (graph_state.captured &&
        (graph_state.captured_grad_accum_steps != grad_accum_steps ||
         graph_state.captured_B != B ||
         graph_state.captured_T != T)) {
        reset_full_step_graph();
    }

    // Update optimizer parameters on device (always, even for graph replay)
    // This allows dynamic learning rate schedules without re-capture
    float opt_params_host[5] = {learning_rate, beta_1, beta_2, epsilon, weight_decay};
    CUDA_CHECK(cudaMemcpyAsync(graph_state.opt_params.Data, opt_params_host,
                               5 * sizeof(float), cudaMemcpyHostToDevice, main_stream));
    CUDA_CHECK(cudaMemcpyAsync(graph_state.opt_step.Data, &step,
                               sizeof(int), cudaMemcpyHostToDevice, main_stream));

    // Non-graph path: execute normally
    if (!use_graph) {
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            forward(inputs, position_ids, comm, j);
            backward(inputs, targets, comm, grad_accum_steps, j);
        }
        update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);
        return;
    }

    // Graph replay path (if already captured)
    if (graph_state.captured) {
        // Load all batches for this step first (data loading cannot be in the graph)
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            // Note: For proper multi-accumulation graphed training, we'd need
            // to store all batches, but for now we support grad_accum_steps=1
        }

        // Launch the captured graph
        CUDA_CHECK(cudaGraphLaunch(graph_state.graph_exec, main_stream));
        return;
    }

    // Graph capture path (first execution)
    // Note: Full graph capture including optimizer is complex due to:
    // 1. Data loading must happen outside the graph
    // 2. Multiple gradient accumulation steps would need all data pre-loaded
    // 3. NCCL collectives need special handling (cudaStreamCaptureModeRelaxed)
    //
    // For now, we capture a single forward+backward+optimizer iteration.
    // For grad_accum > 1, the user should use non-graphed mode or we'd need
    // to pre-allocate buffers for all micro-batches.

    if (grad_accum_steps > 1) {
        // Fall back to non-graphed mode for grad accumulation > 1
        // (Full support would require pre-staging all batches)
        Tensor& inputs = get_input_buffer();
        Tensor& targets = get_target_buffer();
        Tensor& position_ids = get_position_ids_buffer();

        for (int j = 0; j < grad_accum_steps; ++j) {
            loader.load_batch(inputs, targets, &position_ids);
            forward(inputs, position_ids, comm, j);
            backward(inputs, targets, comm, grad_accum_steps, j);
        }
        update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);
        return;
    }

    // Load single batch (outside graph capture)
    Tensor& inputs = get_input_buffer();
    Tensor& targets = get_target_buffer();
    Tensor& position_ids = get_position_ids_buffer();
    loader.load_batch(inputs, targets, &position_ids);
    CUDA_CHECK(cudaStreamSynchronize(main_stream));  // Ensure data is ready

    // Begin graph capture
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeThreadLocal));

    // Execute forward + backward + optimizer
    forward(inputs, position_ids, comm, 0);
    backward(inputs, targets, comm, 1, 0);
    update(comm, learning_rate, beta_1, beta_2, step, epsilon, weight_decay, grad_clip);

    // End capture
    CUDA_CHECK(cudaStreamEndCapture(main_stream, &graph));

    // Instantiate executable graph
    CUDA_CHECK(cudaGraphInstantiate(&graph_state.graph_exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    // Mark as captured
    graph_state.captured = true;
    graph_state.captured_grad_accum_steps = grad_accum_steps;
    graph_state.captured_B = B;
    graph_state.captured_T = T;

    // Launch the newly captured graph
    CUDA_CHECK(cudaGraphLaunch(graph_state.graph_exec, main_stream));
}

