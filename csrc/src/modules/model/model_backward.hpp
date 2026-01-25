#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

// Backward pass

template<typename Block>
void ModularTransformerModel<Block>::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                               int grad_accum_steps, int micro_step) {
    backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, {});
}

template<typename Block>
void ModularTransformerModel<Block>::backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                                         int grad_accum_steps, int micro_step,
                                                         const BackwardBlockHook& hook) {
    NVTX_RANGE_FN();

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());
    long B = inputs.Sizes[0];
    long T = inputs.Sizes[1];
    long C = mConfig.HiddenSize;
    long L = mConfig.NumLayers;

    // LoRA-only mode: skip computing base weight gradients (only compute dinp for gradient flow)
    const bool lora_only = rs.is_lora_only_mode();
    const bool debug_grads = std::getenv("SUROGATE_DEBUG_MODULAR_GRADS") != nullptr;

    auto dump_row95 = [&](const char* tag, const Tensor& t, int layer_idx) {
        if (!t.Data) {
            fprintf(stderr, "[MOD GRADS] %s layer=%d ptr=null\n", tag, layer_idx);
            return;
        }
        long rows = (t.Rank >= 3) ? (t.Sizes[0] * t.Sizes[1]) : t.Sizes[0];
        long hidden = (t.Rank >= 3) ? t.Sizes[2] : (t.Rank >= 2 ? t.Sizes[1] : 0);
        if (rows <= 0 || hidden <= 0) {
            fprintf(stderr, "[MOD GRADS] %s layer=%d ptr=%p\n", tag, layer_idx, (void*)t.Data);
            return;
        }
        long row = rows > 95 ? 95 : (rows - 1);
        const long N = hidden < 64 ? hidden : 64;
        float sum_row = 0.0f;
        if (t.DType == ETensorDType::BF16) {
            std::vector<nv_bfloat16> buf(N);
            CUDA_CHECK(cudaMemcpy(buf.data(), (nv_bfloat16*)t.Data + row * hidden, N * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
            for (long i = 0; i < N; i++) sum_row += fabsf(__bfloat162float(buf[i]));
        } else if (t.DType == ETensorDType::FP32) {
            std::vector<float> buf(N);
            CUDA_CHECK(cudaMemcpy(buf.data(), (float*)t.Data + row * hidden, N * sizeof(float), cudaMemcpyDeviceToHost));
            for (long i = 0; i < N; i++) sum_row += fabsf(buf[i]);
        } else {
            fprintf(stderr, "[MOD GRADS] %s layer=%d ptr=%p dtype=%d\n",
                    tag, layer_idx, (void*)t.Data, static_cast<int>(t.DType));
            return;
        }
        fprintf(stderr, "[MOD GRADS] %s layer=%d row%ld_abssum=%.6f\n", tag, layer_idx, row, sum_row);
    };

    const BackwardBlockHook* hook_ptr = hook ? &hook : nullptr;

    bool last_step = micro_step == grad_accum_steps - 1;

    // Copy targets to device
    {
        NvtxRange r{"copy-targets"};
        // Make sure targets buffer is no longer needed by previous step
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, rs.side_stream()));
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.side_stream()));
    }

    // Initialize gradients on first micro-step
    if (micro_step == 0) {
        NvtxRange r{"zero-gradients"};
        // Zero losses and valid token count
        fill_zero(rs.Losses, main_stream);
        fill_zero(rs.ValidTokenCount, main_stream);
        mGrads->start_micro_step(rs.side_stream(), micro_step, grad_accum_steps);
        CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
    } else {
        mGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);
    }

    // Zero the gradient of final layer norm output
    fill_zero(rs.non_block_gradients().d_ln_final, main_stream);
    // Reset the last layer residual-stream gradient (filled by final-norm backward).
    fill_zero(rs.simplified_grads((int)L - 1).d_res_ffn, main_stream);

    // Backward through LM head
    backward_lmhead(B, T, micro_step, grad_accum_steps, comm);

    if (last_step) {
        reduce_loss(B, T, comm);
        // Aggregate valid-token count across ranks (loss is reduced with ncclAvg)
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, main_stream);
    }

    // Backward through final norm
    bool accumulate;
    cudaStream_t fetch_stream = rs.side_stream();
    auto& d_lnf_w = mGrads->get_final_norm_full(main_stream, comm, accumulate);
    mWeights->gather_final_norm(comm, fetch_stream);
    mWeights->gather_final_norm(comm, main_stream);
    auto& lnf_weight = mWeights->get_final_norm(main_stream);
    static bool final_norm_dbg = false;
    if (debug_grads && !final_norm_dbg) {
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
        dump_row95("final pre d_y", rs.non_block_gradients().d_ln_final, -1);
    }
    rmsnorm_backward(rs.simplified_grads((int)L - 1).d_res_ffn,
                     d_lnf_w,
                     rs.scratch().rmsnorm_scratch,
                     rs.simplified_grads((int)L - 1).d_res_ffn,
                     rs.non_block_gradients().d_ln_final,
                     rs.get_final_residual(),
                     lnf_weight,
                     rs.non_block_activations().ln_final_rstd,
                     rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                     (int)B, (int)T, (int)C,
                     rs.DeviceProp,
                     main_stream,
                     /*skip_weight_grad=*/lora_only);
    if (debug_grads && !final_norm_dbg) {
        CUDA_CHECK(cudaStreamSynchronize(main_stream));
        dump_row95("final post d_input", rs.simplified_grads((int)L - 1).d_res_ffn, (int)L - 1);
        final_norm_dbg = true;
    }
    mWeights->release_final_norm(main_stream);
    mGrads->notify_final_norm(main_stream, comm);

    // Backward through blocks
    if (mOptions.offload_residuals && L > 1) {
        rs.fetch_residual(static_cast<int>(L - 2), fetch_stream);
    }
    mWeights->gather_block(L - 1, comm, fetch_stream);
    // Hooks can be captured safely as long as their topology is stable for the run (e.g. LoRA hooks).
    // Disable per-layer graph capture when we're inside a full-step graph capture.
    const bool use_cuda_graphs = mOptions.use_cuda_graphs;
    if (use_cuda_graphs) {
        rs.configure_backward_graphs(/*hooked=*/hook_ptr != nullptr);
    }
    for (int l = L - 1; l >= 0; --l) {
        // Prefetch previous block
        if (l > 0) {
            if (l > 1) {
                rs.fetch_residual(l - 2, fetch_stream);
            }
            mWeights->gather_block(l - 1, comm, fetch_stream);
        }

        auto& block_weights = mWeights->get_block(l, main_stream);
        auto& block_grads = mGrads->get_block_full(l, main_stream, comm, accumulate);
        auto& block_acts = rs.get_block_activations(l);
        auto& block_d_acts = rs.get_block_gradients(l);

        // Recompute if needed
        Tensor& residual = l == 0 ? rs.non_block_activations().encoded :
                                    rs.get_residual(l - 1, main_stream);
        detail::trace_or_execute_cuda_graph_with_stack([&]() {
            recompute_block(l, block_weights, block_acts, residual);
            backward_block(l, accumulate, block_weights, block_grads, block_acts, block_d_acts, hook_ptr);
        }, main_stream, rs.backward_block_graph(l, accumulate), use_cuda_graphs,
           rs.Stack, rs.backward_block_stack_checkpoint(l, accumulate));

        // LN1 backward
        {
            auto& a = rs.simplified_acts(l);
            auto& da = rs.simplified_grads(l);
            Tensor* d_ln1_w = nullptr;
            if constexpr (requires { block_grads.ln1_grads.d_weight; }) {
                d_ln1_w = &block_grads.ln1_grads.d_weight;
            } else if constexpr (requires { block_grads.ln1.d_weight; }) {
                d_ln1_w = &block_grads.ln1.d_weight;
            }
            if (!d_ln1_w) {
                throw std::logic_error("ModularTransformerModel::backward_with_hook: LN1 weight gradients not available for this block type");
            }
            static bool ln1_dbg_top = false;
            static bool ln1_dbg_mid = false;
            static bool ln1_dbg_l1 = false;
            static bool ln1_dbg_l0 = false;
            const int mid_layer = static_cast<int>(L / 2);
            const bool ln1_dbg_layer =
                (l == static_cast<int>(L) - 1 && !ln1_dbg_top) ||
                (l == mid_layer && !ln1_dbg_mid) ||
                (l == 1 && !ln1_dbg_l1) ||
                (l == 0 && !ln1_dbg_l0);
            if (debug_grads && ln1_dbg_layer) {
                CUDA_CHECK(cudaStreamSynchronize(main_stream));
                dump_row95("ln1 pre d_y", da.d_ln1, l);
                dump_row95("ln1 pre d_residual_next", da.d_res_att, l);
            }
            if (l > 0) {
                auto& prev_da = rs.simplified_grads(l - 1);
                rmsnorm_backward(prev_da.d_res_ffn,
                                 *d_ln1_w,
                                 rs.scratch().rmsnorm_scratch,
                                 da.d_res_att,
                                 da.d_ln1,
                                 residual,
                                 block_weights.ln1.weight,
                                 a.ln1_rstd,
                                 rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                                 (int)B, (int)T, (int)C,
                                 rs.DeviceProp,
                                 main_stream,
                                 /*skip_weight_grad=*/lora_only);
                if (debug_grads && ln1_dbg_layer) {
                    CUDA_CHECK(cudaStreamSynchronize(main_stream));
                    dump_row95("ln1 post d_input", prev_da.d_res_ffn, l);
                    if (l == static_cast<int>(L) - 1) ln1_dbg_top = true;
                    if (l == mid_layer) ln1_dbg_mid = true;
                    if (l == 1) ln1_dbg_l1 = true;
                    if (l == 0) ln1_dbg_l0 = true;
                }
            } else {
                rmsnorm_backward(rs.non_block_gradients().d_embeddings,
                                 *d_ln1_w,
                                 rs.scratch().rmsnorm_scratch,
                                 da.d_res_att,
                                 da.d_ln1,
                                 residual,
                                 block_weights.ln1.weight,
                                 a.ln1_rstd,
                                 nullptr,
                                 (int)B, (int)T, (int)C,
                                 rs.DeviceProp,
                                 main_stream,
                                 /*skip_weight_grad=*/lora_only);
                if (debug_grads && ln1_dbg_layer) {
                    CUDA_CHECK(cudaStreamSynchronize(main_stream));
                    dump_row95("ln1 post d_input", rs.non_block_gradients().d_embeddings, l);
                    if (l == static_cast<int>(L) - 1) ln1_dbg_top = true;
                    if (l == mid_layer) ln1_dbg_mid = true;
                    if (l == 1) ln1_dbg_l1 = true;
                    if (l == 0) ln1_dbg_l0 = true;
                }
            }
        }

        mWeights->release_block(l, main_stream);
        mGrads->notify_block(l, main_stream, comm);

        if (l > 0) {
            rs.release_residual(l - 1, main_stream);
        }
    }

    // Embedding backward: skip in LoRA-only mode (embeddings are frozen)
    if (!lora_only) {
        auto& d_emb = mGrads->get_embeddings_full(main_stream, comm, accumulate);
        encoder_backward(d_emb,
                         rs.scratch().encoder_bwd_scratch,
                         rs.scratch().encoder_bwd_indices,
                         rs.scratch().encoder_bwd_info,
                         rs.non_block_gradients().d_embeddings,
                         rs.Inputs,
                         inputs,
                         (int)B, (int)T, (int)C,
                         static_cast<unsigned int>(mOptimizerRNG()),
                         main_stream,
                         rs.side_stream_event(),
                         rs.side_stream());
        mGrads->notify_embeddings(main_stream, comm);
    }

    // Finalize micro-step
    mGrads->end_micro_step(main_stream, comm);
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, main_stream));
    // Ensure the host-side target buffer is safe to reuse.
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
}
