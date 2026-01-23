// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Backward execution for DSL Graph executor.

#include "dsl/graph_executor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_internal.h"
#include "dsl/graph_executor_tensors.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "modules/fp8_scaling_config.h"
#include "modules/fp8_scaling_state.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_run_state.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

/**
 * @brief Execute a callable with stack checkpoint/restore for CUDA graph compatibility.
 */
template<typename Function>
inline void trace_or_execute_cuda_graph_with_stack(Function&& function, cudaStream_t stream,
                                                    cudaGraphExec_t& instance, bool enabled,
                                                    DeviceMemoryStack& stack,
                                                    DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: restore stack state and replay existing executable.
    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    // Capture path: save checkpoint before capture so we know where to restore to.
    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

}  // namespace


void GraphExecutor::execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step,
                                           const modules::BackwardHook* hook) {
    if (!mBackward) {
        throw std::runtime_error("DSL graph executor: missing backward graph");
    }

    auto& rs = mRunState;
    auto& weights = mWeights;
    auto& grads = mGrads;
    const auto& config = mConfig;
    const bool trace_ops = env_enabled("SUROGATE_DEBUG_DSL_TRACE");
    (void)comm;

    ExecState st{
        .rs = rs,
        .weights = weights,
        .grads = grads,
        .config = config,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
        .view_sources = &mViewSources,
        .view_sources_rev = &mViewSourcesReverse,
    };
    augment_shape_env(st.shape_env, mModule.config);

    // Initialize precomputed recomputation flags (computed once per backward pass)
    // This eliminates per-layer conditional overhead - flags are checked once here.
    const_cast<GraphExecutor*>(this)->init_recompute_flags();
    const bool recompute_any = mRecomputeFlags.any;
    const bool recompute_block = mOptions.RecomputeBlock && !(rs.is_lora_only_mode() && !mOptions.RecomputeLoRA);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(rs.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool use_graphs_enabled = mBackwardGraphsEnabled && mBackwardGraphCut > 0 && !in_capture;
    
    std::vector<char> recomputed;
    if (recompute_any && config.NumLayers > 0) {
        recomputed.assign(static_cast<size_t>(config.NumLayers), 0);
    }
    st.recomputed_layers = recompute_any ? &recomputed : nullptr;
    int last_recompute_layer = -1;

    // Track whether we're in CUDA graph capture mode (must be declared before recompute_layer lambda)
    bool bwd_capturing = false;

    // Enable per-layer CUDA graph capture for recomputation when backward graphs are enabled.
    // This significantly reduces kernel launch overhead by caching the entire recompute
    // sequence per layer (matching the modular path's behavior).
    const bool use_recompute_graphs = use_graphs_enabled && recompute_any;

    // Wrapper lambda that delegates to the optimized method.
    // The optimized method uses precomputed flags and optional CUDA graph caching.
    auto recompute_layer = [&, this](int layer_idx) {
        if (!recompute_any) return;
        if (layer_idx < 0 || layer_idx >= config.NumLayers) return;
        if (recompute_block) {
            if (layer_idx == last_recompute_layer) {
                return;
            }
            last_recompute_layer = layer_idx;
        } else {
            if (recomputed.empty() || recomputed[layer_idx]) return;
        }
        if (!recomputed.empty()) {
            recomputed[layer_idx] = 1;
        }

        // Delegate to optimized recompute with optional CUDA graph caching.
        // The capturing flag is checked inside to avoid graphs during initial capture phase.
        const bool use_graph = use_recompute_graphs && !bwd_capturing;
        const_cast<GraphExecutor*>(this)->recompute_layer_optimized(layer_idx, B, T, use_graph);
    };

    // Bind forward inputs needed by backward ops (e.g., rope uses position_ids).
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);

    // Bind d_logits (produced by classifier) as [B, T, V] view.
    Tensor logits_view = view_tensor(rs.non_block_activations().output, {B, T, config.VocabSize});
    st.tensors.emplace("d_logits", logits_view);

    // Bind views for d_xF_flat / d_xF to write into d_ln_final.
    Tensor d_ln_final_flat = view_tensor(rs.non_block_gradients().d_ln_final, {B * T, config.HiddenSize});
    st.tensors.emplace("d_xF_flat", d_ln_final_flat);
    st.tensors.emplace("d_xF", rs.non_block_gradients().d_ln_final);

    // Bind embedding output gradients to the non-block d_embeddings buffer.
    for (const auto& name : mEmbeddingOutputs) {
        st.tensors.emplace("d_" + name, rs.non_block_gradients().d_embeddings);
    }

    // Bind gradient outputs for parameters.
    std::unordered_set<std::string> accumulate_tensors;

    auto bind_param_grad = [&](const std::string& param_name) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            return;
        }
        bool accumulate = false;
        Tensor* grad_tensor = grads.get_param_grad(param_name, accumulate);
        if (!grad_tensor) {
            return;
        }
        std::string grad_name = "d_" + param_name;
        st.tensors.emplace(grad_name, *grad_tensor);
        if (accumulate) {
            accumulate_tensors.insert(grad_name);
        }
    };

    for (const auto& kv : mForward->params) {
        bind_param_grad(kv.first);
    }

    // Track current layer for backward prefetching (need to declare before lambdas)
    int bwd_current_layer = -1;
    int bwd_last_prefetched = -1;

    std::unordered_map<int, DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::unordered_map<int, std::size_t> layer_temp_marks;

    // Cache whether overlapped gradient reduction is enabled (avoids per-layer check overhead)
    const bool use_overlapped_grad_reduce = grads.is_overlapped_enabled();

    auto extract_layer = [&](const std::string& name, int& layer_idx) -> bool {
        std::string_view view{name};
        if (starts_with(view, "d_")) {
            view = view.substr(2);
        }
        if (starts_with(view, kSavedPrefix)) {
            view = view.substr(kSavedPrefix.size());
        }
        std::string field;
        return parse_block_param(view, layer_idx, field);
    };
    auto maybe_start_layer = [&](const Operation& op) {
        for (const auto& name : op.inputs) {
            int layer_idx = -1;
            if (extract_layer(name, layer_idx)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();

                    // Backward goes from last to first layer, so prefetch l-1 while processing l
                    if (layer_idx != bwd_current_layer) {
                        if (bwd_current_layer >= 0) {
                            wait_for_prefetch(bwd_current_layer, rs.MainStream);
                        }
                        bwd_current_layer = layer_idx;

                        const int prev_layer = layer_idx - 1;
                        if (prev_layer >= 0 && prev_layer != bwd_last_prefetched) {
                            prefetch_layer_weights(prev_layer, rs.side_stream());
                            bwd_last_prefetched = prev_layer;
                        }

                        // Prefetch residual from CPU if offloading is enabled
                        if (rs.has_residual_offloading() && !bwd_capturing) {
                            rs.fetch_residual(layer_idx, rs.side_stream());
                            // Also prefetch previous layer's residual for recompute
                            if (prev_layer >= 0) {
                                rs.fetch_residual(prev_layer, rs.side_stream());
                            }
                        }
                    }
                }
                return;
            }
        }
        for (const auto& name : op.outputs) {
            int layer_idx = -1;
            if (extract_layer(name, layer_idx)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();

                    // Backward goes from last to first layer, so prefetch l-1 while processing l
                    if (layer_idx != bwd_current_layer) {
                        if (bwd_current_layer >= 0) {
                            wait_for_prefetch(bwd_current_layer, rs.MainStream);
                        }
                        bwd_current_layer = layer_idx;

                        const int prev_layer = layer_idx - 1;
                        if (prev_layer >= 0 && prev_layer != bwd_last_prefetched) {
                            prefetch_layer_weights(prev_layer, rs.side_stream());
                            bwd_last_prefetched = prev_layer;
                        }

                        // Prefetch residual from CPU if offloading is enabled
                        if (rs.has_residual_offloading() && !bwd_capturing) {
                            rs.fetch_residual(layer_idx, rs.side_stream());
                            // Also prefetch previous layer's residual for recompute
                            if (prev_layer >= 0) {
                                rs.fetch_residual(prev_layer, rs.side_stream());
                            }
                        }
                    }
                }
                return;
            }
        }
    };
    auto finish_layer = [&](int layer_idx) {
        auto cp_it = layer_checkpoints.find(layer_idx);
        if (cp_it == layer_checkpoints.end()) {
            return;
        }
        rs.Stack.restore(cp_it->second);
        auto mark_it = layer_temp_marks.find(layer_idx);
        if (mark_it != layer_temp_marks.end()) {
            if (st.temps.size() > mark_it->second) {
                st.temps.resize(mark_it->second);
            }
        }
        if (rs.large_bwd_temps_on_stack()) {
            auto& grads_layer = rs.simplified_grads(layer_idx);
            grads_layer.d_qkv.Data = nullptr;
            grads_layer.d_mlp_up.Data = nullptr;
            grads_layer.d_swiglu.Data = nullptr;
        }
        if (rs.ffn_temps_on_stack()) {
            auto& acts = rs.simplified_acts(layer_idx);
            acts.mlp_up.Data = nullptr;
            acts.swiglu.Data = nullptr;
        }
        rs.scratch().cudnn_workspace.Data = nullptr;

        // Release residual buffer after backward pass is done with this layer
        if (rs.has_residual_offloading() && !bwd_capturing) {
            rs.release_residual(layer_idx, rs.MainStream);
        }

        // Notify gradient manager that this layer's backward is complete.
        // This triggers overlapped all-reduce/reduce-scatter for multi-GPU training.
        if (!bwd_capturing && use_overlapped_grad_reduce) {
            grads.notify_block(layer_idx, rs.MainStream, comm);
        }

        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    const bool use_graphs = use_graphs_enabled;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const int graph_idx = (micro_step > 0) ? 1 : 0;
    const bool capturing = use_graphs && mBackwardGraph[graph_idx] == nullptr;
    bwd_capturing = capturing;  // Set the shared flag for lambdas

    // Prefetch last layer's weights for backward
    if (mPrefetchEnabled && config.NumLayers > 0 && !capturing) {
        prefetch_layer_weights(config.NumLayers - 1, rs.side_stream());
        bwd_last_prefetched = config.NumLayers - 1;
    }

    // Prefetch last layer's residual from CPU for backward pass
    if (rs.has_residual_offloading() && config.NumLayers > 0 && !capturing) {
        rs.fetch_residual(config.NumLayers - 1, rs.side_stream());
    }

    auto run_ops_range = [&](std::size_t begin, std::size_t end) {
        end = std::min(end, mBackward->operations.size());
        for (std::size_t op_idx = begin; op_idx < end; ++op_idx) {
            const auto& op = mBackward->operations[op_idx];
            const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
            if (trace_ops) {
                fprintf(stderr, "[DSL TRACE] bwd op %zu/%zu id=%s type=%s\n",
                        op_idx, mBackward->operations.size(), op.id.c_str(), op_type.c_str());
                fflush(stderr);
            }
            maybe_start_layer(op);
            if (recompute_any) {
                int layer_idx = -1;
                bool found = false;
                for (const auto& name : op.inputs) {
                    if (extract_layer(name, layer_idx)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    for (const auto& name : op.outputs) {
                        if (extract_layer(name, layer_idx)) {
                            found = true;
                            break;
                        }
                    }
                }
                if (found) {
                    recompute_layer(layer_idx);
                }
            }

            if (op_type == "view") {
                Tensor& src = get_tensor(st, op.inputs.at(0), mSaved);
                auto shape = resolve_view_shape(op, st.shape_env, st, mSaved);
                Tensor view = view_tensor(src, shape);
                auto it = st.tensors.find(op.outputs.at(0));
                if (it != st.tensors.end()) {
                    it->second = view;
                } else {
                    st.tensors.emplace(op.outputs.at(0), view);
                }
                continue;
            }

            if (op_type == "add") {
                Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
                if (a.DType != b.DType || a.nelem() != b.nelem()) {
                    throw std::runtime_error("DSL graph executor: add op shape or dtype mismatch");
                }
                std::vector<long> shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
                Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
                vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, rs.MainStream);
                continue;
            }

            if (op_type == "matmul_backward") {
                Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& a = get_tensor(st, op.inputs.at(1), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(2), mSaved);

                const std::string& dA_name = (op.outputs.size() > 0) ? op.outputs.at(0) : "";
                const std::string& dB_name = (op.outputs.size() > 1) ? op.outputs.at(1) : "";

                Tensor* dA_ptr = nullptr;
                Tensor* dB_ptr = nullptr;
                if (!dA_name.empty()) {
                    dA_ptr = &ensure_tensor(st, dA_name, a.DType,
                                            {a.Sizes[0], a.Sizes[1]});
                }
                if (!dB_name.empty()) {
                    dB_ptr = &ensure_tensor(st, dB_name, b.DType,
                                            {b.Sizes[0], b.Sizes[1]});
                }

                if (!dA_ptr && !dB_ptr) {
                    continue;
                }

                EMMTranspose mode = parse_transpose(op.attrs);
                bool used_recipe = false;

                // Determine op kind for quant buffers (only for NT layout).
                int layer_idx = -1;
                auto op_kind = matmul_op_from_weight(op.inputs.at(2), layer_idx);
                const bool allow_quant = op_kind.has_value() && allow_quant_layer(mOptions, config, layer_idx);
                const modules::MatmulOp matmul_op = op_kind.value_or(modules::MatmulOp::LMHead);

                bool do_accumulate = dB_ptr && (accumulate_tensors.count(dB_name) > 0);
                bool skip_weight_grad = true;
                if (dB_ptr) {
                    if (auto base_name = base_param_from_grad(dB_name)) {
                        if (mWeights.has(*base_name) && mWeights.is_trainable(*base_name)) {
                            skip_weight_grad = false;
                        }
                    }
                }

                if (mOptions.TrainingRecipe && mode == EMMTranspose::NT && a.Sizes[0] == B * T) {
                    const recipes::Recipe& recipe = *mOptions.TrainingRecipe;

                    Tensor dA_tmp{};
                    Tensor dB_tmp{};
                    Tensor* dA_use = dA_ptr;
                    Tensor* dB_use = dB_ptr;
                    if (!dA_use) {
                        dA_tmp = rs.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
                        st.temps.push_back(dA_tmp);
                        dA_use = &dA_tmp;
                    }
                    if (!dB_use) {
                        dB_tmp = rs.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
                        st.temps.push_back(dB_tmp);
                        dB_use = &dB_tmp;
                    }

                    Tensor* weight_ptr = &b;
                    if (allow_quant) {
                        if (const Tensor* cached = get_fp8_cached_weight(op.inputs.at(2), b, rs.MainStream)) {
                            weight_ptr = const_cast<Tensor*>(cached);
                        }
                    }

                    modules::MatmulContext ctx;
                    ctx.dinp = dA_use;
                    ctx.dweight = dB_use;
                    ctx.dout = &d_out;
                    ctx.inp = &a;
                    ctx.weight = weight_ptr;
                    ctx.B = static_cast<int>(B);
                    ctx.T = static_cast<int>(T);
                    ctx.C_in = static_cast<int>(a.Sizes[1]);
                    ctx.C_out = static_cast<int>(b.Sizes[0]);
                    ctx.run_state = &rs;
                    ctx.stream = rs.MainStream;
                    ctx.layer_idx = layer_idx;
                    ctx.op = matmul_op;
                    ctx.accumulate = do_accumulate;
                    ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
                    ctx.allow_fp8 = allow_quant;
                    ctx.allow_fp4 = allow_quant;
                    if (allow_quant) {
                        ctx.dout_quant = fp8_grad_buffer(rs, matmul_op);
                        if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                            ctx.allow_fp8 = false;
                        }

                        // FP4 cached transposed weights for dgrad (for NVFP4 recipe on Blackwell+)
                        if (const auto* fp4_cache = get_fp4_cached_weight_transposed(op.inputs.at(2), b, rs.MainStream)) {
                            ctx.cached_fp4_data = &fp4_cache->data;
                            ctx.cached_fp4_scales = &fp4_cache->scales;
                            ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
                        }
                    }

                    recipe.backward_matmul(ctx);
                    used_recipe = true;
                }

                if (!used_recipe) {
                    // Fallback: emulate matmul backward with explicit matmuls.
                    EMMTranspose mode_dA = EMMTranspose::NN;
                    EMMTranspose mode_dB = EMMTranspose::NN;
                    switch (mode) {
                        case EMMTranspose::NN:
                            mode_dA = EMMTranspose::NT;
                            mode_dB = EMMTranspose::TN;
                            break;
                        case EMMTranspose::NT:
                            mode_dA = EMMTranspose::NN;
                            mode_dB = EMMTranspose::TN;
                            break;
                        case EMMTranspose::TN:
                            mode_dA = EMMTranspose::NT;
                            mode_dB = EMMTranspose::NN;
                            break;
                        case EMMTranspose::TT:
                            mode_dA = EMMTranspose::TT;
                            mode_dB = EMMTranspose::TT;
                            break;
                    }

                    if (dA_ptr) {
                        int M = 0, N = 0, K = 0;
                        matmul_dims(d_out, b, mode_dA, M, N, K);
                        EMMTranspose mode_col = swap_transpose(mode_dA);
                        matmul(*dA_ptr, b, d_out, std::nullopt, nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               N, M, K, mode_col, false, rs.MainStream);
                    }
                    if (dB_ptr && !skip_weight_grad) {
                        int M = 0, N = 0, K = 0;
                        matmul_dims(d_out, a, mode_dB, M, N, K);
                        EMMTranspose mode_col = swap_transpose(mode_dB);
                        matmul(*dB_ptr, a, d_out, std::nullopt, nullptr, nullptr,
                               rs.CublasLtHandle, rs.CuBlasWorkspace,
                               N, M, K, mode_col, do_accumulate, rs.MainStream);
                    }
                }

                // Map dA output to simplified_grads for LoRA hooks
                // The DSL autodiff generates generic names like "d_blocks[N].view_X" but the
                // LoRA backward hooks expect the tensors to be in simplified_grads.
                if (dA_ptr && op_kind.has_value() && layer_idx >= 0) {
                    auto& grads = rs.simplified_grads(layer_idx);
                    switch (*op_kind) {
                        case modules::MatmulOp::MLPDown:
                            // d_swiglu = dA from MLP down backward
                            if (!grads.d_swiglu.Data) {
                                // Tensor was allocated on stack, update the Data pointer
                                grads.d_swiglu.Data = dA_ptr->Data;
                            }
                            break;
                        case modules::MatmulOp::MLPUp:
                            // d_ln2 = dA from MLP up backward (though this goes through swiglu_backward first)
                            break;
                        case modules::MatmulOp::AttnOut:
                            // d_att = dA from attention out backward
                            if (!grads.d_att.Data) {
                                grads.d_att.Data = dA_ptr->Data;
                            }
                            break;
                        case modules::MatmulOp::QKV:
                            // d_ln1 = dA from QKV backward
                            if (!grads.d_ln1.Data) {
                                grads.d_ln1.Data = dA_ptr->Data;
                            }
                            break;
                        default:
                            break;
                    }
                }

                // Hook invocation for backward (LoRA is now handled via hooks from DslModel)
                if (hook && *hook) {
                    int lora_layer_idx = -1;
                    std::string field;
                    if (parse_block_param(op.inputs.at(2), lora_layer_idx, field)) {
                        modules::BackwardHookPoint hook_point;
                        bool should_invoke = false;
                        // For backward, we invoke the "After" hook point which corresponds
                        // to when LoRA gradients should be computed
                        if (field == "qkv_weight") {
                            hook_point = modules::BackwardHookPoint::AfterQKVBackward;
                            should_invoke = true;
                        } else if (field == "out_weight") {
                            hook_point = modules::BackwardHookPoint::AfterAttnOutBackward;
                            should_invoke = true;
                        } else if (field == "mlp_up_weight") {
                            hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                            should_invoke = true;
                        } else if (field == "mlp_down_weight") {
                            hook_point = modules::BackwardHookPoint::AfterMLPDownBackward;
                            should_invoke = true;
                        }
                        if (should_invoke) {
                            invoke_backward_hook(lora_layer_idx, do_accumulate, hook_point, rs.MainStream, hook);
                        }
                    }
                }
                continue;
            }

            if (op_type == "matmul" || op_type == "matmul_bias") {
                if (op.outputs.at(0).empty()) {
                    continue;
                }
                Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
                Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
                if (a.Rank != 2 || b.Rank != 2) {
                    throw std::runtime_error(
                        "DSL graph executor: matmul expects rank-2 tensors (op=" + op.id +
                        ", a=" + op.inputs.at(0) + " rank=" + std::to_string(a.Rank) +
                        " shape=" + tensor_shape_str(a) +
                        ", b=" + op.inputs.at(1) + " rank=" + std::to_string(b.Rank) +
                        " shape=" + tensor_shape_str(b) + ")"
                    );
                }
                EMMTranspose mode = parse_transpose(op.attrs);
                int M = 0, N = 0, K = 0;
                matmul_dims(a, b, mode, M, N, K);
                const std::string& out_name = op.outputs.at(0);
                bool do_accumulate = accumulate_tensors.count(op.outputs.at(0)) > 0;
                std::optional<Tensor> bias;
                if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                    bias = get_tensor(st, op.inputs.at(2), mSaved);
                }
                bool skip_weight_grad = false;
                bool is_param_grad = false;
                if (auto param_name = base_param_from_grad(out_name)) {
                    if (mWeights.has(*param_name)) {
                        is_param_grad = true;
                        if (!mWeights.is_trainable(*param_name)) {
                            skip_weight_grad = true;
                        }
                    }
                }
                // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
                // swapping M/N, and swapping transpose flags.
                EMMTranspose mode_col = swap_transpose(mode);
                if (!skip_weight_grad) {
                    std::vector<long> shape{M, N};
                    Tensor& out = ensure_tensor(st, out_name, a.DType, shape);
                    matmul(out, b, a, bias, nullptr, nullptr,
                           rs.CublasLtHandle, rs.CuBlasWorkspace,
                           N, M, K, mode_col, do_accumulate, rs.MainStream);
                }

                // Hook invocation for backward (LoRA is now handled via hooks from DslModel)
                // Note: This path is for regular matmul ops that compute weight gradients
                if (hook && *hook) {
                    if (auto base_name = base_param_from_grad(out_name)) {
                        int layer_idx = -1;
                        std::string field;
                        if (parse_block_param(*base_name, layer_idx, field)) {
                            modules::BackwardHookPoint hook_point;
                            bool should_invoke = false;
                            if (field == "qkv_weight") {
                                hook_point = modules::BackwardHookPoint::AfterQKVBackward;
                                should_invoke = true;
                            } else if (field == "out_weight") {
                                hook_point = modules::BackwardHookPoint::AfterAttnOutBackward;
                                should_invoke = true;
                            } else if (field == "mlp_up_weight") {
                                hook_point = modules::BackwardHookPoint::AfterMLPUpBackward;
                                should_invoke = true;
                            } else if (field == "mlp_down_weight") {
                                hook_point = modules::BackwardHookPoint::AfterMLPDownBackward;
                                should_invoke = true;
                            }
                            if (should_invoke) {
                                invoke_backward_hook(layer_idx, do_accumulate, hook_point, rs.MainStream, hook);
                            }
                        }
                    }
                }
                continue;
            }

        if (op_type == "bias_add_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_out);
            }
            int Bv = 1;
            int Tv = 1;
            int OC = 1;
            if (d_out.Rank == 2) {
                Bv = static_cast<int>(d_out.Sizes[0]);
                Tv = 1;
                OC = static_cast<int>(d_out.Sizes[1]);
            } else {
                Bv = static_cast<int>(d_out.Sizes[0]);
                Tv = static_cast<int>(d_out.Sizes[1]);
                OC = static_cast<int>(d_out.Sizes[2]);
            }
            const bool bias_trainable = mWeights.has(op.inputs.at(1)) && mWeights.is_trainable(op.inputs.at(1));
            if (bias_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                Tensor& d_bias = ensure_tensor(st, op.outputs.at(1), d_out.DType, {static_cast<long>(OC)});
                const bool do_accumulate = accumulate_tensors.count(op.outputs.at(1)) > 0;
                const int scratch_bytes = get_bias_backward_scratch_size(d_out.DType, OC, rs.DeviceProp);
                Tensor scratch = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(scratch_bytes / sizeof(float))});
                st.temps.push_back(scratch);
                if (do_accumulate) {
                    Tensor tmp = rs.temp_alloc(d_out.DType, {static_cast<long>(OC)});
                    st.temps.push_back(tmp);
                    backward_bias(tmp, d_out, nullptr, nullptr, scratch,
                                  Bv, Tv, OC, rs.DeviceProp, rs.MainStream);
                    vector_add_sr(d_bias, d_bias, tmp, 1.0f, static_cast<long>(d_bias.nelem()), 0,
                                  rs.MainStream);
                } else {
                    backward_bias(d_bias, d_out, nullptr, nullptr, scratch,
                                  Bv, Tv, OC, rs.DeviceProp, rs.MainStream);
                }
            }
            continue;
        }

        if (op_type == "swiglu_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& inp = get_tensor(st, op.inputs.at(1), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            const long D = inp.Sizes[2] / 2;
            Tensor& d_inp = ensure_tensor(st, op.outputs.at(0), inp.DType, {inp.Sizes[0], inp.Sizes[1], inp.Sizes[2]});
            float* abs_max_ptr = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_mlp_up.abs_max()
                : nullptr;
            swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                            static_cast<int>(inp.Sizes[0]), static_cast<int>(inp.Sizes[1]),
                            static_cast<int>(D), rs.MainStream);
            continue;
        }

        // Fused matmul + swiglu backward
        // d_out: gradient w.r.t. swiglu output (M, D)
        // up_output: saved matmul output (M, 2*D) from forward
        // a: input to matmul from forward (M, K)
        // b: weight matrix (2*D, K) for NT mode
        // Outputs: d_a (M, K), d_b (2*D, K)
        if (op_type == "matmul_swiglu_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& a = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& up_output = get_tensor(st, op.inputs.at(3), mSaved);

            const std::string& d_a_name = (op.outputs.size() > 0) ? op.outputs.at(0) : "";
            const std::string& d_b_name = (op.outputs.size() > 1) ? op.outputs.at(1) : "";

            if (d_a_name.empty() && d_b_name.empty()) {
                continue;
            }

            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            const long D = N / 2;

            // Step 1: swiglu backward to get d_up (M, 2*D)
            // d_out is (M, D), up_output is (M, 2*D)
            Tensor d_up = rs.temp_alloc(d_out.DType, {static_cast<long>(M), static_cast<long>(N)});
            st.temps.push_back(d_up);

            // Reshape for swiglu_backward which expects (B, T, 2*D)
            Tensor d_out_3d = view_tensor(d_out, {B, T, D});
            Tensor up_3d = view_tensor(up_output, {B, T, static_cast<long>(N)});
            Tensor d_up_3d = view_tensor(d_up, {B, T, static_cast<long>(N)});

            float* abs_max_ptr = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_mlp_up.abs_max()
                : nullptr;
            swiglu_backward(d_up_3d, d_out_3d, up_3d, abs_max_ptr,
                            static_cast<int>(B), static_cast<int>(T),
                            static_cast<int>(D), rs.MainStream);

            // Step 2: matmul backward
            // d_a = d_up @ b (if needed)
            // d_b = d_up^T @ a (if needed)
            bool do_accumulate = accumulate_tensors.count(d_b_name) > 0;

            if (!d_a_name.empty()) {
                Tensor& d_a = ensure_tensor(st, d_a_name, a.DType, {static_cast<long>(M), static_cast<long>(K)});
                // For NT mode: d_a = d_up @ b (no transpose on either)
                EMMTranspose mode_da = EMMTranspose::NN;
                matmul(d_a, b, d_up, std::nullopt, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       K, M, N, swap_transpose(mode_da), false, rs.MainStream);
            }

            if (!d_b_name.empty()) {
                // Check if weight is trainable
                bool skip_weight_grad = false;
                if (auto param_name = base_param_from_grad(d_b_name)) {
                    if (mWeights.has(*param_name) && !mWeights.is_trainable(*param_name)) {
                        skip_weight_grad = true;
                    }
                }
                if (!skip_weight_grad) {
                    Tensor& d_b = ensure_tensor(st, d_b_name, b.DType, {static_cast<long>(N), static_cast<long>(K)});
                    // For NT mode: d_b = d_up^T @ a -> (N, M) @ (M, K) = (N, K)
                    EMMTranspose mode_db = EMMTranspose::TN;
                    matmul(d_b, a, d_up, std::nullopt, nullptr, nullptr,
                           rs.CublasLtHandle, rs.CuBlasWorkspace,
                           K, N, M, swap_transpose(mode_db), do_accumulate, rs.MainStream);
                }
            }
            continue;
        }

        if (op_type == "rope_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(2), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            Tensor& d_inp = ensure_tensor(st, op.outputs.at(0), d_out.DType,
                                          {d_out.Sizes[0], d_out.Sizes[1], d_out.Sizes[2], d_out.Sizes[3]});
            int Hq = static_cast<int>(config.NumQueryHeads);
            int Hkv = static_cast<int>(config.NumKeyValHeads);
            int Hs = static_cast<int>(config.head_size());
            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }
            float* abs_max_ptr = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_qkv.abs_max()
                : nullptr;
            rope_backward(d_inp, d_out, freqs, reinterpret_cast<int*>(pos_ids.Data), abs_max_ptr,
                          static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            continue;
        }

        if (op_type == "qkv_qk_norm_rope_backward") {
            Tensor& d_qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& qkv = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& q_norm = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& k_norm = get_tensor(st, op.inputs.at(3), mSaved);
            Tensor& q_rstd = get_tensor(st, op.inputs.at(4), mSaved);
            Tensor& k_rstd = get_tensor(st, op.inputs.at(5), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(6), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(7), mSaved);

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            const int qkv_channels = Hs * (Hq + 2 * Hkv);

            Tensor d_qkv_view = (d_qkv.Rank == 4)
                ? view_tensor(d_qkv, {B, T, qkv_channels})
                : d_qkv;
            Tensor qkv_view = (qkv.Rank == 4)
                ? view_tensor(qkv, {B, T, qkv_channels})
                : qkv;

            int rotary_dim = Hs;
            if (auto* rd_attr = find_attr(op.attrs, "rotary_dim")) {
                if (auto v = attr_int(*rd_attr)) {
                    rotary_dim = static_cast<int>(*v);
                } else if (auto s = attr_string(*rd_attr)) {
                    rotary_dim = static_cast<int>(resolve_dim(Dim::symbolic(*s), st.shape_env));
                }
            }

            const bool q_norm_trainable = mWeights.has(op.inputs.at(2)) && mWeights.is_trainable(op.inputs.at(2));
            const bool k_norm_trainable = mWeights.has(op.inputs.at(3)) && mWeights.is_trainable(op.inputs.at(3));
            const bool rope_fusable = (rotary_dim > 0)
                && ((Hs % 2) == 0)
                && (((Hs / 2) % 32) == 0)
                && (freqs.Rank >= 2)
                && (freqs.Sizes[1] >= Hs);
            const bool disable_fused_qk_rope_bwd = env_enabled("SUROGATE_DISABLE_FUSED_QK_ROPE_BWD");
            float* d_qkv_abs_max = rs.has_grad_quants()
                ? rs.simplified_quant_grads().d_qkv.abs_max()
                : nullptr;

            if (rope_fusable && !disable_fused_qk_rope_bwd) {
                // Fused QK norm + RoPE backward (matches modular path).
                if (d_qkv_abs_max) {
                    CUDA_CHECK(cudaMemsetAsync(d_qkv_abs_max, 0, sizeof(float), rs.MainStream));
                }
                qkv_head_rmsnorm_rope_backward_dx(
                    d_qkv_view, qkv_view, q_norm, q_rstd,
                    freqs, reinterpret_cast<int*>(pos_ids.Data),
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                    rs.MainStream, d_qkv_abs_max);
                qkv_head_rmsnorm_rope_backward_dx(
                    d_qkv_view, qkv_view, k_norm, k_rstd,
                    freqs, reinterpret_cast<int*>(pos_ids.Data),
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                    rs.MainStream, d_qkv_abs_max);
                if (q_norm_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    Tensor& d_q_norm = ensure_tensor(st, op.outputs.at(1), q_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(1)) > 0;
                    qkv_head_rmsnorm_rope_backward_dweight(
                        d_q_norm, d_qkv_view, qkv_view, q_norm,
                        freqs, reinterpret_cast<int*>(pos_ids.Data),
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                        acc, rs.MainStream);
                }
                if (k_norm_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                    Tensor& d_k_norm = ensure_tensor(st, op.outputs.at(2), k_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(2)) > 0;
                    qkv_head_rmsnorm_rope_backward_dweight(
                        d_k_norm, d_qkv_view, qkv_view, k_norm,
                        freqs, reinterpret_cast<int*>(pos_ids.Data),
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                        acc, rs.MainStream);
                }
                if (d_qkv_abs_max) {
                    const int v_offset = (Hq + Hkv) * Hs;
                    const int v_channels = Hkv * Hs;
                    qkv_abs_max_slice(d_qkv_view, static_cast<int>(B), static_cast<int>(T), qkv_channels,
                                      v_offset, v_channels, d_qkv_abs_max, rs.MainStream);
                }
            } else {
                // Undo RoPE on gradients and activations (in-place) before QK RMSNorm backward.
                rope_backward(d_qkv, d_qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), d_qkv_abs_max,
                              static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
                rope_backward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                              static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);

                if (q_norm_trainable && op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    Tensor& d_q_norm = ensure_tensor(st, op.outputs.at(1), q_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(1)) > 0;
                    qkv_head_rmsnorm_backward_dweight(
                        d_q_norm, d_qkv_view, qkv_view, q_norm,
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                        acc, rs.MainStream);
                }
                if (k_norm_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                    Tensor& d_k_norm = ensure_tensor(st, op.outputs.at(2), k_norm.DType, {Hs});
                    const bool acc = accumulate_tensors.count(op.outputs.at(2)) > 0;
                    qkv_head_rmsnorm_backward_dweight(
                        d_k_norm, d_qkv_view, qkv_view, k_norm,
                        static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                        acc, rs.MainStream);
                }

                // Backward dx for Q/K RMSNorm (in-place on d_qkv_view).
                qkv_head_rmsnorm_backward_dx(
                    d_qkv_view, qkv_view, q_norm, q_rstd,
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hq, Hs, 0,
                    rs.MainStream);
                qkv_head_rmsnorm_backward_dx(
                    d_qkv_view, qkv_view, k_norm, k_rstd,
                    static_cast<int>(B), static_cast<int>(T), qkv_channels, Hkv, Hs, Hq * Hs,
                    rs.MainStream);
            }
            if (!op.outputs.at(0).empty()) {
                st.tensors.emplace(op.outputs.at(0), d_qkv);
            }
            continue;
        }

        if (op_type == "flash_attention_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& out = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& lse = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& qkv = get_tensor(st, op.inputs.at(3), mSaved);
            if (op.outputs.at(0).empty()) {
                continue;
            }
            Tensor& d_qkv = ensure_tensor(st, op.outputs.at(0), qkv.DType,
                                          {qkv.Sizes[0], qkv.Sizes[1], qkv.Sizes[2], qkv.Sizes[3]});

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            if (!rs.scratch().cudnn_workspace.Data) {
                rs.temp_acquire(rs.scratch().cudnn_workspace);
                st.temps.push_back(rs.scratch().cudnn_workspace);
            }
            attention_backward_cudnn(d_qkv, lse, out, d_out, qkv,
                                     rs.scratch().cudnn_workspace, rs.CudnnHandle,
                                     static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rs.MainStream);
            continue;
        }

        if (op_type == "zeros") {
            auto* shape_attr = find_attr(op.attrs, "shape");
            if (!shape_attr) {
                throw std::runtime_error("DSL graph executor: zeros missing shape attr");
            }
            auto shape = resolve_attr_shape(*shape_attr, st.shape_env);
            ETensorDType dtype = ETensorDType::BF16;
            if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                if (auto s = attr_string(*dtype_attr)) {
                    dtype = dtype_from_str(*s);
                }
            }
            Tensor& out = ensure_tensor(st, op.outputs.at(0), dtype, shape);
            fill_zero(out, rs.MainStream);
            st.zero_tensors.insert(op.outputs.at(0));
            continue;
        }

        if (op_type == "fused_residual_rmsnorm_backward" || op.name == "fused_residual_rmsnorm_backward") {
            Tensor& d_y = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& residual_out = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor* d_residual_next = nullptr;
            if (!op.inputs.at(1).empty()) {
                d_residual_next = &get_tensor(st, op.inputs.at(1), mSaved);
            }

            Tensor& weight = resolve_param_tensor(st, op.inputs.at(3));

            Tensor& d_input = ensure_tensor(st, op.outputs.at(1), d_y.DType, {B, T, config.HiddenSize});
            Tensor* d_residual_out = nullptr;
            if (!op.outputs.at(0).empty()) {
                d_residual_out = &ensure_tensor(st, op.outputs.at(0), d_y.DType, {B, T, config.HiddenSize});
            }

            Tensor* d_residual = d_residual_next;
            if (!d_residual) {
                std::string zero_name = op.id + "_dres_zero";
                Tensor& d_residual_zero = ensure_tensor(st, zero_name, d_y.DType, {B, T, config.HiddenSize});
                fill_zero(d_residual_zero, rs.MainStream);
                d_residual = &d_residual_zero;
            }

            const bool weight_trainable = mWeights.has(op.inputs.at(3)) && mWeights.is_trainable(op.inputs.at(3));
            Tensor* d_weight = nullptr;
            bool skip_weight_grad = true;
            if (weight_trainable && op.outputs.size() > 2 && !op.outputs.at(2).empty()) {
                d_weight = &ensure_tensor(st, op.outputs.at(2), weight.DType, {config.HiddenSize});
                skip_weight_grad = false;
            }
            Tensor dummy_weight{};
            if (!d_weight) {
                dummy_weight = rs.temp_alloc(weight.DType, {config.HiddenSize});
                st.temps.push_back(dummy_weight);
                d_weight = &dummy_weight;
            }

            const bool skip_weight = skip_weight_grad;
            float* abs_max_ptr = nullptr;
            if (rs.has_grad_quants()) {
                bool use_res_att = false;
                if (op.outputs.size() > 1 && !op.outputs.at(1).empty()) {
                    use_res_att = op.outputs.at(1).find("res_att") != std::string::npos;
                }
                if (!use_res_att) {
                    use_res_att = op.inputs.at(3).find("ln2_weight") != std::string::npos;
                }
                abs_max_ptr = use_res_att
                    ? rs.simplified_quant_grads().d_res_att.abs_max()
                    : rs.simplified_quant_grads().d_res_ffn.abs_max();
            }
            rmsnorm_backward(d_input, *d_weight, rs.scratch().rmsnorm_scratch,
                             *d_residual, d_y, residual_out, weight,
                             get_tensor(st, op.inputs.at(4), mSaved),
                             abs_max_ptr,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             rs.DeviceProp, rs.MainStream,
                             skip_weight);
            if (d_residual_out && op.outputs.at(0) != op.outputs.at(1)) {
                CUDA_CHECK(cudaMemcpyAsync(d_residual_out->Data, d_input.Data, d_input.bytes(),
                                           cudaMemcpyDeviceToDevice, rs.MainStream));
            }
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.inputs.at(3), layer_idx, field) && field == "ln1_weight") {
                finish_layer(layer_idx);
            }
            continue;
        }

        if (op_type == "embedding_backward" || op.name == "embedding_backward") {
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            if (op.outputs.empty() || op.outputs.at(0).empty()) {
                continue;
            }
            auto it = st.tensors.find(op.outputs.at(0));
            if (it == st.tensors.end()) {
                continue;
            }
            Tensor& d_emb = it->second;
            encoder_backward(d_emb,
                             rs.scratch().encoder_bwd_scratch,
                             rs.scratch().encoder_bwd_indices,
                             rs.scratch().encoder_bwd_info,
                             d_out,
                             rs.Inputs,
                             mLastInputsCpu,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             next_rng_seed(),
                             rs.MainStream,
                             rs.side_stream_event(),
                             rs.side_stream());
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported backward op " + op.name);
        }
    };

    if (use_graphs) {
        auto run_graph_ops = [&]() { run_ops_range(0, mBackwardGraphCut); };
        trace_or_execute_cuda_graph_with_stack(run_graph_ops, rs.MainStream, mBackwardGraph[graph_idx], use_graphs, rs.Stack, mBackwardCheckpoint[graph_idx]);
        if (mBackwardGraphCut < mBackward->operations.size()) {
            run_ops_range(mBackwardGraphCut, mBackward->operations.size());
        }
    } else {
        run_ops_range(0, mBackward->operations.size());
    }

    if (!use_graphs || capturing) {
        free_temps(st);
        rs.temp_free(rs.non_block_activations().output);
    }
}

// ============================================================================
// Optimized backward recomputation support
// ============================================================================

void GraphExecutor::init_recompute_flags() {
    const bool disable_recompute_block = mRunState.is_lora_only_mode() && !mOptions.RecomputeLoRA;
    const bool recompute_block = mOptions.RecomputeBlock && !disable_recompute_block;

    // Attention path recomputation
    mRecomputeFlags.att = mOptions.RecomputeAtt || recompute_block;
    mRecomputeFlags.qkv = mOptions.RecomputeQKV || mRecomputeFlags.att;
    mRecomputeFlags.qk_norm = mOptions.RecomputeQKNorm || mRecomputeFlags.qkv;
    mRecomputeFlags.rope = mOptions.RecomputeRoPE || mRecomputeFlags.qkv;
    mRecomputeFlags.out_proj = mOptions.RecomputeOutProj || recompute_block;

    // FFN/MLP path recomputation
    mRecomputeFlags.ffn = mOptions.RecomputeFFN || recompute_block;
    mRecomputeFlags.swiglu = mOptions.RecomputeSwiGLu || mRecomputeFlags.ffn;
    mRecomputeFlags.mlp_down = mOptions.RecomputeMLPDown || recompute_block;

    // Normalization recomputation
    const bool recompute_rmsnorm = mOptions.RecomputeRMSNorm || recompute_block;
    mRecomputeFlags.ln1 = recompute_rmsnorm || mRecomputeFlags.qkv || recompute_block;
    mRecomputeFlags.ln2 = recompute_rmsnorm || mRecomputeFlags.out_proj || mRecomputeFlags.ffn || recompute_block;

    // Overall flag
    mRecomputeFlags.any = mRecomputeFlags.ln1 || mRecomputeFlags.qkv || mRecomputeFlags.att ||
                          mRecomputeFlags.ln2 || mRecomputeFlags.ffn || mRecomputeFlags.swiglu ||
                          mRecomputeFlags.out_proj || mRecomputeFlags.mlp_down;
}


void GraphExecutor::reset_recompute_graphs() {
    for (auto& graph : mRecomputeGraphs) {
        if (graph != nullptr) {
            cudaGraphExecDestroy(graph);
            graph = nullptr;
        }
    }
    mRecomputeGraphsInitialized = false;
}

void GraphExecutor::recompute_layer_optimized(int layer_idx, long B, long T, bool use_graph) {
    if (!mRecomputeFlags.any) return;
    if (layer_idx < 0 || layer_idx >= mConfig.NumLayers) return;

    auto& rs = mRunState;
    const auto& config = mConfig;
    const auto& flags = mRecomputeFlags;

    // Check if we can use cached CUDA graph
    if (use_graph && mRecomputeGraphsInitialized && layer_idx < static_cast<int>(mRecomputeGraphs.size())) {
        cudaGraphExec_t& graph = mRecomputeGraphs[layer_idx];
        if (graph != nullptr) {
            rs.Stack.restore(mRecomputeCheckpoints[layer_idx]);
            CUDA_CHECK(cudaGraphLaunch(graph, rs.MainStream));
            return;
        }
    }

    // Initialize graph arrays if needed
    if (use_graph && !mRecomputeGraphsInitialized) {
        mRecomputeGraphs.resize(static_cast<std::size_t>(config.NumLayers), nullptr);
        mRecomputeCheckpoints.resize(static_cast<std::size_t>(config.NumLayers));
        mRecomputeGraphsInitialized = true;
    }

    // Capture checkpoint for potential graph capture
    DeviceMemoryStack::Checkpoint checkpoint;
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size())) {
        checkpoint = rs.Stack.checkpoint();
        mRecomputeCheckpoints[layer_idx] = checkpoint;
    }

    // Start graph capture if enabled
    cudaGraph_t captured_graph = nullptr;
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size()) && mRecomputeGraphs[layer_idx] == nullptr) {
        CUDA_CHECK(cudaStreamBeginCapture(rs.MainStream, cudaStreamCaptureModeThreadLocal));
    }

    // ========================================================================
    // Recomputation logic (streamlined, no conditionals in hot path)
    // ========================================================================
    const int Bv = static_cast<int>(B);
    const int Tv = static_cast<int>(T);
    const int C = static_cast<int>(config.HiddenSize);
    const int D = static_cast<int>(config.IntermediateSize);
    const int Hq = static_cast<int>(config.NumQueryHeads);
    const int Hkv = static_cast<int>(config.NumKeyValHeads);
    const int Hs = static_cast<int>(config.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int MUp = 2 * D;

    auto& acts = rs.simplified_acts(layer_idx);

    // Helper to ensure activation buffer is allocated
    auto ensure_act = [&](Tensor& t) {
        if (!t.Data) {
            rs.temp_acquire(t);
        }
    };

    Tensor& res_ffn = rs.get_residual(layer_idx, rs.MainStream);

    // LN1 recomputation
    if (flags.ln1) {
        ensure_act(acts.ln1);
        ensure_act(acts.ln1_rstd);
        Tensor& ln1_weight = mWeights.get("blocks[" + std::to_string(layer_idx) + "].ln1_weight");
        rmsnorm_forward(acts.ln1, acts.ln1_rstd, res_ffn, ln1_weight, nullptr,
                        config.RmsNormEps, Bv, Tv, C, rs.MainStream);
    }

    // QKV + RoPE recomputation
    // Recipe-based matmul dispatch (FP8 disabled for recompute to avoid JIT vs delayed scaling mismatch)
    const bool allow_quant = allow_quant_layer(mOptions, config, layer_idx);
    const bool use_recipe_matmul = mOptions.TrainingRecipe != nullptr;

    if (flags.qkv) {
        ensure_act(acts.qkv);
        const std::string qkv_weight_name = "blocks[" + std::to_string(layer_idx) + "].qkv_weight";
        Tensor& qkv_weight = mWeights.get(qkv_weight_name);
        Tensor ln1_flat = view_tensor(acts.ln1, {B * T, C});
        Tensor qkv_flat = view_tensor(acts.qkv, {B * T, qkv_channels});

        // Check for bias
        std::string bias_name = "blocks[" + std::to_string(layer_idx) + "].qkv_bias";
        std::optional<Tensor> qkv_bias;
        if (mWeights.has(bias_name)) {
            qkv_bias = mWeights.get(bias_name);
        }

        if (use_recipe_matmul) {
            // Use recipe dispatch (FP8 disabled, FP4 OK since it uses per-block auto-scaling)
            const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
            modules::MatmulContext ctx;
            ctx.out = &qkv_flat;
            ctx.inp = &ln1_flat;
            ctx.weight = &qkv_weight;
            ctx.bias = qkv_bias ? &*qkv_bias : nullptr;
            ctx.B = Bv;
            ctx.T = Tv;
            ctx.C_in = C;
            ctx.C_out = qkv_channels;
            ctx.run_state = &rs;
            ctx.stream = rs.MainStream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::QKV;
            // Disable FP8 for recompute: JIT scaling produces different results than delayed scaling
            // from forward pass, causing gradient explosion. FP4 is OK (per-block auto-scaling).
            ctx.allow_fp8 = false;
            ctx.allow_fp4 = allow_quant && mOptions.fp4_enabled();
            recipe.forward_matmul(ctx);
        } else {
            // BF16 fallback when recipe not available
            int M = static_cast<int>(B * T);
            int N = qkv_channels;
            int K_dim = C;
            matmul(qkv_flat, qkv_weight, ln1_flat, qkv_bias, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K_dim, EMMTranspose::TN, false, rs.MainStream);
        }

        // NOTE: LoRA contributions are NOT recomputed here.
        // LoRA is applied once during forward pass via hooks, and the LoRA-modified
        // activations are saved. During recompute, we only redo base model ops
        // (matching the modular path's BlockExecutor::recompute behavior).

        // QK normalization + RoPE (fused when possible)
        if (config.UseQKNorm) {
            ensure_act(acts.q_rstd);
            ensure_act(acts.k_rstd);
            Tensor qkv_view = view_tensor(acts.qkv, {B, T, qkv_channels});
            const bool rope_fusable = ((Hs % 2) == 0)
                && (((Hs / 2) % 32) == 0)
                && (rs.non_block_activations().freq_cis.Rank >= 2)
                && (rs.non_block_activations().freq_cis.Sizes[1] >= Hs);
            if (rope_fusable) {
                qkv_qk_norm_rope_forward(qkv_view, acts.q_rstd, acts.k_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
                                         rs.non_block_activations().freq_cis,
                                         reinterpret_cast<int*>(rs.PositionIDs.Data),
                                         static_cast<float>(config.RmsNormEps),
                                         Bv, Tv, Hq, Hkv, Hs, rs.MainStream);
            } else {
                qkv_head_rmsnorm_forward(qkv_view, acts.q_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].q_norm_weight"),
                                         static_cast<float>(config.RmsNormEps),
                                         Bv, Tv, qkv_channels, Hq, Hs, 0, rs.MainStream);
                qkv_head_rmsnorm_forward(qkv_view, acts.k_rstd,
                                         mWeights.get("blocks[" + std::to_string(layer_idx) + "].k_norm_weight"),
                                         static_cast<float>(config.RmsNormEps),
                                         Bv, Tv, qkv_channels, Hkv, Hs, Hq * Hs, rs.MainStream);
                Tensor qkv_rope = (acts.qkv.Rank == 4)
                    ? acts.qkv
                    : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
                rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                             reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                             Bv, Tv, Hq, Hkv, Hs, Hs, rs.MainStream);
            }
        } else {
            Tensor qkv_rope = (acts.qkv.Rank == 4)
                ? acts.qkv
                : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
            rope_forward(qkv_rope, qkv_rope, rs.non_block_activations().freq_cis,
                         reinterpret_cast<int*>(rs.PositionIDs.Data), nullptr,
                         Bv, Tv, Hq, Hkv, Hs, Hs, rs.MainStream);
        }
    }

    // Attention recomputation
    if (flags.att) {
        ensure_act(acts.att);
        ensure_act(acts.lse);
        Tensor qkv_view = (acts.qkv.Rank == 4)
            ? acts.qkv
            : view_tensor(acts.qkv, {B, T, Hq + 2 * Hkv, Hs});
        if (!rs.scratch().cudnn_workspace.Data) {
            rs.temp_acquire(rs.scratch().cudnn_workspace);
        }
        Tensor att_out = view_tensor(acts.att, {B, T, Hq, Hs});
        Tensor lse_view = view_tensor(acts.lse, {B, Hq, T});
        attention_forward_cudnn(att_out, lse_view, qkv_view, rs.scratch().cudnn_workspace,
                                rs.CudnnHandle, Bv, Tv, Hq, Hkv, Hs, rs.MainStream);

        // Attention output projection recomputation
        if (flags.out_proj) {
            ensure_act(acts.att_out);
            const std::string out_weight_name = "blocks[" + std::to_string(layer_idx) + "].out_weight";
            Tensor& out_weight = mWeights.get(out_weight_name);
            Tensor att_flat = view_tensor(acts.att, {B * T, Hq * Hs});
            Tensor att_out_flat = view_tensor(acts.att_out, {B * T, C});

            if (use_recipe_matmul) {
                // Use recipe dispatch (FP8 disabled, FP4 OK)
                const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
                modules::MatmulContext ctx;
                ctx.out = &att_out_flat;
                ctx.inp = &att_flat;
                ctx.weight = &out_weight;
                ctx.bias = nullptr;
                ctx.B = Bv;
                ctx.T = Tv;
                ctx.C_in = Hq * Hs;
                ctx.C_out = C;
                ctx.run_state = &rs;
                ctx.stream = rs.MainStream;
                ctx.layer_idx = layer_idx;
                ctx.op = modules::MatmulOp::AttnOut;
                ctx.allow_fp8 = false;  // Disabled: JIT vs delayed scaling mismatch
                ctx.allow_fp4 = allow_quant && mOptions.fp4_enabled();
                recipe.forward_matmul(ctx);
            } else {
                // BF16 fallback
                int att_dim = Hq * Hs;
                matmul(att_out_flat, out_weight, att_flat, std::nullopt, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       C, static_cast<int>(B * T), att_dim, EMMTranspose::TN, false, rs.MainStream);
            }
            // NOTE: LoRA o_proj contribution is NOT recomputed (applied once during forward)
        }
    }

    // LN2 + residual recomputation
    if (flags.ln2) {
        ensure_act(acts.ln2);
        ensure_act(acts.ln2_rstd);
        Tensor& ln2_weight = mWeights.get("blocks[" + std::to_string(layer_idx) + "].ln2_weight");
        fused_residual_rmsnorm_forward(
            acts.residual_att, acts.ln2, acts.ln2_rstd,
            res_ffn, acts.att_out, ln2_weight, nullptr,
            config.RmsNormEps, static_cast<int>(B * T), C, rs.MainStream);
    }

    // FFN up projection recomputation
    if (flags.ffn) {
        ensure_act(acts.mlp_up);
        const std::string mlp_up_weight_name = "blocks[" + std::to_string(layer_idx) + "].mlp_up_weight";
        Tensor& mlp_up_weight = mWeights.get(mlp_up_weight_name);
        Tensor ln2_flat = view_tensor(acts.ln2, {B * T, C});
        Tensor mlp_up_flat = view_tensor(acts.mlp_up, {B * T, MUp});

        if (use_recipe_matmul) {
            // Use recipe dispatch (FP8 disabled, FP4 OK)
            const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
            modules::MatmulContext ctx;
            ctx.out = &mlp_up_flat;
            ctx.inp = &ln2_flat;
            ctx.weight = &mlp_up_weight;
            ctx.bias = nullptr;
            ctx.B = Bv;
            ctx.T = Tv;
            ctx.C_in = C;
            ctx.C_out = MUp;
            ctx.run_state = &rs;
            ctx.stream = rs.MainStream;
            ctx.layer_idx = layer_idx;
            ctx.op = modules::MatmulOp::MLPUp;
            ctx.allow_fp8 = false;  // Disabled: JIT vs delayed scaling mismatch
            ctx.allow_fp4 = allow_quant && mOptions.fp4_enabled();
            recipe.forward_matmul(ctx);
        } else {
            // BF16 fallback
            matmul(mlp_up_flat, mlp_up_weight, ln2_flat, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   MUp, static_cast<int>(B * T), C, EMMTranspose::TN, false, rs.MainStream);
        }
        // NOTE: LoRA MLP contribution is NOT recomputed (applied once during forward)
    }

    // SwiGLU recomputation
    if (flags.swiglu) {
        ensure_act(acts.swiglu);
        swiglu_forward(acts.swiglu, acts.mlp_up, nullptr,
                       Bv, Tv, D, rs.MainStream);
    }

    // End graph capture if we were capturing
    if (use_graph && layer_idx < static_cast<int>(mRecomputeGraphs.size()) && mRecomputeGraphs[layer_idx] == nullptr) {
        CUDA_CHECK(cudaStreamEndCapture(rs.MainStream, &captured_graph));
        CUDA_CHECK(cudaGraphInstantiate(&mRecomputeGraphs[layer_idx], captured_graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphDestroy(captured_graph));
        CUDA_CHECK(cudaGraphLaunch(mRecomputeGraphs[layer_idx], rs.MainStream));
    }
}

}  // namespace dsl
