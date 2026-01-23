// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Forward execution for DSL Graph executor.

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

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

}  // namespace


void GraphExecutor::execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full,
                                          const modules::ForwardHook* hook) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: missing forward graph");
    }

    auto& rs = mRunState;
    auto& weights = mWeights;
    const auto& config = mConfig;
    const bool trace_ops = env_enabled("SUROGATE_DEBUG_DSL_TRACE");
    (void)comm;

    ExecState st{
        .rs = rs,
        .weights = weights,
        .grads = mGrads,
        .config = config,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
        .view_sources = &mViewSources,
        .view_sources_rev = &mViewSourcesReverse,
    };
    augment_shape_env(st.shape_env, mModule.config);

    // Bind known inputs.
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);
    st.tensors.emplace("x0", rs.non_block_activations().encoded);

    const bool use_graphs = mGraphsEnabled;
    if (use_graphs && (mGraphB != B || mGraphT != T)) {
        reset_cuda_graphs();
        mGraphB = B;
        mGraphT = T;
    }
    const bool capturing = use_graphs && mForwardGraph == nullptr;
    if (!use_graphs || capturing) {
        mSaved.clear();
    }

    std::vector<char> required;
    if (!full) {
        std::vector<std::string> needed = mSaveList;
        for (const auto& kv : mForward->outputs) {
            needed.push_back(kv.first);
        }
        required = compute_required_ops(*mForward, needed);
    }

    if (capturing) {
        prime_fp8_weight_cache(required);
        prime_fp4_weight_cache(required);
    }

    // Build layer-to-weight map for prefetching (once)
    build_layer_weight_map();

    // Track current layer for prefetching
    int current_layer = -1;
    int last_prefetched = -1;

    // Prefetch first layer's weights before starting the loop
    if (mPrefetchEnabled && config.NumLayers > 0 && !capturing) {
        prefetch_layer_weights(0, rs.side_stream());
        last_prefetched = 0;
    }

    // If using weight manager for streaming, gather first layer's weights
    if (mWeightManager && mWeightManager->is_streaming_enabled() && config.NumLayers > 0 && !capturing) {
        mWeightManager->gather_block(0, comm, rs.side_stream());
    }

    auto run_ops = [&]() {
    std::unordered_map<int, DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::unordered_map<int, std::size_t> layer_temp_marks;
    auto maybe_start_layer = [&](const Operation& op) {
        for (const auto& out_name : op.outputs) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(out_name, layer_idx, field)) {
                if (layer_checkpoints.find(layer_idx) == layer_checkpoints.end()) {
                    layer_checkpoints[layer_idx] = rs.Stack.checkpoint();
                    layer_temp_marks[layer_idx] = st.temps.size();

                    // New layer started - wait for any pending prefetch and start next prefetch
                    if (layer_idx != current_layer) {
                        // Wait for prefetch of current layer if needed
                        if (current_layer >= 0) {
                            wait_for_prefetch(current_layer, rs.MainStream);
                            // Release previous layer's weights if using weight manager
                            if (mWeightManager && mWeightManager->is_streaming_enabled()) {
                                mWeightManager->release_block(current_layer, rs.MainStream);
                            }
                        }

                        // Wait for current layer's weights to be ready
                        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
                            mWeightManager->wait_for_gather(layer_idx, rs.MainStream);
                        }

                        current_layer = layer_idx;

                        // Prefetch next layer's weights on side stream
                        const int next_layer = layer_idx + 1;
                        if (next_layer < config.NumLayers && next_layer != last_prefetched) {
                            prefetch_layer_weights(next_layer, rs.side_stream());
                            // Also start gathering next layer's weights if using weight manager
                            if (mWeightManager && mWeightManager->is_streaming_enabled()) {
                                mWeightManager->gather_block(next_layer, comm, rs.side_stream());
                            }
                            last_prefetched = next_layer;
                        }
                    }
                }
                break;
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
        if (rs.ffn_temps_on_stack()) {
            auto& acts = rs.simplified_acts(layer_idx);
            acts.mlp_up.Data = nullptr;
            acts.swiglu.Data = nullptr;
        }
        rs.scratch().cudnn_workspace.Data = nullptr;

        // Offload residual to CPU if enabled (for backward pass later)
        if (rs.has_residual_offloading() && !capturing) {
            rs.mark_residual_ready(layer_idx, rs.MainStream);
            rs.put_residual(layer_idx, rs.side_stream());
        }

        layer_checkpoints.erase(layer_idx);
        layer_temp_marks.erase(layer_idx);
    };

    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!full && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;
        if (trace_ops) {
            fprintf(stderr, "[DSL TRACE] fwd op %zu/%zu id=%s type=%s\n",
                    idx, mForward->operations.size(), op.id.c_str(), op_type.c_str());
            fflush(stderr);
        }
        maybe_start_layer(op);

        if (op_type == "embedding") {
            Tensor& token_ids = get_tensor(st, op.inputs.at(0), mSaved);
            const std::string& emb_name = op.inputs.size() > 1 ? op.inputs.at(1) : "embedding";
            Tensor& emb = resolve_param_tensor(st, emb_name);
            const std::string& out_name = op.outputs.at(0);
            if (st.tensors.find(out_name) == st.tensors.end()) {
                st.tensors.emplace(out_name, rs.non_block_activations().encoded);
            }
            Tensor& out = st.tensors.at(out_name);
            encoder_forward(out, token_ids, emb, std::nullopt,
                            static_cast<int>(B), static_cast<int>(T),
                            config.HiddenSize, config.VocabSize, rs.MainStream);
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

        if (op_type == "fused_residual_rmsnorm") {
            Tensor& residual_in = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& input = get_tensor(st, op.inputs.at(1), mSaved);
            const std::string& weight_name = op.inputs.at(2);
            Tensor& weight = resolve_param_tensor(st, weight_name);

            const bool is_final = (weight_name == "final_norm" || weight_name == "final_norm_weight" || weight_name == "norm");

            Tensor* residual_out_ptr = nullptr;
            Tensor* y_ptr = nullptr;
            Tensor* rstd_ptr = nullptr;
            if (is_final) {
                residual_out_ptr = &rs.get_final_residual();
                y_ptr = &rs.non_block_activations().ln_final;
                rstd_ptr = &rs.non_block_activations().ln_final_rstd;
            } else {
                residual_out_ptr = &ensure_tensor(st, op.outputs.at(0), input.DType, {B, T, config.HiddenSize});
                y_ptr = &ensure_tensor(st, op.outputs.at(1), input.DType, {B, T, config.HiddenSize});
                rstd_ptr = &ensure_tensor(st, op.outputs.at(2), ETensorDType::FP32, {B, T});
            }
            Tensor& residual_out = *residual_out_ptr;
            Tensor& y = *y_ptr;
            Tensor& rstd = *rstd_ptr;

            double eps = config.RmsNormEps;
            if (auto* eps_attr = find_attr(op.attrs, "eps")) {
                if (auto v = attr_double(*eps_attr)) {
                    eps = *v;
                }
            }
            fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                           static_cast<float>(eps), static_cast<int>(B * T),
                                           config.HiddenSize, rs.MainStream);

            st.tensors.emplace(op.outputs.at(0), residual_out);
            st.tensors.emplace(op.outputs.at(1), y);
            st.tensors.emplace(op.outputs.at(2), rstd);
            continue;
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
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(op.outputs.at(0), layer_idx, field) && field == "mlp_down") {
                finish_layer(layer_idx);
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

        if (op_type == "matmul" || op_type == "matmul_bias") {
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
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            std::optional<Tensor> bias;
            if (op_type == "matmul_bias" && op.inputs.size() > 2 && !op.inputs.at(2).empty()) {
                bias = get_tensor(st, op.inputs.at(2), mSaved);
            }
            bool used_recipe = false;
            if (mOptions.TrainingRecipe && mode == EMMTranspose::NT && a.Sizes[0] == B * T) {
                const recipes::Recipe& recipe = *mOptions.TrainingRecipe;
                int layer_idx = -1;
                auto op_kind = matmul_op_from_weight(op.inputs.at(1), layer_idx);
                const bool allow_quant = op_kind.has_value() && allow_quant_layer(mOptions, config, layer_idx);
                const modules::MatmulOp matmul_op = op_kind.value_or(modules::MatmulOp::LMHead);

                modules::MatmulContext ctx;
                ctx.out = &out;
                ctx.inp = &a;
                ctx.weight = &b;
                ctx.bias = bias ? &*bias : nullptr;
                ctx.B = static_cast<int>(B);
                ctx.T = static_cast<int>(T);
                ctx.C_in = K;
                ctx.C_out = N;
                ctx.run_state = &rs;
                ctx.stream = rs.MainStream;
                ctx.layer_idx = layer_idx;
                ctx.op = matmul_op;
                ctx.allow_fp8 = allow_quant;
                ctx.allow_fp4 = allow_quant;
                if (allow_quant) {
                    // FP8 buffers
                    ctx.inp_quant = fp8_forward_buffer(rs, matmul_op);
                    ctx.cached_weight = get_fp8_cached_weight(op.inputs.at(1), b, rs.MainStream);
                    ctx.delayed_quantizer_idx = fp8_quantizer_index(rs, matmul_op, layer_idx);

                    // FP4 cached weights (for NVFP4 recipe on Blackwell+)
                    if (const auto* fp4_cache = get_fp4_cached_weight(op.inputs.at(1), b, rs.MainStream)) {
                        ctx.cached_fp4_data = &fp4_cache->data;
                        ctx.cached_fp4_scales = &fp4_cache->scales;
                        ctx.cached_fp4_amax = fp4_cache->amax.get<float>();
                    }
                }

                recipe.forward_matmul(ctx);
                used_recipe = true;
            }
            if (!used_recipe) {
                // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
                // swapping M/N, and swapping transpose flags.
                EMMTranspose mode_col = swap_transpose(mode);
                matmul(out, b, a, bias, nullptr, nullptr,
                       rs.CublasLtHandle, rs.CuBlasWorkspace,
                       N, M, K, mode_col, false, rs.MainStream);
            }

            // Hook invocation (for LoRA or other extensions)
            // LoRA is now handled via hooks provided by DslModel
            if (hook && *hook) {
                int layer_idx = -1;
                std::string field;
                if (parse_block_param(op.inputs.at(1), layer_idx, field)) {
                    modules::ForwardHookPoint hook_point;
                    bool should_invoke = false;
                    if (field == "qkv_weight") {
                        hook_point = modules::ForwardHookPoint::AfterQKVProjection;
                        should_invoke = true;
                    } else if (field == "out_weight") {
                        hook_point = modules::ForwardHookPoint::AfterAttnOutProjection;
                        should_invoke = true;
                    } else if (field == "mlp_up_weight") {
                        hook_point = modules::ForwardHookPoint::AfterMLPUpProjection;
                        should_invoke = true;
                    } else if (field == "mlp_down_weight") {
                        hook_point = modules::ForwardHookPoint::AfterMLPDownProjection;
                        should_invoke = true;
                    }
                    if (should_invoke) {
                        invoke_forward_hook(layer_idx, hook_point, rs.MainStream, hook);
                    }
                }
            }
            continue;
        }

        if (op_type == "bias_add") {
            Tensor& x = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& bias = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& out = ensure_tensor(st, op.outputs.at(0), x.DType,
                                        {x.Sizes[0], x.Sizes[1], x.Sizes[2]});
            const std::size_t bytes = static_cast<std::size_t>(x.nelem()) * get_dtype_size(x.DType);
            CUDA_CHECK(cudaMemcpyAsync(out.Data, x.Data, bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
            add_bias_tensor(out, bias, static_cast<int>(x.Sizes[0]), static_cast<int>(x.Sizes[1]),
                            static_cast<int>(x.Sizes[2]), rs.MainStream);
            continue;
        }

        if (op_type == "swiglu") {
            Tensor& inp = get_tensor(st, op.inputs.at(0), mSaved);
            // Input shape (B, T, 2*D) -> output (B, T, D)
            const long D = inp.Sizes[2] / 2;
            Tensor& out = ensure_tensor(st, op.outputs.at(0), inp.DType, {inp.Sizes[0], inp.Sizes[1], D});
            swiglu_forward(out, inp, nullptr, static_cast<int>(inp.Sizes[0]),
                           static_cast<int>(inp.Sizes[1]), static_cast<int>(D), rs.MainStream);
            continue;
        }

        // Fused matmul + swiglu for MLP up projection
        // Decomposes into: matmul -> swiglu, but saves intermediate for backward
        if (op_type == "matmul_swiglu") {
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            if (a.Rank != 2 || b.Rank != 2) {
                throw std::runtime_error(
                    "DSL graph executor: matmul_swiglu expects rank-2 tensors (op=" + op.id + ")"
                );
            }
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            // N = 2*D (fused up+gate), output D = N/2
            const long D = N / 2;

            // First output: swiglu result (M, D)
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, {static_cast<long>(M), D});
            // Second output: matmul intermediate for backward (M, N=2*D)
            Tensor& up_out = ensure_tensor(st, op.outputs.at(1), a.DType, {static_cast<long>(M), static_cast<long>(N)});

            // Step 1: matmul
            matmul(up_out, b, a, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K, swap_transpose(mode), false, rs.MainStream);

            // Step 2: swiglu
            // up_out is (M, 2*D), we need to reshape it to (B*T, 2*D) for swiglu
            // swiglu expects (B, T, 2*D) -> (B, T, D), but we have (B*T, 2*D)
            // Reshape to 3D for swiglu kernel
            Tensor up_3d = view_tensor(up_out, {B, T, static_cast<long>(N)});
            Tensor out_3d = view_tensor(out, {B, T, D});
            swiglu_forward(out_3d, up_3d, nullptr, static_cast<int>(B),
                           static_cast<int>(T), static_cast<int>(D), rs.MainStream);
            continue;
        }

        if (op_type == "qkv_qk_norm_rope") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& q_norm = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& k_norm = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(3), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(4), mSaved);

            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());
            const int qkv_channels = Hs * (Hq + 2 * Hkv);

            Tensor& q_rstd = ensure_tensor(st, op.outputs.at(1), ETensorDType::FP32, {B, T, Hq});
            Tensor& k_rstd = ensure_tensor(st, op.outputs.at(2), ETensorDType::FP32, {B, T, Hkv});

            double eps = config.RmsNormEps;
            if (auto* eps_attr = find_attr(op.attrs, "eps")) {
                if (auto v = attr_double(*eps_attr)) {
                    eps = *v;
                }
            }

            // Match modular path: fused QK norm + RoPE when supported, otherwise fall back to separate ops.
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
            const bool rope_fusable = (rotary_dim > 0)
                && ((Hs % 2) == 0)
                && (((Hs / 2) % 32) == 0)
                && (freqs.Rank >= 2)
                && (freqs.Sizes[1] >= Hs)
                && (qkv_view.Rank == 3);
            if (rope_fusable) {
                qkv_qk_norm_rope_forward(qkv_view, q_rstd, k_rstd, q_norm, k_norm,
                                         freqs, reinterpret_cast<int*>(pos_ids.Data),
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         Hq, Hkv, Hs, rs.MainStream);
            } else {
                const int q_rows = Hq * Hs;
                qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         qkv_channels, Hq, Hs, 0, rs.MainStream);
                qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                         static_cast<float>(eps),
                                         static_cast<int>(B), static_cast<int>(T),
                                         qkv_channels, Hkv, Hs, q_rows, rs.MainStream);
                rope_forward(qkv, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                             static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            }
            st.tensors.emplace(op.outputs.at(0), qkv);
            st.tensors.emplace(op.outputs.at(1), q_rstd);
            st.tensors.emplace(op.outputs.at(2), k_rstd);
            continue;
        }

        if (op_type == "rope") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& freqs = get_tensor(st, op.inputs.at(1), mSaved);
            Tensor& pos_ids = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor& out = ensure_tensor(st, op.outputs.at(0), qkv.DType, {qkv.Sizes[0], qkv.Sizes[1], qkv.Sizes[2], qkv.Sizes[3]});

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

            rope_forward(out, qkv, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                         static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs, rotary_dim, rs.MainStream);
            continue;
        }

        if (op_type == "flash_attention" || op_type == "flash_attention_qkv") {
            Tensor& qkv = get_tensor(st, op.inputs.at(0), mSaved);
            const int Hq = static_cast<int>(config.NumQueryHeads);
            const int Hkv = static_cast<int>(config.NumKeyValHeads);
            const int Hs = static_cast<int>(config.head_size());

            Tensor& out = ensure_tensor(st, op.outputs.at(0), qkv.DType, {B, T, Hq, Hs});
            Tensor& lse = ensure_tensor(st, op.outputs.at(1), ETensorDType::FP32, {B, Hq, T});

            // Ensure workspace allocated
            if (!rs.scratch().cudnn_workspace.Data) {
                rs.temp_acquire(rs.scratch().cudnn_workspace);
                st.temps.push_back(rs.scratch().cudnn_workspace);
            }

            attention_forward_cudnn(out, lse, qkv, rs.scratch().cudnn_workspace,
                                    rs.CudnnHandle, static_cast<int>(B), static_cast<int>(T), Hq, Hkv, Hs,
                                    rs.MainStream);
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported forward op " + op.name);
    }

    // Save requested tensors for backward (uses auto-computed list if autodiff was used).
    for (const auto& name : mSaveList) {
        auto it = st.tensors.find(name);
        if (it != st.tensors.end()) {
            mSaved[name] = it->second;
        } else if (name == "token_ids") {
            mSaved[name] = rs.Inputs;
        } else {
            throw std::runtime_error("DSL graph executor: missing save tensor " + name);
        }
    }

    free_temps(st);
    };

    trace_or_execute_cuda_graph_with_stack(run_ops, rs.MainStream, mForwardGraph, use_graphs, rs.Stack, mForwardCheckpoint);
}

}  // namespace dsl
