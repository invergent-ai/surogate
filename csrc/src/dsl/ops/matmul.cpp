#include "dsl/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "modules/fp8_scaling_state.h"
#include "utilities/stack.h"
#include "utilities/utils.h"
#include "utilities/dtype.h"
#include "dsl/graph_executor_helpers.h"

namespace dsl {

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    const int layer_idx = op.attrs.layer_idx;

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

    const int pre_nan_trace = env_int("SUROGATE_MATMUL_IN_NAN_TRACE", 0);
    if (pre_nan_trace && !mCapturing) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        const int trace_layer = env_int("SUROGATE_MATMUL_IN_NAN_LAYER", -1);
        if (trace_layer < 0 || trace_layer == layer_idx) {
            const char* op_name = "matmul";
            if (op.attrs.matmul_op.has_value()) {
                switch (*op.attrs.matmul_op) {
                    case modules::MatmulOp::QKV: op_name = "qkv"; break;
                    case modules::MatmulOp::AttnOut: op_name = "attn_out"; break;
                    case modules::MatmulOp::MLPUp: op_name = "mlp_up"; break;
                    case modules::MatmulOp::MLPDown: op_name = "mlp_down"; break;
                    case modules::MatmulOp::LMHead: op_name = "lm_head"; break;
                    default: break;
                }
            }
            auto log_pre = [&](const Tensor& t, const char* tag) {
                if (!t.Data) {
                    return;
                }
                long row = -1;
                float min_val = 0.0f;
                float max_val = 0.0f;
                if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
                    return;
                }
                std::cerr << fmt::format("[MATMUL_IN_NAN_PRE] op={} layer={} tag={} row={} min={} max={} dtype={}\n",
                                         op_name, layer_idx, tag ? tag : "<unnamed>",
                                         row, min_val, max_val, static_cast<int>(t.DType));
            };
            log_pre(a, "inp");
            log_pre(b, "weight");
            if (bias.has_value()) {
                log_pre(*bias, "bias");
            }
        }
    }

    bool used_recipe = false;
    modules::MatmulContext ctx{};
    modules::MatmulContext* ctx_ptr = nullptr;
    if (mRecipe && op.attrs.transpose == EMMTranspose::NT && a.Sizes[0] == mB * mT) {
        if (op.attrs.allow_quant && op.attrs.matmul_op.has_value()) {
            ctx.out = &out;
            ctx.inp = &a;
            ctx.weight = &b;
            ctx.bias = bias ? &*bias : nullptr;
            ctx.B = static_cast<int>(mB);
            ctx.T = static_cast<int>(mT);
            ctx.C_in = K;
            ctx.C_out = N;
            ctx.run_state = &mRunState;
            ctx.stream = mRunState.MainStream;
            ctx.layer_idx = op.attrs.layer_idx;
            ctx.op = *op.attrs.matmul_op;
            ctx.allow_fp8 = mRecipe->uses_fp8_forward();
            ctx.allow_fp4 = mRecipe->uses_fp4_forward();

            // FP8/FP4 buffers would be set here via pre-resolved cache
            if (ctx.allow_fp8) {
                ctx.inp_quant = fp8_forward_buffer(mRunState, *op.attrs.matmul_op);
                ctx.delayed_quantizer_idx = fp8_quantizer_index(mRunState, *op.attrs.matmul_op, op.attrs.layer_idx);
            }

            mRecipe->forward_matmul(ctx);
            used_recipe = true;
            ctx_ptr = &ctx;
        }
    }

    if (!used_recipe) {
        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(out, b, a, bias, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }

    const int fp8_nan_trace_fwd = env_int("SUROGATE_FP8_NAN_TRACE", 0);
    if (fp8_nan_trace_fwd && !mCapturing && used_recipe && ctx_ptr && ctx_ptr->allow_fp8) {
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(mRunState.MainStream, &cap_status) == cudaSuccess &&
            cap_status != cudaStreamCaptureStatusNone) {
            // Avoid host copies during CUDA graph capture.
        } else {
        static std::atomic<int> fp8_nan_once{0};
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (find_first_nan_row(out, &row, &min_val, &max_val)) {
            const int trace_layer = env_int("SUROGATE_FP8_NAN_TRACE_LAYER", -1);
            if (trace_layer < 0 || trace_layer == layer_idx) {
                if (fp8_nan_once.fetch_add(1) == 0) {
                    float scale_host = 0.0f;
                    float amax_host = 0.0f;
                    const int qidx = ctx_ptr->delayed_quantizer_idx;
                    if (mRunState.has_fp8_delayed_scaling() && qidx >= 0) {
                        auto* fp8_state = mRunState.get_fp8_scaling_state();
                        if (fp8_state) {
                            auto& scales = fp8_state->scales();
                            auto& amaxes = fp8_state->recorded_amaxes();
                            CUDA_CHECK(cudaMemcpyAsync(&scale_host, scales.get<float>() + qidx,
                                                       sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
                            CUDA_CHECK(cudaMemcpyAsync(&amax_host, amaxes.get<float>() + qidx,
                                                       sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
                            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        }
                    }
                    const char* op_name = "matmul";
                    if (op.attrs.matmul_op.has_value()) {
                        switch (*op.attrs.matmul_op) {
                            case modules::MatmulOp::QKV: op_name = "qkv"; break;
                            case modules::MatmulOp::AttnOut: op_name = "attn_out"; break;
                            case modules::MatmulOp::MLPUp: op_name = "mlp_up"; break;
                            case modules::MatmulOp::MLPDown: op_name = "mlp_down"; break;
                            case modules::MatmulOp::LMHead: op_name = "lm_head"; break;
                            default: break;
                        }
                    }
                    std::cerr << fmt::format(
                        "[FP8_NAN_FWD] op={} layer={} qidx={} out_row={} min={} max={} scale={} amax={}\n",
                        op_name, layer_idx, qidx, row, min_val, max_val, scale_host, amax_host);
                }
            }
        }
        }
    }

    if (env_int("SUROGATE_QKV_PTR_TRACE", 0) && op.attrs.matmul_op.has_value() &&
        *op.attrs.matmul_op == modules::MatmulOp::QKV) {
        std::cerr << fmt::format("[QKV_MATMUL_PTR] layer={} name={} ptr={}\n",
                                 layer_idx, op.outputs[0].name,
                                 static_cast<const void*>(out.Data));
    }

    const int qkv_trace = env_int("SUROGATE_QKV_MATMUL_TRACE", 0);
    if (qkv_trace && !mCapturing && op.attrs.matmul_op.has_value() &&
        *op.attrs.matmul_op == modules::MatmulOp::QKV) {
        if (env_int("SUROGATE_QKV_MATMUL_SYNCALL", 0)) {
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        }
        const int trace_layer = env_int("SUROGATE_QKV_MATMUL_TRACE_LAYER", -1);
        if (trace_layer < 0 || trace_layer == layer_idx) {
            static std::atomic<int> qkv_trace_count{0};
            const int limit = env_int("SUROGATE_QKV_MATMUL_TRACE_LIMIT", 4);
            if (limit <= 0 || qkv_trace_count.fetch_add(1) < limit) {
                long out_row = -1;
                float out_min = 0.0f;
                float out_max = 0.0f;
                const bool out_nan = find_first_nan_row(out, &out_row, &out_min, &out_max);
                long in_row = -1;
                float in_min = 0.0f;
                float in_max = 0.0f;
                const bool in_nan = find_first_nan_row(a, &in_row, &in_min, &in_max);
                long w_row = -1;
                float w_min = 0.0f;
                float w_max = 0.0f;
                const bool w_nan = find_first_nan_row(b, &w_row, &w_min, &w_max);
                const int ptr_trace = env_int("SUROGATE_QKV_MATMUL_PTR_TRACE", 0);
                if (ptr_trace) {
                    std::cerr << fmt::format(
                        "[QKV_MATMUL_PTR] layer={} in_ptr={} w_ptr={} out_ptr={}\n",
                        layer_idx,
                        static_cast<const void*>(a.Data),
                        static_cast<const void*>(b.Data),
                        static_cast<const void*>(out.Data));
                }
                std::cerr << fmt::format(
                    "[QKV_MATMUL_TRACE] layer={} micro={} op_idx={} op_id={} in_name={} w_name={} out_name={} "
                    "in_shape={} out_shape={} "
                    "in_nan={} out_nan={} w_nan={} in_row={} in_min={} in_max={} "
                    "out_row={} out_min={} out_max={} w_row={} w_min={} w_max={}\n",
                    layer_idx, mMicroStep, op.original_idx, op.op_id,
                    op.inputs[0].name, op.inputs[1].name, op.outputs[0].name,
                    tensor_shape_str(a), tensor_shape_str(out),
                    in_nan ? 1 : 0, out_nan ? 1 : 0, w_nan ? 1 : 0,
                    in_row, in_min, in_max,
                    out_row, out_min, out_max,
                    w_row, w_min, w_max);

                const int sample_trace = env_int("SUROGATE_QKV_MATMUL_SAMPLE_TRACE", 0);
                if (sample_trace) {
                    std::vector<float> sample;
                    if (copy_tensor_sample_offset_as_f32(out, 0, 8, sample) && !sample.empty()) {
                        std::ostringstream oss;
                        for (std::size_t i = 0; i < sample.size(); ++i) {
                            if (i) {
                                oss << ",";
                            }
                            oss << sample[i];
                        }
                        std::cerr << fmt::format("[QKV_MATMUL_SAMPLE] layer={} micro={} op_idx={} op_id={} out_name={} vals={}\n",
                                                 layer_idx, mMicroStep, op.original_idx, op.op_id,
                                                 op.outputs[0].name, oss.str());
                    }
                }

                if (env_int("SUROGATE_QKV_GUARD", 0)) {
                    std::vector<float> guard_vals;
                    if (copy_tensor_sample_offset_as_f32(out, 0, 8, guard_vals) &&
                        guard_vals.size() >= 8) {
                        std::array<float, 8> cached{};
                        for (std::size_t i = 0; i < 8; ++i) {
                            cached[i] = guard_vals[i];
                        }
                        record_qkv_guard_sample(out.Data, layer_idx, mMicroStep,
                                                op.original_idx, op.op_id, cached);
                        if (env_int("SUROGATE_QKV_STACK_WATCH", 0)) {
                            set_stack_watch_range(out.Data, out.bytes(), "qkv_guard");
                        }
                    }
                }

                if (env_int("SUROGATE_QKV_CANARY", 0) &&
                    env_int("SUROGATE_QKV_CANARY_CHECK_MATMUL", 0)) {
                    mRunState.check_qkv_canary(mRunState.MainStream,
                                               "post_matmul",
                                               layer_idx,
                                               mMicroStep,
                                               op.op_id.c_str());
                }
            }
        }
    }

    const int matmul_nan_trace = env_int("SUROGATE_MATMUL_NAN_TRACE", 0);
    if (matmul_nan_trace && !mCapturing) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        const int trace_layer = env_int("SUROGATE_MATMUL_NAN_LAYER", -1);
        if (trace_layer < 0 || trace_layer == layer_idx) {
            long row = -1;
            float min_val = 0.0f;
            float max_val = 0.0f;
            if (find_first_nan_row(out, &row, &min_val, &max_val)) {
                const char* op_name = "matmul";
                if (op.attrs.matmul_op.has_value()) {
                    switch (*op.attrs.matmul_op) {
                        case modules::MatmulOp::QKV: op_name = "qkv"; break;
                        case modules::MatmulOp::AttnOut: op_name = "attn_out"; break;
                        case modules::MatmulOp::MLPUp: op_name = "mlp_up"; break;
                        case modules::MatmulOp::MLPDown: op_name = "mlp_down"; break;
                        case modules::MatmulOp::LMHead: op_name = "lm_head"; break;
                        default: break;
                    }
                }
                const std::string& weight_name = op.inputs[1].name;
                const bool is_qlora = mWeights.qlora_provider() && !weight_name.empty() && mWeights.is_external(weight_name);
                std::cerr << fmt::format("[MATMUL_NAN] op={} layer={} weight={} row={} min={} max={} dtype={} qlora={}\n",
                                         op_name, layer_idx, weight_name, row, min_val, max_val,
                                         static_cast<int>(out.DType), is_qlora ? 1 : 0);
                auto log_input_nan = [&](const Tensor& t, const char* tag) {
                    if (!t.Data) {
                        return;
                    }
                    long in_row = -1;
                    float in_min = 0.0f;
                    float in_max = 0.0f;
                    if (!find_first_nan_row(t, &in_row, &in_min, &in_max)) {
                        return;
                    }
                    std::cerr << fmt::format("[MATMUL_NAN_IN] op={} layer={} weight={} tag={} row={} min={} max={} dtype={}\n",
                                             op_name, layer_idx, weight_name,
                                             tag ? tag : "<unnamed>", in_row, in_min, in_max,
                                             static_cast<int>(t.DType));
                };
                log_input_nan(a, "inp");
                log_input_nan(b, "weight");
                if (bias.has_value()) {
                    log_input_nan(*bias, "bias");
                }
            }
        }
    }

    const int qlora_nan_trace = env_int("SUROGATE_QLORA_NAN_TRACE", 0);
    if (qlora_nan_trace && !mCapturing && mWeights.qlora_provider()) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        const std::string& weight_name = op.inputs[1].name;
        if (!weight_name.empty() && mWeights.is_external(weight_name)) {
            const int trace_layer = env_int("SUROGATE_QLORA_NAN_LAYER", -1);
            if (trace_layer < 0 || trace_layer == layer_idx) {
                static std::atomic<int> qlora_nan_count{0};
                const int limit = env_int("SUROGATE_QLORA_NAN_LIMIT", 8);
                if (limit <= 0 || qlora_nan_count.fetch_add(1) < limit) {
                    auto matmul_op_name = [&]() -> const char* {
                        if (!op.attrs.matmul_op.has_value()) return "matmul";
                        switch (*op.attrs.matmul_op) {
                            case modules::MatmulOp::QKV: return "qkv";
                            case modules::MatmulOp::AttnOut: return "attn_out";
                            case modules::MatmulOp::MLPUp: return "mlp_up";
                            case modules::MatmulOp::MLPDown: return "mlp_down";
                            case modules::MatmulOp::LMHead: return "lm_head";
                            default: return "matmul";
                        }
                    };
                    const int full_scan = env_int("SUROGATE_QLORA_NAN_FULL", 0);
                    auto check_nan = [&](const Tensor& t, const char* tag) -> bool {
                        if (!t.Data) {
                            return false;
                        }
                        const long rows = (t.Rank > 0) ? static_cast<long>(t.Sizes[0]) : 1;
                        long row = -1;
                        float min_val = 0.0f;
                        float max_val = 0.0f;
                        bool has_nan = false;
                        if (full_scan) {
                            has_nan = find_first_nan_row(t, &row, &min_val, &max_val);
                        } else {
                            const long sample_rows[3] = {0, rows > 0 ? rows / 2 : 0, rows > 0 ? rows - 1 : 0};
                            for (long r : sample_rows) {
                                if (r < 0 || r >= rows) {
                                    continue;
                                }
                                if (tensor_row_has_nan_or_inf(t, r, &min_val, &max_val)) {
                                    row = r;
                                    has_nan = true;
                                    break;
                                }
                            }
                        }
                        if (has_nan) {
                            std::cerr << fmt::format("[QLORA_NAN_MATMUL] op={} layer={} weight={} tag={} row={} min={} max={} dtype={}\n",
                                                     matmul_op_name(), layer_idx, weight_name,
                                                     tag ? tag : "<unnamed>", row, min_val, max_val,
                                                     static_cast<int>(t.DType));
                        }
                        return has_nan;
                    };
                    bool out_nan = check_nan(out, "out");
                    if (out_nan) {
                        check_nan(a, "inp");
                        check_nan(b, "weight");
                        if (bias.has_value()) {
                            check_nan(*bias, "bias");
                        }
                        if (env_int("SUROGATE_QLORA_NAN_ABORT", 0)) {
                            throw std::runtime_error("QLoRA matmul output contains NaN/Inf");
                        }
                    }
                }
            }
        }
    }

    if (mForwardPlan && op.attrs.matmul_op.has_value() && op.attrs.layer_idx >= 0 &&
        static_cast<std::size_t>(op.attrs.layer_idx) < mForwardPlan->size() &&
        *op.attrs.matmul_op != modules::MatmulOp::LMHead) {
        MatmulForwardPlan plan{};
        plan.valid = true;
        plan.use_recipe = used_recipe;
        plan.has_bias = bias.has_value();
        if (used_recipe && ctx_ptr) {
            plan.allow_fp8 = ctx_ptr->allow_fp8;
            plan.allow_fp4 = ctx_ptr->allow_fp4;
            plan.delayed_quantizer_idx = ctx_ptr->delayed_quantizer_idx;
            plan.use_fp8_cache = (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data);
            plan.use_fp4_cache = (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales);
        }
        auto& layer_plan = (*mForwardPlan)[static_cast<std::size_t>(op.attrs.layer_idx)];
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                layer_plan.qkv = plan;
                break;
            case modules::MatmulOp::AttnOut:
                layer_plan.out_proj = plan;
                break;
            case modules::MatmulOp::MLPUp:
                layer_plan.mlp_up = plan;
                break;
            case modules::MatmulOp::MLPDown:
                layer_plan.mlp_down = plan;
                break;
            default:
                break;
        }
    }

    // Hook invocation
    if (hook && *hook && op.attrs.forward_hook_point.has_value()) {
        (*hook)(op.attrs.layer_idx, mRunState.MainStream, *op.attrs.forward_hook_point, mHookContext);
    }

}

void CompiledExecutor::dispatch_matmul_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // inputs: d_out, A, B (weight)
    // outputs: dA, dB
    const std::string& weight_name = (op.inputs.size() > 2) ? op.inputs[2].name : "";
    const bool is_lm_head = (weight_name == "lm_head" || weight_name == "lm_head_weight");
    const bool skip_lm_head = is_lm_head && mOptions.LMHeadChunks > 1;

    EMMTranspose mode = op.attrs.transpose;
    const int layer_idx = op.attrs.layer_idx;
    const bool allow_quant = op.attrs.allow_quant;

    // Check if weight gradient should be skipped BEFORE allocating (frozen weights in LoRA mode)
    bool skip_weight_grad = true;
    const std::string& dB_name = op.outputs.size() > 1 ? op.outputs[1].name : "";
    if (!dB_name.empty()) {
        std::string weight_name;
        if (auto base = base_param_from_grad(dB_name)) {
            weight_name = *base;
        } else {
            weight_name = dB_name;
            if (weight_name.rfind("d_", 0) == 0) {
                weight_name = weight_name.substr(2);
            }
        }
        bool accum = false;
        Tensor* grad = mGrads.get_param_grad(weight_name, accum);
        skip_weight_grad = (grad == nullptr || !grad->Data);
    }

    if (skip_lm_head) {
        if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
            (void)ensure_output_tensor(op.outputs[0]);
        }
        if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
            (void)ensure_output_tensor(op.outputs[1]);
        }
        return;
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);

    const bool is_qkv_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::QKV);
    const bool is_mlp_down_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPDown);

    // Now allocate output tensors - skip dB if weights are frozen
    Tensor* dA_ptr = nullptr;
    Tensor* dB_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        dA_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (!skip_weight_grad && op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        dB_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    // FIX: Zero dA buffer before matmul to ensure consistent results regardless of initial values
    // This is needed because the buffer may contain stale gradients from layer 27 in no-recompute mode.
    // Even though matmul uses accumulate=false (beta=0), we explicitly zero to ensure determinism.
    if (dA_ptr && dA_ptr->Data) {
        fill_zero(*dA_ptr, mRunState.MainStream);
    }

    if (!dA_ptr && !dB_ptr) {
        return;
    }

    const int trace = env_int("SUROGATE_MATMUL_BWD_TRACE", 0);
    const int trace_attn_out = env_int("SUROGATE_MATMUL_BWD_TRACE_ATTN_OUT", 0);
    const int trace_qkv = env_int("SUROGATE_MATMUL_BWD_TRACE_QKV", 0);
    const int trace_mlp_up = env_int("SUROGATE_MATMUL_BWD_TRACE_MLP_UP", 0);
    const int trace_mlp_down = env_int("SUROGATE_MATMUL_BWD_TRACE_MLP_DOWN", 0);
    const int trace_layer = env_int("SUROGATE_MATMUL_BWD_TRACE_LAYER", -1);
    const int trace_limit = env_int("SUROGATE_MATMUL_BWD_TRACE_LIMIT", 8);
    const int trace_samples = env_int("SUROGATE_MATMUL_BWD_TRACE_SAMPLES", 8);
    const int trace_dA = env_int("SUROGATE_MATMUL_BWD_TRACE_DA", 0);
    static std::atomic<int> trace_count{0};
    const bool is_attn_out_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::AttnOut);
    const bool is_mlp_up_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp);
    const bool trace_any = (trace_attn_out || trace_qkv || trace_mlp_up || trace_mlp_down);
    const bool want_op = trace_any
        ? ((trace_attn_out && is_attn_out_op) ||
           (trace_qkv && is_qkv_op) ||
           (trace_mlp_up && is_mlp_up_op) ||
           (trace_mlp_down && is_mlp_down_op))
        : is_mlp_down_op;
    const bool do_trace = trace && !mCapturing && want_op &&
        (trace_layer < 0 || trace_layer == layer_idx) &&
        (trace_limit <= 0 || trace_count.fetch_add(1) < trace_limit);

    auto trace_sample = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            std::cerr << fmt::format("[MATMUL_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr=<null>\n",
                                     layer_idx, mMicroStep, tag ? tag : "<unnamed>",
                                     static_cast<int>(t.DType), tensor_shape_str(t));
            return;
        }
        std::vector<float> vals;
        if (!copy_tensor_token_sample_as_f32(t, 0, static_cast<std::size_t>(trace_samples), vals) || vals.empty()) {
            std::cerr << fmt::format("[MATMUL_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} sample=<unavailable>\n",
                                     layer_idx, mMicroStep, tag ? tag : "<unnamed>",
                                     static_cast<int>(t.DType), tensor_shape_str(t),
                                     static_cast<const void*>(t.Data));
            return;
        }
        float min_v = vals[0];
        float max_v = vals[0];
        float max_abs = std::abs(vals[0]);
        double mean_abs = 0.0;
        for (float v : vals) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            max_abs = std::max(max_abs, std::abs(v));
            mean_abs += static_cast<double>(std::abs(v));
        }
        mean_abs /= static_cast<double>(vals.size());
        std::cerr << fmt::format(
            "[MATMUL_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} min={:.6g} max={:.6g} max_abs={:.6g} mean_abs={:.6g}\n",
            layer_idx, mMicroStep, tag ? tag : "<unnamed>", static_cast<int>(t.DType),
            tensor_shape_str(t), static_cast<const void*>(t.Data),
            min_v, max_v, max_abs, mean_abs);
    };

    if (do_trace) {
        trace_sample(d_out, "d_out");
        trace_sample(a, "a_in");
    }

    bool do_accumulate = mAccumulateTensors.count(dB_name) > 0;
    if (!do_accumulate && !dB_name.empty()) {
        if (auto base = base_param_from_grad(dB_name)) {
            do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
        }
    }

    bool used_recipe = false;
    bool used_fp8 = false;
    bool has_dout_quant = false;

    const bool disable_qkv_recipe_bwd =
        is_qkv_op && skip_weight_grad && (mConfig.NumExperts > 0);
    if (mRecipe && mode == EMMTranspose::NT && a.Sizes[0] == mB * mT && allow_quant && !disable_qkv_recipe_bwd) {
        Tensor dA_tmp{};
        Tensor dB_tmp{};
        Tensor* dA_use = dA_ptr;
        Tensor* dB_use = dB_ptr;

        if (!dA_use) {
            dA_tmp = mRunState.temp_alloc(a.DType, {a.Sizes[0], a.Sizes[1]});
            mTemps.push_back(dA_tmp);
            dA_use = &dA_tmp;
        }
        if (!dB_use) {
            dB_tmp = mRunState.temp_alloc(b.DType, {b.Sizes[0], b.Sizes[1]});
            mTemps.push_back(dB_tmp);
            dB_use = &dB_tmp;
        }

        modules::MatmulContext ctx;
        ctx.dinp = dA_use;
        ctx.dweight = dB_use;
        ctx.dout = &d_out;
        ctx.inp = &a;
        ctx.weight = &b;
        ctx.B = static_cast<int>(mB);
        ctx.T = static_cast<int>(mT);
        ctx.C_in = static_cast<int>(a.Sizes[1]);
        ctx.C_out = static_cast<int>(b.Sizes[0]);
        ctx.run_state = &mRunState;
        ctx.stream = mRunState.MainStream;
        ctx.layer_idx = layer_idx;
        ctx.op = op.attrs.matmul_op.value_or(modules::MatmulOp::LMHead);
        ctx.accumulate = do_accumulate;
        ctx.skip_weight_grad = skip_weight_grad || !dB_ptr;
        ctx.allow_fp8 = allow_quant && mRecipe->uses_fp8_hybrid_backward();
        ctx.allow_fp4 = allow_quant && mRecipe->uses_fp4_forward();

        if (ctx.allow_fp8 && op.attrs.matmul_op.has_value()) {
            ctx.dout_quant = fp8_grad_buffer(mRunState, *op.attrs.matmul_op);
            if (!ctx.dout_quant || !ctx.dout_quant->Data) {
                ctx.allow_fp8 = false;
            }
        }
        used_fp8 = ctx.allow_fp8;
        has_dout_quant = (ctx.dout_quant && ctx.dout_quant->Data);

        mRecipe->backward_matmul(ctx);
        used_recipe = true;
    }

    if (!used_recipe) {
        Tensor d_out_mat = d_out;
        Tensor a_mat = a;
        auto maybe_flatten_bt = [&](Tensor& t) {
            if (t.Rank > 2 && t.Sizes[0] == mB && t.Sizes[1] == mT) {
                t = view_tensor(t, {mB * mT, t.Sizes[t.Rank - 1]});
            }
        };
        // Ensure matmul inputs are rank-2 by flattening [B, T, K] -> [B*T, K].
        // This handles cases where *_flat tensors were not materialized as views.
        if (disable_qkv_recipe_bwd && is_qkv_op) {
            maybe_flatten_bt(d_out_mat);
            maybe_flatten_bt(a_mat);
        } else {
            maybe_flatten_bt(d_out_mat);
            maybe_flatten_bt(a_mat);
        }

        // Fallback: explicit matmuls for dA and dB
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
            matmul_dims(d_out_mat, b, mode_dA, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_dA);
            matmul(*dA_ptr, b, d_out_mat, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, false, mRunState.MainStream);
        }
        if (dB_ptr && !skip_weight_grad) {
            const Tensor* lhs = nullptr;
            const Tensor* rhs = nullptr;
            EMMTranspose mode_rm = EMMTranspose::NN;
            switch (mode) {
                case EMMTranspose::NN:
                    // dB = A^T * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::NT:
                    // dB = d_out^T * A
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TN;
                    break;
                case EMMTranspose::TN:
                    // dB = A * d_out
                    lhs = &a_mat;
                    rhs = &d_out_mat;
                    mode_rm = EMMTranspose::NN;
                    break;
                case EMMTranspose::TT:
                    // dB = d_out^T * A^T
                    lhs = &d_out_mat;
                    rhs = &a_mat;
                    mode_rm = EMMTranspose::TT;
                    break;
            }

            int M = 0, N = 0, K = 0;
            matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
            EMMTranspose mode_col = swap_transpose(mode_rm);
            matmul(*dB_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
                   mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, mRunState.MainStream);
        }
    }

    const int fp8_nan_trace_bwd = env_int("SUROGATE_FP8_NAN_TRACE", 0);
    if (fp8_nan_trace_bwd && !mCapturing && used_fp8) {
        cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(mRunState.MainStream, &cap_status) == cudaSuccess &&
            cap_status != cudaStreamCaptureStatusNone) {
            // Avoid host copies during CUDA graph capture.
        } else {
        static std::atomic<int> fp8_nan_once{0};
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (find_first_nan_row(d_out, &row, &min_val, &max_val)) {
            const int trace_layer = env_int("SUROGATE_FP8_NAN_TRACE_LAYER", -1);
            if (trace_layer < 0 || trace_layer == layer_idx) {
                if (fp8_nan_once.fetch_add(1) == 0) {
                    int qidx = -1;
                    if (mRunState.has_fp8_delayed_scaling() && op.attrs.matmul_op.has_value()) {
                        switch (*op.attrs.matmul_op) {
                            case modules::MatmulOp::QKV:
                                qidx = modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::BWD_D_QKV);
                                break;
                            case modules::MatmulOp::MLPUp:
                                qidx = modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::BWD_D_MLP_UP);
                                break;
                            case modules::MatmulOp::AttnOut:
                                qidx = modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::BWD_D_RES_ATT);
                                break;
                            case modules::MatmulOp::MLPDown:
                                qidx = modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::BWD_D_RES_FFN);
                                break;
                            default:
                                break;
                        }
                    }
                    float scale_host = 0.0f;
                    float amax_host = 0.0f;
                    if (qidx >= 0) {
                        auto* fp8_state = mRunState.get_fp8_scaling_state();
                        if (fp8_state) {
                            auto& scales = fp8_state->scales();
                            auto& amaxes = fp8_state->recorded_amaxes();
                            CUDA_CHECK(cudaMemcpyAsync(&scale_host, scales.get<float>() + qidx,
                                                       sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
                            CUDA_CHECK(cudaMemcpyAsync(&amax_host, amaxes.get<float>() + qidx,
                                                       sizeof(float), cudaMemcpyDeviceToHost, mRunState.MainStream));
                            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
                        }
                    }
                    const char* op_name = "matmul_bwd";
                    if (op.attrs.matmul_op.has_value()) {
                        switch (*op.attrs.matmul_op) {
                            case modules::MatmulOp::QKV: op_name = "qkv"; break;
                            case modules::MatmulOp::AttnOut: op_name = "attn_out"; break;
                            case modules::MatmulOp::MLPUp: op_name = "mlp_up"; break;
                            case modules::MatmulOp::MLPDown: op_name = "mlp_down"; break;
                            case modules::MatmulOp::LMHead: op_name = "lm_head"; break;
                            default: break;
                        }
                    }
                    std::cerr << fmt::format(
                        "[FP8_NAN_BWD] op={} layer={} qidx={} d_out_row={} min={} max={} scale={} amax={}\n",
                        op_name, layer_idx, qidx, row, min_val, max_val, scale_host, amax_host);
                }
            }
        }
        }
    }

    if (do_trace && trace_dA && dA_ptr) {
        trace_sample(*dA_ptr, "dA_out");
    }

    // Record qkv dA pointer for LN1 wiring verification.
    if (is_qkv_op && dA_ptr && layer_idx >= 0) {
        if (g_qkv_dA_ptr_by_layer.empty() && mConfig.NumLayers > 0) {
            g_qkv_dA_ptr_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), nullptr);
            g_qkv_dA_micro_by_layer.assign(static_cast<std::size_t>(mConfig.NumLayers), -1);
        }
        if (layer_idx < static_cast<int>(g_qkv_dA_ptr_by_layer.size())) {
            g_qkv_dA_ptr_by_layer[static_cast<std::size_t>(layer_idx)] =
                reinterpret_cast<std::byte*>(dA_ptr->Data);
            g_qkv_dA_micro_by_layer[static_cast<std::size_t>(layer_idx)] = mMicroStep;
        }
    }


    // Hook invocation for LoRA backward
    // Skip dense MLP hooks for MoE models - MoE has different backward path (grouped GEMM)
    const bool is_moe = mConfig.NumExperts > 0;
    const bool is_mlp_hook = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp ||
         *op.attrs.matmul_op == modules::MatmulOp::MLPDown);
    if (hook && *hook && op.attrs.backward_hook_point.has_value() && !(is_moe && is_mlp_hook)) {
        // Temporarily map grads to current backward tensors for LoRA hooks, then restore.
        struct GradPtrs {
            std::byte* d_swiglu{nullptr};
            std::byte* d_ln2{nullptr};
            std::byte* d_att{nullptr};
            std::byte* d_ln1{nullptr};
            std::byte* d_res_ffn{nullptr};
            std::byte* d_mlp_up{nullptr};
            std::byte* d_res_att{nullptr};
            std::byte* d_qkv{nullptr};
        } prev{};

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            prev.d_swiglu = reinterpret_cast<std::byte*>(grads.d_swiglu.Data);
            prev.d_ln2 = reinterpret_cast<std::byte*>(grads.d_ln2.Data);
            prev.d_att = reinterpret_cast<std::byte*>(grads.d_att.Data);
            prev.d_ln1 = reinterpret_cast<std::byte*>(grads.d_ln1.Data);
            prev.d_res_ffn = reinterpret_cast<std::byte*>(grads.d_res_ffn.Data);
            prev.d_mlp_up = reinterpret_cast<std::byte*>(grads.d_mlp_up.Data);
            prev.d_res_att = reinterpret_cast<std::byte*>(grads.d_res_att.Data);
            prev.d_qkv = reinterpret_cast<std::byte*>(grads.d_qkv.Data);

            if (dA_ptr) {
                switch (*op.attrs.matmul_op) {
                    case modules::MatmulOp::MLPDown:
                        grads.d_swiglu.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::MLPUp:
                        grads.d_ln2.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::AttnOut:
                        grads.d_att.Data = dA_ptr->Data;
                        break;
                    case modules::MatmulOp::QKV:
                        grads.d_ln1.Data = dA_ptr->Data;
                        break;
                    default:
                        break;
                }
            }

            switch (*op.attrs.matmul_op) {
                case modules::MatmulOp::MLPDown:
                    grads.d_res_ffn.Data = d_out.Data;
                    break;
                case modules::MatmulOp::MLPUp:
                    grads.d_mlp_up.Data = d_out.Data;
                    break;
                case modules::MatmulOp::AttnOut:
                    grads.d_res_att.Data = d_out.Data;
                    break;
                case modules::MatmulOp::QKV:
                    grads.d_qkv.Data = d_out.Data;
                    break;
                default:
                    break;
            }
        }

        // Ensure activations needed by LoRA hooks are available.
        if (layer_idx >= 0 && op.attrs.matmul_op.has_value()) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (*op.attrs.matmul_op == modules::MatmulOp::MLPDown) {
                // LoRA backward hook needs acts.swiglu (forward activation).
                // With recompute enabled, swiglu may have been stack-allocated and freed.
                if (!acts.swiglu.Data && acts.mlp_up.Data) {
                    mRunState.temp_acquire(acts.swiglu);
                    const int Bv = static_cast<int>(mB);
                    const int Tv = static_cast<int>(mT);
                    const int D = static_cast<int>(mConfig.IntermediateSize);
                    swiglu_forward(acts.swiglu, acts.mlp_up, nullptr, Bv, Tv, D, mRunState.MainStream);
                }
            }
        }
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);

        if (op.attrs.matmul_op.has_value() && layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(layer_idx);
            grads.d_swiglu.Data = prev.d_swiglu;
            grads.d_ln2.Data = prev.d_ln2;
            grads.d_att.Data = prev.d_att;
            grads.d_ln1.Data = prev.d_ln1;
            grads.d_res_ffn.Data = prev.d_res_ffn;
            grads.d_mlp_up.Data = prev.d_mlp_up;
            grads.d_res_att.Data = prev.d_res_att;
            grads.d_qkv.Data = prev.d_qkv;
        }
    }
}

}  // namespace dsl
