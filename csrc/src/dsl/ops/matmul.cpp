#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "dsl/graph_executor_helpers.h"

namespace dsl {

void CompiledExecutor::dispatch_matmul(const CompiledOp& op, const modules::ForwardHook* hook) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    std::optional<Tensor> bias;
    if (op.type == CompiledOpType::MatmulBias && op.inputs.size() > 2) {
        bias = resolve_tensor(op.inputs[2]);
    }

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);

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

    // NaN detection for forward matmuls (token 3 sample)
    if (op.attrs.matmul_op.has_value()) {
        const long sample_token = 3;
        switch (*op.attrs.matmul_op) {
            case modules::MatmulOp::QKV:
                log_nan_sample("FWD_QKV", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                if (op.attrs.layer_idx == 0) {
                    log_tensor_stats("FWD_QKV", op.attrs.layer_idx, op.outputs[0].name, out, 4096);
                    if (used_recipe) {
                        const auto recipe_name = mRecipe ? mRecipe->name() : std::string_view("none");
                        fprintf(stderr,
                                "[MATMUL_RECIPE] op=QKV layer=%d recipe=%.*s allow_fp8=%d allow_fp4=%d use_fp8_cache=%d use_fp4_cache=%d\n",
                                op.attrs.layer_idx,
                                static_cast<int>(recipe_name.size()),
                                recipe_name.data(),
                                ctx_ptr ? (ctx_ptr->allow_fp8 ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->allow_fp4 ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->cached_weight && ctx_ptr->cached_weight->Data ? 1 : 0) : 0,
                                ctx_ptr ? (ctx_ptr->cached_fp4_data && ctx_ptr->cached_fp4_scales ? 1 : 0) : 0);
                    } else {
                        fprintf(stderr, "[MATMUL_RECIPE] op=QKV layer=%d used_recipe=0\n",
                                op.attrs.layer_idx);
                    }
                }
                break;
            case modules::MatmulOp::AttnOut:
                log_nan_sample("FWD_ATTN_OUT", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            case modules::MatmulOp::MLPUp:
                log_nan_sample("FWD_MLP_UP", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            case modules::MatmulOp::MLPDown:
                log_nan_sample("FWD_MLP_DOWN", op.attrs.layer_idx, op.outputs[0].name, out, sample_token);
                break;
            default:
                break;
        }
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

    const bool trace_matmul_a = (std::getenv("SUROGATE_TRACE_MATMUL_A") != nullptr);
    const bool assert_recompute_a = (std::getenv("SUROGATE_ASSERT_RECOMPUTE_A") != nullptr);
    const bool is_qkv_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::QKV);
    const bool is_mlp_up_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPUp);
    const bool is_mlp_down_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::MLPDown);
    const bool is_attn_out_op = op.attrs.matmul_op.has_value() &&
        (*op.attrs.matmul_op == modules::MatmulOp::AttnOut);
    static int matmul_a_trace = 0;
    if (trace_matmul_a && matmul_a_trace < 12 && (is_qkv_op || is_mlp_up_op)) {
        const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
        if (layer_idx == 0 || layer_idx == 26 || layer_idx == top_layer) {
            matmul_a_trace++;
            const long sample_token = 3;
            std::vector<float> a_vals(4);
            const bool a_ok = copy_tensor_token_sample_as_f32(a, sample_token, a_vals.size(), a_vals);
            fprintf(stderr,
                    "[MATMUL_BWD_A] micro_step=%d layer=%d op=%s kind=%s input=%s ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                    mMicroStep, layer_idx, op.op_id.c_str(),
                    is_qkv_op ? "qkv" : "mlp_up",
                    op.inputs[1].name.c_str(), a.Data, a_ok ? 1 : 0,
                    a_vals[0], a_vals[1], a_vals[2], a_vals[3]);
            log_tensor_stats_ex(is_qkv_op ? "MATMUL_BWD_A_QKV" : "MATMUL_BWD_A_MLP_UP",
                                layer_idx, op.inputs[1].name, a, 4096, true);
        }
    }

    // Targeted matmul backward trace for layers 8/9 (attn out + mlp down paths).
    static int matmul_l8_in_trace = 0;
    if ((layer_idx == 8 || layer_idx == 9) && (is_attn_out_op || is_mlp_down_op) && matmul_l8_in_trace < 12) {
        const char* kind = is_attn_out_op ? "attn_out" : (is_mlp_down_op ? "mlp_down" : "other");
        fprintf(stderr,
                "[MATMUL_BWD_L8_IN] layer=%d kind=%s d_out=%s A=%s W=%s\n",
                layer_idx,
                kind,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_mag_unbounded("MATMUL_BWD_L8_DOUT", layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_L8_A", layer_idx, op.inputs[1].name, a, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_L8_W", layer_idx, op.inputs[2].name, b, 4096);
        matmul_l8_in_trace++;
    }

    if (assert_recompute_a && (is_qkv_op || is_mlp_up_op) &&
        layer_idx >= 0 && layer_idx < static_cast<int>(mRecomputeSamples.size())) {
        const auto& sample = mRecomputeSamples[static_cast<std::size_t>(layer_idx)];
        const bool want_ln1 = is_qkv_op;
        const bool valid = want_ln1 ? sample.ln1_valid : sample.ln2_valid;
        if (valid && sample.micro_step == mMicroStep) {
            const long sample_token = 3;
            std::vector<float> a_vals;
            if (copy_tensor_token_sample_as_f32(a, sample_token, 4, a_vals)) {
                const std::array<float, 4>& ref = want_ln1 ? sample.ln1 : sample.ln2;
                float max_diff = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    const float diff = std::fabs(a_vals[static_cast<std::size_t>(i)] - ref[static_cast<std::size_t>(i)]);
                    if (diff > max_diff) max_diff = diff;
                }
                if (max_diff > 1e-2f) {
                    fprintf(stderr,
                            "[RECOMPUTE_A_MISMATCH] micro_step=%d layer=%d kind=%s input=%s max_diff=%.6f "
                            "a=%.6f,%.6f,%.6f,%.6f ref=%.6f,%.6f,%.6f,%.6f\n",
                            mMicroStep, layer_idx, want_ln1 ? "qkv" : "mlp_up",
                            op.inputs[1].name.c_str(), max_diff,
                            a_vals[0], a_vals[1], a_vals[2], a_vals[3],
                            ref[0], ref[1], ref[2], ref[3]);
                    throw std::runtime_error("Recompute A mismatch: matmul input does not match recomputed LN output");
                }
            }
        }
    }

    // Targeted QKV matmul backward trace for layers 8/9.
    static int qkv_l8_trace = 0;
    if (is_qkv_op && (layer_idx == 8 || layer_idx == 9) && qkv_l8_trace < 12) {
        fprintf(stderr,
                "[MATMUL_BWD_QKV_L8] layer=%d d_out=%s A=%s W=%s\n",
                layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DOUT", layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_A", layer_idx, op.inputs[1].name, a, 4096);
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_W", layer_idx, op.inputs[2].name, b, 4096);
        const std::size_t dout_total = static_cast<std::size_t>(d_out.nelem());
        if (dout_total > 4096) {
            log_tensor_sample_stats("MATMUL_BWD_QKV_L8_DOUT_MID", d_out, dout_total / 2, 4096);
        }
        qkv_l8_trace++;
    }

    static int matmul_ln2_trace = 0;
    const bool outputs_ln2 = (!op.outputs.empty() &&
                              !op.outputs[0].name.empty() &&
                              strip_ssa_suffix(op.outputs[0].name) == "d_blocks[0].ln2");
    if (outputs_ln2 && matmul_ln2_trace < 8) {
        fprintf(stderr,
                "[MATMUL_BWD_LN2] id=%s in=%s out=%s weight=%s\n",
                op.op_id.c_str(),
                op.inputs[0].name.c_str(),
                op.outputs[0].name.c_str(),
                op.inputs[2].name.c_str());
        log_tensor_stats_ex("MATMUL_BWD_LN2_DOUT", layer_idx, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("MATMUL_BWD_LN2_A", layer_idx, op.inputs[1].name, a, 4096, true);
        log_tensor_stats_ex("MATMUL_BWD_LN2_W", layer_idx, op.inputs[2].name, b, 4096, true);
        matmul_ln2_trace++;
    }

    // DEBUG: Print matmul backward input/output for layer 26 QKV backward
    static int matmul_print_count = 0;
    // Trace Layer 25 and 26 QKV backward for explosion debugging
    static int qkv_25_trace = 0;
    if ((layer_idx == 25 || layer_idx == 26) && is_qkv_op && qkv_25_trace < 20) {
        qkv_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> dout_all(N);
        cudaMemcpy(dout_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float dout_sum_sq = 0.0f, dout_max = 0.0f;
        for (int i = 0; i < N; ++i) {
            dout_sum_sq += dout_all[i] * dout_all[i];
            if (std::fabs(dout_all[i]) > dout_max) dout_max = std::fabs(dout_all[i]);
        }
        fprintf(stderr, "[MATMUL_BWD_QKV] Layer %d: d_out name='%s' slot=%d ptr=%p L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, op.inputs[0].name.c_str(), static_cast<int>(op.inputs[0].slot), d_out.Data,
                std::sqrt(dout_sum_sq), dout_max, dout_all[0], dout_all[1], dout_all[2], dout_all[3]);
    }
    if ((layer_idx == 26 && is_qkv_op) && matmul_print_count < 3) {
        matmul_print_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        // Print the input tensor ref to see where d_out is coming from
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: d_out name='%s' slot=%d ptr=%p\n",
                layer_idx, op.inputs[0].name.c_str(), static_cast<int>(op.inputs[0].slot), d_out.Data);
        // Compute L2 norm of dout to see the full tensor magnitude
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> dout_all(N);
        cudaMemcpy(dout_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float dout_sum_sq = 0.0f;
        float dout_max = 0.0f;
        int dout_nonzero = 0;
        for (int i = 0; i < N; ++i) {
            dout_sum_sq += dout_all[i] * dout_all[i];
            if (std::fabs(dout_all[i]) > dout_max) dout_max = std::fabs(dout_all[i]);
            if (std::fabs(dout_all[i]) > 1e-10f) dout_nonzero++;
        }
        float dout_norm = std::sqrt(dout_sum_sq);
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: dout L2 norm=%.9f, max=%.9f, nonzero=%d/%d\n",
                layer_idx, dout_norm, dout_max, dout_nonzero, N);
        // Print some middle values to see if pattern is different
        int mid = N / 2;
        fprintf(stderr, "[MATMUL_BWD] Layer %d QKV: dout[%d..%d]=%.9f,%.9f,%.9f,%.9f\n",
                layer_idx, mid, mid+3, dout_all[mid], dout_all[mid+1], dout_all[mid+2], dout_all[mid+3]);
    }

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
        if (disable_qkv_recipe_bwd && is_qkv_op) {
            if (d_out_mat.Rank > 2 && d_out_mat.Sizes[0] == mB && d_out_mat.Sizes[1] == mT) {
                d_out_mat = view_tensor(d_out_mat, {mB * mT, d_out_mat.Sizes[d_out_mat.Rank - 1]});
            }
            if (a_mat.Rank > 2 && a_mat.Sizes[0] == mB && a_mat.Sizes[1] == mT) {
                a_mat = view_tensor(a_mat, {mB * mT, a_mat.Sizes[a_mat.Rank - 1]});
            }
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

    // One-time NaN watchdog for matmul backward outputs.
    static bool matmul_qkv_nan_logged = false;
    static int matmul_attn_nan_logged = 0;
    auto dump_sample = [](const char* tag, const Tensor& t, const std::string& name) {
        std::vector<float> vals(4, 0.0f);
        const bool ok = copy_tensor_sample_as_f32(t, vals.size(), vals);
        int nan = 0;
        int inf = 0;
        for (float v : vals) {
            if (std::isnan(v)) {
                nan++;
            } else if (std::isinf(v)) {
                inf++;
            }
        }
        fprintf(stderr,
                "[%s] name=%s dtype=%s ok=%d nan=%d inf=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                tag,
                name.c_str(),
                dtype_to_str(t.DType),
                ok ? 1 : 0,
                nan,
                inf,
                vals[0], vals[1], vals[2], vals[3]);
    };
    if (is_qkv_op && dA_ptr && !matmul_qkv_nan_logged && tensor_sample_has_nan_or_inf(*dA_ptr, 3)) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        Tensor d_out_scan = d_out;
        if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
            d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
        }
        float scan_min = 0.0f;
        float scan_max = 0.0f;
        const bool scan_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &scan_min, &scan_max);
        fprintf(stderr,
                "[MATMUL_BWD_QKV_NAN] op=%s layer=%d dA=%s d_out=%s weight=%s allow_quant=%d used_recipe=%d used_fp8=%d has_dout_quant=%d dtype=%s ptr=%p d_out_shape=%s d_out_scan_shape=%s a_shape=%s b_shape=%s d_out_row_nan=%d row_min=%.6f row_max=%.6f\n",
                op.op_id.c_str(),
                layer_idx,
                op.outputs[0].name.c_str(),
                op.inputs[0].name.c_str(),
                op.inputs[2].name.c_str(),
                allow_quant ? 1 : 0,
                used_recipe ? 1 : 0,
                used_fp8 ? 1 : 0,
                has_dout_quant ? 1 : 0,
                dtype_to_str(b.DType),
                b.Data,
                shape_str(d_out).c_str(),
                shape_str(d_out_scan).c_str(),
                shape_str(a).c_str(),
                shape_str(b).c_str(),
                scan_nan ? 1 : 0,
                scan_min,
                scan_max);
        dump_sample("MATMUL_BWD_QKV_NAN_DOUT", d_out, op.inputs[0].name);
        dump_sample("MATMUL_BWD_QKV_NAN_DA", *dA_ptr, op.outputs[0].name);
        dump_sample("MATMUL_BWD_QKV_NAN_W", b, op.inputs[2].name);
        matmul_qkv_nan_logged = true;
    }
    if (is_attn_out_op && dA_ptr && matmul_attn_nan_logged < 4) {
        Tensor d_out_scan = d_out;
        if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
            d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
        }
        Tensor dA_scan = *dA_ptr;
        if (dA_scan.Rank > 2 && dA_scan.Sizes[0] == mB && dA_scan.Sizes[1] == mT) {
            dA_scan = view_tensor(dA_scan, {mB * mT, dA_scan.Sizes[dA_scan.Rank - 1]});
        }
        long d_out_row = -1;
        long dA_row = -1;
        float d_out_min = 0.0f;
        float d_out_max = 0.0f;
        float dA_min = 0.0f;
        float dA_max = 0.0f;
        const bool d_out_nan = find_first_nan_row(d_out_scan, &d_out_row, &d_out_min, &d_out_max);
        const bool dA_nan = find_first_nan_row(dA_scan, &dA_row, &dA_min, &dA_max);
        if (d_out_nan || dA_nan) {
            fprintf(stderr,
                    "[MATMUL_BWD_ATTN_NAN] op=%s layer=%d d_out=%s dA=%s d_out_nan=%d dA_nan=%d\n",
                    op.op_id.c_str(),
                    layer_idx,
                    op.inputs[0].name.c_str(),
                    op.outputs[0].name.c_str(),
                    d_out_nan ? 1 : 0,
                    dA_nan ? 1 : 0);
            if (d_out_nan) {
                const long b = (d_out_scan.Rank >= 2 && d_out_scan.Sizes[0] == mB * mT)
                    ? (d_out_row / static_cast<long>(mT)) : -1;
                const long t = (d_out_scan.Rank >= 2 && d_out_scan.Sizes[0] == mB * mT)
                    ? (d_out_row % static_cast<long>(mT)) : -1;
                fprintf(stderr,
                        "[MATMUL_BWD_ATTN_NAN_DOUT_ROW] row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                        d_out_row, b, t, d_out_min, d_out_max);
            }
            if (dA_nan) {
                const long b = (dA_scan.Rank >= 2 && dA_scan.Sizes[0] == mB * mT)
                    ? (dA_row / static_cast<long>(mT)) : -1;
                const long t = (dA_scan.Rank >= 2 && dA_scan.Sizes[0] == mB * mT)
                    ? (dA_row % static_cast<long>(mT)) : -1;
                fprintf(stderr,
                        "[MATMUL_BWD_ATTN_NAN_DA_ROW] row=%ld b=%ld t=%ld min=%.6f max=%.6f\n",
                        dA_row, b, t, dA_min, dA_max);
            }
            dump_sample("MATMUL_BWD_ATTN_NAN_DOUT", d_out, op.inputs[0].name);
            dump_sample("MATMUL_BWD_ATTN_NAN_DA", *dA_ptr, op.outputs[0].name);
            dump_sample("MATMUL_BWD_ATTN_NAN_W", b, op.inputs[2].name);
            matmul_attn_nan_logged++;
        }
    }

    static int qkv_l8_out_trace = 0;
    if (is_qkv_op && (layer_idx == 8 || layer_idx == 9) && dA_ptr && qkv_l8_out_trace < 12) {
        log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DA", layer_idx,
                                 op.outputs.empty() ? "<none>" : op.outputs[0].name, *dA_ptr, 4096);
        if (dB_ptr) {
            log_tensor_mag_unbounded("MATMUL_BWD_QKV_L8_DB", layer_idx,
                                     op.outputs.size() > 1 ? op.outputs[1].name : "<none>", *dB_ptr, 4096);
        }
        const std::size_t da_total = static_cast<std::size_t>(dA_ptr->nelem());
        if (da_total > 4096) {
            log_tensor_sample_stats("MATMUL_BWD_QKV_L8_DA_MID", *dA_ptr, da_total / 2, 4096);
        }
        qkv_l8_out_trace++;
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

    // Targeted matmul backward output trace for layers 8/9.
    static int matmul_l8_out_trace = 0;
    if ((layer_idx == 8 || layer_idx == 9) && (is_attn_out_op || is_mlp_down_op) &&
        dA_ptr && matmul_l8_out_trace < 12) {
        const char* kind = is_attn_out_op ? "attn_out" : (is_mlp_down_op ? "mlp_down" : "other");
        fprintf(stderr,
                "[MATMUL_BWD_L8_OUT] layer=%d kind=%s dA=%s dB=%s\n",
                layer_idx,
                kind,
                op.outputs.empty() ? "<none>" : op.outputs[0].name.c_str(),
                (op.outputs.size() > 1) ? op.outputs[1].name.c_str() : "<none>");
        log_tensor_mag_unbounded("MATMUL_BWD_L8_DA", layer_idx,
                                 op.outputs.empty() ? "<none>" : op.outputs[0].name, *dA_ptr, 4096);
        if (dB_ptr) {
            log_tensor_mag_unbounded("MATMUL_BWD_L8_DB", layer_idx,
                                     op.outputs.size() > 1 ? op.outputs[1].name : "<none>", *dB_ptr, 4096);
        }
        matmul_l8_out_trace++;
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
