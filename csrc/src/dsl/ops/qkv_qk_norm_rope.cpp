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

namespace dsl {
    
void CompiledExecutor::dispatch_qkv_qk_norm_rope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& q_norm = resolve_tensor(op.inputs[1]);
    Tensor& k_norm = resolve_tensor(op.inputs[2]);
    Tensor& freqs = resolve_tensor(op.inputs[3]);
    Tensor& pos_ids = resolve_tensor(op.inputs[4]);

    // Get output tensor from pre-allocated slot if available
    Tensor& qkv_out = ensure_output_tensor(op.outputs[0]);
    Tensor& q_rstd = ensure_output_tensor(op.outputs[1]);
    Tensor& k_rstd = ensure_output_tensor(op.outputs[2]);

    // DEBUG: Trace qkv_in and qkv_out for layer 0
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[0].name, debug_layer_idx, debug_field);
    static int fwd_qknorm_count = 0;
    if (debug_layer_idx == 0 && fwd_qknorm_count < 3) {
        fwd_qknorm_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> in_vals(4), out_vals(4);
        cudaMemcpy(in_vals.data(), qkv_in.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_vals.data(), qkv_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d: qkv_in=%s ptr=%p, qkv_out=%s ptr=%p, same=%d\n",
                debug_layer_idx, op.inputs[0].name.c_str(), qkv_in.Data,
                op.outputs[0].name.c_str(), qkv_out.Data, (qkv_in.Data == qkv_out.Data) ? 1 : 0);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d: qkv_in values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
    }

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    // If input and output are different buffers, copy input to output first
    // The kernel operates in-place on the output buffer
    if (qkv_in.Data != qkv_out.Data) {
        cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                        qkv_in.bytes(),
                        cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    Tensor qkv_view = (qkv_out.Rank == 4) ? view_tensor(qkv_out, {mB, mT, qkv_channels}) : qkv_out;
    int rotary_dim = op.attrs.rotary_dim;

    const bool rope_fusable = (rotary_dim > 0)
        && ((Hs % 2) == 0)
        && (((Hs / 2) % 32) == 0)
        && (freqs.Rank >= 2)
        && (freqs.Sizes[1] >= Hs)
        && (qkv_view.Rank == 3);

    if (mForwardPlan) {
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(op.inputs[0].name, layer_idx, field) &&
            layer_idx >= 0 && static_cast<std::size_t>(layer_idx) < mForwardPlan->size()) {
            AttnForwardPlan plan{};
            plan.valid = true;
            plan.use_qk_norm = true;
            plan.rope_fused = rope_fusable;
            plan.use_cudnn = cudnn_gqa_ok;
            plan.rotary_dim = rotary_dim;
            (*mForwardPlan)[static_cast<std::size_t>(layer_idx)].attn = plan;
        }
    }

    if (rope_fusable) {
        qkv_qk_norm_rope_forward(qkv_view, q_rstd, k_rstd, q_norm, k_norm,
                                 freqs, reinterpret_cast<int*>(pos_ids.Data),
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        const int q_rows = Hq * Hs;
        qkv_head_rmsnorm_forward(qkv_view, q_rstd, q_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_forward(qkv_view, k_rstd, k_norm,
                                 op.attrs.eps,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 qkv_channels, Hkv, Hs, q_rows, mRunState.MainStream);
        rope_forward(qkv_out, qkv_out, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                     static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
    }

    // DEBUG: Print qkv_in and qkv_out after computation for layer 0
    if (debug_layer_idx == 0 && fwd_qknorm_count <= 3) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> in_vals(4), out_vals(4);
        cudaMemcpy(in_vals.data(), qkv_in.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_vals.data(), qkv_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d AFTER: qkv_in values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, in_vals[0], in_vals[1], in_vals[2], in_vals[3]);
        fprintf(stderr, "[FWD_QK_NORM_ROPE] Layer %d AFTER: qkv_out values=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
    }

    mTensorMap[op.outputs[0].name] = qkv_out;

    // NaN detection for QKV after RoPE/QK-norm (layer 25, token 3)
    int layer_idx = -1;
    std::string field;
    if (parse_block_param(op.outputs[0].name, layer_idx, field)) {
        const long sample_token = 3;
        log_nan_sample("FWD_QKV_ROPE", layer_idx, op.outputs[0].name, qkv_out, sample_token);
        if (op.outputs.size() > 1) {
            Tensor& q_rstd = resolve_tensor(op.outputs[1]);
            log_nan_sample("FWD_Q_RSTD", layer_idx, op.outputs[1].name, q_rstd, sample_token);
        }
        if (op.outputs.size() > 2) {
            Tensor& k_rstd = resolve_tensor(op.outputs[2]);
            log_nan_sample("FWD_K_RSTD", layer_idx, op.outputs[2].name, k_rstd, sample_token);
        }
        if (layer_idx == 0) {
            log_tensor_stats("FWD_QKV_IN", layer_idx, op.inputs[0].name, qkv_in, 4096);
            log_tensor_stats("FWD_QKV_ROPE", layer_idx, op.outputs[0].name, qkv_out, 4096);
            log_tensor_stats("FWD_Q_NORM_W", layer_idx, op.inputs[1].name, q_norm, 2048);
            log_tensor_stats("FWD_K_NORM_W", layer_idx, op.inputs[2].name, k_norm, 2048);
            if (op.outputs.size() > 1) {
                Tensor& q_rstd = resolve_tensor(op.outputs[1]);
                log_tensor_stats("FWD_Q_RSTD", layer_idx, op.outputs[1].name, q_rstd, 2048);
            }
            if (op.outputs.size() > 2) {
                Tensor& k_rstd = resolve_tensor(op.outputs[2]);
                log_tensor_stats("FWD_K_RSTD", layer_idx, op.outputs[2].name, k_rstd, 2048);
            }
        }
    }
}

void CompiledExecutor::dispatch_qkv_qk_norm_rope_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_qkv_out, qkv_out (saved), q_norm_weight, k_norm_weight, q_rstd, k_rstd, freqs, pos_ids
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& qkv = resolve_tensor(op.inputs[1]);       // Saved QKV output from forward
    Tensor& q_norm = resolve_tensor(op.inputs[2]);
    Tensor& k_norm = resolve_tensor(op.inputs[3]);
    Tensor& q_rstd = resolve_tensor(op.inputs[4]);    // Saved RSTD (FP32)
    Tensor& k_rstd = resolve_tensor(op.inputs[5]);    // Saved RSTD (FP32)
    Tensor& freqs = resolve_tensor(op.inputs[6]);
    Tensor& pos_ids = resolve_tensor(op.inputs[7]);

    // DEBUG: Trace inputs with L2 norms
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[1].name, debug_layer_idx, debug_field);
    static int qknorm_trace_count = 0;
    if (debug_layer_idx == 0 && qknorm_trace_count < 10) {
        qknorm_trace_count++;
        cudaStreamSynchronize(mRunState.MainStream);
        // Compute L2 norm of d_out
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> d_out_all(N);
        cudaMemcpy(d_out_all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float d_out_sum_sq = 0.0f, d_out_max = 0.0f;
        for (int i = 0; i < N; ++i) {
            d_out_sum_sq += d_out_all[i] * d_out_all[i];
            if (std::fabs(d_out_all[i]) > d_out_max) d_out_max = std::fabs(d_out_all[i]);
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_out INPUT ptr=%p L2=%.6f, max=%.6f, first4=%.9f,%.9f,%.9f,%.9f\n",
                debug_layer_idx, d_out.Data, std::sqrt(d_out_sum_sq), d_out_max, d_out_all[0], d_out_all[1], d_out_all[2], d_out_all[3]);
    }

    // Targeted trace for layers 8/9.
    static int qk_l8_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && qk_l8_trace < 12) {
        fprintf(stderr,
                "[QK_BWD_L8_IN] layer=%d d_out=%s qkv=%s q_rstd=%s k_rstd=%s\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[4].name.c_str(),
                op.inputs[5].name.c_str());
        log_tensor_mag_unbounded("QK_BWD_L8_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_QKV", debug_layer_idx, op.inputs[1].name, qkv, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_QRSTD", debug_layer_idx, op.inputs[4].name, q_rstd, 4096);
        log_tensor_mag_unbounded("QK_BWD_L8_KRSTD", debug_layer_idx, op.inputs[5].name, k_rstd, 4096);
        const std::size_t dout_total = static_cast<std::size_t>(d_out.nelem());
        const std::size_t qkv_total = static_cast<std::size_t>(qkv.nelem());
        if (dout_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_DOUT_MID", d_out, dout_total / 2, 4096);
        }
        if (qkv_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_QKV_MID", qkv, qkv_total / 2, 4096);
        }
        qk_l8_trace++;
    }

    const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
    // DEBUG: Trace top-layer inputs (qkv/rstd/freqs/pos_ids) to catch NaN sources.
    static int qk_top_trace = 0;
    if (debug_layer_idx == top_layer && qk_top_trace < 6) {
        fprintf(stderr,
                "[QK_NORM_ROPE_BWD_TOP] layer=%d qkv=%p d_out=%p q_rstd=%p k_rstd=%p freqs=%p pos_ids=%p qkv_shape=%s d_out_shape=%s\n",
                debug_layer_idx,
                qkv.Data,
                d_out.Data,
                q_rstd.Data,
                k_rstd.Data,
                freqs.Data,
                pos_ids.Data,
                tensor_shape_str(qkv).c_str(),
                tensor_shape_str(d_out).c_str());
        log_tensor_stats_ex("QK_BWD_TOP_QKV", debug_layer_idx, op.inputs[1].name, qkv, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_Q_RSTD", debug_layer_idx, op.inputs[4].name, q_rstd, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_K_RSTD", debug_layer_idx, op.inputs[5].name, k_rstd, 4096, true);
        log_tensor_stats_ex("QK_BWD_TOP_FREQS", debug_layer_idx, op.inputs[6].name, freqs, 4096, true);
        if (pos_ids.Data && pos_ids.nelem() > 0) {
            const std::size_t n = std::min<std::size_t>(8, static_cast<std::size_t>(pos_ids.nelem()));
            std::vector<int> ids(n, 0);
            cudaMemcpy(ids.data(), pos_ids.Data, n * sizeof(int), cudaMemcpyDeviceToHost);
            fprintf(stderr,
                    "[QK_BWD_TOP_POS_IDS] first=%d,%d,%d,%d,%d,%d,%d,%d\n",
                    ids.size() > 0 ? ids[0] : 0,
                    ids.size() > 1 ? ids[1] : 0,
                    ids.size() > 2 ? ids[2] : 0,
                    ids.size() > 3 ? ids[3] : 0,
                    ids.size() > 4 ? ids[4] : 0,
                    ids.size() > 5 ? ids[5] : 0,
                    ids.size() > 6 ? ids[6] : 0,
                    ids.size() > 7 ? ids[7] : 0);
        }
        qk_top_trace++;
    }

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int q_rows = Hq * Hs;

    Tensor qkv_view = (qkv.Rank == 4) ? view_tensor(qkv, {mB, mT, static_cast<long>(qkv_channels)}) : qkv;
    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    // Initialize d_qkv with upstream gradient (d_out) so V gradients pass through unchanged.
    // The fused or fallback kernels update Q/K channels in-place.
    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    const bool disable_fused = env_enabled("SUROGATE_DISABLE_FUSED_QK_ROPE_BWD");
    if (disable_fused) {
        // Fallback: undo RoPE on gradients and activations, then run non-RoPE QK RMSNorm backward.
        const int rotary_dim = op.attrs.rotary_dim;
        rope_backward(d_qkv_view, d_qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        rope_backward(qkv_view, qkv_view, freqs, reinterpret_cast<int*>(pos_ids.Data), nullptr,
                      static_cast<int>(mB), static_cast<int>(mT),
                      Hq, Hkv, Hs, rotary_dim, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hq, Hs, 0, mRunState.MainStream);
        qkv_head_rmsnorm_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                     static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                     Hkv, Hs, q_rows, mRunState.MainStream);
    } else {
        // Combined backward for Q and K norms with RoPE
        // Q norm backward (with RoPE): channel_offset=0
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, q_norm, q_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hq, Hs, 0, mRunState.MainStream, nullptr);

        // K norm backward (with RoPE): channel_offset=q_rows
        qkv_head_rmsnorm_rope_backward_dx(d_qkv_view, qkv_view, k_norm, k_rstd,
                                           freqs, reinterpret_cast<int*>(pos_ids.Data),
                                           static_cast<int>(mB), static_cast<int>(mT), qkv_channels,
                                           Hkv, Hs, q_rows, mRunState.MainStream, nullptr);
    }

    // One-time NaN watchdog for QK-norm/RoPE backward output.
    static bool qk_bwd_nan_logged = false;
    Tensor d_out_scan = d_out_view;
    Tensor d_qkv_scan = d_qkv_view;
    if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
        d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
    }
    if (d_qkv_scan.Rank > 2 && d_qkv_scan.Sizes[0] == mB && d_qkv_scan.Sizes[1] == mT) {
        d_qkv_scan = view_tensor(d_qkv_scan, {mB * mT, d_qkv_scan.Sizes[d_qkv_scan.Rank - 1]});
    }
    float dq_row_min = 0.0f;
    float dq_row_max = 0.0f;
    const bool dq_row_nan = tensor_row_has_nan_or_inf(d_qkv_scan, 3, &dq_row_min, &dq_row_max);
    if (!qk_bwd_nan_logged && dq_row_nan) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        float row_min = 0.0f;
        float row_max = 0.0f;
        const bool row_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &row_min, &row_max);
        fprintf(stderr,
                "[QK_NORM_ROPE_BWD_NAN] layer=%d d_out=%s d_qkv=%s d_out_shape=%s d_qkv_shape=%s d_out_row_nan=%d row_min=%.6f row_max=%.6f d_qkv_row_min=%.6f d_qkv_row_max=%.6f\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.outputs[0].name.c_str(),
                shape_str(d_out_scan).c_str(),
                shape_str(d_qkv_scan).c_str(),
                row_nan ? 1 : 0,
                row_min,
                row_max,
                dq_row_min,
                dq_row_max);
        qk_bwd_nan_logged = true;
    }

    // V doesn't have normalization - its gradients pass through unchanged
    // The d_out already contains the V gradients at the correct offset

    // DEBUG: Print output with L2 norm (include Layer 25 and top layer for debugging)
    static int qk_25_trace = 0;
    if ((debug_layer_idx == 25 || debug_layer_idx == top_layer) && qk_25_trace < 12) {
        qk_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
        std::vector<float> d_qkv_all(N);
        cudaMemcpy(d_qkv_all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_sq += d_qkv_all[i] * d_qkv_all[i];
            if (std::fabs(d_qkv_all[i]) > max_val) max_val = std::fabs(d_qkv_all[i]);
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_qkv OUTPUT L2=%.6f, max=%.6f, vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, std::sqrt(sum_sq), max_val, d_qkv_all[0], d_qkv_all[1], d_qkv_all[2], d_qkv_all[3]);
    }
    static int qk_l8_out_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && qk_l8_out_trace < 12) {
        log_tensor_mag_unbounded("QK_BWD_L8_DQKV", debug_layer_idx, op.outputs[0].name, d_qkv, 4096);
        const std::size_t dqkv_total = static_cast<std::size_t>(d_qkv.nelem());
        if (dqkv_total > 4096) {
            log_tensor_sample_stats("QK_BWD_L8_DQKV_MID", d_qkv, dqkv_total / 2, 4096);
        }
        qk_l8_out_trace++;
    }
    if ((debug_layer_idx == 0 || debug_layer_idx == 26) && qknorm_trace_count <= 10) {
        cudaStreamSynchronize(mRunState.MainStream);
        const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
        std::vector<float> d_qkv_all(N);
        cudaMemcpy(d_qkv_all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        int nonzero = 0;
        for (int i = 0; i < N; ++i) {
            sum_sq += d_qkv_all[i] * d_qkv_all[i];
            if (std::fabs(d_qkv_all[i]) > max_val) max_val = std::fabs(d_qkv_all[i]);
            if (std::fabs(d_qkv_all[i]) > 1e-10f) nonzero++;
        }
        fprintf(stderr, "[QK_NORM_ROPE_BWD] Layer %d: d_qkv OUTPUT name=%s ptr=%p, L2=%.6f, max=%.6f, nonzero=%d/%d, vals[0..3]=%.9f,%.9f,%.9f,%.9f\n",
                debug_layer_idx, op.outputs[0].name.c_str(), d_qkv.Data, std::sqrt(sum_sq), max_val, nonzero, N,
                d_qkv_all[0], d_qkv_all[1], d_qkv_all[2], d_qkv_all[3]);
    }

    // For FP8 hybrid backward, record abs_max of the final d_qkv for subsequent quantization
    if (mRunState.has_fp8_hybrid_backward()) {
        float* abs_max_ptr = mRunState.simplified_quant_grads().d_qkv.abs_max();
        abs_max(abs_max_ptr, d_qkv_view, static_cast<long>(d_qkv_view.nelem()),
                mRunState.DeviceProp, mRunState.MainStream);
    }
}

}  // namespace dsl
