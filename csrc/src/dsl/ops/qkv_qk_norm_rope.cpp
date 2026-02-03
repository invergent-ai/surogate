#include "dsl/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

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

    const int op_layer_idx = (op.inputs.size() > 0 && op.inputs[0].layer_idx >= 0)
                                 ? op.inputs[0].layer_idx
                                 : op.attrs.layer_idx;

    if (env_int("SUROGATE_QKV_PTR_TRACE", 0)) {
        std::cerr << "[QKV_ROPE_PTR] layer=" << op_layer_idx
                  << " in_name=" << op.inputs[0].name
                  << " in_ptr=" << qkv_in.Data
                  << " out_ptr=" << qkv_out.Data
                  << std::endl;
    }

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    const int copy_trace = env_int("SUROGATE_QK_ROPE_COPY_TRACE", 0);
    if (copy_trace && !mCapturing) {
        std::cerr << "[QK_ROPE_COPY] layer=" << op_layer_idx
                  << " in_ptr=" << qkv_in.Data
                  << " out_ptr=" << qkv_out.Data
                  << " in_dtype=" << static_cast<int>(qkv_in.DType)
                  << " out_dtype=" << static_cast<int>(qkv_out.DType)
                  << " in_shape=" << tensor_shape_str(qkv_in)
                  << " out_shape=" << tensor_shape_str(qkv_out)
                  << " in_bytes=" << qkv_in.bytes()
                  << " out_bytes=" << qkv_out.bytes()
                  << " copy=" << (qkv_in.Data != qkv_out.Data ? 1 : 0)
                  << std::endl;
    }

    // If input and output are different buffers, copy input to output first.
    // The kernel operates in-place on the output buffer.
    if (qkv_in.Data != qkv_out.Data) {
        cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                        qkv_in.bytes(),
                        cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    Tensor qkv_view = (qkv_out.Rank == 4) ? view_tensor(qkv_out, {mB, mT, qkv_channels}) : qkv_out;
    int rotary_dim = op.attrs.rotary_dim;

    if (env_int("SUROGATE_QKV_CANARY", 0)) {
        mRunState.check_qkv_canary(mRunState.MainStream,
                                   "pre_qk_rope",
                                   op_layer_idx,
                                   mMicroStep,
                                   op.op_id.c_str());
    }

    const int sample_trace = env_int("SUROGATE_QK_ROPE_SAMPLE_TRACE", 0);
    const int sample_layer = env_int("SUROGATE_QK_ROPE_SAMPLE_TRACE_LAYER", -1);
    const bool sample_layer_ok = (sample_layer < 0) || (op_layer_idx < 0) || (op_layer_idx == sample_layer);
    if (sample_trace && sample_layer_ok && !mCapturing) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        std::vector<float> sample;
        if (copy_tensor_sample_offset_as_f32(qkv_view, 0, 8, sample) && !sample.empty()) {
            std::ostringstream oss;
            for (std::size_t i = 0; i < sample.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << sample[i];
            }
            std::cerr << "[QK_ROPE_SAMPLE] layer=" << op_layer_idx
                      << " micro=" << mMicroStep
                      << " op_idx=" << op.original_idx
                      << " op_id=" << op.op_id
                      << " tag=qkv_pre"
                      << " vals=" << oss.str()
                      << std::endl;

            if (env_int("SUROGATE_QKV_GUARD", 0) && sample.size() >= 8) {
                std::array<float, 8> curr{};
                for (std::size_t i = 0; i < 8; ++i) {
                    curr[i] = sample[i];
                }
                QkvGuardSample prev;
                if (fetch_qkv_guard_sample(qkv_view.Data, op_layer_idx, mMicroStep, prev)) {
                    float max_abs_diff = 0.0f;
                    bool has_nan = false;
                    for (std::size_t i = 0; i < 8; ++i) {
                        if (std::isnan(curr[i]) || std::isnan(prev.vals[i])) {
                            has_nan = true;
                            continue;
                        }
                        max_abs_diff = std::max(max_abs_diff, std::fabs(curr[i] - prev.vals[i]));
                    }
                    if (has_nan || max_abs_diff > 0.0f) {
                        std::ostringstream prev_oss;
                        for (std::size_t i = 0; i < 8; ++i) {
                            if (i) {
                                prev_oss << ",";
                            }
                            prev_oss << prev.vals[i];
                        }
                        std::string writer_note;
                        QkvLastWriter writer;
                        if (fetch_qkv_last_writer(qkv_view.Data, writer)) {
                            std::ostringstream w;
                            w << " last_op_idx=" << writer.op_idx
                              << " last_op_id=" << writer.op_id
                              << " last_type=" << writer.op_type
                              << " last_out=" << writer.out_name
                              << " last_layer=" << writer.layer
                              << " last_micro=" << writer.micro;
                            writer_note = w.str();
                        }
                        std::string kernel_note;
                        QkvKernelWriter kernel_writer;
                        if (fetch_qkv_kernel_writer(qkv_view.Data, kernel_writer)) {
                            std::ostringstream w;
                            w << " kernel_op_idx=" << kernel_writer.op_idx
                              << " kernel_op_id=" << kernel_writer.op_id
                              << " kernel_type=" << kernel_writer.op_type
                              << " kernel_out=" << kernel_writer.out_name
                              << " kernel_layer=" << kernel_writer.layer
                              << " kernel_micro=" << kernel_writer.micro;
                            kernel_note = w.str();
                        }
                        std::cerr << "[QKV_GUARD_DIFF] layer=" << op_layer_idx
                                  << " micro=" << mMicroStep
                                  << " op_idx=" << op.original_idx
                                  << " op_id=" << op.op_id
                                  << " prev_op_idx=" << prev.op_idx
                                  << " prev_op_id=" << prev.op_id
                                  << " max_abs_diff=" << max_abs_diff
                                  << " has_nan=" << (has_nan ? 1 : 0)
                                  << " prev_vals=" << prev_oss.str()
                                  << " curr_vals=" << oss.str()
                                  << writer_note
                                  << kernel_note
                                  << std::endl;
                    }
                }
            }
        }
    }

    const int nan_trace = env_int("SUROGATE_QK_ROPE_FWD_NAN_TRACE", 0);
    const int trace_layer = env_int("SUROGATE_QK_ROPE_FWD_NAN_TRACE_LAYER", -1);
    const bool trace_layer_ok = (trace_layer < 0) || (op_layer_idx < 0) || (op_layer_idx == trace_layer);
    auto log_nan = [&](const Tensor& t, const char* tag) -> bool {
        if (!nan_trace || !trace_layer_ok || !t.Data || mCapturing) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
            return false;
        }
        std::cerr << "[QK_ROPE_FWD_NAN] layer=" << op_layer_idx
                  << " micro=" << mMicroStep
                  << " op_idx=" << op.original_idx
                  << " op_id=" << op.op_id
                  << " tag=" << (tag ? tag : "<unnamed>")
                  << " shape=" << tensor_shape_str(t)
                  << " row=" << row
                  << " min=" << min_val
                  << " max=" << max_val
                  << " dtype=" << static_cast<int>(t.DType)
                  << std::endl;
        return true;
    };

    log_nan(qkv_view, "qkv_pre");

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

    const bool qkv_nan = log_nan(qkv_out, "qkv_post");
    if (qkv_nan) {
        log_nan(q_rstd, "q_rstd");
        log_nan(k_rstd, "k_rstd");
    }

    mTensorMap[op.outputs[0].name] = qkv_out;
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

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int q_rows = Hq * Hs;

    Tensor qkv_view = (qkv.Rank == 4) ? view_tensor(qkv, {mB, mT, static_cast<long>(qkv_channels)}) : qkv;
    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    const int bwd_trace = env_int("SUROGATE_QK_ROPE_BWD_TRACE", 0);
    const int bwd_trace_layer = env_int("SUROGATE_QK_ROPE_BWD_TRACE_LAYER", -1);
    const int bwd_trace_limit = env_int("SUROGATE_QK_ROPE_BWD_TRACE_LIMIT", 8);
    const int bwd_trace_samples = env_int("SUROGATE_QK_ROPE_BWD_TRACE_SAMPLES", 8);
    static std::atomic<int> trace_count{0};

    const int nan_trace = env_int("SUROGATE_QK_ROPE_BWD_NAN_TRACE", 0);
    const int trace_layer = env_int("SUROGATE_QK_ROPE_BWD_NAN_TRACE_LAYER", -1);
    const int op_layer_idx = (op.inputs.size() > 0 && op.inputs[0].layer_idx >= 0)
                                 ? op.inputs[0].layer_idx
                                 : op.attrs.layer_idx;
    const bool trace_layer_ok = (trace_layer < 0) || (op_layer_idx < 0) || (op_layer_idx == trace_layer);
    auto log_nan = [&](const Tensor& t, const char* tag) -> bool {
        if (!nan_trace || !trace_layer_ok || !t.Data) {
            return false;
        }
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
            return false;
        }
        std::cerr << "[QK_ROPE_BWD_NAN] layer=" << op_layer_idx
                  << " tag=" << (tag ? tag : "<unnamed>")
                  << " row=" << row
                  << " min=" << min_val
                  << " max=" << max_val
                  << " dtype=" << static_cast<int>(t.DType)
                  << std::endl;
        return true;
    };

    const bool do_trace = bwd_trace && !mCapturing &&
        (bwd_trace_layer < 0 || op_layer_idx < 0 || op_layer_idx == bwd_trace_layer) &&
        (bwd_trace_limit <= 0 || trace_count.fetch_add(1) < bwd_trace_limit);

    auto trace_sample = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            std::cerr << "[QK_ROPE_BWD_TRACE] layer=" << op_layer_idx
                      << " micro=" << mMicroStep
                      << " tag=" << (tag ? tag : "<unnamed>")
                      << " dtype=" << static_cast<int>(t.DType)
                      << " shape=" << tensor_shape_str(t)
                      << " ptr=<null>"
                      << std::endl;
            return;
        }
        std::vector<float> vals;
        if (!copy_tensor_token_sample_as_f32(t, 0, static_cast<std::size_t>(bwd_trace_samples), vals) || vals.empty()) {
            std::cerr << "[QK_ROPE_BWD_TRACE] layer=" << op_layer_idx
                      << " micro=" << mMicroStep
                      << " tag=" << (tag ? tag : "<unnamed>")
                      << " dtype=" << static_cast<int>(t.DType)
                      << " shape=" << tensor_shape_str(t)
                      << " ptr=" << t.Data
                      << " sample=<unavailable>"
                      << std::endl;
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
        std::cerr << "[QK_ROPE_BWD_TRACE] layer=" << op_layer_idx
                  << " micro=" << mMicroStep
                  << " tag=" << (tag ? tag : "<unnamed>")
                  << " dtype=" << static_cast<int>(t.DType)
                  << " shape=" << tensor_shape_str(t)
                  << " ptr=" << t.Data
                  << " min=" << min_v
                  << " max=" << max_v
                  << " max_abs=" << max_abs
                  << " mean_abs=" << mean_abs
                  << std::endl;
    };

    // Initialize d_qkv with upstream gradient (d_out) so V gradients pass through unchanged.
    // The fused or fallback kernels update Q/K channels in-place.
    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    if (do_trace) {
        trace_sample(d_out_view, "d_qkv_out");
        trace_sample(qkv_view, "qkv");
        trace_sample(q_rstd, "q_rstd");
        trace_sample(k_rstd, "k_rstd");
    }

    const bool d_out_nan = log_nan(d_out_view, "d_qkv_out");
    if (d_out_nan) {
        log_nan(qkv_view, "qkv");
        log_nan(q_rstd, "q_rstd");
        log_nan(k_rstd, "k_rstd");
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

    // V doesn't have normalization - its gradients pass through unchanged
    // The d_out already contains the V gradients at the correct offset

    // For FP8 hybrid backward, record abs_max of the final d_qkv for subsequent quantization
    if (mRunState.has_fp8_hybrid_backward()) {
        float* abs_max_ptr = mRunState.simplified_quant_grads().d_qkv.abs_max();
        abs_max(abs_max_ptr, d_qkv_view, static_cast<long>(d_qkv_view.nelem()),
                mRunState.DeviceProp, mRunState.MainStream);
    }

    const bool d_qkv_nan = log_nan(d_qkv_view, "d_qkv");
    if (d_qkv_nan && !d_out_nan) {
        log_nan(qkv_view, "qkv");
        log_nan(q_rstd, "q_rstd");
        log_nan(k_rstd, "k_rstd");

        auto log_sample = [&](const Tensor& t, const char* tag, std::size_t max_elems) {
            if (!t.Data) {
                return;
            }
            const std::size_t row_width = tensor_row_width(t);
            if (row_width == 0) {
                return;
            }
            const std::size_t count = std::min(row_width, max_elems);
            std::vector<float> vals;
            if (!copy_tensor_token_sample_as_f32(t, 0, count, vals) || vals.empty()) {
                return;
            }
            float min_v = vals[0];
            float max_v = vals[0];
            double mean_v = 0.0;
            for (float v : vals) {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                mean_v += static_cast<double>(v);
            }
            mean_v /= static_cast<double>(vals.size());
            std::cerr << "[QK_ROPE_BWD_SAMPLE] layer=" << op_layer_idx
                      << " tag=" << (tag ? tag : "<unnamed>")
                      << " row_width=" << row_width
                      << " sample_count=" << count
                      << " min=" << min_v
                      << " max=" << max_v
                      << " mean=" << mean_v
                      << std::endl;
        };

        // Capture representative magnitudes to see if RMS norms or activations blow up.
        log_sample(d_out_view, "d_qkv_out", 4096);
        log_sample(qkv_view, "qkv", 4096);
        log_sample(q_rstd, "q_rstd", 65536);
        log_sample(k_rstd, "k_rstd", 65536);
    }
}

}  // namespace dsl
