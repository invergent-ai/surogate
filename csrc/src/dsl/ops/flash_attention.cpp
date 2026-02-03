#include "dsl/compiled_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Mirror modular path: always use cuDNN attention.
    attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                            mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                            Hq, Hkv, Hs, mRunState.MainStream);

    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);
    }
    const int kernel_trace = env_int("SUROGATE_ATTN_FWD_KERNEL_TRACE", 0);
    const int kernel_layer = env_int("SUROGATE_ATTN_FWD_KERNEL_LAYER", -1);
    const int kernel_b = env_int("SUROGATE_ATTN_FWD_KERNEL_B", 0);
    const int kernel_h = env_int("SUROGATE_ATTN_FWD_KERNEL_H", 0);
    const int kernel_t = env_int("SUROGATE_ATTN_FWD_KERNEL_T", 0);
    const int kernel_l = env_int("SUROGATE_ATTN_FWD_KERNEL_L", 0);
    const bool do_kernel_trace = kernel_trace && !mCapturing &&
        (kernel_layer < 0 || kernel_layer == layer_idx);
    if (do_kernel_trace) {
        AttnFwdDebugConfig cfg;
        cfg.enabled = 1;
        cfg.layer = layer_idx;
        cfg.target_b = kernel_b;
        cfg.target_h = kernel_h;
        cfg.target_t = kernel_t;
        cfg.target_l = kernel_l;
        attention_forward_debug(qkv, lse,
                                static_cast<int>(mB), static_cast<int>(mT),
                                Hq, Hkv, Hs, cfg, mRunState.MainStream);
    }
    const int nan_trace = env_int("SUROGATE_ATTN_FWD_NAN_TRACE", 0);
    const int nan_layer = env_int("SUROGATE_ATTN_FWD_NAN_LAYER", -1);
    if (nan_trace && !mCapturing && (nan_layer < 0 || nan_layer == layer_idx)) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        auto trace_nan = [&](const Tensor& t, const char* tag) -> bool {
            if (!t.Data) {
                return false;
            }
            long row = -1;
            float min_val = 0.0f;
            float max_val = 0.0f;
            if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
                return false;
            }
            std::cerr << fmt::format("[ATTN_FWD_NAN] layer={} tag={} row={} min={} max={} dtype={}\n",
                                     layer_idx,
                                     tag ? tag : "<unnamed>",
                                     row,
                                     min_val,
                                     max_val,
                                     static_cast<int>(t.DType));
            return true;
        };
        const bool out_nan = trace_nan(out, "out");
        if (out_nan) {
            trace_nan(lse, "lse");
            trace_nan(qkv, "qkv");
        }
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && op.inputs.size() > 3) {
        std::string field;
        parse_block_param(op.inputs[3].name, layer_idx, field);
    }

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Zero-initialize d_qkv to avoid stale values.
    fill_zero(d_qkv, mRunState.MainStream);

    const int attn_chunks = mOptions.AttBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const int chunk_B = (attn_chunks == 1)
        ? static_cast<int>(mB)
        : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

    const int kernel_trace = env_int("SUROGATE_ATTN_BWD_KERNEL_TRACE", 0);
    const int kernel_layer = env_int("SUROGATE_ATTN_BWD_KERNEL_LAYER", -1);
    const int kernel_b = env_int("SUROGATE_ATTN_BWD_KERNEL_B", 0);
    const int kernel_h = env_int("SUROGATE_ATTN_BWD_KERNEL_H", 0);
    const int kernel_t = env_int("SUROGATE_ATTN_BWD_KERNEL_T", 0);
    const int kernel_l = env_int("SUROGATE_ATTN_BWD_KERNEL_L", 0);
    const bool do_kernel_trace = kernel_trace && !mCapturing &&
        (kernel_layer < 0 || kernel_layer == layer_idx);
    if (do_kernel_trace) {
        AttnBwdDebugConfig cfg;
        cfg.enabled = 1;
        cfg.layer = layer_idx;
        cfg.micro = mMicroStep;
        cfg.target_b = kernel_b;
        cfg.target_h = kernel_h;
        cfg.target_t = kernel_t;
        cfg.target_l = kernel_l;
        attention_backward_debug(out, d_out, qkv, lse,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, cfg, mRunState.MainStream);
    }

    auto trace_nan = [&](const Tensor& t, const char* tag) -> bool {
        if (!t.Data) {
            return false;
        }
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
            return false;
        }
        std::cerr << fmt::format("[ATTN_BWD_NAN] layer={} tag={} row={} min={} max={} dtype={}\n",
                                 layer_idx,
                                 tag ? tag : "<unnamed>",
                                 row,
                                 min_val,
                                 max_val,
                                 static_cast<int>(t.DType));
        return true;
    };
    const int nan_trace = env_int("SUROGATE_ATTN_BWD_NAN_TRACE", 0);
    const int nan_layer = env_int("SUROGATE_ATTN_BWD_NAN_LAYER", -1);
    const int trace = env_int("SUROGATE_ATTN_BWD_TRACE", 0);
    const int trace_layer = env_int("SUROGATE_ATTN_BWD_TRACE_LAYER", -1);
    const int trace_limit = env_int("SUROGATE_ATTN_BWD_TRACE_LIMIT", 8);
    const int trace_samples = env_int("SUROGATE_ATTN_BWD_TRACE_SAMPLES", 8);
    static std::atomic<int> trace_count{0};

    auto trace_sample = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            std::cerr << fmt::format("[ATTN_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr=<null>\n",
                                     layer_idx, mMicroStep, tag ? tag : "<unnamed>",
                                     static_cast<int>(t.DType), tensor_shape_str(t));
            return;
        }
        std::vector<float> vals;
        if (!copy_tensor_token_sample_as_f32(t, 0, static_cast<std::size_t>(trace_samples), vals) || vals.empty()) {
            std::cerr << fmt::format("[ATTN_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} sample=<unavailable>\n",
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
            "[ATTN_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} min={:.6g} max={:.6g} max_abs={:.6g} mean_abs={:.6g}\n",
            layer_idx, mMicroStep, tag ? tag : "<unnamed>", static_cast<int>(t.DType),
            tensor_shape_str(t), static_cast<const void*>(t.Data),
            min_v, max_v, max_abs, mean_abs);
    };

    const bool do_trace = trace && !mCapturing &&
        (trace_layer < 0 || trace_layer == layer_idx) &&
        (trace_limit <= 0 || trace_count.fetch_add(1) < trace_limit);

    if (attn_chunks == 1) {
        if (do_trace) {
            trace_sample(d_out, "d_out_pre");
            trace_sample(out, "out_pre");
            trace_sample(lse, "lse_pre");
            trace_sample(qkv, "qkv_pre");
        }
        attention_backward_cudnn(d_qkv, lse, out, d_out, qkv,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
        if (do_trace) {
            trace_sample(d_qkv, "d_qkv_post");
        }
        if (nan_trace && !mCapturing && (nan_layer < 0 || nan_layer == layer_idx)) {
            CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
            const bool d_qkv_nan = trace_nan(d_qkv, "d_qkv");
            if (d_qkv_nan) {
                trace_nan(d_out, "d_out");
                trace_nan(out, "out");
                trace_nan(lse, "lse");
                trace_nan(qkv, "qkv");
            }
        }
        return;
    }

    for (int chunk = 0; chunk < attn_chunks; ++chunk) {
        const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
        const long end = start + static_cast<long>(chunk_B);
        Tensor d_out_chunk = slice(d_out, 0, start, end);
        Tensor out_chunk = slice(out, 0, start, end);
        Tensor lse_chunk = slice(lse, 0, start, end);
        Tensor qkv_chunk = slice(qkv, 0, start, end);
        Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

        if (do_trace && chunk == 0) {
            trace_sample(d_out_chunk, "d_out_pre");
            trace_sample(out_chunk, "out_pre");
            trace_sample(lse_chunk, "lse_pre");
            trace_sample(qkv_chunk, "qkv_pre");
        }
        attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 static_cast<int>(chunk_B), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
        if (do_trace && chunk == 0) {
            trace_sample(d_qkv_chunk, "d_qkv_post");
        }
    }

    if (nan_trace && !mCapturing && (nan_layer < 0 || nan_layer == layer_idx)) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        const bool d_qkv_nan = trace_nan(d_qkv, "d_qkv");
        if (d_qkv_nan) {
            trace_nan(d_out, "d_out");
            trace_nan(out, "out");
            trace_nan(lse, "lse");
            trace_nan(qkv, "qkv");
        }
    }
}


}  // namespace dsl
