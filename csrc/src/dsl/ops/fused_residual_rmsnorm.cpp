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

void CompiledExecutor::dispatch_fused_residual_rmsnorm(const CompiledOp& op) {
    Tensor& residual_in = resolve_tensor(op.inputs[0]);
    Tensor& input = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    Tensor& residual_out = ensure_output_tensor(op.outputs[0]);
    Tensor& y = ensure_output_tensor(op.outputs[1]);
    Tensor& rstd = ensure_output_tensor(op.outputs[2]);

    // Validate dtypes before calling kernel
    if (rstd.DType != ETensorDType::FP32) {
        std::ostringstream oss;
        oss << "fused_residual_rmsnorm: rstd dtype mismatch. Expected FP32, got "
            << dtype_to_str(rstd.DType) << ". Output tensor: " << op.outputs[2].name
            << " (slot=" << static_cast<int>(op.outputs[2].slot) << ")";
        throw std::runtime_error(oss.str());
    }

    fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                   op.attrs.eps, static_cast<int>(mB * mT),
                                   mConfig.HiddenSize, mRunState.MainStream);

    const int fwd_nan_trace = env_int("SUROGATE_RMS_FWD_NAN_TRACE", 0);
    const int fwd_nan_layer = env_int("SUROGATE_RMS_FWD_NAN_LAYER", -1);
    int fwd_layer_idx = -1;
    std::string fwd_field;
    if (!op.outputs.empty() && !op.outputs[1].name.empty()) {
        parse_block_param(op.outputs[1].name, fwd_layer_idx, fwd_field);
    }
    if (fwd_nan_trace && !mCapturing && (fwd_nan_layer < 0 || fwd_nan_layer == fwd_layer_idx)) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        auto log_nan = [&](const Tensor& t, const char* tag) -> bool {
            if (!t.Data) {
                return false;
            }
            long row = -1;
            float min_val = 0.0f;
            float max_val = 0.0f;
            if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
                return false;
            }
            const char* field_name = fwd_field.empty() ? "<none>" : fwd_field.c_str();
            std::cerr << fmt::format("[RMS_FWD_NAN] layer={} field={} tag={} row={} min={} max={} dtype={}\n",
                                     fwd_layer_idx, field_name, tag ? tag : "<unnamed>",
                                     row, min_val, max_val, static_cast<int>(t.DType));
            return true;
        };
        const bool y_nan = log_nan(y, "y");
        const bool res_nan = log_nan(residual_out, "residual_out");
        const bool rstd_nan = log_nan(rstd, "rstd");
        if (y_nan || res_nan || rstd_nan) {
            log_nan(residual_in, "residual_in");
            log_nan(input, "input");
            log_nan(weight, "weight");
        }
    }

    // FIX: For LN2 output (res_att), copy to simplified_acts.residual_att when the
    // graph compiler assigned the wrong slot. This happens for the last layer where
    // the output is named "StackedBlocks_N" instead of "blocks[N].res_att".
    if (op.outputs[1].name.find("ln2") != std::string::npos) {
        int layer_idx = -1;
        std::string field;
        parse_block_param(op.outputs[1].name, layer_idx, field);
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            // If the output wasn't written to acts.residual_att, copy it there
            if (residual_out.Data != acts.residual_att.Data && acts.residual_att.Data) {
                CUDA_CHECK(cudaMemcpyAsync(acts.residual_att.Data, residual_out.Data,
                                           residual_out.bytes(), cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
            }
        }
    }
}

void CompiledExecutor::dispatch_fused_residual_rmsnorm_backward(const CompiledOp& op) {
    // inputs: d_y, d_residual_next (may be empty), residual_out, weight, rstd
    // outputs: d_residual, d_input, d_weight (optional)

    Tensor& d_y = resolve_tensor(op.inputs[0]);

    const bool is_final_norm =
        (op.inputs[3].name.find("final_norm") != std::string::npos ||
         op.inputs[3].name.find("ln_final") != std::string::npos ||
         op.inputs[3].name.find("ln_f") != std::string::npos);

    Tensor* residual_out_ptr = &resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);

    int ln_layer_idx = -1;
    std::string ln_field;
    if (!op.inputs[3].name.empty()) {
        parse_block_param(op.inputs[3].name, ln_layer_idx, ln_field);
    }
    if (ln_layer_idx >= 0 && ln_field == "ln1_weight") {
        // LN1 backward expects residual_out from the forward fused residual op.
        // In the DSL graph, residual_out is res_ffn for the SAME layer index.
        // Ensure the correct per-layer residual buffer is used (especially with offloading).
        if (mRunState.has_residual_offloading()) {
            mRunState.fetch_residual(ln_layer_idx, mRunState.side_stream());
        }
        residual_out_ptr = &mRunState.get_residual(ln_layer_idx, mRunState.MainStream);
    }
    // FIX: LN2 backward needs the saved/recomputed residual_att from simplified_acts.
    // The backward graph may have wrong tensor names for the last layer (e.g., "StackedBlocks_N"
    // instead of "blocks[N].res_att"), causing it to resolve to stale/wrong data.
    // Always use the simplified_acts residual_att which is either saved (no recompute) or
    // recomputed (with recompute) to ensure correct gradient computation.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        auto& acts = mRunState.simplified_acts(ln_layer_idx);
        residual_out_ptr = &acts.residual_att;
    }
    Tensor& residual_out = *residual_out_ptr;

    const int nan_trace = env_int("SUROGATE_RMS_BWD_NAN_TRACE", 0);
    const int nan_layer = env_int("SUROGATE_RMS_BWD_NAN_LAYER", -1);
    const int trace = env_int("SUROGATE_RMS_BWD_TRACE", 0);
    const int trace_layer = env_int("SUROGATE_RMS_BWD_TRACE_LAYER", -1);
    const int trace_limit = env_int("SUROGATE_RMS_BWD_TRACE_LIMIT", 8);
    const int trace_samples = env_int("SUROGATE_RMS_BWD_TRACE_SAMPLES", 8);
    static std::atomic<int> trace_count{0};
    const bool trace_layer_ok = (nan_layer < 0) || (nan_layer == ln_layer_idx);
    auto log_nan = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            return;
        }
        long row = -1;
        float min_val = 0.0f;
        float max_val = 0.0f;
        if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
            return;
        }
        const char* field_name = ln_field.empty() ? "<none>" : ln_field.c_str();
        std::cerr << fmt::format("[RMS_BWD_NAN] layer={} field={} tag={} row={} min={} max={} dtype={}\n",
                                 ln_layer_idx, field_name, tag ? tag : "<unnamed>",
                                 row, min_val, max_val, static_cast<int>(t.DType));
    };
    if (nan_trace && !mCapturing && trace_layer_ok) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        log_nan(d_y, "d_y");
        log_nan(residual_out, "residual_out");
        log_nan(weight, "weight");
        log_nan(rstd, "rstd");
    }

    // d_residual_next is the incoming gradient from the next layer (may be zero/empty)
    Tensor d_residual_zero{};
    Tensor* d_residual_next = nullptr;
    if (!op.inputs[1].name.empty()) {
        d_residual_next = &resolve_tensor(op.inputs[1]);
    } else {
        // Allocate and zero a temporary for d_residual if none provided
        d_residual_zero = mRunState.temp_alloc(d_y.DType, {mB, mT, static_cast<long>(mConfig.HiddenSize)});
        fill_zero(d_residual_zero, mRunState.MainStream);
        mTemps.push_back(d_residual_zero);
        d_residual_next = &d_residual_zero;
    }
    Tensor* d_residual_input = d_residual_next;
    Tensor* d_residual_stream = d_residual_next;

    const bool trace_layer_ok_stats = (trace_layer < 0) || (trace_layer == ln_layer_idx);
    const bool do_trace = trace && !mCapturing && trace_layer_ok_stats &&
        (trace_limit <= 0 || trace_count.fetch_add(1) < trace_limit);

    auto trace_sample = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            std::cerr << fmt::format("[RMS_BWD_TRACE] layer={} field={} tag={} dtype={} shape={} ptr=<null>\n",
                                     ln_layer_idx, ln_field.empty() ? "<none>" : ln_field.c_str(),
                                     tag ? tag : "<unnamed>", static_cast<int>(t.DType),
                                     tensor_shape_str(t));
            return;
        }
        std::vector<float> vals;
        if (!copy_tensor_token_sample_as_f32(t, 0, static_cast<std::size_t>(trace_samples), vals) || vals.empty()) {
            std::cerr << fmt::format("[RMS_BWD_TRACE] layer={} field={} tag={} dtype={} shape={} ptr={} sample=<unavailable>\n",
                                     ln_layer_idx, ln_field.empty() ? "<none>" : ln_field.c_str(),
                                     tag ? tag : "<unnamed>", static_cast<int>(t.DType),
                                     tensor_shape_str(t), static_cast<const void*>(t.Data));
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
            "[RMS_BWD_TRACE] layer={} field={} tag={} dtype={} shape={} ptr={} min={:.6g} max={:.6g} max_abs={:.6g} mean_abs={:.6g}\n",
            ln_layer_idx, ln_field.empty() ? "<none>" : ln_field.c_str(),
            tag ? tag : "<unnamed>", static_cast<int>(t.DType),
            tensor_shape_str(t), static_cast<const void*>(t.Data),
            min_v, max_v, max_abs, mean_abs);
    };

    if (do_trace) {
        trace_sample(d_y, "d_y");
        if (d_residual_input) {
            trace_sample(*d_residual_input, "d_residual_next");
        }
        trace_sample(residual_out, "residual_out");
    }

    Tensor& d_input = ensure_output_tensor(op.outputs[1]);

    // d_weight may be nullptr if weight is frozen
    Tensor dummy_weight{};
    Tensor* d_weight_ptr = nullptr;
    bool skip_weight_grad = true;
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[2]);
        skip_weight_grad = false;
        if (op.outputs[2].slot == TensorSlot::Mapped || op.outputs[2].slot == TensorSlot::Temporary) {
            fill_zero(*d_weight_ptr, mRunState.MainStream);
        }
    } else {
        dummy_weight = mRunState.temp_alloc(weight.DType, {static_cast<long>(mConfig.HiddenSize)});
        mTemps.push_back(dummy_weight);
        d_weight_ptr = &dummy_weight;
    }

    const int C = mConfig.HiddenSize;

    // Determine abs_max pointer for FP8 gradient quantization.
    // LN1 backward produces d_res_ffn (gradient for previous layer's residual).
    // LN2 backward produces d_res_att (gradient for attention path).
    float* abs_max_ptr = nullptr;
    if (mRunState.has_grad_quants()) {
        const bool is_ln2 = (ln_field == "ln2_weight");
        abs_max_ptr = is_ln2
            ? mRunState.simplified_quant_grads().d_res_att.abs_max()
            : mRunState.simplified_quant_grads().d_res_ffn.abs_max();
    }


    rmsnorm_backward(d_input, *d_weight_ptr, mRunState.scratch().rmsnorm_scratch,
                     *d_residual_input, d_y, residual_out, weight, rstd,
                     abs_max_ptr,
                     static_cast<int>(mB), static_cast<int>(mT), C,
                     mRunState.DeviceProp, mRunState.MainStream, skip_weight_grad);

    if (nan_trace && !mCapturing && trace_layer_ok) {
        log_nan(d_input, "d_input");
        if (d_residual_input) {
            log_nan(*d_residual_input, "d_residual_in");
        }
    }


    // Copy d_input to d_residual if they're different outputs
    if (!op.outputs[0].name.empty() && op.outputs[0].name != op.outputs[1].name) {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        CUDA_CHECK(cudaMemcpyAsync(d_residual.Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // Update residual_out gradient buffer to include norm contribution.
    if (d_residual_stream && d_residual_stream->Data && d_residual_stream->Data != d_input.Data) {
        CUDA_CHECK(cudaMemcpyAsync(d_residual_stream->Data, d_input.Data, d_input.bytes(),
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    // LN2 backward produces d_res_ffn (gradient for MLP down output). Mirror it into d_mlp_down
    // so downstream matmul backward sees a valid d_out.
    if (ln_layer_idx >= 0 && ln_field == "ln2_weight") {
        Tensor& d_residual = ensure_output_tensor(op.outputs[0]);
        Tensor& d_mlp_down = mRunState.simplified_grads(ln_layer_idx).d_mlp_down;

        if (d_mlp_down.Data && d_mlp_down.Data != d_residual.Data) {
            CUDA_CHECK(cudaMemcpyAsync(d_mlp_down.Data, d_residual.Data, d_residual.bytes(),
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
        }
    }

    // LN1 backward writes grad for previous layer's residual stream into d_mlp_down;
    // mirror it into that layer's d_res_ffn so gradient propagation matches modular.
    if (op.outputs.size() > 1 && op.outputs[1].slot == TensorSlot::BlockDMLPDown) {
        const int prev_layer = op.outputs[1].layer_idx;
        if (prev_layer >= 0) {
            Tensor& d_res_ffn_prev = mRunState.simplified_grads(prev_layer).d_res_ffn;
            if (d_res_ffn_prev.Data && d_res_ffn_prev.Data != d_input.Data) {
                CUDA_CHECK(cudaMemcpyAsync(d_res_ffn_prev.Data, d_input.Data, d_input.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
            }
        }
    }
}


}  // namespace dsl
