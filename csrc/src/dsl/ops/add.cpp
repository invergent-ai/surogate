#include "dsl/compiled_ops.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_add(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, mRunState.MainStream);
}

void CompiledExecutor::dispatch_add_backward(const CompiledOp& op) {
    // Addition backward: gradients pass through unchanged to both inputs
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    const int trace = env_int("SUROGATE_ADD_BWD_TRACE", 0);
    const int trace_layer = env_int("SUROGATE_ADD_BWD_TRACE_LAYER", -1);
    const int trace_limit = env_int("SUROGATE_ADD_BWD_TRACE_LIMIT", 8);
    const int trace_samples = env_int("SUROGATE_ADD_BWD_TRACE_SAMPLES", 8);
    static std::atomic<int> trace_count{0};

    int layer_idx = op.outputs.empty() ? -1 : op.outputs[0].layer_idx;
    std::string field;
    if (layer_idx < 0 && !op.outputs.empty()) {
        parse_block_param(op.outputs[0].name, layer_idx, field);
    }
    const bool do_trace = trace && !mCapturing &&
        (trace_layer < 0 || trace_layer == layer_idx) &&
        (trace_limit <= 0 || trace_count.fetch_add(1) < trace_limit);

    auto trace_sample = [&](const Tensor& t, const char* tag) {
        if (!t.Data) {
            std::cerr << fmt::format("[ADD_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr=<null>\n",
                                     layer_idx, mMicroStep, tag ? tag : "<unnamed>",
                                     static_cast<int>(t.DType), tensor_shape_str(t));
            return;
        }
        std::vector<float> vals;
        if (!copy_tensor_token_sample_as_f32(t, 0, static_cast<std::size_t>(trace_samples), vals) || vals.empty()) {
            std::cerr << fmt::format("[ADD_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} sample=<unavailable>\n",
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
            "[ADD_BWD_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} min={:.6g} max={:.6g} max_abs={:.6g} mean_abs={:.6g} out0={} out1={}\n",
            layer_idx, mMicroStep, tag ? tag : "<unnamed>", static_cast<int>(t.DType),
            tensor_shape_str(t), static_cast<const void*>(t.Data),
            min_v, max_v, max_abs, mean_abs,
            op.outputs.size() > 0 ? op.outputs[0].name : "<none>",
            op.outputs.size() > 1 ? op.outputs[1].name : "<none>");
    };

    if (do_trace) {
        trace_sample(d_out, "d_out");
    }

    // For pre-allocated gradient slots (like d_res_ffn, d_res_att), we must copy the
    // upstream gradient into the original simplified_grads buffer. Simply aliasing
    // the data pointer causes shared storage between residual and branch gradients,
    // which breaks LoRA (it does in-place dx accumulation).
    // IMPORTANT: We must get the base tensor directly from simplified_grads(), not via
    // resolve_tensor(), because resolve_tensor() may return a view from mTensorMap.
    auto assign_output = [&](const TensorRef& ref) {
        Tensor* base_grad = nullptr;
        if (ref.layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(ref.layer_idx);
            switch (ref.slot) {
                case TensorSlot::BlockDResFFN: base_grad = &grads.d_res_ffn; break;
                case TensorSlot::BlockDResAtt: base_grad = &grads.d_res_att; break;
                case TensorSlot::BlockDLN1: base_grad = &grads.d_ln1; break;
                case TensorSlot::BlockDLN2: base_grad = &grads.d_ln2; break;
                case TensorSlot::BlockDSwiGLU: base_grad = &grads.d_swiglu; break;
                case TensorSlot::BlockDAtt: base_grad = &grads.d_att; break;
                case TensorSlot::BlockDQKV: base_grad = &grads.d_qkv; break;
                case TensorSlot::BlockDMLPUp: base_grad = &grads.d_mlp_up; break;
                case TensorSlot::BlockDMLPDown: base_grad = &grads.d_mlp_down; break;
                default: break;
            }
        }

        if (base_grad) {
            if (base_grad->Data) {
                if (base_grad->DType != d_out.DType) {
                    throw std::runtime_error("dispatch_add_backward: dtype mismatch for " + ref.name);
                }
                if (base_grad->Data != d_out.Data) {
                    CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                               cudaMemcpyDeviceToDevice, mRunState.MainStream));
                }
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                return;
            }
            // For stack-allocated gradient temps, allocate proper storage instead of aliasing.
            // Aliasing to d_out can cause stale memory access when the stack is restored at
            // layer boundaries because the aliased memory gets recycled.
            const bool is_stack_grad = mRunState.large_bwd_temps_on_stack() &&
                (ref.slot == TensorSlot::BlockDQKV ||
                 ref.slot == TensorSlot::BlockDMLPUp ||
                 ref.slot == TensorSlot::BlockDSwiGLU);
            if (is_stack_grad) {
                // Allocate proper stack storage and copy data
                mRunState.temp_acquire(*base_grad);
                mTemps.push_back(*base_grad);
                CUDA_CHECK(cudaMemcpyAsync(base_grad->Data, d_out.Data, d_out.bytes(),
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));
                mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
                return;
            }
            // Fall back to aliasing if the base grad has no storage yet (non-stack temps).
            base_grad->Data = d_out.Data;
            mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
            return;
        }
        // Default: just expose d_out as-is.
        mTensorMap[ref.name] = d_out;
    };

    assign_output(op.outputs[0]);
    if (op.outputs.size() > 1) {
        assign_output(op.outputs[1]);
    }
}


}  // namespace dsl
