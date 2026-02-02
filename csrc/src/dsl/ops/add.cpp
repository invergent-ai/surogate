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

void CompiledExecutor::dispatch_add(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, mRunState.MainStream);
}

void CompiledExecutor::dispatch_add_backward(const CompiledOp& op) {
    // Addition backward: gradients pass through unchanged to both inputs
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // For pre-allocated gradient slots (like d_res_ffn, d_res_att), we must copy the
    // upstream gradient into the original simplified_grads buffer. Simply aliasing
    // the data pointer causes shared storage between residual and branch gradients,
    // which breaks LoRA (it does in-place dx accumulation).
    // IMPORTANT: We must get the base tensor directly from simplified_grads(), not via
    // resolve_tensor(), because resolve_tensor() may return a view from mTensorMap.
    auto assign_output = [&](const TensorRef& ref) {
        auto trace_residual_l8 = [&](const Tensor& out_tensor, const char* tag) {
            static int add_res_trace = 0;
            if (add_res_trace < 20 &&
                (ref.name.find("d_blocks[7].mlp_down") != std::string::npos ||
                 ref.name.find("d_blocks[8].res_att") != std::string::npos ||
                 ref.name.find("d_blocks[8].res_ffn") != std::string::npos)) {
                fprintf(stderr,
                        "[ADD_BWD_L8] %s out=%s ptr=%p in=%s slot=%d ptr=%p\n",
                        tag,
                        ref.name.c_str(),
                        out_tensor.Data,
                        op.inputs[0].name.c_str(),
                        static_cast<int>(op.inputs[0].slot),
                        d_out.Data);
                log_tensor_mag_unbounded("ADD_BWD_L8_DOUT",
                                         ref.layer_idx,
                                         op.inputs[0].name,
                                         d_out,
                                         4096);
                log_tensor_mag_unbounded("ADD_BWD_L8_OUT",
                                         ref.layer_idx,
                                         ref.name,
                                         out_tensor,
                                         4096);
                add_res_trace++;
            }
        };
        auto trace_mlp_down = [&](const Tensor& out_tensor, const char* tag) {
            static int add_mlp_down_trace = 0;
            if (add_mlp_down_trace < 20 &&
                ref.name.find("d_blocks[24].mlp_down") != std::string::npos) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> in_vals(4, 0.0f), out_vals(4, 0.0f);
                const bool ok_in = copy_tensor_sample_as_f32(d_out, in_vals.size(), in_vals);
                const bool ok_out = copy_tensor_sample_as_f32(out_tensor, out_vals.size(), out_vals);
                fprintf(stderr,
                        "[ADD_BWD_MLP_DOWN] %s out=%s ptr=%p in=%s slot=%d ptr=%p ok_in=%d ok_out=%d "
                        "in_vals=%.6f,%.6f,%.6f,%.6f out_vals=%.6f,%.6f,%.6f,%.6f\n",
                        tag,
                        ref.name.c_str(),
                        out_tensor.Data,
                        op.inputs[0].name.c_str(),
                        static_cast<int>(op.inputs[0].slot),
                        d_out.Data,
                        ok_in ? 1 : 0,
                        ok_out ? 1 : 0,
                        in_vals[0], in_vals[1], in_vals[2], in_vals[3],
                        out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
                add_mlp_down_trace++;
            }
        };
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
                trace_residual_l8(mTensorMap[ref.name], "copy");
                trace_mlp_down(mTensorMap[ref.name], "copy");
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
                trace_residual_l8(mTensorMap[ref.name], "stack_copy");
                trace_mlp_down(mTensorMap[ref.name], "stack_copy");
                return;
            }
            // Fall back to aliasing if the base grad has no storage yet (non-stack temps).
            base_grad->Data = d_out.Data;
            mTensorMap[ref.name] = view_tensor(*base_grad, ref.shape);
            trace_residual_l8(mTensorMap[ref.name], "alias");
            trace_mlp_down(mTensorMap[ref.name], "alias");
            return;
        }
        // Default: just expose d_out as-is.
        mTensorMap[ref.name] = d_out;
        trace_residual_l8(mTensorMap[ref.name], "passthrough");
        trace_mlp_down(mTensorMap[ref.name], "passthrough");
    };

    assign_output(op.outputs[0]);
    if (op.outputs.size() > 1) {
        assign_output(op.outputs[1]);
    }
}


}  // namespace dsl
