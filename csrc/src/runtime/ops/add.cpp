#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_add(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    // Backward accumulation outputs (e.g. d_*_accum_N) should not reuse aliased
    // buffers: they are reduction nodes and can become incorrect if out aliases an
    // input or if stale mapped storage is reused across accumulation steps.
    const bool is_accum_output = !op.outputs.empty() && op.outputs[0].name.find("_accum_") != std::string::npos;
    const bool aliases_input = out.Data && (out.Data == a.Data || out.Data == b.Data);

    // For element-wise add, output shape must match inputs. Reallocate when shape
    // is missing/wrong, when aliasing would make add in-place unsafe, or for
    // autodiff accumulation outputs to guarantee isolated storage.
    if ((out.nelem() != a.nelem() && a.Rank > 0) || aliases_input || is_accum_output) {
        std::vector<long> shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
        out = mRunState.temp_alloc(a.DType, shape);
        fill_zero(out, mRunState.MainStream);
        mTemps.push_back(out);
    }

    vector_add_sr(out, a, b, 1.0f, static_cast<long>(a.nelem()), 0, mRunState.MainStream);
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_add_backward(const CompiledOp& op) {
    // Addition backward: gradients pass through unchanged to both inputs
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // For pre-allocated gradient slots (like d_res_ffn, d_res_att), we must copy the
    // upstream gradient into the original simplified_grads buffer. Simply aliasing
    // the data pointer causes shared storage between residual and branch gradients,
    // which breaks LoRA (it does in-place dx accumulation).
    // IMPORTANT: We must get the base tensor directly from simplified_grads(), not via
    // resolve_tensor(), because resolve_tensor() may return a cached view from mTensors.
    auto assign_output = [&](const TensorRef& ref) {
        if (!ref.name.empty()) {
            if (auto base = base_param_from_grad(ref.name)) {
                bool accumulate = mAccumulateTensors.count(ref.name) > 0;
                if (!accumulate) {
                    accumulate = mAccumulateTensors.count("d_" + *base) > 0;
                }
                bool grad_accum = false;
                if (Tensor* grad = mGrads.get_param_grad(*base, grad_accum)) {
                    if (grad->Data) {
                        Tensor target = ref.shape.empty() ? *grad : view_tensor(*grad, ref.shape);
                        if (target.DType != d_out.DType) {
                            throw std::runtime_error("dispatch_add_backward: dtype mismatch for " + ref.name);
                        }
                        if (target.nelem() != d_out.nelem()) {
                            throw std::runtime_error("dispatch_add_backward: shape mismatch for " + ref.name);
                        }
                        if (target.Data != d_out.Data) {
                            if (accumulate) {
                                vector_add_sr(target,
                                              target,
                                              d_out,
                                              1.0f,
                                              static_cast<long>(target.nelem()),
                                              0,
                                              mRunState.MainStream);
                            } else {
                                CUDA_CHECK(cudaMemcpyAsync(target.Data,
                                                           d_out.Data,
                                                           target.bytes(),
                                                           cudaMemcpyDeviceToDevice,
                                                           mRunState.MainStream));
                            }
                        }
                        store_tensor(ref, target);
                        return;
                    }
                }
            }
        }

        Tensor* base_grad = nullptr;
        if (ref.layer_idx >= 0) {
            auto& grads = mRunState.simplified_grads(ref.layer_idx);
            switch (ref.slot) {
                case TensorSlot::BlockDResFFN: base_grad = &grads.d_res_ffn; break;
                case TensorSlot::BlockDResAtt: base_grad = &grads.d_res_att; break;
                case TensorSlot::BlockDAttOut: base_grad = &grads.d_att_out; break;
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
                    CUDA_CHECK(cudaMemcpyAsync(base_grad->Data,
                                               d_out.Data,
                                               d_out.bytes(),
                                               cudaMemcpyDeviceToDevice,
                                               mRunState.MainStream));
                }
                store_tensor(ref, view_tensor(*base_grad, ref.shape));
                return;
            }
            // For stack-allocated gradient temps, allocate proper storage instead of aliasing.
            // Aliasing to d_out can cause stale memory access when the stack is restored at
            // layer boundaries because the aliased memory gets recycled.
            const bool is_stack_grad = mRunState.large_bwd_temps_on_stack() &&
                                       (ref.slot == TensorSlot::BlockDQKV || ref.slot == TensorSlot::BlockDMLPUp ||
                                        ref.slot == TensorSlot::BlockDSwiGLU);
            if (is_stack_grad) {
                // Allocate proper stack storage and copy data
                mRunState.temp_acquire(*base_grad);
                mTemps.push_back(*base_grad);
                CUDA_CHECK(cudaMemcpyAsync(base_grad->Data,
                                           d_out.Data,
                                           d_out.bytes(),
                                           cudaMemcpyDeviceToDevice,
                                           mRunState.MainStream));
                store_tensor(ref, view_tensor(*base_grad, ref.shape));
                return;
            }
            // Fall back to aliasing if the base grad has no storage yet (non-stack temps).
            base_grad->Data = d_out.Data;
            store_tensor(ref, view_tensor(*base_grad, ref.shape));
            return;
        }
        // Default: just expose d_out as-is.
        store_tensor(ref, d_out);
    };

    assign_output(op.outputs[0]);
    if (op.outputs.size() > 1) {
        assign_output(op.outputs[1]);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Add backward rule
// Forward: C = A + B
// Backward: dA = dC, dB = dC (with broadcast reduction if needed)
// -----------------------------------------------------------------------------
std::vector<Operation> add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    // Gradient passes through unchanged (identity for addition)
    // Note: if shapes differ due to broadcasting, would need reduce_sum
    // For now, assume same shapes. Emit a single add_backward op so compiled
    // executor can copy into both base gradients in one place.
    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    if (ctx.needs_grad(0) || ctx.needs_grad(1)) {
        ops.push_back(make_operation("add_backward_" + std::to_string(ctx.op_counter++),
                                     "add_backward",
                                     "add_backward",
                                     {ctx.d_output},
                                     outputs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("add", ::dsl::add_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Add (elementwise)
// ------------------------------------------------------------------------
const int _add_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "add";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.size() < 2 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "add requires 2 inputs and 1 output";
            return std::make_optional(err);
        }
        return validators::check_broadcastable(inputs[0], inputs[1], "add");
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// AddBackward
// ------------------------------------------------------------------------
const int _add_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "add_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& d_in1 = outputs[0];
        const auto& d_in2 = outputs[1];

        // Both outputs should match input gradient shape
        if (auto err = validators::check_same_numel(d_in1, d_out, "d_in1", "d_out", "add_backward")) {
            return err;
        }
        if (auto err = validators::check_same_numel(d_in2, d_out, "d_in2", "d_out", "add_backward")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
