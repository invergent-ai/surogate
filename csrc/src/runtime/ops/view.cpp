#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_view(const CompiledOp& op) {
    Tensor& src = resolve_tensor(op.inputs[0]);
    Tensor view = view_tensor(src, op.attrs.shape);
    store_tensor(op.outputs[0], view);
}

void CompiledExecutor::dispatch_view_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    std::vector<long> shape = op.attrs.shape;

    // If shape is empty, try to resolve from shape_like reference
    if (shape.empty() && !op.attrs.shape_like.empty()) {
        std::string ref_name = op.attrs.shape_like;

        // Strip "saved." prefix if present
        const std::string saved_prefix = "saved.";
        if (ref_name.rfind(saved_prefix, 0) == 0) {
            ref_name = ref_name.substr(saved_prefix.length());
        }

        // Try to find the reference tensor
        Tensor* ref = nullptr;

        // Check saved tensors first
        if (mSaved) {
            auto it = mSaved->find(ref_name);
            if (it != mSaved->end()) {
                ref = &it->second;
            }
        }

        // Check flat tensor vector via pre-resolved shape_like_tensor_id
        if (!ref && op.attrs.shape_like_tensor_id >= 0 &&
            static_cast<std::size_t>(op.attrs.shape_like_tensor_id) < mTensors.size() &&
            mTensors[op.attrs.shape_like_tensor_id].Data) {
            ref = &mTensors[op.attrs.shape_like_tensor_id];
        }

        // Fall back to name-based lookup in flat vector
        if (!ref && mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(ref_name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                ref = &mTensors[tid];
            }
        }

        // If reference found and valid, use its shape
        if (ref && ref->Rank > 0) {
            shape.assign(ref->Sizes.begin(), ref->Sizes.begin() + ref->Rank);
        } else {
            // Fallback: infer shape based on output tensor name and input shape
            // View backward typically does one of:
            // 1. Flatten: [B,T,C] -> [B*T,C] (output name contains "_flat")
            // 2. Unflatten: [B*T,C] -> [B,T,C] (output name does not contain "_flat")
            //
            // Check output name for "_flat" suffix to determine direction
            const std::string& out_name = op.outputs[0].name;
            bool wants_flat = out_name.find("_flat") != std::string::npos;

            if (wants_flat) {
                // Flatten to rank 2: [B,T,C] -> [B*T,C] or [B*T,C] -> [B*T,C]
                if (d_out.Rank >= 3) {
                    long flat_dim = 1;
                    for (int i = 0; i < d_out.Rank - 1; ++i) {
                        flat_dim *= d_out.Sizes[i];
                    }
                    shape = {flat_dim, d_out.Sizes[d_out.Rank - 1]};
                } else if (d_out.Rank == 2) {
                    // Already flat, keep shape
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            } else {
                // Unflatten or keep shape
                if (d_out.Rank >= 3) {
                    // Already unflat, keep shape
                    shape.assign(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
                } else if (d_out.Rank == 2 && d_out.Sizes[0] == mB * mT) {
                    // Unflatten: [B*T,C] -> [B,T,C]
                    shape = {mB, mT, d_out.Sizes[1]};
                } else if (d_out.Rank == 2) {
                    // Keep as rank 2
                    shape = {d_out.Sizes[0], d_out.Sizes[1]};
                }
            }
        }
    }

    if (shape.empty()) {
        auto shape_str = [](const Tensor& t) {
            std::string s = "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i > 0) s += ", ";
                s += std::to_string(t.Sizes[i]);
            }
            s += "]";
            return s;
        };
        throw std::runtime_error("CompiledExecutor view_backward: cannot resolve shape for op " + op.op_id +
                                 " input=" + op.inputs[0].name + " shape=" + shape_str(d_out) +
                                 " output=" + op.outputs[0].name + " shape_like=" + op.attrs.shape_like);
    }
    // Sanity check: view_tensor silently reshapes without verifying nelem.
    // If an upstream op produces d_out with the wrong size (e.g. a backward
    // slot pre-sized with mismatched per-layer dims in hybrid models), the
    // view would claim more elements than its allocation holds and downstream
    // ops (split, memcpy) would read past the buffer end. Fail here instead
    // of corrupting memory downstream.
    {
        std::size_t shape_nelem = 1;
        for (long d : shape)
            shape_nelem *= static_cast<std::size_t>(d);
        if (shape_nelem != d_out.nelem()) {
            std::string sh = "[";
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i) sh += ",";
                sh += std::to_string(shape[i]);
            }
            sh += "]";
            std::string ins = "[";
            for (int i = 0; i < d_out.Rank; ++i) {
                if (i) ins += ",";
                ins += std::to_string(d_out.Sizes[i]);
            }
            ins += "]";
            throw std::runtime_error(
                "view_backward: shape nelem mismatch (upstream grad slot sized wrong?)"
                " op=" +
                op.op_id + " input=" + op.inputs[0].name + " input_tid=" + std::to_string(op.inputs[0].tensor_id) +
                " in_shape=" + ins + " in_nelem=" + std::to_string(d_out.nelem()) +
                " in_Data=" + std::to_string(reinterpret_cast<uintptr_t>(d_out.Data)) +
                " output=" + op.outputs[0].name + " output_tid=" + std::to_string(op.outputs[0].tensor_id) +
                " target_shape=" + sh + " target_nelem=" + std::to_string(shape_nelem));
        }
    }

    Tensor view = view_tensor(d_out, shape);
    store_tensor(op.outputs[0], view);
}

namespace {

// -----------------------------------------------------------------------------
// View/Reshape backward rule (no-op, just reshape gradient)
// Forward: y = view(x, shape)
// Backward: dx = view(dy, original_shape)
// -----------------------------------------------------------------------------
std::vector<Operation> view_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;

        // Need to reshape gradient back to input shape.
        // If the forward input is a parameter or graph input, we can reference it without saving.
        // Otherwise, we need to save it for its shape (or use shape_like).
        AttrMap attrs;
        const std::string& fwd_input = fwd.inputs[0];

        // Check if the forward input is a parameter (available at backward time) or an input
        if (ctx.is_param(fwd_input) || ctx.is_input(fwd_input)) {
            // Use the tensor directly (it's available at backward time)
            attrs["shape_like"] = AttrValue{fwd_input};
        } else {
            // Need to save the tensor for its shape
            attrs["shape_like"] = AttrValue{saved_ref(fwd_input)};
        }

        ops.push_back(make_operation("view_backward_" + std::to_string(ctx.op_counter++),
                                     "view",
                                     "view",
                                     {ctx.d_output},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("view", ::dsl::view_backward);
REGISTER_AUTODIFF("reshape", ::dsl::view_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// View / Reshape
// ------------------------------------------------------------------------
const int _view_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "view";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.empty() || outputs.empty()) {
            ShapeValidationError err;
            err.message = "view requires 1 input and 1 output";
            return std::make_optional(err);
        }
        if (outputs[0].empty()) {
            return std::optional<ShapeValidationError>();
        }
        return validators::check_same_numel(inputs[0], outputs[0], "input", "output", "view");
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// ViewBackward
// ------------------------------------------------------------------------
const int _view_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "view_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& d_in = outputs[0];

        // Check element count preserved
        if (auto err = validators::check_same_numel(d_in, d_out, "d_in", "d_out", "view_backward")) {
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
