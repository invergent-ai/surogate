#include "runtime/executor/compiled_ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {
void CompiledExecutor::dispatch_swiglu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Handle both 3D [B, T, 2*D] and 2D [N, 2*D] tensors (MoE produces 2D)
    if (inp.Rank == 2) {
        // 2D input: [N, 2*D] -> [N, D] (used by MoE path)
        const long N = inp.Sizes[0];
        const long D = inp.Sizes[1] / 2;

        // MoE output shape is dynamic, allocate with runtime shape
        std::vector<long> out_shape = {N, D};
        Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "swiglu_out");
        mTemps.push_back(out);

        swiglu_forward(out, inp, nullptr, 1, static_cast<int>(N), static_cast<int>(D), mRunState.MainStream);

        // Store output in tensor map for subsequent ops
        store_tensor(op.outputs[0], out);
    } else {
        // 3D input: [B, T, 2*D] -> [B, T, D] (standard path)
        Tensor& out = ensure_output_tensor(op.outputs[0]);

        const long B = inp.Sizes[0];
        const long T = inp.Sizes[1];
        const long D = inp.Sizes[2] / 2;
        swiglu_forward(out,
                       inp,
                       nullptr,
                       static_cast<int>(B),
                       static_cast<int>(T),
                       static_cast<int>(D),
                       mRunState.MainStream);

        // Pre-quantize swiglu output into FP8 buffer for the downstream MLPDown matmul.
        // This co-locates quantization with the data producer (better L2 locality)
        // and allows the matmul recipe to skip its own quantization pass.
        if (mRecipe && mRecipe->uses_fp8_forward() && mRunState.has_fp8_forward() &&
            !mRunState.has_fp8_delayed_scaling()) {
            auto& fp8_buf = mRunState.fp8_forward_quants().swiglu;
            if (fp8_buf.Data && fp8_buf.abs_max() && fp8_buf.scale()) {
                const long num_elements = B * T * D;
                Tensor out_flat = view_tensor(out, {B * T, D});
                quantize_with_abs_max(fp8_buf,
                                      fp8_buf.scale(),
                                      out_flat,
                                      fp8_buf.abs_max(),
                                      num_elements,
                                      mRunState.DeviceProp,
                                      mRunState.MainStream);
                mRunState.set_fp8_buffer_ready(DslRunState::FP8Ready_SwiGLU);
            }
        }
    }
}

void CompiledExecutor::dispatch_swiglu_backward(const CompiledOp& op) {
    // inputs: d_out, input (the mlp_up output before swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& d_inp = ensure_output_tensor(op.outputs[0]);
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && op.inputs.size() > 1) {
        std::string_view name = op.inputs[1].name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        std::string field;
        parse_block_param(name, layer_idx, field);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr =
        mRunState.has_fp8_hybrid_backward() ? mRunState.simplified_quant_grads().d_mlp_up.abs_max() : nullptr;

    // Handle both 3D [B, T, D] and 2D [N, D] tensors (MoE produces 2D)
    if (d_out.Rank == 2) {
        // 2D case for MoE: d_out is [N, D], inp is [N, 2*D]
        const long N = d_out.Sizes[0];
        const long D = d_out.Sizes[1];

        // EP changes token count dynamically — pre-allocated buffer may be wrong size.
        // Re-allocate if needed (same pattern as moe_grouped_gemm_gate_up_backward).
        Tensor* d_inp_ptr = &d_inp;
        const long expected_nelem = static_cast<long>(inp.nelem());
        if (d_inp_ptr->nelem() != expected_nelem) {
            std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
            Tensor tmp = mRunState.temp_alloc(inp.DType, shape, "swiglu_backward_d_inp");
            mTemps.push_back(tmp);
            store_tensor(op.outputs[0], tmp);
            d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
        }

        swiglu_backward(*d_inp_ptr,
                        d_out,
                        inp,
                        abs_max_ptr,
                        1,
                        static_cast<int>(N),
                        static_cast<int>(D),
                        mRunState.MainStream);
    } else {
        // 3D case: d_out is [B, T, D]
        const long D = d_out.Sizes[2];
        swiglu_backward(d_inp,
                        d_out,
                        inp,
                        abs_max_ptr,
                        static_cast<int>(d_out.Sizes[0]),
                        static_cast<int>(d_out.Sizes[1]),
                        static_cast<int>(D),
                        mRunState.MainStream);
    }
}

namespace {

// -----------------------------------------------------------------------------
// SwiGLU backward rule
// Forward: out = swiglu(gate, up) = silu(gate) * up
// Backward: d_gate, d_up = swiglu_backward(d_out, gate, up)
// -----------------------------------------------------------------------------
std::vector<Operation> swiglu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    // DSL swiglu takes a single gate_up input (packed) -> output
    if (fwd.inputs.size() == 1) {
        std::string gate_up = fwd.inputs[0];
        std::vector<std::string> outputs;
        outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
        ops.push_back(make_operation("swiglu_backward_" + std::to_string(ctx.op_counter++),
                                     "swiglu_backward",
                                     "swiglu_backward",
                                     {ctx.d_output, saved_ref(gate_up)},
                                     outputs));
        return ops;
    }

    // Legacy form: swiglu(gate, up)
    std::string gate = fwd.inputs[0];
    std::string up = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    ops.push_back(make_operation("swiglu_backward_" + std::to_string(ctx.op_counter++),
                                 "swiglu_backward",
                                 "swiglu_backward",
                                 {ctx.d_output, saved_ref(gate), saved_ref(up)},
                                 outputs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("swiglu", ::dsl::swiglu_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// SwiGLU
// ------------------------------------------------------------------------
const int _swiglu_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "swiglu";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.empty() || outputs.empty()) {
            ShapeValidationError err;
            err.message = "swiglu requires 1 input and 1 output";
            return std::make_optional(err);
        }

        const auto& in_shape = inputs[0];
        const auto& out_shape = outputs[0];

        // Input last dim should be 2x output last dim
        if (!in_shape.empty() && !out_shape.empty()) {
            if (in_shape.back() != 2 * out_shape.back()) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "swiglu: input last dim (" << in_shape.back() << ") should be 2x output last dim ("
                    << out_shape.back() << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        // All other dims should match
        if (in_shape.size() != out_shape.size()) {
            ShapeValidationError err;
            err.message = "swiglu: input and output rank must match";
            return std::make_optional(err);
        }

        for (size_t i = 0; i + 1 < in_shape.size(); ++i) {
            if (in_shape[i] != out_shape[i]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "swiglu: dimension [" << i << "] mismatch: " << in_shape[i] << " != " << out_shape[i];
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// SwiGLUBackward
// ------------------------------------------------------------------------
const int _swiglu_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "swiglu_backward";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& mlp_up = inputs[1];
        const auto& d_inp = outputs[0];

        // Check mlp_up last dim is 2x d_out last dim
        if (!mlp_up.empty() && !d_out.empty()) {
            long mlp_up_dim = mlp_up.back();
            long d_out_dim = d_out.back();
            if (mlp_up_dim != 2 * d_out_dim) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "swiglu_backward: mlp_up last dim (" << mlp_up_dim << ") must be 2x d_out last dim ("
                    << d_out_dim << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        // d_inp matches mlp_up shape
        if (auto err = validators::check_same_numel(d_inp, mlp_up, "d_inp", "mlp_up", "swiglu_backward")) {
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
