#include "runtime/executor/compiled_ops.h"

#include <algorithm>
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
namespace {

int normalize_dim(int dim, int rank) {
    int out = dim;
    if (out < 0) {
        out += rank;
    }
    if (out < 0 || out >= rank) {
        throw std::runtime_error("dispatch_transpose: dim out of range");
    }
    return out;
}

bool tensor_matches_shape(const Tensor& t, const std::vector<long>& shape) {
    return tensor_shape_matches(t, shape);
}

}  // namespace

void CompiledExecutor::dispatch_transpose(const CompiledOp& op) {
    if (op.inputs.size() != 1) {
        throw std::runtime_error("dispatch_transpose: expected exactly one input");
    }
    if (op.outputs.empty() || op.outputs[0].name.empty()) {
        throw std::runtime_error("dispatch_transpose: expected one non-empty output");
    }

    Tensor& in = resolve_tensor(op.inputs[0]);
    if (in.Rank <= 0) {
        throw std::runtime_error("dispatch_transpose: input rank must be > 0");
    }

    const int rank = in.Rank;
    const int dim0 = normalize_dim(op.attrs.dim0, rank);
    const int dim1 = normalize_dim(op.attrs.dim1, rank);

    if (dim0 == dim1) {
        store_tensor(op.outputs[0], in);
        return;
    }

    std::vector<long> out_shape(in.Sizes.begin(), in.Sizes.begin() + rank);
    std::swap(out_shape[dim0], out_shape[dim1]);

    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.DType != in.DType || !tensor_matches_shape(out, out_shape)) {
        out = mRunState.temp_alloc(in.DType, out_shape, "transpose_out");
        mTemps.push_back(out);
    }

    // Fast path: 2D transpose.
    if (rank == 2 && ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))) {
        const int rows = static_cast<int>(in.Sizes[0]);
        const int cols = static_cast<int>(in.Sizes[1]);
        transpose(out, in, rows, cols, mRunState.MainStream);
        store_tensor(op.outputs[0], out);
        return;
    }

    // Needed by Qwen3.5 dense linear-attention path:
    // [B, T, C] <-> [B, C, T] (swap dims 1 and 2).
    if (rank == 3 && ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1))) {
        const long B = in.Sizes[0];
        const int rows = static_cast<int>(in.Sizes[1]);
        const int cols = static_cast<int>(in.Sizes[2]);
        const std::size_t slice_bytes =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * get_dtype_size(in.DType);

        for (long b = 0; b < B; ++b) {
            Tensor src2 = in;
            src2.Rank = 2;
            src2.Sizes[0] = rows;
            src2.Sizes[1] = cols;
            src2.Data = in.Data + static_cast<std::size_t>(b) * slice_bytes;

            Tensor dst2 = out;
            dst2.Rank = 2;
            dst2.Sizes[0] = cols;
            dst2.Sizes[1] = rows;
            dst2.Data = out.Data + static_cast<std::size_t>(b) * slice_bytes;

            transpose(dst2, src2, rows, cols, mRunState.MainStream);
        }
        store_tensor(op.outputs[0], out);
        return;
    }

    throw std::runtime_error(
        "dispatch_transpose: unsupported transpose pattern (currently supports rank-2 and rank-3 dim swap 1<->2)");
}

namespace {

// -----------------------------------------------------------------------------
// Transpose backward rule
// Forward: y = transpose(x, dim0, dim1)
// Backward: dx = transpose(dy, dim0, dim1)
// -----------------------------------------------------------------------------
std::vector<Operation> transpose_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    if (!ctx.needs_grad(0)) {
        return ops;
    }

    const auto& fwd = ctx.fwd_op;
    AttrMap attrs = copy_attrs(fwd.attrs, {"dim0", "dim1"});
    ops.push_back(make_operation("transpose_backward_" + std::to_string(ctx.op_counter++),
                                 "transpose",
                                 "transpose",
                                 {ctx.d_output},
                                 {ctx.d_inputs[0]},
                                 attrs));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("transpose", ::dsl::transpose_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Transpose
// ------------------------------------------------------------------------
const int _transpose_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "transpose";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap& attrs, const ShapeEnv&) {
        if (inputs.empty() || outputs.empty()) {
            ShapeValidationError err;
            err.message = "transpose requires 1 input and 1 output";
            return std::make_optional(err);
        }
        if (inputs[0].empty() || outputs[0].empty()) {
            return std::optional<ShapeValidationError>();
        }

        const int rank = static_cast<int>(inputs[0].size());
        long dim0 = 0;
        long dim1 = 1;
        if (const AttrValue* a = find_attr(attrs, "dim0")) {
            if (auto v = attr_int(*a)) dim0 = *v;
        }
        if (const AttrValue* a = find_attr(attrs, "dim1")) {
            if (auto v = attr_int(*a)) dim1 = *v;
        }
        if (dim0 < 0) dim0 += rank;
        if (dim1 < 0) dim1 += rank;
        if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank || dim0 == dim1) {
            ShapeValidationError err;
            err.message = "transpose: invalid dim0/dim1 for input rank";
            return std::make_optional(err);
        }

        auto expected = inputs[0];
        std::swap(expected[dim0], expected[dim1]);
        if (outputs[0].size() != expected.size()) {
            ShapeValidationError err;
            err.message = "transpose: output rank mismatch";
            return std::make_optional(err);
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            if (outputs[0][i] != expected[i]) {
                ShapeValidationError err;
                err.message = "transpose: output shape mismatch";
                return std::make_optional(err);
            }
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
