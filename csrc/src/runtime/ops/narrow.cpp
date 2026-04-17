#include "runtime/executor/compiled_ops.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_narrow(const CompiledOp& op) {
    if (op.inputs.size() != 1) {
        throw std::runtime_error("dispatch_narrow: expected exactly one input");
    }
    if (op.outputs.size() != 1) {
        throw std::runtime_error("dispatch_narrow: expected exactly one output");
    }

    Tensor& in = resolve_tensor(op.inputs[0]);
    if (in.Rank <= 0) {
        throw std::runtime_error("dispatch_narrow: input rank must be > 0");
    }

    const int rank = in.Rank;
    int dim = op.attrs.split_concat_dim;  // reuses the common "dim" attr
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) {
        throw std::runtime_error("dispatch_narrow: dim out of range");
    }

    const int start = op.attrs.narrow_start;
    const int length = op.attrs.narrow_length;
    const long in_dim = in.Sizes[dim];

    if (start < 0 || length <= 0 || start + length > in_dim) {
        std::ostringstream oss;
        oss << "dispatch_narrow: invalid range [" << start << ", " << start + length << ") for dim " << dim
            << " of size " << in_dim;
        throw std::runtime_error(oss.str());
    }

    // Compute output shape (same rank, narrowed along dim)
    std::vector<long> out_shape(in.Sizes.begin(), in.Sizes.begin() + rank);
    out_shape[dim] = length;

    // Compute byte offset and strides
    const std::size_t elem_bytes = get_dtype_size(in.DType);
    long inner = 1;
    for (int i = dim + 1; i < rank; ++i) {
        inner *= in.Sizes[i];
    }
    long outer = 1;
    for (int i = 0; i < dim; ++i) {
        outer *= in.Sizes[i];
    }

    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.DType != in.DType || !tensor_shape_matches(out, out_shape)) {
        out = mRunState.temp_alloc(in.DType, out_shape, "narrow_output");
        mTemps.push_back(out);
    }

    const std::byte* in_ptr = static_cast<const std::byte*>(in.Data);
    std::byte* out_ptr = static_cast<std::byte*>(out.Data);

    const std::size_t src_pitch = static_cast<std::size_t>(in_dim) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::size_t row_bytes = static_cast<std::size_t>(length) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::byte* src_base = in_ptr + static_cast<std::size_t>(start) * static_cast<std::size_t>(inner) * elem_bytes;

    if (outer == 1) {
        CUDA_CHECK(cudaMemcpyAsync(out_ptr, src_base, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(out_ptr,
                                     row_bytes,
                                     src_base,
                                     src_pitch,
                                     row_bytes,
                                     static_cast<std::size_t>(outer),
                                     cudaMemcpyDeviceToDevice,
                                     mRunState.MainStream));
    }

    store_tensor(op.outputs[0], out);
}

// Backward: d_input = zeros_like(input); d_input[..., start:start+length, ...] = d_output
// Inputs: [0] d_output, [1] saved forward input (for shape reference)
// Outputs: [0] d_input
void CompiledExecutor::dispatch_narrow_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& fwd_input = resolve_tensor(op.inputs[1]);

    const int rank = fwd_input.Rank;
    int dim = op.attrs.split_concat_dim;
    if (dim < 0) dim += rank;

    const int start = op.attrs.narrow_start;
    const int length = op.attrs.narrow_length;

    // Allocate d_input with full input shape, zero-filled
    std::vector<long> in_shape(fwd_input.Sizes.begin(), fwd_input.Sizes.begin() + rank);
    Tensor& d_in_ref = ensure_output_tensor(op.outputs[0]);
    Tensor d_in = d_in_ref;
    if (d_in.nelem() != fwd_input.nelem() || d_in.DType != d_out.DType) {
        d_in = mRunState.temp_alloc(d_out.DType, in_shape, "narrow_bwd_din");
        mTemps.push_back(d_in);
    }
    // Zero the entire gradient
    cudaMemsetAsync(d_in.Data, 0, d_in.bytes(), mRunState.MainStream);

    // Copy d_output into the slice
    const std::size_t elem_bytes = get_dtype_size(d_out.DType);
    long inner = 1;
    for (int i = dim + 1; i < rank; ++i)
        inner *= fwd_input.Sizes[i];
    long outer = 1;
    for (int i = 0; i < dim; ++i)
        outer *= fwd_input.Sizes[i];

    std::byte* dst_base = static_cast<std::byte*>(d_in.Data) +
                          static_cast<std::size_t>(start) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::byte* src_ptr = static_cast<const std::byte*>(d_out.Data);

    const std::size_t dst_pitch =
        static_cast<std::size_t>(fwd_input.Sizes[dim]) * static_cast<std::size_t>(inner) * elem_bytes;
    const std::size_t row_bytes = static_cast<std::size_t>(length) * static_cast<std::size_t>(inner) * elem_bytes;

    if (outer == 1) {
        CUDA_CHECK(cudaMemcpyAsync(dst_base, src_ptr, row_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(dst_base,
                                     dst_pitch,
                                     src_ptr,
                                     row_bytes,
                                     row_bytes,
                                     static_cast<std::size_t>(outer),
                                     cudaMemcpyDeviceToDevice,
                                     mRunState.MainStream));
    }

    store_tensor(op.outputs[0], d_in);
}

namespace {

// -----------------------------------------------------------------------------
// Narrow backward rule
// Forward: y = narrow(x, dim, start, length)
// Backward: d_x = zeros_like(x); d_x[..., start:start+length, ...] = d_y
// -----------------------------------------------------------------------------
std::vector<Operation> narrow_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        // Copy dim, start, length attributes from forward
        AttrMap attrs = copy_attrs(ctx.fwd_op.attrs, {"dim", "start", "length"});
        // Pass the forward input name so backward can determine the full shape
        ops.push_back(make_operation("narrow_backward_" + std::to_string(ctx.op_counter++),
                                     "narrow_backward",
                                     "narrow_backward",
                                     {ctx.d_output, saved_ref(ctx.fwd_op.inputs[0])},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("narrow", ::dsl::narrow_backward_rule);
