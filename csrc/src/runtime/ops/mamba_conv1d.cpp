// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba causal conv1d operation dispatch.

#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_conv1d(const CompiledOp& op) {
    // Input: x [B, conv_dim, T]
    // Weights: weight [conv_dim, kernel], bias [conv_dim] (optional)
    // Output: out [B, conv_dim, T]
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor* bias = nullptr;
    if (op.inputs.size() > 2 && !op.inputs[2].name.empty()) {
        bias = &resolve_tensor(op.inputs[2]);
    }

    const int B = static_cast<int>(x.Sizes[0]);
    const int conv_dim = static_cast<int>(x.Sizes[1]);
    const int T = static_cast<int>(x.Sizes[2]);
    const int kernel = static_cast<int>(weight.Sizes[1]);

    // Determine if SiLU activation should be applied
    bool silu = (op.attrs.activation == "silu");

    const long expected = static_cast<long>(B) * conv_dim * T;
    auto shape_matches = [&](const TensorRef& ref) -> bool {
        if (ref.shape.empty()) return false;
        long prod = 1;
        for (auto d : ref.shape) {
            if (d <= 0) return false;
            prod *= d;
        }
        return prod == expected;
    };

    Tensor out_val;
    bool out_is_ref = false;
    if (shape_matches(op.outputs[0])) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.nelem() == expected) {
            out_val = out_ref;
            out_is_ref = true;
        }
    }
    if (!out_is_ref) {
        out_val = mRunState.temp_alloc(x.DType, {B, conv_dim, T}, "mamba_conv1d_out");
        mTemps.push_back(out_val);
    }

    if (sequence_chunk_active() && kernel > 1) {
        // Chunked-sequence carry via the extended-input trick: run the
        // standard kernel (fused activation included) on concat(tail, x) —
        // outputs [K-1, T+K-1) are exactly the values a full-sequence conv
        // would produce, because each sees the true previous K-1 inputs.
        const int Km1 = kernel - 1;
        const std::size_t es = get_dtype_size(x.DType);
        auto& cst = chunk_conv_state(static_cast<long>(B) * conv_dim * Km1, x.DType, op_layer_idx(op));
        const int c = sequence_chunk_idx();
        auto slot = [&](int i) {
            return static_cast<std::byte*>(cst.tails) + static_cast<std::size_t>(i) * cst.elems * es;
        };
        const std::size_t rows = static_cast<std::size_t>(B) * conv_dim;
        Tensor x_ext = mRunState.temp_alloc(x.DType, {B, conv_dim, T + Km1}, "mamba_conv1d_x_ext");
        Tensor out_ext = mRunState.temp_alloc(x.DType, {B, conv_dim, T + Km1}, "mamba_conv1d_out_ext");
        mTemps.push_back(x_ext);
        mTemps.push_back(out_ext);
        cudaStream_t stream = mRunState.MainStream;
        CUDA_CHECK(cudaMemcpy2DAsync(x_ext.Data, (T + Km1) * es, slot(c), Km1 * es, Km1 * es, rows,
                                     cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(x_ext.Data) + Km1 * es, (T + Km1) * es, x.Data, T * es,
                                     T * es, rows, cudaMemcpyDeviceToDevice, stream));
        // Save this chunk's input tail for the next chunk (idempotent on
        // re-forward and replay).
        CUDA_CHECK(cudaMemcpy2DAsync(slot(c + 1), Km1 * es, static_cast<std::byte*>(x.Data) + (T - Km1) * es, T * es,
                                     Km1 * es, rows, cudaMemcpyDeviceToDevice, stream));
        mamba_causal_conv1d_forward(out_ext, x_ext, weight, bias, B, T + Km1, conv_dim, kernel, silu, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(out_val.Data, T * es, static_cast<std::byte*>(out_ext.Data) + Km1 * es,
                                     (T + Km1) * es, T * es, rows, cudaMemcpyDeviceToDevice, stream));
        store_tensor(op.outputs[0], out_val);
        return;
    }

    // Call kernel
    mamba_causal_conv1d_forward(out_val, x, weight, bias, B, T, conv_dim, kernel, silu, mRunState.MainStream);

    store_tensor(op.outputs[0], out_val);
}

void CompiledExecutor::dispatch_mamba_conv1d_backward(const CompiledOp& op) {
    // Inputs: d_out [B, conv_dim, T], x [B, conv_dim, T], weight [conv_dim, kernel]
    // Outputs: dx [B, conv_dim, T], dweight [conv_dim, kernel], dbias [conv_dim] (optional)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(d_out.Sizes[0]);
    const int conv_dim = static_cast<int>(d_out.Sizes[1]);
    const int T = static_cast<int>(d_out.Sizes[2]);
    const int kernel = static_cast<int>(weight.Sizes[1]);

    bool silu = (op.attrs.activation == "silu");

    // Allocate outputs
    Tensor dx = mRunState.temp_alloc(d_out.DType, {B, conv_dim, T}, "mamba_conv1d_backward_dx");
    mTemps.push_back(dx);

    // Weight gradient is accumulated via atomicAdd — must zero-init (stack memory is stale)
    Tensor dweight_fp32 =
        mRunState.temp_alloc(ETensorDType::FP32, {conv_dim, kernel}, "mamba_conv1d_backward_dweight_fp32");
    mTemps.push_back(dweight_fp32);
    fill_zero(dweight_fp32, mRunState.MainStream);

    Tensor dbias_fp32;
    bool has_dbias = false;
    if (op.outputs.size() > 2) {
        dbias_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {conv_dim}, "mamba_conv1d_backward_dbias_fp32");
        mTemps.push_back(dbias_fp32);
        fill_zero(dbias_fp32, mRunState.MainStream);
        has_dbias = true;
    }

    if (sequence_chunk_active() && kernel > 1) {
        // Mirror of the forward's extended-input trick. dx_ext[0:K-1] is the
        // gradient w.r.t. the previous chunk's tail — handed backward via the
        // d_tail carry (chunks run last-to-first); the incoming carry adds
        // into this chunk's last K-1 input-gradient columns.
        const int Km1 = kernel - 1;
        const std::size_t es = get_dtype_size(x.DType);
        auto& cst = chunk_conv_state(static_cast<long>(B) * conv_dim * Km1, x.DType, op_layer_idx(op));
        const int c = sequence_chunk_idx();
        auto slot = [&](int i) {
            return static_cast<std::byte*>(cst.tails) + static_cast<std::size_t>(i) * cst.elems * es;
        };
        const std::size_t rows = static_cast<std::size_t>(B) * conv_dim;
        cudaStream_t stream = mRunState.MainStream;
        Tensor x_ext = mRunState.temp_alloc(x.DType, {B, conv_dim, T + Km1}, "mamba_conv1d_x_ext");
        Tensor dout_ext = mRunState.temp_alloc(x.DType, {B, conv_dim, T + Km1}, "mamba_conv1d_dout_ext");
        Tensor dx_ext = mRunState.temp_alloc(x.DType, {B, conv_dim, T + Km1}, "mamba_conv1d_dx_ext");
        mTemps.push_back(x_ext);
        mTemps.push_back(dout_ext);
        mTemps.push_back(dx_ext);
        CUDA_CHECK(cudaMemcpy2DAsync(x_ext.Data, (T + Km1) * es, slot(c), Km1 * es, Km1 * es, rows,
                                     cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(x_ext.Data) + Km1 * es, (T + Km1) * es, x.Data, T * es,
                                     T * es, rows, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(dout_ext.Data, 0, dout_ext.nelem() * es, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(dout_ext.Data) + Km1 * es, (T + Km1) * es, d_out.Data,
                                     T * es, T * es, rows, cudaMemcpyDeviceToDevice, stream));
        mamba_causal_conv1d_backward(dx_ext,
                                     dweight_fp32,
                                     has_dbias ? &dbias_fp32 : nullptr,
                                     x_ext,
                                     weight,
                                     dout_ext,
                                     B,
                                     T + Km1,
                                     conv_dim,
                                     kernel,
                                     silu,
                                     stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dx.Data, T * es, static_cast<std::byte*>(dx_ext.Data) + Km1 * es, (T + Km1) * es,
                                     T * es, rows, cudaMemcpyDeviceToDevice, stream));
        // Add the incoming carry (next chunk's tail gradient) into this
        // chunk's last K-1 columns via a zero-padded contiguous helper.
        Tensor carry_pad = mRunState.temp_alloc(x.DType, {B, conv_dim, T}, "mamba_conv1d_carry_pad");
        mTemps.push_back(carry_pad);
        CUDA_CHECK(cudaMemsetAsync(carry_pad.Data, 0, carry_pad.nelem() * es, stream));
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(carry_pad.Data) + (T - Km1) * es, T * es, cst.d_tail,
                                     Km1 * es, Km1 * es, rows, cudaMemcpyDeviceToDevice, stream));
        vector_add_sr(dx, dx, carry_pad, 1.0f, dx.nelem(), 0, stream);
        // Emit the new carry AFTER consuming the old one (same stream).
        CUDA_CHECK(cudaMemcpy2DAsync(cst.d_tail, Km1 * es, dx_ext.Data, (T + Km1) * es, Km1 * es, rows,
                                     cudaMemcpyDeviceToDevice, stream));
        store_tensor(op.outputs[0], dx);
        store_tensor(op.outputs[1], dweight_fp32);
        if (op.outputs.size() > 2 && has_dbias) {
            store_tensor(op.outputs[2], dbias_fp32);
        }
        return;
    }

    // Call kernel
    mamba_causal_conv1d_backward(dx,
                                 dweight_fp32,
                                 has_dbias ? &dbias_fp32 : nullptr,
                                 x,
                                 weight,
                                 d_out,
                                 B,
                                 T,
                                 conv_dim,
                                 kernel,
                                 silu,
                                 mRunState.MainStream);

    store_tensor(op.outputs[0], dx);
    store_tensor(op.outputs[1], dweight_fp32);
    if (op.outputs.size() > 2 && has_dbias) {
        store_tensor(op.outputs[2], dbias_fp32);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Mamba conv1d backward rule
// Forward: out = mamba_conv1d(x, weight, bias)
// Backward: dx, dweight, dbias = mamba_conv1d_backward(d_out, x, weight)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_conv1d_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string x = fwd.inputs[0];
    std::string weight = fwd.inputs[1];
    std::string bias = (fwd.inputs.size() > 2) ? fwd.inputs[2] : "";

    std::string x_ref = ctx.is_param(x) ? x : saved_ref(x);
    std::string weight_ref = ctx.is_param(weight) ? weight : saved_ref(weight);

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");
    if (fwd.inputs.size() > 2 && ctx.needs_grad(2)) {
        outputs.push_back(ctx.d_inputs[2]);
    }

    AttrMap attrs = copy_attrs(fwd.attrs, {"activation"}, "mamba_conv1d");

    ops.push_back(make_operation("mamba_conv1d_backward_" + std::to_string(ctx.op_counter++),
                                 "mamba_conv1d_backward",
                                 "mamba_conv1d_backward",
                                 {ctx.d_output, x_ref, weight_ref},
                                 outputs,
                                 attrs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("mamba_conv1d", ::dsl::mamba_conv1d_backward);
