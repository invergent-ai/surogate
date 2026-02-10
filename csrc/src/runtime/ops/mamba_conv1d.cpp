// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba causal conv1d operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
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

    Tensor* out_ptr = nullptr;
    if (shape_matches(op.outputs[0])) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.nelem() == expected) {
            out_ptr = &out_ref;
        }
    }
    if (!out_ptr) {
        Tensor out = mRunState.temp_alloc(x.DType, {B, conv_dim, T});
        mTemps.push_back(out);
        out_ptr = &mTemps.back();
    }

    // Call kernel
    mamba_causal_conv1d_forward(*out_ptr, x, weight, bias,
                                 B, T, conv_dim, kernel, silu,
                                 mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = *out_ptr;
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
    Tensor dx = mRunState.temp_alloc(d_out.DType, {B, conv_dim, T});
    mTemps.push_back(dx);

    // Weight gradient is accumulated via atomicAdd — must zero-init (stack memory is stale)
    Tensor dweight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {conv_dim, kernel});
    mTemps.push_back(dweight_fp32);
    fill_zero(dweight_fp32, mRunState.MainStream);

    Tensor* dbias_fp32 = nullptr;
    if (op.outputs.size() > 2) {
        Tensor dbias = mRunState.temp_alloc(ETensorDType::FP32, {conv_dim});
        mTemps.push_back(dbias);
        fill_zero(mTemps.back(), mRunState.MainStream);
        dbias_fp32 = &mTemps.back();
    }

    // Call kernel
    mamba_causal_conv1d_backward(dx, dweight_fp32, dbias_fp32,
                                  x, weight, d_out,
                                  B, T, conv_dim, kernel, silu,
                                  mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = dx;
    mTensorMap[op.outputs[1].name] = dweight_fp32;
    if (op.outputs.size() > 2 && dbias_fp32) {
        mTensorMap[op.outputs[2].name] = *dbias_fp32;
    }
}

}  // namespace dsl
