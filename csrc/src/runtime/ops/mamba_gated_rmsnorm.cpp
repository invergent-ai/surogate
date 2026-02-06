// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba gated RMSNorm operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_gated_rmsnorm(const CompiledOp& op) {
    // Inputs: x [B, T, D], gate [B, T, D], weight [D]
    // Output: out [B, T, D]
    // This performs: out = gate * rmsnorm(x, weight)
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& gate = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = static_cast<int>(x.Sizes[1]);
    const int D = static_cast<int>(x.Sizes[2]);

    // Get normalization parameters
    const float eps = op.attrs.eps;
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;

    // Allocate outputs
    Tensor out = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(out);

    // Allocate rstd for backward
    Tensor rstd = mRunState.temp_alloc(ETensorDType::FP32, {B * T, groups});
    mTemps.push_back(rstd);

    // First compute RMSNorm(x)
    Tensor normed = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(normed);

    mamba_group_rmsnorm_forward(normed, rstd, x, weight, eps, B, T, D, groups, mRunState.MainStream);

    // Then multiply by gate: out = gate * normed
    // Use element-wise multiply kernel
    if (x.DType == ETensorDType::BF16) {
        elementwise_mul(out.get<nv_bfloat16>(), gate.get<nv_bfloat16>(), normed.get<nv_bfloat16>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else if (x.DType == ETensorDType::FP16) {
        elementwise_mul(out.get<half>(), gate.get<half>(), normed.get<half>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else {
        elementwise_mul(out.get<float>(), gate.get<float>(), normed.get<float>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    }

    mTensorMap[op.outputs[0].name] = out;

    // Save rstd and normed for backward if needed
    if (mSaved) {
        std::string rstd_name = op.op_id + ".rstd";
        (*mSaved)[rstd_name] = rstd;
        std::string normed_name = op.op_id + ".normed";
        (*mSaved)[normed_name] = normed;
    }
}

void CompiledExecutor::dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op) {
    // Inputs: d_out [B, T, D], x [B, T, D], gate [B, T, D], weight [D], rstd [B*T, G], normed [B, T, D]
    // Outputs: d_x [B, T, D], d_gate [B, T, D], d_weight [D]
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& gate = resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);
    Tensor& normed = resolve_tensor(op.inputs[5]);

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = static_cast<int>(x.Sizes[1]);
    const int D = static_cast<int>(x.Sizes[2]);
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;

    // d_gate = d_out * normed
    Tensor d_gate = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_gate);

    if (d_out.DType == ETensorDType::BF16) {
        elementwise_mul(d_gate.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), normed.get<nv_bfloat16>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP16) {
        elementwise_mul(d_gate.get<half>(), d_out.get<half>(), normed.get<half>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else {
        elementwise_mul(d_gate.get<float>(), d_out.get<float>(), normed.get<float>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    }

    // d_normed = d_out * gate
    Tensor d_normed = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_normed);

    if (d_out.DType == ETensorDType::BF16) {
        elementwise_mul(d_normed.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), gate.get<nv_bfloat16>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP16) {
        elementwise_mul(d_normed.get<half>(), d_out.get<half>(), gate.get<half>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    } else {
        elementwise_mul(d_normed.get<float>(), d_out.get<float>(), gate.get<float>(),
                        static_cast<long>(B) * T * D, mRunState.MainStream);
    }

    // d_x = RMSNorm_backward(d_normed, x, weight, rstd)
    Tensor d_x = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_x);

    mamba_group_rmsnorm_backward_dx(d_x, d_normed, x, weight, rstd, B, T, D, groups, mRunState.MainStream);

    // d_weight (accumulated in FP32)
    Tensor d_weight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(d_weight_fp32);

    mamba_group_rmsnorm_backward_dweight_fp32(d_weight_fp32, d_normed, x, rstd, B, T, D, groups, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = d_x;
    mTensorMap[op.outputs[1].name] = d_gate;
    mTensorMap[op.outputs[2].name] = d_weight_fp32;
}

}  // namespace dsl
