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
    //
    // Implements norm_before_gate=False (Nemotron-H / mamba_ssm convention):
    //   gated = x * silu(gate)
    //   out   = GroupRMSNorm(gated, weight)
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& gate = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = static_cast<int>(x.Sizes[1]);
    const int D = static_cast<int>(x.Sizes[2]);
    const long n = static_cast<long>(B) * T * D;

    // Get normalization parameters
    const float eps = op.attrs.eps;
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;

    // 1. silu_gate = silu(gate)
    Tensor silu_gate = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    // 2. gated = x * silu(gate)
    Tensor gated = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(gated);
    if (x.DType == ETensorDType::BF16) {
        elementwise_mul(gated.get<nv_bfloat16>(), x.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
    } else if (x.DType == ETensorDType::FP16) {
        elementwise_mul(gated.get<half>(), x.get<half>(), silu_gate.get<half>(),
                        n, mRunState.MainStream);
    } else {
        elementwise_mul(gated.get<float>(), x.get<float>(), silu_gate.get<float>(),
                        n, mRunState.MainStream);
    }

    // 3. out = GroupRMSNorm(gated, weight)
    // Allocate both output tensors upfront to avoid dangling pointers
    // from vector reallocation when pushing to mTemps.
    Tensor out_t = mRunState.temp_alloc(x.DType, {B, T, D});
    Tensor rstd = mRunState.temp_alloc(ETensorDType::FP32, {B * T, groups});
    mTemps.push_back(out_t);
    mTemps.push_back(rstd);

    mamba_group_rmsnorm_forward(out_t, rstd, gated, weight, eps, B, T, D, groups, mRunState.MainStream);

    store_tensor(op.outputs[0], out_t);

    // Save rstd and gated for backward.
    // Must persist via cudaMalloc because temp_alloc'd stack memory is freed at
    // layer boundaries (Stack.restore), leaving dangling pointers in mSaved.
    if (mSaved) {
        auto persist_save = [&](const std::string& name, const Tensor& src) {
            const size_t bytes = src.bytes();
            if (bytes == 0) return;
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            CUDA_CHECK(cudaMemcpyAsync(mMoeSavedBuffers[name], src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved;
            saved.DType = src.DType;
            saved.Rank = src.Rank;
            for (int d = 0; d < src.Rank; ++d) saved.Sizes[d] = src.Sizes[d];
            saved.Data = static_cast<std::byte*>(mMoeSavedBuffers[name]);
            (*mSaved)[name] = saved;
        };

        persist_save(op.op_id + ".rstd", rstd);
        persist_save(op.op_id + ".normed", gated);
    }
}

void CompiledExecutor::dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op) {
    // Inputs: d_out [B, T, D], x [B, T, D], gate [B, T, D], weight [D], rstd [B*T, G], gated [B, T, D]
    // Outputs: d_x [B, T, D], d_gate [B, T, D], d_weight [D]
    //
    // Forward was: gated = x * silu(gate); out = GroupRMSNorm(gated, weight)
    // Backward:
    //   d_gated  = GroupRMSNorm_bwd(d_out, gated, weight, rstd)  [dx of the norm]
    //   d_weight = GroupRMSNorm_bwd_dweight(d_out, gated, rstd)
    //   d_x      = d_gated * silu(gate)
    //   d_gate   = silu_backward(d_gated * x, gate)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& gate = resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);
    Tensor& gated = resolve_tensor(op.inputs[5]);  // was "normed" slot, now stores gated

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = static_cast<int>(x.Sizes[1]);
    const int D = static_cast<int>(x.Sizes[2]);
    const long n = static_cast<long>(B) * T * D;
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;

    // 1. d_gated = GroupRMSNorm_backward_dx(d_out, gated, weight, rstd)
    //    Note: the norm input was "gated", not "x", so backward uses "gated"
    Tensor d_gated = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_gated);
    mamba_group_rmsnorm_backward_dx(d_gated, d_out, gated, weight, rstd, B, T, D, groups, mRunState.MainStream);

    // 2. d_weight (accumulated in FP32) — norm input was "gated"
    Tensor d_weight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(d_weight_fp32);
    mamba_group_rmsnorm_backward_dweight_fp32(d_weight_fp32, d_out, gated, rstd, B, T, D, groups, mRunState.MainStream);

    // 3. Recompute silu(gate) for d_x
    Tensor silu_gate = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    // 4. d_x = d_gated * silu(gate)
    Tensor d_x = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_x);
    if (d_out.DType == ETensorDType::BF16) {
        elementwise_mul(d_x.get<nv_bfloat16>(), d_gated.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP16) {
        elementwise_mul(d_x.get<half>(), d_gated.get<half>(), silu_gate.get<half>(),
                        n, mRunState.MainStream);
    } else {
        elementwise_mul(d_x.get<float>(), d_gated.get<float>(), silu_gate.get<float>(),
                        n, mRunState.MainStream);
    }

    // 5. d_gate = silu_backward(d_gated * x, gate)
    //    silu_backward computes: d_inp = dout * (sigmoid(inp) + inp * sigmoid(inp) * (1 - sigmoid(inp)))
    //    where dout = d_gated * x, inp = gate
    Tensor d_gated_times_x = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_gated_times_x);
    if (d_out.DType == ETensorDType::BF16) {
        elementwise_mul(d_gated_times_x.get<nv_bfloat16>(), d_gated.get<nv_bfloat16>(), x.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP16) {
        elementwise_mul(d_gated_times_x.get<half>(), d_gated.get<half>(), x.get<half>(),
                        n, mRunState.MainStream);
    } else {
        elementwise_mul(d_gated_times_x.get<float>(), d_gated.get<float>(), x.get<float>(),
                        n, mRunState.MainStream);
    }

    Tensor d_gate = mRunState.temp_alloc(d_out.DType, {B, T, D});
    mTemps.push_back(d_gate);
    silu_backward(d_gate, gate, d_gated_times_x, n, mRunState.MainStream);

    store_tensor(op.outputs[0], d_x);
    store_tensor(op.outputs[1], d_gate);
    store_tensor(op.outputs[2], d_weight_fp32);
}

}  // namespace dsl
