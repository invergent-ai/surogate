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
    const bool norm_before_gate = op.attrs.norm_before_gate;

    // 1. silu_gate = silu(gate)
    Tensor silu_gate = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    Tensor out_t = mRunState.temp_alloc(x.DType, {B, T, D});
    Tensor rstd = mRunState.temp_alloc(ETensorDType::FP32, {B * T, groups});
    mTemps.push_back(out_t);
    mTemps.push_back(rstd);

    Tensor normed_or_gated;
    if (norm_before_gate) {
        // Qwen3.5 style: out = RMSNorm(x) * silu(gate)
        Tensor normed = mRunState.temp_alloc(x.DType, {B, T, D});
        mTemps.push_back(normed);
        mamba_group_rmsnorm_forward(normed, rstd, x, weight, eps, B, T, D, groups, mRunState.MainStream);
        if (x.DType == ETensorDType::BF16) {
            elementwise_mul(out_t.get<nv_bfloat16>(), normed.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (x.DType == ETensorDType::FP16) {
            elementwise_mul(out_t.get<half>(), normed.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(out_t.get<float>(), normed.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }
        normed_or_gated = normed;  // Save normalized x for backward.
    } else {
        // Mamba default: out = RMSNorm(x * silu(gate))
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
        mamba_group_rmsnorm_forward(out_t, rstd, gated, weight, eps, B, T, D, groups, mRunState.MainStream);
        normed_or_gated = gated;  // Save norm input for backward.
    }

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
        persist_save(op.op_id + ".normed", normed_or_gated);
    }
}

void CompiledExecutor::dispatch_mamba_gated_rmsnorm_backward(const CompiledOp& op) {
    // Inputs: d_out [B, T, D], x [B, T, D], gate [B, T, D], weight [D], rstd [B*T, G], gated [B, T, D]
    // Outputs: d_x [B, T, D], d_gate [B, T, D], d_weight [D]
    //
    // Saved input[5] is:
    // - norm_before_gate=False: gated = x * silu(gate) (norm input)
    // - norm_before_gate=True : normed = RMSNorm(x) (pre-gate normalized output)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& gate = resolve_tensor(op.inputs[2]);
    Tensor& weight = resolve_tensor(op.inputs[3]);
    Tensor& rstd = resolve_tensor(op.inputs[4]);
    Tensor& normed_or_gated = resolve_tensor(op.inputs[5]);

    const int B = static_cast<int>(x.Sizes[0]);
    const int T = static_cast<int>(x.Sizes[1]);
    const int D = static_cast<int>(x.Sizes[2]);
    const long n = static_cast<long>(B) * T * D;
    const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
    const bool norm_before_gate = op.attrs.norm_before_gate;

    // Common gate activation for both branches
    Tensor silu_gate = mRunState.temp_alloc(x.DType, {B, T, D});
    mTemps.push_back(silu_gate);
    silu_forward(silu_gate, gate, n, mRunState.MainStream);

    Tensor d_x = mRunState.temp_alloc(d_out.DType, {B, T, D});
    Tensor d_gate = mRunState.temp_alloc(d_out.DType, {B, T, D});
    Tensor d_weight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(d_x);
    mTemps.push_back(d_gate);
    mTemps.push_back(d_weight_fp32);

    if (norm_before_gate) {
        // Forward: out = RMSNorm(x) * silu(gate), saved normed_or_gated = RMSNorm(x)
        // d_normed = d_out * silu(gate)
        Tensor d_normed = mRunState.temp_alloc(d_out.DType, {B, T, D});
        mTemps.push_back(d_normed);
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_normed.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), silu_gate.get<nv_bfloat16>(),
                            n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_normed.get<half>(), d_out.get<half>(), silu_gate.get<half>(),
                            n, mRunState.MainStream);
        } else {
            elementwise_mul(d_normed.get<float>(), d_out.get<float>(), silu_gate.get<float>(),
                            n, mRunState.MainStream);
        }
        // d_x, d_weight via RMSNorm backward on x.
        mamba_group_rmsnorm_backward_dx(d_x, d_normed, x, weight, rstd, B, T, D, groups, mRunState.MainStream);
        mamba_group_rmsnorm_backward_dweight_fp32(
            d_weight_fp32, d_normed, x, rstd, B, T, D, groups, mRunState.MainStream);

        // d_gate = silu_backward(d_out * normed, gate)
        Tensor d_out_times_normed = mRunState.temp_alloc(d_out.DType, {B, T, D});
        mTemps.push_back(d_out_times_normed);
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_out_times_normed.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(),
                            normed_or_gated.get<nv_bfloat16>(), n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_out_times_normed.get<half>(), d_out.get<half>(),
                            normed_or_gated.get<half>(), n, mRunState.MainStream);
        } else {
            elementwise_mul(d_out_times_normed.get<float>(), d_out.get<float>(),
                            normed_or_gated.get<float>(), n, mRunState.MainStream);
        }
        silu_backward(d_gate, gate, d_out_times_normed, n, mRunState.MainStream);
    } else {
        // Forward: out = RMSNorm(gated), gated = x * silu(gate), saved normed_or_gated = gated
        Tensor d_gated = mRunState.temp_alloc(d_out.DType, {B, T, D});
        mTemps.push_back(d_gated);
        mamba_group_rmsnorm_backward_dx(
            d_gated, d_out, normed_or_gated, weight, rstd, B, T, D, groups, mRunState.MainStream);
        mamba_group_rmsnorm_backward_dweight_fp32(
            d_weight_fp32, d_out, normed_or_gated, rstd, B, T, D, groups, mRunState.MainStream);

        // d_x = d_gated * silu(gate)
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

        // d_gate = silu_backward(d_gated * x, gate)
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
        silu_backward(d_gate, gate, d_gated_times_x, n, mRunState.MainStream);
    }

    store_tensor(op.outputs[0], d_x);
    store_tensor(op.outputs[1], d_gate);
    store_tensor(op.outputs[2], d_weight_fp32);
}

}  // namespace dsl
