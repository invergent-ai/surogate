// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba selective scan (SSM) operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_ssm_scan(const CompiledOp& op) {
    // Inputs: u [B, D, T], delta [B, D, T], A_log [H], B [B, G, N, T], C [B, G, N, T], D_param [H]
    //         dt_bias [H] (optional)
    // Outputs: out [B, D, T], ssm_state [B, D, N] (optional, for caching)
    Tensor& u = resolve_tensor(op.inputs[0]);
    Tensor& delta = resolve_tensor(op.inputs[1]);
    Tensor& A_log = resolve_tensor(op.inputs[2]);
    Tensor& B_ssm = resolve_tensor(op.inputs[3]);
    Tensor& C_ssm = resolve_tensor(op.inputs[4]);
    Tensor& D_param = resolve_tensor(op.inputs[5]);

    Tensor* dt_bias = nullptr;
    if (op.inputs.size() > 6 && !op.inputs[6].name.empty()) {
        dt_bias = &resolve_tensor(op.inputs[6]);
    }

    // Use op.attrs for dimensions — tensor Sizes may be wrong after DSL view() ops
    const int num_heads = op.attrs.mamba_num_heads;
    const int head_dim = op.attrs.mamba_head_dim;
    const int D = op.attrs.intermediate_size > 0
        ? op.attrs.intermediate_size
        : num_heads * head_dim;
    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;
    const int B = static_cast<int>(u.Sizes[0]);  // Batch is always first dim
    const int T = static_cast<int>(u.nelem() / (static_cast<long>(B) * D));
    // n_chunks must match the selective_scan kernel's maximum kChunkSize (128 threads * 16 items = 2048).
    // This is NOT the model's Mamba chunk_size attribute — it's the kernel's internal processing granularity.
    constexpr int kKernelMaxChunkSize = 2048;
    const int n_chunks = (T + kKernelMaxChunkSize - 1) / kKernelMaxChunkSize;

    // Expand A_log to A [D, N]
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate});
    mTemps.push_back(A);
    mamba_expand_A(A, A_log, num_heads, head_dim, dstate, mRunState.MainStream);

    // Expand D_param to D_expanded [D]
    Tensor D_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(D_expanded);
    mamba_expand_head_param(D_expanded, D_param, num_heads, head_dim, mRunState.MainStream);

    // Expand dt_bias if present
    Tensor dt_bias_expanded;
    if (dt_bias) {
        dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
        mTemps.push_back(dt_bias_expanded);
        mamba_expand_head_param(dt_bias_expanded, *dt_bias, num_heads, head_dim, mRunState.MainStream);
    } else {
        // Create zero bias
        dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
        mTemps.push_back(dt_bias_expanded);
        CUDA_CHECK(cudaMemsetAsync(dt_bias_expanded.Data, 0, dt_bias_expanded.bytes(), mRunState.MainStream));
    }

    // Allocate output
    Tensor out = mRunState.temp_alloc(u.DType, {B, D, T});
    mTemps.push_back(out);

    // Allocate SSM state buffer (used internally by selective_scan)
    Tensor x = mRunState.temp_alloc(ETensorDType::FP32, {B, D, n_chunks, dstate * 2});
    mTemps.push_back(x);

    // Call selective scan forward
    mamba_selective_scan_forward(out, u, delta, A, B_ssm, C_ssm, D_expanded, dt_bias_expanded,
                                  x, B, T, D, dstate, groups, n_chunks, mRunState.MainStream);

    // Transpose output from [B, D, T] to [B, T, D] for downstream ops (gated RMSNorm, out_proj)
    // which expect standard [B, T, D] layout.  Gate from split_proj is [B, T, I] physical,
    // so SSM output must also be [B, T, D] physical to match for element-wise operations.
    Tensor out_btd = mRunState.temp_alloc(out.DType, {B, T, D});
    mTemps.push_back(out_btd);
    mamba_transpose_bdt_to_btd(out_btd, out, B, T, D, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = out_btd;

    // Optionally save ssm_state for backward
    if (op.outputs.size() > 1) {
        mTensorMap[op.outputs[1].name] = x;
    }
}

void CompiledExecutor::dispatch_mamba_ssm_scan_backward(const CompiledOp& op) {
    // Inputs: d_out [B, D, T], u, delta, A_log, B, C, D_param, dt_bias, ssm_state
    // Outputs: d_u, d_delta, d_A_log, d_B, d_C, d_D, d_dt_bias
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& u = resolve_tensor(op.inputs[1]);
    Tensor& delta = resolve_tensor(op.inputs[2]);
    Tensor& A_log = resolve_tensor(op.inputs[3]);
    Tensor& B_ssm = resolve_tensor(op.inputs[4]);
    Tensor& C_ssm = resolve_tensor(op.inputs[5]);
    Tensor& D_param = resolve_tensor(op.inputs[6]);
    Tensor& dt_bias = resolve_tensor(op.inputs[7]);
    Tensor& x = resolve_tensor(op.inputs[8]);

    // Use op.attrs for dimensions — tensor Sizes may be wrong after DSL view() ops
    const int num_heads = op.attrs.mamba_num_heads;
    const int head_dim = op.attrs.mamba_head_dim;
    const int D = op.attrs.intermediate_size > 0
        ? op.attrs.intermediate_size
        : num_heads * head_dim;
    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;
    const int B = static_cast<int>(u.Sizes[0]);  // Batch is always first dim
    const int T = static_cast<int>(u.nelem() / (static_cast<long>(B) * D));
    // n_chunks must match the selective_scan kernel's maximum kChunkSize (128 threads * 16 items = 2048).
    constexpr int kKernelMaxChunkSize = 2048;
    const int n_chunks = (T + kKernelMaxChunkSize - 1) / kKernelMaxChunkSize;

    // Expand parameters (same as forward)
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate});
    mTemps.push_back(A);
    mamba_expand_A(A, A_log, num_heads, head_dim, dstate, mRunState.MainStream);

    Tensor D_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(D_expanded);
    mamba_expand_head_param(D_expanded, D_param, num_heads, head_dim, mRunState.MainStream);

    Tensor dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(dt_bias_expanded);
    mamba_expand_head_param(dt_bias_expanded, dt_bias, num_heads, head_dim, mRunState.MainStream);

    // Transpose d_out from [B, T, D] to [B, D, T] to match kernel's expected layout.
    // The forward output was transposed from [B, D, T] to [B, T, D], so the incoming
    // gradient is in [B, T, D] layout and must be transposed back.
    Tensor d_out_bdt = mRunState.temp_alloc(d_out.DType, {B, D, T});
    mTemps.push_back(d_out_bdt);
    mamba_transpose_btd_to_bdt(d_out_bdt, d_out, B, T, D, mRunState.MainStream);

    // Allocate gradient outputs — du and ddelta are written directly by the kernel (store_output),
    // while dA/dB/dC/dD/ddelta_bias use atomicAdd, so they MUST be zero-initialized.
    Tensor du = mRunState.temp_alloc(u.DType, {B, D, T});
    mTemps.push_back(du);

    Tensor ddelta = mRunState.temp_alloc(u.DType, {B, D, T});
    mTemps.push_back(ddelta);

    Tensor dA = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate});
    mTemps.push_back(dA);
    fill_zero(dA, mRunState.MainStream);

    Tensor dB = mRunState.temp_alloc(ETensorDType::FP32, {B, groups, dstate, T});
    mTemps.push_back(dB);
    fill_zero(dB, mRunState.MainStream);

    Tensor dC = mRunState.temp_alloc(ETensorDType::FP32, {B, groups, dstate, T});
    mTemps.push_back(dC);
    fill_zero(dC, mRunState.MainStream);

    Tensor dD_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(dD_expanded);
    fill_zero(dD_expanded, mRunState.MainStream);

    Tensor ddelta_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D});
    mTemps.push_back(ddelta_bias_expanded);
    fill_zero(ddelta_bias_expanded, mRunState.MainStream);

    // Call selective scan backward (using transposed d_out in [B, D, T] layout)
    mamba_selective_scan_backward(du, ddelta, dA, dB, dC, &dD_expanded, &ddelta_bias_expanded,
                                   u, delta, A, B_ssm, C_ssm, D_expanded, dt_bias_expanded,
                                   d_out_bdt, x, B, T, D, dstate, groups, n_chunks, mRunState.MainStream);

    // Reduce expanded gradients back to per-head
    Tensor dA_log = mRunState.temp_alloc(ETensorDType::FP32, {num_heads});
    mTemps.push_back(dA_log);
    mamba_reduce_dA_log(dA_log, dA, A, num_heads, head_dim, dstate, false, mRunState.MainStream);

    Tensor dD = mRunState.temp_alloc(ETensorDType::FP32, {num_heads});
    mTemps.push_back(dD);
    mamba_reduce_head_param(dD, dD_expanded, num_heads, head_dim, false, mRunState.MainStream);

    Tensor ddelta_bias = mRunState.temp_alloc(ETensorDType::FP32, {num_heads});
    mTemps.push_back(ddelta_bias);
    mamba_reduce_head_param(ddelta_bias, ddelta_bias_expanded, num_heads, head_dim, false, mRunState.MainStream);

    // Store outputs
    mTensorMap[op.outputs[0].name] = du;
    mTensorMap[op.outputs[1].name] = ddelta;
    mTensorMap[op.outputs[2].name] = dA_log;
    mTensorMap[op.outputs[3].name] = dB;
    mTensorMap[op.outputs[4].name] = dC;
    mTensorMap[op.outputs[5].name] = dD;
    mTensorMap[op.outputs[6].name] = ddelta_bias;
}

}  // namespace dsl
