// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba selective scan (SSM) operation dispatch.

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
    const int D = op.attrs.intermediate_size > 0 ? op.attrs.intermediate_size : num_heads * head_dim;
    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;
    const int B = static_cast<int>(u.Sizes[0]);  // Batch is always first dim
    const int T = static_cast<int>(u.nelem() / (static_cast<long>(B) * D));
    // n_chunks must match the selective_scan kernel's maximum kChunkSize (128 threads * 16 items = 2048).
    // This is NOT the model's Mamba chunk_size attribute — it's the kernel's internal processing granularity.
    constexpr int kKernelMaxChunkSize = 2048;
    const int n_chunks = (T + kKernelMaxChunkSize - 1) / kKernelMaxChunkSize;

    // Expand A_log to A [D, N]
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate}, "mamba_ssm_scan_A");
    mTemps.push_back(A);
    mamba_expand_A(A, A_log, num_heads, head_dim, dstate, mRunState.MainStream);

    // Expand D_param to D_expanded [D]
    Tensor D_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_D_expanded");
    mTemps.push_back(D_expanded);
    mamba_expand_head_param(D_expanded, D_param, num_heads, head_dim, mRunState.MainStream);

    // Expand dt_bias if present
    Tensor dt_bias_expanded;
    if (dt_bias) {
        dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_dt_bias_expanded");
        mTemps.push_back(dt_bias_expanded);
        mamba_expand_head_param(dt_bias_expanded, *dt_bias, num_heads, head_dim, mRunState.MainStream);
    } else {
        // Create zero bias
        dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_dt_bias_expanded_zero");
        mTemps.push_back(dt_bias_expanded);
        CUDA_CHECK(cudaMemsetAsync(dt_bias_expanded.Data, 0, dt_bias_expanded.bytes(), mRunState.MainStream));
    }

    // Allocate output
    Tensor out = mRunState.temp_alloc(u.DType, {B, D, T}, "mamba_ssm_scan_out");
    mTemps.push_back(out);

    // Allocate SSM state buffer (used internally by selective_scan)
    Tensor x = mRunState.temp_alloc(ETensorDType::FP32, {B, D, n_chunks, dstate * 2}, "mamba_ssm_scan_x");
    mTemps.push_back(x);

    // Call selective scan forward
    mamba_selective_scan_forward(out,
                                 u,
                                 delta,
                                 A,
                                 B_ssm,
                                 C_ssm,
                                 D_expanded,
                                 dt_bias_expanded,
                                 op.attrs.dt_min,
                                 op.attrs.dt_max,
                                 x,
                                 B,
                                 T,
                                 D,
                                 dstate,
                                 groups,
                                 n_chunks,
                                 mRunState.MainStream);

    // Transpose output from [B, D, T] to [B, T, D] for downstream ops (gated RMSNorm, out_proj)
    // which expect standard [B, T, D] layout.  Gate from split_proj is [B, T, I] physical,
    // so SSM output must also be [B, T, D] physical to match for element-wise operations.
    const long expected = static_cast<long>(B) * T * D;
    auto shape_matches = [](const TensorRef& ref, long expected_nelem) -> bool {
        if (ref.shape.empty()) return false;
        long prod = 1;
        for (auto d : ref.shape) {
            if (d <= 0) return false;
            prod *= d;
        }
        return prod == expected_nelem;
    };

    Tensor out_btd;
    bool out_is_ref = false;
    if (shape_matches(op.outputs[0], expected)) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.nelem() == expected) {
            out_btd = out_ref;
            out_is_ref = true;
        }
    }
    if (!out_is_ref) {
        out_btd = mRunState.temp_alloc(out.DType, {B, T, D});
        mTemps.push_back(out_btd);
    }

    mamba_transpose_bdt_to_btd(out_btd, out, B, T, D, mRunState.MainStream);
    store_tensor(op.outputs[0], out_btd);

    // Optionally save ssm_state for backward
    if (op.outputs.size() > 1) {
        store_tensor(op.outputs[1], x);
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
    const int D = op.attrs.intermediate_size > 0 ? op.attrs.intermediate_size : num_heads * head_dim;
    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;
    const int B = static_cast<int>(u.Sizes[0]);  // Batch is always first dim
    const int T = static_cast<int>(u.nelem() / (static_cast<long>(B) * D));
    // n_chunks must match the selective_scan kernel's maximum kChunkSize (128 threads * 16 items = 2048).
    constexpr int kKernelMaxChunkSize = 2048;
    const int n_chunks = (T + kKernelMaxChunkSize - 1) / kKernelMaxChunkSize;

    // Expand parameters (same as forward)
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate}, "mamba_ssm_scan_A");
    mTemps.push_back(A);
    mamba_expand_A(A, A_log, num_heads, head_dim, dstate, mRunState.MainStream);

    Tensor D_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_D_expanded");
    mTemps.push_back(D_expanded);
    mamba_expand_head_param(D_expanded, D_param, num_heads, head_dim, mRunState.MainStream);

    Tensor dt_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_dt_bias_expanded");
    mTemps.push_back(dt_bias_expanded);
    mamba_expand_head_param(dt_bias_expanded, dt_bias, num_heads, head_dim, mRunState.MainStream);

    // Transpose d_out from [B, T, D] to [B, D, T] to match kernel's expected layout.
    // The forward output was transposed from [B, D, T] to [B, T, D], so the incoming
    // gradient is in [B, T, D] layout and must be transposed back.
    Tensor d_out_bdt = mRunState.temp_alloc(d_out.DType, {B, D, T}, "mamba_ssm_scan_d_out_bdt");
    mTemps.push_back(d_out_bdt);
    mamba_transpose_btd_to_bdt(d_out_bdt, d_out, B, T, D, mRunState.MainStream);

    // Allocate gradient outputs — du and ddelta are written directly by the kernel (store_output),
    // while dA/dB/dC/dD/ddelta_bias use atomicAdd, so they MUST be zero-initialized.
    Tensor du = mRunState.temp_alloc(u.DType, {B, D, T}, "mamba_ssm_scan_du");
    mTemps.push_back(du);

    Tensor ddelta = mRunState.temp_alloc(u.DType, {B, D, T}, "mamba_ssm_scan_ddelta");
    mTemps.push_back(ddelta);

    Tensor dA = mRunState.temp_alloc(ETensorDType::FP32, {D, dstate}, "mamba_ssm_scan_dA");
    mTemps.push_back(dA);
    fill_zero(dA, mRunState.MainStream);

    Tensor dB = mRunState.temp_alloc(ETensorDType::FP32, {B, groups, dstate, T}, "mamba_ssm_scan_dB");
    mTemps.push_back(dB);
    fill_zero(dB, mRunState.MainStream);

    Tensor dC = mRunState.temp_alloc(ETensorDType::FP32, {B, groups, dstate, T}, "mamba_ssm_scan_dC");
    mTemps.push_back(dC);
    fill_zero(dC, mRunState.MainStream);

    Tensor dD_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_dD_expanded");
    mTemps.push_back(dD_expanded);
    fill_zero(dD_expanded, mRunState.MainStream);

    Tensor ddelta_bias_expanded = mRunState.temp_alloc(ETensorDType::FP32, {D}, "mamba_ssm_scan_ddelta_bias_expanded");
    mTemps.push_back(ddelta_bias_expanded);
    fill_zero(ddelta_bias_expanded, mRunState.MainStream);

    // Call selective scan backward (using transposed d_out in [B, D, T] layout)
    mamba_selective_scan_backward(du,
                                  ddelta,
                                  dA,
                                  dB,
                                  dC,
                                  &dD_expanded,
                                  &ddelta_bias_expanded,
                                  u,
                                  delta,
                                  A,
                                  B_ssm,
                                  C_ssm,
                                  D_expanded,
                                  dt_bias_expanded,
                                  op.attrs.dt_min,
                                  op.attrs.dt_max,
                                  d_out_bdt,
                                  x,
                                  B,
                                  T,
                                  D,
                                  dstate,
                                  groups,
                                  n_chunks,
                                  mRunState.MainStream);

    // Reduce expanded gradients back to per-head
    Tensor dA_log = mRunState.temp_alloc(ETensorDType::FP32, {num_heads}, "mamba_ssm_scan_dA_log");
    mTemps.push_back(dA_log);
    mamba_reduce_dA_log(dA_log, dA, A, num_heads, head_dim, dstate, false, mRunState.MainStream);

    Tensor dD = mRunState.temp_alloc(ETensorDType::FP32, {num_heads}, "mamba_ssm_scan_dD");
    mTemps.push_back(dD);
    mamba_reduce_head_param(dD, dD_expanded, num_heads, head_dim, false, mRunState.MainStream);

    Tensor ddelta_bias = mRunState.temp_alloc(ETensorDType::FP32, {num_heads}, "mamba_ssm_scan_ddelta_bias");
    mTemps.push_back(ddelta_bias);
    mamba_reduce_head_param(ddelta_bias, ddelta_bias_expanded, num_heads, head_dim, false, mRunState.MainStream);

    // Store outputs
    store_tensor(op.outputs[0], du);
    store_tensor(op.outputs[1], ddelta);
    store_tensor(op.outputs[2], dA_log);
    store_tensor(op.outputs[3], dB);
    store_tensor(op.outputs[4], dC);
    store_tensor(op.outputs[5], dD);
    store_tensor(op.outputs[6], ddelta_bias);
}

namespace {

// -----------------------------------------------------------------------------
// Mamba SSM scan backward rule
// Forward: out, ssm_state = mamba_ssm_scan(u, delta, A_log, B, C, D_param, dt_bias)
// Backward: du, ddelta, dA_log, dB, dC, dD, ddelta_bias = mamba_ssm_scan_backward(...)
// -----------------------------------------------------------------------------
std::vector<Operation> mamba_ssm_scan_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string u = fwd.inputs[0];
    std::string delta = fwd.inputs[1];
    std::string A_log = fwd.inputs[2];
    std::string B_ssm = fwd.inputs[3];
    std::string C_ssm = fwd.inputs[4];
    std::string D_param = fwd.inputs[5];
    std::string dt_bias = (fwd.inputs.size() > 6) ? fwd.inputs[6] : "";

    std::string u_ref = ctx.is_param(u) ? u : saved_ref(u);
    std::string delta_ref = ctx.is_param(delta) ? delta : saved_ref(delta);
    std::string A_log_ref = ctx.is_param(A_log) ? A_log : saved_ref(A_log);
    std::string B_ssm_ref = ctx.is_param(B_ssm) ? B_ssm : saved_ref(B_ssm);
    std::string C_ssm_ref = ctx.is_param(C_ssm) ? C_ssm : saved_ref(C_ssm);
    std::string D_param_ref = ctx.is_param(D_param) ? D_param : saved_ref(D_param);
    std::string dt_bias_ref = dt_bias.empty() ? "" : (ctx.is_param(dt_bias) ? dt_bias : saved_ref(dt_bias));

    // ssm_state is the second output from forward, referenced via saved
    std::string ssm_state_ref = saved_ref(fwd.outputs[1]);

    std::vector<std::string> inputs = {ctx.d_output,  // d_out
                                       u_ref,
                                       delta_ref,
                                       A_log_ref,
                                       B_ssm_ref,
                                       C_ssm_ref,
                                       D_param_ref,
                                       dt_bias_ref,
                                       ssm_state_ref};

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // du
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // ddelta
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");  // dA_log
    outputs.push_back(ctx.needs_grad(3) ? ctx.d_inputs[3] : "");  // dB
    outputs.push_back(ctx.needs_grad(4) ? ctx.d_inputs[4] : "");  // dC
    outputs.push_back(ctx.needs_grad(5) ? ctx.d_inputs[5] : "");  // dD
    if (fwd.inputs.size() > 6 && ctx.needs_grad(6)) {
        outputs.push_back(ctx.d_inputs[6]);  // ddelta_bias
    }

    AttrMap attrs =
        copy_attrs(fwd.attrs,
                   {"num_heads", "head_dim", "chunk_size", "ssm_state_size", "n_groups", "intermediate_size"},
                   "mamba_ssm_scan");

    ops.push_back(make_operation("mamba_ssm_scan_backward_" + std::to_string(ctx.op_counter++),
                                 "mamba_ssm_scan_backward",
                                 "mamba_ssm_scan_backward",
                                 inputs,
                                 outputs,
                                 attrs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("mamba_ssm_scan", ::dsl::mamba_ssm_scan_backward);
