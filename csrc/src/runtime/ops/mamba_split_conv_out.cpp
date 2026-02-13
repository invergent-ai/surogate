// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba split_conv_out operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_split_conv_out(const CompiledOp& op) {
    // Input: conv_out [B, conv_dim, T] where conv_dim = D + 2*groups*dstate
    // Outputs: u [B, D, T], B_ssm [B, groups, dstate, T], C_ssm [B, groups, dstate, T]
    Tensor& conv_out = resolve_tensor(op.inputs[0]);

    const int B = static_cast<int>(conv_out.Sizes[0]);
    const int conv_dim = static_cast<int>(conv_out.Sizes[1]);
    const int T = static_cast<int>(conv_out.Sizes[2]);

    // Get dimensions from attributes
    const int intermediate_size = op.attrs.intermediate_size;
    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;

    // D = intermediate_size (mamba_num_heads * mamba_head_dim)
    const int D = intermediate_size;

    // Allocate all output tensors upfront (before any mTemps.push_back)
    // to avoid dangling pointers from vector reallocation.
    Tensor u_t = mRunState.temp_alloc(conv_out.DType, {B, D, T});
    Tensor b_t = mRunState.temp_alloc(conv_out.DType, {B, groups, dstate, T});
    Tensor c_t = mRunState.temp_alloc(conv_out.DType, {B, groups, dstate, T});
    mTemps.push_back(u_t);
    mTemps.push_back(b_t);
    mTemps.push_back(c_t);

    // Call kernel
    mamba_split_conv_out(u_t, b_t, c_t, conv_out,
                         B, T, D, groups, dstate,
                         mRunState.MainStream);

    store_tensor(op.outputs[0], u_t);
    store_tensor(op.outputs[1], b_t);
    store_tensor(op.outputs[2], c_t);
}

void CompiledExecutor::dispatch_mamba_split_conv_out_backward(const CompiledOp& op) {
    // Inputs: d_u [B, D, T], d_B [B, groups, dstate, T], d_C [B, groups, dstate, T]
    // Output: d_conv_out [B, conv_dim, T]
    Tensor& d_u = resolve_tensor(op.inputs[0]);
    Tensor& d_B = resolve_tensor(op.inputs[1]);
    Tensor& d_C = resolve_tensor(op.inputs[2]);

    const int groups = op.attrs.n_groups;
    const int dstate = op.attrs.ssm_state_size;
    const int D = op.attrs.intermediate_size;
    const int B = (mB > 0) ? static_cast<int>(mB) : static_cast<int>(d_u.Sizes[0]);
    const int T = (mT > 0) ? static_cast<int>(mT)
                           : static_cast<int>(d_u.nelem() / (static_cast<long>(B) * D));
    const int conv_dim = D + 2 * groups * dstate;

    // Allocate output
    Tensor d_conv_out = mRunState.temp_alloc(d_u.DType, {B, conv_dim, T});
    mTemps.push_back(d_conv_out);

    // Call kernel (d_B and d_C are expected to be FP32 from selective_scan backward)
    mamba_pack_conv_out(d_conv_out, d_u, d_B, d_C,
                        B, T, D, groups, dstate,
                        mRunState.MainStream);

    store_tensor(op.outputs[0], d_conv_out);
}

}  // namespace dsl
