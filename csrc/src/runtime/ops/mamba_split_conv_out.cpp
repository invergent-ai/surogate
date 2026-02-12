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

    auto shape_matches = [](const TensorRef& ref, long expected) -> bool {
        if (ref.shape.empty()) return false;
        long prod = 1;
        for (auto d : ref.shape) {
            if (d <= 0) return false;
            prod *= d;
        }
        return prod == expected;
    };

    const long u_expected = static_cast<long>(B) * D * T;
    const long bc_expected = static_cast<long>(B) * groups * dstate * T;

    Tensor* u_ptr = nullptr;
    Tensor* b_ptr = nullptr;
    Tensor* c_ptr = nullptr;

    if (shape_matches(op.outputs[0], u_expected)) {
        Tensor& u_ref = ensure_output_tensor(op.outputs[0]);
        if (u_ref.nelem() == u_expected) {
            u_ptr = &u_ref;
        }
    }
    if (!u_ptr) {
        Tensor u = mRunState.temp_alloc(conv_out.DType, {B, D, T});
        mTemps.push_back(u);
        u_ptr = &mTemps.back();
    }

    if (shape_matches(op.outputs[1], bc_expected)) {
        Tensor& b_ref = ensure_output_tensor(op.outputs[1]);
        if (b_ref.nelem() == bc_expected) {
            b_ptr = &b_ref;
        }
    }
    if (!b_ptr) {
        Tensor B_ssm = mRunState.temp_alloc(conv_out.DType, {B, groups, dstate, T});
        mTemps.push_back(B_ssm);
        b_ptr = &mTemps.back();
    }

    if (shape_matches(op.outputs[2], bc_expected)) {
        Tensor& c_ref = ensure_output_tensor(op.outputs[2]);
        if (c_ref.nelem() == bc_expected) {
            c_ptr = &c_ref;
        }
    }
    if (!c_ptr) {
        Tensor C_ssm = mRunState.temp_alloc(conv_out.DType, {B, groups, dstate, T});
        mTemps.push_back(C_ssm);
        c_ptr = &mTemps.back();
    }

    // Call kernel
    mamba_split_conv_out(*u_ptr, *b_ptr, *c_ptr, conv_out,
                         B, T, D, groups, dstate,
                         mRunState.MainStream);

    store_tensor(op.outputs[0], *u_ptr);
    store_tensor(op.outputs[1], *b_ptr);
    store_tensor(op.outputs[2], *c_ptr);
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
