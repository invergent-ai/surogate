#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_matmul_swiglu(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& up_out = ensure_output_tensor(op.outputs[1]);

    int M = 0, N = 0, K = 0;
    matmul_dims(a, b, op.attrs.transpose, M, N, K);
    const long D = N / 2;

    matmul(up_out, b, a, std::nullopt, nullptr, nullptr,
           mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
           N, M, K, swap_transpose(op.attrs.transpose), false, mRunState.MainStream);

    Tensor up_3d = view_tensor(up_out, {mB, mT, static_cast<long>(N)});
    Tensor out_3d = view_tensor(out, {mB, mT, D});
    swiglu_forward(out_3d, up_3d, nullptr, static_cast<int>(mB),
                   static_cast<int>(mT), static_cast<int>(D), mRunState.MainStream);
}

void CompiledExecutor::dispatch_matmul_swiglu_backward(const CompiledOp& op, const modules::BackwardHook* hook) {
    // Combined backward for matmul + swiglu (fused op in forward)
    // inputs: d_swiglu_out, ln2 (matmul input), mlp_up_weight, mlp_up (pre-swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);
    Tensor mlp_up = resolve_tensor(op.inputs[3]);

    const int layer_idx = op.attrs.layer_idx;

    // Recompute mlp_up if the saved tensor was stack-allocated and freed
    bool recomputed_mlp_up = false;
    if (!mlp_up.Data || (mRunState.Stack.owns(mlp_up.Data) && !mRunState.Stack.is_live(mlp_up.Data))) {
        int M = 0, N = 0, K = 0;
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});
        matmul_dims(inp_flat, weight, op.attrs.transpose, M, N, K);
        const long D2 = N;
        Tensor mlp_up_flat = mRunState.temp_alloc(inp.DType, {mB * mT, D2});
        mTemps.push_back(mlp_up_flat);

        EMMTranspose mode_col = swap_transpose(op.attrs.transpose);
        matmul(mlp_up_flat, weight, inp_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);

        mlp_up = view_tensor(mlp_up_flat, {mB, mT, D2});
        if (layer_idx >= 0) {
            auto& acts = mRunState.simplified_acts(layer_idx);
            acts.mlp_up.Data = mlp_up.Data;
        }
        recomputed_mlp_up = true;
    }

    // First: swiglu backward
    Tensor* d_mlp_up_ptr = nullptr;
    if (layer_idx >= 0) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        d_mlp_up_ptr = &grads.d_mlp_up;
        if (!d_mlp_up_ptr->Data) {
            mRunState.temp_acquire(*d_mlp_up_ptr);
            mTemps.push_back(*d_mlp_up_ptr);
        }
    }
    Tensor d_mlp_up = d_mlp_up_ptr ? *d_mlp_up_ptr
                                   : mRunState.temp_alloc(mlp_up.DType, {mlp_up.Sizes[0], mlp_up.Sizes[1], mlp_up.Sizes[2]});
    if (!d_mlp_up_ptr) {
        mTemps.push_back(d_mlp_up);
    }

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    const long D = d_out.Sizes[2];
    swiglu_backward(d_mlp_up, d_out, mlp_up, abs_max_ptr,
                    static_cast<int>(d_out.Sizes[0]),
                    static_cast<int>(d_out.Sizes[1]),
                    static_cast<int>(D), mRunState.MainStream);

    // Then: matmul backward
    Tensor d_mlp_up_flat = view_tensor(d_mlp_up, {mB * mT, 2 * D});

    Tensor* d_inp_ptr = nullptr;
    Tensor* d_weight_ptr = nullptr;

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_weight_ptr = &ensure_output_tensor(op.outputs[1]);
    }

    bool do_accumulate = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        do_accumulate = (mAccumulateTensors.count(op.outputs[1].name) > 0);
        if (!do_accumulate) {
            if (auto base = base_param_from_grad(op.outputs[1].name)) {
                do_accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }
    }

    if (d_inp_ptr) {
        EMMTranspose mode_dA = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::NT:
                mode_dA = EMMTranspose::NN;
                break;
            case EMMTranspose::TN:
                mode_dA = EMMTranspose::NT;
                break;
            case EMMTranspose::TT:
                mode_dA = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(d_mlp_up_flat, weight, mode_dA, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_dA);
        matmul(*d_inp_ptr, weight, d_mlp_up_flat, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, false, mRunState.MainStream);
    }
    if (d_weight_ptr) {
        Tensor inp_flat = (inp.Rank == 2) ? inp : view_tensor(inp, {mB * mT, inp.Sizes[inp.Rank - 1]});

        const Tensor* lhs = nullptr;
        const Tensor* rhs = nullptr;
        EMMTranspose mode_rm = EMMTranspose::NN;
        switch (op.attrs.transpose) {
            case EMMTranspose::NN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::NT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TN;
                break;
            case EMMTranspose::TN:
                lhs = &inp_flat;
                rhs = &d_mlp_up_flat;
                mode_rm = EMMTranspose::NN;
                break;
            case EMMTranspose::TT:
                lhs = &d_mlp_up_flat;
                rhs = &inp_flat;
                mode_rm = EMMTranspose::TT;
                break;
        }

        int M = 0, N = 0, K = 0;
        matmul_dims(*lhs, *rhs, mode_rm, M, N, K);
        EMMTranspose mode_col = swap_transpose(mode_rm);
        matmul(*d_weight_ptr, *rhs, *lhs, std::nullopt, nullptr, nullptr,
               mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
               N, M, K, mode_col, do_accumulate, mRunState.MainStream);
    }

    if (layer_idx >= 0 && d_inp_ptr) {
        auto& grads = mRunState.simplified_grads(layer_idx);
        grads.d_ln2.Data = d_inp_ptr->Data;
    }

    // Hook invocation for LoRA backward (MLP up/gate)
    // Skip dense MLP hooks for MoE models - MoE has different backward path (grouped GEMM)
    const bool is_moe = mConfig.NumExperts > 0;
    if (hook && *hook && op.attrs.backward_hook_point.has_value() && !is_moe) {
        (*hook)(layer_idx, do_accumulate, mRunState.MainStream, *op.attrs.backward_hook_point, mHookContext);
    }
}

}  // namespace dsl
