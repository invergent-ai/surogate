#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

// Forward: y = rmsnorm(x, weight)
// Inputs: [0] x, [1] weight
// Outputs: [0] y, [1] rstd
void CompiledExecutor::dispatch_rmsnorm(const CompiledOp& op) {
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);

    // Infer shape from input x: y has same shape, rstd covers all rows.
    // C = last dimension (normalization axis).
    // rows = product of all other dimensions.
    // For 2D [rows, C]: total_rows = rows.
    // For 3D [B, T, C]: total_rows = B*T.
    const int C = static_cast<int>(x.Sizes[x.Rank - 1]);
    const int total_rows = static_cast<int>(x.nelem() / C);
    const float eps = op.attrs.eps;

    Tensor& y = ensure_output_tensor(op.outputs[0]);
    // rstd must be FP32; allocate temp if the slot is wrong dtype
    Tensor& rstd_ref = ensure_output_tensor(op.outputs[1]);
    Tensor rstd = rstd_ref;
    if (rstd.DType != ETensorDType::FP32 || rstd.nelem() != static_cast<std::size_t>(total_rows)) {
        rstd = mRunState.temp_alloc(ETensorDType::FP32, {total_rows}, "rmsnorm_rstd");
        mTemps.push_back(rstd);
    }

    // The pre-allocated slot may be larger than the input (e.g., V-norm
    // [B*T*Hkv, D] mapped to [B, T, C]).  The kernel only writes total_rows*C
    // elements, so an oversized buffer is safe.  If the buffer is too small
    // (shared-KV Q-norm), allocate a temp.
    if (y.nelem() < x.nelem()) {
        const std::vector<long> x_shape(x.Sizes.begin(), x.Sizes.begin() + x.Rank);
        y = mRunState.temp_alloc(x.DType, x_shape, "rmsnorm_y");
        mTemps.push_back(y);
    }

    rmsnorm_forward(y, rstd, x, weight, /*abs_max_ptr=*/nullptr, eps, total_rows, 1, C,
                    mRunState.MainStream);
    store_tensor(op.outputs[1], rstd);
}

// Backward: d_x = rmsnorm_backward(d_out, x, weight, rstd)
// Inputs: [0] d_out, [1] saved_x, [2] weight, [3] saved_rstd
// Outputs: [0] d_x, [1] d_weight (may be empty)
void CompiledExecutor::dispatch_rmsnorm_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);

    // Weight may be a forward-only "ones" tensor (V-norm uses ones as weight).
    // If it's not available, recreate it on the fly — it's just a tensor of 1.0s.
    Tensor weight_local{};
    Tensor* weight_ptr = nullptr;
    const std::string& weight_name = op.inputs[2].name;
    if (weight_name.find("ones") != std::string::npos) {
        // Allocate and fill with ones on demand
        const long C_ = static_cast<long>(x.Sizes[x.Rank - 1]);
        weight_local = mRunState.temp_alloc(x.DType, {C_}, "rmsnorm_bwd_ones");
        mTemps.push_back(weight_local);
        fill_constant(weight_local, 1.0f, static_cast<std::size_t>(C_), mRunState.MainStream);
        weight_ptr = &weight_local;
    } else {
        weight_ptr = &resolve_tensor(op.inputs[2]);
    }
    Tensor& weight = *weight_ptr;
    Tensor& rstd = resolve_tensor(op.inputs[3]);

    Tensor& d_x = ensure_output_tensor(op.outputs[0]);

    // d_weight accumulation
    Tensor d_weight_buf;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        d_weight_buf = ensure_output_tensor(op.outputs[1]);
    }

    const int C = static_cast<int>(d_out.Sizes[d_out.Rank - 1]);
    const int total_rows = static_cast<int>(d_out.nelem() / C);

    // Standalone rmsnorm backward: dresidual = zero (no upstream residual gradient)
    Tensor scratch = mRunState.temp_alloc(ETensorDType::BF16, {total_rows}, "rmsnorm_bwd_scratch");
    mTemps.push_back(scratch);

    Tensor zero_dresidual = mRunState.temp_alloc(d_out.DType, {total_rows, C}, "rmsnorm_bwd_zero_dres");
    mTemps.push_back(zero_dresidual);
    cudaMemsetAsync(zero_dresidual.Data, 0, zero_dresidual.bytes(), mRunState.MainStream);

    rmsnorm_backward(d_x, d_weight_buf, scratch, zero_dresidual, d_out, x, weight, rstd,
                     /*abs_max_ptr=*/nullptr, total_rows, 1, C,
                     mRunState.DeviceProp, mRunState.MainStream);
}

}  // namespace dsl
