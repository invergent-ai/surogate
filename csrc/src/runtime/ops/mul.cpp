#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mul(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    if (a.DType != b.DType) {
        throw std::runtime_error("dispatch_mul: a and b must have the same dtype");
    }

    // Check for broadcast: (N, M) * (N, 1) or (N, 1) * (N, M)
    // The larger tensor determines the output shape.
    Tensor* big = &a;
    Tensor* small = &b;
    bool broadcast = false;
    if (a.nelem() != b.nelem()) {
        // Try to identify broadcast pattern: one tensor has trailing dim 1
        if (a.nelem() > b.nelem()) {
            big = &a;
            small = &b;
        } else {
            big = &b;
            small = &a;
        }
        // Validate: small must broadcast to big
        if (small->nelem() == 1) {
            // Scalar broadcast: out = big * scalar
            broadcast = true;
        } else if (big->Rank == 2 && small->Rank == 2 && big->Sizes[0] == small->Sizes[0] && small->Sizes[1] == 1) {
            broadcast = true;
        } else {
            throw std::runtime_error("dispatch_mul: shape mismatch and unsupported broadcast pattern"
                                     " (a.nelem=" +
                                     std::to_string(a.nelem()) + ", b.nelem=" + std::to_string(b.nelem()) + ")");
        }
    }

    const Tensor& bigger = *big;
    std::vector<long> out_shape(bigger.Sizes.begin(), bigger.Sizes.begin() + bigger.Rank);
    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.nelem() != bigger.nelem() || out.DType != bigger.DType) {
        out = mRunState.temp_alloc(bigger.DType, out_shape, "mul_out");
        mTemps.push_back(out);
    }

    if (!broadcast) {
        // Standard element-wise multiply
        const long n = static_cast<long>(a.nelem());
        if (a.DType == ETensorDType::BF16) {
            elementwise_mul(out.get<nv_bfloat16>(),
                            a.get<nv_bfloat16>(),
                            b.get<nv_bfloat16>(),
                            n,
                            mRunState.MainStream);
        } else if (a.DType == ETensorDType::FP16) {
            elementwise_mul(out.get<half>(), a.get<half>(), b.get<half>(), n, mRunState.MainStream);
        } else if (a.DType == ETensorDType::FP32) {
            elementwise_mul(out.get<float>(), a.get<float>(), b.get<float>(), n, mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul: unsupported dtype");
        }
    } else if (small->nelem() == 1) {
        // Scalar broadcast: out = big * scalar. The scalar lives on device;
        // copying it to host here would sync the stream (fine eagerly, fatal
        // under CUDA graph capture — cudaStreamSynchronize is unsupported on
        // a capturing stream). Reuse scale_rows with N=1 so the kernel reads
        // the scalar via device pointer and we never touch the host.
        const long n = static_cast<long>(big->nelem());
        if (big->DType == ETensorDType::BF16) {
            scale_rows(out.get<nv_bfloat16>(),
                       big->get<nv_bfloat16>(),
                       small->get<nv_bfloat16>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else if (big->DType == ETensorDType::FP16) {
            scale_rows(out.get<half>(),
                       big->get<half>(),
                       small->get<half>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else if (big->DType == ETensorDType::FP32) {
            scale_rows(out.get<float>(),
                       big->get<float>(),
                       small->get<float>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul scalar: unsupported dtype");
        }
    } else {
        // Row broadcast: (N, M) * (N, 1) → (N, M)
        const long N = bigger.Sizes[0];
        const long M = bigger.Sizes[1];
        if (bigger.DType == ETensorDType::BF16) {
            scale_rows(out.get<nv_bfloat16>(),
                       big->get<nv_bfloat16>(),
                       small->get<nv_bfloat16>(),
                       N,
                       M,
                       mRunState.MainStream);
        } else if (bigger.DType == ETensorDType::FP16) {
            scale_rows(out.get<half>(), big->get<half>(), small->get<half>(), N, M, mRunState.MainStream);
        } else if (bigger.DType == ETensorDType::FP32) {
            scale_rows(out.get<float>(), big->get<float>(), small->get<float>(), N, M, mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul broadcast: unsupported dtype");
        }
    }
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_mul_backward(const CompiledOp& op) {
    // Inputs: d_out, a, b
    // Outputs: d_a, d_b
    // Forward was: out = a * b (element-wise or broadcast)
    // Backward:   d_a = d_out * b,  d_b = d_out * a
    // With broadcast (N,M)*(N,1): d_data = d_out * scale (row-scale)
    //                              d_scale = sum(d_out * data, dim=-1)
    if (op.inputs.size() < 3) {
        throw std::runtime_error("dispatch_mul_backward: expected inputs (d_out, a, b)");
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);

    if (d_out.DType != a.DType || d_out.DType != b.DType) {
        throw std::runtime_error("dispatch_mul_backward: dtype mismatch between d_out/a/b");
    }

    // Detect broadcast: one of a,b has nelem=1 (scalar) or trailing dim 1
    bool broadcast = false;
    bool scalar_broadcast = false;
    Tensor* big = &a;
    Tensor* small = &b;
    int big_input_idx = 0;
    if (a.nelem() != b.nelem()) {
        if (a.nelem() < b.nelem()) {
            big = &b;
            small = &a;
            big_input_idx = 1;
        }
        if (small->nelem() == 1) {
            broadcast = true;
            scalar_broadcast = true;
        } else if (big->Rank == 2 && small->Rank == 2 && big->Sizes[0] == small->Sizes[0] && small->Sizes[1] == 1) {
            broadcast = true;
        } else {
            std::ostringstream oss;
            oss << "dispatch_mul_backward: unsupported broadcast pattern. "
                << "a.name=" << op.inputs[1].name << " a.shape=[";
            for (int i = 0; i < a.Rank; ++i)
                oss << a.Sizes[i] << (i + 1 < a.Rank ? "," : "");
            oss << "] b.name=" << op.inputs[2].name << " b.shape=[";
            for (int i = 0; i < b.Rank; ++i)
                oss << b.Sizes[i] << (i + 1 < b.Rank ? "," : "");
            oss << "]";
            throw std::runtime_error(oss.str());
        }
    }

    auto allocate_like = [&](std::size_t out_idx, const Tensor& like) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            if (out_ref.nelem() == like.nelem() && out_ref.DType == like.DType) {
                return out_ref;
            }
        }
        std::vector<long> shape(like.Sizes.begin(), like.Sizes.begin() + like.Rank);
        Tensor out = mRunState.temp_alloc(like.DType, shape, "mul_backward_out");
        mTemps.push_back(out);
        return out;
    };

    if (!broadcast) {
        // Standard element-wise backward: d_a = d_out * b, d_b = d_out * a
        Tensor d_a = allocate_like(0, a);
        Tensor d_b = allocate_like(1, b);
        const long n = static_cast<long>(d_out.nelem());
        if (d_out.DType == ETensorDType::BF16) {
            elementwise_mul(d_a.get<nv_bfloat16>(),
                            d_out.get<nv_bfloat16>(),
                            b.get<nv_bfloat16>(),
                            n,
                            mRunState.MainStream);
            elementwise_mul(d_b.get<nv_bfloat16>(),
                            d_out.get<nv_bfloat16>(),
                            a.get<nv_bfloat16>(),
                            n,
                            mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            elementwise_mul(d_a.get<half>(), d_out.get<half>(), b.get<half>(), n, mRunState.MainStream);
            elementwise_mul(d_b.get<half>(), d_out.get<half>(), a.get<half>(), n, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP32) {
            elementwise_mul(d_a.get<float>(), d_out.get<float>(), b.get<float>(), n, mRunState.MainStream);
            elementwise_mul(d_b.get<float>(), d_out.get<float>(), a.get<float>(), n, mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul_backward: unsupported dtype");
        }
        if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], d_a);
        if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], d_b);
    } else if (scalar_broadcast) {
        // Scalar broadcast backward: forward was big * scalar -> out
        // d_big = d_out * scalar, d_scalar = sum(d_out * big). Same constraint
        // as the forward path — the scalar lives on device and we must NOT
        // sync the stream to read it (capture-unsafe). scale_rows with N=1
        // does the d_big multiply via device pointer.
        int d_big_idx = big_input_idx;
        int d_small_idx = 1 - big_input_idx;
        Tensor d_big = allocate_like(static_cast<std::size_t>(d_big_idx), *big);
        const long n = static_cast<long>(big->nelem());
        if (d_out.DType == ETensorDType::BF16) {
            scale_rows(d_big.get<nv_bfloat16>(),
                       d_out.get<nv_bfloat16>(),
                       small->get<nv_bfloat16>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            scale_rows(d_big.get<half>(),
                       d_out.get<half>(),
                       small->get<half>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP32) {
            scale_rows(d_big.get<float>(),
                       d_out.get<float>(),
                       small->get<float>(),
                       /*N=*/1,
                       /*M=*/n,
                       mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul_backward scalar: unsupported dtype");
        }
        if (op.outputs.size() > static_cast<std::size_t>(d_big_idx) && !op.outputs[d_big_idx].name.empty())
            store_tensor(op.outputs[d_big_idx], d_big);
        // d_scalar: skip for frozen layer_scalar (gradient output is usually empty)
        if (op.outputs.size() > static_cast<std::size_t>(d_small_idx) && !op.outputs[d_small_idx].name.empty()) {
            Tensor d_small = allocate_like(static_cast<std::size_t>(d_small_idx), *small);
            // d_scalar = sum(d_out * big) — for frozen scalars this is unused
            cudaMemsetAsync(d_small.Data, 0, d_small.bytes(), mRunState.MainStream);
            store_tensor(op.outputs[d_small_idx], d_small);
        }
    } else {
        // Row broadcast backward: forward was big(N,M) * small(N,1) -> out(N,M)
        // d_big  = d_out * small  (row-scale: same as forward broadcast)
        // d_small = sum(d_out * big, dim=-1)  (reduce to (N,1))
        const long N = big->Sizes[0];
        const long M = big->Sizes[1];

        // d_big output index: same as big_input_idx (a=0, b=1)
        int d_big_idx = big_input_idx;
        int d_small_idx = 1 - big_input_idx;

        Tensor d_big = allocate_like(static_cast<std::size_t>(d_big_idx), *big);
        Tensor d_small = allocate_like(static_cast<std::size_t>(d_small_idx), *small);

        if (d_out.DType == ETensorDType::BF16) {
            scale_rows(d_big.get<nv_bfloat16>(),
                       d_out.get<nv_bfloat16>(),
                       small->get<nv_bfloat16>(),
                       N,
                       M,
                       mRunState.MainStream);
            reduce_row_mul(d_small.get<nv_bfloat16>(),
                           d_out.get<nv_bfloat16>(),
                           big->get<nv_bfloat16>(),
                           N,
                           M,
                           mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP16) {
            scale_rows(d_big.get<half>(), d_out.get<half>(), small->get<half>(), N, M, mRunState.MainStream);
            reduce_row_mul(d_small.get<half>(), d_out.get<half>(), big->get<half>(), N, M, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP32) {
            scale_rows(d_big.get<float>(), d_out.get<float>(), small->get<float>(), N, M, mRunState.MainStream);
            reduce_row_mul(d_small.get<float>(), d_out.get<float>(), big->get<float>(), N, M, mRunState.MainStream);
        } else {
            throw std::runtime_error("dispatch_mul_backward broadcast: unsupported dtype");
        }

        if (op.outputs.size() > static_cast<std::size_t>(d_big_idx) && !op.outputs[d_big_idx].name.empty())
            store_tensor(op.outputs[d_big_idx], d_big);
        if (op.outputs.size() > static_cast<std::size_t>(d_small_idx) && !op.outputs[d_small_idx].name.empty())
            store_tensor(op.outputs[d_small_idx], d_small);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Multiply backward rule
// Forward: C = A * B (elementwise)
// Backward: dA = dC * B, dB = dC * A
// -----------------------------------------------------------------------------
std::vector<Operation> multiply_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    const std::string& A = fwd.inputs[0];
    const std::string& B = fwd.inputs[1];

    if (ctx.needs_grad(0) || ctx.needs_grad(1)) {
        std::string a_ref = ctx.is_param(A) ? A : saved_ref(A);
        std::string b_ref = ctx.is_param(B) ? B : saved_ref(B);
        ops.push_back(
            make_operation("mul_backward_" + std::to_string(ctx.op_counter++),
                           "mul_backward",
                           "mul_backward",
                           {ctx.d_output, a_ref, b_ref},
                           {ctx.needs_grad(0) ? ctx.d_inputs[0] : "", ctx.needs_grad(1) ? ctx.d_inputs[1] : ""}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("multiply", ::dsl::multiply_backward);
REGISTER_AUTODIFF("mul", ::dsl::multiply_backward);
