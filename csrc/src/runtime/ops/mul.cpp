#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mul(const CompiledOp& op) {
    Tensor& a = resolve_tensor(op.inputs[0]);
    Tensor& b = resolve_tensor(op.inputs[1]);
    if (a.nelem() != b.nelem()) {
        throw std::runtime_error("dispatch_mul: a and b must have the same number of elements");
    }
    if (a.DType != b.DType) {
        throw std::runtime_error("dispatch_mul: a and b must have the same dtype");
    }

    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.nelem() != a.nelem() || out.DType != a.DType) {
        std::vector<long> shape(a.Sizes.begin(), a.Sizes.begin() + a.Rank);
        out = mRunState.temp_alloc(a.DType, shape);
        mTemps.push_back(out);
    }

    const long n = static_cast<long>(a.nelem());
    if (a.DType == ETensorDType::BF16) {
        elementwise_mul(out.get<nv_bfloat16>(), a.get<nv_bfloat16>(), b.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
    } else if (a.DType == ETensorDType::FP16) {
        elementwise_mul(out.get<half>(), a.get<half>(), b.get<half>(),
                        n, mRunState.MainStream);
    } else if (a.DType == ETensorDType::FP32) {
        elementwise_mul(out.get<float>(), a.get<float>(), b.get<float>(),
                        n, mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_mul: unsupported dtype");
    }
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_mul_backward(const CompiledOp& op) {
    // Inputs: d_out, a, b
    // Outputs: d_a, d_b
    if (op.inputs.size() < 3) {
        throw std::runtime_error("dispatch_mul_backward: expected inputs (d_out, a, b)");
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& a = resolve_tensor(op.inputs[1]);
    Tensor& b = resolve_tensor(op.inputs[2]);
    if (a.nelem() != b.nelem() || d_out.nelem() != a.nelem()) {
        throw std::runtime_error("dispatch_mul_backward: all tensors must have the same number of elements");
    }

    auto allocate_like_input = [&](std::size_t out_idx, const Tensor& like) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            if (out_ref.nelem() == like.nelem() && out_ref.DType == like.DType) {
                return out_ref;
            }
        }
        std::vector<long> shape(like.Sizes.begin(), like.Sizes.begin() + like.Rank);
        Tensor out = mRunState.temp_alloc(like.DType, shape);
        mTemps.push_back(out);
        return out;
    };

    Tensor d_a = allocate_like_input(0, a);
    Tensor d_b = allocate_like_input(1, b);

    if (d_out.DType != a.DType || d_out.DType != b.DType) {
        throw std::runtime_error("dispatch_mul_backward: dtype mismatch between d_out/a/b");
    }

    const long n = static_cast<long>(d_out.nelem());
    if (d_out.DType == ETensorDType::BF16) {
        // d_a = d_out * b
        elementwise_mul(d_a.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), b.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
        // d_b = d_out * a
        elementwise_mul(d_b.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), a.get<nv_bfloat16>(),
                        n, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP16) {
        elementwise_mul(d_a.get<half>(), d_out.get<half>(), b.get<half>(),
                        n, mRunState.MainStream);
        elementwise_mul(d_b.get<half>(), d_out.get<half>(), a.get<half>(),
                        n, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        elementwise_mul(d_a.get<float>(), d_out.get<float>(), b.get<float>(),
                        n, mRunState.MainStream);
        elementwise_mul(d_b.get<float>(), d_out.get<float>(), a.get<float>(),
                        n, mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_mul_backward: unsupported dtype");
    }

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_a);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        store_tensor(op.outputs[1], d_b);
    }
}

}  // namespace dsl
