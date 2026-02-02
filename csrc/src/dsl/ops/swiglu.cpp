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

void CompiledExecutor::dispatch_swiglu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Handle both 3D [B, T, 2*D] and 2D [N, 2*D] tensors (MoE produces 2D)
    if (inp.Rank == 2) {
        // 2D input: [N, 2*D] -> [N, D] (used by MoE path)
        const long N = inp.Sizes[0];
        const long D = inp.Sizes[1] / 2;

        // MoE output shape is dynamic, allocate with runtime shape
        std::vector<long> out_shape = {N, D};
        Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
        mTemps.push_back(out);

        swiglu_forward(out, inp, nullptr, 1, static_cast<int>(N), static_cast<int>(D), mRunState.MainStream);

        // Store output in tensor map for subsequent ops
        mTensorMap[op.outputs[0].name] = out;
    } else {
        // 3D input: [B, T, 2*D] -> [B, T, D] (standard path)
        Tensor& out = ensure_output_tensor(op.outputs[0]);

        const long B = inp.Sizes[0];
        const long T = inp.Sizes[1];
        const long D = inp.Sizes[2] / 2;
        swiglu_forward(out, inp, nullptr, static_cast<int>(B),
                       static_cast<int>(T), static_cast<int>(D), mRunState.MainStream);
    }

}

void CompiledExecutor::dispatch_swiglu_backward(const CompiledOp& op) {
    // inputs: d_out, input (the mlp_up output before swiglu)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor& d_inp = ensure_output_tensor(op.outputs[0]);

    // For FP8 hybrid backward, record abs_max of d_mlp_up for subsequent quantization
    float* abs_max_ptr = mRunState.has_fp8_hybrid_backward()
        ? mRunState.simplified_quant_grads().d_mlp_up.abs_max()
        : nullptr;

    // Handle both 3D [B, T, D] and 2D [N, D] tensors (MoE produces 2D)
    if (d_out.Rank == 2) {
        // 2D case for MoE: d_out is [N, D], inp is [N, 2*D]
        const long N = d_out.Sizes[0];
        const long D = d_out.Sizes[1];
        const long expected_inp = N * D * 2;
        const long inp_nelem = static_cast<long>(inp.nelem());
        const long d_inp_nelem = static_cast<long>(d_inp.nelem());
        if (inp_nelem != expected_inp || d_inp_nelem != expected_inp) {
            std::ostringstream oss;
            oss << "swiglu_backward: shape mismatch for 2D tensors: "
                << "d_out=[" << N << "," << D << "]"
                << " inp_nelem=" << inp_nelem
                << " d_inp_nelem=" << d_inp_nelem
                << " expected_inp_nelem=" << expected_inp
                << " inp_shape=" << tensor_shape_str(inp)
                << " d_inp_shape=" << tensor_shape_str(d_inp)
                << " d_out_name=" << op.inputs[0].name
                << " inp_name=" << op.inputs[1].name
                << " out_name=" << op.outputs[0].name;
            throw std::runtime_error(oss.str());
        }
        swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                        1, static_cast<int>(N), static_cast<int>(D), mRunState.MainStream);
    } else {
        // 3D case: d_out is [B, T, D]
        const long D = d_out.Sizes[2];
        swiglu_backward(d_inp, d_out, inp, abs_max_ptr,
                        static_cast<int>(d_out.Sizes[0]),
                        static_cast<int>(d_out.Sizes[1]),
                        static_cast<int>(D), mRunState.MainStream);
    }
}

}  // namespace dsl
