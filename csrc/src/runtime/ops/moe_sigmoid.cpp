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

void CompiledExecutor::dispatch_moe_sigmoid(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Determine shape - input might have rank=0 if shape wasn't propagated at compile time
    // In MoE context, the input is router logits with shape [num_tokens, num_experts]
    std::vector<long> shape;
    if (inp.Rank == 2) {
        shape = {inp.Sizes[0], inp.Sizes[1]};
    } else if (inp.Rank == 0 && mConfig.NumExperts > 0) {
        // Infer shape from config and current dimensions
        const long num_tokens = mB * mT;
        const long num_experts = static_cast<long>(mConfig.NumExperts);
        shape = {num_tokens, num_experts};
        // Also fix the input tensor shape
        inp.Rank = 2;
        inp.Sizes[0] = num_tokens;
        inp.Sizes[1] = num_experts;
    } else {
        // Fallback to input shape if available
        for (int i = 0; i < inp.Rank; ++i) {
            shape.push_back(inp.Sizes[i]);
        }
    }

    const ETensorDType out_dtype = op.outputs[0].dtype;

    // Allocate output with same shape as input
    Tensor out = mRunState.temp_alloc(out_dtype, shape);
    mTemps.push_back(out);

    const int num_elements = static_cast<int>(out.nelem());

    if (out_dtype == inp.DType) {
        if (inp.DType == ETensorDType::BF16) {
            moe_sigmoid_forward(out.get<nv_bfloat16>(),
                                inp.get<nv_bfloat16>(),
                                num_elements, mRunState.MainStream);
        } else if (inp.DType == ETensorDType::FP32) {
            moe_sigmoid_forward(out.get<float>(),
                                inp.get<float>(),
                                num_elements, mRunState.MainStream);
        } else {
            throw std::logic_error("moe_sigmoid_forward: unsupported input dtype");
        }
    } else if (out_dtype == ETensorDType::FP32 && inp.DType == ETensorDType::BF16) {
        Tensor inp_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape);
        mTemps.push_back(inp_f32);
        convert_dtype(inp_f32.get<float>(), inp.get<nv_bfloat16>(),
                      num_elements, mRunState.MainStream);
        moe_sigmoid_forward(out.get<float>(),
                            inp_f32.get<float>(),
                            num_elements, mRunState.MainStream);
    } else if (out_dtype == ETensorDType::BF16 && inp.DType == ETensorDType::FP32) {
        Tensor out_f32 = mRunState.temp_alloc(ETensorDType::FP32, shape);
        mTemps.push_back(out_f32);
        moe_sigmoid_forward(out_f32.get<float>(),
                            inp.get<float>(),
                            num_elements, mRunState.MainStream);
        convert_dtype(out.get<nv_bfloat16>(), out_f32.get<float>(),
                      num_elements, mRunState.MainStream);
    } else {
        throw std::logic_error("moe_sigmoid_forward: unsupported dtype conversion");
    }

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_moe_sigmoid_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& sigmoid_out = resolve_tensor(op.inputs[1]);

    // Allocate output with same shape as d_out (not from compile-time inference)
    std::vector<long> d_inp_shape;
    for (int i = 0; i < d_out.Rank; ++i) {
        d_inp_shape.push_back(d_out.Sizes[i]);
    }
    const ETensorDType out_dtype = op.outputs[0].dtype;
    Tensor d_inp = mRunState.temp_alloc(out_dtype, d_inp_shape);
    mTemps.push_back(d_inp);

    const int num_elements = static_cast<int>(d_out.nelem());

    if (out_dtype == ETensorDType::FP32) {
        Tensor d_out_f32 = d_out;
        Tensor sig_f32 = sigmoid_out;
        if (d_out.DType == ETensorDType::BF16) {
            d_out_f32 = mRunState.temp_alloc(ETensorDType::FP32, d_inp_shape);
            mTemps.push_back(d_out_f32);
            convert_dtype(d_out_f32.get<float>(), d_out.get<nv_bfloat16>(),
                          num_elements, mRunState.MainStream);
        } else if (d_out.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_sigmoid_backward: unsupported d_out dtype");
        }
        if (sigmoid_out.DType == ETensorDType::BF16) {
            sig_f32 = mRunState.temp_alloc(ETensorDType::FP32, d_inp_shape);
            mTemps.push_back(sig_f32);
            convert_dtype(sig_f32.get<float>(), sigmoid_out.get<nv_bfloat16>(),
                          num_elements, mRunState.MainStream);
        } else if (sigmoid_out.DType != ETensorDType::FP32) {
            throw std::logic_error("moe_sigmoid_backward: unsupported sigmoid_out dtype");
        }
        moe_sigmoid_backward(d_inp.get<float>(),
                             d_out_f32.get<float>(),
                             sig_f32.get<float>(),
                             num_elements, mRunState.MainStream);
    } else if (out_dtype == ETensorDType::BF16) {
        if (d_out.DType == ETensorDType::BF16 && sigmoid_out.DType == ETensorDType::BF16) {
            moe_sigmoid_backward(d_inp.get<nv_bfloat16>(),
                                 d_out.get<nv_bfloat16>(),
                                 sigmoid_out.get<nv_bfloat16>(),
                                 num_elements, mRunState.MainStream);
        } else {
            Tensor d_out_f32 = d_out;
            Tensor sig_f32 = sigmoid_out;
            if (d_out.DType == ETensorDType::BF16) {
                d_out_f32 = mRunState.temp_alloc(ETensorDType::FP32, d_inp_shape);
                mTemps.push_back(d_out_f32);
                convert_dtype(d_out_f32.get<float>(), d_out.get<nv_bfloat16>(),
                              num_elements, mRunState.MainStream);
            } else if (d_out.DType != ETensorDType::FP32) {
                throw std::logic_error("moe_sigmoid_backward: unsupported d_out dtype");
            }
            if (sigmoid_out.DType == ETensorDType::BF16) {
                sig_f32 = mRunState.temp_alloc(ETensorDType::FP32, d_inp_shape);
                mTemps.push_back(sig_f32);
                convert_dtype(sig_f32.get<float>(), sigmoid_out.get<nv_bfloat16>(),
                              num_elements, mRunState.MainStream);
            } else if (sigmoid_out.DType != ETensorDType::FP32) {
                throw std::logic_error("moe_sigmoid_backward: unsupported sigmoid_out dtype");
            }
            Tensor d_inp_f32 = mRunState.temp_alloc(ETensorDType::FP32, d_inp_shape);
            mTemps.push_back(d_inp_f32);
            moe_sigmoid_backward(d_inp_f32.get<float>(),
                                 d_out_f32.get<float>(),
                                 sig_f32.get<float>(),
                                 num_elements, mRunState.MainStream);
            convert_dtype(d_inp.get<nv_bfloat16>(), d_inp_f32.get<float>(),
                          num_elements, mRunState.MainStream);
        }
    } else {
        throw std::logic_error("moe_sigmoid_backward: unsupported output dtype");
    }

    mTensorMap[op.outputs[0].name] = d_inp;
}


}  // namespace dsl
