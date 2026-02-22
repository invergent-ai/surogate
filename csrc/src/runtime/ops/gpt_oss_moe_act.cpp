#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_gpt_oss_moe_act(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    float alpha = (op.attrs.gpt_oss_alpha > 0.0f) ? op.attrs.gpt_oss_alpha : 1.702f;
    float limit = (op.attrs.gpt_oss_limit > 0.0f) ? op.attrs.gpt_oss_limit : 7.0f;

    if (inp.Rank == 2) {
        const long N = inp.Sizes[0];
        const long D = inp.Sizes[1] / 2;
        std::vector<long> out_shape = {N, D};
        Tensor out = mRunState.temp_alloc(inp.DType, out_shape);
        mTemps.push_back(out);

        if (inp.DType == ETensorDType::BF16) {
            gpt_oss_moe_act_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(),
                                    static_cast<int>(N), static_cast<int>(D),
                                    alpha, limit, mRunState.MainStream);
        } else if (inp.DType == ETensorDType::FP32) {
            gpt_oss_moe_act_forward(out.get<float>(), inp.get<float>(),
                                    static_cast<int>(N), static_cast<int>(D),
                                    alpha, limit, mRunState.MainStream);
        } else {
            throw std::logic_error("gpt_oss_moe_act: unsupported input dtype");
        }

        store_tensor(op.outputs[0], out);

        return;
    }

    Tensor& out = ensure_output_tensor(op.outputs[0]);
    const long B = inp.Sizes[0];
    const long T = inp.Sizes[1];
    const long D = inp.Sizes[2] / 2;
    const int N = static_cast<int>(B * T);

    if (inp.DType == ETensorDType::BF16) {
        gpt_oss_moe_act_forward(out.get<nv_bfloat16>(), inp.get<nv_bfloat16>(),
                                N, static_cast<int>(D),
                                alpha, limit, mRunState.MainStream);
    } else if (inp.DType == ETensorDType::FP32) {
        gpt_oss_moe_act_forward(out.get<float>(), inp.get<float>(),
                                N, static_cast<int>(D),
                                alpha, limit, mRunState.MainStream);
    } else {
        throw std::logic_error("gpt_oss_moe_act: unsupported input dtype");
    }
}

void CompiledExecutor::dispatch_gpt_oss_moe_act_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor* d_inp_ptr = nullptr;
    Tensor d_inp_local;

    float alpha = (op.attrs.gpt_oss_alpha > 0.0f) ? op.attrs.gpt_oss_alpha : 1.702f;
    float limit = (op.attrs.gpt_oss_limit > 0.0f) ? op.attrs.gpt_oss_limit : 7.0f;

    if (d_out.Rank == 2) {
        const long N = d_out.Sizes[0];
        const long D = d_out.Sizes[1];
        const long expected_inp = N * D * 2;
        if (inp.nelem() != expected_inp) {
            std::ostringstream oss;
            oss << "gpt_oss_moe_act_backward: shape mismatch: d_out=[" << N << "," << D
                << "] inp_nelem=" << inp.nelem();
            throw std::runtime_error(oss.str());
        }
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
        if (static_cast<long>(d_inp_ptr->nelem()) != expected_inp) {
            d_inp_local = mRunState.temp_alloc(inp.DType, {N, D * 2});
            mTemps.push_back(d_inp_local);
            store_tensor(op.outputs[0], d_inp_local);
            d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
        }
        Tensor& d_inp = *d_inp_ptr;
        if (d_out.DType == ETensorDType::BF16) {
            gpt_oss_moe_act_backward(d_inp.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(),
                                     inp.get<nv_bfloat16>(),
                                     static_cast<int>(N), static_cast<int>(D),
                                     alpha, limit, mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP32) {
            gpt_oss_moe_act_backward(d_inp.get<float>(), d_out.get<float>(),
                                     inp.get<float>(),
                                     static_cast<int>(N), static_cast<int>(D),
                                     alpha, limit, mRunState.MainStream);
        } else {
            throw std::logic_error("gpt_oss_moe_act_backward: unsupported dtype");
        }
        return;
    }

    const long D = d_out.Sizes[2];
    const int N = static_cast<int>(d_out.Sizes[0] * d_out.Sizes[1]);
    d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    const long expected_inp = static_cast<long>(N) * D * 2;
    if (static_cast<long>(d_inp_ptr->nelem()) != expected_inp) {
        d_inp_local = mRunState.temp_alloc(inp.DType, {d_out.Sizes[0], d_out.Sizes[1], D * 2});
        mTemps.push_back(d_inp_local);
        store_tensor(op.outputs[0], d_inp_local);
        d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
    }
    Tensor& d_inp = *d_inp_ptr;
    if (d_out.DType == ETensorDType::BF16) {
        gpt_oss_moe_act_backward(d_inp.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(),
                                 inp.get<nv_bfloat16>(),
                                 N, static_cast<int>(D),
                                 alpha, limit, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        gpt_oss_moe_act_backward(d_inp.get<float>(), d_out.get<float>(),
                                 inp.get<float>(),
                                 N, static_cast<int>(D),
                                 alpha, limit, mRunState.MainStream);
    } else {
        throw std::logic_error("gpt_oss_moe_act_backward: unsupported dtype");
    }
}

}  // namespace dsl
