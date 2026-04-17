#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_mrope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);

    const std::vector<long> qkv_shape(
        qkv_in.Sizes.begin(), qkv_in.Sizes.begin() + qkv_in.Rank);
    Tensor qkv_out = ensure_output_tensor_or_persistent(
        ensure_output_tensor(op.outputs[0]),
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".qkv_out",
        qkv_in.DType, qkv_shape, "mrope");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(qkv_in, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    if (qkv_in.Data != qkv_out.Data) {
        CUDA_CHECK(cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                                   qkv_in.bytes(), cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    Tensor qkv_view = qkv_out;
    const long needed = static_cast<long>(mB) * static_cast<long>(mT) * qkv_channels;
    if ((qkv_out.Rank == 4 || (qkv_out.Rank == 3 && qkv_out.Sizes[2] != qkv_channels)) &&
        static_cast<long>(qkv_out.nelem()) >= needed) {
        qkv_view = view_tensor(qkv_out, {mB, mT, qkv_channels});
    }
    int rotary_dim = op.attrs.rotary_dim;

    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_forward(qkv_view, qkv_view, freqs, pos_ptr, pos_planes,
                 op.attrs.mrope_section[0], op.attrs.mrope_section[1], op.attrs.mrope_section[2],
                 nullptr, static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim,
                 mRunState.MainStream);

    store_tensor(op.outputs[0], qkv_out);
}

void CompiledExecutor::dispatch_mrope_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // Allow inputs: [d_out, freqs, position_ids] or legacy [d_out, qkv, freqs, position_ids]
    const bool has_qkv = op.inputs.size() == 4;
    Tensor& freqs = resolve_tensor(op.inputs[has_qkv ? 2 : 1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[has_qkv ? 3 : 2]);

    const std::vector<long> d_qkv_shape(
        d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
    Tensor d_qkv = ensure_output_tensor_or_persistent(
        ensure_output_tensor(op.outputs[0]),
        mRunState, mMoeSavedBuffers, mMoeSavedSizes,
        op.op_id + "." + op.outputs[0].name + ".d_qkv",
        d_out.DType, d_qkv_shape, "mrope_backward");

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = derive_head_size(d_out, Hq, Hkv, static_cast<int>(mConfig.head_size()));
    const int qkv_channels = Hs * (Hq + 2 * Hkv);
    const int rotary_dim = op.attrs.rotary_dim;

    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    const int* pos_ptr = reinterpret_cast<int*>(pos_ids.Data);
    int pos_planes = 1;
    if (pos_ids.Rank == 3) {
        pos_planes = static_cast<int>(pos_ids.Sizes[0]);
        if (pos_planes == 4) {
            pos_ptr += static_cast<int>(mB * mT);
            pos_planes = 3;
        }
    }

    mrope_backward(d_qkv_view, d_qkv_view, freqs, pos_ptr, pos_planes,
                   op.attrs.mrope_section[0], op.attrs.mrope_section[1], op.attrs.mrope_section[2],
                   nullptr, static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim,
                   mRunState.MainStream);

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_qkv);
    }
}

}  // namespace dsl
