#include "runtime/dsl/compiled_ops.h"

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

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_mrope(const CompiledOp& op) {
    Tensor& qkv_in = resolve_tensor(op.inputs[0]);
    Tensor& freqs = resolve_tensor(op.inputs[1]);
    Tensor& pos_ids = resolve_tensor(op.inputs[2]);

    Tensor& qkv_out = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    if (qkv_in.Data != qkv_out.Data) {
        CUDA_CHECK(cudaMemcpyAsync(qkv_out.Data, qkv_in.Data,
                                   qkv_in.bytes(), cudaMemcpyDeviceToDevice, mRunState.MainStream));
    }

    Tensor qkv_view = (qkv_out.Rank == 4) ? view_tensor(qkv_out, {mB, mT, qkv_channels}) : qkv_out;
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

    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const int qkv_channels = Hs * (Hq + 2 * Hkv);

    Tensor d_out_view = (d_out.Rank == 4) ? view_tensor(d_out, {mB, mT, static_cast<long>(qkv_channels)}) : d_out;
    Tensor d_qkv_view = (d_qkv.Rank == 4) ? view_tensor(d_qkv, {mB, mT, static_cast<long>(qkv_channels)}) : d_qkv;

    if (d_qkv_view.Data != d_out_view.Data) {
        const std::size_t bytes = static_cast<std::size_t>(d_out_view.nelem()) * get_dtype_size(d_out_view.DType);
        CUDA_CHECK(cudaMemcpyAsync(d_qkv_view.Data, d_out_view.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));
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

    mrope_backward(d_qkv_view, d_qkv_view, freqs, pos_ptr, pos_planes,
                   op.attrs.mrope_section[0], op.attrs.mrope_section[1], op.attrs.mrope_section[2],
                   nullptr, static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs, rotary_dim,
                   mRunState.MainStream);
}

}  // namespace dsl
