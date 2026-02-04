#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Mirror modular path: always use cuDNN attention.
    attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                            mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                            Hq, Hkv, Hs, mRunState.MainStream);

}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && op.inputs.size() > 3) {
        std::string field;
        parse_block_param(op.inputs[3].name, layer_idx, field);
    }

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    const int attn_chunks = mOptions.AttBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const int chunk_B = (attn_chunks == 1)
        ? static_cast<int>(mB)
        : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

    if (attn_chunks == 1) {
        attention_backward_cudnn(d_qkv, lse, out, d_out, qkv,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
        return;
    }

    for (int chunk = 0; chunk < attn_chunks; ++chunk) {
        const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
        const long end = start + static_cast<long>(chunk_B);
        Tensor d_out_chunk = slice(d_out, 0, start, end);
        Tensor out_chunk = slice(out, 0, start, end);
        Tensor lse_chunk = slice(lse, 0, start, end);
        Tensor qkv_chunk = slice(qkv, 0, start, end);
        Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

        attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                 mRunState.scratch().cudnn_workspace,
                                 mRunState.CudnnHandle,
                                 static_cast<int>(chunk_B), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    }
}


}  // namespace dsl
