// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GDN fused projection dispatch for Qwen3.5 linear-attention decode.
//
// Merges 4 matmuls into 2 by concatenating weights, then splits the
// merged outputs via zero-copy pointer views (no D2D copies).
//
// Savings: 2 fewer kernel launches per linear-attention layer per decode step.

#include "runtime/dsl/compiled_ops.h"
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace dsl {

void CompiledExecutor::dispatch_gdn_fused_proj(const CompiledOp& op) {
    if (op.inputs.size() < 5) {
        throw std::runtime_error("gdn_fused_proj: expected 5 inputs");
    }
    if (op.outputs.size() < 4) {
        throw std::runtime_error("gdn_fused_proj: expected 4 outputs");
    }

    Tensor& ln1_flat   = resolve_tensor(op.inputs[0]);
    Tensor& qkv_weight = resolve_tensor(op.inputs[1]);
    Tensor& z_weight   = resolve_tensor(op.inputs[2]);
    Tensor& b_weight   = resolve_tensor(op.inputs[3]);
    Tensor& a_weight   = resolve_tensor(op.inputs[4]);

    const long BT = ln1_flat.Sizes[0];
    const long C  = ln1_flat.Sizes[1];
    const long ConvDim = qkv_weight.Sizes[0];
    const long HvVd    = z_weight.Sizes[0];
    const long Hv_b    = b_weight.Sizes[0];
    const long Hv_a    = a_weight.Sizes[0];

    const long qkvz_out = ConvDim + HvVd;
    const long ba_out   = Hv_b + Hv_a;

    const long elem_size = get_dtype_size(qkv_weight.DType);

    // ── Lazily create cached merged weights (one-time cost) ──
    const long qkvz_bytes = qkvz_out * C * elem_size;
    if (!mGdnQkvzWeightData || mGdnQkvzWeightSize != qkvz_bytes) {
        if (mGdnQkvzWeightData) cudaFree(mGdnQkvzWeightData);
        cudaMalloc(&mGdnQkvzWeightData, qkvz_bytes);
        mGdnQkvzWeightSize = qkvz_bytes;

        const long qkv_bytes = ConvDim * C * elem_size;
        const long z_bytes   = HvVd * C * elem_size;
        cudaMemcpyAsync(mGdnQkvzWeightData, qkv_weight.Data,
                        qkv_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream);
        cudaMemcpyAsync(static_cast<std::byte*>(mGdnQkvzWeightData) + qkv_bytes,
                        z_weight.Data, z_bytes,
                        cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    const long ba_bytes = ba_out * C * elem_size;
    if (!mGdnBaWeightData || mGdnBaWeightSize != ba_bytes) {
        if (mGdnBaWeightData) cudaFree(mGdnBaWeightData);
        cudaMalloc(&mGdnBaWeightData, ba_bytes);
        mGdnBaWeightSize = ba_bytes;

        const long b_bytes = Hv_b * C * elem_size;
        const long a_bytes = Hv_a * C * elem_size;
        cudaMemcpyAsync(mGdnBaWeightData, b_weight.Data,
                        b_bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream);
        cudaMemcpyAsync(static_cast<std::byte*>(mGdnBaWeightData) + b_bytes,
                        a_weight.Data, a_bytes,
                        cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    // ── 2 merged matmuls (saves 2 kernel launches vs 4 separate) ──
    // Allocate merged output buffers
    Tensor qkvz_flat = mRunState.temp_alloc(ln1_flat.DType, {BT, qkvz_out}, "gdn_qkvz_flat");
    mTemps.push_back(qkvz_flat);

    Tensor ba_flat = mRunState.temp_alloc(ln1_flat.DType, {BT, ba_out}, "gdn_ba_flat");
    mTemps.push_back(ba_flat);

    Tensor qkvz_w;
    qkvz_w.Data = static_cast<std::byte*>(mGdnQkvzWeightData);
    qkvz_w.Sizes = {qkvz_out, C, 0, 0};
    qkvz_w.Rank = 2;
    qkvz_w.DType = qkv_weight.DType;

    Tensor ba_w;
    ba_w.Data = static_cast<std::byte*>(mGdnBaWeightData);
    ba_w.Sizes = {ba_out, C, 0, 0};
    ba_w.Rank = 2;
    ba_w.DType = b_weight.DType;

    // out = ln1_flat @ weight^T  (row-major NT → cuBLAS col-major TN)
    matmul(qkvz_flat, qkvz_w, ln1_flat, std::nullopt, nullptr, nullptr,
           mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
           qkvz_out, BT, C, EMMTranspose::TN, false, mRunState.MainStream);

    matmul(ba_flat, ba_w, ln1_flat, std::nullopt, nullptr, nullptr,
           mRunState.CublasLtHandle, mRunState.CuBlasWorkspace,
           ba_out, BT, C, EMMTranspose::TN, false, mRunState.MainStream);

    // ── Zero-copy split: pointer views into merged outputs ──
    // qkvz_flat row layout: [qkv (ConvDim) | z (HvVd)]
    // ba_flat row layout:   [b (Hv_b) | a (Hv_a)]
    //
    // For decode (BT=1), each row is contiguous, so we can create
    // view tensors pointing directly into the merged buffers.
    // No D2D copies, no additional kernel launches.

    Tensor& out_qkv = ensure_output_tensor(op.outputs[0]);
    Tensor& out_z   = ensure_output_tensor(op.outputs[1]);
    Tensor& out_b   = ensure_output_tensor(op.outputs[2]);
    Tensor& out_a   = ensure_output_tensor(op.outputs[3]);

    if (BT == 1) {
        // Zero-copy: point outputs directly into merged matmul results
        out_qkv.Data  = qkvz_flat.Data;
        out_qkv.Sizes = {1, ConvDim, 0, 0};
        out_qkv.Rank  = 2;
        out_qkv.DType = ln1_flat.DType;

        out_z.Data  = qkvz_flat.Data + ConvDim * elem_size;
        out_z.Sizes = {1, HvVd, 0, 0};
        out_z.Rank  = 2;
        out_z.DType = ln1_flat.DType;

        out_b.Data  = ba_flat.Data;
        out_b.Sizes = {1, Hv_b, 0, 0};
        out_b.Rank  = 2;
        out_b.DType = ln1_flat.DType;

        out_a.Data  = ba_flat.Data + Hv_b * elem_size;
        out_a.Sizes = {1, Hv_a, 0, 0};
        out_a.Rank  = 2;
        out_a.DType = ln1_flat.DType;
    } else {
        // General case (BT > 1): rows are strided, need actual copies
        cudaMemcpy2DAsync(
            out_qkv.Data, ConvDim * elem_size,
            qkvz_flat.Data, qkvz_out * elem_size,
            ConvDim * elem_size, BT,
            cudaMemcpyDeviceToDevice, mRunState.MainStream);
        cudaMemcpy2DAsync(
            out_z.Data, HvVd * elem_size,
            qkvz_flat.Data + ConvDim * elem_size, qkvz_out * elem_size,
            HvVd * elem_size, BT,
            cudaMemcpyDeviceToDevice, mRunState.MainStream);
        cudaMemcpy2DAsync(
            out_b.Data, Hv_b * elem_size,
            ba_flat.Data, ba_out * elem_size,
            Hv_b * elem_size, BT,
            cudaMemcpyDeviceToDevice, mRunState.MainStream);
        cudaMemcpy2DAsync(
            out_a.Data, Hv_a * elem_size,
            ba_flat.Data + Hv_b * elem_size, ba_out * elem_size,
            Hv_a * elem_size, BT,
            cudaMemcpyDeviceToDevice, mRunState.MainStream);
    }

    // Store output tensors so downstream ops can find them
    store_tensor(op.outputs[0], out_qkv);
    store_tensor(op.outputs[1], out_z);
    store_tensor(op.outputs[2], out_b);
    store_tensor(op.outputs[3], out_a);
}

} // namespace dsl
