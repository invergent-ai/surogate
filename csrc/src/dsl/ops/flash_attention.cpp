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

void CompiledExecutor::dispatch_flash_attention(const CompiledOp& op) {
    Tensor& qkv = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    Tensor& lse = ensure_output_tensor(op.outputs[1]);

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool is_decode = (mT <= 1);
    const bool gqa_divisible = (Hkv > 0) ? (Hq % Hkv == 0) : false;
    const bool allow_gqa_cudnn = (!is_decode && gqa_divisible);
    const bool cudnn_gqa_ok = (Hq == Hkv) || allow_gqa_cudnn;
    const bool force_custom_fwd = (std::getenv("SUROGATE_ATTN_FWD_CUSTOM") != nullptr);

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }


    // cuDNN attention uses custom strides that map logical (B, Hq, T, HS) dims
    // to (B, T, Hq, HS) contiguous memory layout:
    //   Output strides: {Hq*HS*T, HS, Hq*HS, 1} for dims {B, Hq, T, HS}
    //   This maps element [b,h,t,s] to offset: b*Hq*HS*T + t*Hq*HS + h*HS + s
    //   Which is exactly (B, T, Hq, HS) contiguous layout.
    // DSL allocates output as (B, T, Hq*HS) = (B, T, Hq, HS) contiguous, so
    // we can pass it directly to cuDNN without any transpose.
    //
    // Similarly for QKV input: cuDNN expects (B, T, H, HS) contiguous where H = Hq + 2*Hkv.

    if (!cudnn_gqa_ok || force_custom_fwd) {
        attention_forward_custom(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                                mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                                Hq, Hkv, Hs, mRunState.MainStream);
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    Tensor* out_ptr = &out;
    Tensor* lse_ptr = &lse;
    Tensor* qkv_ptr = &qkv;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool is_decode = (mT <= 1);
    const bool gqa_divisible = (Hkv > 0) ? (Hq % Hkv == 0) : false;
    const bool allow_gqa_cudnn = (!is_decode && gqa_divisible);
    const bool cudnn_gqa_ok = (Hq == Hkv) || allow_gqa_cudnn;
    const bool force_custom_bwd = (std::getenv("SUROGATE_ATTN_BWD_CUSTOM") != nullptr);
    const bool force_cudnn_bwd = (std::getenv("SUROGATE_ATTN_BWD_FORCE_CUDNN") != nullptr);
    const bool use_cudnn_bwd = force_cudnn_bwd || (cudnn_gqa_ok && !force_custom_bwd);
    const bool gqa_fallback_full = !cudnn_gqa_ok;
    auto shape_vec = [](const Tensor& t) {
        return std::vector<long>(t.Sizes.begin(), t.Sizes.begin() + t.Rank);
    };

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Zero-initialize d_qkv before cuDNN attention backward to prevent NaN from uninitialized memory.
    // The d_qkv buffer may contain stale values from previous operations, and cuDNN attention backward
    // may read parts of this buffer even though it's expected to be output-only. Without this zero-init,
    // NaN values can appear in the gradient computation and propagate through the backward pass.
    fill_zero(d_qkv, mRunState.MainStream);

    const int attn_chunks = mOptions.AttBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const int chunk_B = (attn_chunks == 1)
        ? static_cast<int>(mB)
        : static_cast<int>(div_exact(mB, static_cast<long>(attn_chunks)));

    // Signature: attention_backward_cudnn(dqkv, stats, out, dout, qkv, workspace, handle, B, T, Hq, Hkv, HS, stream)
    if (attn_chunks == 1) {
        if (!use_cudnn_bwd) {
            if (d_out.DType == ETensorDType::BF16) {
                auto& scratch = mRunState.scratch();
                bool have_fallback_bufs =
                    scratch.attn_qkv_f32.Data && scratch.attn_out_f32.Data &&
                    scratch.attn_d_out_f32.Data && scratch.attn_d_qkv_f32.Data;
                if (have_fallback_bufs) {
                    const std::size_t need_qkv = static_cast<std::size_t>(qkv.nelem());
                    const std::size_t need_out = static_cast<std::size_t>(out.nelem());
                    const std::size_t need_d_out = static_cast<std::size_t>(d_out.nelem());
                    const std::size_t need_d_qkv = static_cast<std::size_t>(d_qkv.nelem());
                    const std::size_t have_qkv = static_cast<std::size_t>(scratch.attn_qkv_f32.nelem());
                    const std::size_t have_out = static_cast<std::size_t>(scratch.attn_out_f32.nelem());
                    const std::size_t have_d_out = static_cast<std::size_t>(scratch.attn_d_out_f32.nelem());
                    const std::size_t have_d_qkv = static_cast<std::size_t>(scratch.attn_d_qkv_f32.nelem());
                    const bool too_small =
                        (have_qkv < need_qkv) || (have_out < need_out) ||
                        (have_d_out < need_d_out) || (have_d_qkv < need_d_qkv);
                    if (too_small) {
                        if (mRunState.Allocator) {
                            if (have_qkv < need_qkv) {
                                scratch.attn_qkv_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_qkv_f32", EAllocationType::ON_DEVICE, shape_vec(qkv));
                            }
                            if (have_out < need_out) {
                                scratch.attn_out_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_out_f32", EAllocationType::ON_DEVICE, shape_vec(out));
                            }
                            if (have_d_out < need_d_out) {
                                scratch.attn_d_out_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_d_out_f32", EAllocationType::ON_DEVICE, shape_vec(d_out));
                            }
                            if (have_d_qkv < need_d_qkv) {
                                scratch.attn_d_qkv_f32 = mRunState.Allocator->allocate(
                                    ETensorDType::FP32, "attn_d_qkv_f32", EAllocationType::ON_DEVICE, shape_vec(d_qkv));
                            }
                        }
                        have_fallback_bufs =
                            scratch.attn_qkv_f32.nelem() >= static_cast<long>(need_qkv) &&
                            scratch.attn_out_f32.nelem() >= static_cast<long>(need_out) &&
                            scratch.attn_d_out_f32.nelem() >= static_cast<long>(need_d_out) &&
                            scratch.attn_d_qkv_f32.nelem() >= static_cast<long>(need_d_qkv);
                        if (!have_fallback_bufs) {
                        }
                    }
                }
                Tensor qkv_f32 = have_fallback_bufs ? view_tensor(scratch.attn_qkv_f32, shape_vec(qkv))
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(qkv), "qkv_f32");
                Tensor out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_out_f32, shape_vec(out))
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out), "attn_out_f32");
                Tensor d_out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_d_out_f32, shape_vec(d_out))
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_out), "d_attn_out_f32");
                Tensor d_qkv_f32 = have_fallback_bufs ? view_tensor(scratch.attn_d_qkv_f32, shape_vec(d_qkv))
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_qkv), "d_qkv_f32");
                convert_dtype(qkv_f32.get<float>(), qkv.get<nv_bfloat16>(), qkv.nelem(), mRunState.MainStream);
                convert_dtype(d_out_f32.get<float>(), d_out.get<nv_bfloat16>(), d_out.nelem(), mRunState.MainStream);
                // attention_backward_custom uses atomicAdd into d_qkv_f32; ensure it's zeroed.
                fill_zero(d_qkv_f32, mRunState.MainStream);

                if (gqa_fallback_full) {
                    Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                        ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_f32,
                                             static_cast<int>(mB), static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    convert_dtype(out_f32.get<float>(), out.get<nv_bfloat16>(), out.nelem(), mRunState.MainStream);
                    if (lse.DType == ETensorDType::BF16) {
                        Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                            ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                            : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                        convert_dtype(lse_f32.get<float>(), lse.get<nv_bfloat16>(), lse.nelem(), mRunState.MainStream);
                        attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                                  static_cast<int>(mB), static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    } else {
                        attention_backward_custom(d_qkv_f32, *lse_ptr, out_f32, d_out_f32, qkv_f32,
                                                  static_cast<int>(mB), static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    }
                }

                convert_dtype(d_qkv.get<nv_bfloat16>(), d_qkv_f32.get<float>(), d_qkv.nelem(), mRunState.MainStream);
            } else {
                if (gqa_fallback_full) {
                    auto& scratch = mRunState.scratch();
                    const bool have_fallback_bufs = scratch.attn_out_f32.Data && scratch.attn_lse_f32.Data;
                    Tensor out_f32 = have_fallback_bufs ? view_tensor(scratch.attn_out_f32, shape_vec(out))
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out), "attn_out_f32");
                    Tensor lse_f32 = have_fallback_bufs ? view_tensor(scratch.attn_lse_f32, shape_vec(lse))
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, *qkv_ptr,
                                             static_cast<int>(mB), static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv, lse_f32, out_f32, d_out, *qkv_ptr,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    attention_backward_custom(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                              static_cast<int>(mB), static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                }
            }
        } else {
            attention_backward_cudnn(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }
        return;
    }

    for (int chunk = 0; chunk < attn_chunks; ++chunk) {
        const long start = static_cast<long>(chunk) * static_cast<long>(chunk_B);
        const long end = start + static_cast<long>(chunk_B);
        Tensor d_out_chunk = slice(d_out, 0, start, end);
        Tensor out_chunk = slice(*out_ptr, 0, start, end);
        Tensor lse_chunk = slice(*lse_ptr, 0, start, end);
        Tensor qkv_chunk = slice(*qkv_ptr, 0, start, end);
        Tensor d_qkv_chunk = slice(d_qkv, 0, start, end);

        if (!use_cudnn_bwd) {
            if (d_out_chunk.DType == ETensorDType::BF16) {
                auto& scratch = mRunState.scratch();
                const bool have_fallback_bufs =
                    scratch.attn_qkv_f32.Data && scratch.attn_out_f32.Data &&
                    scratch.attn_d_out_f32.Data && scratch.attn_d_qkv_f32.Data;
                Tensor qkv_f32 = have_fallback_bufs ? slice(scratch.attn_qkv_f32, 0, start, end)
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(qkv_chunk), "qkv_f32");
                Tensor out_f32 = have_fallback_bufs ? slice(scratch.attn_out_f32, 0, start, end)
                                                    : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out_chunk), "attn_out_f32");
                Tensor d_out_f32 = have_fallback_bufs ? slice(scratch.attn_d_out_f32, 0, start, end)
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_out_chunk), "d_attn_out_f32");
                Tensor d_qkv_f32 = have_fallback_bufs ? slice(scratch.attn_d_qkv_f32, 0, start, end)
                                                      : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(d_qkv_chunk), "d_qkv_f32");
                convert_dtype(qkv_f32.get<float>(), qkv_chunk.get<nv_bfloat16>(), qkv_chunk.nelem(), mRunState.MainStream);
                convert_dtype(d_out_f32.get<float>(), d_out_chunk.get<nv_bfloat16>(), d_out_chunk.nelem(), mRunState.MainStream);

                if (gqa_fallback_full) {
                    Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                        ? slice(scratch.attn_lse_f32, 0, start, end)
                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_f32,
                                             chunk_B, static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    convert_dtype(out_f32.get<float>(), out_chunk.get<nv_bfloat16>(), out_chunk.nelem(), mRunState.MainStream);
                    if (lse_chunk.DType == ETensorDType::BF16) {
                        Tensor lse_f32 = have_fallback_bufs && scratch.attn_lse_f32.Data
                            ? slice(scratch.attn_lse_f32, 0, start, end)
                            : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                        convert_dtype(lse_f32.get<float>(), lse_chunk.get<nv_bfloat16>(), lse_chunk.nelem(), mRunState.MainStream);
                        attention_backward_custom(d_qkv_f32, lse_f32, out_f32, d_out_f32, qkv_f32,
                                                  chunk_B, static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    } else {
                        attention_backward_custom(d_qkv_f32, lse_chunk, out_f32, d_out_f32, qkv_f32,
                                                  chunk_B, static_cast<int>(mT),
                                                  Hq, Hkv, Hs, mRunState.MainStream);
                    }
                }

                convert_dtype(d_qkv_chunk.get<nv_bfloat16>(), d_qkv_f32.get<float>(), d_qkv_chunk.nelem(), mRunState.MainStream);
            } else {
                if (gqa_fallback_full) {
                    auto& scratch = mRunState.scratch();
                    const bool have_fallback_bufs = scratch.attn_out_f32.Data && scratch.attn_lse_f32.Data;
                    Tensor out_f32 = have_fallback_bufs ? slice(scratch.attn_out_f32, 0, start, end)
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(out_chunk), "attn_out_f32");
                    Tensor lse_f32 = have_fallback_bufs ? slice(scratch.attn_lse_f32, 0, start, end)
                                                        : mRunState.Stack.allocate(ETensorDType::FP32, shape_vec(lse_chunk), "attn_lse_f32");
                    attention_forward_custom(out_f32, lse_f32, qkv_chunk,
                                             chunk_B, static_cast<int>(mT),
                                             Hq, Hkv, Hs, mRunState.MainStream);
                    attention_backward_custom(d_qkv_chunk, lse_f32, out_f32, d_out_chunk, qkv_chunk,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                } else {
                    attention_backward_custom(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                              chunk_B, static_cast<int>(mT),
                                              Hq, Hkv, Hs, mRunState.MainStream);
                }
            }
        } else {
            attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     chunk_B, static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }
    }
}


}  // namespace dsl
