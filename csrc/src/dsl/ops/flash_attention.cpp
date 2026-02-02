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
    const bool cudnn_gqa_ok = (Hq == Hkv);

    if (!mRunState.scratch().cudnn_workspace.Data) {
        mRunState.temp_acquire(mRunState.scratch().cudnn_workspace);
        mTemps.push_back(mRunState.scratch().cudnn_workspace);
    }

    // Debug: log workspace size vs required cuDNN size (limited)
    {
        static int ws_log_count = 0;
        if (ws_log_count < 4) {
            const std::size_t ws_bytes =
                static_cast<std::size_t>(mRunState.scratch().cudnn_workspace.nelem()) *
                static_cast<std::size_t>(get_dtype_size(mRunState.scratch().cudnn_workspace.DType));
            const std::size_t ws_needed =
                cudnn_get_workspace_size(static_cast<int>(mB), static_cast<int>(mT),
                                         Hq, Hkv, Hs, mRunState.CudnnHandle);
            fprintf(stderr,
                    "[CUDNN_WS] B=%d T=%d Hq=%d Hkv=%d HS=%d ws_ptr=%p ws_bytes=%zu ws_needed=%zu\n",
                    static_cast<int>(mB), static_cast<int>(mT), Hq, Hkv, Hs,
                    mRunState.scratch().cudnn_workspace.Data, ws_bytes, ws_needed);
            ws_log_count++;
        }
    }

    // NaN detection on flash-attention input (qkv) before cuDNN call.
    {
        const long sample_token = 3;
        log_nan_sample("FWD_ATTN_IN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, sample_token);
    }
    // One-time stats for layer 0 qkv input to detect large magnitudes or NaNs
    if (op.inputs[0].layer_idx == 0) {
        log_tensor_stats("FWD_ATTN_IN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, 4096);
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

    if (!cudnn_gqa_ok) {
        static int cudnn_skip_count = 0;
        if (cudnn_skip_count < 4) {
            fprintf(stderr,
                    "[CUDNN_ATTN_SKIP] Hq=%d Hkv=%d causal=1 reason=GQA\n",
                    Hq, Hkv);
            cudnn_skip_count++;
        }
        attention_forward_custom(out, lse, qkv,
                                 static_cast<int>(mB), static_cast<int>(mT),
                                 Hq, Hkv, Hs, mRunState.MainStream);
    } else {
        attention_forward_cudnn(out, lse, qkv, mRunState.scratch().cudnn_workspace,
                                mRunState.CudnnHandle, static_cast<int>(mB), static_cast<int>(mT),
                                Hq, Hkv, Hs, mRunState.MainStream);
    }

    // DEBUG: Print first att values for layer 0
    if (op.outputs[0].layer_idx == 0) {
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> vals(8);
        cudaMemcpy(vals.data(), out.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[FWD_ATTN] Layer 0 att output ptr=%p, values=%.6f, %.6f, %.6f, %.6f\n",
                out.Data, vals[0], vals[1], vals[2], vals[3]);
        log_tensor_stats_ex("FWD_ATTN_OUT", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, false);
        log_tensor_stats_ex("FWD_LSE", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, false);
    }

    // NaN detection for attention forward (layer 25, token 3)
    {
        const long sample_token = 3;
        log_nan_sample("FWD_ATTN", op.outputs[0].layer_idx, op.outputs[0].name, out, sample_token);
        log_nan_sample("FWD_LSE", op.outputs[0].layer_idx, op.outputs[1].name, lse, sample_token);
        const bool out_nan = tensor_sample_has_nan_or_inf(out, sample_token);
        const bool lse_nan = tensor_sample_has_nan_or_inf(lse, sample_token);
        if (out_nan || lse_nan) {
            log_tensor_stats_ex("FWD_ATTN_OUT_NAN", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, true);
            log_tensor_stats_ex("FWD_LSE_NAN", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, true);
            log_tensor_stats_ex("FWD_ATTN_IN_NAN", op.inputs[0].layer_idx, op.inputs[0].name, qkv, 4096, true);
            // Fallback: recompute attention using the custom kernel to isolate cuDNN issues.
            fprintf(stderr, "[FWD_ATTN_FALLBACK] layer=%d using custom kernel\n", op.outputs[0].layer_idx);
            attention_forward_custom(out, lse, qkv,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
            log_tensor_stats_ex("FWD_ATTN_FALLBACK_OUT", op.outputs[0].layer_idx, op.outputs[0].name, out, 4096, true);
            log_tensor_stats_ex("FWD_LSE_FALLBACK", op.outputs[0].layer_idx, op.outputs[1].name, lse, 4096, true);
        }
    }
}

void CompiledExecutor::dispatch_flash_attention_backward(const CompiledOp& op) {
    // inputs (from autodiff): d_out, out (attention output), lse, qkv
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& out = resolve_tensor(op.inputs[1]);
    Tensor& lse = resolve_tensor(op.inputs[2]);
    Tensor& qkv = resolve_tensor(op.inputs[3]);
    Tensor& d_qkv = ensure_output_tensor(op.outputs[0]);

    // DEBUG: Print attention backward inputs for layer 0, 25, and 26
    int debug_layer_idx = -1;
    std::string debug_field;
    parse_block_param(op.inputs[1].name, debug_layer_idx, debug_field);

    // Trace Layer 25 flash attention backward for explosion debugging
    static int attn_25_trace = 0;
    if (debug_layer_idx == 25 && attn_25_trace < 10) {
        attn_25_trace++;
        cudaStreamSynchronize(mRunState.MainStream);
        std::vector<float> d_out_vals(4);
        cudaMemcpy(d_out_vals.data(), d_out.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        // Compute L2 norm of d_out
        const int N = static_cast<int>(std::min(static_cast<long>(d_out.nelem()), 10000L));
        std::vector<float> all(N);
        cudaMemcpy(all.data(), d_out.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_sq = 0.0f, max_val = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum_sq += all[i] * all[i];
            if (std::fabs(all[i]) > max_val) max_val = std::fabs(all[i]);
        }
        fprintf(stderr, "[FLASH_ATTN_BWD] Layer %d: d_out INPUT L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, std::sqrt(sum_sq), max_val, d_out_vals[0], d_out_vals[1], d_out_vals[2], d_out_vals[3]);
        // Also trace saved activations (qkv, out, lse) to check if they're corrupted
        const long sample_token = 3;
        std::vector<float> qkv_vals(4), out_vals(4), lse_vals(4);
        const bool qkv_ok = copy_tensor_token_sample_as_f32(qkv, sample_token, qkv_vals.size(), qkv_vals);
        const bool out_ok = copy_tensor_token_sample_as_f32(out, sample_token, out_vals.size(), out_vals);
        const bool lse_ok = copy_tensor_token_sample_as_f32(lse, sample_token, lse_vals.size(), lse_vals);
        fprintf(stderr,
                "[FLASH_ATTN_BWD] Layer %d: token=%ld qkv ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f  "
                "out ptr=%p ok=%d vals=%.6f,%.6f,%.6f,%.6f  lse ok=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                debug_layer_idx, sample_token, qkv.Data, qkv_ok ? 1 : 0,
                qkv_vals[0], qkv_vals[1], qkv_vals[2], qkv_vals[3],
                out.Data, out_ok ? 1 : 0,
                out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                lse_ok ? 1 : 0,
                lse_vals[0], lse_vals[1], lse_vals[2], lse_vals[3]);
    }

    // Targeted trace for layers 8/9 where residual spikes first appear.
    static int attn_8_9_trace = 0;
    if ((debug_layer_idx == 8 || debug_layer_idx == 9) && attn_8_9_trace < 12) {
        fprintf(stderr,
                "[ATTN_BWD_L8] layer=%d d_out=%s out=%s lse=%s qkv=%s\n",
                debug_layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[3].name.c_str());
        log_tensor_mag_unbounded("ATTN_BWD_L8_DOUT", debug_layer_idx, op.inputs[0].name, d_out, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_OUT", debug_layer_idx, op.inputs[1].name, out, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_LSE", debug_layer_idx, op.inputs[2].name, lse, 4096);
        log_tensor_mag_unbounded("ATTN_BWD_L8_QKV", debug_layer_idx, op.inputs[3].name, qkv, 4096);
        attn_8_9_trace++;
    }

    const int top_layer = static_cast<int>(mConfig.NumLayers) - 1;
    if (debug_layer_idx == 0 || debug_layer_idx == 26 || debug_layer_idx == top_layer) {
        cudaStreamSynchronize(mRunState.MainStream);
        const long sample_token = 3;
        std::vector<float> qkv_vals(4), out_vals(4), d_out_vals(4);
        const bool qkv_ok = copy_tensor_token_sample_as_f32(qkv, sample_token, qkv_vals.size(), qkv_vals);
        const bool out_ok = copy_tensor_token_sample_as_f32(out, sample_token, out_vals.size(), out_vals);
        const bool d_out_ok = copy_tensor_token_sample_as_f32(d_out, sample_token, d_out_vals.size(), d_out_vals);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld qkv_rope ptr=%p ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, qkv.Data, qkv_ok ? 1 : 0,
                qkv_vals[0], qkv_vals[1], qkv_vals[2], qkv_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld att ptr=%p ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, out.Data, out_ok ? 1 : 0,
                out_vals[0], out_vals[1], out_vals[2], out_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d token=%ld d_out(d_att) ok=%d values=%.6f, %.6f, %.6f, %.6f\n",
                debug_layer_idx, sample_token, d_out_ok ? 1 : 0,
                d_out_vals[0], d_out_vals[1], d_out_vals[2], d_out_vals[3]);
        fprintf(stderr,
                "[ATTN_BWD] Layer %d shapes: d_out=%s qkv=%s out=%s lse=%s\n",
                debug_layer_idx,
                tensor_shape_str(d_out).c_str(),
                tensor_shape_str(qkv).c_str(),
                tensor_shape_str(out).c_str(),
                tensor_shape_str(lse).c_str());
    }

    Tensor* out_ptr = &out;
    Tensor* lse_ptr = &lse;
    Tensor* qkv_ptr = &qkv;

    const int Hq = static_cast<int>(mConfig.NumQueryHeads);
    const int Hkv = static_cast<int>(mConfig.NumKeyValHeads);
    const int Hs = static_cast<int>(mConfig.head_size());
    const bool cudnn_gqa_ok = (Hq == Hkv);
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

    // Parse layer_idx for use in cuDNN call
    int layer_idx = -1;
    std::string field;
    parse_block_param(op.inputs[3].name, layer_idx, field);

    // NaN watchdog for attention backward inputs (log a few occurrences).
    static int attn_bwd_nan_in_logged = 0;
    Tensor d_out_scan = d_out;
    if (d_out_scan.Rank > 2 && d_out_scan.Sizes[0] == mB && d_out_scan.Sizes[1] == mT) {
        d_out_scan = view_tensor(d_out_scan, {mB * mT, d_out_scan.Sizes[d_out_scan.Rank - 1]});
    }
    float attn_row_min = 0.0f;
    float attn_row_max = 0.0f;
    const bool attn_row_nan = tensor_row_has_nan_or_inf(d_out_scan, 3, &attn_row_min, &attn_row_max);
    if (attn_bwd_nan_in_logged < 4 && attn_row_nan) {
        auto dump_sample = [](const char* tag, const Tensor& t, const std::string& name) {
            std::vector<float> vals(4, 0.0f);
            const bool ok = copy_tensor_sample_as_f32(t, vals.size(), vals);
            int nan = 0;
            int inf = 0;
            for (float v : vals) {
                if (std::isnan(v)) {
                    nan++;
                } else if (std::isinf(v)) {
                    inf++;
                }
            }
            fprintf(stderr,
                    "[%s] name=%s dtype=%s ok=%d nan=%d inf=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                    tag,
                    name.c_str(),
                    dtype_to_str(t.DType),
                    ok ? 1 : 0,
                    nan,
                    inf,
                    vals[0], vals[1], vals[2], vals[3]);
        };
        fprintf(stderr,
                "[FLASH_ATTN_BWD_NAN_IN] layer=%d d_out=%s out=%s lse=%s qkv=%s row_min=%.6f row_max=%.6f\n",
                layer_idx,
                op.inputs[0].name.c_str(),
                op.inputs[1].name.c_str(),
                op.inputs[2].name.c_str(),
                op.inputs[3].name.c_str(),
                attn_row_min,
                attn_row_max);
        dump_sample("FLASH_ATTN_BWD_NAN_DOUT", d_out, op.inputs[0].name);
        dump_sample("FLASH_ATTN_BWD_NAN_OUT", out, op.inputs[1].name);
        dump_sample("FLASH_ATTN_BWD_NAN_LSE", lse, op.inputs[2].name);
        dump_sample("FLASH_ATTN_BWD_NAN_QKV", qkv, op.inputs[3].name);
        attn_bwd_nan_in_logged++;
    }

    // FIX: Zero-initialize d_qkv before cuDNN attention backward to prevent NaN from uninitialized memory.
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
            static int bwd_skip_count = 0;
            if (bwd_skip_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_SKIP] Hq=%d Hkv=%d reason=%s\n",
                        Hq, Hkv, force_custom_bwd ? "forced_custom" : "gqa");
                bwd_skip_count++;
            }
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
                        static int scratch_small_log = 0;
                        if (scratch_small_log < 8) {
                            fprintf(stderr,
                                    "[ATTN_F32_SCRATCH_TOO_SMALL] have_qkv=%zu need_qkv=%zu have_out=%zu need_out=%zu "
                                    "have_d_out=%zu need_d_out=%zu have_d_qkv=%zu need_d_qkv=%zu\\n",
                                    have_qkv, need_qkv, have_out, need_out,
                                    have_d_out, need_d_out, have_d_qkv, need_d_qkv);
                            scratch_small_log++;
                        }
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
                            static int scratch_disable_log = 0;
                            if (scratch_disable_log < 4) {
                                fprintf(stderr, "[ATTN_F32_SCRATCH_DISABLE] using stack fallback buffers\n");
                                scratch_disable_log++;
                            }
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
                const bool trace_ranges = (std::getenv("SUROGATE_TRACE_ATTN_F32_RANGES") != nullptr);
                static int attn_f32_range_log = 0;
                if (trace_ranges && attn_f32_range_log < 8) {
                    attn_f32_range_log++;
                    auto log_range = [](const char* tag, const Tensor& t) {
                        const auto start = reinterpret_cast<std::uintptr_t>(t.Data);
                        const auto end = start + t.bytes();
                        fprintf(stderr,
                                "[%s] ptr=%p bytes=%zu range=[0x%zx..0x%zx) shape=%s dtype=%s\n",
                                tag,
                                t.Data,
                                t.bytes(),
                                static_cast<std::size_t>(start),
                                static_cast<std::size_t>(end),
                                tensor_shape_str(t).c_str(),
                                dtype_to_str(t.DType));
                    };
                    log_range("ATTN_F32_QKV", qkv_f32);
                    log_range("ATTN_F32_OUT", out_f32);
                    log_range("ATTN_F32_DOUT", d_out_f32);
                    log_range("ATTN_F32_DQKV", d_qkv_f32);
                    log_range("ATTN_BF16_QKV", qkv);
                    log_range("ATTN_BF16_OUT", out);
                    log_range("ATTN_BF16_DOUT", d_out);
                    log_range("ATTN_BF16_DQKV", d_qkv);
                    fprintf(stderr,
                            "[ATTN_F32_STACK] used=%zu unused=%zu\n",
                            mRunState.Stack.bytes_used(),
                            mRunState.Stack.unused_capacity());
                }
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
            static int bwd_force_count = 0;
            if (force_cudnn_bwd && bwd_force_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_FORCE] Hq=%d Hkv=%d\n",
                        Hq, Hkv);
                bwd_force_count++;
            }
            attention_backward_cudnn(d_qkv, *lse_ptr, *out_ptr, d_out, *qkv_ptr,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     static_cast<int>(mB), static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }

    // DEBUG: Print d_qkv output for layer 0, 25, and 26
    if (debug_layer_idx == 25 && attn_25_trace <= 10) {
            cudaStreamSynchronize(mRunState.MainStream);
            const int N = static_cast<int>(std::min(static_cast<long>(d_qkv.nelem()), 10000L));
            std::vector<float> all(N);
            cudaMemcpy(all.data(), d_qkv.Data, N * sizeof(float), cudaMemcpyDeviceToHost);
            float sum_sq = 0.0f, max_val = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum_sq += all[i] * all[i];
                if (std::fabs(all[i]) > max_val) max_val = std::fabs(all[i]);
            }
            fprintf(stderr, "[FLASH_ATTN_BWD] Layer %d: d_qkv OUTPUT L2=%.6f max=%.6f vals=%.6f,%.6f,%.6f,%.6f\n",
                    debug_layer_idx, std::sqrt(sum_sq), max_val, all[0], all[1], all[2], all[3]);
        }
        if (debug_layer_idx == 0 || debug_layer_idx == 26 || debug_layer_idx == top_layer) {
            cudaStreamSynchronize(mRunState.MainStream);
            std::vector<float> vals(8);
            cudaMemcpy(vals.data(), d_qkv.Data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[ATTN_BWD] Layer %d d_qkv OUTPUT ptr=%p, values=%.6f, %.6f, %.6f, %.6f\n",
                    debug_layer_idx, d_qkv.Data, vals[0], vals[1], vals[2], vals[3]);
        }

        static int attn_8_9_out_trace = 0;
        if ((debug_layer_idx == 8 || debug_layer_idx == 9) && attn_8_9_out_trace < 12) {
            log_tensor_mag_unbounded("ATTN_BWD_L8_DQKV", debug_layer_idx, op.outputs[0].name, d_qkv, 4096);
            const std::size_t dqkv_total = static_cast<std::size_t>(d_qkv.nelem());
            if (dqkv_total > 4096) {
                log_tensor_sample_stats("ATTN_BWD_L8_DQKV_MID", d_qkv, dqkv_total / 2, 4096);
            }
            attn_8_9_out_trace++;
        }

        // NaN watchdog for attention backward outputs (log a few occurrences).
        static int attn_bwd_nan_out_logged = 0;
        Tensor d_qkv_scan = d_qkv;
        if (d_qkv_scan.Rank > 2 && d_qkv_scan.Sizes[0] == mB && d_qkv_scan.Sizes[1] == mT) {
            d_qkv_scan = view_tensor(d_qkv_scan, {mB * mT, d_qkv_scan.Sizes[d_qkv_scan.Rank - 1]});
        }
        float dq_row_min = 0.0f;
        float dq_row_max = 0.0f;
        const bool dq_row_nan = tensor_row_has_nan_or_inf(d_qkv_scan, 3, &dq_row_min, &dq_row_max);
        if (attn_bwd_nan_out_logged < 4 && dq_row_nan) {
            auto dump_sample = [](const char* tag, const Tensor& t, const std::string& name) {
                std::vector<float> vals(4, 0.0f);
                const bool ok = copy_tensor_sample_as_f32(t, vals.size(), vals);
                int nan = 0;
                int inf = 0;
                for (float v : vals) {
                    if (std::isnan(v)) {
                        nan++;
                    } else if (std::isinf(v)) {
                        inf++;
                    }
                }
                fprintf(stderr,
                        "[%s] name=%s dtype=%s ok=%d nan=%d inf=%d vals=%.6f,%.6f,%.6f,%.6f\n",
                        tag,
                        name.c_str(),
                        dtype_to_str(t.DType),
                        ok ? 1 : 0,
                        nan,
                        inf,
                        vals[0], vals[1], vals[2], vals[3]);
            };
            fprintf(stderr,
                    "[FLASH_ATTN_BWD_NAN_OUT] layer=%d d_qkv=%s row_min=%.6f row_max=%.6f\n",
                    layer_idx,
                    op.outputs[0].name.c_str(),
                    dq_row_min,
                    dq_row_max);
            dump_sample("FLASH_ATTN_BWD_NAN_DQKV", d_qkv, op.outputs[0].name);
            auto log_nan_row = [&](const char* tag, const Tensor& t, const std::string& name) {
                long nan_row = -1;
                float row_min = 0.0f;
                float row_max = 0.0f;
                if (!find_first_nan_row(t, &nan_row, &row_min, &row_max)) {
                    fprintf(stderr, "[%s] name=%s any_nan=0\n", tag, name.c_str());
                    return;
                }
                if (t.Rank >= 2 && t.Sizes[0] == mB * mT) {
                    const long b = nan_row / static_cast<long>(mT);
                    const long tok = nan_row % static_cast<long>(mT);
                    fprintf(stderr,
                            "[%s] name=%s any_nan=1 row=%ld (b=%ld t=%ld) row_min=%.6f row_max=%.6f\n",
                            tag, name.c_str(), nan_row, b, tok, row_min, row_max);
                } else {
                    fprintf(stderr,
                            "[%s] name=%s any_nan=1 row=%ld row_min=%.6f row_max=%.6f\n",
                            tag, name.c_str(), nan_row, row_min, row_max);
                }
            };
            Tensor d_out_scan2 = d_out;
            if (d_out_scan2.Rank > 2 && d_out_scan2.Sizes[0] == mB && d_out_scan2.Sizes[1] == mT) {
                d_out_scan2 = view_tensor(d_out_scan2, {mB * mT, d_out_scan2.Sizes[d_out_scan2.Rank - 1]});
            }
            Tensor out_scan2 = out;
            if (out_scan2.Rank > 2 && out_scan2.Sizes[0] == mB && out_scan2.Sizes[1] == mT) {
                out_scan2 = view_tensor(out_scan2, {mB * mT, out_scan2.Sizes[out_scan2.Rank - 1]});
            }
            Tensor qkv_scan2 = qkv;
            if (qkv_scan2.Rank > 2 && qkv_scan2.Sizes[0] == mB && qkv_scan2.Sizes[1] == mT) {
                qkv_scan2 = view_tensor(qkv_scan2, {mB * mT, qkv_scan2.Sizes[qkv_scan2.Rank - 1]});
            }
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_DOUT", d_out_scan2, op.inputs[0].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_OUT", out_scan2, op.inputs[1].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_QKV", qkv_scan2, op.inputs[3].name);
            log_nan_row("FLASH_ATTN_BWD_NAN_INPUT_LSE", lse, op.inputs[2].name);
            attn_bwd_nan_out_logged++;
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
            static int bwd_force_count = 0;
            if (force_cudnn_bwd && bwd_force_count < 4) {
                fprintf(stderr,
                        "[CUDNN_ATTN_BWD_FORCE] Hq=%d Hkv=%d\n",
                        Hq, Hkv);
                bwd_force_count++;
            }
            attention_backward_cudnn(d_qkv_chunk, lse_chunk, out_chunk, d_out_chunk, qkv_chunk,
                                     mRunState.scratch().cudnn_workspace,
                                     mRunState.CudnnHandle,
                                     chunk_B, static_cast<int>(mT),
                                     Hq, Hkv, Hs, mRunState.MainStream);
        }
    }
}


}  // namespace dsl
