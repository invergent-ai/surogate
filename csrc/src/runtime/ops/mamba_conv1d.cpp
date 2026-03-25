// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Mamba causal conv1d operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_mamba_conv1d(const CompiledOp& op) {
    // Input: x [B, conv_dim, T]
    // Weights: weight [conv_dim, kernel], bias [conv_dim] (optional)
    // Output: out [B, conv_dim, T]
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& weight = resolve_tensor(op.inputs[1]);
    Tensor* bias = nullptr;
    if (op.inputs.size() > 2 && !op.inputs[2].name.empty()) {
        bias = &resolve_tensor(op.inputs[2]);
    }

    const int B = static_cast<int>(x.Sizes[0]);
    const int conv_dim = static_cast<int>(x.Sizes[1]);
    const int T = static_cast<int>(x.Sizes[2]);
    const int kernel = static_cast<int>(weight.Sizes[1]);

    // Determine if SiLU activation should be applied
    bool silu = (op.attrs.activation == "silu");

    const long expected = static_cast<long>(B) * conv_dim * T;
    auto shape_matches = [&](const TensorRef& ref) -> bool {
        if (ref.shape.empty()) return false;
        long prod = 1;
        for (auto d : ref.shape) {
            if (d <= 0) return false;
            prod *= d;
        }
        return prod == expected;
    };

    Tensor* out_ptr = nullptr;
    if (shape_matches(op.outputs[0])) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.nelem() == expected) {
            out_ptr = &out_ref;
        }
    }
    if (!out_ptr) {
        Tensor out = mRunState.temp_alloc(x.DType, {B, conv_dim, T}, "mamba_conv1d_out");
        mTemps.push_back(out);
        out_ptr = &mTemps.back();
    }

    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty() && op.inputs[0].layer_idx >= 0) {
        layer_idx = op.inputs[0].layer_idx;
    }
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string field;
        parse_block_param(op.inputs[0].name, layer_idx, field);
    }

    // Decode/prefill generation path:
    // keep and update per-layer causal-conv tail state [B, conv_dim, kernel-1].
    // This state is independent of KV paging, so it must be active for both
    // paged and non-paged decode.
    if (mDecodeState && mDecodeState->conv_states && layer_idx >= 0 && kernel > 1) {
        auto& conv_states = *mDecodeState->conv_states;
        const bool strict_conv_state = mDecodeState->strict_state_buffers;
        const bool flat_mode = mDecodeState->flat_token_mode
                            && mDecodeState->flat_batch_size > 1
                            && mDecodeState->q_indptr_host != nullptr;
        const int state_B = flat_mode ? mDecodeState->flat_batch_size : B;
        const int state_len = kernel - 1;
        const std::size_t state_elems =
            static_cast<std::size_t>(state_B)
            * static_cast<std::size_t>(conv_dim)
            * static_cast<std::size_t>(state_len);
        const std::size_t state_bytes = state_elems * get_dtype_size(x.DType);
        const std::size_t per_seq_bytes = static_cast<std::size_t>(conv_dim)
                                        * static_cast<std::size_t>(state_len)
                                        * get_dtype_size(x.DType);

        auto it_state = conv_states.find(layer_idx);
        void* state_ptr = (it_state != conv_states.end()) ? it_state->second : nullptr;
        bool need_realloc = (state_ptr == nullptr);
        // In flat mode the op tensor has B=1, but the persistent conv-state must
        // track one row per request. Check if current capacity is sufficient.
        if (!strict_conv_state && mDecodeState->conv_state_bytes) {
            auto& conv_state_bytes_map = *mDecodeState->conv_state_bytes;
            auto it = conv_state_bytes_map.find(layer_idx);
            if (it == conv_state_bytes_map.end()) {
                need_realloc = true;
            } else if (it->second != per_seq_bytes) {
                need_realloc = true;
            } else if (flat_mode) {
                // In flat mode, check if the buffer is large enough for state_bytes.
                // The buffer may have been allocated for a different batch size.
                // Use a capacity tracking map to avoid unnecessary reallocation.
                // Convention: conv_state_bytes stores per_seq_bytes; total capacity
                // is tracked by whether state_bytes fits in the existing allocation.
                // We use a static thread-local map to track capacities.
                static thread_local std::unordered_map<int, std::size_t> conv_state_capacity;
                auto cap_it = conv_state_capacity.find(layer_idx);
                if (cap_it == conv_state_capacity.end() || cap_it->second < state_bytes) {
                    need_realloc = true;
                    conv_state_capacity[layer_idx] = state_bytes;  // will be updated below
                }
            }
        } else if (strict_conv_state) {
            if (mDecodeState->conv_state_bytes) {
                auto& conv_state_bytes_map = *mDecodeState->conv_state_bytes;
                auto it = conv_state_bytes_map.find(layer_idx);
                if (it == conv_state_bytes_map.end() || it->second != per_seq_bytes) {
                    throw std::runtime_error(
                        "mamba_conv1d: strict decode conv-state missing or byte-size mismatch for layer "
                        + std::to_string(layer_idx));
                }
            } else {
                throw std::runtime_error(
                    "mamba_conv1d: strict decode conv-state requires conv_state_bytes map");
            }
        }

        if (need_realloc) {
            // Only grow — never shrink. This avoids cudaFree during CUDA graph capture.
            // Allocate max(state_bytes, existing_capacity) to handle B changes.
            static thread_local std::unordered_map<int, std::size_t> conv_state_capacity;
            auto cap_it = conv_state_capacity.find(layer_idx);
            const std::size_t existing_cap = (cap_it != conv_state_capacity.end()) ? cap_it->second : 0;
            const std::size_t alloc_bytes = std::max(state_bytes, existing_cap);
            if (alloc_bytes > existing_cap || !state_ptr) {
                if (state_ptr) {
                    CUDA_CHECK(cudaFree(state_ptr));
                    state_ptr = nullptr;
                }
                CUDA_CHECK(cudaMalloc(&state_ptr, alloc_bytes));
                conv_state_capacity[layer_idx] = alloc_bytes;
            }
            CUDA_CHECK(cudaMemsetAsync(state_ptr, 0, alloc_bytes, mRunState.MainStream));
            conv_states[layer_idx] = state_ptr;
        }
        if (mDecodeState->conv_state_bytes) {
            if (!strict_conv_state) {
                (*mDecodeState->conv_state_bytes)[layer_idx] = per_seq_bytes;
            }
        }

        Tensor conv_state = Tensor::from_pointer(
            static_cast<std::byte*>(state_ptr), x.Device, x.DType,
            std::vector<long>{state_B, conv_dim, state_len});

        if (flat_mode) {
            const int B_real = mDecodeState->flat_batch_size;
            const int T_total = T;
            const int32_t* indptr = mDecodeState->q_indptr_host;
            const std::size_t elem_bytes = get_dtype_size(x.DType);
            const std::size_t flat_pitch = static_cast<std::size_t>(T_total) * elem_bytes;
            cudaStream_t stream = mRunState.MainStream;

            // Split decode (q_len<=1) and prefill (q_len>1) into separate
            // kernel calls to prevent zero-padding from corrupting conv state.
            std::vector<int> dec_idx, pre_idx;
            int max_pre_q = 0;
            for (int i = 0; i < B_real; ++i) {
                const int ql = indptr[i+1] - indptr[i];
                if (ql <= 1) dec_idx.push_back(i);
                else { pre_idx.push_back(i); max_pre_q = std::max(max_pre_q, ql); }
            }
            const std::size_t state_row = static_cast<std::size_t>(conv_dim) * state_len * elem_bytes;

            // ---- PREFILL: process each slot individually with exact q_len ----
            // No padding → no conv state corruption from zero-padded positions.
            // Cost: one kernel call per prefill slot per layer. Acceptable for
            // correctness (typically 1-5 prefill slots per step).
            for (int p = 0; p < static_cast<int>(pre_idx.size()); ++p) {
                const int i = pre_idx[p];
                const int qs = indptr[i];
                const int ql = indptr[i+1] - indptr[i];
                if (ql <= 0) continue;
                const std::size_t ql_pitch = static_cast<std::size_t>(ql) * elem_bytes;

                Tensor sx = mRunState.temp_alloc(x.DType, {1, conv_dim, ql}, "conv1d_pre_sx");
                mTemps.push_back(sx);
                Tensor so = mRunState.temp_alloc(x.DType, {1, conv_dim, ql}, "conv1d_pre_so");
                mTemps.push_back(so);

                // Extract this slot's conv state.
                Tensor ss = mRunState.temp_alloc(conv_state.DType,
                    {1, conv_dim, state_len}, "conv1d_pre_ss");
                mTemps.push_back(ss);
                CUDA_CHECK(cudaMemcpyAsync(ss.Data,
                    (const char*)conv_state.Data + i * state_row,
                    state_row, cudaMemcpyDeviceToDevice, stream));

                // Scatter this slot's tokens: flat [1, conv_dim, T_total] → [1, conv_dim, ql]
                CUDA_CHECK(cudaMemcpy2DAsync(
                    sx.Data, ql_pitch,
                    (const char*)x.Data + std::size_t(qs) * elem_bytes, flat_pitch,
                    ql_pitch, conv_dim,
                    cudaMemcpyDeviceToDevice, stream));

                mamba_causal_conv1d_update(so, ss, sx, weight, bias,
                                           1, ql, conv_dim, kernel, silu, stream);

                // Gather output back to flat.
                CUDA_CHECK(cudaMemcpy2DAsync(
                    (char*)out_ptr->Data + std::size_t(qs) * elem_bytes, flat_pitch,
                    so.Data, ql_pitch,
                    ql_pitch, conv_dim,
                    cudaMemcpyDeviceToDevice, stream));

                // Write conv state back.
                CUDA_CHECK(cudaMemcpyAsync(
                    (char*)conv_state.Data + i * state_row,
                    ss.Data, state_row, cudaMemcpyDeviceToDevice, stream));
            }

            // ---- DECODE sub-batch (T=1, exact) ----
            if (!dec_idx.empty()) {
                const int Bd = static_cast<int>(dec_idx.size());

                Tensor dx = mRunState.temp_alloc(x.DType, {Bd, conv_dim, 1}, "conv1d_dec_x");
                mTemps.push_back(dx);
                Tensor dout_t = mRunState.temp_alloc(x.DType, {Bd, conv_dim, 1}, "conv1d_dec_out");
                mTemps.push_back(dout_t);

                Tensor dstate = mRunState.temp_alloc(conv_state.DType,
                    {Bd, conv_dim, state_len}, "conv1d_dec_state");
                mTemps.push_back(dstate);
                for (int d = 0; d < Bd; ++d) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)dstate.Data + d * state_row,
                        (const char*)conv_state.Data + dec_idx[d] * state_row,
                        state_row, cudaMemcpyDeviceToDevice, stream));
                }

                for (int d = 0; d < Bd; ++d) {
                    const int qs = indptr[dec_idx[d]];
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        (char*)dx.Data + std::size_t(d) * conv_dim * elem_bytes,
                        elem_bytes,
                        (const char*)x.Data + std::size_t(qs) * elem_bytes,
                        flat_pitch,
                        elem_bytes,
                        conv_dim,
                        cudaMemcpyDeviceToDevice, stream));
                }

                mamba_causal_conv1d_update(dout_t, dstate, dx, weight, bias,
                                           Bd, 1, conv_dim, kernel, silu, stream);

                for (int d = 0; d < Bd; ++d) {
                    const int qs = indptr[dec_idx[d]];
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        (char*)out_ptr->Data + std::size_t(qs) * elem_bytes,
                        flat_pitch,
                        (const char*)dout_t.Data + std::size_t(d) * conv_dim * elem_bytes,
                        elem_bytes,
                        elem_bytes,
                        conv_dim,
                        cudaMemcpyDeviceToDevice, stream));
                }
                for (int d = 0; d < Bd; ++d) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)conv_state.Data + dec_idx[d] * state_row,
                        (const char*)dstate.Data + d * state_row,
                        state_row, cudaMemcpyDeviceToDevice, stream));
                }
            }
        } else {
            mamba_causal_conv1d_update(*out_ptr, conv_state, x, weight, bias,
                                       B, T, conv_dim, kernel, silu, mRunState.MainStream);
        }
        store_tensor(op.outputs[0], *out_ptr);
        return;
    }

    // Call kernel
    mamba_causal_conv1d_forward(*out_ptr, x, weight, bias,
                                 B, T, conv_dim, kernel, silu,
                                 mRunState.MainStream);

    store_tensor(op.outputs[0], *out_ptr);
}

void CompiledExecutor::dispatch_mamba_conv1d_backward(const CompiledOp& op) {
    // Inputs: d_out [B, conv_dim, T], x [B, conv_dim, T], weight [conv_dim, kernel]
    // Outputs: dx [B, conv_dim, T], dweight [conv_dim, kernel], dbias [conv_dim] (optional)
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& x = resolve_tensor(op.inputs[1]);
    Tensor& weight = resolve_tensor(op.inputs[2]);

    const int B = static_cast<int>(d_out.Sizes[0]);
    const int conv_dim = static_cast<int>(d_out.Sizes[1]);
    const int T = static_cast<int>(d_out.Sizes[2]);
    const int kernel = static_cast<int>(weight.Sizes[1]);

    bool silu = (op.attrs.activation == "silu");

    // Allocate outputs
    Tensor dx = mRunState.temp_alloc(d_out.DType, {B, conv_dim, T}, "mamba_conv1d_backward_dx");
    mTemps.push_back(dx);

    // Weight gradient is accumulated via atomicAdd — must zero-init (stack memory is stale)
    Tensor dweight_fp32 = mRunState.temp_alloc(ETensorDType::FP32, {conv_dim, kernel}, "mamba_conv1d_backward_dweight_fp32");
    mTemps.push_back(dweight_fp32);
    fill_zero(dweight_fp32, mRunState.MainStream);

    Tensor* dbias_fp32 = nullptr;
    if (op.outputs.size() > 2) {
        Tensor dbias = mRunState.temp_alloc(ETensorDType::FP32, {conv_dim}, "mamba_conv1d_backward_dbias_fp32");
        mTemps.push_back(dbias);
        fill_zero(mTemps.back(), mRunState.MainStream);
        dbias_fp32 = &mTemps.back();
    }

    // Call kernel
    mamba_causal_conv1d_backward(dx, dweight_fp32, dbias_fp32,
                                  x, weight, d_out,
                                  B, T, conv_dim, kernel, silu,
                                  mRunState.MainStream);

    store_tensor(op.outputs[0], dx);
    store_tensor(op.outputs[1], dweight_fp32);
    if (op.outputs.size() > 2 && dbias_fp32) {
        store_tensor(op.outputs[2], *dbias_fp32);
    }
}

}  // namespace dsl
