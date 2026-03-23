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
            const int T_padded = std::max(1, mDecodeState->flat_max_q_len);
            const int32_t* indptr = mDecodeState->q_indptr_host;
            const std::size_t elem_bytes = get_dtype_size(x.DType);
            const std::size_t src_pitch = static_cast<std::size_t>(T_total) * elem_bytes;
            const std::size_t dst_pitch = static_cast<std::size_t>(T_padded) * elem_bytes;

            Tensor x_padded = mRunState.temp_alloc(x.DType, {B_real, conv_dim, T_padded}, "mamba_conv1d_flat_x");
            mTemps.push_back(x_padded);
            Tensor out_padded = mRunState.temp_alloc(x.DType, {B_real, conv_dim, T_padded}, "mamba_conv1d_flat_out");
            mTemps.push_back(out_padded);

            CUDA_CHECK(cudaMemsetAsync(x_padded.Data, 0, x_padded.bytes(), mRunState.MainStream));
            CUDA_CHECK(cudaMemsetAsync(out_padded.Data, 0, out_padded.bytes(), mRunState.MainStream));

            // Scatter flat [1, conv_dim, total_tokens] into padded [B_real, conv_dim, T_padded].
            for (int i = 0; i < B_real; ++i) {
                const int q_start = indptr[i];
                const int q_len = indptr[i + 1] - indptr[i];
                if (q_len <= 0) continue;
                auto* dst_base = reinterpret_cast<char*>(x_padded.Data)
                               + static_cast<std::size_t>(i) * static_cast<std::size_t>(conv_dim) * dst_pitch;
                const auto* src_base = reinterpret_cast<const char*>(x.Data)
                                     + static_cast<std::size_t>(q_start) * elem_bytes;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    dst_base, dst_pitch,
                    src_base, src_pitch,
                    static_cast<std::size_t>(q_len) * elem_bytes,
                    static_cast<std::size_t>(conv_dim),
                    cudaMemcpyDeviceToDevice,
                    mRunState.MainStream));
            }

            mamba_causal_conv1d_update(out_padded, conv_state, x_padded, weight, bias,
                                       B_real, T_padded, conv_dim, kernel, silu, mRunState.MainStream);

            // Gather padded output back to flat [1, conv_dim, total_tokens].
            for (int i = 0; i < B_real; ++i) {
                const int q_start = indptr[i];
                const int q_len = indptr[i + 1] - indptr[i];
                if (q_len <= 0) continue;
                auto* dst_base = reinterpret_cast<char*>(out_ptr->Data)
                               + static_cast<std::size_t>(q_start) * elem_bytes;
                const auto* src_base = reinterpret_cast<const char*>(out_padded.Data)
                                     + static_cast<std::size_t>(i) * static_cast<std::size_t>(conv_dim) * dst_pitch;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    dst_base, src_pitch,
                    src_base, dst_pitch,
                    static_cast<std::size_t>(q_len) * elem_bytes,
                    static_cast<std::size_t>(conv_dim),
                    cudaMemcpyDeviceToDevice,
                    mRunState.MainStream));
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
