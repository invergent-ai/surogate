// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert Parallelism combine: route expert output back to originating GPUs via reverse A2A.
// After local expert GEMMs, ep_combine:
// 1. Un-sorts expert output from local expert order to recv (peer) order
// 2. Reverse all-to-all to send results back to originating GPUs
// 3. Output is in the original permuted order (ready for moe_unpermute)

#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_bf16.h>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"

namespace dsl {

void CompiledExecutor::dispatch_ep_combine(const CompiledOp& op) {
    Tensor& expert_output = resolve_tensor(op.inputs[0]);  // [total_recv, C] sorted by local expert

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(expert_output.Sizes[1]);

    // No-op if EP not active
    if (ep_size <= 1 || !mComm || !mComm->ep_enabled()) {
        store_tensor(op.outputs[0], expert_output);
        return;
    }

    // ---- Parse layer index ----
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) name.remove_prefix(6);
        std::string field;
        parse_block_param(name, layer_idx, field);
    }
    const int ep_key = ep_state_key(layer_idx);

    auto it = mEpStates.find(ep_key);
    if (it == mEpStates.end()) {
        throw std::runtime_error(
            "ep_combine: no EP state found for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = it->second;

    // ---- 1. Un-sort expert output (reverse local re-permutation from dispatch) ----
    // expert_output is in sorted (by local expert) order
    const int elem_sz = (expert_output.DType == ETensorDType::BF16) ? 2 : 4;
    const size_t unsorted_bytes = static_cast<size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* unsorted_ptr = ep_buf_acquire(unsorted_bytes);

    Tensor unsorted_output = make_raw_tensor(unsorted_ptr, expert_output.DType,
        {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
        expert_output.Device);

    if (expert_output.DType == ETensorDType::BF16) {
        moe_permute_tokens(unsorted_output.get<nv_bfloat16>(),
                           expert_output.get<nv_bfloat16>(),
                           static_cast<int*>(ep_state.recv_reorder_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    } else {
        moe_permute_tokens(unsorted_output.get<float>(),
                           expert_output.get<float>(),
                           static_cast<int*>(ep_state.recv_reorder_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    }

    // ---- 2. Reverse A2A: send results back to originating GPUs ----
    // Use per-layer persistent buffer: this activation is needed by backward.
    const size_t combined_need = static_cast<size_t>(ep_state.total_send) * hidden_size * elem_sz;
    auto& ep_state_mut = mEpStates[ep_key];
    if (ep_state_mut.combined_bytes < combined_need) {
        if (ep_state_mut.combined_gpu) CUDA_CHECK(cudaFree(ep_state_mut.combined_gpu));
        CUDA_CHECK(cudaMalloc(&ep_state_mut.combined_gpu, combined_need));
        ep_state_mut.combined_bytes = combined_need;
    }
    Tensor combined = make_raw_tensor(ep_state_mut.combined_gpu, expert_output.DType,
        {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
        expert_output.Device);

    std::vector<int> reverse_send_elem(ep_size), reverse_recv_elem(ep_size);
    for (int p = 0; p < ep_size; ++p) {
        reverse_send_elem[p] = ep_state.recv_splits[p] * hidden_size;
        reverse_recv_elem[p] = ep_state.send_splits[p] * hidden_size;
    }

    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(unsorted_output.Data),
        reinterpret_cast<std::byte*>(combined.Data),
        reverse_send_elem.data(), reverse_recv_elem.data(),
        elem_sz, mRunState.MainStream);

    // Return buffer to pool (stream ordering ensures A2A completes before reuse)
    ep_buf_release(unsorted_ptr, unsorted_bytes);

    // ---- 3. If LLEP was active, undo the send reorder ----
    // Forward ep_dispatch reordered the send buffer: send_buf[i] = permuted_input[send_order[i]]
    // After reverse A2A, combined[i] = result for send_buf[i] = result for permuted_input[send_order[i]]
    // We need output[j] = result for permuted_input[j], so: output[j] = combined[inverse_order[j]]
    const bool sync_combine_fwd_out =
        (std::getenv("SUROGATE_EP_SYNC_COMBINE_FWD_OUT") != nullptr) &&
        mComm && (mComm->dp_size() == 1);

    if (ep_state.llep_send_reorder_gpu) {
        const int N = ep_state.total_send;
        std::vector<int> send_order_host(N);
        CUDA_CHECK(cudaMemcpyAsync(send_order_host.data(), ep_state.llep_send_reorder_gpu,
                                    N * sizeof(int), cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

        std::vector<int> inverse_order(N);
        for (int i = 0; i < N; ++i) {
            inverse_order[send_order_host[i]] = i;
        }

        const size_t inv_gpu_bytes = static_cast<size_t>(N) * sizeof(int);
        void* inv_gpu_ptr = ep_buf_acquire(inv_gpu_bytes);
        CUDA_CHECK(cudaMemcpyAsync(inv_gpu_ptr, inverse_order.data(),
                                    N * sizeof(int), cudaMemcpyHostToDevice, mRunState.MainStream));

        // Allocate per-layer persistent output for LLEP reorder.
        if (ep_state_mut.llep_combined_bytes < combined_need) {
            if (ep_state_mut.llep_combined_gpu) CUDA_CHECK(cudaFree(ep_state_mut.llep_combined_gpu));
            CUDA_CHECK(cudaMalloc(&ep_state_mut.llep_combined_gpu, combined_need));
            ep_state_mut.llep_combined_bytes = combined_need;
        }
        Tensor reordered = make_raw_tensor(ep_state_mut.llep_combined_gpu, expert_output.DType,
            {static_cast<long>(N), static_cast<long>(hidden_size)}, expert_output.Device);

        if (expert_output.DType == ETensorDType::BF16) {
            moe_permute_tokens(reordered.get<nv_bfloat16>(), combined.get<nv_bfloat16>(),
                               static_cast<int*>(inv_gpu_ptr),
                               N, N, hidden_size, 1, mRunState.MainStream);
        } else {
            moe_permute_tokens(reordered.get<float>(), combined.get<float>(),
                               static_cast<int*>(inv_gpu_ptr),
                               N, N, hidden_size, 1, mRunState.MainStream);
        }

        ep_buf_release(inv_gpu_ptr, inv_gpu_bytes);
        if (sync_combine_fwd_out) {
            // In EP-only setups (dp_size=1), all EP ranks should reconstruct
            // the same combined activation. This optional sync can suppress
            // cross-rank drift in sensitive recurrent stacks.
            mComm->all_reduce_avg(reordered, mRunState.MainStream);
        }
        store_tensor(op.outputs[0], reordered);
    } else {
        // ---- 3. Store output ----
        if (sync_combine_fwd_out) {
            // In EP-only setups (dp_size=1), all EP ranks should reconstruct
            // the same combined activation. This optional sync can suppress
            // cross-rank drift in sensitive recurrent stacks.
            mComm->all_reduce_avg(combined, mRunState.MainStream);
        }
        store_tensor(op.outputs[0], combined);
    }
}

void CompiledExecutor::dispatch_ep_combine_backward(const CompiledOp& op) {
    // Backward of ep_combine: A2A + re-sort (same as ep_dispatch forward direction)
    // Input: d_combined [total_send, C]
    // Output: d_expert_output [total_recv, C] in local expert order
    Tensor& d_combined = resolve_tensor(op.inputs[0]);

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(d_combined.Sizes[1]);

    if (ep_size <= 1 || !mComm || !mComm->ep_enabled()) {
        Tensor& d_expert = ensure_output_tensor(op.outputs[0]);
        d_expert = d_combined;
        store_tensor(op.outputs[0], d_expert);
        return;
    }

    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("d_", 0) == 0) name.remove_prefix(2);
        if (name.rfind("saved.", 0) == 0) name.remove_prefix(6);
        std::string field;
        parse_block_param(name, layer_idx, field);
    }
    const int input_total_send = static_cast<int>(d_combined.Sizes[0]);
    const int ep_key_default = ep_state_key(layer_idx);
    int ep_key_selected = ep_key_default;
    auto it = mEpStates.find(ep_key_default);
    if (mOptions.EPSize > 1 && layer_idx >= 0) {
        const int ep_key_r0 = (layer_idx << 1);
        const int ep_key_r1 = ep_key_r0 | 1;
        auto it_r0 = mEpStates.find(ep_key_r0);
        auto it_r1 = mEpStates.find(ep_key_r1);
        // Backward usually consumes replay activations when recompute is enabled.
        // Prefer replay EP state when present, but keep shape sanity checks.
        if (it_r1 != mEpStates.end() && it_r1->second.total_send == input_total_send) {
            it = it_r1;
            ep_key_selected = ep_key_r1;
        } else if (it_r0 != mEpStates.end() && it_r0->second.total_send == input_total_send) {
            it = it_r0;
            ep_key_selected = ep_key_r0;
        } else if (it_r1 != mEpStates.end()) {
            it = it_r1;
            ep_key_selected = ep_key_r1;
        } else if (it_r0 != mEpStates.end()) {
            it = it_r0;
            ep_key_selected = ep_key_r0;
        }
    }
    if (it == mEpStates.end()) {
        throw std::runtime_error(
            "ep_combine_backward: no EP state for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = it->second;
    auto& ep_state_mut = mEpStates[ep_key_selected];
    if (ep_state.total_send != input_total_send) {
        std::ostringstream oss;
        oss << "ep_combine_backward: EP state/input mismatch at layer " << layer_idx
            << " (selected_key=" << ep_key_selected
            << ", state.total_send=" << ep_state.total_send
            << ", input.rows=" << input_total_send << ")";
        throw std::runtime_error(oss.str());
    }
    if (mComm && (mComm->dp_size() == 1) && (ep_size > 1)) {
        // In replicated-input EP mode (dp_size=1), each EP rank contributes the same
        // token gradients. Scale once here to keep expert-path gradient magnitude
        // comparable to EP=1.
        const float inv_ep = 1.0f / static_cast<float>(ep_size);
        if (d_combined.DType == ETensorDType::BF16) {
            moe_scale_forward(d_combined.get<nv_bfloat16>(),
                              d_combined.get<nv_bfloat16>(),
                              inv_ep, static_cast<int>(d_combined.nelem()),
                              mRunState.MainStream);
        } else {
            moe_scale_forward(d_combined.get<float>(),
                              d_combined.get<float>(),
                              inv_ep, static_cast<int>(d_combined.nelem()),
                              mRunState.MainStream);
        }
    }

    // ---- 0. If LLEP was active, redo the send reorder on gradients ----
    // Forward: output[j] = combined[inverse_order[j]]
    // Backward of this step: d_combined[i] = d_output[send_order[i]]
    const int elem_sz = (d_combined.DType == ETensorDType::BF16) ? 2 : 4;
    const Tensor* d_a2a_input = &d_combined;
    void* d_reordered_ptr = nullptr;
    size_t d_reordered_bytes = 0;
    Tensor d_reordered;  // Must outlive d_a2a_input usage below

    if (ep_state.llep_send_reorder_gpu) {
        d_reordered_bytes = static_cast<size_t>(ep_state.total_send) * hidden_size * elem_sz;
        d_reordered_ptr = ep_buf_acquire(d_reordered_bytes);
        d_reordered = make_raw_tensor(d_reordered_ptr, d_combined.DType,
            {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
            d_combined.Device);

        // llep_send_reorder_gpu = send_order, so d_reordered[i] = d_combined[send_order[i]]
        if (d_combined.DType == ETensorDType::BF16) {
            moe_permute_tokens(d_reordered.get<nv_bfloat16>(), d_combined.get<nv_bfloat16>(),
                               static_cast<int*>(ep_state.llep_send_reorder_gpu),
                               ep_state.total_send, ep_state.total_send,
                               hidden_size, 1, mRunState.MainStream);
        } else {
            moe_permute_tokens(d_reordered.get<float>(), d_combined.get<float>(),
                               static_cast<int*>(ep_state.llep_send_reorder_gpu),
                               ep_state.total_send, ep_state.total_send,
                               hidden_size, 1, mRunState.MainStream);
        }
        d_a2a_input = &d_reordered;
    }

    // ---- 1. Forward A2A (original direction: send to expert-owning GPUs) ----
    const size_t bwd_unsorted_bytes = static_cast<size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* d_recv_unsorted_ptr = ep_buf_acquire(bwd_unsorted_bytes);

    Tensor d_recv_unsorted = make_raw_tensor(d_recv_unsorted_ptr, d_combined.DType,
        {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
        d_combined.Device);

    std::vector<int> send_elem(ep_size), recv_elem(ep_size);
    for (int p = 0; p < ep_size; ++p) {
        send_elem[p] = ep_state.send_splits[p] * hidden_size;
        recv_elem[p] = ep_state.recv_splits[p] * hidden_size;
    }

    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(d_a2a_input->Data),
        reinterpret_cast<std::byte*>(d_recv_unsorted.Data),
        send_elem.data(), recv_elem.data(),
        elem_sz, mRunState.MainStream);

    // Release LLEP reorder buffer if used
    if (d_reordered_ptr) {
        ep_buf_release(d_reordered_ptr, d_reordered_bytes);
    }

    // ---- 2. Re-sort by local expert (same permutation as dispatch forward) ----
    // Use per-layer persistent output: shared cross-layer buffers can be overwritten
    // before this tensor is consumed by downstream backward ops.
    const size_t sorted_need = static_cast<size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    if (ep_state_mut.combine_bwd_sorted_bytes < sorted_need) {
        if (ep_state_mut.combine_bwd_sorted_gpu) CUDA_CHECK(cudaFree(ep_state_mut.combine_bwd_sorted_gpu));
        CUDA_CHECK(cudaMalloc(&ep_state_mut.combine_bwd_sorted_gpu, sorted_need));
        ep_state_mut.combine_bwd_sorted_bytes = sorted_need;
    }
    Tensor d_expert_sorted = make_raw_tensor(ep_state_mut.combine_bwd_sorted_gpu, d_combined.DType,
        {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
        d_combined.Device);

    if (d_combined.DType == ETensorDType::BF16) {
        moe_permute_tokens(d_expert_sorted.get<nv_bfloat16>(),
                           d_recv_unsorted.get<nv_bfloat16>(),
                           static_cast<int*>(ep_state.send_order_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    } else {
        moe_permute_tokens(d_expert_sorted.get<float>(),
                           d_recv_unsorted.get<float>(),
                           static_cast<int*>(ep_state.send_order_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    }

    // Return buffer to pool (stream ordering ensures re-sort completes before reuse)
    ep_buf_release(d_recv_unsorted_ptr, bwd_unsorted_bytes);

    store_tensor(op.outputs[0], d_expert_sorted);
}

}  // namespace dsl
