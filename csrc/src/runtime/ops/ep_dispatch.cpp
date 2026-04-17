// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert Parallelism dispatch: route permuted tokens to expert-owning GPUs via A2A.
// Includes LLEP (Least-Loaded EP) adaptive load balancing:
//   - When routing imbalance exceeds threshold, LPT schedules expert spilling
//   - Expert weights are P2P-transferred from native to helper GPUs
//   - Modified A2A sends tokens to LPT-assigned destinations
//   - Merged weight tensors built for GEMM (native + foreign experts)

#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <sstream>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/ep/lpt_planner.h"
#include "runtime/ep/weight_transfer.h"
#include "runtime/qlora/quantized_tensor.h"
#include "runtime/qlora/generic_quantizer.h"
#include "runtime/training/runtime_options.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"

namespace dsl {

namespace {

// Helper to persist a GPU buffer (realloc if needed)
void persist_gpu_buffer(void*& buf, size_t& buf_bytes, const void* src,
                        size_t need, cudaStream_t stream) {
    if (buf_bytes < need) {
        if (buf) CUDA_CHECK(cudaFree(buf));
        CUDA_CHECK(cudaMalloc(&buf, need));
        buf_bytes = need;
    }
    CUDA_CHECK(cudaMemcpyAsync(buf, src, need, cudaMemcpyDeviceToDevice, stream));
}

// Helper to persist a tensor to mMoeSavedBuffers
void save_persistent_buffer(
    std::unordered_map<std::string, void*>& buffers,
    std::unordered_map<std::string, size_t>& sizes,
    const std::string& key, const Tensor& src, cudaStream_t stream) {
    if (!src.Data) return;
    const size_t bytes = src.bytes();
    if (bytes == 0) return;
    auto buf_it = buffers.find(key);
    if (buf_it == buffers.end() || sizes[key] < bytes) {
        if (buf_it != buffers.end() && buf_it->second)
            CUDA_CHECK(cudaFree(buf_it->second));
        void* new_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&new_buf, bytes));
        buffers[key] = new_buf;
        sizes[key] = bytes;
    }
    CUDA_CHECK(cudaMemcpyAsync(buffers[key], src.Data, bytes,
                                cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

void CompiledExecutor::dispatch_ep_dispatch(const CompiledOp& op) {
    Tensor& permuted_input = resolve_tensor(op.inputs[0]);   // [total_send, C]
    Tensor& routing_indices = resolve_tensor(op.inputs[1]);   // [BT, K]
    Tensor& scatter_indices_in = resolve_tensor(op.inputs[2]); // [total_send]
    (void)routing_indices;

    const int ep_size = op.attrs.ep_size;
    const int num_experts = op.attrs.num_experts;
    const int num_local = num_experts / ep_size;
    const int hidden_size = static_cast<int>(permuted_input.Sizes[1]);
    const int device = permuted_input.Device;
    const int total_send = static_cast<int>(permuted_input.Sizes[0]);
    const int elem_sz = (permuted_input.DType == ETensorDType::BF16) ? 2 : 4;

    // No-op if EP not active
    if (ep_size <= 1 || !mComm || !mComm->ep_enabled()) {
        store_tensor(op.outputs[0], permuted_input);
        store_tensor(op.outputs[1], scatter_indices_in);
        return;
    }

    const int ep_rank = mComm->ep_rank();
    // ---- Parse layer index ----
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx < 0 && !op.inputs.empty()) {
        std::string_view name = op.inputs[0].name;
        if (name.rfind("saved.", 0) == 0) name.remove_prefix(6);
        std::string field;
        parse_block_param(name, layer_idx, field);
    }
    const int ep_key = ep_state_key(layer_idx);

    const std::string layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
    // Detect up-projection weight name: fused "experts_gate_up" (Qwen3-MoE, GPT-OSS)
    // or separate "experts_up" (Nemotron-H).
    const std::string up_weight_name = mWeights.has(layer_prefix + "experts_gate_up")
        ? "experts_gate_up" : "experts_up";
    const bool separate_up_projection = (up_weight_name == "experts_up");
    const bool force_llep = (std::getenv("SUROGATE_FORCE_LLEP") != nullptr);
    const bool force_llep_for_separate_up =
        (std::getenv("SUROGATE_FORCE_LLEP_EXPERTS_UP") != nullptr);
    const bool llep_supported_for_layer =
        force_llep && (!separate_up_projection || force_llep_for_separate_up);
    const float threshold = mOptions.EPLoadBalanceThreshold;
    {
        static bool printed_ep_dispatch_marker = false;
        if (!printed_ep_dispatch_marker && mComm && mComm->rank() == 0) {
            fprintf(stderr,
                    "[EP] ep_dispatch active: layer=%d up_weight=%s ep_size=%d\n",
                    layer_idx, up_weight_name.c_str(), ep_size);
            printed_ep_dispatch_marker = true;
        }
    }
    if (!force_llep && threshold < 100.0f) {
        static bool warned_llep_disabled_default = false;
        if (!warned_llep_disabled_default && mComm && mComm->rank() == 0) {
            fprintf(stderr,
                    "[EP] LLEP is disabled by default for correctness; using static EP mapping. "
                    "Set SUROGATE_FORCE_LLEP=1 to re-enable the experimental LLEP path.\n");
            warned_llep_disabled_default = true;
        }
    }
    if (separate_up_projection && force_llep && !force_llep_for_separate_up) {
        static bool warned_experts_up_detected = false;
        if (!warned_experts_up_detected && mComm && mComm->rank() == 0) {
            fprintf(stderr,
                    "[EP] Detected experts_up layer (Nemotron-style MoE); "
                    "LLEP will remain disabled for this path unless "
                    "SUROGATE_FORCE_LLEP_EXPERTS_UP=1 is set.\n");
            warned_experts_up_detected = true;
        }
    }

    // ---- Get expert offsets from host cache (populated by moe_permute) ----
    auto offsets_it = mMoEHostOffsetsCache.find(ep_key);
    if (offsets_it == mMoEHostOffsetsCache.end()) {
        // Backward-compatible fallback for non-EP-keyed caches.
        offsets_it = mMoEHostOffsetsCache.find(layer_idx);
    }
    if (offsets_it == mMoEHostOffsetsCache.end()) {
        throw std::runtime_error(
            "ep_dispatch: host expert offsets not found for layer " + std::to_string(layer_idx));
    }
    const std::vector<int>& expert_offsets = offsets_it->second;

    // ---- Compute local per-expert counts ----
    std::vector<int> local_expert_counts(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        local_expert_counts[e] = expert_offsets[e + 1] - expert_offsets[e];
    }
    // ---- LLEP imbalance detection (only when threshold is reachable) ----
    bool use_llep = false;
    std::vector<int> global_expert_counts;

    if (llep_supported_for_layer && threshold < 100.0f) {
        // All-reduce expert counts across EP group
        Tensor counts_gpu = mRunState.temp_alloc(ETensorDType::INT32,
            {static_cast<long>(num_experts)}, "ep_expert_counts");
        CUDA_CHECK(cudaMemcpyAsync(counts_gpu.Data, local_expert_counts.data(),
                                    num_experts * sizeof(int), cudaMemcpyHostToDevice,
                                    mRunState.MainStream));
        mComm->all_reduce_sum_int_ep(counts_gpu.get<int>(), num_experts, mRunState.MainStream);

        global_expert_counts.resize(num_experts);
        CUDA_CHECK(cudaMemcpyAsync(global_expert_counts.data(), counts_gpu.Data,
                                    num_experts * sizeof(int), cudaMemcpyDeviceToHost,
                                    mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mTemps.push_back(counts_gpu);

        const float imbalance = ep::compute_imbalance_ratio(
            global_expert_counts.data(), num_experts, ep_size, num_local);
        use_llep = (imbalance >= threshold);
    }

    auto& ep_state = mEpStates[ep_key];
    if (!use_llep && ep_state.llep_send_reorder_gpu) {
        CUDA_CHECK(cudaFree(ep_state.llep_send_reorder_gpu));
        ep_state.llep_send_reorder_gpu = nullptr;
        ep_state.llep_send_reorder_bytes = 0;
    }

    // Clear ALL previous LLEP states (free per-layer LoRA + foreign weight GPU memory).
    // Foreign expert weight buffers are large (~20MB each) and accumulate across 23 MoE
    // layers, causing OOM if kept alive. Only the last MoE layer's LLEP state survives.
    // Earlier layers' backward uses mEPLayerMeta (lightweight) to reconstruct weight pointers.
    {
        for (auto& [_, state] : mLLEPStates) {
            state.free_lora_gpu();
            state.free_foreign_gpu();
        }
        mLLEPStates.clear();
    }

    // ---- Determine expert→GPU mapping (default or LLEP) ----
    // expert_to_gpu[e] = GPU that will process expert e's tokens
    std::vector<int> expert_to_gpu(num_experts);
    ep::LPTPlan plan;

    if (use_llep) {
        plan = ep::compute_lpt_plan(
            global_expert_counts.data(), num_experts, ep_size, ep_rank, num_local);
        expert_to_gpu = plan.expert_to_gpu;
    } else {
        for (int e = 0; e < num_experts; ++e) {
            expert_to_gpu[e] = e / num_local;
        }
    }

    // ---- Determine merged expert set for this GPU ----
    std::vector<int> merged_experts;
    for (int e = 0; e < num_experts; ++e) {
        if (expert_to_gpu[e] == ep_rank) {
            merged_experts.push_back(e);
        }
    }
    const int num_merged = static_cast<int>(merged_experts.size());

    std::vector<int> global_to_merged(num_experts, -1);
    for (int m = 0; m < num_merged; ++m) {
        global_to_merged[merged_experts[m]] = m;
    }

    // ---- Save lightweight EP metadata for backward (survives LLEP state clearing) ----
    {
        auto& meta = mEPLayerMeta[ep_key];
        meta.num_merged = num_merged;
        meta.native_start = ep_rank * num_local;
        meta.num_local = num_local;
        meta.merged_to_global = merged_experts;
    }

    // ---- Compute send splits ----
    std::vector<int> send_splits(ep_size, 0);
    for (int e = 0; e < num_experts; ++e) {
        send_splits[expert_to_gpu[e]] += local_expert_counts[e];
    }

    // ---- Reorder send buffer if LLEP changed the routing ----
    // In standard EP, experts are naturally grouped by destination peer
    // (experts [p*num_local, (p+1)*num_local) → peer p). LLEP may change this.
    const Tensor* send_ptr = &permuted_input;
    Tensor send_buf;

    if (use_llep) {
        // Upload expert_offsets and expert_to_gpu to GPU (small H2D copies)
        Tensor offsets_gpu = mRunState.temp_alloc(ETensorDType::INT32,
            {static_cast<long>(num_experts + 1)}, "ep_offsets_gpu");
        CUDA_CHECK(cudaMemcpyAsync(offsets_gpu.Data, expert_offsets.data(),
                                    (num_experts + 1) * sizeof(int), cudaMemcpyHostToDevice,
                                    mRunState.MainStream));

        Tensor e2g_gpu = mRunState.temp_alloc(ETensorDType::INT32,
            {static_cast<long>(num_experts)}, "ep_e2g_gpu");
        CUDA_CHECK(cudaMemcpyAsync(e2g_gpu.Data, expert_to_gpu.data(),
                                    num_experts * sizeof(int), cudaMemcpyHostToDevice,
                                    mRunState.MainStream));

        // Allocate output send buffer
        send_buf = mRunState.temp_alloc(permuted_input.DType,
            {static_cast<long>(total_send), static_cast<long>(hidden_size)}, "ep_send_buf");
        mTemps.push_back(send_buf);

        // Temporary buffer for per-expert write offsets
        Tensor pwo_gpu = mRunState.temp_alloc(ETensorDType::INT32,
            {static_cast<long>(num_experts)}, "ep_pwo_gpu");

        // Output buffer for send_order mapping (needed for backward)
        Tensor send_order_gpu = mRunState.temp_alloc(ETensorDType::INT32,
            {static_cast<long>(total_send)}, "ep_send_order_gpu");

        // Fused GPU kernel: compute write offsets + scatter tokens + record send_order
        if (permuted_input.DType == ETensorDType::BF16) {
            ep_fused_prepare_send_buffer_bf16(
                send_buf.get<nv_bfloat16>(), permuted_input.get<nv_bfloat16>(),
                offsets_gpu.get<int>(), e2g_gpu.get<int>(),
                pwo_gpu.get<int>(), send_order_gpu.get<int>(),
                num_experts, ep_size, hidden_size, total_send,
                mRunState.MainStream);
        } else {
            ep_fused_prepare_send_buffer_fp32(
                send_buf.get<float>(), permuted_input.get<float>(),
                offsets_gpu.get<int>(), e2g_gpu.get<int>(),
                pwo_gpu.get<int>(), send_order_gpu.get<int>(),
                num_experts, ep_size, hidden_size, total_send,
                mRunState.MainStream);
        }
        send_ptr = &send_buf;

        // Persist LLEP send order for backward (to invert the reorder)
        persist_gpu_buffer(ep_state.llep_send_reorder_gpu, ep_state.llep_send_reorder_bytes,
                           send_order_gpu.Data, send_order_gpu.bytes(), mRunState.MainStream);

        mTemps.push_back(offsets_gpu);
        mTemps.push_back(e2g_gpu);
        mTemps.push_back(pwo_gpu);
        mTemps.push_back(send_order_gpu);
    }

    // ---- Exchange splits + per-expert counts (batched: 2 A2As, single sync) ----
    // Issue both A2As back-to-back on the stream, then batch D2H copies before one sync.
    // A2A 1: exchange per-GPU token splits
    Tensor send_splits_gpu = mRunState.temp_alloc(ETensorDType::INT32,
        {static_cast<long>(ep_size)}, "ep_send_splits_gpu");
    Tensor recv_splits_gpu = mRunState.temp_alloc(ETensorDType::INT32,
        {static_cast<long>(ep_size)}, "ep_recv_splits_gpu");
    CUDA_CHECK(cudaMemcpyAsync(send_splits_gpu.Data, send_splits.data(),
                                ep_size * sizeof(int), cudaMemcpyHostToDevice,
                                mRunState.MainStream));

    std::vector<int> ones(ep_size, 1);
    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(send_splits_gpu.Data),
        reinterpret_cast<std::byte*>(recv_splits_gpu.Data),
        ones.data(), ones.data(), sizeof(int), mRunState.MainStream);

    // A2A 2: exchange per-expert counts (fixed-size, independent of A2A 1 results)
    Tensor send_ec_gpu = mRunState.temp_alloc(ETensorDType::INT32,
        {static_cast<long>(num_experts * ep_size)}, "ep_send_ec_gpu");
    Tensor recv_ec_gpu = mRunState.temp_alloc(ETensorDType::INT32,
        {static_cast<long>(num_experts * ep_size)}, "ep_recv_ec_gpu");
    for (int p = 0; p < ep_size; ++p) {
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<std::byte*>(send_ec_gpu.Data) + static_cast<size_t>(p) * num_experts * sizeof(int),
            local_expert_counts.data(),
            num_experts * sizeof(int), cudaMemcpyHostToDevice,
            mRunState.MainStream));
    }

    std::vector<int> ec_send_splits(ep_size, num_experts);
    std::vector<int> ec_recv_splits(ep_size, num_experts);
    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(send_ec_gpu.Data),
        reinterpret_cast<std::byte*>(recv_ec_gpu.Data),
        ec_send_splits.data(), ec_recv_splits.data(),
        sizeof(int), mRunState.MainStream);

    // Batch D2H copies: both A2A results in one sync
    std::vector<int> recv_splits(ep_size);
    CUDA_CHECK(cudaMemcpyAsync(recv_splits.data(), recv_splits_gpu.Data,
                                ep_size * sizeof(int), cudaMemcpyDeviceToHost,
                                mRunState.MainStream));

    std::vector<int> recv_all_counts(num_experts * ep_size);
    CUDA_CHECK(cudaMemcpyAsync(recv_all_counts.data(), recv_ec_gpu.Data,
                                num_experts * ep_size * sizeof(int), cudaMemcpyDeviceToHost,
                                mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
    mTemps.push_back(send_splits_gpu);
    mTemps.push_back(recv_splits_gpu);
    mTemps.push_back(send_ec_gpu);
    mTemps.push_back(recv_ec_gpu);

    const int total_recv = std::accumulate(recv_splits.begin(), recv_splits.end(), 0);

    // ---- Start async weight transfer for LLEP (overlaps with A2A below) ----
    ep::ForeignExpertWeights foreign_weights;
    Tensor* native_gate_up_ptr = nullptr;
    Tensor* native_down_ptr = nullptr;
    bool wt_started = false;
    if (use_llep && (!plan.weights_to_send.empty() || !plan.weights_to_receive.empty())) {
        native_gate_up_ptr = &mWeights.get(layer_prefix + up_weight_name);
        native_down_ptr = &mWeights.get(layer_prefix + "experts_down");

        // Lazy create weight transfer stream
        if (!mWeightTransferStream) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&mWeightTransferStream, cudaStreamNonBlocking));
        }
        // Make WeightTransferStream wait for MainStream (native weights must be resolved)
        cudaEvent_t wt_start;
        CUDA_CHECK(cudaEventCreateWithFlags(&wt_start, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(wt_start, mRunState.MainStream));
        CUDA_CHECK(cudaStreamWaitEvent(mWeightTransferStream, wt_start));
        CUDA_CHECK(cudaEventDestroy(wt_start));

        // Prefer quantized transfer (2-8x less P2P bandwidth) when available
        auto* provider = mWeights.qlora_provider();
        const qlora::QuantizedTensor* qt_gu = provider
            ? provider->try_get_quantized(layer_prefix + up_weight_name) : nullptr;
        const qlora::QuantizedTensor* qt_dn = provider
            ? provider->try_get_quantized(layer_prefix + "experts_down") : nullptr;
        qlora::IQuantizer* quantizer = provider ? provider->get_quantizer() : nullptr;

        if (qt_gu && qt_dn && qt_gu->is_quantized() && qt_dn->is_quantized()
            && quantizer && !qt_gu->is_on_host()) {
            ep::transfer_expert_weights_quantized(
                plan, *mComm, mRunState, mWeightTransferStream,
                foreign_weights, *qt_gu, *qt_dn,
                *native_gate_up_ptr, *native_down_ptr,
                quantizer, num_local, ep_rank);
        } else {
            ep::transfer_expert_weights(
                plan, *mComm, mRunState, mWeightTransferStream,
                foreign_weights, *native_gate_up_ptr, *native_down_ptr,
                num_local, ep_rank);
        }
        // Transfer per-expert LoRA adapters alongside base weights (on same stream).
        // LoRA is low-rank so overhead is small (<1% of base weight transfer).
        if (mLoRAConfig && mLoRAWeights && mLoRAConfig->enabled() && mLoRAWeights->enabled()) {
            auto& lora_block = mLoRAWeights->get_block(layer_idx, mWeightTransferStream);
            if (lora_block.moe.use_grouped) {
                const auto& g = lora_block.moe.grouped;
                Tensor null_tensor;
                ep::transfer_expert_lora(
                    plan, *mComm, mRunState, mWeightTransferStream,
                    foreign_weights,
                    g.gate_up.has_value() ? g.gate_up->A : null_tensor,
                    g.gate_up.has_value() ? g.gate_up->B : null_tensor,
                    g.gate.has_value() ? g.gate->A : null_tensor,
                    g.gate.has_value() ? g.gate->B : null_tensor,
                    g.up.has_value() ? g.up->A : null_tensor,
                    g.up.has_value() ? g.up->B : null_tensor,
                    g.down.has_value() ? g.down->A : null_tensor,
                    g.down.has_value() ? g.down->B : null_tensor,
                    num_local, ep_rank);
            }
        }

        wt_started = true;
    }

    // ---- A2A hidden states (concurrent with weight transfer above) ----
    const size_t recv_hidden_bytes = static_cast<size_t>(total_recv) * hidden_size * elem_sz;
    void* recv_hidden_ptr = ep_buf_acquire(recv_hidden_bytes);

    Tensor recv_hidden = make_raw_tensor(recv_hidden_ptr, permuted_input.DType,
        {static_cast<long>(total_recv), static_cast<long>(hidden_size)}, device);

    std::vector<int> send_elem_splits(ep_size), recv_elem_splits(ep_size);
    for (int p = 0; p < ep_size; ++p) {
        send_elem_splits[p] = send_splits[p] * hidden_size;
        recv_elem_splits[p] = recv_splits[p] * hidden_size;
    }

    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(send_ptr->Data),
        reinterpret_cast<std::byte*>(recv_hidden.Data),
        send_elem_splits.data(), recv_elem_splits.data(),
        elem_sz, mRunState.MainStream);

    // ---- Build recv-side merged expert IDs for re-sort ----
    // After A2A, tokens from peer 0 first, then peer 1, etc.
    // Within each peer's chunk, tokens are in the order the peer sent them.
    // The peer grouped tokens by destination, and within each dest-peer group,
    // experts appear in global expert-ID order. For our tokens (dest == ep_rank),
    // the experts arrive in expert-ID order within each source peer's chunk.
    std::vector<int> recv_merged_ids(total_recv);
    {
        int pos = 0;
        for (int p = 0; p < ep_size; ++p) {
            // Peer p sent tokens for experts assigned to us (expert_to_gpu[e] == ep_rank)
            // in ascending expert-ID order
            for (int e = 0; e < num_experts; ++e) {
                if (expert_to_gpu[e] != ep_rank) continue;
                int count = recv_all_counts[p * num_experts + e];
                int merged_idx = global_to_merged[e];
                if (merged_idx < 0 || merged_idx >= num_merged) {
                    throw std::runtime_error(
                        "ep_dispatch: invalid merged expert index at layer "
                        + std::to_string(layer_idx) + " e=" + std::to_string(e)
                        + " merged_idx=" + std::to_string(merged_idx)
                        + " num_merged=" + std::to_string(num_merged));
                }
                for (int t = 0; t < count; ++t) {
                    if (pos < total_recv) {
                        recv_merged_ids[pos++] = merged_idx;
                    }
                }
            }
        }
        if (pos != total_recv) {
            throw std::runtime_error(
                "ep_dispatch: recv_merged_ids fill mismatch at layer "
                + std::to_string(layer_idx) + " pos=" + std::to_string(pos)
                + " total_recv=" + std::to_string(total_recv));
        }
    }

    // All EP dispatch temporaries and outputs use cudaMalloc (not the stack allocator)
    // to avoid stack OOM: EP adds ~14+ MB per layer that doesn't exist in the non-EP path.

    // Small temporary: recv_routing [total_recv, 1] INT32
    const size_t recv_routing_bytes = static_cast<size_t>(total_recv) * sizeof(int);
    void* recv_routing_ptr = ep_buf_acquire(recv_routing_bytes);
    CUDA_CHECK(cudaMemcpyAsync(recv_routing_ptr, recv_merged_ids.data(),
                                total_recv * sizeof(int), cudaMemcpyHostToDevice,
                                mRunState.MainStream));

    Tensor recv_routing = make_raw_tensor(recv_routing_ptr, ETensorDType::INT32,
        {static_cast<long>(total_recv), 1L}, device);

    // ---- Re-permute received tokens by merged expert index ----
    // Per-layer persistent output: sorted_recv [total_recv, hidden].
    // This tensor is consumed by MoE GEMM and must remain valid until backward.
    const size_t sorted_recv_need = static_cast<size_t>(total_recv) * hidden_size * elem_sz;
    auto& ep_state_out = mEpStates[ep_key];
    if (ep_state_out.sorted_recv_bytes < sorted_recv_need) {
        if (ep_state_out.sorted_recv_gpu) CUDA_CHECK(cudaFree(ep_state_out.sorted_recv_gpu));
        CUDA_CHECK(cudaMalloc(&ep_state_out.sorted_recv_gpu, sorted_recv_need));
        ep_state_out.sorted_recv_bytes = sorted_recv_need;
    }
    Tensor sorted_recv = make_raw_tensor(ep_state_out.sorted_recv_gpu, permuted_input.DType,
        {static_cast<long>(total_recv), static_cast<long>(hidden_size)}, device);

    // Per-layer persistent output: local_scatter [total_recv] INT32
    const size_t local_scatter_need = static_cast<size_t>(total_recv) * sizeof(int);
    if (ep_state_out.local_scatter_bytes < local_scatter_need) {
        if (ep_state_out.local_scatter_gpu) CUDA_CHECK(cudaFree(ep_state_out.local_scatter_gpu));
        CUDA_CHECK(cudaMalloc(&ep_state_out.local_scatter_gpu, local_scatter_need));
        ep_state_out.local_scatter_bytes = local_scatter_need;
    }
    Tensor local_scatter = make_raw_tensor(ep_state_out.local_scatter_gpu, ETensorDType::INT32,
        {static_cast<long>(total_recv)}, device);

    // Small temporaries for MoE index computation
    const size_t helper_bytes =
        (num_merged + (num_merged + 1) + num_merged + total_recv) * sizeof(int);
    void* helper_buf = ep_buf_acquire(helper_bytes);

    int* merged_counts_ptr = static_cast<int*>(helper_buf);
    int* merged_offsets_ptr = merged_counts_ptr + num_merged;
    int* merged_positions_ptr = merged_offsets_ptr + (num_merged + 1);
    int* local_gather_ptr = merged_positions_ptr + num_merged;

    Tensor merged_counts_t = make_raw_tensor(merged_counts_ptr, ETensorDType::INT32,
        {static_cast<long>(num_merged)}, device);
    Tensor merged_offsets_t = make_raw_tensor(merged_offsets_ptr, ETensorDType::INT32,
        {static_cast<long>(num_merged + 1)}, device);
    Tensor merged_positions_t = make_raw_tensor(merged_positions_ptr, ETensorDType::INT32,
        {static_cast<long>(num_merged)}, device);
    Tensor local_gather = make_raw_tensor(local_gather_ptr, ETensorDType::INT32,
        {static_cast<long>(total_recv)}, device);

    fill_zero(merged_counts_t, mRunState.MainStream);
    fill_zero(merged_positions_t, mRunState.MainStream);
    CUDA_CHECK(cudaMemsetAsync(local_gather.Data, 0, local_gather.bytes(), mRunState.MainStream));
    CUDA_CHECK(cudaMemsetAsync(local_scatter.Data, 0xFF,
                                local_scatter.bytes(), mRunState.MainStream));

    moe_compute_expert_counts(merged_counts_t.get<int>(),
                              recv_routing.get<int>(),
                              total_recv, 1, num_merged, mRunState.MainStream);

    moe_compute_expert_offsets(merged_offsets_t.get<int>(),
                               merged_counts_t.get<int>(),
                               num_merged, mRunState.MainStream);

    moe_build_indices(local_gather.get<int>(),
                      local_scatter.get<int>(),
                      recv_routing.get<int>(),
                      merged_offsets_t.get<int>(),
                      merged_positions_t.get<int>(),
                      total_recv, 1, num_merged, mRunState.MainStream);

    if (recv_hidden.DType == ETensorDType::BF16) {
        moe_permute_tokens(sorted_recv.get<nv_bfloat16>(),
                           recv_hidden.get<nv_bfloat16>(),
                           local_gather.get<int>(),
                           total_recv, total_recv, hidden_size, 1,
                           mRunState.MainStream);
    } else {
        moe_permute_tokens(sorted_recv.get<float>(),
                           recv_hidden.get<float>(),
                           local_gather.get<int>(),
                           total_recv, total_recv, hidden_size, 1,
                           mRunState.MainStream);
    }

    // Return intermediate buffers to pool (stream ordering ensures safe reuse)
    ep_buf_release(recv_hidden_ptr, recv_hidden_bytes);
    ep_buf_release(recv_routing_ptr, recv_routing_bytes);

    // ---- Cache host offsets for grouped GEMM ----
    {
        std::vector<int> merged_offsets_host(num_merged + 1);
        CUDA_CHECK(cudaMemcpyAsync(merged_offsets_host.data(),
                                    merged_offsets_t.get<int>(),
                                    (num_merged + 1) * sizeof(int),
                                    cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mMoEHostOffsetsCache[ep_key] = merged_offsets_host;

    }

    // ---- Persist expert offsets and EP state ----
    // IMPORTANT: Must persist all data from helper_buf BEFORE releasing to pool.
    // local_gather.Data points into helper_buf — pool may recycle it for next layer.
    const std::string saved_offsets_key = moe_saved_key(layer_idx, "moe_expert_offsets");
    save_persistent_buffer(mMoeSavedBuffers, mMoeSavedSizes,
                           saved_offsets_key,
                           merged_offsets_t, mRunState.MainStream);

    // Persist EP backward state (send_order = local_gather, recv_reorder = local_scatter)
    // Must happen BEFORE pool release of helper_buf since local_gather.Data is inside it.
    auto& ep_state_persist = mEpStates[ep_key];
    persist_gpu_buffer(ep_state_persist.send_order_gpu, ep_state_persist.send_order_bytes,
                       local_gather.Data, local_gather.bytes(), mRunState.MainStream);
    persist_gpu_buffer(ep_state_persist.recv_reorder_gpu, ep_state_persist.recv_reorder_bytes,
                       local_scatter.Data, local_scatter.bytes(), mRunState.MainStream);

    // Return helper buf to pool — bind from persistent copies instead
    ep_buf_release(helper_buf, helper_bytes);

    // Bind from persistent copies (helper_buf returned to pool, may be overwritten)
    {
        Tensor offsets_persisted = merged_offsets_t;
        offsets_persisted.Data = static_cast<std::byte*>(
            mMoeSavedBuffers[saved_offsets_key]);
        bind_tensor("moe_expert_offsets", offsets_persisted);
    }

    // ---- Sync weight transfer and build merged weights for LLEP ----
    if (wt_started) {
        // Synchronize: make MainStream wait for weight transfer to complete
        cudaEvent_t wt_done;
        CUDA_CHECK(cudaEventCreateWithFlags(&wt_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(wt_done, mWeightTransferStream));
        CUDA_CHECK(cudaStreamWaitEvent(mRunState.MainStream, wt_done));
        CUDA_CHECK(cudaEventDestroy(wt_done));

        // Check for errors from weight transfer stream
        {
            auto wt_err = cudaGetLastError();
            if (wt_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("ep_dispatch: weight transfer stream error at layer ")
                    + std::to_string(layer_idx) + ": " + cudaGetErrorString(wt_err));
            }
        }

        const Tensor& native_gate_up = *native_gate_up_ptr;
        const Tensor& native_down = *native_down_ptr;

        // Build per-expert weight pointer arrays (no contiguous merged buffer needed).
        // Each pointer targets a single expert's weight slice in its original location:
        //  - Native experts → pointer into QLoRA-resolved dequant buffer
        //  - Foreign experts → pointer into P2P receive buffer (kept alive in LLEP state)
        // This eliminates the ~465 MB merged buffer allocation that caused OOM.
        const long gu_rows = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[1] : native_gate_up.Sizes[0];
        const long gu_cols = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[2] : native_gate_up.Sizes[1];
        const long dn_rows = (native_down.Rank >= 3) ? native_down.Sizes[1] : native_down.Sizes[0];
        const long dn_cols = (native_down.Rank >= 3) ? native_down.Sizes[2] : native_down.Sizes[1];
        const size_t gu_expert_bytes = static_cast<size_t>(gu_rows * gu_cols) * elem_sz;
        const size_t dn_expert_bytes = static_cast<size_t>(dn_rows * dn_cols) * elem_sz;

        auto& llep = mLLEPStates[ep_key];
        llep.active = true;
        llep.num_merged_experts = num_merged;
        llep.merged_to_global = merged_experts;
        llep.global_to_merged = global_to_merged;
        llep.weight_dtype = native_gate_up.DType;

        llep.gate_up_weight_ptrs.resize(num_merged);
        llep.down_weight_ptrs.resize(num_merged);

        const int native_start = ep_rank * num_local;
        for (int m = 0; m < num_merged; ++m) {
            const int global_e = merged_experts[m];
            const int local_native = global_e - native_start;

            if (local_native >= 0 && local_native < num_local) {
                // Native expert — point directly into dequant buffer (no copy)
                llep.gate_up_weight_ptrs[m] = static_cast<const std::byte*>(native_gate_up.Data)
                    + static_cast<size_t>(local_native) * gu_expert_bytes;
                llep.down_weight_ptrs[m] = static_cast<const std::byte*>(native_down.Data)
                    + static_cast<size_t>(local_native) * dn_expert_bytes;
            } else {
                // Foreign expert — point into P2P receive buffer (kept alive below)
                auto wit = foreign_weights.weights.find(global_e);
                if (wit != foreign_weights.weights.end()) {
                    llep.gate_up_weight_ptrs[m] = wit->second.gate_up.Data;
                    llep.down_weight_ptrs[m] = wit->second.down.Data;
                } else {
                    // Should not happen — LPT plan guarantees all merged experts
                    // are either native or received. Use native[0] as safe fallback.
                    llep.gate_up_weight_ptrs[m] = native_gate_up.Data;
                    llep.down_weight_ptrs[m] = native_down.Data;
                }
            }
        }

        // ---- Build merged LoRA tensors (native + foreign expert LoRA) ----
        if (mLoRAConfig && mLoRAWeights && mLoRAConfig->enabled() && mLoRAWeights->enabled()) {
            auto& lora_block = mLoRAWeights->get_block(layer_idx, mRunState.MainStream);
            if (lora_block.moe.use_grouped && lora_block.moe.grouped.has_any()) {
                const auto& g = lora_block.moe.grouped;
                auto& ml = llep.merged_lora;
                llep.has_merged_lora = true;

                // Helper: build merged [num_merged, ...] from local [num_local, ...] + foreign.
                // Per-layer owned GPU allocations stored in llep.owned_lora_ptrs.
                // Unlike base weights, LoRA tensors are small (~9 MB/layer) so per-layer
                // storage is feasible and avoids the shared-buffer aliasing bug where a later
                // layer's reallocation would free memory still referenced by earlier layers.
                auto build_merged_pair = [&](
                    const std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& local_opt,
                    std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& merged_opt,
                    auto get_foreign_A, auto get_foreign_B) {
                    if (!local_opt.has_value() || !local_opt->has_value()) return;
                    const auto& local = *local_opt;
                    const long a_rows = local.A.Sizes[1];
                    const long a_cols = local.A.Sizes[2];
                    const long b_rows = local.B.Sizes[1];
                    const long b_cols = local.B.Sizes[2];
                    const size_t a_slice = static_cast<size_t>(a_rows * a_cols) * get_dtype_size(local.A.DType);
                    const size_t b_slice = static_cast<size_t>(b_rows * b_cols) * get_dtype_size(local.B.DType);

                    const size_t need_A = static_cast<size_t>(num_merged) * a_slice;
                    const size_t need_B = static_cast<size_t>(num_merged) * b_slice;
                    void* ptr_A = nullptr;
                    void* ptr_B = nullptr;
                    CUDA_CHECK(cudaMalloc(&ptr_A, need_A));
                    llep.owned_lora_ptrs.push_back(ptr_A);
                    CUDA_CHECK(cudaMalloc(&ptr_B, need_B));
                    llep.owned_lora_ptrs.push_back(ptr_B);

                    modules::LoRAGroupedLayerWeights<Tensor> merged;
                    merged.A = make_raw_tensor(ptr_A, local.A.DType,
                        {static_cast<long>(num_merged), a_rows, a_cols}, device);
                    merged.B = make_raw_tensor(ptr_B, local.B.DType,
                        {static_cast<long>(num_merged), b_rows, b_cols}, device);

                    for (int m = 0; m < num_merged; ++m) {
                        const int global_e = merged_experts[m];
                        const int local_native = global_e - native_start;
                        auto* a_dst = static_cast<std::byte*>(merged.A.Data) + static_cast<size_t>(m) * a_slice;
                        auto* b_dst = static_cast<std::byte*>(merged.B.Data) + static_cast<size_t>(m) * b_slice;

                        if (local_native >= 0 && local_native < num_local) {
                            const auto* a_src = static_cast<const std::byte*>(local.A.Data) + static_cast<size_t>(local_native) * a_slice;
                            const auto* b_src = static_cast<const std::byte*>(local.B.Data) + static_cast<size_t>(local_native) * b_slice;
                            CUDA_CHECK(cudaMemcpyAsync(a_dst, a_src, a_slice, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                            CUDA_CHECK(cudaMemcpyAsync(b_dst, b_src, b_slice, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                        } else {
                            auto wit = foreign_weights.weights.find(global_e);
                            if (wit != foreign_weights.weights.end()) {
                                const Tensor& fa = get_foreign_A(wit->second.lora);
                                const Tensor& fb = get_foreign_B(wit->second.lora);
                                if (!fa.is_null()) {
                                    CUDA_CHECK(cudaMemcpyAsync(a_dst, fa.Data, a_slice, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                                    CUDA_CHECK(cudaMemcpyAsync(b_dst, fb.Data, b_slice, cudaMemcpyDeviceToDevice, mRunState.MainStream));
                                }
                            }
                        }
                    }
                    merged_opt = std::move(merged);
                };

                build_merged_pair(g.gate_up, ml.gate_up,
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_up_A; },
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_up_B; });
                build_merged_pair(g.gate, ml.gate,
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_A; },
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_B; });
                build_merged_pair(g.up, ml.up,
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.up_A; },
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.up_B; });
                build_merged_pair(g.down, ml.down,
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.down_A; },
                    [](const ep::ExpertLoRA& l) -> const Tensor& { return l.down_B; });
            }
        }

        llep.merged_offsets_host = mMoEHostOffsetsCache[ep_key];
        llep.merged_offsets_gpu = mMoeSavedBuffers[saved_offsets_key];
        llep.merged_offsets_gpu_bytes = mMoeSavedSizes[saved_offsets_key];

        // Transfer foreign weight buffer ownership to LLEP state.
        // Per-expert pointers reference these buffers, so they must stay alive
        // until the LLEP state is cleared (at start of next layer's ep_dispatch).
        llep.owned_foreign_ptrs = std::move(foreign_weights.owned_gpu_ptrs);
        foreign_weights.owned_gpu_ptrs.clear();  // Prevent double-free
    } else if (ep_size > 1) {
        // When EP is active but no foreign experts were transferred, some architectures
        // use a native-only pointer path to avoid recipe issues on EP-modified offsets.
        //
        // IMPORTANT: with QLoRA expert offloading, resolved dequant pointers can rotate
        // across steps/layers. Caching native per-expert pointers here can leave stale
        // pointers and corrupt EP MoE forward/backward. Disable the native-pointer
        // fallback whenever expert offloading is active.
        bool allow_native_pointer_path = true;
        if (auto* provider = mWeights.qlora_provider()) {
            if (provider->has_offloading()) {
                allow_native_pointer_path = false;
                static bool warned_ep_native_ptrs_offload = false;
                if (!warned_ep_native_ptrs_offload && mComm && mComm->rank() == 0) {
                    fprintf(stderr,
                            "[EP] QLoRA offloading detected: disabling native-pointer EP fallback "
                            "to avoid stale expert pointers.\n");
                    warned_ep_native_ptrs_offload = true;
                }
            }
        }

        if (allow_native_pointer_path) {
            native_gate_up_ptr = &mWeights.get(layer_prefix + up_weight_name);
            native_down_ptr = &mWeights.get(layer_prefix + "experts_down");

            const Tensor& native_gate_up = *native_gate_up_ptr;
            const Tensor& native_down = *native_down_ptr;

            const long gu_rows = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[1] : native_gate_up.Sizes[0];
            const long gu_cols = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[2] : native_gate_up.Sizes[1];
            const long dn_rows = (native_down.Rank >= 3) ? native_down.Sizes[1] : native_down.Sizes[0];
            const long dn_cols = (native_down.Rank >= 3) ? native_down.Sizes[2] : native_down.Sizes[1];
            const size_t gu_expert_bytes = static_cast<size_t>(gu_rows * gu_cols) * elem_sz;
            const size_t dn_expert_bytes = static_cast<size_t>(dn_rows * dn_cols) * elem_sz;

            auto& llep = mLLEPStates[ep_key];
            llep.active = true;
            llep.num_merged_experts = num_merged;
            llep.merged_to_global = merged_experts;
            llep.global_to_merged = global_to_merged;
            llep.weight_dtype = native_gate_up.DType;

            llep.gate_up_weight_ptrs.resize(num_merged);
            llep.down_weight_ptrs.resize(num_merged);

            const int native_start = ep_rank * num_local;
            for (int m = 0; m < num_merged; ++m) {
                const int global_e = merged_experts[m];
                const int local_native = global_e - native_start;
                llep.gate_up_weight_ptrs[m] = static_cast<const std::byte*>(native_gate_up.Data)
                    + static_cast<size_t>(local_native) * gu_expert_bytes;
                llep.down_weight_ptrs[m] = static_cast<const std::byte*>(native_down.Data)
                    + static_cast<size_t>(local_native) * dn_expert_bytes;
            }

            llep.merged_offsets_host = mMoEHostOffsetsCache[ep_key];
            llep.merged_offsets_gpu = mMoeSavedBuffers[saved_offsets_key];
            llep.merged_offsets_gpu_bytes = mMoeSavedSizes[saved_offsets_key];
        }
    }

    // ---- Save remaining EP state for backward / combine ----
    // Note: send_order_gpu and recv_reorder_gpu already persisted above (before helper_buf free).
    ep_state.send_splits = send_splits;
    ep_state.recv_splits = recv_splits;
    ep_state.total_send = total_send;
    ep_state.total_recv = total_recv;

    // ---- Store outputs ----
    store_tensor(op.outputs[0], sorted_recv);
    store_tensor(op.outputs[1], local_scatter);

}

void CompiledExecutor::dispatch_ep_dispatch_backward(const CompiledOp& op) {
    // Backward of ep_dispatch: un-sort + reverse A2A + (LLEP) undo send reorder
    // Input: d_recv_sorted [total_recv, C]
    // Output: d_permuted_input [total_send, C]
    //
    // NOTE: We use cudaMalloc/cudaFree for intermediate buffers instead of the
    // stack allocator. The stack has limited capacity and EP backward temporaries
    // (~20+ MB each) would exhaust it, causing OOM for subsequent ops in the same
    // layer (e.g., attention matmul backward).
    Tensor& d_recv_sorted = resolve_tensor(op.inputs[0]);

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(d_recv_sorted.Sizes[1]);

    if (ep_size <= 1 || !mComm || !mComm->ep_enabled()) {
        Tensor& d_permuted = ensure_output_tensor(op.outputs[0]);
        d_permuted = d_recv_sorted;
        store_tensor(op.outputs[0], d_permuted);
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
    const int input_total_recv = static_cast<int>(d_recv_sorted.Sizes[0]);
    const int ep_key_default = ep_state_key(layer_idx);
    int ep_key_selected = ep_key_default;

    auto it = mEpStates.find(ep_key_default);
    if (mOptions.EPSize > 1 && layer_idx >= 0) {
        const int ep_key_r0 = (layer_idx << 1);
        const int ep_key_r1 = ep_key_r0 | 1;
        auto it_r0 = mEpStates.find(ep_key_r0);
        auto it_r1 = mEpStates.find(ep_key_r1);

        // Backward can consume replay-generated activations even when mInReplay=false.
        // Pick the EP state whose total_recv matches the current input rows.
        if (it_r1 != mEpStates.end() && it_r1->second.total_recv == input_total_recv) {
            it = it_r1;
            ep_key_selected = ep_key_r1;
        } else if (it_r0 != mEpStates.end() && it_r0->second.total_recv == input_total_recv) {
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
            "ep_dispatch_backward: no EP state for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = it->second;
    auto& ep_state_mut = mEpStates[ep_key_selected];
    if (ep_state.total_recv != input_total_recv) {
        std::ostringstream oss;
        oss << "ep_dispatch_backward: EP state/input mismatch at layer " << layer_idx
            << " (selected_key=" << ep_key_selected
            << ", state.total_recv=" << ep_state.total_recv
            << ", input.rows=" << input_total_recv << ")";
        throw std::runtime_error(oss.str());
    }
    const int elem_sz = (d_recv_sorted.DType == ETensorDType::BF16) ? 2 : 4;

    // ---- 1. Un-sort gradient (reverse local re-permutation) ----
    const size_t unsorted_bytes = static_cast<size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* d_recv_unsorted_ptr = ep_buf_acquire(unsorted_bytes);
    Tensor d_recv_unsorted = make_raw_tensor(d_recv_unsorted_ptr, d_recv_sorted.DType,
        {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
        d_recv_sorted.Device);

    if (d_recv_sorted.DType == ETensorDType::BF16) {
        moe_permute_tokens(d_recv_unsorted.get<nv_bfloat16>(),
                           d_recv_sorted.get<nv_bfloat16>(),
                           static_cast<int*>(ep_state.recv_reorder_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    } else {
        moe_permute_tokens(d_recv_unsorted.get<float>(),
                           d_recv_sorted.get<float>(),
                           static_cast<int*>(ep_state.recv_reorder_gpu),
                           ep_state.total_recv, ep_state.total_recv,
                           hidden_size, 1, mRunState.MainStream);
    }

    // ---- 2. Reverse A2A (swap send/recv splits) ----
    // Use per-layer persistent output: shared cross-layer buffers can be overwritten
    // before this tensor is consumed by downstream backward ops.
    const size_t send_need = static_cast<size_t>(ep_state.total_send) * hidden_size * elem_sz;
    if (ep_state_mut.dispatch_bwd_send_bytes < send_need) {
        if (ep_state_mut.dispatch_bwd_send_gpu) CUDA_CHECK(cudaFree(ep_state_mut.dispatch_bwd_send_gpu));
        CUDA_CHECK(cudaMalloc(&ep_state_mut.dispatch_bwd_send_gpu, send_need));
        ep_state_mut.dispatch_bwd_send_bytes = send_need;
    }
    Tensor d_send_buf = make_raw_tensor(ep_state_mut.dispatch_bwd_send_gpu, d_recv_sorted.DType,
        {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
        d_recv_sorted.Device);

    std::vector<int> reverse_send_elem(ep_size), reverse_recv_elem(ep_size);
    for (int p = 0; p < ep_size; ++p) {
        reverse_send_elem[p] = ep_state.recv_splits[p] * hidden_size;
        reverse_recv_elem[p] = ep_state.send_splits[p] * hidden_size;
    }

    mComm->all_to_all_single(
        reinterpret_cast<const std::byte*>(d_recv_unsorted.Data),
        reinterpret_cast<std::byte*>(d_send_buf.Data),
        reverse_send_elem.data(), reverse_recv_elem.data(),
        elem_sz, mRunState.MainStream);

    // Return buffer to pool (stream ordering ensures A2A completes before reuse)
    ep_buf_release(d_recv_unsorted_ptr, unsorted_bytes);

    // ---- 3. If LLEP was active, undo the send reorder ----
    if (ep_state.llep_send_reorder_gpu) {
        // Forward: send_buf[i] = permuted_input[send_order[i]]
        // Backward: d_permuted_input[send_order[i]] += d_send_buf[i]
        // Since send_order is a bijection, compute inverse and use gather.
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

        // Per-layer persistent output for LLEP reorder.
        if (ep_state_mut.dispatch_bwd_out_bytes < send_need) {
            if (ep_state_mut.dispatch_bwd_out_gpu) CUDA_CHECK(cudaFree(ep_state_mut.dispatch_bwd_out_gpu));
            CUDA_CHECK(cudaMalloc(&ep_state_mut.dispatch_bwd_out_gpu, send_need));
            ep_state_mut.dispatch_bwd_out_bytes = send_need;
        }
        Tensor d_permuted = make_raw_tensor(ep_state_mut.dispatch_bwd_out_gpu, d_recv_sorted.DType,
            {static_cast<long>(N), static_cast<long>(hidden_size)}, d_recv_sorted.Device);

        if (d_recv_sorted.DType == ETensorDType::BF16) {
            moe_permute_tokens(d_permuted.get<nv_bfloat16>(),
                               d_send_buf.get<nv_bfloat16>(),
                               static_cast<int*>(inv_gpu_ptr),
                               N, N, hidden_size, 1, mRunState.MainStream);
        } else {
            moe_permute_tokens(d_permuted.get<float>(),
                               d_send_buf.get<float>(),
                               static_cast<int*>(inv_gpu_ptr),
                               N, N, hidden_size, 1, mRunState.MainStream);
        }

        ep_buf_release(inv_gpu_ptr, inv_gpu_bytes);
        store_tensor(op.outputs[0], d_permuted);
    } else {
        store_tensor(op.outputs[0], d_send_buf);
    }
}

}  // namespace dsl
