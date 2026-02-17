// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert weight transfer implementation for LLEP load balancing.
// Uses batched P2P via ncclGroupStart/End on the weight_transfer_comm.
//
// Two transfer modes:
//   1. BF16 transfer: send dequantized BF16 slices per expert.
//      Simple but high bandwidth (2 bytes per element).
//   2. Quantized transfer: send raw quantized bytes + scales per expert.
//      NF4: ~4x bandwidth reduction, FP8: ~2x. Dequantized on receiver.

#include "runtime/ep/weight_transfer.h"

#include "runtime/dsl/dsl_run_state.h"
#include "runtime/qlora/quantized_tensor.h"
#include "runtime/qlora/generic_quantizer.h"
#include "utilities/comm.h"
#include "utilities/tensor.h"

namespace ep {

namespace {

/// Build a Tensor wrapping a raw GPU pointer with proper Rank/Device set.
inline Tensor make_tensor_from_ptr(void* ptr, ETensorDType dtype,
                                   const std::vector<long>& shape, int device = 0) {
    Tensor t{};
    t.Data = static_cast<std::byte*>(ptr);
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    for (int i = 0; i < t.Rank && i < 4; ++i) t.Sizes[i] = shape[static_cast<size_t>(i)];
    t.Device = device;
    return t;
}

/// Compute per-expert byte sizes for each component of a QuantizedTensor.
/// All common formats store data linearly by expert, so dividing total bytes
/// by num_experts gives per-expert byte counts.
struct PerExpertQuantSizes {
    size_t data_bytes = 0;
    size_t scale_bytes = 0;
    size_t meta_bytes = 0;
    size_t meta2_bytes = 0;
};

PerExpertQuantSizes compute_per_expert_sizes(
    const qlora::QuantizedTensor& qt, int num_experts) {
    PerExpertQuantSizes s;
    if (num_experts <= 0) return s;
    if (!qt.data.is_null())   s.data_bytes  = qt.data.bytes()  / static_cast<size_t>(num_experts);
    if (!qt.scales.is_null()) s.scale_bytes = qt.scales.bytes() / static_cast<size_t>(num_experts);
    if (!qt.meta.is_null())   s.meta_bytes  = qt.meta.bytes()  / static_cast<size_t>(num_experts);
    if (!qt.meta2.is_null())  s.meta2_bytes = qt.meta2.bytes() / static_cast<size_t>(num_experts);
    return s;
}

/// Total bytes for one quantized expert transfer (all components).
size_t total_quant_expert_bytes(const PerExpertQuantSizes& s) {
    return s.data_bytes + s.scale_bytes + s.meta_bytes + s.meta2_bytes;
}

/// Create a QuantizedTensor sub-view for a single expert's quantized data.
/// The returned QuantizedTensor has M = per_expert_M, K = qt.K, and pointers
/// into the original qt's buffers at the correct per-expert offset.
qlora::QuantizedTensor make_expert_slice_view(
    const qlora::QuantizedTensor& qt,
    int expert_idx,
    int per_expert_M,
    const PerExpertQuantSizes& sizes) {
    qlora::QuantizedTensor slice;
    slice.M = per_expert_M;
    slice.K = qt.K;
    slice.format = qt.format;
    slice.block_size = qt.block_size;
    slice.double_quant = qt.double_quant;
    slice.double_quant_group_size = qt.double_quant_group_size;
    slice.global_scale = qt.global_scale;

    const auto idx = static_cast<size_t>(expert_idx);

    if (!qt.data.is_null() && sizes.data_bytes > 0) {
        slice.data = qt.data;
        slice.data.Data = static_cast<std::byte*>(qt.data.Data) + idx * sizes.data_bytes;
        slice.data.Sizes[0] = static_cast<long>(sizes.data_bytes / get_dtype_size(qt.data.DType));
    }
    if (!qt.scales.is_null() && sizes.scale_bytes > 0) {
        slice.scales = qt.scales;
        slice.scales.Data = static_cast<std::byte*>(qt.scales.Data) + idx * sizes.scale_bytes;
        slice.scales.Sizes[0] = static_cast<long>(sizes.scale_bytes / get_dtype_size(qt.scales.DType));
    }
    if (!qt.meta.is_null() && sizes.meta_bytes > 0) {
        slice.meta = qt.meta;
        slice.meta.Data = static_cast<std::byte*>(qt.meta.Data) + idx * sizes.meta_bytes;
        slice.meta.Sizes[0] = static_cast<long>(sizes.meta_bytes / get_dtype_size(qt.meta.DType));
    }
    if (!qt.meta2.is_null() && sizes.meta2_bytes > 0) {
        slice.meta2 = qt.meta2;
        slice.meta2.Data = static_cast<std::byte*>(qt.meta2.Data) + idx * sizes.meta2_bytes;
        slice.meta2.Sizes[0] = static_cast<long>(sizes.meta2_bytes / get_dtype_size(qt.meta2.DType));
    }
    return slice;
}

/// Allocate GPU receive buffers matching a quantized expert slice layout.
/// Returns a QuantizedTensor with freshly allocated GPU buffers (cudaMalloc).
/// Tracks allocated pointers in `owned` for cleanup.
qlora::QuantizedTensor alloc_recv_quant_expert(
    const qlora::QuantizedTensor& template_qt,
    int per_expert_M,
    const PerExpertQuantSizes& sizes,
    std::vector<void*>& owned) {
    qlora::QuantizedTensor recv;
    recv.M = per_expert_M;
    recv.K = template_qt.K;
    recv.format = template_qt.format;
    recv.block_size = template_qt.block_size;
    recv.double_quant = template_qt.double_quant;
    recv.double_quant_group_size = template_qt.double_quant_group_size;
    recv.global_scale = template_qt.global_scale;

    auto alloc = [&](ETensorDType dtype, size_t bytes) -> Tensor {
        const long n = static_cast<long>(bytes / get_dtype_size(dtype));
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        owned.push_back(ptr);
        return make_tensor_from_ptr(ptr, dtype, {n}, 0);
    };

    if (sizes.data_bytes > 0 && !template_qt.data.is_null()) {
        recv.data = alloc(template_qt.data.DType, sizes.data_bytes);
    }
    if (sizes.scale_bytes > 0 && !template_qt.scales.is_null()) {
        recv.scales = alloc(template_qt.scales.DType, sizes.scale_bytes);
    }
    if (sizes.meta_bytes > 0 && !template_qt.meta.is_null()) {
        recv.meta = alloc(template_qt.meta.DType, sizes.meta_bytes);
    }
    if (sizes.meta2_bytes > 0 && !template_qt.meta2.is_null()) {
        recv.meta2 = alloc(template_qt.meta2.DType, sizes.meta2_bytes);
    }
    return recv;
}

/// Send all components of a quantized expert slice via P2P.
void send_quant_expert(
    NCCLCommunicator& comm,
    const qlora::QuantizedTensor& slice,
    const PerExpertQuantSizes& sizes,
    int dst_rank, cudaStream_t stream) {
    if (sizes.data_bytes > 0)
        comm.send_wt(slice.data.Data, sizes.data_bytes, dst_rank, stream);
    if (sizes.scale_bytes > 0)
        comm.send_wt(slice.scales.Data, sizes.scale_bytes, dst_rank, stream);
    if (sizes.meta_bytes > 0)
        comm.send_wt(slice.meta.Data, sizes.meta_bytes, dst_rank, stream);
    if (sizes.meta2_bytes > 0)
        comm.send_wt(slice.meta2.Data, sizes.meta2_bytes, dst_rank, stream);
}

/// Receive all components of a quantized expert into pre-allocated buffers.
void recv_quant_expert(
    NCCLCommunicator& comm,
    qlora::QuantizedTensor& recv,
    const PerExpertQuantSizes& sizes,
    int src_rank, cudaStream_t stream) {
    if (sizes.data_bytes > 0)
        comm.recv_wt(recv.data.Data, sizes.data_bytes, src_rank, stream);
    if (sizes.scale_bytes > 0)
        comm.recv_wt(recv.scales.Data, sizes.scale_bytes, src_rank, stream);
    if (sizes.meta_bytes > 0)
        comm.recv_wt(recv.meta.Data, sizes.meta_bytes, src_rank, stream);
    if (sizes.meta2_bytes > 0)
        comm.recv_wt(recv.meta2.Data, sizes.meta2_bytes, src_rank, stream);
}

/// Allocate a Tensor backed by cudaMalloc (not the stack allocator).
/// Tracks the pointer in received.owned_gpu_ptrs for cleanup.
Tensor alloc_foreign_tensor(
    ForeignExpertWeights& received,
    ETensorDType dtype,
    const std::vector<long>& shape) {
    size_t bytes = static_cast<size_t>(get_dtype_size(dtype));
    for (auto s : shape) bytes *= static_cast<size_t>(s);
    if (bytes == 0) return Tensor{};
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    received.owned_gpu_ptrs.push_back(ptr);
    return make_tensor_from_ptr(ptr, dtype, shape, 0);
}

}  // namespace

void transfer_expert_weights(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    dsl::DslRunState& run_state,
    cudaStream_t stream,
    ForeignExpertWeights& received,
    const Tensor& native_gate_up,
    const Tensor& native_down,
    int num_local_experts,
    int ep_rank) {

    received.free_gpu();

    if (plan.weights_to_send.empty() && plan.weights_to_receive.empty()) {
        return;
    }

    const long gate_up_rows = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[1] : 0;
    const long gate_up_cols = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[2] : 0;
    const long down_rows = (native_down.Rank >= 3) ? native_down.Sizes[1] : 0;
    const long down_cols = (native_down.Rank >= 3) ? native_down.Sizes[2] : 0;
    const int elem_size = (native_gate_up.DType == ETensorDType::BF16) ? 2 : 4;

    const size_t gate_up_expert_bytes = static_cast<size_t>(gate_up_rows * gate_up_cols) * elem_size;
    const size_t down_expert_bytes = static_cast<size_t>(down_rows * down_cols) * elem_size;

    for (const auto& [expert_id, src_rank] : plan.weights_to_receive) {
        auto& pair = received.weights[expert_id];
        pair.gate_up = alloc_foreign_tensor(received, native_gate_up.DType,
            {gate_up_rows, gate_up_cols});
        pair.down = alloc_foreign_tensor(received, native_down.DType,
            {down_rows, down_cols});
    }

    comm.weight_transfer_group_start();

    for (const auto& [expert_id, dst_rank] : plan.weights_to_send) {
        const int local_idx = expert_id - ep_rank * num_local_experts;
        if (local_idx < 0 || local_idx >= num_local_experts) continue;

        const std::byte* gate_up_ptr = static_cast<const std::byte*>(native_gate_up.Data)
            + static_cast<size_t>(local_idx) * gate_up_expert_bytes;
        comm.send_wt(gate_up_ptr, gate_up_expert_bytes, dst_rank, stream);

        const std::byte* down_ptr = static_cast<const std::byte*>(native_down.Data)
            + static_cast<size_t>(local_idx) * down_expert_bytes;
        comm.send_wt(down_ptr, down_expert_bytes, dst_rank, stream);
    }

    for (const auto& [expert_id, src_rank] : plan.weights_to_receive) {
        auto& pair = received.weights[expert_id];
        comm.recv_wt(pair.gate_up.Data, gate_up_expert_bytes, src_rank, stream);
        comm.recv_wt(pair.down.Data, down_expert_bytes, src_rank, stream);
    }

    comm.weight_transfer_group_end();
}

void transfer_expert_weights_quantized(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    dsl::DslRunState& run_state,
    cudaStream_t stream,
    ForeignExpertWeights& received,
    const qlora::QuantizedTensor& native_gate_up_qt,
    const qlora::QuantizedTensor& native_down_qt,
    const Tensor& native_gate_up,
    const Tensor& native_down,
    qlora::IQuantizer* quantizer,
    int num_local_experts,
    int ep_rank) {

    received.free_gpu();

    if (plan.weights_to_send.empty() && plan.weights_to_receive.empty()) {
        return;
    }

    // Compute per-expert quantized byte sizes
    const auto gu_sizes = compute_per_expert_sizes(native_gate_up_qt, num_local_experts);
    const auto dn_sizes = compute_per_expert_sizes(native_down_qt, num_local_experts);

    // Per-expert M dimensions (for dequantization)
    const int gu_per_M = native_gate_up_qt.M / num_local_experts;
    const int dn_per_M = native_down_qt.M / num_local_experts;

    // BF16 output shape per expert (from the dequantized tensor)
    const long gate_up_rows = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[1] : 0;
    const long gate_up_cols = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[2] : 0;
    const long down_rows = (native_down.Rank >= 3) ? native_down.Sizes[1] : 0;
    const long down_cols = (native_down.Rank >= 3) ? native_down.Sizes[2] : 0;

    // Allocate BF16 output buffers (same as BF16 path) + quantized receive buffers
    struct RecvEntry {
        int expert_id;
        int src_rank;
        qlora::QuantizedTensor gu_qt;
        qlora::QuantizedTensor dn_qt;
    };
    std::vector<RecvEntry> recv_entries;

    for (const auto& [expert_id, src_rank] : plan.weights_to_receive) {
        auto& pair = received.weights[expert_id];
        pair.gate_up = alloc_foreign_tensor(received, native_gate_up.DType,
            {gate_up_rows, gate_up_cols});
        pair.down = alloc_foreign_tensor(received, native_down.DType,
            {down_rows, down_cols});

        RecvEntry entry;
        entry.expert_id = expert_id;
        entry.src_rank = src_rank;
        entry.gu_qt = alloc_recv_quant_expert(native_gate_up_qt, gu_per_M, gu_sizes, received.owned_gpu_ptrs);
        entry.dn_qt = alloc_recv_quant_expert(native_down_qt, dn_per_M, dn_sizes, received.owned_gpu_ptrs);
        recv_entries.push_back(std::move(entry));
    }

    // Batched P2P: send quantized slices, receive into quantized buffers
    comm.weight_transfer_group_start();

    for (const auto& [expert_id, dst_rank] : plan.weights_to_send) {
        const int local_idx = expert_id - ep_rank * num_local_experts;
        if (local_idx < 0 || local_idx >= num_local_experts) continue;

        auto gu_slice = make_expert_slice_view(native_gate_up_qt, local_idx, gu_per_M, gu_sizes);
        send_quant_expert(comm, gu_slice, gu_sizes, dst_rank, stream);

        auto dn_slice = make_expert_slice_view(native_down_qt, local_idx, dn_per_M, dn_sizes);
        send_quant_expert(comm, dn_slice, dn_sizes, dst_rank, stream);
    }

    for (auto& entry : recv_entries) {
        recv_quant_expert(comm, entry.gu_qt, gu_sizes, entry.src_rank, stream);
        recv_quant_expert(comm, entry.dn_qt, dn_sizes, entry.src_rank, stream);
    }

    comm.weight_transfer_group_end();

    // Dequantize received quantized data into BF16 output buffers
    for (auto& entry : recv_entries) {
        auto& pair = received.weights[entry.expert_id];
        quantizer->dequantize(entry.gu_qt, pair.gate_up, stream);
        quantizer->dequantize(entry.dn_qt, pair.down, stream);
    }
}

void transfer_weight_gradients_back(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    cudaStream_t stream,
    const ForeignExpertWeights& foreign_grads,
    Tensor& native_gate_up_grad,
    Tensor& native_down_grad,
    int num_local_experts,
    int ep_rank) {

    if (plan.weights_to_send.empty() && plan.weights_to_receive.empty()) {
        return;
    }

    const long gate_up_rows = (native_gate_up_grad.Rank >= 3) ? native_gate_up_grad.Sizes[1] : 0;
    const long gate_up_cols = (native_gate_up_grad.Rank >= 3) ? native_gate_up_grad.Sizes[2] : 0;
    const long down_rows = (native_down_grad.Rank >= 3) ? native_down_grad.Sizes[1] : 0;
    const long down_cols = (native_down_grad.Rank >= 3) ? native_down_grad.Sizes[2] : 0;
    const int elem_size = (native_gate_up_grad.DType == ETensorDType::BF16) ? 2 : 4;
    const size_t gate_up_expert_bytes = static_cast<size_t>(gate_up_rows * gate_up_cols) * elem_size;
    const size_t down_expert_bytes = static_cast<size_t>(down_rows * down_cols) * elem_size;

    comm.weight_transfer_group_start();

    for (const auto& [expert_id, src_rank] : plan.weights_to_receive) {
        auto it = foreign_grads.weights.find(expert_id);
        if (it == foreign_grads.weights.end()) continue;
        comm.send_wt(it->second.gate_up.Data, gate_up_expert_bytes, src_rank, stream);
        comm.send_wt(it->second.down.Data, down_expert_bytes, src_rank, stream);
    }

    for (const auto& [expert_id, dst_rank] : plan.weights_to_send) {
        const int local_idx = expert_id - ep_rank * num_local_experts;
        if (local_idx < 0 || local_idx >= num_local_experts) continue;

        std::byte* gate_up_ptr = static_cast<std::byte*>(native_gate_up_grad.Data)
            + static_cast<size_t>(local_idx) * gate_up_expert_bytes;
        comm.recv_wt(gate_up_ptr, gate_up_expert_bytes, dst_rank, stream);

        std::byte* down_ptr = static_cast<std::byte*>(native_down_grad.Data)
            + static_cast<size_t>(local_idx) * down_expert_bytes;
        comm.recv_wt(down_ptr, down_expert_bytes, dst_rank, stream);
    }

    comm.weight_transfer_group_end();
}

namespace {

/// Helper: send a single expert's LoRA slice (A and B) for one projection.
/// The grouped tensor is [E, ...], each expert is a contiguous slice.
void send_lora_pair(
    NCCLCommunicator& comm,
    const Tensor& grouped_A,  // [E_local, rank, in]
    const Tensor& grouped_B,  // [E_local, out, rank]
    int local_idx,
    int dst_rank,
    cudaStream_t stream) {
    if (grouped_A.is_null() || grouped_B.is_null()) return;
    const size_t a_expert_bytes = grouped_A.bytes() / static_cast<size_t>(grouped_A.Sizes[0]);
    const size_t b_expert_bytes = grouped_B.bytes() / static_cast<size_t>(grouped_B.Sizes[0]);
    const auto* a_ptr = static_cast<const std::byte*>(grouped_A.Data)
        + static_cast<size_t>(local_idx) * a_expert_bytes;
    const auto* b_ptr = static_cast<const std::byte*>(grouped_B.Data)
        + static_cast<size_t>(local_idx) * b_expert_bytes;
    comm.send_wt(a_ptr, a_expert_bytes, dst_rank, stream);
    comm.send_wt(b_ptr, b_expert_bytes, dst_rank, stream);
}

/// Helper: allocate and receive a single expert's LoRA pair.
void recv_lora_pair(
    NCCLCommunicator& comm,
    ForeignExpertWeights& received,
    const Tensor& template_A,  // [E_local, rank, in] (for shape/dtype)
    const Tensor& template_B,  // [E_local, out, rank]
    Tensor& out_A,
    Tensor& out_B,
    int src_rank,
    cudaStream_t stream) {
    if (template_A.is_null() || template_B.is_null()) return;
    // Single expert shape: template.Sizes[1..] (remove expert dimension)
    const long a_rows = template_A.Sizes[1];
    const long a_cols = template_A.Sizes[2];
    const long b_rows = template_B.Sizes[1];
    const long b_cols = template_B.Sizes[2];
    out_A = alloc_foreign_tensor(received, template_A.DType, {a_rows, a_cols});
    out_B = alloc_foreign_tensor(received, template_B.DType, {b_rows, b_cols});
    comm.recv_wt(out_A.Data, out_A.bytes(), src_rank, stream);
    comm.recv_wt(out_B.Data, out_B.bytes(), src_rank, stream);
}

}  // anonymous namespace

void transfer_expert_lora(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    dsl::DslRunState& run_state,
    cudaStream_t stream,
    ForeignExpertWeights& received,
    const Tensor& local_gate_up_A, const Tensor& local_gate_up_B,
    const Tensor& local_gate_A, const Tensor& local_gate_B,
    const Tensor& local_up_A, const Tensor& local_up_B,
    const Tensor& local_down_A, const Tensor& local_down_B,
    int num_local_experts,
    int ep_rank) {

    // Check if any LoRA is present
    const bool has_gate_up = !local_gate_up_A.is_null();
    const bool has_gate = !local_gate_A.is_null();
    const bool has_up = !local_up_A.is_null();
    const bool has_down = !local_down_A.is_null();
    if (!has_gate_up && !has_gate && !has_up && !has_down) return;

    comm.weight_transfer_group_start();

    // Send LoRA slices for each expert we're sending
    for (const auto& [expert_id, dst_rank] : plan.weights_to_send) {
        const int local_idx = expert_id - ep_rank * num_local_experts;
        if (local_idx < 0 || local_idx >= num_local_experts) continue;

        if (has_gate_up)
            send_lora_pair(comm, local_gate_up_A, local_gate_up_B, local_idx, dst_rank, stream);
        if (has_gate)
            send_lora_pair(comm, local_gate_A, local_gate_B, local_idx, dst_rank, stream);
        if (has_up)
            send_lora_pair(comm, local_up_A, local_up_B, local_idx, dst_rank, stream);
        if (has_down)
            send_lora_pair(comm, local_down_A, local_down_B, local_idx, dst_rank, stream);
    }

    // Receive LoRA slices for each foreign expert
    for (const auto& [expert_id, src_rank] : plan.weights_to_receive) {
        auto it = received.weights.find(expert_id);
        if (it == received.weights.end()) continue;
        auto& lora = it->second.lora;

        if (has_gate_up)
            recv_lora_pair(comm, received, local_gate_up_A, local_gate_up_B,
                           lora.gate_up_A, lora.gate_up_B, src_rank, stream);
        if (has_gate)
            recv_lora_pair(comm, received, local_gate_A, local_gate_B,
                           lora.gate_A, lora.gate_B, src_rank, stream);
        if (has_up)
            recv_lora_pair(comm, received, local_up_A, local_up_B,
                           lora.up_A, lora.up_B, src_rank, stream);
        if (has_down)
            recv_lora_pair(comm, received, local_down_A, local_down_B,
                           lora.down_A, lora.down_B, src_rank, stream);
    }

    comm.weight_transfer_group_end();
}

}  // namespace ep
