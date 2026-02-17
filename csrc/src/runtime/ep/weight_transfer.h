// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert weight transfer for LLEP load balancing.
// Uses P2P send/recv via NCCL weight_transfer_comm for transferring
// expert weights from native GPUs to helper GPUs.
//
// Supports two transfer modes:
//   1. BF16 transfer: send dequantized BF16 weights (Phase 3, always works)
//   2. Quantized transfer: send raw quantized bytes + scales (Phase 4, 2-8x less bandwidth)

#ifndef SUROGATE_SRC_RUNTIME_EP_WEIGHT_TRANSFER_H
#define SUROGATE_SRC_RUNTIME_EP_WEIGHT_TRANSFER_H

#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/ep/lpt_planner.h"
#include "utilities/tensor.h"

namespace dsl {
class DslRunState;
}

namespace qlora {
struct QuantizedTensor;
class IQuantizer;
}  // namespace qlora

class NCCLCommunicator;

namespace ep {

/// Per-expert LoRA adapter weights (optional, present when LoRA is enabled).
/// Each tensor is a single expert's slice: A=[rank, in], B=[out, rank].
struct ExpertLoRA {
    Tensor gate_up_A, gate_up_B;  ///< Fused gate+up LoRA (Qwen3-MoE style)
    Tensor gate_A, gate_B;        ///< Gate-only LoRA
    Tensor up_A, up_B;            ///< Up-only LoRA
    Tensor down_A, down_B;        ///< Down projection LoRA

    [[nodiscard]] bool empty() const {
        return !gate_up_A.Data && !gate_A.Data && !up_A.Data && !down_A.Data;
    }
};

/// Holds received foreign expert weights on a helper GPU.
/// Maps global expert_id → BF16 weight tensor (dequantized on source or receiver).
struct ForeignExpertWeights {
    /// Received weight tensors: expert_id → {gate_up_weight, down_weight, optional LoRA}
    struct ExpertWeightPair {
        Tensor gate_up;   ///< [gate_up_rows, hidden] or [2*intermediate, hidden]
        Tensor down;      ///< [hidden, intermediate]
        ExpertLoRA lora;  ///< Per-expert LoRA adapters (empty if LoRA not enabled)
    };
    std::unordered_map<int, ExpertWeightPair> weights;

    /// GPU pointers allocated with cudaMalloc (tracked for cleanup).
    /// Populated when use_cuda_malloc=true in transfer functions.
    std::vector<void*> owned_gpu_ptrs;

    void clear() {
        weights.clear();
        // Do NOT free GPU memory here — use free_gpu() explicitly
    }
    bool empty() const { return weights.empty(); }

    /// Free all cudaMalloc'd GPU buffers owned by this struct.
    /// Call after the data has been consumed (e.g., merged into combined tensors).
    void free_gpu() {
        for (void* p : owned_gpu_ptrs) {
            if (p) cudaFree(p);
        }
        owned_gpu_ptrs.clear();
        weights.clear();
    }
};

/// Transfer expert weights from native to helper GPUs according to LPT plan.
///
/// Uses the weight_transfer_comm (separate from EP A2A comm) for overlap.
/// Weights are sent as dequantized BF16 slices from the 3D weight tensors.
///
/// @param plan              LPT plan with weight transfer entries
/// @param comm              NCCL communicator with weight_transfer_comm
/// @param run_state         DSL run state for temporary allocations
/// @param stream            CUDA stream for transfers
/// @param[out] received     Received foreign expert weights on this GPU
/// @param native_gate_up    Native dequantized gate_up weights [E_local, rows, cols]
/// @param native_down       Native dequantized down weights [E_local, rows, cols]
/// @param num_local_experts Number of native experts per GPU
/// @param ep_rank           This GPU's EP rank
void transfer_expert_weights(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    dsl::DslRunState& run_state,
    cudaStream_t stream,
    ForeignExpertWeights& received,
    const Tensor& native_gate_up,
    const Tensor& native_down,
    int num_local_experts,
    int ep_rank);

/// Transfer expert weights using quantized format for reduced P2P bandwidth.
///
/// Sends raw quantized data + scales instead of dequantized BF16, achieving
/// 2-8x bandwidth reduction depending on format (NF4: ~4x, FP8: ~2x).
/// The receiver dequantizes locally using the provided quantizer.
///
/// Output is still BF16 in ForeignExpertWeights (same as BF16 path).
///
/// @param plan                LPT plan with weight transfer entries
/// @param comm                NCCL communicator with weight_transfer_comm
/// @param run_state           DSL run state for temporary allocations
/// @param stream              CUDA stream for transfers
/// @param[out] received       Received foreign expert weights (BF16, dequantized on receiver)
/// @param native_gate_up_qt   Quantized gate_up tensor (all local experts)
/// @param native_down_qt      Quantized down tensor (all local experts)
/// @param native_gate_up      BF16 gate_up (for shape info: [E_local, rows, cols])
/// @param native_down         BF16 down (for shape info: [E_local, rows, cols])
/// @param quantizer           Quantizer for dequantization on receiver
/// @param num_local_experts   Number of native experts per GPU
/// @param ep_rank             This GPU's EP rank
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
    int ep_rank);

/// Transfer expert weight gradients back from helper to native GPUs.
///
/// After backward pass, helper GPUs have computed gradients for foreign experts.
/// These gradients are sent back to the native GPU for accumulation.
///
/// @param plan              LPT plan (same as forward)
/// @param comm              NCCL communicator
/// @param stream            CUDA stream for transfers
/// @param foreign_grads     Gradients computed on this GPU for foreign experts
/// @param native_gate_up_grad  Native gate_up gradient tensor (receive target)
/// @param native_down_grad     Native down gradient tensor (receive target)
/// @param num_local_experts    Number of native experts per GPU
/// @param ep_rank              This GPU's EP rank
void transfer_weight_gradients_back(
    const LPTPlan& plan,
    NCCLCommunicator& comm,
    cudaStream_t stream,
    const ForeignExpertWeights& foreign_grads,
    Tensor& native_gate_up_grad,
    Tensor& native_down_grad,
    int num_local_experts,
    int ep_rank);

/// Transfer per-expert LoRA adapters from native to helper GPUs.
///
/// Sends LoRA A/B matrices for each transferred expert. Small overhead
/// since LoRA is low-rank (typically <1% of base weight size).
/// Runs on the same weight_transfer_comm/stream as base weight transfer.
///
/// @param plan              LPT plan with weight transfer entries
/// @param comm              NCCL communicator with weight_transfer_comm
/// @param run_state         DSL run state for temporary allocations
/// @param stream            CUDA stream for transfers
/// @param[in,out] received  Foreign expert weights (LoRA fields populated)
/// @param local_gate_up_A   Local gate_up LoRA A [E_local, rank, in] (null if no gate_up LoRA)
/// @param local_gate_up_B   Local gate_up LoRA B [E_local, out, rank] (null if no gate_up LoRA)
/// @param local_gate_A      Local gate LoRA A [E_local, rank, in] (null if no gate LoRA)
/// @param local_gate_B      Local gate LoRA B [E_local, out, rank] (null if no gate LoRA)
/// @param local_up_A        Local up LoRA A [E_local, rank, in] (null if no up LoRA)
/// @param local_up_B        Local up LoRA B [E_local, out, rank] (null if no up LoRA)
/// @param local_down_A      Local down LoRA A [E_local, rank, D] (null if no down LoRA)
/// @param local_down_B      Local down LoRA B [E_local, C, rank] (null if no down LoRA)
/// @param num_local_experts Number of native experts per GPU
/// @param ep_rank           This GPU's EP rank
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
    int ep_rank);

}  // namespace ep

#endif  // SUROGATE_SRC_RUNTIME_EP_WEIGHT_TRANSFER_H
