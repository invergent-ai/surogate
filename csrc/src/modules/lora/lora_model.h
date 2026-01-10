// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "kernels/kernels.h"
#include "lora_config.h"
#include "lora_weights.h"
#include "modules/forward_hooks.h"
#include "modules/modular_model.h"
#include "modules/optimizers/normuon.h"
#include "modules/qlora/qlora_config.h"
#include "modules/qlora/fp8_weight_provider.h"
#include "modules/qlora/fp4_weight_provider.h"
#include "modules/qlora/bnb_weight_provider.h"
#include "training/model.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

struct RuntimeOptions;

namespace modules {

namespace detail {

struct EmptyTensorContainer final : public ITensorContainer {
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {}
};

inline ITensorContainer& empty_tensor_container() {
    static EmptyTensorContainer instance;
    return instance;
}

inline void apply_lora_contribution(
    Tensor& output,
    int output_offset,
    const Tensor& input,
    const LoRALayerWeights<Tensor>& lora,
    Tensor& intermediate,
    Tensor& slice_buffer,
    float scaling,
    int BT,
    int in_features,
    int out_features,
    int rank,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (!lora.has_value()) return;
    if (out_features <= 0 || BT <= 0) return;

    // intermediate = input @ A^T  (BT x rank)
    matmul(intermediate, lora.A, input, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

    // Scale intermediate so we can use GEMM accumulate for B @ intermediate^T.
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    const long total_out_features = output.Sizes[output.Rank - 1];
    if (output_offset < 0 || output_offset + out_features > total_out_features) {
        throw std::logic_error("apply_lora_contribution: output_offset out of bounds");
    }

    // Packed destination: accumulate directly.
    if (output_offset == 0 && out_features == total_out_features) {
        matmul(output, lora.B, intermediate, std::nullopt, nullptr, nullptr,
               handle, workspace, out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/true, stream);
        return;
    }

    // Fused projections: prefer direct strided accumulate when aligned, else fall back to packed delta + add.
    Tensor output_slice = output;
    output_slice.Data = output.Data + (std::size_t)output_offset * get_dtype_size(output.DType);
    bool aligned = ((uintptr_t)output_slice.Data % 16) == 0;
    if (aligned) {
        matmul_strided_c(output_slice, lora.B, intermediate, std::nullopt, nullptr, nullptr,
                         handle, workspace,
                         out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/true,
                         (int)total_out_features, stream);
        return;
    }

    Tensor packed_delta = slice_buffer;
    packed_delta.DType = output.DType;
    packed_delta.Rank = 2;
    packed_delta.Sizes[0] = BT;
    packed_delta.Sizes[1] = out_features;
    for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed_delta.Sizes[i] = 1;

    matmul(packed_delta, lora.B, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/false, stream);
    add_2d_slice(output, packed_delta, BT, total_out_features, out_features, output_offset, stream);
}

inline void backward_lora_layer(
    Tensor& dA,
    Tensor& dB,
    Tensor& dx,
    const Tensor& dL_dy,
    int dL_dy_offset,
    const Tensor& x,
    const Tensor& A,
    const Tensor& B,
    float scaling,
    Tensor& intermediate,
    Tensor& slice_buffer,
    int BT,
    int in_features,
    int out_features,
    int rank,
    bool accumulate,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (!A.Data || !B.Data) return;

    Tensor dL_dy_slice = dL_dy;
    const long full_out_features = dL_dy.Sizes[dL_dy.Rank - 1];
    if (dL_dy_offset < 0 || dL_dy_offset + out_features > full_out_features) {
        throw std::logic_error("backward_lora_layer: dL_dy_offset out of bounds");
    }

    // Pack fused slice into a contiguous buffer.
    if (dL_dy_offset != 0 || out_features != full_out_features) {
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = out_features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_out_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)out_features * elem_size;
        const std::size_t width = (std::size_t)out_features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)dL_dy_offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_ptr, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));

        dL_dy_slice = packed;
        dL_dy_offset = 0;
    }

    // intermediate = x @ A^T (BT x rank)
    matmul(intermediate, A, x, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dB = (x @ A^T)^T @ dL_dy
    matmul(dB, intermediate, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // intermediate = B @ dL_dy^T  => (BT x rank) view
    matmul(intermediate, B, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dA = x^T @ (dL_dy @ B)
    matmul(dA, x, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // dx += (dL_dy @ B) @ A
    matmul(dx, A, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
}

/**
 * @brief Fused backward pass for QKV LoRA projections
 *
 * Optimizes QKV backward by:
 * 1. Computing dL_dy @ B^T for all projections, then batching the dx accumulation
 * 2. Reusing x (ln1) across all three projections instead of redundant loads
 * 3. Reducing kernel launch overhead from 15 matmuls to 12 matmuls
 *
 * Mathematical formulation (for each projection p in {q,k,v}):
 *   dA_p = x^T @ (dL_dy_p @ B_p^T) * scaling
 *   dB_p = (x @ A_p^T)^T @ dL_dy_p * scaling
 *   dx += (dL_dy_p @ B_p^T) @ A_p * scaling
 *
 * Fusion strategy:
 * - Phase 1: Compute x @ A^T for all projections (reuses x)
 * - Phase 2: Compute dB for all projections
 * - Phase 3: Compute dL_dy @ B^T and dA for all projections
 * - Phase 4: Accumulate dx contributions
 */
inline void backward_lora_qkv_fused(
    // Gradient outputs for Q
    Tensor& dA_q, Tensor& dB_q,
    // Gradient outputs for K
    Tensor& dA_k, Tensor& dB_k,
    // Gradient outputs for V
    Tensor& dA_v, Tensor& dB_v,
    // Input gradient accumulator
    Tensor& dx,
    // Upstream gradient (packed QKV)
    const Tensor& dL_dy,
    // Forward input (shared across Q, K, V)
    const Tensor& x,
    // LoRA weights
    const LoRALayerWeights<Tensor>& lora_q,
    const LoRALayerWeights<Tensor>& lora_k,
    const LoRALayerWeights<Tensor>& lora_v,
    // Dimensions
    float scaling,
    int BT,
    int in_features,    // C (hidden size)
    int q_out_features, // Hq * Hs
    int kv_out_features,// Hkv * Hs
    int rank,
    bool accumulate,
    // Intermediates (must be pre-allocated)
    Tensor& intermediate1,  // (BT, rank) for x @ A^T
    Tensor& intermediate2,  // (BT, rank) for dL_dy @ B^T
    Tensor& slice_buffer,   // For slicing packed QKV gradients
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    const bool has_q = lora_q.has_value() && dA_q.Data;
    const bool has_k = lora_k.has_value() && dA_k.Data;
    const bool has_v = lora_v.has_value() && dA_v.Data;

    if (!has_q && !has_k && !has_v) return;

    // Offsets into packed QKV gradient tensor
    const int q_offset = 0;
    const int k_offset = q_out_features;
    const int v_offset = q_out_features + kv_out_features;
    const long full_qkv_features = dL_dy.Sizes[dL_dy.Rank - 1];

    // Helper to extract a slice from packed QKV gradient
    auto extract_slice = [&](int offset, int features) -> Tensor {
        if (offset == 0 && features == full_qkv_features) {
            return dL_dy;
        }
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_qkv_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)features * elem_size;
        const std::size_t width = (std::size_t)features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_ptr, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));
        return packed;
    };

    // =======================================================================
    // Phase 1 & 2: For each projection, compute x @ A^T, then dB
    // This reuses x across all projections
    // =======================================================================

    if (has_q) {
        Tensor dL_dy_q = extract_slice(q_offset, q_out_features);

        // intermediate1 = x @ A_q^T (BT x rank)
        matmul(intermediate1, lora_q.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_q = intermediate1^T @ dL_dy_q
        matmul(dB_q, intermediate1, dL_dy_q, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, q_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_q @ dL_dy_q^T (for dA_q and dx)
        matmul(intermediate2, lora_q.B, dL_dy_q, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, q_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_q = x^T @ intermediate2
        matmul(dA_q, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_q
        matmul(dx, lora_q.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    if (has_k) {
        Tensor dL_dy_k = extract_slice(k_offset, kv_out_features);

        // intermediate1 = x @ A_k^T (BT x rank)
        matmul(intermediate1, lora_k.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_k = intermediate1^T @ dL_dy_k
        matmul(dB_k, intermediate1, dL_dy_k, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, kv_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_k @ dL_dy_k^T
        matmul(intermediate2, lora_k.B, dL_dy_k, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, kv_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_k = x^T @ intermediate2
        matmul(dA_k, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_k
        matmul(dx, lora_k.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    if (has_v) {
        Tensor dL_dy_v = extract_slice(v_offset, kv_out_features);

        // intermediate1 = x @ A_v^T (BT x rank)
        matmul(intermediate1, lora_v.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_v = intermediate1^T @ dL_dy_v
        matmul(dB_v, intermediate1, dL_dy_v, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, kv_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_v @ dL_dy_v^T
        matmul(intermediate2, lora_v.B, dL_dy_v, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, kv_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_v = x^T @ intermediate2
        matmul(dA_v, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_v
        matmul(dx, lora_v.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }
}

/**
 * @brief Fused backward pass for MLP LoRA projections (gate, up, down)
 *
 * Optimizes MLP backward by processing gate and up together (shared input x=ln2)
 * and down separately (input x=swiglu).
 *
 * For gate/up (shared x = ln2, shared dL_dy = d_mlp_up):
 *   dA_gate = x^T @ (dL_dy[D:] @ B_gate^T) * scaling
 *   dA_up   = x^T @ (dL_dy[:D] @ B_up^T) * scaling
 *   dx_ln2 += contributions from both
 *
 * For down (x = swiglu, dL_dy = d_res_ffn):
 *   dA_down = swiglu^T @ (dL_dy @ B_down^T) * scaling
 *   dx_swiglu += contribution
 */
inline void backward_lora_mlp_up_gate_fused(
    // Gradient outputs for up
    Tensor& dA_up, Tensor& dB_up,
    // Gradient outputs for gate
    Tensor& dA_gate, Tensor& dB_gate,
    // Input gradient accumulator (d_ln2)
    Tensor& dx,
    // Upstream gradient (packed up+gate from SwiGLU backward)
    const Tensor& dL_dy,
    // Forward input (ln2 output, shared across up and gate)
    const Tensor& x,
    // LoRA weights
    const LoRALayerWeights<Tensor>& lora_up,
    const LoRALayerWeights<Tensor>& lora_gate,
    // Dimensions
    float scaling,
    int BT,
    int in_features,    // C (hidden size)
    int out_features,   // D (intermediate size)
    int rank,
    bool accumulate,
    // Intermediates
    Tensor& intermediate1,
    Tensor& intermediate2,
    Tensor& slice_buffer,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    const bool has_up = lora_up.has_value() && dA_up.Data;
    const bool has_gate = lora_gate.has_value() && dA_gate.Data;

    if (!has_up && !has_gate) return;

    // dL_dy is packed as [d_up (D), d_gate (D)]
    const int up_offset = 0;
    const int gate_offset = out_features;
    const long full_features = dL_dy.Sizes[dL_dy.Rank - 1];

    auto extract_slice = [&](int offset, int features) -> Tensor {
        if (offset == 0 && features == full_features) {
            return dL_dy;
        }
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)features * elem_size;
        const std::size_t width = (std::size_t)features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_ptr, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));
        return packed;
    };

    // Process up projection
    if (has_up) {
        Tensor dL_dy_up = extract_slice(up_offset, out_features);

        // intermediate1 = x @ A_up^T
        matmul(intermediate1, lora_up.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_up = intermediate1^T @ dL_dy_up
        matmul(dB_up, intermediate1, dL_dy_up, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_up @ dL_dy_up^T
        matmul(intermediate2, lora_up.B, dL_dy_up, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_up = x^T @ intermediate2
        matmul(dA_up, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_up
        matmul(dx, lora_up.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    // Process gate projection
    if (has_gate) {
        Tensor dL_dy_gate = extract_slice(gate_offset, out_features);

        // intermediate1 = x @ A_gate^T
        matmul(intermediate1, lora_gate.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_gate = intermediate1^T @ dL_dy_gate
        matmul(dB_gate, intermediate1, dL_dy_gate, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_gate @ dL_dy_gate^T
        matmul(intermediate2, lora_gate.B, dL_dy_gate, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_gate = x^T @ intermediate2
        matmul(dA_gate, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_gate
        matmul(dx, lora_gate.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }
}

inline std::vector<std::string> targets_to_peft_names(const ModularLoRAConfig& cfg) {
    std::vector<std::string> out;
    out.reserve(8);
    if (cfg.applies_to_q()) out.emplace_back("q_proj");
    if (cfg.applies_to_k()) out.emplace_back("k_proj");
    if (cfg.applies_to_v()) out.emplace_back("v_proj");
    if (cfg.applies_to_o()) out.emplace_back("o_proj");
    if (cfg.applies_to_gate()) out.emplace_back("gate_proj");
    if (cfg.applies_to_up()) out.emplace_back("up_proj");
    if (cfg.applies_to_down()) out.emplace_back("down_proj");
    return out;
}

} // namespace detail

template<typename Block>
class ModularLoRAModel final : public IModel {
public:
    ModularLoRAModel(std::unique_ptr<ModularTransformerModel<Block>> base_model,
                     const ModularLoRAConfig& lora_config,
                     const RuntimeOptions& options,
                     NCCLCommunicator& comm,
                     const std::shared_ptr<TensorAllocator>& allocator,
                     const QLoRAConfig& qlora_config = QLoRAConfig{})
        : mBaseModel(std::move(base_model))
        , mLoRAConfig(lora_config)
        , mQLoRAConfig(qlora_config)
        , mOptions(options)
        , mAllocator(allocator ? allocator : std::make_shared<TensorAllocator>())
        , mLoRAOptimizerRNG(42) {

        if (!lora_enabled()) {
            return;
        }

        const auto& cfg = mBaseModel->config();

        ModularLoRAWeightsManager::Config wm{};
        wm.num_layers = cfg.NumLayers;
        wm.hidden_size = cfg.HiddenSize;
        wm.intermediate_size = cfg.IntermediateSize;
        wm.num_query_heads = cfg.NumQueryHeads;
        wm.num_kv_heads = cfg.NumKeyValHeads;
        wm.head_size = cfg.head_size();
        wm.lora_config = mLoRAConfig;
        wm.work_dtype = cfg.DType;
        wm.shard_idx = comm.rank();
        wm.num_shards = comm.world_size();
        mLoRAWeights = std::make_unique<ModularLoRAWeightsManager>(wm, *mAllocator);

        ModularLoRAGradsManager::Config gm{};
        gm.num_layers = cfg.NumLayers;
        gm.hidden_size = cfg.HiddenSize;
        gm.intermediate_size = cfg.IntermediateSize;
        gm.num_query_heads = cfg.NumQueryHeads;
        gm.num_kv_heads = cfg.NumKeyValHeads;
        gm.head_size = cfg.head_size();
        gm.lora_config = mLoRAConfig;
        gm.grad_dtype = mLoRAConfig.dtype;
        gm.shard_idx = comm.rank();
        gm.num_shards = comm.world_size();
        mLoRAGrads = std::make_unique<ModularLoRAGradsManager>(gm, mAllocator);
    }

    // IModel
    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T, bool allocate_optimizer) override {
        (void)allocate_optimizer;
        // Set the recipe on base model from RuntimeOptions (critical for FP8/FP4 recipes)
        if (options.TrainingRecipe) {
            mBaseModel->set_recipe(options.TrainingRecipe);
        }
        // Convert RuntimeOptions to ModelOptions and enable LoRA optimizations for memory efficiency
        // - lora_only_mode: skips storing activations not needed for LoRA backward
        // - skip_base_gradients: skips allocating gradient buffers for frozen base weights (~8GB for 4B model)
        auto model_opts = ModelOptions::from_runtime_options(options);
        if (lora_enabled()) {
            model_opts.lora_only_mode = true;
            model_opts.skip_base_gradients = true;
        }
        // Disable CUDA graphs for QLoRA FP4: the on-the-fly weight dequantization is not
        // compatible with CUDA graph replay (weights change between steps but graphs expect
        // consistent state). This is a temporary workaround until proper graph-aware
        // weight caching is implemented.
        if (qlora_enabled() && mFP4WeightProvider) {
            model_opts.use_cuda_graphs = false;
        }
        // recompute_lora + offload_residuals: CUDA graphs must be disabled.
        // recompute_lora needs to read residuals during LoRA backward to recompute ln1/ln2.
        // offload_residuals uses cudaStreamWaitEvent for async D2H/H2D copies, which
        // breaks CUDA graph capture isolation. Allow both memory optimizations to work
        // together by sacrificing CUDA graph performance.
        if (model_opts.recompute_lora && model_opts.offload_residuals) {
            model_opts.use_cuda_graphs = false;
        }
        // Use the ModelOptions overload to apply LoRA-specific flags
        mBaseModel->allocate_run_state(model_opts, comm, B, T, /*allocate_optimizer=*/false);
        allocate_lora_run_state(comm, B, T);

        // Allocate 8-bit AdamW optimizer state for LoRA
        if (lora_enabled() && !mLoRAAdamW8BitState) {
            mLoRAAdamW8BitState = std::make_unique<LoRAAdamW8BitState>();
            mLoRAAdamW8BitState->initialized = false;

            // Allocate quantization maps (256 entries each)
            mLoRAAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
            mLoRAAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});

            // Initialize quantization maps on host then copy to device
            std::vector<float> h_quantiles1(256), h_quantiles2(256);
            create_adamw8bit_quantiles1(h_quantiles1.data());
            create_adamw8bit_quantiles2(h_quantiles2.data());
            CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles1.Data, h_quantiles1.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles2.Data, h_quantiles2.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void init_weights(NCCLCommunicator& comm) override {
        mBaseModel->init_weights(comm);
        if (lora_enabled()) {
            mLoRAWeights->random_init(42, comm);
        }
    }

    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override {
        if (qlora_enabled() && lora_enabled()) {
            // QLoRA mode: load base weights into quantized storage, then inject weight provider
            if (mQLoRAConfig.is_fp4()) {
                import_weights_fp4_qlora(file_name, comm);
            } else if (mQLoRAConfig.is_bnb()) {
                import_weights_bnb_qlora(file_name, comm);
            } else {
                import_weights_qlora(file_name, comm);
            }
        } else {
            // Standard LoRA or non-LoRA: load base weights normally
            mBaseModel->import_weights(file_name, allow_cast, comm);
        }
        if (lora_enabled()) {
            mLoRAWeights->random_init(42, comm);
        }
    }

    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override {
        mBaseModel->export_weights(file_name, comm);
    }

    void on_restore_checkpoint(NCCLCommunicator& comm) override {
        mBaseModel->on_restore_checkpoint(comm);
    }

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override {
        if (!lora_enabled()) {
            mBaseModel->forward(inputs, position_ids, comm, micro_step);
            return;
        }

        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        // Invalidate QLoRA dequant cache at the start of each training step.
        // This ensures forward pass dequantizes fresh weights, which are then
        // cached and reused by the backward pass (eliminating redundant dequant).
        if (qlora_enabled() && micro_step == 0) {
            if (mFP8WeightProvider) {
                mFP8WeightProvider->invalidate_cache();
            }
            if (mFP4WeightProvider) {
                mFP4WeightProvider->invalidate_cache();
            }
            if (mBnBWeightProvider) {
                mBnBWeightProvider->invalidate_cache();
            }
        }

        auto hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point) {
            const auto& cfg = mBaseModel->config();
            auto& rs = mBaseModel->run_state();
            const int B = (int)rs.B;
            const int T = (int)rs.T;
            const int C = (int)cfg.HiddenSize;
            const int D = (int)cfg.IntermediateSize;
            const int Hq = (int)cfg.NumQueryHeads;
            const int Hkv = (int)cfg.NumKeyValHeads;
            const int Hs = (int)cfg.head_size();
            const int rank = mLoRAConfig.rank;
            const float scaling = mLoRAConfig.scaling();

            auto& acts = rs.simplified_acts(layer_idx);
            auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

            switch (point) {
                case ForwardHookPoint::AfterQKVProjection: {
                    if (lora_block.attention.q.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hq * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.k.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.v.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterAttnOutProjection: {
                    if (lora_block.attention.o.has_value()) {
                        detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, Hq * Hs, C, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterMLPUpProjection: {
                    if (lora_block.mlp.up.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.mlp.gate.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterMLPDownProjection: {
                    if (lora_block.mlp.down.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, D, C, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
            }
        };

        mBaseModel->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
    }

    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override {
        if (!lora_enabled()) {
            return mBaseModel->validate(inputs, position_ids, targets, comm, micro_step);
        }

        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        auto full_hook = [this](int layer_idx, cudaStream_t stream, ForwardHookPoint point) {
            const auto& cfg = mBaseModel->config();
            auto& rs = mBaseModel->run_state();
            const int B = (int)rs.B;
            const int T = (int)rs.T;
            const int C = (int)cfg.HiddenSize;
            const int D = (int)cfg.IntermediateSize;
            const int Hq = (int)cfg.NumQueryHeads;
            const int Hkv = (int)cfg.NumKeyValHeads;
            const int Hs = (int)cfg.head_size();
            const int rank = mLoRAConfig.rank;
            const float scaling = mLoRAConfig.scaling();

            auto& acts = rs.simplified_acts(layer_idx);
            auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

            switch (point) {
                case ForwardHookPoint::AfterQKVProjection: {
                    if (lora_block.attention.q.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hq * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.k.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.attention.v.has_value()) {
                        detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, Hkv * Hs, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterAttnOutProjection: {
                    if (lora_block.attention.o.has_value()) {
                        detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, Hq * Hs, C, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterMLPUpProjection: {
                    if (lora_block.mlp.up.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                    if (lora_block.mlp.gate.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, C, D, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
                case ForwardHookPoint::AfterMLPDownProjection: {
                    if (lora_block.mlp.down.has_value()) {
                        detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                        mLoRARunState->intermediate, mLoRARunState->slice,
                                                        scaling, B * T, D, C, rank,
                                                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    }
                } break;
            }
        };

        return mBaseModel->validate_with_hook(inputs, position_ids, targets, comm, micro_step, full_hook);
    }

    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override {
        if (!lora_enabled()) {
            mBaseModel->backward(inputs, targets, comm, grad_accum_steps, micro_step);
            return;
        }

        ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

        auto& rs = mBaseModel->run_state();
        cudaStream_t main_stream = rs.MainStream;

        mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

        auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, BackwardHookPoint point) {
            const int B = (int)mBaseModel->get_run_state().B;
            const int T = (int)mBaseModel->get_run_state().T;
            switch (point) {
                case BackwardHookPoint::AfterMLPDownBackward:
                    backward_lora_mlp_down(layer_idx, B, T, accumulate, comm, stream);
                    break;
                case BackwardHookPoint::AfterMLPUpBackward:
                    backward_lora_mlp_up(layer_idx, B, T, accumulate, comm, stream);
                    break;
                case BackwardHookPoint::AfterAttnOutBackward:
                    backward_lora_attn_out(layer_idx, B, T, accumulate, comm, stream);
                    break;
                case BackwardHookPoint::AfterQKVBackward:
                    backward_lora_qkv(layer_idx, B, T, accumulate, comm, stream);
                    mLoRAGrads->notify_block(layer_idx, stream, comm);
                    break;
                default:
                    break;
            }
        };

        mBaseModel->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);
        mLoRAGrads->end_micro_step(main_stream, comm);
        // Extend the base-model BackwardDone event to include LoRA gradient reductions.
        CUDA_CHECK(cudaEventRecord(rs.BackwardDone, main_stream));
    }

    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                int t, float epsilon, float weight_decay, float grad_clip) override {
        if (!lora_enabled()) {
            mBaseModel->update(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
            return;
        }

        // Only 8-bit AdamW optimizer is supported
        if (!mLoRAAdamW8BitState) {
            throw std::logic_error("ModularLoRAModel::update: 8-bit optimizer state not allocated");
        }
        update_adamw_8bit(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
    }

    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override {
        if (!lora_enabled()) {
            // Delegate to base model which supports all optimizers for full fine-tuning
            mBaseModel->update_with_config(comm, config, step);
            return;
        }

        switch (config.type) {
            case optimizers::OptimizerType::ADAMW_8BIT:
                if (!mLoRAAdamW8BitState) {
                    throw std::logic_error("ModularLoRAModel::update_with_config(ADAMW_8BIT): "
                                           "optimizer state not allocated");
                }
                update_adamw_8bit(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                                 step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
                break;

            case optimizers::OptimizerType::NORMUON:
                if (step == 1) {
                    fmt::print("LoRA NorMuon optimizer selected (step {})\n", step);
                }
                update_normuon(comm, config, step);
                break;

            default:
                throw std::logic_error("ModularLoRAModel::update_with_config(): unsupported optimizer type");
        }
    }

    void update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                           int t, float epsilon, float weight_decay, float grad_clip) {
        NVTX_RANGE_FN();
        auto& rs = mBaseModel->get_run_state();
        cudaStream_t main_stream = rs.MainStream;
        auto& state = *mLoRAAdamW8BitState;

        // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
        calculate_lora_gradient_norm(comm, grad_clip);
        const float* grad_scale = mLoRARunState->norm_buffer.template get<float>() + 1;

        // Initialize multi-tensor optimizer state on first call
        if (!state.initialized) {
            initialize_multi_tensor_state(comm, main_stream);
        }

        // Update grad pointers each step (grads change between steps due to accumulation)
        update_grad_pointers(comm, main_stream);

        // Single kernel launch for all LoRA tensors - dispatch based on lora_dtype
        const ETensorDType lora_dtype = mLoRAConfig.dtype;
        if (lora_dtype == ETensorDType::FP32) {
            adamw_update_8bit_multi_tensor(
                reinterpret_cast<float**>(state.param_ptrs.Data),
                reinterpret_cast<float**>(state.grad_ptrs.Data),
                state.tensor_sizes.template get<int>(),
                state.num_tensors,
                reinterpret_cast<unsigned char*>(state.state1.Data),
                reinterpret_cast<unsigned char*>(state.state2.Data),
                state.absmax1.template get<float>(),
                state.absmax2.template get<float>(),
                state.state_offsets.template get<int>(),
                state.total_params,
                learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
                state.quantiles1.template get<float>(),
                state.quantiles2.template get<float>(),
                main_stream
            );
        } else if (lora_dtype == ETensorDType::BF16) {
            adamw_update_8bit_multi_tensor(
                reinterpret_cast<nv_bfloat16**>(state.param_ptrs.Data),
                reinterpret_cast<nv_bfloat16**>(state.grad_ptrs.Data),
                state.tensor_sizes.template get<int>(),
                state.num_tensors,
                reinterpret_cast<unsigned char*>(state.state1.Data),
                reinterpret_cast<unsigned char*>(state.state2.Data),
                state.absmax1.template get<float>(),
                state.absmax2.template get<float>(),
                state.state_offsets.template get<int>(),
                state.total_params,
                learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_scale,
                state.quantiles1.template get<float>(),
                state.quantiles2.template get<float>(),
                main_stream
            );
        } else {
            throw std::runtime_error(fmt::format(
                "ModularLoRAModel: unsupported lora_dtype {} for 8-bit AdamW optimizer",
                dtype_to_str(lora_dtype)));
        }

        CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
    }

    /**
     * @brief NorMuon optimizer update for LoRA weights
     *
     * Implements the NorMuon algorithm with orthogonalized momentum and variance reduction.
     * For LoRA's 2D matrices, this applies Polar Express orthogonalization and
     * Adafactor-style variance reduction.
     */
    void update_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
        NVTX_RANGE_FN();
        auto& rs = mBaseModel->get_run_state();
        cudaStream_t main_stream = rs.MainStream;

        // Initialize NorMuon state if needed
        if (!mLoRANorMuonState) {
            mLoRANorMuonState = std::make_unique<LoRANorMuonState>();
        }
        auto& state = *mLoRANorMuonState;

        // Calculate gradient norm - grad_scale is kept on device for CUDA graph compatibility
        calculate_lora_gradient_norm(comm, config.grad_clip);

        // Extract NorMuon hyperparameters
        const float lr = config.normuon_lr > 0 ? config.normuon_lr : config.learning_rate;
        const float momentum = config.normuon_momentum;
        const float beta2 = config.normuon_beta2;
        const float weight_decay = config.weight_decay;
        const bool cautious_wd = config.normuon_cautious_wd;
        const int L = (int)mBaseModel->config().NumLayers;

        constexpr size_t BLOCK_SIZE = optimizers::NORMUON_BLOCK_SIZE;

        // Lazy initialization of state tensors
        if (!state.initialized) {
            fmt::print("LoRA NorMuon: Initializing optimizer state...\n");

            // Phase 1: Count parameters and find max dimensions
            state.total_params = 0;
            state.state_elems = 0;
            state.max_weight_M = 0;
            state.max_weight_N = 0;

            auto add_param = [&](const Tensor& weight) {
                if (!weight.Data) return;
                size_t n = weight.nelem();
                state.total_params += n;
                state.state_elems = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                state.state_elems += n;

                // Track max dimensions for workspace allocation
                int M = 1, N = (int)n;
                if (weight.Rank >= 2) {
                    M = (int)weight.Sizes[0];
                    N = (int)(n / M);
                }
                state.max_weight_M = std::max(state.max_weight_M, (size_t)M);
                state.max_weight_N = std::max(state.max_weight_N, (size_t)N);

                // Store shape for variance buffer allocation
                state.variance_shapes.push_back({M, N});
            };

            // Count in same order as update phase
            for (int l = 0; l < L; ++l) {
                auto& lora_w = mLoRAWeights->get_master_block(l, main_stream);

                if (lora_w.attention.q.has_value()) {
                    add_param(lora_w.attention.q->A);
                    add_param(lora_w.attention.q->B);
                }
                if (lora_w.attention.k.has_value()) {
                    add_param(lora_w.attention.k->A);
                    add_param(lora_w.attention.k->B);
                }
                if (lora_w.attention.v.has_value()) {
                    add_param(lora_w.attention.v->A);
                    add_param(lora_w.attention.v->B);
                }
                if (lora_w.attention.o.has_value()) {
                    add_param(lora_w.attention.o->A);
                    add_param(lora_w.attention.o->B);
                }
                if (lora_w.mlp.gate.has_value()) {
                    add_param(lora_w.mlp.gate->A);
                    add_param(lora_w.mlp.gate->B);
                }
                if (lora_w.mlp.up.has_value()) {
                    add_param(lora_w.mlp.up->A);
                    add_param(lora_w.mlp.up->B);
                }
                if (lora_w.mlp.down.has_value()) {
                    add_param(lora_w.mlp.down->A);
                    add_param(lora_w.mlp.down->B);
                }
            }

            state.num_blocks = (state.state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Allocate momentum quantization map
            state.momentum_quantiles = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_quantiles", {256});
            std::vector<float> h_quantiles(256);
            optimizers::create_normuon_quantiles(h_quantiles.data());
            CUDA_CHECK(cudaMemcpy(state.momentum_quantiles.Data, h_quantiles.data(),
                                  256 * sizeof(float), cudaMemcpyHostToDevice));

            // Allocate 8-bit momentum state
            state.momentum_state = mAllocator->allocate(ETensorDType::BYTE, "lora_normuon_momentum",
                                                         {static_cast<long>(state.state_elems)});
            state.momentum_absmax = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_absmax",
                                                          {static_cast<long>(state.num_blocks)});

            // Initialize momentum state
            optimizers::init_normuon_momentum_state(
                reinterpret_cast<unsigned char*>(state.momentum_state.Data),
                state.momentum_absmax.template get<float>(),
                state.state_elems,
                main_stream
            );

            // Allocate variance buffers (one per tensor)
            for (const auto& shape : state.variance_shapes) {
                int M = shape.first;
                int N = shape.second;
                size_t var_size = optimizers::normuon_variance_buffer_size(M, N);
                Tensor var_buf = mAllocator->allocate(ETensorDType::FP32, "lora_normuon_variance",
                                                       {static_cast<long>(var_size)});
                // Initialize to 1.0 for stable first update
                std::vector<float> ones(var_size, 1.0f);
                CUDA_CHECK(cudaMemcpyAsync(var_buf.Data, ones.data(),
                                           var_size * sizeof(float), cudaMemcpyHostToDevice, main_stream));
                state.variance_buffers.push_back(std::move(var_buf));
            }

            // Allocate Polar Express workspace
            // Polar Express needs: A (M x M), B (M x M), C (M x N), X_tmp (M x N), plus scale floats
            // where M = max(rows, cols) after potential transpose. Total ~4 * max(M,N)^2 elements.
            // Also need space for momentum output before Polar Express (M x N elements)
            size_t max_dim = std::max(state.max_weight_M, state.max_weight_N);
            size_t max_weight_elems = state.max_weight_M * state.max_weight_N;
            size_t polar_workspace_elems = 4 * max_dim * max_dim + 1;  // +1 for scale float (as BF16)
            size_t polar_size = max_weight_elems + polar_workspace_elems;  // momentum_out + polar workspace
            state.polar_workspace = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_polar",
                                                          {static_cast<long>(polar_size)});

            // Allocate momentum temp buffer
            size_t max_weight_size = state.max_weight_M * state.max_weight_N;
            state.momentum_temp = mAllocator->allocate(ETensorDType::BF16, "lora_normuon_temp",
                                                        {static_cast<long>(max_weight_size)});

            // Create cuBLAS handle
            CUBLAS_CHECK(cublasCreate(&state.cublas_handle));
            CUBLAS_CHECK(cublasSetStream(state.cublas_handle, main_stream));

            fmt::print("LoRA NorMuon: {} params, {} blocks, max weight {}x{}\n",
                       state.total_params, state.num_blocks, state.max_weight_M, state.max_weight_N);
            fmt::print("LoRA NorMuon state: momentum={} bytes, variance={} buffers, polar={} elems\n",
                       state.state_elems, state.variance_buffers.size(), polar_size);

            state.initialized = true;
        }

        // Phase 2: Update parameters
        const ETensorDType lora_dtype = mLoRAConfig.dtype;
        size_t state_offset = 0;
        size_t var_idx = 0;
        bool unused_acc = false;

        auto update_param = [&](Tensor& param, Tensor& grad) {
            if (!param.Data) return;

            const auto& shape = state.variance_shapes[var_idx];
            int M = shape.first;
            int N = shape.second;
            size_t n = param.nelem();

            // Get state pointers with proper offset
            size_t aligned_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            unsigned char* momentum_ptr = reinterpret_cast<unsigned char*>(state.momentum_state.Data) + aligned_offset;
            float* absmax_ptr = state.momentum_absmax.template get<float>() + (aligned_offset / BLOCK_SIZE);
            float* variance_ptr = state.variance_buffers[var_idx].template get<float>();

            if (lora_dtype == ETensorDType::BF16) {
                // Note: normuon_update_2d applies lr_mult internally based on M, N
                optimizers::normuon_update_2d(
                    state.cublas_handle,
                    param.template get<nv_bfloat16>(),
                    grad.template get<nv_bfloat16>(),
                    momentum_ptr,
                    variance_ptr,
                    state.polar_workspace.template get<nv_bfloat16>(),
                    M, N,
                    lr,
                    momentum,
                    beta2,
                    cautious_wd ? weight_decay : 0.0f,  // NorMuon uses cautious WD internally
                    state.momentum_quantiles.template get<float>(),
                    absmax_ptr,
                    main_stream
                );
            } else {
                // FP32 LoRA weights - need to use FP32 overload
                // For now, fall back to momentum update + simple SGD
                // (Full NorMuon for FP32 requires additional kernel overloads)
                throw std::runtime_error("LoRA NorMuon optimizer currently only supports BF16 LoRA weights");
            }

            state_offset = aligned_offset + n;
            var_idx++;
        };

        // Update all LoRA tensors
        for (int l = 0; l < L; ++l) {
            auto& lora_w = mLoRAWeights->get_master_block(l, main_stream);
            auto& lora_g = mLoRAGrads->get_block_full(l, main_stream, comm, unused_acc);

            if (lora_w.attention.q.has_value()) {
                update_param(lora_w.attention.q->A, lora_g.attention.q->A);
                update_param(lora_w.attention.q->B, lora_g.attention.q->B);
            }
            if (lora_w.attention.k.has_value()) {
                update_param(lora_w.attention.k->A, lora_g.attention.k->A);
                update_param(lora_w.attention.k->B, lora_g.attention.k->B);
            }
            if (lora_w.attention.v.has_value()) {
                update_param(lora_w.attention.v->A, lora_g.attention.v->A);
                update_param(lora_w.attention.v->B, lora_g.attention.v->B);
            }
            if (lora_w.attention.o.has_value()) {
                update_param(lora_w.attention.o->A, lora_g.attention.o->A);
                update_param(lora_w.attention.o->B, lora_g.attention.o->B);
            }
            if (lora_w.mlp.gate.has_value()) {
                update_param(lora_w.mlp.gate->A, lora_g.mlp.gate->A);
                update_param(lora_w.mlp.gate->B, lora_g.mlp.gate->B);
            }
            if (lora_w.mlp.up.has_value()) {
                update_param(lora_w.mlp.up->A, lora_g.mlp.up->A);
                update_param(lora_w.mlp.up->B, lora_g.mlp.up->B);
            }
            if (lora_w.mlp.down.has_value()) {
                update_param(lora_w.mlp.down->A, lora_g.mlp.down->A);
                update_param(lora_w.mlp.down->B, lora_g.mlp.down->B);
            }
        }

        CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, main_stream));
    }

    /**
     * @brief Initialize multi-tensor optimizer state buffers.
     *
     * Pre-gathers all LoRA tensor pointers, sizes, and state offsets into device arrays.
     * This is done once at the start of training.
     */
    void initialize_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream) {
        auto& state = *mLoRAAdamW8BitState;
        const int L = (int)mBaseModel->config().NumLayers;

        // Count tensors and total params - use void* to handle both FP32 and BF16
        std::vector<void*> h_param_ptrs;
        std::vector<int> h_sizes;
        std::vector<int> h_state_offsets;
        size_t total_params = 0;

        auto collect_tensor = [&](Tensor& param) {
            if (!param.Data) return;
            h_param_ptrs.push_back(param.Data);  // Store raw pointer, type-checked at kernel dispatch
            int n = (int)param.nelem();
            h_sizes.push_back(n);
            h_state_offsets.push_back((int)total_params);
            total_params += n;
        };

        // Collect all LoRA tensors (master weights dtype matches lora_dtype)
        for (int l = 0; l < L; ++l) {
            auto& lora_w = mLoRAWeights->get_master_block(l, stream);

            if (lora_w.attention.q.has_value()) {
                collect_tensor(lora_w.attention.q->A);
                collect_tensor(lora_w.attention.q->B);
            }
            if (lora_w.attention.k.has_value()) {
                collect_tensor(lora_w.attention.k->A);
                collect_tensor(lora_w.attention.k->B);
            }
            if (lora_w.attention.v.has_value()) {
                collect_tensor(lora_w.attention.v->A);
                collect_tensor(lora_w.attention.v->B);
            }
            if (lora_w.attention.o.has_value()) {
                collect_tensor(lora_w.attention.o->A);
                collect_tensor(lora_w.attention.o->B);
            }
            if (lora_w.mlp.gate.has_value()) {
                collect_tensor(lora_w.mlp.gate->A);
                collect_tensor(lora_w.mlp.gate->B);
            }
            if (lora_w.mlp.up.has_value()) {
                collect_tensor(lora_w.mlp.up->A);
                collect_tensor(lora_w.mlp.up->B);
            }
            if (lora_w.mlp.down.has_value()) {
                collect_tensor(lora_w.mlp.down->A);
                collect_tensor(lora_w.mlp.down->B);
            }
        }

        state.num_tensors = (int)h_param_ptrs.size();
        state.total_params = total_params;
        constexpr size_t BLOCK_SIZE = 2048;
        state.num_blocks = (total_params + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate device buffers for pointer arrays (pointer size is same for all types)
        state.param_ptrs = mAllocator->allocate(ETensorDType::BYTE, "lora_mt_param_ptrs",
            EAllocationType::ON_DEVICE, {(long)(state.num_tensors * sizeof(void*))});
        state.grad_ptrs = mAllocator->allocate(ETensorDType::BYTE, "lora_mt_grad_ptrs",
            EAllocationType::ON_DEVICE, {(long)(state.num_tensors * sizeof(void*))});
        state.tensor_sizes = mAllocator->allocate(ETensorDType::INT32, "lora_mt_sizes",
            EAllocationType::ON_DEVICE, {(long)state.num_tensors});
        state.state_offsets = mAllocator->allocate(ETensorDType::INT32, "lora_mt_offsets",
            EAllocationType::ON_DEVICE, {(long)state.num_tensors});

        // Copy param pointers, sizes, and offsets to device
        CUDA_CHECK(cudaMemcpyAsync(state.param_ptrs.Data, h_param_ptrs.data(),
            h_param_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state.tensor_sizes.Data, h_sizes.data(),
            h_sizes.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(state.state_offsets.Data, h_state_offsets.data(),
            h_state_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Determine allocation location based on offload options
        state.offload_state = mOptions.OffloadOptimizer;
        state.use_zero_copy = mOptions.UseZeroCopy;

        EAllocationType alloc_kind = EAllocationType::ON_DEVICE;
        if (state.offload_state) {
            if (state.use_zero_copy) {
                alloc_kind = mOptions.offload_alloc();
            } else {
                alloc_kind = EAllocationType::ON_DEVICE;
            }
        }

        // Allocate optimizer state tensors
        state.state1 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state1",
            alloc_kind, {(long)total_params});
        state.state2 = mAllocator->allocate(ETensorDType::BYTE, "lora_adamw8bit_state2",
            alloc_kind, {(long)total_params});
        state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax1",
            alloc_kind, {(long)state.num_blocks});
        state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_absmax2",
            alloc_kind, {(long)state.num_blocks});

        // Initialize optimizer state
        init_adamw8bit_state(
            reinterpret_cast<unsigned char*>(state.state1.Data),
            reinterpret_cast<unsigned char*>(state.state2.Data),
            state.absmax1.template get<float>(),
            state.absmax2.template get<float>(),
            total_params, stream
        );

        state.initialized = true;
    }

    /**
     * @brief Update gradient pointers in device buffer.
     *
     * Called each step because gradient buffers may change (e.g., after accumulation reset).
     * This is a small host->device copy but avoids the overhead of 336 kernel launches.
     */
    void update_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream) {
        auto& state = *mLoRAAdamW8BitState;
        const int L = (int)mBaseModel->config().NumLayers;

        // Use void* to handle both FP32 and BF16 gradient pointers
        std::vector<void*> h_grad_ptrs;
        h_grad_ptrs.reserve(state.num_tensors);

        bool unused_acc = false;

        auto collect_grad = [&](std::optional<LoRALayerWeights<Tensor>>& grad_opt) {
            if (!grad_opt.has_value()) return;
            if (grad_opt->A.Data) h_grad_ptrs.push_back(grad_opt->A.Data);
            if (grad_opt->B.Data) h_grad_ptrs.push_back(grad_opt->B.Data);
        };

        for (int l = 0; l < L; ++l) {
            auto& lora_g = mLoRAGrads->get_block_full(l, stream, comm, unused_acc);

            collect_grad(lora_g.attention.q);
            collect_grad(lora_g.attention.k);
            collect_grad(lora_g.attention.v);
            collect_grad(lora_g.attention.o);
            collect_grad(lora_g.mlp.gate);
            collect_grad(lora_g.mlp.up);
            collect_grad(lora_g.mlp.down);
        }

        // Copy gradient pointers to device
        CUDA_CHECK(cudaMemcpyAsync(state.grad_ptrs.Data, h_grad_ptrs.data(),
            h_grad_ptrs.size() * sizeof(void*), cudaMemcpyHostToDevice, stream));
    }

    float get_loss() const override { return mBaseModel->get_loss(); }

    ITensorContainer& weights() override {
        if (lora_enabled()) return *mLoRAWeights;
        return mBaseModel->weights();
    }
    ITensorContainer& opt_momentum() override {
        // 8-bit optimizer doesn't expose state as ITensorContainer
        if (lora_enabled()) return detail::empty_tensor_container();
        return mBaseModel->opt_momentum();
    }
    ITensorContainer& opt_momentum_scales() override {
        // 8-bit optimizer doesn't use FP8 scales
        if (lora_enabled()) return detail::empty_tensor_container();
        return mBaseModel->opt_momentum_scales();
    }
    ITensorContainer& opt_variance() override {
        // 8-bit optimizer doesn't expose state as ITensorContainer
        if (lora_enabled()) return detail::empty_tensor_container();
        return mBaseModel->opt_variance();
    }
    ITensorContainer& opt_variance_scales() override {
        // 8-bit optimizer doesn't use FP8 scales
        if (lora_enabled()) return detail::empty_tensor_container();
        return mBaseModel->opt_variance_scales();
    }

    std::vector<std::byte> rng_state() const override { return mBaseModel->rng_state(); }
    void set_rng_state(const std::vector<std::byte>& state) override { mBaseModel->set_rng_state(state); }
    std::string_view model_type() const override { return mBaseModel->model_type(); }
    IRunState& get_run_state() const override { return mBaseModel->get_run_state(); }

    // LoRA API (used by train.cpp debug / export)
    [[nodiscard]] bool lora_enabled() const { return mLoRAConfig.enabled(); }
    [[nodiscard]] bool qlora_enabled() const { return mQLoRAConfig.is_quantized(); }
    [[nodiscard]] std::size_t lora_num_parameters() const { return mLoRAWeights ? mLoRAWeights->num_parameters() : 0; }

    ModularTransformerModel<Block>& base_model() { return *mBaseModel; }
    const ModularLoRAConfig& lora_config() const { return mLoRAConfig; }
    const QLoRAConfig& qlora_config() const { return mQLoRAConfig; }
    ModularLoRAWeightsManager& lora_weights() { return *mLoRAWeights; }
    ModularLoRAGradsManager& lora_grads() { return *mLoRAGrads; }

    // QLoRA memory stats (supports FP8, FP4, and BnB NF4)
    [[nodiscard]] std::size_t qlora_quantized_weights_bytes() const {
        if (mFP4WeightProvider) return mFP4WeightProvider->quantized_weights_bytes();
        if (mFP8WeightProvider) return mFP8WeightProvider->quantized_weights_bytes();
        if (mBnBWeightProvider) return mBnBWeightProvider->quantized_weights_bytes();
        return 0;
    }
    [[nodiscard]] float qlora_memory_savings_ratio() const {
        if (mFP4WeightProvider) return mFP4WeightProvider->memory_savings_ratio();
        if (mFP8WeightProvider) return mFP8WeightProvider->memory_savings_ratio();
        if (mBnBWeightProvider) return mBnBWeightProvider->memory_savings_ratio();
        return 1.0f;
    }

    void export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path = "") {
        if (!lora_enabled()) return;
        namespace fs = std::filesystem;
        fs::path dir(directory);

        // Only global rank 0 creates the directory to avoid NFS race conditions
        // when multiple nodes try to create the same directory simultaneously
        if (comm.rank() == 0) {
            fs::create_directories(dir);
        }

        // Full barrier to ensure directory exists before any rank proceeds
        comm.barrier();
        
        mLoRAWeights->export_to_file((dir / "adapter_model.safetensors").string(), comm);
        
        if (comm.rank() == 0) {
            nlohmann::json adapter_config;
            adapter_config["base_model_name_or_path"] = base_model_path;
            adapter_config["peft_type"] = "LORA";
            adapter_config["task_type"] = "CAUSAL_LM";
            adapter_config["r"] = mLoRAConfig.rank;
            adapter_config["lora_alpha"] = mLoRAConfig.alpha;
            adapter_config["lora_dropout"] = mLoRAConfig.dropout;
            adapter_config["fan_in_fan_out"] = false;
            adapter_config["bias"] = "none";
            adapter_config["use_rslora"] = mLoRAConfig.use_rs_lora;
            adapter_config["target_modules"] = detail::targets_to_peft_names(mLoRAConfig);

            std::ofstream config_file(dir / "adapter_config.json");
            config_file << adapter_config.dump(2);
        }

        // Note: No barrier needed here - export_to_file already synchronizes all ranks
    }

    void import_adapter(const std::string& file_name, NCCLCommunicator& comm) {
        if (!lora_enabled()) return;
        mLoRAWeights->import_from_file(file_name, comm);
    }

    void save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override {
        if (!lora_enabled()) return;
        export_adapter(checkpoint_dir, comm);
    }

    void load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override {
        if (!lora_enabled()) return;
        namespace fs = std::filesystem;
        fs::path adapter_file = fs::path(checkpoint_dir) / "adapter_model.safetensors";
        if (fs::exists(adapter_file)) {
            import_adapter(adapter_file.string(), comm);
        }
    }

    float debug_get_grad_scale() const {
        if (!lora_enabled() || !mLoRARunState) return std::numeric_limits<float>::quiet_NaN();
        float v = std::numeric_limits<float>::quiet_NaN();
        const float* device_ptr = mLoRARunState->norm_buffer.template get<float>() + 1;
        CUDA_CHECK(cudaMemcpy(&v, device_ptr, sizeof(v), cudaMemcpyDeviceToHost));
        return v;
    }

    int debug_get_valid_tokens() const {
        int v = 0;
        auto& rs = mBaseModel->get_run_state();
        CUDA_CHECK(cudaMemcpy(&v, rs.ValidTokenCount.template get<int>(), sizeof(v), cudaMemcpyDeviceToHost));
        return v;
    }

    std::vector<float> debug_get_grad_norms_by_layer(NCCLCommunicator& comm) const {
        if (!lora_enabled() || !mLoRARunState || !mLoRAGrads) {
            return {};
        }

        auto& rs = mBaseModel->get_run_state();
        cudaStream_t stream = rs.MainStream;
        const auto& cfg = mBaseModel->config();

        std::vector<float> norms;
        norms.resize((std::size_t)cfg.NumLayers, 0.0f);

        Tensor& buf = mLoRARunState->norm_buffer;

        auto add = [&](const Tensor& grad) {
            if (grad.Data) {
                global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
            }
        };

        for (int l = 0; l < cfg.NumLayers; ++l) {
            fill_zero(buf, stream);

            bool unused_accumulate = false;
            auto& g = mLoRAGrads->get_block_full(l, stream, comm, unused_accumulate);

            if (g.attention.q.has_value()) { add(g.attention.q->A); add(g.attention.q->B); }
            if (g.attention.k.has_value()) { add(g.attention.k->A); add(g.attention.k->B); }
            if (g.attention.v.has_value()) { add(g.attention.v->A); add(g.attention.v->B); }
            if (g.attention.o.has_value()) { add(g.attention.o->A); add(g.attention.o->B); }
            if (g.mlp.gate.has_value()) { add(g.mlp.gate->A); add(g.mlp.gate->B); }
            if (g.mlp.up.has_value()) { add(g.mlp.up->A); add(g.mlp.up->B); }
            if (g.mlp.down.has_value()) { add(g.mlp.down->A); add(g.mlp.down->B); }

            deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem(), stream);
            if (comm.world_size() > 1) {
                comm.reduce_norm(buf.template get<float>(), stream);
            }
            CUDA_CHECK(cudaMemcpy(&norms[(std::size_t)l], buf.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
            norms[(std::size_t)l] = std::sqrt(std::max(0.0f, norms[(std::size_t)l]));
        }

        return norms;
    }

    std::vector<std::pair<std::string, float>> debug_get_grad_norms_by_module(int layer_idx, NCCLCommunicator& comm) const {
        if (!lora_enabled() || !mLoRARunState || !mLoRAGrads) {
            return {};
        }

        const auto& cfg = mBaseModel->config();
        if (layer_idx < 0 || layer_idx >= cfg.NumLayers) {
            return {};
        }

        auto& rs = mBaseModel->get_run_state();
        cudaStream_t stream = rs.MainStream;

        Tensor& buf = mLoRARunState->norm_buffer;

        auto add = [&](const Tensor& grad) {
            if (grad.Data) {
                global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
            }
        };

        auto module_norm = [&](const Tensor& A, const Tensor& B) -> float {
            fill_zero(buf, stream);
            add(A);
            add(B);
            deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem(), stream);
            if (comm.world_size() > 1) {
                comm.reduce_norm(buf.template get<float>(), stream);
            }
            float v = 0.0f;
            CUDA_CHECK(cudaMemcpy(&v, buf.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost));
            return std::sqrt(std::max(0.0f, v));
        };

        bool unused_accumulate = false;
        auto& g = mLoRAGrads->get_block_full(layer_idx, stream, comm, unused_accumulate);

        std::vector<std::pair<std::string, float>> out;
        out.reserve(8);

        if (g.attention.q.has_value()) out.emplace_back("q_proj", module_norm(g.attention.q->A, g.attention.q->B));
        if (g.attention.k.has_value()) out.emplace_back("k_proj", module_norm(g.attention.k->A, g.attention.k->B));
        if (g.attention.v.has_value()) out.emplace_back("v_proj", module_norm(g.attention.v->A, g.attention.v->B));
        if (g.attention.o.has_value()) out.emplace_back("o_proj", module_norm(g.attention.o->A, g.attention.o->B));
        if (g.mlp.gate.has_value()) out.emplace_back("gate_proj", module_norm(g.mlp.gate->A, g.mlp.gate->B));
        if (g.mlp.up.has_value()) out.emplace_back("up_proj", module_norm(g.mlp.up->A, g.mlp.up->B));
        if (g.mlp.down.has_value()) out.emplace_back("down_proj", module_norm(g.mlp.down->A, g.mlp.down->B));

        return out;
    }

private:
    struct LoRARunState {
        Tensor intermediate;   // (BT, rank) - first intermediate buffer
        Tensor intermediate2;  // (BT, rank) - second intermediate buffer for fused ops
        Tensor slice;
        Tensor norm_buffer;
        Tensor recompute_ln;   // (B, T, C) - buffer for recomputed ln1/ln2 activations
        Tensor recompute_rstd; // (B, T) - buffer for recomputed rstd (unused but required by kernel)
        int B = 0;
        int T = 0;
    };

    // 8-bit AdamW optimizer state for LoRA weights
    struct LoRAAdamW8BitState {
        bool initialized = false;
        size_t total_params = 0;
        size_t num_blocks = 0;
        int num_tensors = 0;

        // Offloading configuration
        bool offload_state = false;  // If true, state tensors are in pinned host memory
        bool use_zero_copy = false;  // If true, use zero-copy access instead of transfers

        Tensor quantiles1;   // 256 entries - quantization map for m
        Tensor quantiles2;   // 256 entries - quantization map for v
        Tensor state1;       // uint8 quantized m, size = total_params
        Tensor state2;       // uint8 quantized v, size = total_params
        Tensor absmax1;      // per-block absmax for m
        Tensor absmax2;      // per-block absmax for v

        // Multi-tensor optimizer buffers (device memory)
        // Pre-allocated arrays of pointers/sizes to avoid per-step CPU work
        Tensor param_ptrs;      // float** or nv_bfloat16** - array of param pointers
        Tensor grad_ptrs;       // float** or nv_bfloat16** - array of grad pointers
        Tensor tensor_sizes;    // int* - array of tensor sizes
        Tensor state_offsets;   // int* - element offset for each tensor in state buffers
    };

    // NorMuon optimizer state for LoRA weights
    // Uses 8-bit quantized momentum + FP32 variance buffers
    struct LoRANorMuonState {
        bool initialized = false;
        size_t total_params = 0;
        size_t state_elems = 0;
        size_t num_blocks = 0;

        // 8-bit quantized momentum buffer (combined for all LoRA weights)
        Tensor momentum_quantiles;  // float[256] - signed quantization map
        Tensor momentum_state;      // uint8[state_elems]
        Tensor momentum_absmax;     // float[num_blocks]

        // Variance buffers - stored per LoRA tensor as FP32
        // For LoRA, each A/B matrix is a 2D weight
        std::vector<Tensor> variance_buffers;
        std::vector<std::pair<int, int>> variance_shapes;  // (M, N) for each buffer

        // Polar Express workspace (reused across layers)
        Tensor polar_workspace;
        size_t max_weight_M = 0;  // Max weight rows seen
        size_t max_weight_N = 0;  // Max weight cols seen

        // Temporary buffer for dequantized momentum (reused per weight)
        Tensor momentum_temp;  // BF16[max_weight_size]

        // cuBLAS handle for Polar Express matrix multiplications
        cublasHandle_t cublas_handle = nullptr;

        ~LoRANorMuonState() {
            if (cublas_handle) {
                cublasDestroy(cublas_handle);
                cublas_handle = nullptr;
            }
        }
    };

    std::unique_ptr<ModularTransformerModel<Block>> mBaseModel;
    ModularLoRAConfig mLoRAConfig;
    QLoRAConfig mQLoRAConfig;
    RuntimeOptions mOptions;
    std::shared_ptr<TensorAllocator> mAllocator;

    std::unique_ptr<ModularLoRAWeightsManager> mLoRAWeights;
    std::unique_ptr<ModularLoRAGradsManager> mLoRAGrads;
    std::unique_ptr<LoRAAdamW8BitState> mLoRAAdamW8BitState;
    std::unique_ptr<LoRANorMuonState> mLoRANorMuonState;
    std::unique_ptr<LoRARunState> mLoRARunState;

    // QLoRA: quantized base weight provider (FP8 with per-block scales)
    std::unique_ptr<FP8WeightProvider<Block>> mFP8WeightProvider;

    // QLoRA FP4: quantized base weight provider (FP4 E2M1 with two-level block scales)
    std::unique_ptr<FP4WeightProvider<Block>> mFP4WeightProvider;

    // QLoRA BnB: BitsAndBytes NF4 quantized base weight provider (works on any GPU)
    std::unique_ptr<BnBWeightProvider<Block>> mBnBWeightProvider;

    std::minstd_rand mLoRAOptimizerRNG;

    void allocate_lora_run_state(NCCLCommunicator& comm, int B, int T) {
        (void)comm;
        if (!lora_enabled()) return;

        mLoRARunState = std::make_unique<LoRARunState>();
        mLoRARunState->B = B;
        mLoRARunState->T = T;

        auto ctx = mAllocator->with_context("Modular_LoRA_RunState");

        const auto& cfg = mBaseModel->config();
        const int rank = mLoRAConfig.rank;
        const int BT = B * T;
        const int max_features = std::max(cfg.HiddenSize, cfg.IntermediateSize);
        const long max_slice_elems = (long)BT * (long)max_features;

        ETensorDType work_dtype = cfg.DType;

        mLoRARunState->intermediate = mAllocator->allocate(
            work_dtype, "lora_intermediate", EAllocationType::ON_DEVICE, {BT, rank});
        mLoRARunState->intermediate2 = mAllocator->allocate(
            work_dtype, "lora_intermediate2", EAllocationType::ON_DEVICE, {BT, rank});
        mLoRARunState->slice = mAllocator->allocate(
            work_dtype, "lora_slice", EAllocationType::ON_DEVICE, {max_slice_elems});

        auto& rs = mBaseModel->run_state();
        const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(rs.DeviceProp)));
        mLoRARunState->norm_buffer = mAllocator->allocate(
            ETensorDType::FP32, "lora_norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums + 2});

        // Allocate recompute buffers only when recompute_lora is enabled
        if (rs.config().recompute_lora) {
            const int C = cfg.HiddenSize;
            mLoRARunState->recompute_ln = mAllocator->allocate(
                work_dtype, "lora_recompute_ln", EAllocationType::ON_DEVICE, {B, T, C});
            mLoRARunState->recompute_rstd = mAllocator->allocate(
                ETensorDType::FP32, "lora_recompute_rstd", EAllocationType::ON_DEVICE, {B, T});
        }
    }

    void ensure_lora_run_state(NCCLCommunicator& comm, int B, int T) {
        if (!lora_enabled()) return;
        if (!mLoRARunState || mLoRARunState->B != B || mLoRARunState->T != T) {
            allocate_lora_run_state(comm, B, T);
        }
    }

    /**
     * @brief Recompute RMSNorm output for LoRA backward when recompute_lora is enabled.
     *
     * Instead of storing ln1/ln2 per-layer, we recompute them from the residual stream
     * during LoRA backward. This trades compute for memory (~350MB for 4B model).
     *
     * @param residual Input residual tensor (B, T, C)
     * @param ln_weight RMSNorm weight tensor (C,)
     * @param epsilon RMSNorm epsilon
     * @param B Batch size
     * @param T Sequence length
     * @param C Hidden size
     * @param stream CUDA stream
     * @return Reference to the recomputed LN output in mLoRARunState->recompute_ln
     */
    Tensor& recompute_rmsnorm(const Tensor& residual, const Tensor& ln_weight,
                              float epsilon, int B, int T, int C, cudaStream_t stream) {
        rmsnorm_forward(mLoRARunState->recompute_ln, mLoRARunState->recompute_rstd,
                        residual, ln_weight, nullptr, epsilon, B, T, C, stream);
        return mLoRARunState->recompute_ln;
    }

    /**
     * @brief Load and quantize base model weights for QLoRA mode
     *
     * Creates FP8WeightProvider, loads and quantizes base model weights to FP8,
     * then injects the weight provider into the base model's weight manager.
     *
     * @param file_name Path to safetensors model file
     * @param comm NCCL communicator for multi-GPU sync
     */
    void import_weights_qlora(const std::string& file_name, NCCLCommunicator& comm) {
        const auto& cfg = mBaseModel->config();

        // Get CUDA device properties
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_props{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

        // Create a CUDA stream for quantization
        cudaStream_t quant_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&quant_stream));

        // Create QLoRA weight provider config
        typename FP8WeightProvider<Block>::Config qwp_config{};
        qwp_config.num_layers = cfg.NumLayers;
        qwp_config.hidden_size = cfg.HiddenSize;
        qwp_config.intermediate_size = cfg.IntermediateSize;
        qwp_config.num_query_heads = cfg.NumQueryHeads;
        qwp_config.num_kv_heads = cfg.NumKeyValHeads;
        qwp_config.head_size = cfg.head_size();
        qwp_config.vocab_size = cfg.VocabSize;
        qwp_config.qlora_config = mQLoRAConfig;
        qwp_config.lora_config = mLoRAConfig;
        qwp_config.model_dtype = cfg.DType;
        qwp_config.use_qk_norm = cfg.UseQKNorm;
        qwp_config.tied_embeddings = cfg.TiedWordEmbeddings;
        qwp_config.shard_idx = comm.rank();
        qwp_config.num_shards = comm.world_size();
        // Pass FP8 flags to enable native FP8 path (skip dequantization)
        qwp_config.enable_fp8_forward = mOptions.fp8_forward_enabled();
        qwp_config.enable_fp8_hybrid = mOptions.fp8_hybrid_enabled();

        // Create the QLoRA weight provider
        mFP8WeightProvider = std::make_unique<FP8WeightProvider<Block>>(
            qwp_config, *mAllocator, device_props);

        // Import and quantize weights
        mFP8WeightProvider->import_and_quantize(file_name, comm, quant_stream);

        // Synchronize before cleanup
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        CUDA_CHECK(cudaStreamDestroy(quant_stream));

        // Inject the weight provider into the base model's weight manager
        // This makes get_block() return dequantized weights from QLoRA storage
        auto& weights_manager = mBaseModel->weights_manager();
        weights_manager.set_weight_provider(
            [this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
                return mFP8WeightProvider->get_block(layer_idx, stream);
            });

        // Also inject providers for non-block weights (embeddings, final_norm, lm_head)
        weights_manager.set_embeddings_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP8WeightProvider->get_embeddings(stream);
            });
        weights_manager.set_final_norm_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP8WeightProvider->get_final_norm(stream);
            });
        weights_manager.set_lm_head_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP8WeightProvider->get_lm_head(stream);
            });

        // If FP8 hybrid mode is enabled, also set the FP8 cache provider
        // This allows the forward pass to access FP8 weights via fp8_weight_cache()
        if (mFP8WeightProvider->has_fp8_forward_cache()) {
            weights_manager.set_fp8_cache_provider(
                [this]() -> typename ModularWeightManager<Block>::FP8WeightCache& {
                    return mFP8WeightProvider->get_fp8_cache();
                });
        }
    }

    /**
     * @brief Load and quantize base model weights for FP4 QLoRA mode
     *
     * Creates FP4WeightProvider, loads and quantizes base model weights to FP4,
     * then injects the weight provider into the base model's weight manager.
     *
     * @param file_name Path to safetensors model file
     * @param comm NCCL communicator for multi-GPU sync
     */
    void import_weights_fp4_qlora(const std::string& file_name, NCCLCommunicator& comm) {
        const auto& cfg = mBaseModel->config();

        // Get CUDA device properties
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_props{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

        // Check for Blackwell (SM100+) requirement
        if (device_props.major < 10) {
            std::cerr << "[FP4-QLoRA WARNING] FP4 (NVFP4) requires Blackwell (SM100+), "
                      << "but detected SM" << device_props.major << "." << device_props.minor
                      << ". Proceeding anyway, but native FP4 matmuls won't work.\n";
        }

        // Create a CUDA stream for quantization
        cudaStream_t quant_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&quant_stream));

        // Create FP4 weight provider config
        typename FP4WeightProvider<Block>::Config fp4_config{};
        fp4_config.num_layers = cfg.NumLayers;
        fp4_config.hidden_size = cfg.HiddenSize;
        fp4_config.intermediate_size = cfg.IntermediateSize;
        fp4_config.num_query_heads = cfg.NumQueryHeads;
        fp4_config.num_kv_heads = cfg.NumKeyValHeads;
        fp4_config.head_size = cfg.head_size();
        fp4_config.vocab_size = cfg.VocabSize;
        fp4_config.qlora_config = mQLoRAConfig;
        fp4_config.lora_config = mLoRAConfig;
        fp4_config.model_dtype = cfg.DType;
        fp4_config.use_qk_norm = cfg.UseQKNorm;
        fp4_config.tied_embeddings = cfg.TiedWordEmbeddings;
        fp4_config.shard_idx = comm.rank();
        fp4_config.num_shards = comm.world_size();

        // Create the FP4 weight provider
        mFP4WeightProvider = std::make_unique<FP4WeightProvider<Block>>(
            fp4_config, *mAllocator, device_props);

        // Import and quantize weights to FP4
        mFP4WeightProvider->import_and_quantize(file_name, comm, quant_stream);

        // Synchronize before cleanup
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        CUDA_CHECK(cudaStreamDestroy(quant_stream));

        // Inject the weight provider into the base model's weight manager
        // This makes get_block() return dequantized weights from FP4 storage
        auto& weights_manager = mBaseModel->weights_manager();
        weights_manager.set_weight_provider(
            [this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
                return mFP4WeightProvider->get_block(layer_idx, stream);
            });

        // Also inject providers for non-block weights (embeddings, final_norm, lm_head)
        weights_manager.set_embeddings_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP4WeightProvider->get_embeddings(stream);
            });
        weights_manager.set_final_norm_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP4WeightProvider->get_final_norm(stream);
            });
        weights_manager.set_lm_head_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mFP4WeightProvider->get_lm_head(stream);
            });
    }

    /**
     * @brief Load and quantize base model weights for BitsAndBytes NF4 QLoRA mode
     *
     * Creates BnBWeightProvider, loads and quantizes base model weights to NF4,
     * then injects the weight provider into the base model's weight manager.
     *
     * This mode works on any CUDA GPU (no SM89+ or SM100+ requirement).
     *
     * @param file_name Path to safetensors model file
     * @param comm NCCL communicator for multi-GPU sync
     */
    void import_weights_bnb_qlora(const std::string& file_name, NCCLCommunicator& comm) {
        const auto& cfg = mBaseModel->config();

        // Get CUDA device properties
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_props{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

        // Create a CUDA stream for quantization
        cudaStream_t quant_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&quant_stream));

        // Create BnB weight provider config
        typename BnBWeightProvider<Block>::Config bnb_config{};
        bnb_config.num_layers = cfg.NumLayers;
        bnb_config.hidden_size = cfg.HiddenSize;
        bnb_config.intermediate_size = cfg.IntermediateSize;
        bnb_config.num_query_heads = cfg.NumQueryHeads;
        bnb_config.num_kv_heads = cfg.NumKeyValHeads;
        bnb_config.head_size = cfg.head_size();
        bnb_config.vocab_size = cfg.VocabSize;
        bnb_config.qlora_config = mQLoRAConfig;
        bnb_config.lora_config = mLoRAConfig;
        bnb_config.model_dtype = cfg.DType;
        bnb_config.use_qk_norm = cfg.UseQKNorm;
        bnb_config.tied_embeddings = cfg.TiedWordEmbeddings;
        bnb_config.shard_idx = comm.rank();
        bnb_config.num_shards = comm.world_size();

        // Create the BnB weight provider
        mBnBWeightProvider = std::make_unique<BnBWeightProvider<Block>>(
            bnb_config, *mAllocator, device_props);

        // Import and quantize weights to NF4
        mBnBWeightProvider->import_and_quantize(file_name, comm, quant_stream);

        // Synchronize before cleanup
        CUDA_CHECK(cudaStreamSynchronize(quant_stream));
        CUDA_CHECK(cudaStreamDestroy(quant_stream));

        // Inject the weight provider into the base model's weight manager
        // This makes get_block() return dequantized weights from BnB NF4 storage
        auto& weights_manager = mBaseModel->weights_manager();
        weights_manager.set_weight_provider(
            [this](int layer_idx, cudaStream_t stream) -> typename Block::Weights& {
                return mBnBWeightProvider->get_block(layer_idx, stream);
            });

        // Also inject providers for non-block weights (embeddings, final_norm, lm_head)
        weights_manager.set_embeddings_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mBnBWeightProvider->get_embeddings(stream);
            });
        weights_manager.set_final_norm_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mBnBWeightProvider->get_final_norm(stream);
            });
        weights_manager.set_lm_head_provider(
            [this](cudaStream_t stream) -> Tensor& {
                return mBnBWeightProvider->get_lm_head(stream);
            });
    }

    void backward_lora_qkv(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
        const auto& cfg = mBaseModel->config();
        const int C = cfg.HiddenSize;
        const int Hq = cfg.NumQueryHeads;
        const int Hkv = cfg.NumKeyValHeads;
        const int Hs = cfg.head_size();
        const int rank = mLoRAConfig.rank;

        auto& rs = mBaseModel->run_state();
        auto& a = rs.simplified_acts(layer_idx);
        auto& da = rs.simplified_grads(layer_idx);

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
        bool lora_accum = false;
        auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
        lora_accum = lora_accum || accumulate;

        // Get ln1 input: either from stored activation or recompute from residual
        Tensor ln1_input;
        if (rs.config().recompute_lora) {
            // Recompute ln1 from the residual input
            Tensor& residual = rs.get_residual(layer_idx, stream);
            auto& block_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);
            ln1_input = recompute_rmsnorm(residual, block_weights.ln1.weight,
                                          cfg.RmsNormEps, B, T, C, stream);
        } else {
            ln1_input = a.ln1;
        }

        // Prepare gradient tensors (use empty tensor if projection not enabled)
        Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
        LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

        if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
            dA_q = lora_grads.attention.q->A;
            dB_q = lora_grads.attention.q->B;
            lora_q = *lora_block.attention.q;
        }
        if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
            dA_k = lora_grads.attention.k->A;
            dB_k = lora_grads.attention.k->B;
            lora_k = *lora_block.attention.k;
        }
        if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
            dA_v = lora_grads.attention.v->A;
            dB_v = lora_grads.attention.v->B;
            lora_v = *lora_block.attention.v;
        }

        detail::backward_lora_qkv_fused(
            dA_q, dB_q,
            dA_k, dB_k,
            dA_v, dB_v,
            da.d_ln1,
            da.d_qkv,
            ln1_input,
            lora_q, lora_k, lora_v,
            mLoRAConfig.scaling(),
            B * T,
            C,
            Hq * Hs,
            Hkv * Hs,
            rank,
            lora_accum,
            mLoRARunState->intermediate,
            mLoRARunState->intermediate2,
            mLoRARunState->slice,
            rs.CublasLtHandle,
            rs.CuBlasWorkspace,
            stream);
    }

    void backward_lora_attn_out(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
        const auto& cfg = mBaseModel->config();
        const int C = cfg.HiddenSize;
        const int Hq = cfg.NumQueryHeads;
        const int Hs = cfg.head_size();
        const int rank = mLoRAConfig.rank;

        auto& rs = mBaseModel->run_state();
        auto& a = rs.simplified_acts(layer_idx);
        auto& da = rs.simplified_grads(layer_idx);

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
        if (!lora_block.attention.o.has_value()) return;

        bool lora_accum = false;
        auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
        lora_accum = lora_accum || accumulate;
        if (!lora_grads.attention.o.has_value()) return;

        Tensor x = a.att;
        Tensor dL_dy = da.d_res_att;

        detail::backward_lora_layer(lora_grads.attention.o->A, lora_grads.attention.o->B,
                                   da.d_att,
                                   dL_dy, 0,
                                   x,
                                   lora_block.attention.o->A, lora_block.attention.o->B,
                                   mLoRAConfig.scaling(),
                                   mLoRARunState->intermediate, mLoRARunState->slice,
                                   B * T, Hq * Hs, C, rank, lora_accum,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
    }

    void backward_lora_mlp_up(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
        const auto& cfg = mBaseModel->config();
        const int C = cfg.HiddenSize;
        const int D = cfg.IntermediateSize;
        const int rank = mLoRAConfig.rank;

        auto& rs = mBaseModel->run_state();
        auto& a = rs.simplified_acts(layer_idx);
        auto& da = rs.simplified_grads(layer_idx);

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
        bool lora_accum = false;
        auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
        lora_accum = lora_accum || accumulate;

        // Get ln2 input: either from stored activation or recompute from residual_att
        Tensor ln2_input;
        if (rs.config().recompute_lora) {
            // Recompute ln2 from residual_att (which is residual + att_out)
            // Note: residual_att is always stored even in recompute_block mode
            auto& block_weights = mBaseModel->weights_manager().get_block(layer_idx, stream);
            ln2_input = recompute_rmsnorm(a.residual_att, block_weights.ln2.weight,
                                          cfg.RmsNormEps, B, T, C, stream);
        } else {
            ln2_input = a.ln2;
        }

        // Prepare gradient tensors (use empty tensor if projection not enabled)
        Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
        LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

        if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
            dA_up = lora_grads.mlp.up->A;
            dB_up = lora_grads.mlp.up->B;
            lora_up = *lora_block.mlp.up;
        }
        if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
            dA_gate = lora_grads.mlp.gate->A;
            dB_gate = lora_grads.mlp.gate->B;
            lora_gate = *lora_block.mlp.gate;
        }

        detail::backward_lora_mlp_up_gate_fused(
            dA_up, dB_up,
            dA_gate, dB_gate,
            da.d_ln2,
            da.d_mlp_up,
            ln2_input,
            lora_up, lora_gate,
            mLoRAConfig.scaling(),
            B * T,
            C,
            D,
            rank,
            lora_accum,
            mLoRARunState->intermediate,
            mLoRARunState->intermediate2,
            mLoRARunState->slice,
            rs.CublasLtHandle,
            rs.CuBlasWorkspace,
            stream);
    }

    void backward_lora_mlp_down(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream) {
        const auto& cfg = mBaseModel->config();
        const int C = cfg.HiddenSize;
        const int D = cfg.IntermediateSize;
        const int rank = mLoRAConfig.rank;

        auto& rs = mBaseModel->run_state();
        auto& a = rs.simplified_acts(layer_idx);
        auto& da = rs.simplified_grads(layer_idx);

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);
        if (!lora_block.mlp.down.has_value()) return;

        bool lora_accum = false;
        auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
        lora_accum = lora_accum || accumulate;
        if (!lora_grads.mlp.down.has_value()) return;

        Tensor x = a.swiglu;
        Tensor dL_dy = da.d_res_ffn;

        detail::backward_lora_layer(lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                                   da.d_swiglu,
                                   dL_dy, 0,
                                   x,
                                   lora_block.mlp.down->A, lora_block.mlp.down->B,
                                   mLoRAConfig.scaling(),
                                   mLoRARunState->intermediate, mLoRARunState->slice,
                                   B * T, D, C, rank, lora_accum,
                                   rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
    }

    void calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip) {
        NVTX_RANGE_FN();
        auto& rs = mBaseModel->get_run_state();
        cudaStream_t stream = rs.MainStream;

        Tensor& buf = mLoRARunState->norm_buffer;
        fill_zero(buf, stream);

        auto norm_squared = [&](const Tensor& grad) {
            if (grad.Data) {
                global_norm_squared(buf, grad, grad.nelem(), rs.DeviceProp, stream);
            }
        };

        const auto& cfg = mBaseModel->config();
        for (int l = 0; l < cfg.NumLayers; ++l) {
            bool unused_acc = false;
            auto& g = mLoRAGrads->get_block_full(l, stream, comm, unused_acc);

            if (g.attention.q.has_value()) { norm_squared(g.attention.q->A); norm_squared(g.attention.q->B); }
            if (g.attention.k.has_value()) { norm_squared(g.attention.k->A); norm_squared(g.attention.k->B); }
            if (g.attention.v.has_value()) { norm_squared(g.attention.v->A); norm_squared(g.attention.v->B); }
            if (g.attention.o.has_value()) { norm_squared(g.attention.o->A); norm_squared(g.attention.o->B); }
            if (g.mlp.gate.has_value()) { norm_squared(g.mlp.gate->A); norm_squared(g.mlp.gate->B); }
            if (g.mlp.up.has_value()) { norm_squared(g.mlp.up->A); norm_squared(g.mlp.up->B); }
            if (g.mlp.down.has_value()) { norm_squared(g.mlp.down->A); norm_squared(g.mlp.down->B); }
        }

        deterministic_sum(buf.template get<float>(), buf.template get<float>(), buf.nelem() - 2, stream);

        float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                           * static_cast<float>(std::max(1, rs.GradAccumSteps))
                           * static_cast<float>(std::max(1, comm.world_size()));
        global_norm_sqrt(buf.template get<float>(), rs.NormHost, grad_clip,
                         rs.ValidTokenCount.template get<int>(), total_tokens,
                         rs.DeviceProp, stream);
        CUDA_CHECK(cudaEventRecord(rs.NormDone, stream));
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_H
