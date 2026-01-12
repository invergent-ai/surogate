// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Weight Fusion Helpers - Static methods for weight transformation
//
// This file provides static helper methods for fusing/splitting weight tensors
// during HuggingFace format import/export. These utilities handle models that
// store weights in split format (separate Q, K, V projections) vs fused format
// (single QKV tensor).
//
// Key transformations:
// 1. QKV fusion: Combine separate Q, K, V projections into fused tensor
// 2. Gate+Up fusion: Combine separate gate and up projections for SwiGLU
// 3. Expert batching: Combine per-expert weights into batched tensor
//
// Note: The actual fusion during weight loading is handled by load_intersect()
// in weight_manager_io.h using ranges from weight_mapping.h. These helpers are
// for explicit fusion/split operations when needed.

#ifndef SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_FUSION_H
#define SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_FUSION_H

#include <cstring>
#include <stdexcept>

#include "config/pretrained_config.h"
#include "utilities/tensor.h"

namespace modules {

// ============================================================================
// QKV Fusion Helpers
// ============================================================================

/**
 * @brief Static helpers for QKV weight fusion/split
 *
 * Internal fused layout: [Q, K, V] = (qkv_rows, hidden_size)
 * where qkv_rows = q_rows + 2 * kv_rows
 *       q_rows = num_query_heads * head_size
 *       kv_rows = num_kv_heads * head_size
 *
 * HuggingFace split layout:
 *   q_proj.weight: (q_rows, hidden_size)
 *   k_proj.weight: (kv_rows, hidden_size)
 *   v_proj.weight: (kv_rows, hidden_size)
 */
struct QKVFusion {
    /**
     * @brief Calculate Q, K, V row counts from config
     */
    static void get_qkv_dims(const PretrainedConfig& cfg, int& q_rows, int& kv_rows, int& c) {
        const int hs = cfg.head_size();
        q_rows = hs * cfg.NumQueryHeads;
        kv_rows = hs * cfg.NumKeyValHeads;
        c = cfg.HiddenSize;
    }

    /**
     * @brief Get offset and size for Q portion in fused tensor (elements)
     */
    static void q_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        int q_rows, kv_rows, c;
        get_qkv_dims(cfg, q_rows, kv_rows, c);
        begin = 0;
        end = static_cast<std::ptrdiff_t>(q_rows) * c;
    }

    /**
     * @brief Get offset and size for K portion in fused tensor (elements)
     */
    static void k_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        int q_rows, kv_rows, c;
        get_qkv_dims(cfg, q_rows, kv_rows, c);
        begin = static_cast<std::ptrdiff_t>(q_rows) * c;
        end = begin + static_cast<std::ptrdiff_t>(kv_rows) * c;
    }

    /**
     * @brief Get offset and size for V portion in fused tensor (elements)
     */
    static void v_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        int q_rows, kv_rows, c;
        get_qkv_dims(cfg, q_rows, kv_rows, c);
        begin = static_cast<std::ptrdiff_t>(q_rows + kv_rows) * c;
        end = begin + static_cast<std::ptrdiff_t>(kv_rows) * c;
    }

    /**
     * @brief Fuse separate Q, K, V tensors into single QKV tensor
     *
     * @param dst Destination fused tensor (qkv_rows, hidden_size)
     * @param q Q projection weight (q_rows, hidden_size)
     * @param k K projection weight (kv_rows, hidden_size)
     * @param v V projection weight (kv_rows, hidden_size)
     */
    static void fuse(Tensor& dst, const Tensor& q, const Tensor& k, const Tensor& v) {
        if (dst.Data == nullptr) {
            throw std::runtime_error("QKVFusion::fuse: destination tensor not allocated");
        }

        const std::size_t elem_size = get_dtype_size(dst.DType);
        const long q_elems = q.nelem();
        const long kv_elems = k.nelem();

        // Copy Q
        std::memcpy(dst.Data, q.Data, q_elems * elem_size);

        // Copy K
        std::byte* k_dst = dst.Data + q_elems * elem_size;
        std::memcpy(k_dst, k.Data, kv_elems * elem_size);

        // Copy V
        std::byte* v_dst = k_dst + kv_elems * elem_size;
        std::memcpy(v_dst, v.Data, kv_elems * elem_size);
    }

    /**
     * @brief Split fused QKV tensor into separate Q, K, V tensors
     *
     * @param qkv Source fused tensor
     * @param q Output Q tensor (must be pre-allocated)
     * @param k Output K tensor (must be pre-allocated)
     * @param v Output V tensor (must be pre-allocated)
     */
    static void split(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
        const std::size_t elem_size = get_dtype_size(qkv.DType);
        const long q_elems = q.nelem();
        const long kv_elems = k.nelem();

        // Copy Q
        std::memcpy(q.Data, qkv.Data, q_elems * elem_size);

        // Copy K
        const std::byte* k_src = qkv.Data + q_elems * elem_size;
        std::memcpy(k.Data, k_src, kv_elems * elem_size);

        // Copy V
        const std::byte* v_src = k_src + kv_elems * elem_size;
        std::memcpy(v.Data, v_src, kv_elems * elem_size);
    }
};

// ============================================================================
// QKV Bias Fusion Helpers
// ============================================================================

/**
 * @brief Static helpers for QKV bias fusion/split
 *
 * Same layout pattern as QKV weights but 1D: (qkv_rows,)
 */
struct QKVBiasFusion {
    static void q_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        const int hs = cfg.head_size();
        begin = 0;
        end = hs * cfg.NumQueryHeads;
    }

    static void k_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        const int hs = cfg.head_size();
        const int q_rows = hs * cfg.NumQueryHeads;
        const int kv_rows = hs * cfg.NumKeyValHeads;
        begin = q_rows;
        end = q_rows + kv_rows;
    }

    static void v_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        const int hs = cfg.head_size();
        const int q_rows = hs * cfg.NumQueryHeads;
        const int kv_rows = hs * cfg.NumKeyValHeads;
        begin = q_rows + kv_rows;
        end = q_rows + 2 * kv_rows;
    }

    static void fuse(Tensor& dst, const Tensor& q, const Tensor& k, const Tensor& v) {
        const std::size_t elem_size = get_dtype_size(dst.DType);
        const long q_elems = q.nelem();
        const long kv_elems = k.nelem();

        std::memcpy(dst.Data, q.Data, q_elems * elem_size);

        std::byte* k_dst = dst.Data + q_elems * elem_size;
        std::memcpy(k_dst, k.Data, kv_elems * elem_size);

        std::byte* v_dst = k_dst + kv_elems * elem_size;
        std::memcpy(v_dst, v.Data, kv_elems * elem_size);
    }

    static void split(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
        const std::size_t elem_size = get_dtype_size(qkv.DType);
        const long q_elems = q.nelem();
        const long kv_elems = k.nelem();

        std::memcpy(q.Data, qkv.Data, q_elems * elem_size);

        const std::byte* k_src = qkv.Data + q_elems * elem_size;
        std::memcpy(k.Data, k_src, kv_elems * elem_size);

        const std::byte* v_src = k_src + kv_elems * elem_size;
        std::memcpy(v.Data, v_src, kv_elems * elem_size);
    }
};

// ============================================================================
// Gate+Up Fusion Helpers
// ============================================================================

/**
 * @brief Static helpers for gate+up MLP weight fusion/split
 *
 * Internal fused layout: [up, gate] = (2 * intermediate_size, hidden_size)
 *
 * HuggingFace split layout:
 *   up_proj.weight: (intermediate_size, hidden_size)
 *   gate_proj.weight: (intermediate_size, hidden_size)
 *
 * Note: We store up first, then gate (matching the order in SwiGLU computation).
 */
struct GateUpFusion {
    /**
     * @brief Get offset and size for up portion in fused tensor (elements)
     */
    static void up_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        begin = 0;
        end = static_cast<std::ptrdiff_t>(cfg.IntermediateSize) * cfg.HiddenSize;
    }

    /**
     * @brief Get offset and size for gate portion in fused tensor (elements)
     */
    static void gate_range(const PretrainedConfig& cfg, std::ptrdiff_t& begin, std::ptrdiff_t& end) {
        begin = static_cast<std::ptrdiff_t>(cfg.IntermediateSize) * cfg.HiddenSize;
        end = 2 * begin;
    }

    /**
     * @brief Fuse separate gate and up tensors into single tensor
     */
    static void fuse(Tensor& dst, const Tensor& up, const Tensor& gate) {
        const std::size_t elem_size = get_dtype_size(dst.DType);
        const long d_elems = up.nelem();

        // Copy up (first half)
        std::memcpy(dst.Data, up.Data, d_elems * elem_size);

        // Copy gate (second half)
        std::byte* gate_dst = dst.Data + d_elems * elem_size;
        std::memcpy(gate_dst, gate.Data, d_elems * elem_size);
    }

    /**
     * @brief Split fused tensor into separate up and gate tensors
     */
    static void split(const Tensor& fused, Tensor& up, Tensor& gate) {
        const std::size_t elem_size = get_dtype_size(fused.DType);
        const long d_elems = up.nelem();

        // Copy up
        std::memcpy(up.Data, fused.Data, d_elems * elem_size);

        // Copy gate
        const std::byte* gate_src = fused.Data + d_elems * elem_size;
        std::memcpy(gate.Data, gate_src, d_elems * elem_size);
    }
};

// ============================================================================
// Expert Weight Batching Helpers
// ============================================================================

/**
 * @brief Static helpers for batching per-expert weights
 *
 * Some models store expert weights separately:
 *   experts.0.gate_proj.weight, experts.0.up_proj.weight, experts.0.down_proj.weight
 *   experts.1.gate_proj.weight, ...
 *
 * We batch these into efficient grouped GEMM layout:
 *   gate_up_proj: (num_experts, 2 * intermediate_size, hidden_size)
 *   down_proj: (num_experts, hidden_size, intermediate_size)
 */
struct ExpertBatching {
    /**
     * @brief Get offset for a specific expert's weights in batched tensor
     */
    static std::ptrdiff_t expert_offset(int expert_idx, int expert_size) {
        return static_cast<std::ptrdiff_t>(expert_idx) * expert_size;
    }

    /**
     * @brief Copy a single expert's gate+up weights into batched tensor
     *
     * @param dst_data Batched destination (num_experts, 2*D, C)
     * @param expert_idx Which expert this is
     * @param gate_data Gate projection weight (D, C)
     * @param up_data Up projection weight (D, C)
     * @param intermediate_size D dimension
     * @param hidden_size C dimension
     * @param dtype Data type
     */
    static void copy_expert_gate_up_into_batched(
        void* dst_data,
        int expert_idx,
        const void* gate_data, std::size_t gate_bytes,
        const void* up_data, std::size_t up_bytes,
        int intermediate_size, int hidden_size, ETensorDType dtype
    ) {
        const int d = intermediate_size;
        const int c = hidden_size;
        const std::size_t elem_size = get_dtype_size(dtype);
        const std::size_t expert_gate_up_bytes = 2 * d * c * elem_size;

        // Calculate offset for this expert
        char* expert_base = static_cast<char*>(dst_data) + expert_idx * expert_gate_up_bytes;

        // Copy up first (at offset 0), then gate
        std::memcpy(expert_base, up_data, up_bytes);
        std::memcpy(expert_base + up_bytes, gate_data, gate_bytes);
    }

    /**
     * @brief Copy a single expert's down projection into batched tensor
     */
    static void copy_expert_down_into_batched(
        void* dst_data,
        int expert_idx,
        const void* down_data, std::size_t down_bytes,
        int intermediate_size, int hidden_size, ETensorDType dtype
    ) {
        const int d = intermediate_size;
        const int c = hidden_size;
        const std::size_t elem_size = get_dtype_size(dtype);
        const std::size_t expert_down_bytes = c * d * elem_size;

        char* expert_base = static_cast<char*>(dst_data) + expert_idx * expert_down_bytes;
        std::memcpy(expert_base, down_data, down_bytes);
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_WEIGHTS_WEIGHT_FUSION_H
