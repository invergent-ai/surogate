// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor: tensor save / persistence methods.
// Extracted from compiled_ops.cpp to reduce file size; behavior unchanged.

#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/core/backward_hooks.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/moe/moe_types.h"
#include "recipes/recipe.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::save_moe_layer_tensors(int layer_idx) {
    // Copy MoE tensors from this layer to persistent storage before stack restore.
    // This allows stack memory to be reclaimed while preserving tensors for backward.
    if (mCapturing) {
        return;
    }
    if (mConfig.NumExperts == 0) {
        return;
    }

    if (!mCurrentGraph) return;

    // Build layer prefix pattern (e.g., "blocks[5]." or "layer5.")
    // Check the first block tensor name to determine naming convention
    std::string layer_prefix;
    for (const auto& [n, _] : mCurrentGraph->tensor_name_to_id) {
        if (n.rfind("layer", 0) == 0 && n.size() > 5 && std::isdigit(n[5])) {
            layer_prefix = "layer" + std::to_string(layer_idx) + ".";
            break;
        }
        if (n.rfind("blocks[", 0) == 0) {
            layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
            break;
        }
    }
    if (layer_prefix.empty()) {
        layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";
    }

    // Iterate through compile-time tensor name map looking for MoE tensors from this layer
    for (const auto& [name, tid] : mCurrentGraph->tensor_name_to_id) {
        // Skip global MoE tensors - these are scratch space reused each layer
        // and are NOT needed for backward (backward uses mMoEExpertOffsetsGPU).
        if (name == "moe_expert_offsets" || name == "moe_gather_indices") {
            continue;
        }

        // Check if tensor belongs to this layer
        if (name.find(layer_prefix) != 0) {
            continue;
        }

        // Check if this is an MoE-related tensor that needs persistent storage
        bool is_moe_tensor =
            (name.find("moe_") != std::string::npos || name.find("ep_") != std::string::npos ||
             name.find("scatter_indices") != std::string::npos || name.find("routing_weights") != std::string::npos ||
             name.find("routing_indices") != std::string::npos || name.find("router_") != std::string::npos ||
             name.find("permuted") != std::string::npos || name.find("expert_") != std::string::npos);

        if (!is_moe_tensor) continue;
        if (tid < 0 || static_cast<std::size_t>(tid) >= mTensors.size()) continue;
        auto& tensor = mTensors[static_cast<std::size_t>(tid)];
        if (!tensor.Data) continue;

        const size_t bytes = tensor.bytes();
        if (bytes == 0) {
            continue;
        }

        // Allocate or resize persistent buffer if needed
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            // Free old buffer if exists
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            // Allocate new buffer
            void* new_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
            mMoeSavedBuffers[name] = new_buffer;
            mMoeSavedSizes[name] = bytes;
        }

        // Copy data to persistent buffer
        void* dst_buffer = mMoeSavedBuffers[name];
        CUDA_CHECK(cudaMemcpyAsync(dst_buffer, tensor.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Update all authoritative tables to point at the persistent buffer.
        // If we only patch mTensors, resolve_tensor() can still pick a stale
        // stack-backed alias from mNamedTensors first.
        Tensor saved_tensor = tensor;
        saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
        tensor = saved_tensor;
        mNamedTensors[name] = saved_tensor;
        if (mSaved) {
            auto saved_it = mSaved->find(name);
            if (saved_it != mSaved->end() || mSaveSet.find(name) != mSaveSet.end()) {
                (*mSaved)[name] = saved_tensor;
            }
        }
    }
}

void CompiledExecutor::prepare_saved_buffers_for_capture(const std::vector<std::string>& save_list,
                                                         const CompiledGraph* capture_graph) {
    // Only needed when recompute is enabled or MoE tensors require persistence.
    if (!mSaved) {
        return;
    }

    const bool recompute_enabled = mRecomputeEnabled;
    // When forward replay is active, ALL block tensors will be regenerated
    // by replay_layer_forward during backward — no persistent buffers needed.
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Forward replay: all block tensors are replayed, no persistent save needed
        if (forward_replay_active) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(tensor_name, layer_idx, field)) {
                return true;
            }
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int lid = -1;
        std::string fld;
        if (parse_block_param(name, lid, fld)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(fld))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist) -> bool {
        if (force_persist) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        // Unknown tensors default to non-persistent to avoid over-allocating
        // during graph-capture preallocation.
        return false;
    };

    const bool debug_save_buffers = (std::getenv("SUROGATE_DEBUG_SAVE_BUFFERS") != nullptr);
    auto ensure_buffer = [&](const std::string& name, size_t bytes) {
        if (bytes == 0) {
            return;
        }
        auto buf_it = mMoeSavedBuffers.find(name);
        if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
            if (debug_save_buffers) {
                std::cerr << "[SAVE-BUF] alloc name=" << name << " bytes=" << bytes
                          << " old_bytes=" << (buf_it == mMoeSavedBuffers.end() ? 0 : mMoeSavedSizes[name])
                          << std::endl;
            }
            if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            void* new_buffer = nullptr;
            cudaError_t alloc_err = cudaMalloc(&new_buffer, bytes);
            if (alloc_err != cudaSuccess) {
                std::size_t free_bytes = 0;
                std::size_t total_bytes = 0;
                (void)cudaMemGetInfo(&free_bytes, &total_bytes);
                std::ostringstream oss;
                oss << "CompiledExecutor::prepare_saved_buffers_for_capture: cudaMalloc failed for saved tensor '"
                    << name << "' (" << bytes << " bytes, " << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB)"
                    << ", free=" << static_cast<double>(free_bytes) / (1024.0 * 1024.0) << " MiB"
                    << ", total=" << static_cast<double>(total_bytes) / (1024.0 * 1024.0) << " MiB"
                    << ", error=" << cudaGetErrorString(alloc_err);
                throw std::runtime_error(oss.str());
            }
            mMoeSavedBuffers[name] = new_buffer;
            mMoeSavedSizes[name] = bytes;
        }
    };

    auto resolve_source = [&](const std::string& name) -> std::optional<Tensor> {
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                return mTensors[tid];
            }
        }
        if (name == "token_ids") {
            return mRunState.Inputs;
        }
        if (name == "position_ids") {
            return mRunState.PositionIDs;
        }

        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
                return std::nullopt;
            }
            const std::string base_field = strip_ssa_suffix(field);
            auto& acts = mRunState.simplified_acts(layer_idx);
            if (base_field == "ln1_rstd" || base_field == "ln_rstd") return acts.ln1_rstd;
            if (base_field == "ln2_rstd") return acts.ln2_rstd;
            if (base_field == "q_rstd") return acts.q_rstd;
            if (base_field == "k_rstd") return acts.k_rstd;
            if (base_field == "lse") return acts.lse;
            if (base_field == "ln1" || base_field == "ln1_flat" || base_field == "ln" || base_field == "ln_flat")
                return acts.ln1;
            if (base_field == "ln2" || base_field == "ln2_flat") return acts.ln2;
            if (base_field == "qkv" || base_field == "qkv_norm") return acts.qkv;
            if (base_field == "qkv_rope") {
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                return src;
            }
            if (base_field == "qkv_flat") {
                Tensor qkv = acts.qkv;
                return view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
            }
            if (base_field == "att" || base_field == "att_flat") return acts.att;
            if (base_field == "att_out" || base_field == "att_out_flat") return acts.att_out;
            if (base_field == "mlp_up" || base_field == "mlp_up_flat") return acts.mlp_up;
            if (base_field == "swiglu") return acts.swiglu;
            if (base_field == "swiglu_flat") {
                Tensor swiglu = acts.swiglu;
                return view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
            }
            if (base_field == "mlp_down" || base_field == "mlp_down_flat") return acts.mlp_down;
            if (base_field == "res_att" || base_field == "residual_att") return acts.residual_att;
            if (base_field == "res_ffn" || base_field == "residual_ffn" || base_field == "res_in") {
                Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
                return res;
            }
            if (base_field == "router_logits") return acts.router_logits;
            if (base_field == "router_probs") return acts.router_probs;
            if (base_field == "routing_weights") return acts.routing_weights;
            if (base_field == "routing_indices") return acts.routing_indices;
            if (base_field == "permuted_input") return acts.permuted_input;
            if (base_field == "scatter_indices") return acts.scatter_indices;
            if (base_field == "expert_gate_up") return acts.expert_gate_up;
            if (base_field == "expert_act") return acts.expert_act;
            if (base_field == "expert_down") return acts.expert_down;
            if (base_field == "moe_out" || base_field == "moe_out_flat") return acts.moe_out;
        } else if (name == "ln_final" || name == "xF") {
            return mRunState.non_block_activations().ln_final;
        } else if (name == "final_residual" || name == "residual_final") {
            return mRunState.get_final_residual();
        } else if (name == "xF_flat") {
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            return view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
        } else if (name == "ln_final_rstd") {
            return mRunState.non_block_activations().ln_final_rstd;
        } else if (name == "encoded" || name == "x0") {
            return mRunState.non_block_activations().encoded;
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            return mRunState.non_block_activations().freq_cis;
        }

        return std::nullopt;
    };

    auto infer_block_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(name, layer_idx, field)) {
            return false;
        }
        const std::string base_field = strip_ssa_suffix(field);
        const long B = (mB > 0) ? mB : mRunState.B;
        const long T = (mT > 0) ? mT : mRunState.T;
        if (B <= 0 || T <= 0) {
            return false;
        }
        const long C = mConfig.HiddenSize;
        const long D = mConfig.IntermediateSize;
        const long Hq = mConfig.NumQueryHeads;
        const long Hkv = mConfig.NumKeyValHeads;
        const long Hs = mConfig.head_size();
        const long QKV = Hs * (Hq + 2 * Hkv);
        const long AttnDim = Hq * Hs;
        const long MUp = mConfig.mlp_up_rows();

        std::vector<long> shape;
        if (base_field == "qkv_flat" || base_field == "qkv_biased") {
            shape = {B * T, QKV};
        } else if (base_field == "ln1_flat" || base_field == "ln2_flat" || base_field == "ln_flat") {
            shape = {B * T, C};
        } else if (base_field == "att_out_flat") {
            shape = {B * T, C};
        } else if (base_field == "att_flat") {
            shape = {B * T, AttnDim};
        } else if (base_field == "mlp_up_flat") {
            shape = {B * T, MUp};
        } else if (base_field == "mlp_down_flat") {
            shape = {B * T, C};
        } else if (base_field == "swiglu_flat") {
            shape = {B * T, D};
        } else if (base_field == "ln1" || base_field == "ln2" || base_field == "ln" || base_field == "res_att" ||
                   base_field == "residual_att" || base_field == "res_ffn" || base_field == "residual_ffn" ||
                   base_field == "res_in" || base_field == "att_out" || base_field == "mlp_down") {
            shape = {B, T, C};
        } else if (base_field == "ln1_rstd" || base_field == "ln2_rstd" || base_field == "ln_rstd") {
            shape = {B, T};
        } else if (base_field == "mlp_up") {
            shape = {B, T, MUp};
        } else if (base_field == "swiglu") {
            shape = {B, T, D};
        } else if (base_field == "qkv" || base_field == "qkv_rope" || base_field == "qkv_norm") {
            shape = {B, T, QKV};
        } else if (base_field == "att") {
            shape = {B, T, AttnDim};
        } else if (base_field == "q_rstd") {
            shape = {B, T, Hq};
        } else if (base_field == "k_rstd") {
            shape = {B, T, Hkv};
        } else if (base_field == "lse") {
            shape = {B, Hq, T};
        } else {
            return false;
        }

        ETensorDType dtype = ETensorDType::BF16;
        if (mConfig.NumLayers > 0) {
            dtype = mRunState.simplified_acts(0).ln1.DType;
        }
        if (base_field == "ln1_rstd" || base_field == "ln2_rstd" || base_field == "ln_rstd" || base_field == "q_rstd" ||
            base_field == "k_rstd" || base_field == "lse") {
            dtype = ETensorDType::FP32;
        }
        const std::size_t nelem = shape_nelem(shape);
        out_bytes = nelem * static_cast<std::size_t>(get_dtype_size(dtype));
        return out_bytes > 0;
    };

    auto infer_graph_ref_bytes = [&](const std::string& name, std::size_t& out_bytes) -> bool {
        const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
        if (!graph) {
            return false;
        }
        auto ref_bytes = [&](const TensorRef& ref, std::size_t& bytes_out) -> bool {
            if (ref.name != name || ref.shape.empty()) {
                return false;
            }
            const std::size_t nelem = shape_nelem(ref.shape);
            if (nelem == 0) {
                return false;
            }
            bytes_out = nelem * static_cast<std::size_t>(get_dtype_size(ref.dtype));
            return bytes_out > 0;
        };
        // Prefer producer shape (outputs), then fallback to consumer refs (inputs).
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.outputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        for (const auto& op : graph->ops) {
            for (const auto& ref : op.inputs) {
                if (ref_bytes(ref, out_bytes)) {
                    return true;
                }
            }
        }
        return false;
    };

    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const std::string base = strip_ssa_suffix(field);
        return (base == "swiglu" || base == "swiglu_flat" || base == "att" || base == "att_flat");
    };

    for (const auto& name : save_list) {
        if (mWeights.has(name)) {
            continue;
        }

        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" || name == "ln_final_rstd" ||
             name == "residual_final" || name == "final_residual" || force_lora_hook);
        const bool is_moe_tensor =
            (name.find("moe_") != std::string::npos || name.find("scatter_indices") != std::string::npos ||
             name.find("routing_weights") != std::string::npos || name.find("routing_indices") != std::string::npos ||
             name.find("router_probs") != std::string::npos || name.find("router_logits") != std::string::npos ||
             name.find("permuted_input") != std::string::npos || name.find("expert_") != std::string::npos);
        const bool prefer_live = prefer_live_tensor(name);
        const bool force_persist = is_moe_tensor && mConfig.NumExperts > 0;
        const bool need_persist = should_persist(name, prefer_live, force_persist || force_persist_name);

        // Block-level tensors may be stack-backed and persisted by
        // persist_saved_layer_tensors() during execute_forward. Pre-allocate
        // buffers for those only when they can actually require persistence.
        int layer_idx = -1;
        std::string field;
        const bool is_block_tensor = parse_block_param(name, layer_idx, field);
        if (!need_persist && (!is_block_tensor || prefer_live)) {
            continue;
        }

        auto src_opt = resolve_source(name);
        // For capture safety, preallocate block save tensors even when they
        // appear non-stack-backed at prep time. Runtime dispatch may route
        // them through stack-backed storage depending on recompute/layout.
        if (src_opt.has_value() && src_opt->Data) {
            ensure_buffer(name, src_opt->bytes());
            continue;
        }
        std::size_t inferred_bytes = 0;
        if (infer_block_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        if (infer_graph_ref_bytes(name, inferred_bytes)) {
            ensure_buffer(name, inferred_bytes);
            continue;
        }
        // Infer sizes for non-block tensors (xF_flat, ln_final, etc.)
        {
            const long B = (mB > 0) ? mB : mRunState.B;
            const long T = (mT > 0) ? mT : mRunState.T;
            const long C = mConfig.HiddenSize;
            if (B > 0 && T > 0 && C > 0) {
                const std::size_t elem_size = static_cast<std::size_t>(get_dtype_size(ETensorDType::BF16));
                std::size_t nbytes = 0;
                if (name == "xF_flat") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "ln_final" || name == "xF") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "ln_final_rstd") {
                    nbytes = static_cast<std::size_t>(B * T) * sizeof(float);
                } else if (name == "encoded" || name == "x0") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                } else if (name == "final_residual" || name == "residual_final") {
                    nbytes = static_cast<std::size_t>(B * T * C) * elem_size;
                }
                if (nbytes > 0) {
                    ensure_buffer(name, nbytes);
                }
            }
        }
    }

    // Some ops persist internal saved tensors with op-scoped names that are not
    // in forward.save (e.g., mamba_gated_rmsnorm saves "<op_id>.rstd"/".normed").
    // Pre-allocate those buffers before capture to avoid cudaMalloc in dispatch.
    const CompiledGraph* graph = capture_graph ? capture_graph : mCurrentGraph;
    if (graph) {
        for (const auto& op : graph->ops) {
            if (op.type != CompiledOpType::MambaGatedRMSNorm || op.op_id.empty()) {
                continue;
            }
            if (op.inputs.empty() || op.inputs[0].shape.empty()) {
                continue;
            }
            const auto& x_ref = op.inputs[0];
            const auto x_elems = shape_nelem(x_ref.shape);
            if (x_elems == 0) {
                continue;
            }

            const std::size_t normed_bytes = x_elems * static_cast<std::size_t>(get_dtype_size(x_ref.dtype));
            ensure_buffer(op.op_id + ".out_fallback", normed_bytes);
            ensure_buffer(op.op_id + ".normed", normed_bytes);

            const int groups = op.attrs.n_groups > 0 ? op.attrs.n_groups : 1;
            long rows = x_ref.shape[0];
            if (x_ref.shape.size() >= 3) {
                rows *= x_ref.shape[1];
            }
            if (rows > 0 && groups > 0) {
                const std::size_t rstd_bytes =
                    static_cast<std::size_t>(rows) * static_cast<std::size_t>(groups) * sizeof(float);
                ensure_buffer(op.op_id + ".rstd", rstd_bytes);
            }
        }
    }
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list, bool force_persist) {
    if (!mSaved) {
        return;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture = (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone);
    const bool capturing = mCapturing || in_capture;

    // Recompute is only active when explicitly enabled for this execution.
    // This gate is set by GraphExecutor after validating runtime options + plan.
    const bool recompute_enabled = mRecomputeEnabled;
    const bool forward_replay_active = recompute_enabled && static_cast<bool>(mRecomputeFn);
    auto contains_ci = [](std::string_view haystack, std::string_view needle) {
        std::string h(haystack);
        std::string n(needle);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return h.find(n) != std::string::npos;
    };
    const bool is_qwen3_5_model =
        contains_ci(mConfig.ModelTypeName, "qwen3_5") || contains_ci(mConfig.ModelTypeName, "qwen3.5") ||
        contains_ci(mConfig.ArchitectureName, "qwen3_5") || contains_ci(mConfig.ArchitectureName, "qwen3.5");

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (force_persist) {
            return false;
        }
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            // When forward replay is active, ALL block tensors will be regenerated
            if (forward_replay_active) {
                // Qwen3.5 replay drifts at LN1 if we drop the exact original normalized
                // activation into metadata-only mode. Keep these tensors persistent so
                // replay can consume the saved forward values instead of recomputing them.
                if (is_qwen3_5_model &&
                    (base_field == "ln1" || base_field == "ln1_flat" || base_field == "ln" || base_field == "ln_flat" ||
                     base_field == "ln1_rstd" || base_field == "ln_rstd")) {
                    return false;
                }
                return true;
            }
            return mSlotRegistry->will_recompute(base_field, lora_only_mode);
        }
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // Helper to copy tensor to persistent buffer when needed in recompute mode.
    // Returns true if tensor was copied to persistent storage, false if metadata-only save.
    auto is_shared_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->memory_hint == ActivationMemoryHint::Shared;
            }
            return std::nullopt;
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->memory_hint == ActivationMemoryHint::Shared;
        }
        return std::nullopt;
    };

    auto is_mapped_slot = [&](const std::string& name) -> std::optional<bool> {
        if (!mSlotRegistry) {
            return std::nullopt;
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(field))) {
                return entry->slot == TensorSlot::Mapped;
            }
        }
        if (auto entry = mSlotRegistry->lookup(strip_ssa_suffix(name))) {
            return entry->slot == TensorSlot::Mapped;
        }
        return std::nullopt;
    };

    auto should_persist = [&](const std::string& name, bool prefer_live, bool force_persist_name) -> bool {
        if (force_persist || force_persist_name) {
            return true;
        }
        if (!recompute_enabled || prefer_live) {
            return false;
        }
        auto mapped = is_mapped_slot(name);
        if (mapped.has_value() && mapped.value()) {
            // Slot resolves to Mapped: no persistent buffer, so saved data must be copied.
            return true;
        }
        auto shared = is_shared_slot(name);
        if (shared.has_value()) {
            return shared.value();
        }
        // Unknown tensors default to non-persistent unless explicitly forced.
        return false;
    };

    auto save_tensor_with_policy =
        [&](const std::string& name, const Tensor& src, bool prefer_live, bool force_persist_name) -> void {
        if (force_persist) {
            prefer_live = false;
            force_persist_name = true;
        }
        if (prefer_live) {
            // Save metadata only - will resolve from live buffer or recompute
            Tensor meta = src;
            meta.Data = nullptr;
            (*mSaved)[name] = meta;
            return;
        }

        const bool src_stack_backed = src.Data && mRunState.Stack.owns(src.Data);
        const bool need_persist =
            (should_persist(name, prefer_live, force_persist_name) || src_stack_backed) && src.Data != nullptr;
        if (need_persist && src.Data != nullptr) {
            const size_t bytes = src.bytes();
            auto buf_it = mMoeSavedBuffers.find(name);
            if (buf_it == mMoeSavedBuffers.end() || mMoeSavedSizes[name] < bytes) {
                if (capturing) {
                    // During CUDA graph capture we cannot allocate new buffers.
                    // Fall back to metadata-only save so backward can resolve
                    // from live/recomputed tensors instead of aborting capture.
                    Tensor meta = src;
                    meta.Data = nullptr;
                    (*mSaved)[name] = meta;
                    return;
                }
                if (buf_it != mMoeSavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoeSavedBuffers[name] = new_buffer;
                mMoeSavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoeSavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor;
            saved_tensor.DType = src.DType;
            saved_tensor.Rank = src.Rank;
            for (int i = 0; i < src.Rank; ++i) {
                saved_tensor.Sizes[i] = src.Sizes[i];
            }
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
            return;
        }

        // Non-recompute mode: just store reference
        (*mSaved)[name] = src;
    };

    auto is_lora_hook_activation = [&](const std::string& tensor_name) -> bool {
        if (!mLoRAConfig) {
            return false;
        }
        int layer_idx = -1;
        std::string field;
        if (!parse_block_param(tensor_name, layer_idx, field)) {
            return false;
        }
        const std::string base = strip_ssa_suffix(field);
        return (base == "swiglu" || base == "swiglu_flat" || base == "att" || base == "att_flat");
    };

    for (const auto& name : save_list) {
        // LoRA hook activations (att, swiglu) need persistence for backward LoRA
        // gradient computation — BUT only when forward replay is NOT active.
        // When replay regenerates these tensors, no persistent buffer is needed.
        const bool force_lora_hook = !forward_replay_active && is_lora_hook_activation(name);
        const bool force_persist_name =
            (name == "xF_flat" || name == "xF" || name == "ln_final" || name == "ln_final_rstd" ||
             name == "residual_final" || name == "final_residual" || force_lora_hook);
        const bool prefer_live = prefer_live_tensor(name);
        // Skip tensors already saved directly by dispatch (e.g. mamba_gated_rmsnorm saves rstd/normed),
        // unless we are forcing a persistent copy to replace metadata-only entries.
        auto saved_it = mSaved->find(name);
        if (saved_it != mSaved->end()) {
            // Layer-end replay bookkeeping pre-seeds metadata-only entries for block tensors.
            // If this tensor should not remain metadata-only, refresh it now with live data.
            const bool needs_refresh =
                (saved_it->second.Data == nullptr) && (force_persist || force_persist_name || !prefer_live);
            if (!needs_refresh) {
                continue;
            }
            mSaved->erase(saved_it);
        }

        // First check intermediate tensors via flat vector (fast path) or mirror map
        Tensor* found_tensor = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                found_tensor = &mTensors[tid];
            }
        }
        if (found_tensor) {
            const bool is_moe_tensor =
                (name.find("moe_") != std::string::npos || name.find("ep_") != std::string::npos ||
                 name.find("scatter_indices") != std::string::npos ||
                 name.find("routing_weights") != std::string::npos ||
                 name.find("routing_indices") != std::string::npos || name.find("router_probs") != std::string::npos ||
                 name.find("router_logits") != std::string::npos || name.find("permuted_input") != std::string::npos ||
                 name.find("expert_") != std::string::npos);
            const bool force_persist = is_moe_tensor && mConfig.NumExperts > 0;
            save_tensor_with_policy(name, *found_tensor, prefer_live, force_persist || force_persist_name);
            continue;
        }

        // Check special tensors
        if (name == "token_ids") {
            save_tensor_with_policy(name, mRunState.Inputs, prefer_live_tensor(name), force_persist_name);
            continue;
        }
        if (name == "position_ids") {
            save_tensor_with_policy(name, mRunState.PositionIDs, prefer_live_tensor(name), force_persist_name);
            continue;
        }

        // Try to look up as a pre-allocated activation by creating a TensorRef
        // This handles tensors like "blocks[0].ln1_rstd" that map to slots
        TensorRef ref;
        ref.name = name;
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            ref.layer_idx = layer_idx;
            // Map common saved fields
            const bool prefer_live = prefer_live_tensor(name);
            if (field == "ln1_rstd" || field == "ln_rstd") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).ln1_rstd,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "ln2_rstd") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).ln2_rstd,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "q_rstd") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).q_rstd,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "k_rstd") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).k_rstd,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "lse") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).lse,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "ln1" || field == "ln1_flat" || field == "ln" || field == "ln_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).ln1,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "ln2" || field == "ln2_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).ln2,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "qkv" || field == "qkv_norm") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).qkv,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "qkv_rope") {
                // qkv_rope has RoPE applied - save it if available, otherwise fall back to qkv
                auto& acts = mRunState.simplified_acts(layer_idx);
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                save_tensor_with_policy(name, src, prefer_live, force_persist_name);
            } else if (field == "qkv_flat") {
                // Save the flattened version for matmul backward shape resolution
                Tensor qkv = mRunState.simplified_acts(layer_idx).qkv;
                Tensor flat = view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
            } else if (field == "att" || field == "att_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).att,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "att_out" || field == "att_out_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).att_out,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).mlp_up,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "swiglu") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).swiglu,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "swiglu_flat") {
                Tensor swiglu = mRunState.simplified_acts(layer_idx).swiglu;
                Tensor flat = view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live, force_persist_name);
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).mlp_down,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "res_att" || field == "residual_att") {
                save_tensor_with_policy(name,
                                        mRunState.simplified_acts(layer_idx).residual_att,
                                        prefer_live,
                                        force_persist_name);
            } else if (field == "res_ffn" || field == "residual_ffn" || field == "res_in") {
                // res_ffn is the residual stream after FFN block (residual_att + mlp_down)
                Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
                save_tensor_with_policy(name, res, prefer_live, force_persist_name);
            } else if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
            } else {
                throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
            }
        } else if (name == "ln_final" || name == "xF") {
            save_tensor_with_policy(name,
                                    mRunState.non_block_activations().ln_final,
                                    prefer_live_tensor(name),
                                    force_persist_name);
        } else if (name == "final_residual" || name == "residual_final") {
            save_tensor_with_policy(name, mRunState.get_final_residual(), prefer_live_tensor(name), force_persist_name);
        } else if (name == "xF_flat") {
            // Save the flattened version for matmul backward
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            Tensor flat = view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
            save_tensor_with_policy(name, flat, prefer_live_tensor(name), force_persist_name);
        } else if (name == "ln_final_rstd") {
            save_tensor_with_policy(name,
                                    mRunState.non_block_activations().ln_final_rstd,
                                    prefer_live_tensor(name),
                                    force_persist_name);
        } else if (name == "encoded" || name == "x0") {
            save_tensor_with_policy(name,
                                    mRunState.non_block_activations().encoded,
                                    prefer_live_tensor(name),
                                    force_persist_name);
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            save_tensor_with_policy(name,
                                    mRunState.non_block_activations().freq_cis,
                                    prefer_live_tensor(name),
                                    force_persist_name);
        } else if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
        } else {
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }
    }

    // For MoE models, copy expert_offsets data to persistent storage for backward pass
    // The original tensor is stack-allocated and will be freed before backward runs
    if (mConfig.NumExperts > 0) {
        Tensor* moe_offsets = nullptr;
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id("moe_expert_offsets");
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size() && mTensors[tid].Data) {
                moe_offsets = &mTensors[tid];
            }
        }
        if (moe_offsets) {
            const Tensor& src = *moe_offsets;
            const int num_elements = static_cast<int>(src.nelem());
            mMoEExpertOffsetsData.resize(num_elements);
            CUDA_CHECK(
                cudaMemcpy(mMoEExpertOffsetsData.data(), src.Data, num_elements * sizeof(int), cudaMemcpyDeviceToHost));
            // Store metadata for reconstruction in backward
            mMoEExpertOffsets = src;           // Copy the tensor metadata (shape, dtype, etc.)
            mMoEExpertOffsets.Data = nullptr;  // Data will be restored from CPU storage
        }
    }
}

Tensor* CompiledExecutor::try_resolve_saved_live(const std::string& name, const Tensor& saved) {
    std::vector<long> shape;
    shape.reserve(static_cast<std::size_t>(saved.Rank));
    for (int i = 0; i < saved.Rank; ++i) {
        shape.push_back(saved.Sizes[i]);
    }

    auto map_view = [&](Tensor& base) -> Tensor* {
        if (!base.Data) {
            return nullptr;
        }
        if (shape.empty() || tensor_shape_matches(base, shape)) {
            return &base;
        }
        if (shape_nelem(shape) != base.nelem()) {
            return nullptr;
        }
        Tensor view = view_tensor(base, shape);
        if (mCurrentGraph) {
            int tid = mCurrentGraph->find_tensor_id(name);
            if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                mTensors[tid] = view;
                return &mTensors[tid];
            }
        }
        return &base;  // Can't cache view without tensor_id, return base
    };

    if (name == "token_ids") {
        return map_view(mRunState.Inputs);
    }
    if (name == "position_ids") {
        return map_view(mRunState.PositionIDs);
    }
    if (name == "encoded" || name == "x0") {
        return map_view(mRunState.non_block_activations().encoded);
    }
    if (name == "ln_final" || name == "xF" || name == "xF_flat") {
        return map_view(mRunState.non_block_activations().ln_final);
    }
    if (name == "ln_final_rstd") {
        return map_view(mRunState.non_block_activations().ln_final_rstd);
    }
    if (name == "final_residual" || name == "residual_final") {
        return map_view(mRunState.get_final_residual());
    }
    if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
        return map_view(mRunState.non_block_activations().freq_cis);
    }

    int layer_idx = -1;
    std::string field;
    if (parse_block_param(name, layer_idx, field)) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return nullptr;
        }
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (field == "ln1" || field == "ln1_flat") return map_view(acts.ln1);
        if (field == "ln1_rstd") return map_view(acts.ln1_rstd);
        if (field == "ln2" || field == "ln2_flat") return map_view(acts.ln2);
        if (field == "ln2_rstd") return map_view(acts.ln2_rstd);
        if (field == "q_rstd") return map_view(acts.q_rstd);
        if (field == "k_rstd") return map_view(acts.k_rstd);
        if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased" || field == "qkv_norm")
            return map_view(acts.qkv);
        if (field == "qkv_rope") {
            Tensor* base = acts.qkv_rope.Data ? &acts.qkv_rope : &acts.qkv;
            return map_view(*base);
        }
        if (field == "lse") return map_view(acts.lse);
        if (field == "att" || field == "att_flat") return map_view(acts.att);
        if (field == "att_out" || field == "att_out_flat") return map_view(acts.att_out);
        if (field == "res_att" || field == "residual_att") return map_view(acts.residual_att);
        if (field == "mlp_up" || field == "mlp_up_flat") return map_view(acts.mlp_up);
        if (field == "swiglu" || field == "swiglu_flat") return map_view(acts.swiglu);
        if (field == "mlp_down" || field == "mlp_down_flat") return map_view(acts.mlp_down);
        if (field == "router_logits") return map_view(acts.router_logits);
        if (field == "router_probs") return map_view(acts.router_probs);
        if (field == "routing_weights") return map_view(acts.routing_weights);
        if (field == "routing_indices") return map_view(acts.routing_indices);
        if (field == "permuted_input") return map_view(acts.permuted_input);
        if (field == "scatter_indices") return map_view(acts.scatter_indices);
        if (field == "expert_gate_up") return map_view(acts.expert_gate_up);
        if (field == "expert_act") return map_view(acts.expert_act);
        if (field == "expert_down") return map_view(acts.expert_down);
        if (field == "moe_out" || field == "moe_out_flat") return map_view(acts.moe_out);
        if (field == "res_ffn" || field == "residual_ffn" || field == "res_in") {
            Tensor& res = mRunState.get_residual(layer_idx, mRunState.MainStream);
            return map_view(res);
        }
        if (field == "rope_freqs" || field == "freq_cis") {
            return map_view(mRunState.non_block_activations().freq_cis);
        }
    }

    return nullptr;
}

Tensor& CompiledExecutor::resolve_tensor(const TensorRef& ref) {
    auto& rs = mRunState;
    const int tid = ref.tensor_id;
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    auto log_tensor = [&](const Tensor& t, const char* tag) {
        if (debug_name) {
            fprintf(stderr,
                    "[DEBUG_DTYPE_RUNTIME] resolve_tensor %s %s dtype=%s\n",
                    ref.name.c_str(),
                    tag,
                    dtype_to_str(t.DType));
        }
    };

    if (!ref.name.empty()) {
        auto name_it = mNamedTensors.find(ref.name);
        if (name_it != mNamedTensors.end() && name_it->second.Data) {
            log_tensor(name_it->second, "named");
            return name_it->second;
        }
    }

    // Only check gradient resolution for tensors flagged as gradients at compile time.
    // This avoids calling base_param_from_grad() (string substr + find) for the ~80%
    // of tensors that are activations/weights, not gradients.
    if (ref.is_gradient && !ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Need to create a view from the base tensor
        if (ref.name.find("att_flat") != std::string::npos && ref.layer_idx == 4) {
            fprintf(stderr,
                    "[RESOLVE] %s slot=%d layer=%d shape=[%ld,%ld] tid=%d\n",
                    ref.name.c_str(),
                    (int)ref.slot,
                    ref.layer_idx,
                    ref.shape.size() > 0 ? ref.shape[0] : -1,
                    ref.shape.size() > 1 ? ref.shape[1] : -1,
                    tid);
        }
        Tensor* base = nullptr;
        switch (ref.slot) {
            case TensorSlot::TokenIDs: base = &rs.Inputs; break;
            case TensorSlot::PositionIDs: base = &rs.PositionIDs; break;
            case TensorSlot::Targets: base = &rs.Targets; break;
            case TensorSlot::Losses: base = &rs.Losses; break;
            case TensorSlot::DLoss: base = &rs.scratch().cross_entropy_dloss; break;
            case TensorSlot::BlockDLN1: base = &rs.simplified_grads(ref.layer_idx).d_ln1; break;
            case TensorSlot::BlockDQKV: base = &rs.simplified_grads(ref.layer_idx).d_qkv; break;
            case TensorSlot::BlockDAtt: base = &rs.simplified_grads(ref.layer_idx).d_att; break;
            case TensorSlot::BlockDSwiGLU: base = &rs.simplified_grads(ref.layer_idx).d_swiglu; break;
            case TensorSlot::BlockDMLPUp: base = &rs.simplified_grads(ref.layer_idx).d_mlp_up; break;
            case TensorSlot::BlockDMLPDown: base = &rs.simplified_grads(ref.layer_idx).d_mlp_down; break;
            case TensorSlot::BlockDLN2: base = &rs.simplified_grads(ref.layer_idx).d_ln2; break;
            case TensorSlot::BlockDResAtt: base = &rs.simplified_grads(ref.layer_idx).d_res_att; break;
            case TensorSlot::BlockDAttOut: base = &rs.simplified_grads(ref.layer_idx).d_att_out; break;
            case TensorSlot::BlockDResFFN: base = &rs.simplified_grads(ref.layer_idx).d_res_ffn; break;
            case TensorSlot::BlockLN1: base = &rs.simplified_acts(ref.layer_idx).ln1; break;
            case TensorSlot::BlockLN2: base = &rs.simplified_acts(ref.layer_idx).ln2; break;
            case TensorSlot::BlockQKV: base = &rs.simplified_acts(ref.layer_idx).qkv; break;
            case TensorSlot::BlockAtt: base = &rs.simplified_acts(ref.layer_idx).att; break;
            case TensorSlot::BlockAttOut: base = &rs.simplified_acts(ref.layer_idx).att_out; break;
            case TensorSlot::BlockMLPUp: base = &rs.simplified_acts(ref.layer_idx).mlp_up; break;
            case TensorSlot::BlockSwiGLU: base = &rs.simplified_acts(ref.layer_idx).swiglu; break;
            case TensorSlot::BlockMLPDown: base = &rs.simplified_acts(ref.layer_idx).mlp_down; break;
            default: break;
        }
        if (base && base->Data) {
            Tensor view = view_tensor(*base, ref.shape);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return *base;
        }
    }

    // Check flat tensor vector first for cached/aliased tensors (e.g., view_backward aliases).
    // This is critical because view_backward stores aliases, and subsequent ops
    // (like rmsnorm_backward) must use that aliased tensor, not the pre-allocated simplified_grads buffer.
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        log_tensor(mTensors[static_cast<std::size_t>(tid)], "cached");
        return mTensors[static_cast<std::size_t>(tid)];
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs: return rs.Inputs;
        case TensorSlot::PositionIDs: return rs.PositionIDs;
        case TensorSlot::Targets: return rs.Targets;
        case TensorSlot::Losses: return rs.Losses;
        case TensorSlot::DLoss: return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded: return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal: return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD: return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual: return rs.get_final_residual();
        case TensorSlot::FreqCis: return rs.non_block_activations().freq_cis;
        case TensorSlot::BlockLN1: return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD: return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2: return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD: return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD: return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD: return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV: return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE: return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt: return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut: return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt: return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp: return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU: return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown: return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockResidualFFN: return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockDLN1: return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV: return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt: return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU: return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp: return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown: return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDLN2: return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt: return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDAttOut: return rs.simplified_grads(ref.layer_idx).d_att_out;
        case TensorSlot::BlockDResFFN: return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter: return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    // If the saved tensor has actual data, use it directly.
                    // Only resolve from live buffers when Data == nullptr (metadata-only mode).
                    if (it->second.Data != nullptr) {
                        return it->second;
                    }
                    // Metadata-only: resolve from current live tensors first.
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    // As a last resort, reuse cached tensor-id only if it points to
                    // persistent (non-stack) memory; stack pointers can become stale
                    // across layer checkpoint restores under recompute/capture.
                    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data &&
                        !mRunState.Stack.owns(mTensors[static_cast<std::size_t>(tid)].Data)) {
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            // Already checked mTensors[tid] above; if we get here, tensor was not found
            if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
                return mTensors[static_cast<std::size_t>(tid)];
            }
            // Materialize effective tensors on demand (e.g. *_weight_eff = *_weight + 1).
            if (!ref.name.empty() && ref.name.size() > 4 && ref.name.compare(ref.name.size() - 4, 4, "_eff") == 0) {
                const std::string base_name = ref.name.substr(0, ref.name.size() - 4);
                Tensor* base_ptr = nullptr;
                if (mWeights.has(base_name)) {
                    base_ptr = &mWeights.get(base_name);
                } else if (mSaved) {
                    auto it = mSaved->find(base_name);
                    if (it != mSaved->end() && it->second.Data) {
                        base_ptr = &it->second;
                    }
                }
                if (!base_ptr && mCurrentGraph) {
                    const int base_tid = mCurrentGraph->find_tensor_id(base_name);
                    if (base_tid >= 0 && static_cast<std::size_t>(base_tid) < mTensors.size() &&
                        mTensors[static_cast<std::size_t>(base_tid)].Data) {
                        base_ptr = &mTensors[static_cast<std::size_t>(base_tid)];
                    }
                }
                if (base_ptr && base_ptr->Data) {
                    std::vector<long> shape(base_ptr->Sizes.begin(), base_ptr->Sizes.begin() + base_ptr->Rank);
                    Tensor ones = mRunState.temp_alloc(base_ptr->DType, shape, "ones");
                    Tensor eff = mRunState.temp_alloc(base_ptr->DType, shape, "eff");
                    mTemps.push_back(ones);
                    mTemps.push_back(eff);
                    fill_constant(ones, 1.0f, static_cast<std::size_t>(ones.nelem()), mRunState.MainStream);
                    vector_add_sr(eff,
                                  *base_ptr,
                                  ones,
                                  1.0f,
                                  static_cast<long>(base_ptr->nelem()),
                                  0,
                                  mRunState.MainStream);
                    if (tid >= 0 && static_cast<std::size_t>(tid) < mTensors.size()) {
                        mTensors[static_cast<std::size_t>(tid)] = eff;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    throw std::runtime_error("CompiledExecutor: mapped effective tensor requires valid tensor_id: " +
                                             ref.name);
                }
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary: throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    throw std::runtime_error("CompiledExecutor: invalid tensor slot");
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    const int tid = ref.tensor_id;
    const bool debug_dtype = []() {
        const char* env = std::getenv("SUROGATE_DEBUG_DTYPE_RUNTIME");
        return env && std::string(env) == "1";
    }();
    const bool debug_name = debug_dtype && (ref.name.find("d_xF") != std::string::npos);
    if (debug_name) {
        fprintf(stderr,
                "[DEBUG_DTYPE_RUNTIME] ensure_output_tensor enter %s slot=%d ref=%s\n",
                ref.name.c_str(),
                static_cast<int>(ref.slot),
                dtype_to_str(ref.dtype));
    }

    // Fast path: pre-allocated block slots with existing data bypass string parsing.
    // This covers most activation and gradient outputs during forward/backward.
    // Only Mapped/Temporary/Parameter/Saved slots need the string-heavy resolution below.
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary && ref.slot != TensorSlot::Parameter &&
        ref.slot != TensorSlot::Saved) {
        Tensor& t = resolve_tensor(ref);
        if (t.Data) {
            if (!ref.shape.empty()) {
                Tensor view = view_tensor(t, ref.shape);
                if (tid >= 0) {
                    mTensors[static_cast<std::size_t>(tid)] = view;
                    return mTensors[static_cast<std::size_t>(tid)];
                }
                return t;
            }
            return t;
        }
        // No data — fall through to slow path for alias resolution or temp allocation
    }

    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    Tensor resolved = *grad;
                    if (!ref.shape.empty() && shape_nelem(ref.shape) == static_cast<std::size_t>(grad->nelem())) {
                        resolved = view_tensor(*grad, ref.shape);
                    }
                    if (tid >= 0) {
                        mTensors[static_cast<std::size_t>(tid)] = resolved;
                        return mTensors[static_cast<std::size_t>(tid)];
                    }
                    return *grad;
                }
            }
        }
    }

    // DSL-driven aliasing: allow gradients to reuse existing activation buffers.
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout() && !ref.name.empty()) {
        std::string base_name = strip_ssa_suffix(ref.name);
        const bool is_grad_name = starts_with(base_name, "d_");
        std::string parse_name = is_grad_name ? base_name.substr(2) : base_name;
        int layer_idx = -1;
        std::string field;
        std::string lookup_name = base_name;
        if (parse_block_param(parse_name, layer_idx, field)) {
            const std::string base_field = strip_ssa_suffix(field);
            lookup_name = is_grad_name ? ("d_" + base_field) : base_field;
        }
        if (auto slot_entry = mSlotRegistry->lookup(lookup_name)) {
            if (!slot_entry->alias_of.empty()) {
                const std::string alias_field = slot_entry->alias_of;
                std::string alias_name = alias_field;
                if (layer_idx >= 0) {
                    // Reconstruct using the same naming convention as the source tensor
                    if (parse_name.rfind("layer", 0) == 0) {
                        alias_name = "layer" + std::to_string(layer_idx) + "." + alias_field;
                    } else {
                        alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                    }
                }
                if (auto alias_entry = mSlotRegistry->lookup(alias_field)) {
                    TensorRef alias_ref;
                    alias_ref.name = alias_name;
                    alias_ref.layer_idx = layer_idx;
                    alias_ref.slot = alias_entry->slot;
                    alias_ref.shape = ref.shape;
                    alias_ref.dtype = ref.dtype;
                    // Resolve alias tensor_id from current graph
                    if (mCurrentGraph) {
                        alias_ref.tensor_id = mCurrentGraph->find_tensor_id(alias_name);
                    }
                    if (mSaveSet.find(alias_name) != mSaveSet.end()) {
                        alias_ref.slot = TensorSlot::Saved;
                    }
                    try {
                        Tensor& base = resolve_tensor(alias_ref);
                        Tensor view = ref.shape.empty() ? base : view_tensor(base, ref.shape);
                        if (tid >= 0) {
                            mTensors[static_cast<std::size_t>(tid)] = view;
                            return mTensors[static_cast<std::size_t>(tid)];
                        }
                        return base;
                    } catch (const std::exception&) {
                        // Fall through to normal allocation if alias resolution fails.
                    }
                }
            }
        }
    }

    // For pre-allocated slots, just return the tensor
    if (ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Temporary) {
        Tensor& t = resolve_tensor(ref);
        if (!t.Data) {
            mRunState.temp_acquire(t);
            mTemps.push_back(t);
        }
        if (!ref.shape.empty()) {
            Tensor view = view_tensor(t, ref.shape);
            if (tid >= 0) {
                mTensors[static_cast<std::size_t>(tid)] = view;
                return mTensors[static_cast<std::size_t>(tid)];
            }
            return t;  // tid < 0 should not happen for pre-allocated slots
        }
        return t;
    }

    // For mapped/temporary tensors, check flat vector first
    if (tid >= 0 && mTensors[static_cast<std::size_t>(tid)].Data) {
        if (debug_name) {
            const auto& t = mTensors[static_cast<std::size_t>(tid)];
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] reuse tid=%d dtype=%s\n", tid, dtype_to_str(t.DType));
        }
        return mTensors[static_cast<std::size_t>(tid)];
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape, "t");

    // Zero gradient tensors to prevent stale values from accumulating.
    if (ref.is_gradient) {
        fill_zero(t, mRunState.MainStream);
    }

    mTemps.push_back(t);
    if (tid >= 0) {
        mTensors[static_cast<std::size_t>(tid)] = t;
        if (debug_name) {
            fprintf(stderr, "[DEBUG_DTYPE_RUNTIME] alloc tid=%d dtype=%s\n", tid, dtype_to_str(t.DType));
        }
        return mTensors[static_cast<std::size_t>(tid)];
    }
    throw std::runtime_error("CompiledExecutor: ensure_output_tensor requires valid tensor_id for: " + ref.name);
}

}  // namespace dsl
