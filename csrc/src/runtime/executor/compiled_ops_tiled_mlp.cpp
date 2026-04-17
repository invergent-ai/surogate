// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// CompiledExecutor: tiled MLP forward/backward for long-context mode.
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
#include "runtime/lora/lora_slice_dispatch.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "runtime/moe/moe_types.h"
#include "recipes/recipe.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

// ============================================================================
// Tiled MLP execution for long-context mode
// ============================================================================
// Executes an MLP tile group (view→matmul→view→swiglu→view→matmul→view) in
// chunks along the flattened B*T dimension. Reduces peak MLP activation memory
// from O(B*T * intermediate) to O(chunk_size * intermediate).
//
// Operates in flat 2D space to avoid shape mismatches with compiled view ops:
//   chunk_input[chunk, C] → matmul → up[chunk, MUp] → swiglu → act[chunk, M]
//   → matmul → down[chunk, C] → write into full_output
void CompiledExecutor::execute_tiled_mlp(const CompiledGraph& graph,
                                         const MlpTileGroup& group,
                                         long B,
                                         long T,
                                         const modules::ForwardHook* hook) {
    // The first op is a view: [B,T,C] → [B*T,C]. Get the 3D input.
    const auto& first_view_op = graph.ops[group.start_op_idx];
    Tensor full_input_3d = resolve_tensor(first_view_op.inputs[0]);
    const long BT = B * T;
    const long C = full_input_3d.Sizes[full_input_3d.Rank - 1];
    Tensor full_input = (full_input_3d.Rank == 3) ? view_tensor(full_input_3d, {BT, C}) : full_input_3d;

    // Find up-weight and down-weight from the matmul ops in the group
    Tensor up_weight{};
    Tensor down_weight{};
    const CompiledOp* up_matmul_op = nullptr;
    const CompiledOp* down_matmul_op = nullptr;
    for (std::size_t idx = group.start_op_idx; idx <= group.end_op_idx; ++idx) {
        const auto& op = graph.ops[idx];
        if (op.type != CompiledOpType::Matmul && op.type != CompiledOpType::MatmulBias) continue;
        for (const auto& inp : op.inputs) {
            if (inp.name.size() >= 13 && inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                up_weight = resolve_tensor(inp);
                up_matmul_op = &op;
            }
            if (inp.name.size() >= 15 && inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                down_weight = resolve_tensor(inp);
                down_matmul_op = &op;
            }
        }
    }
    if (!up_weight.Data || !down_weight.Data || !up_matmul_op || !down_matmul_op) {
        throw std::runtime_error("execute_tiled_mlp: could not find MLP weights in tile group");
    }

    const long MUp = up_weight.Sizes[0];  // [MUp, C] with NT transpose
    const long M = MUp / 2;               // SwiGLU halves the dimension
    const ETensorDType dtype = full_input.DType;

    // Pre-allocate full output [B*T, C]
    Tensor full_output = mRunState.temp_alloc(dtype, {BT, C}, "tiled_mlp_out");

    // Chunk size: min(B*T, C) — Unsloth's "arctic" strategy
    const long chunk_size = std::min(BT, C);
    const long num_chunks = (BT + chunk_size - 1) / chunk_size;

    for (long chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const long offset = chunk_idx * chunk_size;
        const long N = std::min(chunk_size, BT - offset);
        const std::size_t byte_offset = static_cast<std::size_t>(offset) * static_cast<std::size_t>(C) *
                                        static_cast<std::size_t>(get_dtype_size(dtype));

        // Narrow input: [N, C]
        Tensor chunk_in = full_input;
        chunk_in.Data = static_cast<std::byte*>(full_input.Data) + byte_offset;
        chunk_in.Sizes[0] = N;

        // Save stack checkpoint for chunk intermediates
        auto ckpt = mRunState.Stack.checkpoint();
        auto temp_mark = mTemps.size();

        // 1) Up-proj matmul: [N, C] × [MUp, C]^T → [N, MUp]
        Tensor up_out = mRunState.temp_alloc(dtype, {N, MUp}, "up_out");
        mTemps.push_back(up_out);
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(chunk_in, up_weight, up_matmul_op->attrs.transpose, mm_M, mm_N, mm_K);
            matmul(up_out,
                   up_weight,
                   chunk_in,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(up_matmul_op->attrs.transpose),
                   false,
                   mRunState.MainStream);
        }

        // 2) SwiGLU: [N, MUp] → [N, M] (operate in 2D, swiglu_forward handles it as B=1, T=N)
        Tensor act_out = mRunState.temp_alloc(dtype, {N, M}, "act_out");
        mTemps.push_back(act_out);
        swiglu_forward(act_out, up_out, nullptr, 1, static_cast<int>(N), static_cast<int>(M), mRunState.MainStream);

        // 3) Down-proj matmul: [N, M] × [C, M]^T → [N, C]
        //    Write directly into the full_output at the correct offset
        Tensor chunk_out = full_output;
        chunk_out.Data = static_cast<std::byte*>(full_output.Data) + byte_offset;
        chunk_out.Sizes[0] = N;
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(act_out, down_weight, down_matmul_op->attrs.transpose, mm_M, mm_N, mm_K);
            matmul(chunk_out,
                   down_weight,
                   act_out,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(down_matmul_op->attrs.transpose),
                   false,
                   mRunState.MainStream);
        }

        // Restore stack — free chunk intermediates (up_out, act_out)
        mRunState.Stack.restore(ckpt);
        if (mTemps.size() > temp_mark) {
            mTemps.resize(temp_mark);
        }
    }

    // Store results under the tensor IDs that subsequent ops expect:
    // - down_matmul output [B*T, C]
    store_tensor(down_matmul_op->outputs[0], full_output);
    mTemps.push_back(full_output);
    // - last view output [B, T, C]
    const auto& last_view_op = graph.ops[group.end_op_idx];
    Tensor full_output_3d = view_tensor(full_output, {B, T, C});
    store_tensor(last_view_op.outputs[0], full_output_3d);
    // - first view output [B*T, C] (for any references to ln_flat)
    store_tensor(first_view_op.outputs[0], full_input);
    // - intermediate tensor IDs (mlp_up, swiglu, etc.) are not needed
    //   since they only live within the tile group
}

// ============================================================================
// Tiled MLP Backward: combined forward recompute + backward per chunk
// ============================================================================
//
// Instead of running the full backward ops (which need full-size intermediates
// from replay), we recompute the forward MLP per-chunk and immediately compute
// gradients, then free intermediates. This reduces memory from O(B*T * intermediate)
// to O(chunk_size * intermediate) during backward.
//
// Backward MLP group structure (reverse order):
//   view_bwd → matmul_bwd(down) → view_bwd → swiglu_bwd → view_bwd → matmul_bwd(up) → view_bwd
//
// We replace all ops in the group with per-chunk:
//   1. Forward recompute: up_out = ln2_chunk @ up_weight^T, act_out = swiglu(up_out)
//   2. Backward down-proj: d_act = d_out_chunk @ down_weight, d_down_weight += act_out^T @ d_out_chunk
//   3. Backward swiglu: d_up = swiglu_backward(d_act, up_out)
//   4. Backward up-proj: d_ln2_chunk = d_up @ up_weight, d_up_weight += ln2_chunk^T @ d_up
void CompiledExecutor::execute_tiled_mlp_backward(const CompiledGraph& bwd_graph,
                                                  const MlpTileGroup& group,
                                                  long B,
                                                  long T,
                                                  const modules::BackwardHook* hook) {
    const long BT = B * T;

    // Find the matmul_backward ops and extract weight/gradient references
    const CompiledOp* down_bwd_op = nullptr;
    const CompiledOp* up_bwd_op = nullptr;
    for (std::size_t idx = group.start_op_idx; idx <= group.end_op_idx; ++idx) {
        const auto& op = bwd_graph.ops[idx];
        if (op.type != CompiledOpType::MatmulBackward) continue;
        for (const auto& inp : op.inputs) {
            if (inp.name.size() >= 15 && inp.name.compare(inp.name.size() - 15, 15, "mlp_down_weight") == 0) {
                down_bwd_op = &op;
            }
            if (inp.name.size() >= 13 && inp.name.compare(inp.name.size() - 13, 13, "mlp_up_weight") == 0) {
                up_bwd_op = &op;
            }
        }
    }
    if (!down_bwd_op || !up_bwd_op) {
        throw std::runtime_error("execute_tiled_mlp_backward: could not find MLP backward ops");
    }

    // Resolve inputs from backward ops:
    // down_bwd inputs: [d_out (d_mlp_down_flat), activation(swiglu_flat), down_weight]
    // up_bwd inputs: [d_mlp_up_flat, activation(ln2_flat), up_weight]
    //
    // The group may or may not start/end with ViewBackward ops. The actual structure is:
    //   [ViewBackward?] MatmulBackward(down) [View] SwiGLUBackward [View] MatmulBackward(up) [ViewBackward?]
    // We skip all ops in the group and compute everything from raw kernel calls.

    // If the group starts with a ViewBackward, dispatch it to produce the flat gradient
    const auto& first_op = bwd_graph.ops[group.start_op_idx];
    if (first_op.type == CompiledOpType::ViewBackward) {
        dispatch_view_backward(first_op);
    }

    // Get the incoming gradient (d_out for down-proj backward) — already available from prior ops
    Tensor d_mlp_out_flat = resolve_tensor(down_bwd_op->inputs[0]);

    // Get ln2_flat from the up-proj backward's activation input (input[1])
    Tensor ln2_flat = resolve_tensor(up_bwd_op->inputs[1]);
    // Get weights
    Tensor up_weight = resolve_tensor(up_bwd_op->inputs[2]);
    Tensor down_weight = resolve_tensor(down_bwd_op->inputs[2]);

    const long C = ln2_flat.Sizes[ln2_flat.Rank - 1];
    const long MUp = up_weight.Sizes[0];  // [MUp, C] with NT
    const long M = MUp / 2;               // SwiGLU halves
    const ETensorDType dtype = ln2_flat.DType;
    const EMMTranspose fwd_mode = down_bwd_op->attrs.transpose;  // NT for standard matmul
    const int layer_idx = down_bwd_op->attrs.layer_idx;

    // Allocate full d_ln2_flat [B*T, C] output
    Tensor d_ln2_flat = mRunState.temp_alloc(dtype, {BT, C}, "d_ln2_flat");
    fill_zero(d_ln2_flat, mRunState.MainStream);
    mTemps.push_back(d_ln2_flat);

    // Resolve weight gradient tensors (pre-allocated by gradient store)
    // down_bwd outputs: [d_swiglu_flat, d_down_weight]
    // up_bwd outputs: [d_ln2_flat, d_up_weight]
    // Check if weight gradients should be computed (frozen in LoRA)
    auto get_weight_grad = [&](const CompiledOp& op, int out_idx) -> Tensor* {
        if (static_cast<int>(op.outputs.size()) <= out_idx) return nullptr;
        const std::string& dB_name = op.outputs[out_idx].name;
        if (dB_name.empty()) return nullptr;
        std::string weight_name = dB_name;
        if (weight_name.rfind("d_", 0) == 0) weight_name = weight_name.substr(2);
        bool accum = false;
        return mGrads.get_param_grad(weight_name, accum);
    };
    Tensor* d_down_weight_ptr = get_weight_grad(*down_bwd_op, 1);
    Tensor* d_up_weight_ptr = get_weight_grad(*up_bwd_op, 1);

    // Determine accumulation: first chunk uses existing flag, subsequent chunks always accumulate
    const std::string& dB_down_name = down_bwd_op->outputs.size() > 1 ? down_bwd_op->outputs[1].name : "";
    const std::string& dB_up_name = up_bwd_op->outputs.size() > 1 ? up_bwd_op->outputs[1].name : "";
    bool base_accumulate_down = mAccumulateTensors.count(dB_down_name) > 0;
    bool base_accumulate_up = mAccumulateTensors.count(dB_up_name) > 0;

    // Compute backward transpose modes for NT forward
    EMMTranspose mode_dA = EMMTranspose::NN;     // dA = d_out @ B (NN for NT forward)
    EMMTranspose mode_dB_rm = EMMTranspose::TN;  // dB = d_out^T @ A (TN for NT forward)

    // Chunk size: min(B*T, C) — same as forward
    const long chunk_size = std::min(BT, C);
    const long num_chunks = (BT + chunk_size - 1) / chunk_size;

    (void)hook;

    // Per-chunk LoRA backward: the op supplies slice declarations via
    // ``op.attrs.lora_slices`` and the caller passes chunk-sized views.
    auto apply_chunk_lora_backward =
        [&](const CompiledOp& op, Tensor input_2d, Tensor d_output_2d, Tensor dx_2d, int BT_chunk, bool accumulate) {
            if (op.attrs.lora_slices.empty() || mLoRAConfig == nullptr || !mLoRAConfig->enabled() ||
                mLoRAWeights == nullptr || mLoRAGrads == nullptr || mLoRARunState == nullptr || mComm == nullptr ||
                layer_idx < 0) {
                return;
            }
            modules::detail::apply_lora_slices_backward(op.attrs.lora_slices,
                                                        layer_idx,
                                                        input_2d,
                                                        d_output_2d,
                                                        dx_2d,
                                                        BT_chunk,
                                                        mLoRAWeights,
                                                        mLoRAGrads,
                                                        mLoRAConfig,
                                                        mLoRARunState,
                                                        *mComm,
                                                        accumulate,
                                                        mRunState.CublasLtHandle,
                                                        mRunState.CuBlasWorkspace,
                                                        mRunState.MainStream);
        };

    for (long chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const long offset = chunk_idx * chunk_size;
        const long N = std::min(chunk_size, BT - offset);

        auto ckpt = mRunState.Stack.checkpoint();

        // Narrow input tensors to chunk
        auto narrow = [&](const Tensor& full, long cols) -> Tensor {
            Tensor chunk = full;
            const std::size_t byte_off = static_cast<std::size_t>(offset) * static_cast<std::size_t>(cols) *
                                         static_cast<std::size_t>(get_dtype_size(dtype));
            chunk.Data = static_cast<std::byte*>(full.Data) + byte_off;
            chunk.Sizes[0] = N;
            return chunk;
        };

        Tensor ln2_chunk = narrow(ln2_flat, C);
        Tensor d_out_chunk = narrow(d_mlp_out_flat, C);
        Tensor d_ln2_chunk = narrow(d_ln2_flat, C);

        // ---- Forward recompute for this chunk ----
        // 1) Up-proj: up_out = ln2_chunk @ up_weight^T → [N, MUp]
        Tensor up_out = mRunState.temp_alloc(dtype, {N, MUp}, "up_out");
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(ln2_chunk, up_weight, fwd_mode, mm_M, mm_N, mm_K);
            matmul(up_out,
                   up_weight,
                   ln2_chunk,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(fwd_mode),
                   false,
                   mRunState.MainStream);
        }

        // 2) SwiGLU: act_out = swiglu(up_out) → [N, M]
        Tensor act_out = mRunState.temp_alloc(dtype, {N, M}, "act_out");
        swiglu_forward(act_out, up_out, nullptr, 1, static_cast<int>(N), static_cast<int>(M), mRunState.MainStream);

        // ---- Backward for this chunk ----
        // 3) Down-proj backward: dA = d_out_chunk @ down_weight (activation grad)
        Tensor d_act = mRunState.temp_alloc(dtype, {N, M}, "d_act");
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_out_chunk, down_weight, mode_dA, mm_M, mm_N, mm_K);
            matmul(d_act,
                   down_weight,
                   d_out_chunk,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(mode_dA),
                   false,
                   mRunState.MainStream);
        }
        // dB = d_out_chunk^T @ act_out (weight grad, accumulated)
        if (d_down_weight_ptr && d_down_weight_ptr->Data) {
            bool accum = base_accumulate_down || (chunk_idx > 0);
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_out_chunk, act_out, mode_dB_rm, mm_M, mm_N, mm_K);
            matmul(*d_down_weight_ptr,
                   act_out,
                   d_out_chunk,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(mode_dB_rm),
                   accum,
                   mRunState.MainStream);
        }

        // Down-proj LoRA: input = chunk swiglu output, d_out = chunk's
        // incoming gradient, dx = chunk's d_swiglu.
        {
            const bool lora_accum = base_accumulate_down || (chunk_idx > 0);
            apply_chunk_lora_backward(*down_bwd_op,
                                      /*input_2d=*/act_out,
                                      /*d_output_2d=*/d_out_chunk,
                                      /*dx_2d=*/d_act,
                                      static_cast<int>(N),
                                      lora_accum);
        }

        // 4) SwiGLU backward: d_up = swiglu_backward(d_act, up_out) → [N, MUp]
        Tensor d_up = mRunState.temp_alloc(dtype, {N, MUp}, "d_up");
        swiglu_backward(d_up,
                        d_act,
                        up_out,
                        nullptr,
                        1,
                        static_cast<int>(N),
                        static_cast<int>(M),
                        mRunState.MainStream);

        // 5) Up-proj backward: dA = d_up @ up_weight (activation grad → d_ln2_chunk)
        {
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_up, up_weight, mode_dA, mm_M, mm_N, mm_K);
            matmul(d_ln2_chunk,
                   up_weight,
                   d_up,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(mode_dA),
                   false,
                   mRunState.MainStream);
        }
        // dB = d_up^T @ ln2_chunk (weight grad, accumulated)
        if (d_up_weight_ptr && d_up_weight_ptr->Data) {
            bool accum = base_accumulate_up || (chunk_idx > 0);
            int mm_M = 0, mm_N = 0, mm_K = 0;
            matmul_dims(d_up, ln2_chunk, mode_dB_rm, mm_M, mm_N, mm_K);
            matmul(*d_up_weight_ptr,
                   ln2_chunk,
                   d_up,
                   std::nullopt,
                   nullptr,
                   nullptr,
                   mRunState.CublasLtHandle,
                   mRunState.CuBlasWorkspace,
                   mm_N,
                   mm_M,
                   mm_K,
                   swap_transpose(mode_dB_rm),
                   accum,
                   mRunState.MainStream);
        }

        // Up-proj LoRA (fused up+gate): the DSL-declared slices on
        // mlp_up_weight cover both the up and gate halves.
        {
            const bool lora_accum = base_accumulate_up || (chunk_idx > 0);
            apply_chunk_lora_backward(*up_bwd_op,
                                      /*input_2d=*/ln2_chunk,
                                      /*d_output_2d=*/d_up,
                                      /*dx_2d=*/d_ln2_chunk,
                                      static_cast<int>(N),
                                      lora_accum);
        }

        // Free chunk intermediates
        mRunState.Stack.restore(ckpt);
    }

    // Store outputs at tensor IDs the subsequent ops expect.
    // If the group ends with a ViewBackward, simulate its reshape.
    // Otherwise store directly at the up-proj backward output.
    const auto& last_op = bwd_graph.ops[group.end_op_idx];
    if (last_op.type == CompiledOpType::ViewBackward) {
        // The ViewBackward reshapes d_ln2_flat [B*T,C] → d_ln2 [B,T,C]
        Tensor d_ln2_3d = view_tensor(d_ln2_flat, {B, T, C});
        store_tensor(last_op.outputs[0], d_ln2_3d);
    }
    // Always store the flat version at the up-proj backward output ID
    store_tensor(up_bwd_op->outputs[0], d_ln2_flat);

    // Store weight gradients at backward op output tensor IDs
    if (d_down_weight_ptr && d_down_weight_ptr->Data && down_bwd_op->outputs.size() > 1) {
        store_tensor(down_bwd_op->outputs[1], *d_down_weight_ptr);
    }
    if (d_up_weight_ptr && d_up_weight_ptr->Data && up_bwd_op->outputs.size() > 1) {
        store_tensor(up_bwd_op->outputs[1], *d_up_weight_ptr);
    }

    // Also store the intermediate gradient tensor IDs that other backward ops might reference
    // d_swiglu_flat (down_bwd output[0]) and d_mlp_up_flat (from swiglu backward) are NOT needed
    // outside the tiled group — they're consumed within the MLP backward sequence.
}

}  // namespace dsl
