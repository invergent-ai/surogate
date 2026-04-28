// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// EPStrategy implementation.
//
// Organisation:
//   1. File-scope helpers (permute dtype-dispatch, GPU buffer resize,
//      layer-idx parsing, EP-state lookup by total size).
//   2. EPStrategy lifecycle (ctor/dtor, cleanup, lazy streams, state key).
//   3. LLEP subclass hook.
//   4. create_strategy() factory.
//   5. DispatchForwardCtx struct + forward-dispatch phases.
//   6. dispatch_forward, dispatch_backward.
//   7. combine_forward, combine_backward.
//
// Each forward-dispatch phase lives in its own method so that the top-level
// `dispatch_forward` reads as a linear sequence of operations.

#include "runtime/ep/ep_strategy.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

#include "runtime/ep/lpt_planner.h"
#include "runtime/ep/weight_transfer.h"
#include "runtime/executor/compiled_ops.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/qlora/generic_quantizer.h"
#include "runtime/qlora/quantized_tensor.h"
#include "runtime/training/runtime_options.h"
#include "kernels/kernels.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace ep {

namespace {

using dsl::CompiledExecutor;
using dsl::CompiledOp;
using dsl::make_raw_tensor;
using dsl::parse_block_param;

// ============================================================================
// File-scope helpers
// ============================================================================

/// DType-dispatched call to `moe_permute_tokens`. Covers the six call sites
/// that previously inlined a BF16/FP32 switch.
void permute_tokens_dtyped(Tensor& dst,
                           const Tensor& src,
                           const int* indices,
                           int output_rows,
                           int input_rows,
                           int hidden,
                           int top_k,
                           cudaStream_t stream) {
    if (src.DType == ETensorDType::BF16) {
        moe_permute_tokens(dst.get<nv_bfloat16>(),
                           src.get<nv_bfloat16>(),
                           indices,
                           output_rows,
                           input_rows,
                           hidden,
                           top_k,
                           stream);
    } else {
        moe_permute_tokens(dst.get<float>(), src.get<float>(), indices, output_rows, input_rows, hidden, top_k, stream);
    }
}

/// Ensure a cudaMalloc'd buffer is at least `need` bytes. Reallocates
/// (cudaFree → cudaMalloc) when the current capacity is insufficient.
/// Returns true when the buffer grew.
bool alloc_or_resize(void*& ptr, std::size_t& cur_bytes, std::size_t need) {
    if (cur_bytes >= need) return false;
    if (ptr) CUDA_CHECK(cudaFree(ptr));
    CUDA_CHECK(cudaMalloc(&ptr, need));
    cur_bytes = need;
    return true;
}

/// Persist a GPU buffer: grow if too small, then memcpy `need` bytes from
/// `src` on the given stream.
void persist_gpu_buffer(void*& buf, std::size_t& buf_bytes, const void* src, std::size_t need, cudaStream_t stream) {
    alloc_or_resize(buf, buf_bytes, need);
    CUDA_CHECK(cudaMemcpyAsync(buf, src, need, cudaMemcpyDeviceToDevice, stream));
}

/// Persist a tensor into the MoE saved-buffers map (per-layer cudaMalloc
/// buffers kept alive through backward).
void save_persistent_buffer(std::unordered_map<std::string, void*>& buffers,
                            std::unordered_map<std::string, std::size_t>& sizes,
                            const std::string& key,
                            const Tensor& src,
                            cudaStream_t stream) {
    if (!src.Data) return;
    const std::size_t bytes = src.bytes();
    if (bytes == 0) return;
    auto buf_it = buffers.find(key);
    if (buf_it == buffers.end() || sizes[key] < bytes) {
        if (buf_it != buffers.end() && buf_it->second) CUDA_CHECK(cudaFree(buf_it->second));
        void* new_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&new_buf, bytes));
        buffers[key] = new_buf;
        sizes[key] = bytes;
    }
    CUDA_CHECK(cudaMemcpyAsync(buffers[key], src.Data, bytes, cudaMemcpyDeviceToDevice, stream));
}

/// Strip a leading prefix from a string_view in-place. Returns true if
/// the prefix was present and removed.
bool strip_prefix(std::string_view& name, std::string_view prefix) {
    if (name.rfind(prefix, 0) == 0) {
        name.remove_prefix(prefix.size());
        return true;
    }
    return false;
}

/// Resolve a layer index from an op. Uses op.attrs.layer_idx when valid,
/// otherwise parses the first input's name (stripping optional prefixes).
int resolve_layer_idx(const CompiledOp& op, std::initializer_list<std::string_view> optional_prefixes) {
    int layer_idx = op.attrs.layer_idx;
    if (layer_idx >= 0 || op.inputs.empty()) return layer_idx;

    std::string_view name = op.inputs[0].name;
    for (auto prefix : optional_prefixes) {
        if (strip_prefix(name, prefix)) break;
    }
    std::string field;
    parse_block_param(name, layer_idx, field);
    return layer_idx;
}

/// Pick the EP state entry whose recorded total_send/total_recv matches the
/// current input's row count. EP replay-forward and the originating forward
/// write state keyed by `(layer_idx << 1) | replay_slot`; backward reads
/// whichever key's shape matches.
struct EpStateSelection {
    int ep_key;
    bool found;
};

EpStateSelection select_ep_state_by_total(std::unordered_map<int, EpLayerState>& map,
                                          int layer_idx,
                                          bool ep_active,
                                          int expected_total,
                                          int EpLayerState::* size_field,
                                          int default_key) {
    EpStateSelection sel{default_key, map.find(default_key) != map.end()};
    if (!ep_active || layer_idx < 0) return sel;

    const int r0 = (layer_idx << 1);
    const int r1 = r0 | 1;
    auto it_r0 = map.find(r0);
    auto it_r1 = map.find(r1);

    auto match = [&](decltype(it_r0) it) {
        return it != map.end() && it->second.*size_field == expected_total;
    };

    if (match(it_r1)) {
        return {r1, true};
    }
    if (match(it_r0)) {
        return {r0, true};
    }
    if (it_r1 != map.end()) {
        return {r1, true};
    }
    if (it_r0 != map.end()) {
        return {r0, true};
    }
    return sel;
}

}  // namespace

// ============================================================================
// EPStrategy: lifecycle
// ============================================================================

EPStrategy::EPStrategy(const RuntimeOptions& options)
    : mOptions(options) {
}

EPStrategy::~EPStrategy() {
    cleanup_all();
}

cudaStream_t EPStrategy::weight_transfer_stream() {
    if (!mWeightTransferStream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&mWeightTransferStream, cudaStreamNonBlocking));
    }
    return mWeightTransferStream;
}

int EPStrategy::ep_state_key(int layer_idx, bool in_replay) const {
    if (layer_idx < 0) return layer_idx;
    if (mOptions.EPSize <= 1) return layer_idx;
    return (layer_idx << 1) | (in_replay ? 1 : 0);
}

void EPStrategy::free_all_llep_layers() {
    for (auto& [_, state] : mLLEPStates) {
        state.free_lora_gpu();
        state.free_foreign_gpu();
    }
    mLLEPStates.clear();
}

void EPStrategy::cleanup_all() {
    for (auto& [layer, state] : mEpStates) {
        state.free_gpu();
    }
    mEpStates.clear();
    free_all_llep_layers();
    mEPLayerMeta.clear();
    mBufferPool.clear_all();
    if (mWeightTransferStream) {
        cudaStreamDestroy(mWeightTransferStream);
        mWeightTransferStream = nullptr;
    }
}

// ============================================================================
// LLEP subclass
// ============================================================================

bool LLEP::enable_llep_for_layer(bool separate_up_projection) const {
    // Nemotron-style `experts_up` layers are not LLEP-compatible yet
    // (foreign-weight transfer assumes fused gate_up). Fall back to static
    // mapping for those layers transparently.
    return !separate_up_projection;
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<EPStrategy> create_strategy(const RuntimeOptions& options) {
    // LLEP is always the default: it decides per-layer whether to rebalance
    // (driven by ep_load_balance_threshold) and falls back to static mapping
    // transparently when rebalancing isn't warranted or supported.
    return std::make_unique<LLEP>(options);
}

// ============================================================================
// DispatchForwardCtx — locals shared across dispatch_forward phases
// ============================================================================

struct EPStrategy::DispatchForwardCtx {
    int layer_idx = -1;
    int ep_size = 1;
    int ep_rank = 0;
    int ep_key = -1;
    int num_experts = 0;
    int num_local = 0;
    int num_merged = 0;
    int hidden_size = 0;
    int elem_sz = 0;
    int device = -1;
    int total_send = 0;
    int total_recv = 0;
    ETensorDType dtype = ETensorDType::BF16;
    std::string layer_prefix;
    std::string up_weight_name;
    bool separate_up_projection = false;

    bool llep_supported_for_layer = false;
    bool use_llep = false;

    std::vector<int> expert_offsets;
    std::vector<int> local_expert_counts;
    std::vector<int> global_expert_counts;
    ep::LPTPlan plan;

    std::vector<int> expert_to_gpu;
    std::vector<int> merged_experts;
    std::vector<int> global_to_merged;
    std::vector<int> send_splits;

    std::vector<int> recv_splits;
    std::vector<int> recv_all_counts;

    ep::ForeignExpertWeights foreign_weights;
    bool wt_started = false;

    std::string saved_offsets_key;
};

// ============================================================================
// Forward dispatch — phase implementations
// ============================================================================

void EPStrategy::parse_forward_layout(CompiledExecutor& exec,
                                      const CompiledOp& op,
                                      const Tensor& permuted_input,
                                      DispatchForwardCtx& ctx) {
    ctx.ep_size = op.attrs.ep_size;
    ctx.ep_rank = exec.mComm->ep_rank();
    ctx.num_experts = op.attrs.num_experts;
    ctx.num_local = ctx.num_experts / ctx.ep_size;
    ctx.hidden_size = static_cast<int>(permuted_input.Sizes[1]);
    ctx.device = permuted_input.Device;
    ctx.total_send = static_cast<int>(permuted_input.Sizes[0]);
    ctx.dtype = permuted_input.DType;
    ctx.elem_sz = static_cast<int>(get_dtype_size(ctx.dtype));

    ctx.layer_idx = resolve_layer_idx(op, {"saved."});
    ctx.ep_key = ep_state_key(ctx.layer_idx, exec.mInReplay);

    ctx.layer_prefix = "blocks[" + std::to_string(ctx.layer_idx) + "].";
    ctx.up_weight_name = exec.mWeights.has(ctx.layer_prefix + "experts_gate_up") ? "experts_gate_up" : "experts_up";
    ctx.separate_up_projection = (ctx.up_weight_name == "experts_up");
    ctx.llep_supported_for_layer = enable_llep_for_layer(ctx.separate_up_projection);

    // One-shot warnings for production diagnostics.
    {
        static bool printed_marker = false;
        if (!printed_marker && exec.mComm && exec.mComm->rank() == 0) {
            fprintf(stderr,
                    "[EP] ep_dispatch active: strategy=%s layer=%d up_weight=%s ep_size=%d\n",
                    name(),
                    ctx.layer_idx,
                    ctx.up_weight_name.c_str(),
                    ctx.ep_size);
            printed_marker = true;
        }
    }
    if (ctx.separate_up_projection && supports_llep() && !ctx.llep_supported_for_layer) {
        static bool warned = false;
        if (!warned && exec.mComm && exec.mComm->rank() == 0) {
            fprintf(stderr,
                    "[EP] Detected experts_up layer (Nemotron-style MoE); "
                    "LLEP is not supported on this block shape — falling back to static mapping.\n");
            warned = true;
        }
    }

    // Pull expert offsets from the host cache populated by moe_permute.
    auto& cache = exec.mMoEHostOffsetsCache;
    auto it = cache.find(ctx.ep_key);
    if (it == cache.end()) it = cache.find(ctx.layer_idx);  // non-EP-keyed fallback
    if (it == cache.end()) {
        throw std::runtime_error("ep_dispatch: host expert offsets not found for layer " +
                                 std::to_string(ctx.layer_idx));
    }
    ctx.expert_offsets = it->second;

    ctx.local_expert_counts.resize(ctx.num_experts);
    for (int e = 0; e < ctx.num_experts; ++e) {
        ctx.local_expert_counts[e] = ctx.expert_offsets[e + 1] - ctx.expert_offsets[e];
    }
}

void EPStrategy::detect_llep_imbalance(CompiledExecutor& exec, DispatchForwardCtx& ctx) {
    ctx.use_llep = false;
    const float threshold = mOptions.EPLoadBalanceThreshold;
    if (!ctx.llep_supported_for_layer || threshold >= 100.0f) return;

    Tensor counts_gpu =
        exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.num_experts)}, "ep_expert_counts");
    CUDA_CHECK(cudaMemcpyAsync(counts_gpu.Data,
                               ctx.local_expert_counts.data(),
                               ctx.num_experts * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));
    exec.mComm->all_reduce_sum_int_ep(counts_gpu.get<int>(), ctx.num_experts, exec.mRunState.MainStream);

    ctx.global_expert_counts.assign(ctx.num_experts, 0);
    CUDA_CHECK(cudaMemcpyAsync(ctx.global_expert_counts.data(),
                               counts_gpu.Data,
                               ctx.num_experts * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               exec.mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(exec.mRunState.MainStream));
    exec.mTemps.push_back(counts_gpu);

    const float imbalance =
        ep::compute_imbalance_ratio(ctx.global_expert_counts.data(), ctx.num_experts, ctx.ep_size, ctx.num_local);
    ctx.use_llep = (imbalance >= threshold);
}

void EPStrategy::plan_expert_mapping(DispatchForwardCtx& ctx) {
    ctx.expert_to_gpu.assign(ctx.num_experts, 0);
    if (ctx.use_llep) {
        ctx.plan = ep::compute_lpt_plan(ctx.global_expert_counts.data(),
                                        ctx.num_experts,
                                        ctx.ep_size,
                                        ctx.ep_rank,
                                        ctx.num_local);
        ctx.expert_to_gpu = ctx.plan.expert_to_gpu;
    } else {
        for (int e = 0; e < ctx.num_experts; ++e) {
            ctx.expert_to_gpu[e] = e / ctx.num_local;
        }
    }

    ctx.merged_experts.clear();
    for (int e = 0; e < ctx.num_experts; ++e) {
        if (ctx.expert_to_gpu[e] == ctx.ep_rank) ctx.merged_experts.push_back(e);
    }
    ctx.num_merged = static_cast<int>(ctx.merged_experts.size());

    ctx.global_to_merged.assign(ctx.num_experts, -1);
    for (int m = 0; m < ctx.num_merged; ++m)
        ctx.global_to_merged[ctx.merged_experts[m]] = m;

    auto& meta = mEPLayerMeta[ctx.ep_key];
    meta.num_merged = ctx.num_merged;
    meta.native_start = ctx.ep_rank * ctx.num_local;
    meta.num_local = ctx.num_local;
    meta.merged_to_global = ctx.merged_experts;

    ctx.send_splits.assign(ctx.ep_size, 0);
    for (int e = 0; e < ctx.num_experts; ++e) {
        ctx.send_splits[ctx.expert_to_gpu[e]] += ctx.local_expert_counts[e];
    }
}

Tensor EPStrategy::apply_llep_send_reorder(CompiledExecutor& exec,
                                           DispatchForwardCtx& ctx,
                                           const Tensor& permuted_input,
                                           EpLayerState& ep_state) {
    // H2D uploads for the fused kernel.
    Tensor offsets_gpu =
        exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.num_experts + 1)}, "ep_offsets_gpu");
    CUDA_CHECK(cudaMemcpyAsync(offsets_gpu.Data,
                               ctx.expert_offsets.data(),
                               (ctx.num_experts + 1) * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));

    Tensor e2g_gpu = exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.num_experts)}, "ep_e2g_gpu");
    CUDA_CHECK(cudaMemcpyAsync(e2g_gpu.Data,
                               ctx.expert_to_gpu.data(),
                               ctx.num_experts * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));

    Tensor send_buf = exec.mRunState.temp_alloc(ctx.dtype,
                                                {static_cast<long>(ctx.total_send), static_cast<long>(ctx.hidden_size)},
                                                "ep_send_buf");
    exec.mTemps.push_back(send_buf);

    Tensor pwo_gpu = exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.num_experts)}, "ep_pwo_gpu");
    Tensor send_order_gpu =
        exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.total_send)}, "ep_send_order_gpu");

    if (ctx.dtype == ETensorDType::BF16) {
        ep_fused_prepare_send_buffer_bf16(send_buf.get<nv_bfloat16>(),
                                          permuted_input.get<nv_bfloat16>(),
                                          offsets_gpu.get<int>(),
                                          e2g_gpu.get<int>(),
                                          pwo_gpu.get<int>(),
                                          send_order_gpu.get<int>(),
                                          ctx.num_experts,
                                          ctx.ep_size,
                                          ctx.hidden_size,
                                          ctx.total_send,
                                          exec.mRunState.MainStream);
    } else {
        ep_fused_prepare_send_buffer_fp32(send_buf.get<float>(),
                                          permuted_input.get<float>(),
                                          offsets_gpu.get<int>(),
                                          e2g_gpu.get<int>(),
                                          pwo_gpu.get<int>(),
                                          send_order_gpu.get<int>(),
                                          ctx.num_experts,
                                          ctx.ep_size,
                                          ctx.hidden_size,
                                          ctx.total_send,
                                          exec.mRunState.MainStream);
    }

    persist_gpu_buffer(ep_state.llep_send_reorder_gpu,
                       ep_state.llep_send_reorder_bytes,
                       send_order_gpu.Data,
                       send_order_gpu.bytes(),
                       exec.mRunState.MainStream);

    exec.mTemps.push_back(offsets_gpu);
    exec.mTemps.push_back(e2g_gpu);
    exec.mTemps.push_back(pwo_gpu);
    exec.mTemps.push_back(send_order_gpu);
    return send_buf;
}

void EPStrategy::exchange_token_splits(CompiledExecutor& exec, DispatchForwardCtx& ctx) {
    // A2A #1: per-GPU token splits (1 int per peer).
    Tensor send_splits_gpu =
        exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.ep_size)}, "ep_send_splits_gpu");
    Tensor recv_splits_gpu =
        exec.mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(ctx.ep_size)}, "ep_recv_splits_gpu");
    CUDA_CHECK(cudaMemcpyAsync(send_splits_gpu.Data,
                               ctx.send_splits.data(),
                               ctx.ep_size * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));
    std::vector<int> ones(ctx.ep_size, 1);
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(send_splits_gpu.Data),
                                  reinterpret_cast<std::byte*>(recv_splits_gpu.Data),
                                  ones.data(),
                                  ones.data(),
                                  sizeof(int),
                                  exec.mRunState.MainStream);

    // A2A #2: per-expert counts (num_experts ints per peer).
    Tensor send_ec_gpu = exec.mRunState.temp_alloc(ETensorDType::INT32,
                                                   {static_cast<long>(ctx.num_experts * ctx.ep_size)},
                                                   "ep_send_ec_gpu");
    Tensor recv_ec_gpu = exec.mRunState.temp_alloc(ETensorDType::INT32,
                                                   {static_cast<long>(ctx.num_experts * ctx.ep_size)},
                                                   "ep_recv_ec_gpu");
    for (int p = 0; p < ctx.ep_size; ++p) {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(send_ec_gpu.Data) +
                                       static_cast<std::size_t>(p) * ctx.num_experts * sizeof(int),
                                   ctx.local_expert_counts.data(),
                                   ctx.num_experts * sizeof(int),
                                   cudaMemcpyHostToDevice,
                                   exec.mRunState.MainStream));
    }
    std::vector<int> ec_send_splits(ctx.ep_size, ctx.num_experts);
    std::vector<int> ec_recv_splits(ctx.ep_size, ctx.num_experts);
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(send_ec_gpu.Data),
                                  reinterpret_cast<std::byte*>(recv_ec_gpu.Data),
                                  ec_send_splits.data(),
                                  ec_recv_splits.data(),
                                  sizeof(int),
                                  exec.mRunState.MainStream);

    ctx.recv_splits.assign(ctx.ep_size, 0);
    CUDA_CHECK(cudaMemcpyAsync(ctx.recv_splits.data(),
                               recv_splits_gpu.Data,
                               ctx.ep_size * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               exec.mRunState.MainStream));
    ctx.recv_all_counts.assign(ctx.num_experts * ctx.ep_size, 0);
    CUDA_CHECK(cudaMemcpyAsync(ctx.recv_all_counts.data(),
                               recv_ec_gpu.Data,
                               ctx.num_experts * ctx.ep_size * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               exec.mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(exec.mRunState.MainStream));
    exec.mTemps.push_back(send_splits_gpu);
    exec.mTemps.push_back(recv_splits_gpu);
    exec.mTemps.push_back(send_ec_gpu);
    exec.mTemps.push_back(recv_ec_gpu);

    ctx.total_recv = std::accumulate(ctx.recv_splits.begin(), ctx.recv_splits.end(), 0);
}

bool EPStrategy::launch_llep_transfers(CompiledExecutor& exec,
                                       DispatchForwardCtx& ctx,
                                       Tensor*& native_gate_up_out,
                                       Tensor*& native_down_out) {
    native_gate_up_out = nullptr;
    native_down_out = nullptr;
    if (!ctx.use_llep || (ctx.plan.weights_to_send.empty() && ctx.plan.weights_to_receive.empty())) {
        return false;
    }

    native_gate_up_out = &exec.mWeights.get(ctx.layer_prefix + ctx.up_weight_name);
    native_down_out = &exec.mWeights.get(ctx.layer_prefix + "experts_down");

    cudaStream_t wt_stream = weight_transfer_stream();
    // Make the weight-transfer stream wait for MainStream (native dequant
    // pointers must be resolved first).
    cudaEvent_t wt_start;
    CUDA_CHECK(cudaEventCreateWithFlags(&wt_start, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(wt_start, exec.mRunState.MainStream));
    CUDA_CHECK(cudaStreamWaitEvent(wt_stream, wt_start));
    CUDA_CHECK(cudaEventDestroy(wt_start));

    // Prefer quantized transfer (2-8× less P2P bandwidth) when available.
    auto* provider = exec.mWeights.qlora_provider();
    const qlora::QuantizedTensor* qt_gu =
        provider ? provider->try_get_quantized(ctx.layer_prefix + ctx.up_weight_name) : nullptr;
    const qlora::QuantizedTensor* qt_dn =
        provider ? provider->try_get_quantized(ctx.layer_prefix + "experts_down") : nullptr;
    qlora::IQuantizer* quantizer = provider ? provider->get_quantizer() : nullptr;

    if (qt_gu && qt_dn && qt_gu->is_quantized() && qt_dn->is_quantized() && quantizer && !qt_gu->is_on_host()) {
        ep::transfer_expert_weights_quantized(ctx.plan,
                                              *exec.mComm,
                                              exec.mRunState,
                                              wt_stream,
                                              ctx.foreign_weights,
                                              *qt_gu,
                                              *qt_dn,
                                              *native_gate_up_out,
                                              *native_down_out,
                                              quantizer,
                                              ctx.num_local,
                                              ctx.ep_rank);
    } else {
        ep::transfer_expert_weights(ctx.plan,
                                    *exec.mComm,
                                    exec.mRunState,
                                    wt_stream,
                                    ctx.foreign_weights,
                                    *native_gate_up_out,
                                    *native_down_out,
                                    ctx.num_local,
                                    ctx.ep_rank);
    }

    // Transfer per-expert LoRA adapters alongside base weights.
    if (exec.mLoRAConfig && exec.mLoRAWeights && exec.mLoRAConfig->enabled() && exec.mLoRAWeights->enabled()) {
        auto& lora_block = exec.mLoRAWeights->get_block(ctx.layer_idx, wt_stream);
        if (lora_block.moe.use_grouped) {
            const auto& g = lora_block.moe.grouped;
            Tensor null_tensor;
            ep::transfer_expert_lora(ctx.plan,
                                     *exec.mComm,
                                     exec.mRunState,
                                     wt_stream,
                                     ctx.foreign_weights,
                                     g.gate_up.has_value() ? g.gate_up->A : null_tensor,
                                     g.gate_up.has_value() ? g.gate_up->B : null_tensor,
                                     g.gate.has_value() ? g.gate->A : null_tensor,
                                     g.gate.has_value() ? g.gate->B : null_tensor,
                                     g.up.has_value() ? g.up->A : null_tensor,
                                     g.up.has_value() ? g.up->B : null_tensor,
                                     g.down.has_value() ? g.down->A : null_tensor,
                                     g.down.has_value() ? g.down->B : null_tensor,
                                     ctx.num_local,
                                     ctx.ep_rank);
        }
    }

    ctx.wt_started = true;
    return true;
}

Tensor EPStrategy::run_token_a2a(CompiledExecutor& exec,
                                 const DispatchForwardCtx& ctx,
                                 const Tensor& send_src,
                                 void*& recv_buf_out,
                                 std::size_t& recv_bytes_out) {
    recv_bytes_out = static_cast<std::size_t>(ctx.total_recv) * ctx.hidden_size * ctx.elem_sz;
    recv_buf_out = mBufferPool.acquire(recv_bytes_out);

    Tensor recv = make_raw_tensor(recv_buf_out,
                                  ctx.dtype,
                                  {static_cast<long>(ctx.total_recv), static_cast<long>(ctx.hidden_size)},
                                  ctx.device);

    std::vector<int> send_elem(ctx.ep_size), recv_elem(ctx.ep_size);
    for (int p = 0; p < ctx.ep_size; ++p) {
        send_elem[p] = ctx.send_splits[p] * ctx.hidden_size;
        recv_elem[p] = ctx.recv_splits[p] * ctx.hidden_size;
    }
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(send_src.Data),
                                  reinterpret_cast<std::byte*>(recv.Data),
                                  send_elem.data(),
                                  recv_elem.data(),
                                  ctx.elem_sz,
                                  exec.mRunState.MainStream);
    return recv;
}

void EPStrategy::permute_recv_tokens(CompiledExecutor& exec,
                                     DispatchForwardCtx& ctx,
                                     const Tensor& recv_hidden,
                                     Tensor& sorted_recv_out,
                                     Tensor& local_scatter_out,
                                     Tensor& merged_offsets_out) {
    // Build recv-side merged expert IDs. After A2A, tokens are in source-peer
    // order; within each peer's chunk, experts appear in ascending global ID.
    std::vector<int> recv_merged_ids(ctx.total_recv);
    {
        int pos = 0;
        for (int p = 0; p < ctx.ep_size; ++p) {
            for (int e = 0; e < ctx.num_experts; ++e) {
                if (ctx.expert_to_gpu[e] != ctx.ep_rank) continue;
                const int count = ctx.recv_all_counts[p * ctx.num_experts + e];
                const int merged_idx = ctx.global_to_merged[e];
                if (merged_idx < 0 || merged_idx >= ctx.num_merged) {
                    throw std::runtime_error("ep_dispatch: invalid merged expert index at layer " +
                                             std::to_string(ctx.layer_idx) + " e=" + std::to_string(e) +
                                             " merged_idx=" + std::to_string(merged_idx) +
                                             " num_merged=" + std::to_string(ctx.num_merged));
                }
                for (int t = 0; t < count; ++t) {
                    if (pos < ctx.total_recv) recv_merged_ids[pos++] = merged_idx;
                }
            }
        }
        if (pos != ctx.total_recv) {
            throw std::runtime_error("ep_dispatch: recv_merged_ids fill mismatch at layer " +
                                     std::to_string(ctx.layer_idx) + " pos=" + std::to_string(pos) +
                                     " total_recv=" + std::to_string(ctx.total_recv));
        }
    }

    const std::size_t recv_routing_bytes = static_cast<std::size_t>(ctx.total_recv) * sizeof(int);
    void* recv_routing_ptr = mBufferPool.acquire(recv_routing_bytes);
    CUDA_CHECK(cudaMemcpyAsync(recv_routing_ptr,
                               recv_merged_ids.data(),
                               ctx.total_recv * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));
    Tensor recv_routing =
        make_raw_tensor(recv_routing_ptr, ETensorDType::INT32, {static_cast<long>(ctx.total_recv), 1L}, ctx.device);

    // Per-layer persistent outputs (consumed by GEMM, kept until backward).
    auto& ep_state_out = mEpStates[ctx.ep_key];
    const std::size_t sorted_need = static_cast<std::size_t>(ctx.total_recv) * ctx.hidden_size * ctx.elem_sz;
    alloc_or_resize(ep_state_out.sorted_recv_gpu, ep_state_out.sorted_recv_bytes, sorted_need);
    sorted_recv_out = make_raw_tensor(ep_state_out.sorted_recv_gpu,
                                      ctx.dtype,
                                      {static_cast<long>(ctx.total_recv), static_cast<long>(ctx.hidden_size)},
                                      ctx.device);

    const std::size_t local_scatter_need = static_cast<std::size_t>(ctx.total_recv) * sizeof(int);
    alloc_or_resize(ep_state_out.local_scatter_gpu, ep_state_out.local_scatter_bytes, local_scatter_need);
    local_scatter_out = make_raw_tensor(ep_state_out.local_scatter_gpu,
                                        ETensorDType::INT32,
                                        {static_cast<long>(ctx.total_recv)},
                                        ctx.device);

    // Helper buffer (4 int blocks packed): counts, offsets, positions, local_gather.
    const std::size_t helper_bytes =
        (ctx.num_merged + (ctx.num_merged + 1) + ctx.num_merged + ctx.total_recv) * sizeof(int);
    void* helper_buf = mBufferPool.acquire(helper_bytes);
    int* merged_counts_ptr = static_cast<int*>(helper_buf);
    int* merged_offsets_ptr = merged_counts_ptr + ctx.num_merged;
    int* merged_positions_ptr = merged_offsets_ptr + (ctx.num_merged + 1);
    int* local_gather_ptr = merged_positions_ptr + ctx.num_merged;

    Tensor merged_counts_t =
        make_raw_tensor(merged_counts_ptr, ETensorDType::INT32, {static_cast<long>(ctx.num_merged)}, ctx.device);
    Tensor merged_offsets_t =
        make_raw_tensor(merged_offsets_ptr, ETensorDType::INT32, {static_cast<long>(ctx.num_merged + 1)}, ctx.device);
    Tensor merged_positions_t =
        make_raw_tensor(merged_positions_ptr, ETensorDType::INT32, {static_cast<long>(ctx.num_merged)}, ctx.device);
    Tensor local_gather =
        make_raw_tensor(local_gather_ptr, ETensorDType::INT32, {static_cast<long>(ctx.total_recv)}, ctx.device);

    fill_zero(merged_counts_t, exec.mRunState.MainStream);
    fill_zero(merged_positions_t, exec.mRunState.MainStream);
    CUDA_CHECK(cudaMemsetAsync(local_gather.Data, 0, local_gather.bytes(), exec.mRunState.MainStream));
    CUDA_CHECK(cudaMemsetAsync(local_scatter_out.Data, 0xFF, local_scatter_out.bytes(), exec.mRunState.MainStream));

    moe_compute_expert_counts(merged_counts_t.get<int>(),
                              recv_routing.get<int>(),
                              ctx.total_recv,
                              1,
                              ctx.num_merged,
                              exec.mRunState.MainStream);
    moe_compute_expert_offsets(merged_offsets_t.get<int>(),
                               merged_counts_t.get<int>(),
                               ctx.num_merged,
                               exec.mRunState.MainStream);
    moe_build_indices(local_gather.get<int>(),
                      local_scatter_out.get<int>(),
                      recv_routing.get<int>(),
                      merged_offsets_t.get<int>(),
                      merged_positions_t.get<int>(),
                      ctx.total_recv,
                      1,
                      ctx.num_merged,
                      exec.mRunState.MainStream);

    permute_tokens_dtyped(sorted_recv_out,
                          recv_hidden,
                          local_gather.get<int>(),
                          ctx.total_recv,
                          ctx.total_recv,
                          ctx.hidden_size,
                          1,
                          exec.mRunState.MainStream);

    // Cache host offsets for grouped GEMM fast-path lookup.
    std::vector<int> merged_offsets_host(ctx.num_merged + 1);
    CUDA_CHECK(cudaMemcpyAsync(merged_offsets_host.data(),
                               merged_offsets_t.get<int>(),
                               (ctx.num_merged + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               exec.mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(exec.mRunState.MainStream));
    exec.mMoEHostOffsetsCache[ctx.ep_key] = std::move(merged_offsets_host);

    // Persist send_order (local_gather) and recv_reorder (local_scatter)
    // BEFORE returning the helper buffer to the pool — their .Data pointers
    // live inside helper_buf and the pool may hand it out to the next caller.
    auto& ep_state_persist = mEpStates[ctx.ep_key];
    persist_gpu_buffer(ep_state_persist.send_order_gpu,
                       ep_state_persist.send_order_bytes,
                       local_gather.Data,
                       local_gather.bytes(),
                       exec.mRunState.MainStream);
    persist_gpu_buffer(ep_state_persist.recv_reorder_gpu,
                       ep_state_persist.recv_reorder_bytes,
                       local_scatter_out.Data,
                       local_scatter_out.bytes(),
                       exec.mRunState.MainStream);

    merged_offsets_out = merged_offsets_t;

    mBufferPool.release(recv_routing_ptr, recv_routing_bytes);
    mBufferPool.release(helper_buf, helper_bytes);
}

void EPStrategy::persist_dispatch_state(CompiledExecutor& exec,
                                        DispatchForwardCtx& ctx,
                                        const Tensor& merged_offsets_t,
                                        Tensor& merged_offsets_persisted_out) {
    ctx.saved_offsets_key = exec.moe_saved_key(ctx.layer_idx, "moe_expert_offsets");
    save_persistent_buffer(exec.mMoeSavedBuffers,
                           exec.mMoeSavedSizes,
                           ctx.saved_offsets_key,
                           merged_offsets_t,
                           exec.mRunState.MainStream);

    merged_offsets_persisted_out = merged_offsets_t;
    merged_offsets_persisted_out.Data = static_cast<std::byte*>(exec.mMoeSavedBuffers[ctx.saved_offsets_key]);
    exec.bind_tensor("moe_expert_offsets", merged_offsets_persisted_out);
}

void EPStrategy::finalize_llep_state(CompiledExecutor& exec,
                                     DispatchForwardCtx& ctx,
                                     const Tensor& native_gate_up,
                                     const Tensor& native_down) {
    cudaEvent_t wt_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&wt_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(wt_done, mWeightTransferStream));
    CUDA_CHECK(cudaStreamWaitEvent(exec.mRunState.MainStream, wt_done));
    CUDA_CHECK(cudaEventDestroy(wt_done));
    if (auto err = cudaGetLastError(); err != cudaSuccess) {
        throw std::runtime_error(std::string("ep_dispatch: weight transfer stream error at layer ") +
                                 std::to_string(ctx.layer_idx) + ": " + cudaGetErrorString(err));
    }

    auto& llep = mLLEPStates[ctx.ep_key];
    populate_llep_weight_pointers(llep, ctx, native_gate_up, native_down, &ctx.foreign_weights);
    const int native_start = ctx.ep_rank * ctx.num_local;

    // Build merged LoRA tensors (native + foreign adapters concatenated).
    if (exec.mLoRAConfig && exec.mLoRAWeights && exec.mLoRAConfig->enabled() && exec.mLoRAWeights->enabled()) {
        auto& lora_block = exec.mLoRAWeights->get_block(ctx.layer_idx, exec.mRunState.MainStream);
        if (lora_block.moe.use_grouped && lora_block.moe.grouped.has_any()) {
            const auto& g = lora_block.moe.grouped;
            auto& ml = llep.merged_lora;
            llep.has_merged_lora = true;

            auto build_merged_pair = [&](const std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& local_opt,
                                         std::optional<modules::LoRAGroupedLayerWeights<Tensor>>& merged_opt,
                                         auto get_foreign_A,
                                         auto get_foreign_B) {
                if (!local_opt.has_value() || !local_opt->has_value()) return;
                const auto& local = *local_opt;
                const long a_rows = local.A.Sizes[1];
                const long a_cols = local.A.Sizes[2];
                const long b_rows = local.B.Sizes[1];
                const long b_cols = local.B.Sizes[2];
                const std::size_t a_slice = static_cast<std::size_t>(a_rows * a_cols) * get_dtype_size(local.A.DType);
                const std::size_t b_slice = static_cast<std::size_t>(b_rows * b_cols) * get_dtype_size(local.B.DType);
                const std::size_t need_A = static_cast<std::size_t>(ctx.num_merged) * a_slice;
                const std::size_t need_B = static_cast<std::size_t>(ctx.num_merged) * b_slice;

                // Per-layer owned allocations — LoRA is small (~9 MB/layer)
                // so per-layer storage avoids the shared-buffer aliasing bug.
                void* ptr_A = nullptr;
                void* ptr_B = nullptr;
                CUDA_CHECK(cudaMalloc(&ptr_A, need_A));
                llep.owned_lora_ptrs.push_back(ptr_A);
                CUDA_CHECK(cudaMalloc(&ptr_B, need_B));
                llep.owned_lora_ptrs.push_back(ptr_B);

                modules::LoRAGroupedLayerWeights<Tensor> merged;
                merged.A = make_raw_tensor(ptr_A,
                                           local.A.DType,
                                           {static_cast<long>(ctx.num_merged), a_rows, a_cols},
                                           ctx.device);
                merged.B = make_raw_tensor(ptr_B,
                                           local.B.DType,
                                           {static_cast<long>(ctx.num_merged), b_rows, b_cols},
                                           ctx.device);

                for (int m = 0; m < ctx.num_merged; ++m) {
                    const int global_e = ctx.merged_experts[m];
                    const int ln = global_e - native_start;
                    auto* a_dst = static_cast<std::byte*>(merged.A.Data) + static_cast<std::size_t>(m) * a_slice;
                    auto* b_dst = static_cast<std::byte*>(merged.B.Data) + static_cast<std::size_t>(m) * b_slice;

                    if (ln >= 0 && ln < ctx.num_local) {
                        const auto* a_src =
                            static_cast<const std::byte*>(local.A.Data) + static_cast<std::size_t>(ln) * a_slice;
                        const auto* b_src =
                            static_cast<const std::byte*>(local.B.Data) + static_cast<std::size_t>(ln) * b_slice;
                        CUDA_CHECK(cudaMemcpyAsync(a_dst,
                                                   a_src,
                                                   a_slice,
                                                   cudaMemcpyDeviceToDevice,
                                                   exec.mRunState.MainStream));
                        CUDA_CHECK(cudaMemcpyAsync(b_dst,
                                                   b_src,
                                                   b_slice,
                                                   cudaMemcpyDeviceToDevice,
                                                   exec.mRunState.MainStream));
                    } else {
                        auto wit = ctx.foreign_weights.weights.find(global_e);
                        if (wit != ctx.foreign_weights.weights.end()) {
                            const Tensor& fa = get_foreign_A(wit->second.lora);
                            const Tensor& fb = get_foreign_B(wit->second.lora);
                            if (!fa.is_null()) {
                                CUDA_CHECK(cudaMemcpyAsync(a_dst,
                                                           fa.Data,
                                                           a_slice,
                                                           cudaMemcpyDeviceToDevice,
                                                           exec.mRunState.MainStream));
                                CUDA_CHECK(cudaMemcpyAsync(b_dst,
                                                           fb.Data,
                                                           b_slice,
                                                           cudaMemcpyDeviceToDevice,
                                                           exec.mRunState.MainStream));
                            }
                        }
                    }
                }
                merged_opt = std::move(merged);
            };

            build_merged_pair(
                g.gate_up,
                ml.gate_up,
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_up_A; },
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_up_B; });
            build_merged_pair(
                g.gate,
                ml.gate,
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_A; },
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.gate_B; });
            build_merged_pair(
                g.up,
                ml.up,
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.up_A; },
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.up_B; });
            build_merged_pair(
                g.down,
                ml.down,
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.down_A; },
                [](const ep::ExpertLoRA& l) -> const Tensor& { return l.down_B; });
        }
    }

    llep.merged_offsets_host = exec.mMoEHostOffsetsCache[ctx.ep_key];
    llep.merged_offsets_gpu = exec.mMoeSavedBuffers[ctx.saved_offsets_key];
    llep.merged_offsets_gpu_bytes = exec.mMoeSavedSizes[ctx.saved_offsets_key];

    // Transfer foreign-weight buffer ownership to LLEP state. Pointers in
    // gate_up_weight_ptrs / down_weight_ptrs reference these buffers, so
    // they must stay alive until the LLEP state is cleared.
    llep.owned_foreign_ptrs = std::move(ctx.foreign_weights.owned_gpu_ptrs);
    ctx.foreign_weights.owned_gpu_ptrs.clear();  // prevent double-free
}

void EPStrategy::populate_llep_weight_pointers(LLEPLayerState& llep,
                                               const DispatchForwardCtx& ctx,
                                               const Tensor& native_gate_up,
                                               const Tensor& native_down,
                                               const ep::ForeignExpertWeights* foreign_weights) {
    const long gu_rows = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[1] : native_gate_up.Sizes[0];
    const long gu_cols = (native_gate_up.Rank >= 3) ? native_gate_up.Sizes[2] : native_gate_up.Sizes[1];
    const long dn_rows = (native_down.Rank >= 3) ? native_down.Sizes[1] : native_down.Sizes[0];
    const long dn_cols = (native_down.Rank >= 3) ? native_down.Sizes[2] : native_down.Sizes[1];
    const std::size_t gu_expert_bytes = static_cast<std::size_t>(gu_rows * gu_cols) * ctx.elem_sz;
    const std::size_t dn_expert_bytes = static_cast<std::size_t>(dn_rows * dn_cols) * ctx.elem_sz;

    llep.active = true;
    llep.num_merged_experts = ctx.num_merged;
    llep.merged_to_global = ctx.merged_experts;
    llep.global_to_merged = ctx.global_to_merged;
    llep.weight_dtype = native_gate_up.DType;
    llep.gate_up_weight_ptrs.resize(ctx.num_merged);
    llep.down_weight_ptrs.resize(ctx.num_merged);

    const int native_start = ctx.ep_rank * ctx.num_local;
    for (int m = 0; m < ctx.num_merged; ++m) {
        const int global_e = ctx.merged_experts[m];
        const int local_native = global_e - native_start;

        if (local_native >= 0 && local_native < ctx.num_local) {
            llep.gate_up_weight_ptrs[m] = static_cast<const std::byte*>(native_gate_up.Data) +
                                          static_cast<std::size_t>(local_native) * gu_expert_bytes;
            llep.down_weight_ptrs[m] = static_cast<const std::byte*>(native_down.Data) +
                                       static_cast<std::size_t>(local_native) * dn_expert_bytes;
        } else if (foreign_weights) {
            auto wit = foreign_weights->weights.find(global_e);
            if (wit != foreign_weights->weights.end()) {
                llep.gate_up_weight_ptrs[m] = wit->second.gate_up.Data;
                llep.down_weight_ptrs[m] = wit->second.down.Data;
            } else {
                // LPT plan guarantees every merged expert is either native or
                // received; fall back to native[0] if that invariant is violated.
                llep.gate_up_weight_ptrs[m] = native_gate_up.Data;
                llep.down_weight_ptrs[m] = native_down.Data;
            }
        } else {
            // Native-only path: non-native merged experts aren't expected here;
            // pointing at native[0] keeps GEMM launches from dereferencing null.
            llep.gate_up_weight_ptrs[m] = native_gate_up.Data;
            llep.down_weight_ptrs[m] = native_down.Data;
        }
    }
}

Tensor EPStrategy::apply_llep_inverse_reorder(CompiledExecutor& exec,
                                              const EpLayerState& ep_state,
                                              const Tensor& input,
                                              int hidden_size,
                                              int elem_sz,
                                              void*& out_buf,
                                              std::size_t& out_bytes) {
    const int N = ep_state.total_send;
    std::vector<int> send_order_host(N);
    CUDA_CHECK(cudaMemcpyAsync(send_order_host.data(),
                               ep_state.llep_send_reorder_gpu,
                               N * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               exec.mRunState.MainStream));
    CUDA_CHECK(cudaStreamSynchronize(exec.mRunState.MainStream));

    std::vector<int> inverse_order(N);
    for (int i = 0; i < N; ++i)
        inverse_order[send_order_host[i]] = i;

    const std::size_t inv_gpu_bytes = static_cast<std::size_t>(N) * sizeof(int);
    void* inv_gpu_ptr = mBufferPool.acquire(inv_gpu_bytes);
    CUDA_CHECK(cudaMemcpyAsync(inv_gpu_ptr,
                               inverse_order.data(),
                               N * sizeof(int),
                               cudaMemcpyHostToDevice,
                               exec.mRunState.MainStream));

    const std::size_t need = static_cast<std::size_t>(N) * hidden_size * elem_sz;
    alloc_or_resize(out_buf, out_bytes, need);

    Tensor output =
        make_raw_tensor(out_buf, input.DType, {static_cast<long>(N), static_cast<long>(hidden_size)}, input.Device);
    permute_tokens_dtyped(output,
                          input,
                          static_cast<int*>(inv_gpu_ptr),
                          N,
                          N,
                          hidden_size,
                          1,
                          exec.mRunState.MainStream);
    mBufferPool.release(inv_gpu_ptr, inv_gpu_bytes);
    return output;
}

void EPStrategy::finalize_native_only_state(CompiledExecutor& exec, DispatchForwardCtx& ctx) {
    if (ctx.ep_size <= 1) return;

    // Native-only pointer path for architectures that skip LLEP weight
    // transfer. QLoRA offloading rotates resolved dequant pointers across
    // steps, so disable this fallback when offloading is active to avoid
    // stale pointers corrupting EP MoE.
    bool allow = true;
    if (auto* provider = exec.mWeights.qlora_provider()) {
        if (provider->has_offloading()) {
            allow = false;
        }
    }
    if (!allow) return;

    const Tensor& native_gate_up = exec.mWeights.get(ctx.layer_prefix + ctx.up_weight_name);
    const Tensor& native_down = exec.mWeights.get(ctx.layer_prefix + "experts_down");

    auto& llep = mLLEPStates[ctx.ep_key];
    populate_llep_weight_pointers(llep, ctx, native_gate_up, native_down, /*foreign_weights=*/nullptr);
    llep.merged_offsets_host = exec.mMoEHostOffsetsCache[ctx.ep_key];
    llep.merged_offsets_gpu = exec.mMoeSavedBuffers[ctx.saved_offsets_key];
    llep.merged_offsets_gpu_bytes = exec.mMoeSavedSizes[ctx.saved_offsets_key];
}

// ============================================================================
// dispatch_forward — orchestrator
// ============================================================================

void EPStrategy::dispatch_forward(CompiledExecutor& exec, const CompiledOp& op) {
    Tensor& permuted_input = exec.resolve_tensor(op.inputs[0]);
    Tensor& routing_indices = exec.resolve_tensor(op.inputs[1]);
    Tensor& scatter_indices_in = exec.resolve_tensor(op.inputs[2]);
    (void)routing_indices;  // reserved for future strategies

    // Early-out when EP is inactive.
    const int ep_size = op.attrs.ep_size;
    if (ep_size <= 1 || !exec.mComm || !exec.mComm->ep_enabled()) {
        exec.store_tensor(op.outputs[0], permuted_input);
        exec.store_tensor(op.outputs[1], scatter_indices_in);
        return;
    }

    DispatchForwardCtx ctx;
    parse_forward_layout(exec, op, permuted_input, ctx);
    detect_llep_imbalance(exec, ctx);

    // Drop any stale LLEP state from earlier layers. Foreign expert weight
    // buffers are ~20 MB each and accumulate across MoE layers, causing OOM
    // if kept alive. Only the current layer's LLEP state survives this call;
    // earlier layers' backward uses mEPLayerMeta to reconstruct pointers.
    auto& ep_state = mEpStates[ctx.ep_key];
    if (!ctx.use_llep && ep_state.llep_send_reorder_gpu) {
        CUDA_CHECK(cudaFree(ep_state.llep_send_reorder_gpu));
        ep_state.llep_send_reorder_gpu = nullptr;
        ep_state.llep_send_reorder_bytes = 0;
    }
    free_all_llep_layers();

    plan_expert_mapping(ctx);

    // Token buffer — either the caller's permuted input, or a LLEP-reordered copy.
    Tensor llep_send_buf;
    const Tensor* send_ptr = &permuted_input;
    if (ctx.use_llep) {
        llep_send_buf = apply_llep_send_reorder(exec, ctx, permuted_input, ep_state);
        send_ptr = &llep_send_buf;
    }

    exchange_token_splits(exec, ctx);

    // Launch LLEP base + LoRA transfers on a separate stream (overlaps the
    // token A2A below). Returns pointers to the native weights (needed later
    // for merged-pointer bookkeeping).
    Tensor* native_gate_up_ptr = nullptr;
    Tensor* native_down_ptr = nullptr;
    launch_llep_transfers(exec, ctx, native_gate_up_ptr, native_down_ptr);

    // Token A2A.
    void* recv_hidden_ptr = nullptr;
    std::size_t recv_hidden_bytes = 0;
    Tensor recv_hidden = run_token_a2a(exec, ctx, *send_ptr, recv_hidden_ptr, recv_hidden_bytes);
    dsl::CommunicationHookPayload comm_payload;
    comm_payload.send_tensor = send_ptr;
    comm_payload.recv_tensor = &recv_hidden;
    comm_payload.ep_size = ctx.ep_size;
    comm_payload.total_send = ctx.total_send;
    comm_payload.total_recv = ctx.total_recv;
    comm_payload.token_all_to_all_completed = true;
    exec.dispatch_schema_layer_hooks(dsl::HookEventKind::AfterAllToAll, ctx.layer_idx, &comm_payload);
    exec.dispatch_schema_layer_hooks(dsl::HookEventKind::AfterCommunication, ctx.layer_idx, &comm_payload);

    // Build recv-side merged IDs, re-permute, and persist send_order / recv_reorder.
    Tensor sorted_recv;
    Tensor local_scatter;
    Tensor merged_offsets_t;
    permute_recv_tokens(exec, ctx, recv_hidden, sorted_recv, local_scatter, merged_offsets_t);
    mBufferPool.release(recv_hidden_ptr, recv_hidden_bytes);

    // Persist merged_offsets to the MoE saved-buffers map and bind as
    // `moe_expert_offsets` for grouped GEMM ops. Expose the persisted
    // tensor back on the stack via `merged_offsets_persisted`.
    Tensor merged_offsets_persisted;
    persist_dispatch_state(exec, ctx, merged_offsets_t, merged_offsets_persisted);

    // Publish EP state for backward / combine.
    ep_state.send_splits = ctx.send_splits;
    ep_state.recv_splits = ctx.recv_splits;
    ep_state.total_send = ctx.total_send;
    ep_state.total_recv = ctx.total_recv;

    // Build the LLEP merged-weight view, or fall back to native-only pointers
    // when no foreign weights were transferred.
    if (ctx.wt_started) {
        finalize_llep_state(exec, ctx, *native_gate_up_ptr, *native_down_ptr);
    } else {
        finalize_native_only_state(exec, ctx);
    }

    exec.store_tensor(op.outputs[0], sorted_recv);
    exec.store_tensor(op.outputs[1], local_scatter);
}

// ============================================================================
// dispatch_backward
// ============================================================================

void EPStrategy::dispatch_backward(CompiledExecutor& exec, const CompiledOp& op) {
    Tensor& d_recv_sorted = exec.resolve_tensor(op.inputs[0]);

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(d_recv_sorted.Sizes[1]);

    if (ep_size <= 1 || !exec.mComm || !exec.mComm->ep_enabled()) {
        Tensor& d_permuted = exec.ensure_output_tensor(op.outputs[0]);
        d_permuted = d_recv_sorted;
        exec.store_tensor(op.outputs[0], d_permuted);
        return;
    }

    const int layer_idx = resolve_layer_idx(op, {"d_", "saved."});
    const int input_total_recv = static_cast<int>(d_recv_sorted.Sizes[0]);
    const int default_key = ep_state_key(layer_idx, exec.mInReplay);

    auto sel = select_ep_state_by_total(mEpStates,
                                        layer_idx,
                                        mOptions.EPSize > 1,
                                        input_total_recv,
                                        &EpLayerState::total_recv,
                                        default_key);
    if (!sel.found) {
        throw std::runtime_error("ep_dispatch_backward: no EP state for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = mEpStates[sel.ep_key];
    auto& ep_state_mut = mEpStates[sel.ep_key];
    if (ep_state.total_recv != input_total_recv) {
        std::ostringstream oss;
        oss << "ep_dispatch_backward: EP state/input mismatch at layer " << layer_idx << " (selected_key=" << sel.ep_key
            << ", state.total_recv=" << ep_state.total_recv << ", input.rows=" << input_total_recv << ")";
        throw std::runtime_error(oss.str());
    }
    const int elem_sz = static_cast<int>(get_dtype_size(d_recv_sorted.DType));

    // 1. Un-sort gradient: reverse the local re-permutation from dispatch.
    const std::size_t unsorted_bytes = static_cast<std::size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* d_recv_unsorted_ptr = mBufferPool.acquire(unsorted_bytes);
    Tensor d_recv_unsorted = make_raw_tensor(d_recv_unsorted_ptr,
                                             d_recv_sorted.DType,
                                             {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
                                             d_recv_sorted.Device);
    permute_tokens_dtyped(d_recv_unsorted,
                          d_recv_sorted,
                          static_cast<int*>(ep_state.recv_reorder_gpu),
                          ep_state.total_recv,
                          ep_state.total_recv,
                          hidden_size,
                          1,
                          exec.mRunState.MainStream);

    // 2. Reverse A2A (swap send/recv splits).
    const std::size_t send_need = static_cast<std::size_t>(ep_state.total_send) * hidden_size * elem_sz;
    alloc_or_resize(ep_state_mut.dispatch_bwd_send_gpu, ep_state_mut.dispatch_bwd_send_bytes, send_need);
    Tensor d_send_buf = make_raw_tensor(ep_state_mut.dispatch_bwd_send_gpu,
                                        d_recv_sorted.DType,
                                        {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
                                        d_recv_sorted.Device);

    std::vector<int> reverse_send_elem, reverse_recv_elem;
    ep_state.build_reverse_a2a_elem_splits(hidden_size, reverse_send_elem, reverse_recv_elem);
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(d_recv_unsorted.Data),
                                  reinterpret_cast<std::byte*>(d_send_buf.Data),
                                  reverse_send_elem.data(),
                                  reverse_recv_elem.data(),
                                  elem_sz,
                                  exec.mRunState.MainStream);
    mBufferPool.release(d_recv_unsorted_ptr, unsorted_bytes);

    // 3. If LLEP was active, undo the send reorder.
    if (!ep_state.llep_send_reorder_gpu) {
        exec.store_tensor(op.outputs[0], d_send_buf);
        return;
    }
    Tensor d_permuted = apply_llep_inverse_reorder(exec,
                                                   ep_state,
                                                   d_send_buf,
                                                   hidden_size,
                                                   elem_sz,
                                                   ep_state_mut.dispatch_bwd_out_gpu,
                                                   ep_state_mut.dispatch_bwd_out_bytes);
    exec.store_tensor(op.outputs[0], d_permuted);
}

// ============================================================================
// combine_forward
// ============================================================================

void EPStrategy::combine_forward(CompiledExecutor& exec, const CompiledOp& op) {
    Tensor& expert_output = exec.resolve_tensor(op.inputs[0]);  // [total_recv, C]

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(expert_output.Sizes[1]);
    if (ep_size <= 1 || !exec.mComm || !exec.mComm->ep_enabled()) {
        exec.store_tensor(op.outputs[0], expert_output);
        return;
    }

    const int layer_idx = resolve_layer_idx(op, {"saved."});
    const int ep_key = ep_state_key(layer_idx, exec.mInReplay);

    auto it = mEpStates.find(ep_key);
    if (it == mEpStates.end()) {
        throw std::runtime_error("ep_combine: no EP state found for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = it->second;
    auto& ep_state_mut = mEpStates[ep_key];
    const int elem_sz = static_cast<int>(get_dtype_size(expert_output.DType));

    // 1. Un-sort expert output (reverse local re-permutation).
    const std::size_t unsorted_bytes = static_cast<std::size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* unsorted_ptr = mBufferPool.acquire(unsorted_bytes);
    Tensor unsorted_output = make_raw_tensor(unsorted_ptr,
                                             expert_output.DType,
                                             {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
                                             expert_output.Device);
    permute_tokens_dtyped(unsorted_output,
                          expert_output,
                          static_cast<int*>(ep_state.recv_reorder_gpu),
                          ep_state.total_recv,
                          ep_state.total_recv,
                          hidden_size,
                          1,
                          exec.mRunState.MainStream);

    // 2. Reverse A2A: send results back to originating GPUs.
    const std::size_t combined_need = static_cast<std::size_t>(ep_state.total_send) * hidden_size * elem_sz;
    alloc_or_resize(ep_state_mut.combined_gpu, ep_state_mut.combined_bytes, combined_need);
    Tensor combined = make_raw_tensor(ep_state_mut.combined_gpu,
                                      expert_output.DType,
                                      {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
                                      expert_output.Device);

    std::vector<int> reverse_send_elem, reverse_recv_elem;
    ep_state.build_reverse_a2a_elem_splits(hidden_size, reverse_send_elem, reverse_recv_elem);
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(unsorted_output.Data),
                                  reinterpret_cast<std::byte*>(combined.Data),
                                  reverse_send_elem.data(),
                                  reverse_recv_elem.data(),
                                  elem_sz,
                                  exec.mRunState.MainStream);
    mBufferPool.release(unsorted_ptr, unsorted_bytes);

    // 3. Optional: undo LLEP send reorder.
    if (!ep_state.llep_send_reorder_gpu) {
        exec.store_tensor(op.outputs[0], combined);
        return;
    }
    Tensor reordered = apply_llep_inverse_reorder(exec,
                                                  ep_state,
                                                  combined,
                                                  hidden_size,
                                                  elem_sz,
                                                  ep_state_mut.llep_combined_gpu,
                                                  ep_state_mut.llep_combined_bytes);
    exec.store_tensor(op.outputs[0], reordered);
}

// ============================================================================
// combine_backward
// ============================================================================

void EPStrategy::combine_backward(CompiledExecutor& exec, const CompiledOp& op) {
    Tensor& d_combined = exec.resolve_tensor(op.inputs[0]);

    const int ep_size = op.attrs.ep_size;
    const int hidden_size = static_cast<int>(d_combined.Sizes[1]);
    if (ep_size <= 1 || !exec.mComm || !exec.mComm->ep_enabled()) {
        Tensor& d_expert = exec.ensure_output_tensor(op.outputs[0]);
        d_expert = d_combined;
        exec.store_tensor(op.outputs[0], d_expert);
        return;
    }

    const int layer_idx = resolve_layer_idx(op, {"d_", "saved."});
    const int input_total_send = static_cast<int>(d_combined.Sizes[0]);
    const int default_key = ep_state_key(layer_idx, exec.mInReplay);

    auto sel = select_ep_state_by_total(mEpStates,
                                        layer_idx,
                                        mOptions.EPSize > 1,
                                        input_total_send,
                                        &EpLayerState::total_send,
                                        default_key);
    if (!sel.found) {
        throw std::runtime_error("ep_combine_backward: no EP state for layer " + std::to_string(layer_idx));
    }
    const auto& ep_state = mEpStates[sel.ep_key];
    auto& ep_state_mut = mEpStates[sel.ep_key];
    if (ep_state.total_send != input_total_send) {
        std::ostringstream oss;
        oss << "ep_combine_backward: EP state/input mismatch at layer " << layer_idx << " (selected_key=" << sel.ep_key
            << ", state.total_send=" << ep_state.total_send << ", input.rows=" << input_total_send << ")";
        throw std::runtime_error(oss.str());
    }

    // Replicated-input EP (dp_size=1): each EP rank sees the same token gradient,
    // scale by 1/ep_size so expert-path magnitude matches EP=1.
    if (exec.mComm && exec.mComm->dp_size() == 1 && ep_size > 1) {
        const float inv_ep = 1.0f / static_cast<float>(ep_size);
        const int nelem = static_cast<int>(d_combined.nelem());
        if (d_combined.DType == ETensorDType::BF16) {
            moe_scale_forward(d_combined.get<nv_bfloat16>(),
                              d_combined.get<nv_bfloat16>(),
                              inv_ep,
                              nelem,
                              exec.mRunState.MainStream);
        } else {
            moe_scale_forward(d_combined.get<float>(),
                              d_combined.get<float>(),
                              inv_ep,
                              nelem,
                              exec.mRunState.MainStream);
        }
    }

    const int elem_sz = static_cast<int>(get_dtype_size(d_combined.DType));

    // 0. If LLEP was active, redo the send reorder on gradients before A2A.
    const Tensor* d_a2a_input = &d_combined;
    void* d_reordered_ptr = nullptr;
    std::size_t d_reordered_bytes = 0;
    Tensor d_reordered;

    if (ep_state.llep_send_reorder_gpu) {
        d_reordered_bytes = static_cast<std::size_t>(ep_state.total_send) * hidden_size * elem_sz;
        d_reordered_ptr = mBufferPool.acquire(d_reordered_bytes);
        d_reordered = make_raw_tensor(d_reordered_ptr,
                                      d_combined.DType,
                                      {static_cast<long>(ep_state.total_send), static_cast<long>(hidden_size)},
                                      d_combined.Device);
        permute_tokens_dtyped(d_reordered,
                              d_combined,
                              static_cast<int*>(ep_state.llep_send_reorder_gpu),
                              ep_state.total_send,
                              ep_state.total_send,
                              hidden_size,
                              1,
                              exec.mRunState.MainStream);
        d_a2a_input = &d_reordered;
    }

    // 1. Forward A2A (same direction as dispatch forward).
    const std::size_t bwd_unsorted_bytes = static_cast<std::size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    void* d_recv_unsorted_ptr = mBufferPool.acquire(bwd_unsorted_bytes);
    Tensor d_recv_unsorted = make_raw_tensor(d_recv_unsorted_ptr,
                                             d_combined.DType,
                                             {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
                                             d_combined.Device);

    std::vector<int> send_elem, recv_elem;
    ep_state.build_forward_a2a_elem_splits(hidden_size, send_elem, recv_elem);
    exec.mComm->all_to_all_single(reinterpret_cast<const std::byte*>(d_a2a_input->Data),
                                  reinterpret_cast<std::byte*>(d_recv_unsorted.Data),
                                  send_elem.data(),
                                  recv_elem.data(),
                                  elem_sz,
                                  exec.mRunState.MainStream);
    if (d_reordered_ptr) mBufferPool.release(d_reordered_ptr, d_reordered_bytes);

    // 2. Re-sort by local expert (same permutation as dispatch forward).
    const std::size_t sorted_need = static_cast<std::size_t>(ep_state.total_recv) * hidden_size * elem_sz;
    alloc_or_resize(ep_state_mut.combine_bwd_sorted_gpu, ep_state_mut.combine_bwd_sorted_bytes, sorted_need);
    Tensor d_expert_sorted = make_raw_tensor(ep_state_mut.combine_bwd_sorted_gpu,
                                             d_combined.DType,
                                             {static_cast<long>(ep_state.total_recv), static_cast<long>(hidden_size)},
                                             d_combined.Device);
    permute_tokens_dtyped(d_expert_sorted,
                          d_recv_unsorted,
                          static_cast<int*>(ep_state.send_order_gpu),
                          ep_state.total_recv,
                          ep_state.total_recv,
                          hidden_size,
                          1,
                          exec.mRunState.MainStream);
    mBufferPool.release(d_recv_unsorted_ptr, bwd_unsorted_bytes);

    exec.store_tensor(op.outputs[0], d_expert_sorted);
}

}  // namespace ep
