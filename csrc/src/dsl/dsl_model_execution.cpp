// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model execution functions (forward, backward, validation, run state allocation).

#include "dsl/dsl_model.h"
#include "dsl/dsl_model_internal.h"
#include "dsl/dsl_runtime.h"
#include "dsl/graph_executor.h"
#include "dsl/graph_executor_helpers.h"

namespace dsl {
class DslParamStore;
}  // namespace dsl
#include "dsl/compiled_ops_helpers.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <fmt/format.h>
#include <cuda_runtime_api.h>

#include "kernels/kernels.h"
#include "modules/forward_hooks.h"
#include "modules/backward_hooks.h"
#include "modules/fp8_scaling_state.h"
#include "modules/lora/lora_utils.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_weights_manager.h"
#include "modules/optimizers/adamw_8bit.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace {
inline bool lora_fwd_nan_trace_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_NAN_TRACE");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled == 1;
}

inline bool lora_qkv_guard_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("SUROGATE_LORA_QKV_GUARD");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled == 1;
}

inline int lora_qkv_guard_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_LORA_QKV_GUARD_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

inline int lora_qkv_guard_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char* env = std::getenv("SUROGATE_LORA_QKV_GUARD_LIMIT");
        limit = env ? std::atoi(env) : 8;
    }
    return limit;
}

inline bool lora_qkv_guard_should_log(int layer_idx) {
    if (!lora_qkv_guard_enabled()) return false;
    const int target = lora_qkv_guard_layer();
    if (target >= 0 && target != layer_idx) return false;
    static std::atomic<int> counter{0};
    const int limit = lora_qkv_guard_limit();
    if (limit <= 0) return false;
    const int idx = counter.fetch_add(1);
    return idx < limit;
}

inline bool lora_b_guard_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("SUROGATE_LORA_B_GUARD");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled == 1;
}

inline int lora_fwd_nan_trace_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

inline int lora_fwd_nan_trace_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_LIMIT");
        limit = env ? std::atoi(env) : 8;
    }
    return limit;
}

inline int lora_fwd_nan_trace_samples() {
    static int samples = -1;
    if (samples < 0) {
        const char* env = std::getenv("SUROGATE_LORA_FWD_TRACE_SAMPLES");
        samples = env ? std::atoi(env) : 8;
        if (samples < 1) samples = 1;
    }
    return samples;
}

inline bool lora_fwd_nan_trace_should_log(int layer_idx) {
    if (!lora_fwd_nan_trace_enabled()) return false;
    const int target = lora_fwd_nan_trace_layer();
    if (target >= 0 && target != layer_idx) return false;
    static std::atomic<int> counter{0};
    const int limit = lora_fwd_nan_trace_limit();
    if (limit <= 0) return false;
    const int idx = counter.fetch_add(1);
    return idx < limit;
}

inline bool lora_fwd_nan_trace_can_copy(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status == cudaStreamCaptureStatusNone;
}

inline float lora_to_float(float v) { return v; }
inline float lora_to_float(nv_bfloat16 v) { return __bfloat162float(v); }
inline float lora_to_float(half v) { return __half2float(v); }

template <typename T>
inline void lora_trace_row(const char* tag,
                           int layer_idx,
                           const Tensor& t,
                           long row_idx,
                           long col_offset,
                           int samples,
                           cudaStream_t stream) {
    if (!t.Data || samples <= 0) return;
    if (!lora_fwd_nan_trace_can_copy(stream)) return;

    const long stride = t.Sizes[t.Rank - 1];
    const long rows = [&]() {
        long r = 1;
        for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
        return r;
    }();
    if (row_idx < 0 || row_idx >= rows) return;
    if (col_offset < 0 || col_offset >= stride) return;
    const int count = std::min<long>(samples, stride - col_offset);
    const long elem_offset = row_idx * stride + col_offset;

    std::vector<T> host(count);
    const std::size_t bytes = (std::size_t)count * sizeof(T);
    CUDA_CHECK(cudaMemcpyAsync(host.data(),
                               t.Data + elem_offset * sizeof(T),
                               bytes,
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int nan_count = 0;
    int inf_count = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    if (!host.empty()) {
        float v0 = lora_to_float(host[0]);
        min_val = v0;
        max_val = v0;
        for (int i = 0; i < count; ++i) {
            float v = lora_to_float(host[i]);
            if (std::isnan(v)) nan_count++;
            if (std::isinf(v)) inf_count++;
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }

    fprintf(stderr,
            "[LORA_FWD_NAN] layer=%d tag=%s dtype=%s row=%ld offset=%ld samples=%d nan=%d inf=%d min=%g max=%g vals=%g,%g,%g,%g\n",
            layer_idx,
            tag,
            dtype_to_str(t.DType),
            row_idx,
            col_offset,
            count,
            nan_count,
            inf_count,
            min_val,
            max_val,
            count > 0 ? lora_to_float(host[0]) : 0.0f,
            count > 1 ? lora_to_float(host[1]) : 0.0f,
            count > 2 ? lora_to_float(host[2]) : 0.0f,
            count > 3 ? lora_to_float(host[3]) : 0.0f);
}

inline void lora_trace_tensor(const char* tag,
                              int layer_idx,
                              const Tensor& t,
                              long col_offset,
                              int samples,
                              cudaStream_t stream) {
    if (t.DType == ETensorDType::BF16) {
        lora_trace_row<nv_bfloat16>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<nv_bfloat16>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
    if (t.DType == ETensorDType::FP16) {
        lora_trace_row<half>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<half>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
    if (t.DType == ETensorDType::FP32) {
        lora_trace_row<float>(tag, layer_idx, t, 0, col_offset, samples, stream);
        const long rows = [&]() {
            long r = 1;
            for (int i = 0; i < t.Rank - 1; ++i) r *= t.Sizes[i];
            return r;
        }();
        if (rows > 1) {
            lora_trace_row<float>(tag, layer_idx, t, rows / 2, col_offset, samples, stream);
        }
        return;
    }
}
}  // namespace

namespace dsl {

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        mExecutor->forward(inputs, position_ids, comm, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);
    if (qlora_enabled() && micro_step == 0 && mQLoRAProvider) {
        mQLoRAProvider->invalidate_cache();
    }

    // Store micro_step for dropout seed computation (needed by backward pass)
    mLoRARunState->micro_step = micro_step;
    mLoRARunState->is_training = true;

    auto hook = [this, micro_step](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            // seed = base_seed + layer_idx * 1000000 + proj_type * 100000 + micro_step * 10000
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                const bool qkv_guard = lora_qkv_guard_should_log(layer_idx) &&
                                       lora_fwd_nan_trace_can_copy(stream);
                const bool b_guard = lora_b_guard_enabled() && lora_fwd_nan_trace_can_copy(stream);
                auto guard_check = [&](const char* proj,
                                       long offset,
                                       const modules::LoRALayerWeights<Tensor>& lora_w,
                                       bool pre_nan,
                                       long pre_row,
                                       float pre_min,
                                       float pre_max) {
                    if (!qkv_guard) return;
                    long post_row = -1;
                    float post_min = 0.0f;
                    float post_max = 0.0f;
                    const bool post_nan = find_first_nan_row(acts.qkv, &post_row, &post_min, &post_max);
                    if (!post_nan || pre_nan) return;
                    long ln1_row = -1;
                    float ln1_min = 0.0f;
                    float ln1_max = 0.0f;
                    long a_row = -1;
                    float a_min = 0.0f;
                    float a_max = 0.0f;
                    long b_row = -1;
                    float b_min = 0.0f;
                    float b_max = 0.0f;
                    const bool ln1_nan = find_first_nan_row(acts.ln1, &ln1_row, &ln1_min, &ln1_max);
                    const bool a_nan = find_first_nan_row(lora_w.A, &a_row, &a_min, &a_max);
                    const bool b_nan = find_first_nan_row(lora_w.B, &b_row, &b_min, &b_max);
                    std::cerr << fmt::format(
                        "[LORA_QKV_GUARD] layer={} micro={} proj={} offset={} qkv_ptr={} pre_row={} pre_min={} pre_max={} "
                        "post_row={} post_min={} post_max={} ln1_nan={} A_nan={} B_nan={} ln1_ptr={} A_ptr={} B_ptr={}\n",
                        layer_idx,
                        micro_step,
                        proj ? proj : "?",
                        offset,
                        static_cast<const void*>(acts.qkv.Data),
                        pre_row,
                        pre_min,
                        pre_max,
                        post_row,
                        post_min,
                        post_max,
                        ln1_nan ? 1 : 0,
                        a_nan ? 1 : 0,
                        b_nan ? 1 : 0,
                        static_cast<const void*>(acts.ln1.Data),
                        static_cast<const void*>(lora_w.A.Data),
                        static_cast<const void*>(lora_w.B.Data));
                };
                auto b_guard_check = [&](const char* proj, const modules::LoRALayerWeights<Tensor>& lora_w, const char* stage) {
                    if (!b_guard || !lora_w.B.Data) return;
                    modules::LoRABGuardSample cached;
                    if (!modules::fetch_lora_b_guard_sample(lora_w.B.Data, layer_idx, cached)) {
                        return;
                    }
                    std::vector<float> cur_vals;
                    if (!copy_tensor_sample_offset_as_f32(lora_w.B, 0, 8, cur_vals)) {
                        return;
                    }
                    float max_diff = 0.0f;
                    bool cur_nan = false;
                    for (std::size_t i = 0; i < cached.vals.size(); ++i) {
                        const float cur = i < cur_vals.size() ? cur_vals[i] : 0.0f;
                        if (std::isnan(cur) || std::isinf(cur)) {
                            cur_nan = true;
                        }
                        max_diff = std::max(max_diff, std::abs(cur - cached.vals[i]));
                    }
                    if (max_diff > 0.0f || cur_nan) {
                        std::cerr << fmt::format(
                            "[LORA_B_GUARD] layer={} micro={} proj={} stage={} ptr={} cached_tag={} max_diff={} cur_nan={} "
                            "cached={:.6g},{:.6g},{:.6g},{:.6g} cur={:.6g},{:.6g},{:.6g},{:.6g}\n",
                            layer_idx,
                            micro_step,
                            proj ? proj : "?",
                            stage ? stage : "?",
                            static_cast<const void*>(lora_w.B.Data),
                            cached.tag.empty() ? "<none>" : cached.tag,
                            max_diff,
                            cur_nan ? 1 : 0,
                            cached.vals[0], cached.vals[1], cached.vals[2], cached.vals[3],
                            cur_vals.size() > 0 ? cur_vals[0] : 0.0f,
                            cur_vals.size() > 1 ? cur_vals[1] : 0.0f,
                            cur_vals.size() > 2 ? cur_vals[2] : 0.0f,
                            cur_vals.size() > 3 ? cur_vals[3] : 0.0f);
                        const auto b_base = reinterpret_cast<std::uintptr_t>(lora_w.B.Data);
                        const auto b_end = b_base + static_cast<std::uintptr_t>(lora_w.B.bytes());
                        const auto qkv_base = reinterpret_cast<std::uintptr_t>(acts.qkv.Data);
                        const auto qkv_end = qkv_base + static_cast<std::uintptr_t>(acts.qkv.bytes());
                        const bool overlap = (b_base < qkv_end) && (qkv_base < b_end);
                        std::size_t gap = 0;
                        if (!overlap) {
                            gap = b_base < qkv_base
                                  ? static_cast<std::size_t>(qkv_base - b_end)
                                  : static_cast<std::size_t>(b_base - qkv_end);
                        }
                        std::cerr << fmt::format(
                            "[LORA_B_GUARD_RANGE] layer={} micro={} proj={} stage={} b_base={} b_bytes={} "
                            "qkv_base={} qkv_bytes={} overlap={} gap={}\n",
                            layer_idx,
                            micro_step,
                            proj ? proj : "?",
                            stage ? stage : "?",
                            static_cast<const void*>(lora_w.B.Data),
                            lora_w.B.bytes(),
                            static_cast<const void*>(acts.qkv.Data),
                            acts.qkv.bytes(),
                            overlap ? 1 : 0,
                            gap);
                        if (mAllocator) {
                            mAllocator->debug_log_allocation_for_ptr(lora_w.B.Data, "lora_b_guard");
                            mAllocator->debug_log_allocation_for_ptr(acts.qkv.Data, "lora_b_guard_qkv");
                        }
                    }
                };

                // Projection types: 0=Q, 1=K, 2=V, 3=O, 4=Up, 5=Gate, 6=Down
                if (lora_block.attention.q.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    long pre_row = -1;
                    float pre_min = 0.0f;
                    float pre_max = 0.0f;
                    const bool pre_nan = qkv_guard
                                         ? find_first_nan_row(acts.qkv, &pre_row, &pre_min, &pre_max)
                                         : false;
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_Q", layer_idx, acts.qkv, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_Q", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_Q", layer_idx, lora_block.attention.q->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_Q", layer_idx, lora_block.attention.q->B, 0, samples, stream);
                    }
                    b_guard_check("Q", lora_block.attention.q.value(), "pre");
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(0), is_training,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    guard_check("Q", 0, lora_block.attention.q.value(), pre_nan, pre_row, pre_min, pre_max);
                    b_guard_check("Q", lora_block.attention.q.value(), "post");
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_Q", layer_idx, acts.qkv, 0, samples, stream);
                    }
                }
                if (lora_block.attention.k.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    const long k_offset = (long)Hq * Hs;
                    long pre_row = -1;
                    float pre_min = 0.0f;
                    float pre_max = 0.0f;
                    const bool pre_nan = qkv_guard
                                         ? find_first_nan_row(acts.qkv, &pre_row, &pre_min, &pre_max)
                                         : false;
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_K", layer_idx, acts.qkv, k_offset, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_K", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_K", layer_idx, lora_block.attention.k->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_K", layer_idx, lora_block.attention.k->B, 0, samples, stream);
                    }
                    b_guard_check("K", lora_block.attention.k.value(), "pre");
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(1), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    guard_check("K", k_offset, lora_block.attention.k.value(), pre_nan, pre_row, pre_min, pre_max);
                    b_guard_check("K", lora_block.attention.k.value(), "post");
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_K", layer_idx, acts.qkv, k_offset, samples, stream);
                    }
                }
                if (lora_block.attention.v.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    const long v_offset = (long)(Hq + Hkv) * Hs;
                    long pre_row = -1;
                    float pre_min = 0.0f;
                    float pre_max = 0.0f;
                    const bool pre_nan = qkv_guard
                                         ? find_first_nan_row(acts.qkv, &pre_row, &pre_min, &pre_max)
                                         : false;
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_V", layer_idx, acts.qkv, v_offset, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_V", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_V", layer_idx, lora_block.attention.v->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_V", layer_idx, lora_block.attention.v->B, 0, samples, stream);
                    }
                    b_guard_check("V", lora_block.attention.v.value(), "pre");
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(2), is_training,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    guard_check("V", v_offset, lora_block.attention.v.value(), pre_nan, pre_row, pre_min, pre_max);
                    b_guard_check("V", lora_block.attention.v.value(), "post");
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_V", layer_idx, acts.qkv, v_offset, samples, stream);
                    }
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(3), is_training,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(4), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(5), is_training,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, dropout, get_dropout_seed(6), is_training,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };

    mExecutor->forward_with_hook(inputs, position_ids, comm, micro_step, hook);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto hook = [this](int layer_idx, cudaStream_t stream, modules::ForwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float scaling = mLoRAConfig->scaling();

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                // Validation: no dropout (is_training=false)
                if (lora_block.attention.q.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_Q", layer_idx, acts.qkv, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_Q", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_Q", layer_idx, lora_block.attention.q->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_Q", layer_idx, lora_block.attention.q->B, 0, samples, stream);
                    }
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hq * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_Q", layer_idx, acts.qkv, 0, samples, stream);
                    }
                }
                if (lora_block.attention.k.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    const long k_offset = (long)Hq * Hs;
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_K", layer_idx, acts.qkv, k_offset, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_K", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_K", layer_idx, lora_block.attention.k->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_K", layer_idx, lora_block.attention.k->B, 0, samples, stream);
                    }
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_K", layer_idx, acts.qkv, k_offset, samples, stream);
                    }
                }
                if (lora_block.attention.v.has_value()) {
                    const bool trace = lora_fwd_nan_trace_should_log(layer_idx);
                    const long v_offset = (long)(Hq + Hkv) * Hs;
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_PRE_V", layer_idx, acts.qkv, v_offset, samples, stream);
                        lora_trace_tensor("LORA_QKV_IN_V", layer_idx, acts.ln1, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_A_V", layer_idx, lora_block.attention.v->A, 0, samples, stream);
                        lora_trace_tensor("LORA_QKV_B_V", layer_idx, lora_block.attention.v->B, 0, samples, stream);
                    }
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, Hkv * Hs, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                    if (trace) {
                        const int samples = lora_fwd_nan_trace_samples();
                        lora_trace_tensor("LORA_QKV_POST_V", layer_idx, acts.qkv, v_offset, samples, stream);
                    }
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, Hq * Hs, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, C, D, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                    mLoRARunState->intermediate, mLoRARunState->slice,
                                                    scaling, 0.0f, 0, false,
                                                    B * T, D, C, rank,
                                                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };

    return mExecutor->validate_with_hook(inputs, position_ids, targets, comm, micro_step, hook);
}

void DslModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward called before allocate_run_state()");
    }

    if (!lora_enabled()) {
        mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
        return;
    }

    ensure_lora_run_state(comm, (int)inputs.Sizes[0], (int)inputs.Sizes[1]);

    auto& rs = *mRunState;
    cudaStream_t main_stream = rs.MainStream;

    mLoRAGrads->start_micro_step(main_stream, micro_step, grad_accum_steps);

    auto hook = [this, &comm](int layer_idx, bool accumulate, cudaStream_t stream, modules::BackwardHookPoint point, void* context) {
        (void)context;
        const auto& cfg = mModelConfig;
        auto& rs = *mRunState;
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = mLoRAConfig->rank;
        const float dropout = mLoRAConfig->dropout;
        const bool is_training = mLoRARunState->is_training;
        const int micro_step = mLoRARunState->micro_step;
        const int qlora_trace = env_int("SUROGATE_QLORA_TRACE", 0);
        const int qlora_trace_layer = env_int("SUROGATE_QLORA_TRACE_LAYER", -1);
        const int qlora_trace_micro = env_int("SUROGATE_QLORA_TRACE_MICRO", -1);
        const int qlora_trace_limit = env_int("SUROGATE_QLORA_TRACE_LIMIT", 8);
        static std::atomic<int> qlora_trace_count{0};

        auto tensor_shape = [](const Tensor& t) {
            std::ostringstream oss;
            oss << "[";
            for (int i = 0; i < t.Rank; ++i) {
                if (i) oss << ",";
                oss << t.Sizes[i];
            }
            oss << "]";
            return oss.str();
        };

        auto log_tensor_sample = [&](const Tensor& t, const char* tag) {
            if (!tag) tag = "<unnamed>";
            if (!t.Data) {
                std::cerr << fmt::format("[QLORA_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr=<null>\n",
                                         layer_idx, micro_step, tag, static_cast<int>(t.DType), tensor_shape(t));
                return;
            }
            std::vector<float> sample;
            sample.reserve(8);
            if (!copy_tensor_sample_offset_as_f32(t, 0, 8, sample) || sample.empty()) {
                std::cerr << fmt::format("[QLORA_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} sample=<unavailable>\n",
                                         layer_idx, micro_step, tag, static_cast<int>(t.DType), tensor_shape(t), (const void*)t.Data);
                return;
            }
            float min_v = sample[0];
            float max_v = sample[0];
            float mean_v = 0.0f;
            for (float v : sample) {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                mean_v += v;
            }
            mean_v /= static_cast<float>(sample.size());
            std::ostringstream vals;
            vals << sample[0];
            for (std::size_t i = 1; i < sample.size(); ++i) {
                vals << "," << sample[i];
            }
            std::cerr << fmt::format("[QLORA_TRACE] layer={} micro={} tag={} dtype={} shape={} ptr={} min={:.6g} max={:.6g} mean={:.6g} vals={}\n",
                                     layer_idx, micro_step, tag, static_cast<int>(t.DType), tensor_shape(t),
                                     (const void*)t.Data, min_v, max_v, mean_v, vals.str());
        };

        auto should_trace = [&](modules::BackwardHookPoint hook_point) {
            if (!qlora_enabled() || !qlora_trace) {
                return false;
            }
            if (internal::stream_is_capturing(stream)) {
                return false;
            }
            if (qlora_trace_layer >= 0 && qlora_trace_layer != layer_idx) {
                return false;
            }
            if (qlora_trace_micro >= 0 && qlora_trace_micro != micro_step) {
                return false;
            }
            if (qlora_trace_limit > 0) {
                const int idx = qlora_trace_count.fetch_add(1);
                if (idx >= qlora_trace_limit) {
                    return false;
                }
            }
            std::cerr << fmt::format("[QLORA_TRACE_BEGIN] layer={} micro={} hook={}\n",
                                     layer_idx, micro_step, static_cast<int>(hook_point));
            return true;
        };

        // Helper to compute unique dropout seed per layer and projection type
        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return mLoRARunState->dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(micro_step) * 10000u;
        };

        auto& lora_block = mLoRAWeights->get_block(layer_idx, stream);

        switch (point) {
            case modules::BackwardHookPoint::AfterMLPDownBackward: {
                if (!lora_block.mlp.down.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.mlp.down.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                const bool trace = should_trace(point);

                // Projection type 6 = Down
                const unsigned int dropout_seed = get_dropout_seed(6);

                modules::detail::backward_lora_layer(
                    lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                    da.d_swiglu,
                    da.d_res_ffn, 0,
                    a.swiglu,
                    lora_block.mlp.down->A, lora_block.mlp.down->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, D, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);

                if (trace) {
                    log_tensor_sample(da.d_swiglu, "d_swiglu");
                    log_tensor_sample(da.d_res_ffn, "d_res_ffn");
                    log_tensor_sample(a.swiglu, "swiglu");
                    log_tensor_sample(lora_block.mlp.down->A, "lora_down.A");
                    log_tensor_sample(lora_block.mlp.down->B, "lora_down.B");
                }
            } break;
            case modules::BackwardHookPoint::AfterMLPUpBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                const bool trace = should_trace(point);

                // Get ln2 input: either from stored activation or recompute from residual stream
                // LN2 input is residual_att = res_ffn[L-1] + att_out[L]
                Tensor ln2_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        const std::string ln2_name = "blocks[" + std::to_string(layer_idx) + "].ln2_weight";
                        Tensor& ln2_weight = mParams->get(ln2_name);
                        // Prefer using recomputed residual_att from simplified_acts.
                        // Fallback uses res_ffn[L-1], but this path shouldn't be hit
                        // when recompute is working correctly since residual_att should be populated.
                        Tensor ln2_residual;
                        if (a.residual_att.Data) {
                            ln2_residual = a.residual_att;
                        } else if (layer_idx == 0) {
                            ln2_residual = rs.non_block_activations().encoded;
                        } else {
                            // Ensure residual is fetched when offloading is enabled
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            }
                            ln2_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln2_input = recompute_lora_rmsnorm(*mLoRARunState, ln2_residual, ln2_weight,
                                                          mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln2_input = a.ln2;
                    }
                } else {
                    ln2_input = a.ln2;
                }
                if (mOptions.recompute_enabled() && !internal::stream_is_capturing(stream)) {
                    long nan_row = -1;
                    float nan_min = 0.0f;
                    float nan_max = 0.0f;
                    if (find_first_nan_row(ln2_input, &nan_row, &nan_min, &nan_max)) {
                        std::cerr << fmt::format(
                            "[LORA_RECOMP_NAN] layer={} micro={} tag=ln2_input row={} min={} max={} using_saved={}\n",
                            layer_idx, micro_step, nan_row, nan_min, nan_max,
                            a.ln2.Data ? 1 : 0);
                        if (a.ln2.Data) {
                            ln2_input = a.ln2;
                        }
                    }
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

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

                if (!dA_up.Data && !dA_gate.Data) break;

                // Projection types: 4=Up, 5=Gate
                modules::detail::backward_lora_mlp_up_gate_fused(
                    dA_up, dB_up,
                    dA_gate, dB_gate,
                    da.d_ln2,
                    da.d_mlp_up,
                    ln2_input,
                    lora_up, lora_gate,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(4), get_dropout_seed(5), is_training,
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

                if (trace) {
                    log_tensor_sample(da.d_ln2, "d_ln2");
                    log_tensor_sample(da.d_mlp_up, "d_mlp_up");
                    log_tensor_sample(ln2_input, "ln2_input");
                    log_tensor_sample(a.ln2, "ln2_saved");
                    if (lora_block.mlp.up.has_value()) {
                        log_tensor_sample(lora_block.mlp.up->A, "lora_up.A");
                        log_tensor_sample(lora_block.mlp.up->B, "lora_up.B");
                    }
                    if (lora_block.mlp.gate.has_value()) {
                        log_tensor_sample(lora_block.mlp.gate->A, "lora_gate.A");
                        log_tensor_sample(lora_block.mlp.gate->B, "lora_gate.B");
                    }
                }
            } break;
            case modules::BackwardHookPoint::AfterAttnOutBackward: {
                if (!lora_block.attention.o.has_value()) break;

                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;
                if (!lora_grads.attention.o.has_value()) break;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                const bool trace = should_trace(point);

                // Projection type 3 = O
                const unsigned int dropout_seed = get_dropout_seed(3);

                modules::detail::backward_lora_layer(
                    lora_grads.attention.o->A, lora_grads.attention.o->B,
                    da.d_att,
                    da.d_res_att, 0,
                    a.att,
                    lora_block.attention.o->A, lora_block.attention.o->B,
                    mLoRAConfig->scaling(),
                    dropout, dropout_seed, is_training,
                    mLoRARunState->intermediate, mLoRARunState->slice,
                    B * T, Hq * Hs, C, rank, lora_accum,
                    rs.CublasLtHandle, rs.CuBlasWorkspace, stream);

                if (trace) {
                    log_tensor_sample(da.d_att, "d_att");
                    log_tensor_sample(da.d_res_att, "d_res_att");
                    log_tensor_sample(a.att, "att");
                    log_tensor_sample(lora_block.attention.o->A, "lora_o.A");
                    log_tensor_sample(lora_block.attention.o->B, "lora_o.B");
                }
            } break;
            case modules::BackwardHookPoint::AfterQKVBackward: {
                bool lora_accum = false;
                auto& lora_grads = mLoRAGrads->get_block_full(layer_idx, stream, comm, lora_accum);
                lora_accum = lora_accum || accumulate;

                auto& a = rs.simplified_acts(layer_idx);
                auto& da = rs.simplified_grads(layer_idx);
                const bool trace = should_trace(point);

                // Get ln1 input: either from stored activation or recompute from residual
                // LN1 input is res_ffn[L-1] (output of previous layer) for layer L > 0
                Tensor ln1_input;
                if (mOptions.recompute_enabled()) {
                    if (mLoRARunState && mLoRARunState->recompute_ln.Data) {
                        const std::string ln1_name = "blocks[" + std::to_string(layer_idx) + "].ln1_weight";
                        Tensor& ln1_weight = mParams->get(ln1_name);
                        Tensor ln1_residual;
                        if (layer_idx == 0) {
                            ln1_residual = rs.non_block_activations().encoded;
                        } else {
                            // Ensure residual is fetched when offloading is enabled
                            if (rs.has_residual_offloading()) {
                                rs.fetch_residual(layer_idx - 1, rs.side_stream());
                            }
                            ln1_residual = rs.get_residual(layer_idx - 1, stream);
                        }
                        ln1_input = recompute_lora_rmsnorm(*mLoRARunState, ln1_residual, ln1_weight,
                                                          mModelConfig.RmsNormEps, B, T, C, stream);
                    } else {
                        ln1_input = a.ln1;
                    }
                } else {
                    ln1_input = a.ln1;
                }
                if (mOptions.recompute_enabled() && !internal::stream_is_capturing(stream)) {
                    long nan_row = -1;
                    float nan_min = 0.0f;
                    float nan_max = 0.0f;
                    if (find_first_nan_row(ln1_input, &nan_row, &nan_min, &nan_max)) {
                        std::cerr << fmt::format(
                            "[LORA_RECOMP_NAN] layer={} micro={} tag=ln1_input row={} min={} max={} using_saved={}\n",
                            layer_idx, micro_step, nan_row, nan_min, nan_max,
                            a.ln1.Data ? 1 : 0);
                        if (a.ln1.Data) {
                            ln1_input = a.ln1;
                        }
                    }
                }

                // Prepare gradient tensors (use empty tensor if projection not enabled)
                Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

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

                if (!dA_q.Data && !dA_k.Data && !dA_v.Data) break;

                const int trace_qkv = env_int("SUROGATE_LORA_QKV_TRACE", 0);
                const int trace_layer = env_int("SUROGATE_LORA_QKV_TRACE_LAYER", -1);
                const int nan_abort = env_int("SUROGATE_LORA_NAN_ABORT", 0);
                if (trace_qkv && !internal::stream_is_capturing(stream) &&
                    (trace_layer < 0 || trace_layer == layer_idx)) {
                    auto log_nan = [&](const Tensor& t, const char* tag) {
                        if (!t.Data) return;
                        long row = -1;
                        float min_val = 0.0f;
                        float max_val = 0.0f;
                        if (!find_first_nan_row(t, &row, &min_val, &max_val)) {
                            return;
                        }
                        std::cerr << fmt::format("[LORA_QKV_NAN] layer={} tag={} row={} min={} max={} dtype={}\n",
                                                 layer_idx,
                                                 tag ? tag : "<unnamed>",
                                                 row,
                                                 min_val,
                                                 max_val,
                                                 static_cast<int>(t.DType));
                        if (nan_abort) {
                            throw std::runtime_error("LoRA QKV inputs contain NaN/Inf");
                        }
                    };
                    log_nan(da.d_qkv, "d_qkv");
                    log_nan(da.d_ln1, "d_ln1");
                    log_nan(ln1_input, "ln1_input");
                    if (lora_q.has_value()) {
                        log_nan(lora_q.A, "lora_q.A");
                        log_nan(lora_q.B, "lora_q.B");
                    }
                    if (lora_k.has_value()) {
                        log_nan(lora_k.A, "lora_k.A");
                        log_nan(lora_k.B, "lora_k.B");
                    }
                    if (lora_v.has_value()) {
                        log_nan(lora_v.A, "lora_v.A");
                        log_nan(lora_v.B, "lora_v.B");
                    }
                }

                // Projection types: 0=Q, 1=K, 2=V
                modules::detail::backward_lora_qkv_fused(
                    dA_q, dB_q,
                    dA_k, dB_k,
                    dA_v, dB_v,
                    da.d_ln1,
                    da.d_qkv,
                    ln1_input,
                    lora_q, lora_k, lora_v,
                    mLoRAConfig->scaling(),
                    dropout, get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
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

                mLoRAGrads->notify_block(layer_idx, stream, comm);

                if (trace) {
                    log_tensor_sample(da.d_ln1, "d_ln1");
                    log_tensor_sample(da.d_qkv, "d_qkv");
                    log_tensor_sample(ln1_input, "ln1_input");
                    log_tensor_sample(a.ln1, "ln1_saved");
                    if (lora_block.attention.q.has_value()) {
                        log_tensor_sample(lora_block.attention.q->A, "lora_q.A");
                        log_tensor_sample(lora_block.attention.q->B, "lora_q.B");
                    }
                    if (lora_block.attention.k.has_value()) {
                        log_tensor_sample(lora_block.attention.k->A, "lora_k.A");
                        log_tensor_sample(lora_block.attention.k->B, "lora_k.B");
                    }
                    if (lora_block.attention.v.has_value()) {
                        log_tensor_sample(lora_block.attention.v->A, "lora_v.A");
                        log_tensor_sample(lora_block.attention.v->B, "lora_v.B");
                    }
                }
            } break;
            default:
                break;
        }
    };

    mExecutor->backward_with_hook(inputs, targets, comm, grad_accum_steps, micro_step, hook);

    mLoRAGrads->end_micro_step(main_stream, comm);
    // Extend the base-model BackwardDone event to include LoRA gradient reductions.
    internal::record_event_if_not_capturing(rs.BackwardDone, main_stream);
}

void DslModel::allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T,
                                  bool allocate_optimizer) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mOptions = options;
    if (qlora_enabled() && mQLoRAConfig.is_fp4()) {
        mOptions.UseCudaGraphs = false;
    }
    const std::size_t dummy_stack_bytes = 1024ULL * 1024ULL * 1024ULL * 1024ULL;  // 1TB dummy stack
    const ActivationLayoutIR* layout = mModule->activation_layout.has_value()
                                           ? &*mModule->activation_layout
                                           : nullptr;
    mRunState = std::make_unique<DslRunState>(*mConfig, mOptions, B, T, mAllocator, lora_enabled(),
                                              dummy_stack_bytes, /*allocate_stack=*/false, layout);
    mRunState->WorldSize = comm.world_size();
    if (mParams) {
        mParams->set_default_stream(mRunState->MainStream);
        if (mQLoRAProvider) {
            mParams->set_qlora_provider(mQLoRAProvider.get());
        }
    }

    const long base_size = static_cast<long>(mRunState->Stack.max_utilization());
    long moe_extra = 0;
    if (mModelConfig.NumExperts > 0) {
        const long moe_intermediate = (mModelConfig.MoeIntermediateSize > 0)
                                          ? mModelConfig.MoeIntermediateSize
                                          : mModelConfig.IntermediateSize;
        const long hidden = mModelConfig.HiddenSize;
        const long num_experts = mModelConfig.NumExperts;
        const long top_k = std::max(1, mModelConfig.NumExpertsPerTok);
        const long dtype_bytes = 2;  // BF16 bytes (matches modular sizing heuristic)
        const long up_factor = mModelConfig.mlp_up_factor();
        const long expert_gate_up_tp = num_experts * up_factor * moe_intermediate * hidden * dtype_bytes;
        const long expert_down_tp = num_experts * moe_intermediate * hidden * dtype_bytes;
        const long permuted_tokens = 2L * B * T * top_k * hidden * dtype_bytes;
        moe_extra = expert_gate_up_tp + expert_down_tp + permuted_tokens;
    }
    ETensorDType act_dtype = mOptions.ModelType.value_or(mConfig->DType);
    if (is_fp8_dtype(act_dtype)) {
        act_dtype = ETensorDType::BF16;
    }
    const long dtype_bytes = static_cast<long>(get_dtype_size(act_dtype));
    const long BT = static_cast<long>(B) * static_cast<long>(T);
    const long C = mModelConfig.HiddenSize;
    const long QKV = mModelConfig.head_size() * (mModelConfig.NumQueryHeads + 2 * mModelConfig.NumKeyValHeads);
    const long MUp = static_cast<long>(mModelConfig.mlp_up_rows());
    const long extra_tmp = std::max({BT * C, BT * QKV, BT * MUp}) * dtype_bytes;
    long attn_fallback_bytes = 0;
    const bool lora_stack_tight = lora_enabled();
    const long safety_floor = lora_stack_tight ? (32L * 1024 * 1024) : (64L * 1024 * 1024);
    const long safety_bytes = std::max(safety_floor, base_size / 8);
    const long base_multiplier = lora_stack_tight ? 1L : 2L;
    long required_size = std::max(1024L * 1024,
                                  base_size * base_multiplier + moe_extra + safety_bytes + extra_tmp + attn_fallback_bytes);
    const long slack_bytes = lora_stack_tight ? (128L * 1024 * 1024) : (512L * 1024 * 1024);
    required_size += slack_bytes;  // extra slack for unmodeled temps
    long moe_stack_slack = 0;
    if (mModelConfig.NumExperts > 0) {
        moe_stack_slack = 2048L * 1024 * 1024;  // MoE backward temps can spike beyond simulated high-water mark
    }
    if (const char* env = std::getenv("SUROGATE_STACK_SLACK_MB")) {
        const long mb = std::max(0L, std::atol(env));
        moe_stack_slack = std::max(moe_stack_slack, mb * 1024 * 1024);
    }
    required_size += moe_stack_slack;
    const long min_stack_base = lora_stack_tight ? (512L * 1024 * 1024) : (3L * 1024 * 1024 * 1024);
    const long min_stack_bytes = min_stack_base + attn_fallback_bytes + moe_stack_slack;
    required_size = std::max(required_size, min_stack_bytes);  // Full fine-tune keeps 3GB+fallback; LoRA can use tighter floor.
    const auto high_mark = mRunState->Stack.get_high_mark();
    Tensor stack_buffer = mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {required_size});
    mRunState->set_stack_buffer(std::move(stack_buffer), high_mark);
    comm.barrier();

    // Configure gradient manager for multi-GPU overlapped reduction
    if (mGrads && comm.world_size() > 1) {
        DslGradStoreConfig grad_config;
        grad_config.num_shards = comm.world_size();
        grad_config.shard_idx = comm.rank();
        grad_config.shard_gradients = mOptions.ShardGradients;  // ZeRO-2
        grad_config.use_all_to_all_reduce = mOptions.UseAllToAllReduce;
        grad_config.num_layers = mModelConfig.NumLayers;
        mGrads->configure(grad_config);
    }

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    mExecutor = std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, mOptions, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
    }

    // Wire weight manager for streaming/sharding
    if (mWeightManager) {
        if (auto* exec = dynamic_cast<GraphExecutor*>(mExecutor.get())) {
            exec->set_weight_manager(mWeightManager.get());
        }
    }

    if (lora_enabled()) {
        ensure_lora_run_state(comm, B, T);
        mExecutor->set_lora_state(mLoRAConfig ? &*mLoRAConfig : nullptr,
                                  mLoRAWeights.get(), mLoRAGrads.get(), mLoRARunState.get());
    }

    if (allocate_optimizer) {
        if (lora_enabled()) {
            if (!mLoRAAdamW8BitState) {
                mLoRAAdamW8BitState = std::make_unique<modules::LoRAAdamW8BitState>();
            }
            if (!mLoRAAdamW8BitState->quantiles1.Data) {
                mLoRAAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles1", {256});
                mLoRAAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "lora_adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mLoRAAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        } else {
            if (!mAdamW8BitState) {
                mAdamW8BitState = std::make_unique<AdamW8BitState>();
            }
            if (!mAdamW8BitState->quantiles1.Data) {
                mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
                mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
                std::vector<float> h_q1(256), h_q2(256);
                create_adamw8bit_quantiles1(h_q1.data());
                create_adamw8bit_quantiles2(h_q2.data());
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }
}

void DslModel::zero_grads(cudaStream_t stream) {
    if (mGrads) {
        mGrads->zero_all(stream);
    }
}

void DslModel::set_internal_graphs_enabled(bool enabled) {
    if (mExecutor) {
        mExecutor->set_internal_graphs_enabled(enabled);
    }
}

bool DslModel::internal_graphs_enabled() const {
    return mExecutor ? mExecutor->internal_graphs_enabled() : false;
}

}  // namespace dsl
