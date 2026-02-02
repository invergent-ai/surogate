// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Compiled operation dispatch for DSL Graph executor.

#include "dsl/compiled_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dsl/compiled_ops_helpers.h"
#include "dsl/dsl_runtime.h"
#include "dsl/dsl_weight_manager.h"
#include "dsl/graph_executor_helpers.h"
#include "dsl/graph_executor_utils.h"
#include "dsl/op_shape_signatures.h"
#include "modules/backward_hooks.h"
#include "modules/forward_hooks.h"
#include "modules/fp8_scaling_config.h"
#include "modules/lora/lora_config.h"
#include "modules/lora/lora_weights_manager.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "modules/moe/moe_types.h"
#include "recipes/recipe.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"

namespace dsl {

// MoE compact weight information (moved out of anonymous namespace for split files)
MoeCompactInfo build_moe_compact_info(const int* expert_offsets_dev,
                                      int num_experts,
                                      int weight_experts,
                                      cudaStream_t stream,
                                      int layer_idx,
                                      const char* tag) {
    MoeCompactInfo info;
    if (!expert_offsets_dev || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.host_offsets.resize(num_experts + 1, 0);
    CUDA_CHECK(cudaMemcpyAsync(info.host_offsets.data(),
                               expert_offsets_dev,
                               static_cast<std::size_t>(num_experts + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (info.host_offsets[e + 1] > info.host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        static int warn_count = 0;
        if (warn_count < 8) {
            fprintf(stderr,
                    "[MOE_COMPACT_WARN] tag=%s layer=%d num_active=%d weight_experts=%d num_experts=%d\n",
                    tag, layer_idx, info.num_active, weight_experts, num_experts);
            warn_count++;
        }
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}

MoeCompactInfo build_moe_compact_info_from_host(const int* host_offsets,
                                                int num_experts,
                                                int weight_experts,
                                                int layer_idx,
                                                const char* tag) {
    MoeCompactInfo info;
    if (!host_offsets || num_experts <= 0 || weight_experts <= 0) {
        return info;
    }
    info.weight_is_compact = (weight_experts != num_experts);
    if (!info.weight_is_compact) {
        return info;
    }

    info.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            info.active_experts.push_back(e);
        }
    }
    info.num_active = static_cast<int>(info.active_experts.size());

    if (weight_experts > 0 && info.num_active != weight_experts) {
        static int warn_count = 0;
        if (warn_count < 8) {
            fprintf(stderr,
                    "[MOE_COMPACT_WARN] tag=%s layer=%d num_active=%d weight_experts=%d num_experts=%d\n",
                    tag, layer_idx, info.num_active, weight_experts, num_experts);
            warn_count++;
        }
        if (info.num_active > weight_experts) {
            info.active_experts.resize(weight_experts);
            info.num_active = weight_experts;
        }
    }

    return info;
}

int moe_trace_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_MOE_TRACE_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}


int moe_trace_limit() {
    static int limit = -2;
    if (limit == -2) {
        const char* env = std::getenv("SUROGATE_MOE_TRACE_LIMIT");
        limit = env ? std::atoi(env) : 8;
    }
    return limit;
}

bool should_trace_moe_layer(int layer_idx, int& counter) {
    const int target = moe_trace_layer();
    if (target < 0 || layer_idx != target) {
        return false;
    }
    const int limit = moe_trace_limit();
    if (limit >= 0 && counter >= limit) {
        return false;
    }
    counter++;
    return true;
}


// Helper functions that need to be accessible from split op files
void log_cuda_ptr_attr(const char* tag, const void* ptr, int layer_idx, const char* name) {
    if (!ptr) {
        fprintf(stderr, "[%s] layer=%d name=%s ptr=null\n", tag, layer_idx, name ? name : "<unnamed>");
        return;
    }
    cudaPointerAttributes attr{};
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[%s] layer=%d name=%s ptr=%p attr_err=%s\n",
                tag, layer_idx, name ? name : "<unnamed>", ptr, cudaGetErrorString(err));
        cudaGetLastError();
        return;
    }
#if CUDART_VERSION >= 10000
    fprintf(stderr,
            "[%s] layer=%d name=%s ptr=%p type=%d device=%d\n",
            tag, layer_idx, name ? name : "<unnamed>", ptr,
            static_cast<int>(attr.type), attr.device);
#else
    fprintf(stderr,
            "[%s] layer=%d name=%s ptr=%p type=%d device=%d\n",
            tag, layer_idx, name ? name : "<unnamed>", ptr,
            static_cast<int>(attr.memoryType), attr.device);
#endif
}

bool copy_tensor_sample_as_f32(const Tensor& t, std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (count == 0 || !t.Data) {
        return false;
    }
    switch (t.DType) {
    case ETensorDType::FP32:
        cudaMemcpy(out.data(), t.Data, count * sizeof(float), cudaMemcpyDeviceToHost);
        return true;
    case ETensorDType::BF16: {
        std::vector<nv_bfloat16> tmp(count);
        cudaMemcpy(tmp.data(), t.Data, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    case ETensorDType::FP16: {
        std::vector<half> tmp(count);
        cudaMemcpy(tmp.data(), t.Data, count * sizeof(half), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    default:
        return false;
    }
}

double sample_mean_abs(const Tensor& t, std::size_t count) {
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, count, vals)) {
        return 0.0;
    }
    double sum = 0.0;
    for (float v : vals) {
        sum += std::abs(static_cast<double>(v));
    }
    return vals.empty() ? 0.0 : (sum / static_cast<double>(vals.size()));
}

bool copy_tensor_sample_offset_as_f32(const Tensor& t, std::size_t elem_offset,
                                      std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (count == 0 || !t.Data) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (elem_offset + count > total) {
        return false;
    }
    const std::size_t byte_offset = elem_offset * get_dtype_size(t.DType);
    const std::byte* base = static_cast<const std::byte*>(t.Data) + byte_offset;
    switch (t.DType) {
    case ETensorDType::FP32:
        cudaMemcpy(out.data(), base, count * sizeof(float), cudaMemcpyDeviceToHost);
        return true;
    case ETensorDType::BF16: {
        std::vector<nv_bfloat16> tmp(count);
        cudaMemcpy(tmp.data(), base, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    case ETensorDType::FP16: {
        std::vector<half> tmp(count);
        cudaMemcpy(tmp.data(), base, count * sizeof(half), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }
    default:
        return false;
    }
}

void log_moe_gate_up_weight_sample(const char* tag,
                                   int layer_idx,
                                   int micro_step,
                                   DslParamStore& weights,
                                   const modules::ModelConfig& cfg) {
    if (layer_idx != 2) {
        return;
    }
    static int trace_count = 0;
    if (trace_count >= 64) {
        return;
    }
    const std::string wname = "blocks[" + std::to_string(layer_idx) + "].experts_gate_up";
    if (!weights.has(wname)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_MISS] tag=%s micro_step=%d layer=%d name=%s\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str());
        trace_count++;
        return;
    }
    Tensor& gw = weights.get(wname);
    const int hidden_size = static_cast<int>(cfg.HiddenSize);
    const int intermediate_size = (cfg.MoeIntermediateSize > 0)
        ? static_cast<int>(cfg.MoeIntermediateSize)
        : static_cast<int>(cfg.IntermediateSize);
    const int expert_id = 122;
    const std::size_t stride = static_cast<std::size_t>(2 * intermediate_size) *
                               static_cast<std::size_t>(hidden_size);
    const std::size_t offset = stride * static_cast<std::size_t>(expert_id);
    if (gw.nelem() < static_cast<long>(offset + 1)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_OOR] tag=%s micro_step=%d layer=%d name=%s nelem=%ld offset=%zu\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str(),
                static_cast<long>(gw.nelem()), offset);
        trace_count++;
        return;
    }
    const std::size_t sample = std::min<std::size_t>(stride, 512);
    std::vector<float> wvals;
    if (!copy_tensor_sample_offset_as_f32(gw, offset, sample, wvals)) {
        fprintf(stderr,
                "[MOE_W_SAMPLE_FAIL] tag=%s micro_step=%d layer=%d name=%s\n",
                tag ? tag : "<null>", micro_step, layer_idx, wname.c_str());
        trace_count++;
        return;
    }
    int nan = 0;
    float max_abs = 0.0f;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (float v : wvals) {
        if (std::isnan(v) || std::isinf(v)) {
            nan++;
            continue;
        }
        max_abs = std::max(max_abs, std::fabs(v));
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
    }
    fprintf(stderr,
            "[MOE_W_SAMPLE] tag=%s micro_step=%d layer=%d expert=%d nan=%d min=%.6f max=%.6f max_abs=%.6f dtype=%s ptr=%p\n",
            tag ? tag : "<null>", micro_step, layer_idx, expert_id, nan,
            min_v, max_v, max_abs, dtype_to_str(gw.DType), static_cast<void*>(gw.Data));
    trace_count++;
}

bool build_selective_info_from_offsets(const int* host_offsets,
                                       int num_experts,
                                       modules::SelectiveExpertInfo& selection) {
    if (!host_offsets || num_experts <= 0) {
        selection.reset();
        return false;
    }
    selection.reset();
    selection.enabled = true;
    selection.num_total = num_experts;
    selection.expert_to_compact.assign(num_experts, -1);
    selection.active_experts.reserve(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        if (host_offsets[e + 1] > host_offsets[e]) {
            selection.expert_to_compact[e] = static_cast<int>(selection.active_experts.size());
            selection.active_experts.push_back(e);
        }
    }
    selection.num_active = static_cast<int>(selection.active_experts.size());
    if (selection.num_active == 0) {
        selection.enabled = false;
        return false;
    }
    return true;
}

bool refresh_moe_experts_if_needed(int layer_idx,
                                   const int* host_offsets,
                                   int num_experts,
                                   DslParamStore& weights,
                                   cudaStream_t stream) {
    if (layer_idx < 0) {
        return false;
    }
    auto* provider = weights.qlora_provider();
    if (!provider) {
        return false;
    }
    modules::SelectiveExpertInfo selection;
    if (!build_selective_info_from_offsets(host_offsets, num_experts, selection)) {
        return false;
    }
    const bool refreshed = provider->refresh_moe_experts(layer_idx, selection, stream);
    static int moe_refresh_trace = 0;
    if (layer_idx == 2 && moe_refresh_trace < 8) {
        const int first = selection.active_experts.empty() ? -1 : selection.active_experts[0];
        fprintf(stderr,
                "[MOE_REFRESH] layer=%d num_active=%d first=%d refreshed=%d\n",
                layer_idx, selection.num_active, first, refreshed ? 1 : 0);
        moe_refresh_trace++;
    }
    return refreshed;
}

bool copy_tensor_token_sample_as_f32(const Tensor& t, long token_idx,
                                     std::size_t count, std::vector<float>& out) {
    out.assign(count, 0.0f);
    if (!t.Data || token_idx < 0 || t.Rank < 2) {
        return false;
    }
    std::size_t row_width = 0;
    if (t.Rank == 2) {
        row_width = static_cast<std::size_t>(t.Sizes[1]);
    } else if (t.Rank >= 3) {
        row_width = 1;
        for (int i = 2; i < t.Rank; ++i) {
            row_width *= static_cast<std::size_t>(t.Sizes[i]);
        }
    }
    if (row_width == 0) {
        return false;
    }
    const std::size_t elem_offset = static_cast<std::size_t>(token_idx) * row_width;
    return copy_tensor_sample_offset_as_f32(t, elem_offset, count, out);
}

bool sample_has_nan_or_inf(const std::vector<float>& vals) {
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            return true;
        }
    }
    return false;
}

bool tensor_sample_has_nan_or_inf(const Tensor& t, long token_idx) {
    std::vector<float> vals(4);
    const bool ok = copy_tensor_token_sample_as_f32(t, token_idx, vals.size(), vals);
    if (!ok) {
        return false;
    }
    return sample_has_nan_or_inf(vals);
}

bool log_tensor_scalar_at(const char* tag, const Tensor& t, long elem_idx) {
    if (!tag || !t.Data || elem_idx < 0) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (static_cast<std::size_t>(elem_idx) >= total) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, static_cast<std::size_t>(elem_idx), 1, vals)) {
        return false;
    }
    const float v = vals.empty() ? 0.0f : vals[0];
    fprintf(stderr, "[%s] idx=%ld val=%.6f dtype=%s\n", tag, elem_idx, v, dtype_to_str(t.DType));
    return true;
}

bool log_tensor_token_row_stats(const char* tag, const Tensor& t, long token_idx) {
    if (!tag || !t.Data || token_idx < 0) {
        return false;
    }
    std::size_t row_width = 0;
    if (t.Rank == 1) {
        if (static_cast<std::size_t>(token_idx) >= static_cast<std::size_t>(t.nelem())) {
            return false;
        }
        row_width = 1;
    } else if (t.Rank == 2) {
        row_width = static_cast<std::size_t>(t.Sizes[1]);
    } else if (t.Rank >= 3) {
        row_width = 1;
        for (int i = 2; i < t.Rank; ++i) {
            row_width *= static_cast<std::size_t>(t.Sizes[i]);
        }
    }
    if (row_width == 0) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_token_sample_as_f32(t, token_idx, row_width, vals)) {
        return false;
    }
    std::size_t nan = 0;
    std::size_t inf = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    bool has_finite = false;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const float v = vals[i];
        if (std::isnan(v)) {
            nan++;
            continue;
        }
        if (std::isinf(v)) {
            inf++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
            max_idx = i;
        }
    }
    const double mean_abs = vals.empty() ? 0.0 : (sum_abs / static_cast<double>(vals.size()));
    fprintf(stderr,
            "[%s] token=%ld n=%zu nan=%zu inf=%zu min=%.6f max=%.6f max_abs=%.6f max_idx=%zu mean_abs=%.6f\n",
            tag, token_idx, vals.size(), nan, inf, min_val, max_val, max_abs, max_idx, mean_abs);
    return true;
}

std::size_t tensor_row_width(const Tensor& t);

bool log_tensor_row_stats(const char* tag, const Tensor& t, long row_idx) {
    if (!tag || !t.Data || row_idx < 0) {
        return false;
    }
    const std::size_t row_width = tensor_row_width(t);
    if (row_width == 0) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t elem_offset = static_cast<std::size_t>(row_idx) * row_width;
    if (elem_offset + row_width > total) {
        return false;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, elem_offset, row_width, vals)) {
        return false;
    }
    std::size_t nan = 0;
    std::size_t inf = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float max_abs = 0.0f;
    double sum_abs = 0.0;
    bool has_finite = false;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < vals.size(); ++i) {
        const float v = vals[i];
        if (std::isnan(v)) {
            nan++;
            continue;
        }
        if (std::isinf(v)) {
            inf++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
            max_idx = i;
        }
    }
    const double mean_abs = vals.empty() ? 0.0 : (sum_abs / static_cast<double>(vals.size()));
    fprintf(stderr,
            "[%s] row=%ld n=%zu nan=%zu inf=%zu min=%.6f max=%.6f max_abs=%.6f max_idx=%zu mean_abs=%.6f\n",
            tag, row_idx, vals.size(), nan, inf, min_val, max_val, max_abs, max_idx, mean_abs);
    return true;
}

std::size_t tensor_row_width(const Tensor& t) {
    if (t.Rank <= 1) {
        return static_cast<std::size_t>(t.nelem());
    }
    std::size_t row_width = 1;
    for (int i = 1; i < t.Rank; ++i) {
        row_width *= static_cast<std::size_t>(t.Sizes[i]);
    }
    return row_width;
}

bool tensor_row_has_nan_or_inf(const Tensor& t, long token_idx, float* out_min, float* out_max) {
    if (!t.Data) {
        return false;
    }
    const std::size_t row_width = tensor_row_width(t);
    if (row_width == 0) {
        return false;
    }
    std::vector<float> vals(row_width);
    if (!copy_tensor_token_sample_as_f32(t, token_idx, row_width, vals)) {
        return false;
    }
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    bool has_nan = false;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            has_nan = true;
            continue;
        }
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }
    if (out_min) {
        *out_min = std::isfinite(min_val) ? min_val : 0.0f;
    }
    if (out_max) {
        *out_max = std::isfinite(max_val) ? max_val : 0.0f;
    }
    return has_nan;
}

bool find_first_nan_row(const Tensor& t, long* out_row, float* out_min, float* out_max) {
    if (t.Rank < 1) {
        return false;
    }
    const long rows = static_cast<long>(t.Sizes[0]);
    for (long r = 0; r < rows; ++r) {
        if (tensor_row_has_nan_or_inf(t, r, out_min, out_max)) {
            if (out_row) {
                *out_row = r;
            }
            return true;
        }
    }
    return false;
}

void log_tensor_mag(const char* tag,
                    int layer_idx,
                    const std::string& name,
                    const Tensor& t,
                    std::size_t max_elems) {
    static int mag_log_count = 0;
    if (!t.Data || mag_log_count >= 64) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            continue;
        }
        const float av = std::fabs(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_MAG] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu l2=%.6e max_abs=%.6e mean_abs=%.6e dtype=%s\n",
            tag, layer_idx, name.c_str(), t.Data, n, total,
            std::sqrt(sum_sq), max_abs, mean_abs, dtype_to_str(t.DType));
    mag_log_count++;
}

void log_tensor_mag_unbounded(const char* tag,
                              int layer_idx,
                              const std::string& name,
                              const Tensor& t,
                              std::size_t max_elems) {
    if (!t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    double sum_sq = 0.0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        if (std::isnan(v) || std::isinf(v)) {
            continue;
        }
        const float av = std::fabs(v);
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_MAG_EX] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu l2=%.6e max_abs=%.6e mean_abs=%.6e dtype=%s\n",
            tag, layer_idx, name.c_str(), t.Data, n, total,
            std::sqrt(sum_sq), max_abs, mean_abs, dtype_to_str(t.DType));
}

// Global state for QKV gradient tracking (shared across split op files)
std::vector<std::byte*> g_qkv_dA_ptr_by_layer;
std::vector<int> g_qkv_dA_micro_by_layer;

void log_nan_sample(const char* tag,
                    int layer_idx,
                    const std::string& name,
                    const Tensor& t,
                    long token_idx) {
    static int nan_log_count = 0;
    static bool first_logged = false;
    std::vector<float> vals(4);
    const bool ok = copy_tensor_token_sample_as_f32(t, token_idx, vals.size(), vals);
    if (!ok || !sample_has_nan_or_inf(vals)) {
        return;
    }
    const char* prefix = first_logged ? "[NAN_DETECT]" : "[FIRST_NAN]";
    if (!first_logged) {
        first_logged = true;
    }
    if (nan_log_count < 100) {
        fprintf(stderr,
                "%s tag=%s layer=%d name=%s token=%ld ptr=%p vals=%.6f,%.6f,%.6f,%.6f\n",
                prefix, tag, layer_idx, name.c_str(), token_idx, t.Data,
                vals[0], vals[1], vals[2], vals[3]);
        nan_log_count++;
    }
}

void log_tensor_stats(const char* tag,
                      int layer_idx,
                      const std::string& name,
                      const Tensor& t,
                      std::size_t max_elems) {
    static int stats_log_count = 0;
    if (stats_log_count >= 32 || !t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (n > 0) ? (sum_abs / static_cast<double>(n)) : 0.0;
    fprintf(stderr,
            "[TENSOR_STATS] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu nan=%zu inf=%zu max_abs=%.6f mean_abs=%.6f\n",
            tag, layer_idx, name.c_str(), t.Data, n, total, nan_count, inf_count, max_abs, mean_abs);
    stats_log_count++;
}

void log_tensor_stats_ex(const char* tag,
                         int layer_idx,
                         const std::string& name,
                         const Tensor& t,
                         std::size_t max_elems,
                         bool force) {
    static int stats_log_count = 0;
    static int forced_log_count = 0;
    if (!t.Data) {
        return;
    }
    if (!force && stats_log_count >= 16) {
        return;
    }
    if (force && forced_log_count >= 16) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    const std::size_t n = std::min(max_elems, total);
    if (n == 0) {
        return;
    }
    std::vector<float> vals;
    if (!copy_tensor_sample_as_f32(t, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t finite_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    bool has_finite = false;
    for (float v : vals) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) {
                min_val = v;
            }
            if (v > max_val) {
                max_val = v;
            }
        }
        finite_count++;
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (finite_count > 0) ? (sum_abs / static_cast<double>(finite_count)) : 0.0;
    fprintf(stderr,
            "[TENSOR_STATS_EX] tag=%s layer=%d name=%s ptr=%p n=%zu total=%zu nan=%zu inf=%zu "
            "min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
            tag, layer_idx, name.c_str(), t.Data, n, total, nan_count, inf_count,
            min_val, max_val, max_abs, mean_abs);
    if (force) {
        forced_log_count++;
    } else {
        stats_log_count++;
    }
}

void log_tensor_sample_stats(const char* tag,
                             const Tensor& t,
                             std::size_t elem_offset,
                             std::size_t sample_elems) {
    if (!t.Data) {
        return;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (total == 0 || sample_elems == 0 || elem_offset >= total) {
        return;
    }
    const std::size_t n = std::min(sample_elems, total - elem_offset);
    std::vector<float> vals;
    if (!copy_tensor_sample_offset_as_f32(t, elem_offset, n, vals)) {
        return;
    }
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t finite_count = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    bool has_finite = false;
    for (float v : vals) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (std::isinf(v)) {
            inf_count++;
            continue;
        }
        if (!has_finite) {
            min_val = v;
            max_val = v;
            has_finite = true;
        } else {
            if (v < min_val) {
                min_val = v;
            }
            if (v > max_val) {
                max_val = v;
            }
        }
        finite_count++;
        const float av = std::fabs(v);
        sum_abs += static_cast<double>(av);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const double mean_abs = (finite_count > 0) ? (sum_abs / static_cast<double>(finite_count)) : 0.0;
    fprintf(stderr,
            "[TENSOR_SAMPLE_STATS] tag=%s offset=%zu n=%zu total=%zu nan=%zu inf=%zu "
            "min=%.6f max=%.6f max_abs=%.6f mean_abs=%.6f\n",
            tag, elem_offset, n, total, nan_count, inf_count, min_val, max_val, max_abs, mean_abs);
}

float env_float(const char* name, float fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    float out = std::strtof(value, &end);
    if (end == value) {
        return fallback;
    }
    return out;
}

int env_int(const char* name, int fallback) {
    if (!name || !*name) {
        return fallback;
    }
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end = nullptr;
    long out = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(out);
}

namespace {  // Reopen anonymous namespace for internal helpers

}  // namespace

// ============================================================================
// Operation type conversion
// ============================================================================

const char* op_type_to_string(CompiledOpType type) {
    switch (type) {
        case CompiledOpType::Embedding: return "embedding";
        case CompiledOpType::Zeros: return "zeros";
        case CompiledOpType::FusedResidualRMSNorm: return "fused_residual_rmsnorm";
        case CompiledOpType::View: return "view";
        case CompiledOpType::Add: return "add";
        case CompiledOpType::Matmul: return "matmul";
        case CompiledOpType::MatmulBias: return "matmul_bias";
        case CompiledOpType::BiasAdd: return "bias_add";
        case CompiledOpType::SwiGLU: return "swiglu";
        case CompiledOpType::Silu: return "silu";
        case CompiledOpType::Mul: return "mul";
        case CompiledOpType::MatmulSwiGLU: return "matmul_swiglu";
        case CompiledOpType::QKVQKNormRoPE: return "qkv_qk_norm_rope";
        case CompiledOpType::RoPE: return "rope";
        case CompiledOpType::FlashAttention: return "flash_attention";
        case CompiledOpType::CrossEntropyLoss: return "cross_entropy_loss";
        case CompiledOpType::FusedLMHeadLoss: return "fused_lm_head_loss";
        // MoE forward
        case CompiledOpType::MoESoftmax: return "moe_softmax";
        case CompiledOpType::MoESigmoid: return "moe_sigmoid";
        case CompiledOpType::MoETopK: return "moe_topk";
        case CompiledOpType::MoEPermute: return "moe_permute";
        case CompiledOpType::MoEGroupedGemmGateUp: return "moe_grouped_gemm_gate_up";
        case CompiledOpType::MoEGroupedGemmDown: return "moe_grouped_gemm_down";
        case CompiledOpType::MoEUnpermute: return "moe_unpermute";
        // Backward
        case CompiledOpType::ViewBackward: return "view_backward";
        case CompiledOpType::AddBackward: return "add_backward";
        case CompiledOpType::MatmulBackward: return "matmul_backward";
        case CompiledOpType::BiasAddBackward: return "bias_add_backward";
        case CompiledOpType::SwiGLUBackward: return "swiglu_backward";
        case CompiledOpType::SiluBackward: return "silu_backward";
        case CompiledOpType::MulBackward: return "mul_backward";
        case CompiledOpType::MatmulSwiGLUBackward: return "matmul_swiglu_backward";
        case CompiledOpType::RoPEBackward: return "rope_backward";
        case CompiledOpType::QKVQKNormRoPEBackward: return "qkv_qk_norm_rope_backward";
        case CompiledOpType::FlashAttentionBackward: return "flash_attention_backward";
        case CompiledOpType::ZerosBackward: return "zeros_backward";
        case CompiledOpType::FusedResidualRMSNormBackward: return "fused_residual_rmsnorm_backward";
        case CompiledOpType::EmbeddingBackward: return "embedding_backward";
        case CompiledOpType::CrossEntropyLossBackward: return "cross_entropy_backward";
        case CompiledOpType::FusedLMHeadLossBackward: return "fused_lm_head_loss_backward";
        // MoE backward
        case CompiledOpType::MoESoftmaxBackward: return "moe_softmax_backward";
        case CompiledOpType::MoESigmoidBackward: return "moe_sigmoid_backward";
        case CompiledOpType::MoETopKBackward: return "moe_topk_backward";
        case CompiledOpType::MoEPermuteBackward: return "moe_permute_backward";
        case CompiledOpType::MoEGroupedGemmGateUpBackward: return "moe_grouped_gemm_gate_up_backward";
        case CompiledOpType::MoEGroupedGemmDownBackward: return "moe_grouped_gemm_down_backward";
        case CompiledOpType::MoEUnpermuteBackward: return "moe_unpermute_backward";
        case CompiledOpType::Unknown: return "unknown";
    }
    return "unknown";
}


// ============================================================================
// CompiledExecutor implementation
// ============================================================================

CompiledExecutor::CompiledExecutor(DslRunState& run_state,
                                   DslParamStore& weights,
                                   DslGradStore& grads,
                                   const modules::ModelConfig& config,
                                   const RuntimeOptions& options)
    : mRunState(run_state)
    , mWeights(weights)
    , mGrads(grads)
    , mConfig(config)
    , mOptions(options)
{}

CompiledExecutor::~CompiledExecutor() {
    // Free persistent GPU buffers
    if (mMoEExpertOffsetsGPU) {
        cudaFree(mMoEExpertOffsetsGPU);
        mMoEExpertOffsetsGPU = nullptr;
        mMoEExpertOffsetsGPUSize = 0;
    }

    // Free persistent MoE saved tensor buffers
    for (auto& [name, buffer] : mMoESavedBuffers) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
    mMoESavedBuffers.clear();
    mMoESavedSizes.clear();
}

void CompiledExecutor::set_lora_state(const modules::ModularLoRAConfig* config,
                                      modules::ModularLoRAWeightsManager* weights,
                                      modules::ModularLoRAGradsManager* grads,
                                      modules::LoRARunState* run_state) {
    mLoRAConfig = config;
    mLoRAWeights = weights;
    mLoRAGrads = grads;
    mLoRARunState = run_state;
}

void CompiledExecutor::set_weight_manager(DslWeightManager* weight_manager) {
    mWeightManager = weight_manager;
}

void CompiledExecutor::set_recipe(const recipes::Recipe* recipe) {
    mRecipe = recipe;
}

void CompiledExecutor::set_hook_context(void* context) {
    mHookContext = context;
}

void CompiledExecutor::set_recompute_fn(std::function<void(int, long, long, bool)> fn) {
    mRecomputeFn = std::move(fn);
}

void CompiledExecutor::set_recompute_enabled(bool enabled) {
    mRecomputeEnabled = enabled;
    mLastRecomputeLayer = -1;
    static int set_count = 0;
    if (set_count < 5) {
        fprintf(stderr, "[SET_RECOMPUTE_ENABLED] enabled=%d, mRecomputeFn=%p\n",
                enabled, mRecomputeFn ? (void*)1 : nullptr);
        set_count++;
    }
}

void CompiledExecutor::set_fp8_cache(std::unordered_map<std::string, FP8WeightCacheEntry>* cache) {
    mFP8Cache = cache;
}

void CompiledExecutor::set_fp4_cache(std::unordered_map<std::string, FP4WeightCacheEntry>* cache,
                                     std::unordered_map<std::string, FP4WeightCacheEntry>* cache_t) {
    mFP4Cache = cache;
    mFP4CacheT = cache_t;
}

void CompiledExecutor::set_saved_tensors(std::unordered_map<std::string, Tensor>* saved) {
    mSaved = saved;
}

void CompiledExecutor::set_save_list(const std::vector<std::string>* save_list) {
    mSaveList = save_list;
    mSaveSet.clear();
    if (save_list) {
        mSaveSet.insert(save_list->begin(), save_list->end());
    }
}

void CompiledExecutor::set_last_inputs_cpu(const Tensor* inputs_cpu) {
    mLastInputsCpu = inputs_cpu;
}

void CompiledExecutor::set_rng_seed_fn(std::function<unsigned int()> fn) {
    mRngSeedFn = std::move(fn);
}

const Tensor* CompiledExecutor::try_get_tensor(const std::string& name) const {
    auto it = mTensorMap.find(name);
    if (it == mTensorMap.end()) {
        return nullptr;
    }
    return &it->second;
}

void CompiledExecutor::save_moe_layer_tensors(int layer_idx) {
    // Copy MoE tensors from this layer to persistent storage before stack restore.
    // This allows stack memory to be reclaimed while preserving tensors for backward.
    if (mConfig.NumExperts == 0) {
        return;
    }

    // Build layer prefix pattern (e.g., "blocks[5].")
    std::string layer_prefix = "blocks[" + std::to_string(layer_idx) + "].";

    // Iterate through tensor map looking for MoE tensors from this layer
    for (auto& [name, tensor] : mTensorMap) {
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
        bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                              name.find("scatter_indices") != std::string::npos ||
                              name.find("routing_weights") != std::string::npos ||
                              name.find("routing_indices") != std::string::npos ||
                              name.find("router_") != std::string::npos ||
                              name.find("permuted") != std::string::npos ||
                              name.find("expert_") != std::string::npos);

        if (!is_moe_tensor || tensor.Data == nullptr) {
            continue;
        }

        const size_t bytes = tensor.bytes();
        if (bytes == 0) {
            continue;
        }

        // Allocate or resize persistent buffer if needed
        auto buf_it = mMoESavedBuffers.find(name);
        if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
            // Free old buffer if exists
            if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                CUDA_CHECK(cudaFree(buf_it->second));
            }
            // Allocate new buffer
            void* new_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
            mMoESavedBuffers[name] = new_buffer;
            mMoESavedSizes[name] = bytes;
        }

        // Copy data to persistent buffer
        void* dst_buffer = mMoESavedBuffers[name];
        CUDA_CHECK(cudaMemcpyAsync(dst_buffer, tensor.Data, bytes,
                                   cudaMemcpyDeviceToDevice, mRunState.MainStream));

        // Update tensor to point to persistent buffer (so backward finds it)
        tensor.Data = static_cast<std::byte*>(dst_buffer);
    }
}

void CompiledExecutor::save_tensors(const std::vector<std::string>& save_list) {
    if (!mSaved) {
        return;
    }

    // Recompute is only active when enabled in runtime options.
    // Do NOT use mRecomputeFn here: it's always set when the graph is compiled,
    // even for no-recompute runs, and would cause metadata-only saves.
    const bool recompute_enabled = mOptions.recompute_enabled();

    auto prefer_live_tensor = [&](const std::string& tensor_name) -> bool {
        if (!recompute_enabled || !mSlotRegistry) {
            return false;
        }
        // Use will_recompute which checks the recompute_policy
        // In FFT mode (!lora_only), tensors with lora_only policy should NOT prefer live
        // because they won't be recomputed and the live buffer may have stale data
        const bool lora_only_mode = mRunState.is_lora_only_mode();
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(tensor_name, layer_idx, field)) {
            return mSlotRegistry->will_recompute(strip_ssa_suffix(field), lora_only_mode);
        }
        const std::string base_name = strip_ssa_suffix(tensor_name);
        return mSlotRegistry->will_recompute(strip_ssa_suffix(tensor_name), lora_only_mode);
    };

    // Helper to copy tensor to persistent buffer when needed in recompute mode.
    // Returns true if tensor was copied to persistent storage, false if metadata-only save.
    auto save_tensor_with_policy = [&](const std::string& name, const Tensor& src,
                                        bool prefer_live) -> void {
        if (prefer_live) {
            // Save metadata only - will resolve from live buffer or recompute
            Tensor meta = src;
            meta.Data = nullptr;
            (*mSaved)[name] = meta;
        } else if (recompute_enabled && src.Data != nullptr) {
            // In recompute mode but tensor won't be recomputed (e.g., lora_only in FFT mode).
            // Copy data to persistent buffer since live buffers will be reused.
            const size_t bytes = src.bytes();
            auto buf_it = mMoESavedBuffers.find(name);
            if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
                if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                    CUDA_CHECK(cudaFree(buf_it->second));
                }
                void* new_buffer = nullptr;
                CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                mMoESavedBuffers[name] = new_buffer;
                mMoESavedSizes[name] = bytes;
            }
            void* dst_buffer = mMoESavedBuffers[name];
            CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
            Tensor saved_tensor;
            saved_tensor.DType = src.DType;
            saved_tensor.Rank = src.Rank;
            for (int i = 0; i < src.Rank; ++i) {
                saved_tensor.Sizes[i] = src.Sizes[i];
            }
            saved_tensor.Data = static_cast<std::byte*>(dst_buffer);
            (*mSaved)[name] = saved_tensor;
        } else {
            // Non-recompute mode: just store reference
            (*mSaved)[name] = src;
        }
    };

    for (const auto& name : save_list) {
        // First check the tensor map (intermediate tensors)
        auto it = mTensorMap.find(name);
        if (it != mTensorMap.end()) {
            const bool is_moe_tensor = (name.find("moe_") != std::string::npos ||
                                        name.find("scatter_indices") != std::string::npos ||
                                        name.find("routing_weights") != std::string::npos ||
                                        name.find("routing_indices") != std::string::npos ||
                                        name.find("router_probs") != std::string::npos ||
                                        name.find("router_logits") != std::string::npos ||
                                        name.find("permuted_input") != std::string::npos ||
                                        name.find("expert_") != std::string::npos);
            const bool prefer_live = prefer_live_tensor(name);

            // For MoE tensors, copy to persistent buffer to prevent buffer reuse corruption
            if (is_moe_tensor && mConfig.NumExperts > 0 && it->second.Data != nullptr) {
                if (prefer_live) {
                    // Recompute mode: store metadata only and recompute later.
                    save_tensor_with_policy(name, it->second, true);
                    continue;
                }
                const Tensor& src = it->second;
                const size_t bytes = src.bytes();

                // Allocate or resize persistent buffer if needed
                auto buf_it = mMoESavedBuffers.find(name);
                if (buf_it == mMoESavedBuffers.end() || mMoESavedSizes[name] < bytes) {
                    // Free old buffer if exists
                    if (buf_it != mMoESavedBuffers.end() && buf_it->second != nullptr) {
                        CUDA_CHECK(cudaFree(buf_it->second));
                    }
                    // Allocate new buffer
                    void* new_buffer = nullptr;
                    CUDA_CHECK(cudaMalloc(&new_buffer, bytes));
                    mMoESavedBuffers[name] = new_buffer;
                    mMoESavedSizes[name] = bytes;
                }

                // Copy data to persistent buffer
                void* dst_buffer = mMoESavedBuffers[name];
                CUDA_CHECK(cudaMemcpyAsync(dst_buffer, src.Data, bytes,
                                           cudaMemcpyDeviceToDevice, mRunState.MainStream));

                // Create tensor struct pointing to persistent buffer
                Tensor saved_tensor;
                saved_tensor.DType = src.DType;
                saved_tensor.Rank = src.Rank;
                for (int i = 0; i < src.Rank; ++i) {
                    saved_tensor.Sizes[i] = src.Sizes[i];
                }
                saved_tensor.Data = static_cast<std::byte*>(dst_buffer);

                (*mSaved)[name] = saved_tensor;
            } else {
                // Non-MoE tensor: use standard policy-based saving
                // DEBUG: Print res_ffn values being saved
                static int map_save_trace = 0;
                if (name.find("blocks[25].res_ffn") != std::string::npos && map_save_trace < 16) {
                    map_save_trace++;
                    cudaStreamSynchronize(mRunState.MainStream);
                    std::vector<float> vals(4);
                    cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                    bool pl = prefer_live;
                    fprintf(stderr, "[SAVE_MAP_RES_FFN] %s src.Data=%p vals=%.6f,%.6f,%.6f,%.6f prefer_live=%d\n",
                            name.c_str(), it->second.Data, vals[0], vals[1], vals[2], vals[3], pl ? 1 : 0);
                }
                save_tensor_with_policy(name, it->second, prefer_live);
            }
            continue;
        }

        // Check special tensors
        if (name == "token_ids") {
            save_tensor_with_policy(name, mRunState.Inputs, prefer_live_tensor(name));
            continue;
        }
        if (name == "position_ids") {
            save_tensor_with_policy(name, mRunState.PositionIDs, prefer_live_tensor(name));
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
            if (field == "ln1_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1_rstd, prefer_live);
            } else if (field == "ln2_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2_rstd, prefer_live);
            } else if (field == "q_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).q_rstd, prefer_live);
            } else if (field == "k_rstd") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).k_rstd, prefer_live);
            } else if (field == "lse") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).lse, prefer_live);
            } else if (field == "ln1" || field == "ln1_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln1, prefer_live);
            } else if (field == "ln2" || field == "ln2_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).ln2, prefer_live);
            } else if (field == "qkv") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).qkv, prefer_live);
            } else if (field == "qkv_rope") {
                // qkv_rope has RoPE applied - save it if available, otherwise fall back to qkv
                auto& acts = mRunState.simplified_acts(layer_idx);
                Tensor& src = acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
                save_tensor_with_policy(name, src, prefer_live);
            } else if (field == "qkv_flat") {
                // Save the flattened version for matmul backward shape resolution
                Tensor qkv = mRunState.simplified_acts(layer_idx).qkv;
                Tensor flat = view_tensor(qkv, {qkv.Sizes[0] * qkv.Sizes[1], qkv.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live);
            } else if (field == "att" || field == "att_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att, prefer_live);
            } else if (field == "att_out" || field == "att_out_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).att_out, prefer_live);
            } else if (field == "mlp_up" || field == "mlp_up_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_up, prefer_live);
            } else if (field == "swiglu") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).swiglu, prefer_live);
            } else if (field == "swiglu_flat") {
                Tensor swiglu = mRunState.simplified_acts(layer_idx).swiglu;
                Tensor flat = view_tensor(swiglu, {swiglu.Sizes[0] * swiglu.Sizes[1], swiglu.Sizes[2]});
                save_tensor_with_policy(name, flat, prefer_live);
            } else if (field == "mlp_down" || field == "mlp_down_flat") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).mlp_down, prefer_live);
            } else if (field == "res_att" || field == "residual_att") {
                save_tensor_with_policy(name, mRunState.simplified_acts(layer_idx).residual_att, prefer_live);
            } else if (field == "res_ffn" || field == "residual_ffn") {
                // res_ffn is computed dynamically (residual_att + mlp_down), check mTensorMap
                auto it = mTensorMap.find(name);
                if (it != mTensorMap.end()) {
                    // DEBUG: Print res_ffn values being saved for layer 25
                    static int save_trace = 0;
                    if (layer_idx == 25 && save_trace < 16) {
                        save_trace++;
                        cudaStreamSynchronize(mRunState.MainStream);
                        std::vector<float> vals(4);
                        cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                        fprintf(stderr, "[SAVE_RES_FFN] %s src.Data=%p vals=%.6f,%.6f,%.6f,%.6f prefer_live=%d\n",
                                name.c_str(), it->second.Data, vals[0], vals[1], vals[2], vals[3], prefer_live ? 1 : 0);
                    }
                    save_tensor_with_policy(name, it->second, prefer_live);
                } else {
                    throw std::runtime_error("CompiledExecutor: res_ffn tensor not found in map: " + name);
                }
            } else if (mWeights.has(name)) {
                (*mSaved)[name] = mWeights.get(name);
            } else {
                throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
            }
        } else if (name == "ln_final" || name == "xF") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final, prefer_live_tensor(name));
        } else if (name == "final_residual" || name == "residual_final") {
            save_tensor_with_policy(name, mRunState.get_final_residual(), prefer_live_tensor(name));
        } else if (name == "xF_flat") {
            // Save the flattened version for matmul backward
            Tensor ln_final = mRunState.non_block_activations().ln_final;
            Tensor flat = view_tensor(ln_final, {ln_final.Sizes[0] * ln_final.Sizes[1], ln_final.Sizes[2]});
            save_tensor_with_policy(name, flat, prefer_live_tensor(name));
        } else if (name == "ln_final_rstd") {
            save_tensor_with_policy(name, mRunState.non_block_activations().ln_final_rstd, prefer_live_tensor(name));
        } else if (name == "encoded" || name == "x0") {
            save_tensor_with_policy(name, mRunState.non_block_activations().encoded, prefer_live_tensor(name));
        } else if (name.find("rope_freqs") != std::string::npos || name.find("freq_cis") != std::string::npos) {
            save_tensor_with_policy(name, mRunState.non_block_activations().freq_cis, prefer_live_tensor(name));
        } else if (mWeights.has(name)) {
            (*mSaved)[name] = mWeights.get(name);
        } else {
            throw std::runtime_error("CompiledExecutor: cannot save tensor " + name);
        }
    }

    // For MoE models, copy expert_offsets data to persistent storage for backward pass
    // The original tensor is stack-allocated and will be freed before backward runs
    if (mConfig.NumExperts > 0) {
        auto it = mTensorMap.find("moe_expert_offsets");
        if (it != mTensorMap.end() && it->second.Data) {
            const Tensor& src = it->second;
            const int num_elements = static_cast<int>(src.nelem());
            mMoEExpertOffsetsData.resize(num_elements);
            CUDA_CHECK(cudaMemcpy(mMoEExpertOffsetsData.data(), src.Data,
                                  num_elements * sizeof(int), cudaMemcpyDeviceToHost));
            // Store metadata for reconstruction in backward
            mMoEExpertOffsets = src;  // Copy the tensor metadata (shape, dtype, etc.)
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
            // DEBUG: Log when element count mismatch
            int layer_idx = -1;
            std::string field;
            parse_block_param(name, layer_idx, field);
            if (layer_idx == 0) {
                fprintf(stderr, "[MAP_VIEW_FAIL] %s: nelem mismatch base=%ld vs shape=%ld\n",
                        name.c_str(), base.nelem(), shape_nelem(shape));
            }
            return nullptr;
        }
        auto [it, _] = mTensorMap.insert_or_assign(name, view_tensor(base, shape));
        return &it->second;
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
        if (field == "qkv" || field == "qkv_flat" || field == "qkv_biased") return map_view(acts.qkv);
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
        if (field == "res_ffn" || field == "residual_ffn") {
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

    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
                }
            }
        }
    }

    // DEBUG: Trace resolution for top-layer d_qkv_rope gradients
    static int d_qkv_rope_resolve_trace = 0;
    if (d_qkv_rope_resolve_trace < 8 &&
        ref.name.find("d_blocks[") != std::string::npos &&
        ref.name.find("qkv_rope") != std::string::npos &&
        ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
        fprintf(stderr,
                "[RESOLVE_D_QKV_ROPE] name=%s slot=%d shape_rank=%zu\n",
                ref.name.c_str(), static_cast<int>(ref.slot), ref.shape.size());
        d_qkv_rope_resolve_trace++;
    }

    // If shape is specified and this is a pre-allocated slot, we may need to create a view
    if (!ref.shape.empty() && ref.slot != TensorSlot::Mapped && ref.slot != TensorSlot::Saved &&
        ref.slot != TensorSlot::Parameter && ref.slot != TensorSlot::Temporary) {
        // Check if we already have a tensor in the map (e.g., from MoE temp allocation)
        auto it = mTensorMap.find(ref.name);
        // DEBUG: Trace mTensorMap lookup for top-layer d_ln1
        static int shape_trace_count = 0;
        if (shape_trace_count < 8 &&
            ref.name.find(".ln1") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_SHAPE] ref.name=%s, shape.size=%zu, found_in_map=%s, slot=%d\n",
                    ref.name.c_str(), ref.shape.size(),
                    (it != mTensorMap.end() && it->second.Data) ? "YES" : "NO",
                    static_cast<int>(ref.slot));
            if (it != mTensorMap.end() && it->second.Data) {
                cudaStreamSynchronize(mRunState.MainStream);
                std::vector<float> vals(4);
                cudaMemcpy(vals.data(), it->second.Data, 4 * sizeof(float), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[RESOLVE_SHAPE] d_ln1 ptr=%p, values=%.9f,%.9f,%.9f,%.9f\n",
                        it->second.Data, vals[0], vals[1], vals[2], vals[3]);
            }
            shape_trace_count++;
        }
        // DEBUG: Trace when top-layer d_qkv_rope is already in the map
        static int qkv_rope_shape_trace = 0;
        if (qkv_rope_shape_trace < 8 &&
            ref.name.find("d_blocks[") != std::string::npos &&
            ref.name.find("qkv_rope") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_SHAPE_QKV_ROPE] ref.name=%s shape.size=%zu found_in_map=%s slot=%d\n",
                    ref.name.c_str(), ref.shape.size(),
                    (it != mTensorMap.end() && it->second.Data) ? "YES" : "NO",
                    static_cast<int>(ref.slot));
            if (it != mTensorMap.end() && it->second.Data) {
                fprintf(stderr,
                        "[RESOLVE_SHAPE_QKV_ROPE] ptr=%p shape=%s\n",
                        it->second.Data,
                        tensor_shape_str(it->second).c_str());
            }
            qkv_rope_shape_trace++;
        }
        if (it != mTensorMap.end() && it->second.Data) {
            // For MoE operations, the tensor map may contain dynamically-shaped temps
            // that differ from the statically-compiled shapes. Prioritize the map tensor
            // if it has valid data, even if shapes differ.
            // Verify shape matches
            bool shape_matches = (it->second.Rank == static_cast<int>(ref.shape.size()));
            if (shape_matches) {
                for (int i = 0; i < it->second.Rank && shape_matches; ++i) {
                    shape_matches = (it->second.Sizes[i] == ref.shape[i]);
                }
            }
            if (shape_matches) {
                return it->second;
            }
            // Shape doesn't match, but we have valid data - use it for MoE dynamic shapes
            // This handles cases like swiglu output [total_tokens, D] vs expected [B, T, D]
            return it->second;
        }
        // Need to create a view - get the base tensor and create view
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
            auto [ins_it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*base, ref.shape));
            return ins_it->second;
        }
    }

    // FIX: Always check mTensorMap first for gradient slots before falling back to simplified_grads.
    // This is critical because view_backward stores aliases in mTensorMap, and subsequent ops
    // (like rmsnorm_backward) must use that aliased tensor, not the pre-allocated simplified_grads buffer.
    // Without this check, the gradient chain can break when view_backward creates an alias that
    // points to a different buffer than the pre-allocated slot.
    if (!ref.name.empty()) {
        auto it = mTensorMap.find(ref.name);
        if (it != mTensorMap.end() && it->second.Data) {
            return it->second;
        }
        // DEBUG: Trace when mTensorMap lookup fails for top-layer d_ln1
        static int miss_trace_count = 0;
        if (miss_trace_count < 8 &&
            ref.name.find(".ln1") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_FIX] NOT in mTensorMap: %s, slot=%d\n",
                    ref.name.c_str(), static_cast<int>(ref.slot));
            miss_trace_count++;
        }
        // DEBUG: Trace when mTensorMap lookup fails for top-layer d_qkv_rope
        static int qkv_rope_miss_trace = 0;
        if (qkv_rope_miss_trace < 8 &&
            ref.name.find("d_blocks[") != std::string::npos &&
            ref.name.find("qkv_rope") != std::string::npos &&
            ref.layer_idx == static_cast<int>(mConfig.NumLayers) - 1) {
            fprintf(stderr, "[RESOLVE_FIX_QKV_ROPE] NOT in mTensorMap: %s, slot=%d\n",
                    ref.name.c_str(), static_cast<int>(ref.slot));
            qkv_rope_miss_trace++;
        }
    }

    switch (ref.slot) {
        case TensorSlot::TokenIDs:
            return rs.Inputs;
        case TensorSlot::PositionIDs:
            return rs.PositionIDs;
        case TensorSlot::Targets:
            return rs.Targets;
        case TensorSlot::Losses:
            return rs.Losses;
        case TensorSlot::DLoss:
            return rs.scratch().cross_entropy_dloss;
        case TensorSlot::Encoded:
            return rs.non_block_activations().encoded;
        case TensorSlot::LNFinal:
            return rs.non_block_activations().ln_final;
        case TensorSlot::LNFinalRSTD:
            return rs.non_block_activations().ln_final_rstd;
        case TensorSlot::FinalResidual:
            return rs.get_final_residual();
        case TensorSlot::FreqCis:
            return rs.non_block_activations().freq_cis;
        case TensorSlot::BlockLN1:
            return rs.simplified_acts(ref.layer_idx).ln1;
        case TensorSlot::BlockLN1RSTD:
            return rs.simplified_acts(ref.layer_idx).ln1_rstd;
        case TensorSlot::BlockLN2:
            return rs.simplified_acts(ref.layer_idx).ln2;
        case TensorSlot::BlockLN2RSTD:
            return rs.simplified_acts(ref.layer_idx).ln2_rstd;
        case TensorSlot::BlockQRSTD:
            return rs.simplified_acts(ref.layer_idx).q_rstd;
        case TensorSlot::BlockKRSTD:
            return rs.simplified_acts(ref.layer_idx).k_rstd;
        case TensorSlot::BlockQKV:
            return rs.simplified_acts(ref.layer_idx).qkv;
        case TensorSlot::BlockQKVRoPE: {
            auto& acts = rs.simplified_acts(ref.layer_idx);
            return acts.qkv_rope.Data ? acts.qkv_rope : acts.qkv;
        }
        case TensorSlot::BlockLSE:
            return rs.simplified_acts(ref.layer_idx).lse;
        case TensorSlot::BlockAtt:
            return rs.simplified_acts(ref.layer_idx).att;
        case TensorSlot::BlockAttOut:
            return rs.simplified_acts(ref.layer_idx).att_out;
        case TensorSlot::BlockResidualAtt:
            return rs.simplified_acts(ref.layer_idx).residual_att;
        case TensorSlot::BlockMLPUp:
            return rs.simplified_acts(ref.layer_idx).mlp_up;
        case TensorSlot::BlockSwiGLU:
            return rs.simplified_acts(ref.layer_idx).swiglu;
        case TensorSlot::BlockMLPDown:
            return rs.simplified_acts(ref.layer_idx).mlp_down;
        case TensorSlot::BlockResidualFFN:
            return rs.get_residual(ref.layer_idx, rs.MainStream);
        case TensorSlot::BlockDLN1:
            return rs.simplified_grads(ref.layer_idx).d_ln1;
        case TensorSlot::BlockDQKV:
            return rs.simplified_grads(ref.layer_idx).d_qkv;
        case TensorSlot::BlockDAtt:
            return rs.simplified_grads(ref.layer_idx).d_att;
        case TensorSlot::BlockDSwiGLU:
            return rs.simplified_grads(ref.layer_idx).d_swiglu;
        case TensorSlot::BlockDMLPUp:
            return rs.simplified_grads(ref.layer_idx).d_mlp_up;
        case TensorSlot::BlockDMLPDown:
            return rs.simplified_grads(ref.layer_idx).d_mlp_down;
        case TensorSlot::BlockDLN2:
            return rs.simplified_grads(ref.layer_idx).d_ln2;
        case TensorSlot::BlockDResAtt:
            return rs.simplified_grads(ref.layer_idx).d_res_att;
        case TensorSlot::BlockDResFFN:
            return rs.simplified_grads(ref.layer_idx).d_res_ffn;
        case TensorSlot::Parameter:
            return mWeights.get(ref.name);
        case TensorSlot::Saved:
            if (mSaved) {
                auto it = mSaved->find(ref.name);
                if (it != mSaved->end()) {
                    // If the saved tensor has actual data, use it directly.
                    // Only resolve from live buffers when Data == nullptr (metadata-only mode).
                    // This is critical for FFT mode where tensors with lora_only recompute_policy
                    // are saved with actual data and should NOT use live buffers.
                    if (it->second.Data != nullptr) {
                        return it->second;
                    }
                    // Metadata-only: try to resolve from live buffer or recompute
                    if (auto live_it = mTensorMap.find(ref.name); live_it != mTensorMap.end()) {
                        return live_it->second;
                    }
                    if (Tensor* live = try_resolve_saved_live(ref.name, it->second)) {
                        return *live;
                    }
                    return it->second;
                }
            }
            throw std::runtime_error("CompiledExecutor: saved tensor not found: " + ref.name);
        case TensorSlot::Mapped: {
            auto it = mTensorMap.find(ref.name);
            if (it != mTensorMap.end()) {
                return it->second;
            }
            throw std::runtime_error("CompiledExecutor: tensor not found: " + ref.name);
        }
        case TensorSlot::Temporary:
            throw std::runtime_error("CompiledExecutor: temporary slot requires allocation");
    }
    throw std::runtime_error("CompiledExecutor: invalid tensor slot");
}

Tensor& CompiledExecutor::ensure_output_tensor(const TensorRef& ref) {
    if (!ref.name.empty()) {
        if (auto base = base_param_from_grad(ref.name)) {
            bool accum = false;
            if (Tensor* grad = mGrads.get_param_grad(*base, accum)) {
                if (grad->Data) {
                    if (!ref.shape.empty()) {
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view_tensor(*grad, ref.shape));
                        return it->second;
                    }
                    auto [it, _] = mTensorMap.insert_or_assign(ref.name, *grad);
                    return it->second;
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
                    alias_name = "blocks[" + std::to_string(layer_idx) + "]." + alias_field;
                }
                if (auto alias_entry = mSlotRegistry->lookup(alias_field)) {
                    TensorRef alias_ref;
                    alias_ref.name = alias_name;
                    alias_ref.layer_idx = layer_idx;
                    alias_ref.slot = alias_entry->slot;
                    alias_ref.shape = ref.shape;
                    alias_ref.dtype = ref.dtype;
                    if (mSaveSet.find(alias_name) != mSaveSet.end()) {
                        alias_ref.slot = TensorSlot::Saved;
                    }
                    try {
                        Tensor& base = resolve_tensor(alias_ref);
                        Tensor view = ref.shape.empty() ? base : view_tensor(base, ref.shape);
                        if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
                            ref.name.find("permuted_input") != std::string::npos) {
                            std::ostringstream oss;
                            oss << "[MOE_ALIAS] name=" << ref.name << " alias=" << alias_name << " ref_shape=[";
                            for (std::size_t i = 0; i < ref.shape.size(); ++i) {
                                if (i) oss << ",";
                                oss << ref.shape[i];
                            }
                            oss << "] base_shape=[";
                            for (int i = 0; i < base.Rank; ++i) {
                                if (i) oss << ",";
                                oss << base.Sizes[i];
                            }
                            oss << "] view_shape=[";
                            for (int i = 0; i < view.Rank; ++i) {
                                if (i) oss << ",";
                                oss << view.Sizes[i];
                            }
                            oss << "]\n";
                            std::fputs(oss.str().c_str(), stderr);
                        }
                        auto [it, _] = mTensorMap.insert_or_assign(ref.name, view);
                        return it->second;
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
            // Create a view if needed
            auto [it, inserted] = mTensorMap.emplace(ref.name, view_tensor(t, ref.shape));
            if (!inserted) {
                it->second = view_tensor(t, ref.shape);
            }
            return it->second;
        }
        return t;
    }

    // For mapped/temporary tensors, allocate if needed
    auto it = mTensorMap.find(ref.name);
    if (it != mTensorMap.end()) {
        if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
            ref.name.find("permuted_input") != std::string::npos) {
            std::ostringstream oss;
            oss << "[MOE_ALLOC_HIT] name=" << ref.name << " shape=[";
            for (int i = 0; i < it->second.Rank; ++i) {
                if (i) oss << ",";
                oss << it->second.Sizes[i];
            }
            oss << "]\n";
            std::fputs(oss.str().c_str(), stderr);
        }
        return it->second;
    }

    Tensor t = mRunState.temp_alloc(ref.dtype, ref.shape);
    if (std::getenv("SUROGATE_TRACE_MOE_ALLOC") &&
        ref.name.find("permuted_input") != std::string::npos) {
        std::ostringstream oss;
        oss << "[MOE_ALLOC_NEW] name=" << ref.name << " ref_shape=[";
        for (std::size_t i = 0; i < ref.shape.size(); ++i) {
            if (i) oss << ",";
            oss << ref.shape[i];
        }
        oss << "] alloc_shape=[";
        for (int i = 0; i < t.Rank; ++i) {
            if (i) oss << ",";
            oss << t.Sizes[i];
        }
        oss << "]\n";
        std::fputs(oss.str().c_str(), stderr);
    }

    // Zero gradient tensors to prevent stale values from accumulating.
    // Gradient tensor names start with "d_" (e.g., "d_blocks[0].ln1").
    // Many backward kernels accumulate (+=) to their outputs, so they
    // need zeroed buffers to start with. Stack memory is reused across
    // micro-batches and contains stale values from previous backward passes.
    if (ref.name.size() >= 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
        fill_zero(t, mRunState.MainStream);
        static int zero_trace = 0;
        if (zero_trace < 5) {
            fprintf(stderr, "[ZERO_GRAD] Zeroed %s ptr=%p\n", ref.name.c_str(), t.Data);
            zero_trace++;
        }
    }

    mTemps.push_back(t);
    auto [ins_it, inserted] = mTensorMap.emplace(ref.name, t);
    return ins_it->second;
}

void CompiledExecutor::handle_layer_start(int layer_idx) {
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        // Wait for current layer's weights
        mWeightManager->wait_for_gather(layer_idx, mRunState.MainStream);
    }

    // Prefetch next layer
    const int next_layer = layer_idx + 1;
    if (next_layer < static_cast<int>(mConfig.NumLayers) && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            if (mComm) {
                mWeightManager->gather_block(next_layer, *mComm, mRunState.side_stream());
            }
        }
    }

    mCurrentLayer = layer_idx;

    // DEBUG: Check previous layer MLP output for NaNs at layer boundary.
    static int pre_ln1_nan_trace = 0;
    if (pre_ln1_nan_trace < 4 && layer_idx == 3) {
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        auto& prev_acts = mRunState.simplified_acts(layer_idx - 1);
        log_nan_sample("PRE_LN1_MLP_DOWN", layer_idx - 1, "blocks[2].mlp_down", prev_acts.mlp_down, 3);
        pre_ln1_nan_trace++;
    }
}

void CompiledExecutor::handle_layer_end(int layer_idx) {
    // Release previous layer's weights
    if (mWeightManager && mWeightManager->is_streaming_enabled() && !mCapturing) {
        mWeightManager->release_block(layer_idx, mRunState.MainStream);
    }

    // Offload residual if enabled
    if (mRunState.has_residual_offloading() && !mCapturing) {
        mRunState.mark_residual_ready(layer_idx, mRunState.MainStream);
        mRunState.put_residual(layer_idx, mRunState.side_stream());
    }
}


void CompiledExecutor::execute_forward(const CompiledGraph& graph,
                                       NCCLCommunicator& comm,
                                       bool full,
                                       const modules::ForwardHook* hook) {
    mComm = &comm;
    mTemps.clear();
    mTensorMap.clear();
    mCurrentLayer = -1;

    // Match GraphExecutor behavior: initialize loss/counter buffers for full forward runs.
    // This avoids stale accumulation when tests call CompiledExecutor directly.
    if (full) {
        bool has_loss_op = false;
        for (const auto& op : graph.ops) {
            if (op.type == CompiledOpType::CrossEntropyLoss ||
                op.type == CompiledOpType::FusedLMHeadLoss) {
                has_loss_op = true;
                break;
            }
        }
        if (has_loss_op) {
            fill_zero(mRunState.Losses, mRunState.MainStream);
            fill_zero(mRunState.ValidTokenCount, mRunState.MainStream);
            fill_zero(mRunState.CorrectCount, mRunState.MainStream);
        }
    }
    const int num_layers = static_cast<int>(mConfig.NumLayers);
    std::vector<DeviceMemoryStack::Checkpoint> layer_checkpoints;
    std::vector<std::size_t> layer_temp_marks;
    std::vector<char> layer_active;
    if (num_layers > 0) {
        layer_checkpoints.resize(static_cast<std::size_t>(num_layers));
        layer_temp_marks.resize(static_cast<std::size_t>(num_layers));
        layer_active.assign(static_cast<std::size_t>(num_layers), 0);
    }
    auto prune_stack_tensors = [&]() {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            // Skip tensors that are needed for backward (in save list)
            if (mSaveSet.count(it->first) > 0) {
                ++it;
                continue;
            }
            // Skip MoE expert_offsets - needed for backward but not in autodiff save list
            if (mConfig.NumExperts > 0 && (it->first == "moe_expert_offsets" || it->first == "moe_gather_indices")) {
                ++it;
                continue;
            }
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind known inputs
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;
    mTensorMap["x0"] = mRunState.non_block_activations().encoded;

    // Ensure non-block weights are gathered if streaming/offload is enabled
    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_embeddings(comm, mRunState.MainStream);
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
    }

    // Prefetch layer 0 before loop
    if (mConfig.NumLayers > 0 && !mCapturing) {
        if (mWeightManager && mWeightManager->is_streaming_enabled()) {
            mWeightManager->gather_block(0, comm, mRunState.side_stream());
        }
    }

    // Main dispatch loop - no string comparisons, direct function pointer dispatch
    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        if (!full && !graph.required_mask.empty() && !graph.required_mask[idx]) {
            continue;
        }

        const auto& op = graph.ops[idx];

        // Handle layer boundaries
        if (op.layer_start >= 0) {
            if (op.layer_start < num_layers &&
                !layer_active[static_cast<std::size_t>(op.layer_start)]) {
                layer_checkpoints[static_cast<std::size_t>(op.layer_start)] = mRunState.Stack.checkpoint();
                layer_temp_marks[static_cast<std::size_t>(op.layer_start)] = mTemps.size();
                layer_active[static_cast<std::size_t>(op.layer_start)] = 1;
            }
            handle_layer_start(op.layer_start);
        }

        try {
            // Direct dispatch via switch (branch predictor friendly, no string compare)
            switch (op.type) {
                case CompiledOpType::Embedding:
                    dispatch_embedding(op);
                    break;
                case CompiledOpType::Zeros:
                    dispatch_zeros(op);
                    break;
                case CompiledOpType::FusedResidualRMSNorm:
                    dispatch_fused_residual_rmsnorm(op);
                    break;
                case CompiledOpType::View:
                    dispatch_view(op);
                    break;
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                case CompiledOpType::Matmul:
                case CompiledOpType::MatmulBias:
                    dispatch_matmul(op, hook);
                    break;
                case CompiledOpType::BiasAdd:
                    dispatch_bias_add(op);
                    break;
                case CompiledOpType::SwiGLU:
                    dispatch_swiglu(op);
                    break;
                case CompiledOpType::Silu:
                    dispatch_silu(op);
                    break;
                case CompiledOpType::Mul:
                    dispatch_mul(op);
                    break;
                case CompiledOpType::MatmulSwiGLU:
                    dispatch_matmul_swiglu(op);
                    break;
                case CompiledOpType::QKVQKNormRoPE:
                    dispatch_qkv_qk_norm_rope(op);
                    break;
                case CompiledOpType::RoPE:
                    dispatch_rope(op);
                    break;
                case CompiledOpType::FlashAttention:
                    dispatch_flash_attention(op);
                    break;
                case CompiledOpType::CrossEntropyLoss:
                    dispatch_cross_entropy_loss(op);
                    break;
                case CompiledOpType::FusedLMHeadLoss:
                    dispatch_fused_lm_head_loss(op);
                    break;
                // MoE operations
                case CompiledOpType::MoESoftmax:
                    dispatch_moe_softmax(op);
                    break;
                case CompiledOpType::MoESigmoid:
                    dispatch_moe_sigmoid(op);
                    break;
                case CompiledOpType::MoETopK:
                    dispatch_moe_topk(op);
                    break;
                case CompiledOpType::MoEPermute:
                    dispatch_moe_permute(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUp:
                    dispatch_moe_grouped_gemm_gate_up(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDown:
                    dispatch_moe_grouped_gemm_down(op);
                    break;
                case CompiledOpType::MoEUnpermute:
                    dispatch_moe_unpermute(op);
                    break;
                default:
                    throw std::runtime_error("CompiledExecutor: unsupported forward op type");
            }
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor forward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            throw std::runtime_error(oss.str());
        }

        // Handle layer end
        if (op.layer_end >= 0) {
            // Note: Forward activation stats are not printed because with recompute_block=true,
            // the activation buffers are shared across layers, so they only contain the last
            // layer's data at this point, not the per-layer values.
            if (op.layer_end < num_layers &&
                layer_active[static_cast<std::size_t>(op.layer_end)]) {
                // For MoE models, skip stack restore because:
                // 1. MoE backward needs forward activations (routing_weights, scatter_indices, etc.)
                // 2. The recompute mechanism (recompute_block) doesn't support MoE ops
                // 3. Without recompute, we must preserve all forward tensors for backward
                // TODO: Implement MoE-specific recompute to enable memory savings
                if (mConfig.NumExperts == 0) {
                    mRunState.Stack.restore(layer_checkpoints[static_cast<std::size_t>(op.layer_end)]);
                    if (mTemps.size() > layer_temp_marks[static_cast<std::size_t>(op.layer_end)]) {
                        mTemps.resize(layer_temp_marks[static_cast<std::size_t>(op.layer_end)]);
                    }
                    prune_stack_tensors();
                    if (mRunState.ffn_temps_on_stack()) {
                        auto& acts = mRunState.simplified_acts(op.layer_end);
                        acts.mlp_up.Data = nullptr;
                        acts.swiglu.Data = nullptr;
                    }
                    // Note: cudnn_workspace is persistently allocated, don't clear
                    layer_active[static_cast<std::size_t>(op.layer_end)] = 0;
                }
            }
            handle_layer_end(op.layer_end);
        }
    }

    // DEBUG: Summarize parameter gradient samples to spot divergence.
    static int param_grad_trace_enabled = -1;
    if (param_grad_trace_enabled < 0) {
        param_grad_trace_enabled = (std::getenv("SUROGATE_TRACE_PARAM_GRADS") != nullptr) ? 1 : 0;
    }
    if (param_grad_trace_enabled) {
        struct ParamSample {
            std::string name;
            double mean_abs{0.0};
            double max_abs{0.0};
            std::size_t count{0};
        };
        std::vector<ParamSample> samples;
        double total_sum_sq = 0.0;
        double total_sum_abs = 0.0;
        std::size_t total_count = 0;
        constexpr std::size_t kSampleN = 1024;
        for (const auto& param_name : mGrads.param_names()) {
            bool accumulate = false;
            Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
            if (!grad_tensor || !grad_tensor->Data || grad_tensor->nelem() <= 0) {
                continue;
            }
            const std::size_t n = std::min<std::size_t>(kSampleN, static_cast<std::size_t>(grad_tensor->nelem()));
            std::vector<float> vals;
            if (!copy_tensor_sample_as_f32(*grad_tensor, n, vals)) {
                continue;
            }
            double sum_abs = 0.0;
            double sum_sq = 0.0;
            double max_abs = 0.0;
            for (float v : vals) {
                if (std::isnan(v) || std::isinf(v)) {
                    continue;
                }
                const double av = std::abs(static_cast<double>(v));
                sum_abs += av;
                sum_sq += av * av;
                if (av > max_abs) {
                    max_abs = av;
                }
            }
            if (!vals.empty()) {
                ParamSample s;
                s.name = param_name;
                s.count = vals.size();
                s.mean_abs = sum_abs / static_cast<double>(vals.size());
                s.max_abs = max_abs;
                samples.push_back(std::move(s));
                total_sum_abs += sum_abs;
                total_sum_sq += sum_sq;
                total_count += vals.size();
            }
        }
        if (!samples.empty()) {
            std::sort(samples.begin(), samples.end(),
                      [](const ParamSample& a, const ParamSample& b) { return a.mean_abs > b.mean_abs; });
            const std::size_t top_n = std::min<std::size_t>(10, samples.size());
            for (std::size_t i = 0; i < top_n; ++i) {
                const auto& s = samples[i];
                fprintf(stderr,
                        "[PARAM_GRAD_SAMPLE] name=%s mean_abs=%.6e max_abs=%.6e n=%zu\n",
                        s.name.c_str(), s.mean_abs, s.max_abs, s.count);
            }
            if (total_count > 0) {
                const double mean_abs = total_sum_abs / static_cast<double>(total_count);
                const double l2 = std::sqrt(total_sum_sq);
                fprintf(stderr,
                        "[PARAM_GRAD_SAMPLE_NORM] sample_l2=%.6e mean_abs=%.6e count=%zu\n",
                        l2, mean_abs, total_count);
            }
        }
    }

    // Free temporaries
    for (auto it = mTemps.rbegin(); it != mTemps.rend(); ++it) {
        mRunState.temp_free(*it);
    }
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_embeddings(mRunState.MainStream);
        mWeightManager->release_final_norm(mRunState.MainStream);
    }
}

void CompiledExecutor::execute_backward(const CompiledGraph& graph,
                                        NCCLCommunicator& comm,
                                        int grad_accum_steps,
                                        int micro_step,
                                        const modules::BackwardHook* hook) {
    mComm = &comm;
    mRunState.reset_simplified_gradients();
    mTemps.clear();
    mTensorMap.clear();
    mAccumulateTensors.clear();
    mCurrentLayer = -1;
    mLastRecomputeLayer = -1;
    mMicroStep = micro_step;

    // Clear activation/non-block gradients for each micro-step.
    // Compiled executor does not go through GraphExecutor's zeroing path.
    fill_zero(mRunState.non_block_gradients().d_ln_final, mRunState.MainStream);
    if (mRunState.non_block_gradients().d_embeddings.Data && !mRunState.is_lora_only_mode()) {
        fill_zero(mRunState.non_block_gradients().d_embeddings, mRunState.MainStream);
    }
    if (mConfig.NumLayers > 0) {
        fill_zero(mRunState.simplified_grads(static_cast<int>(mConfig.NumLayers) - 1).d_res_ffn,
                  mRunState.MainStream);
    }
    mRunState.zero_activation_gradients(mRunState.MainStream);

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->gather_final_norm(comm, mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->gather_lm_head(comm, mRunState.MainStream);
        }
    }

    // Save stack checkpoint at start of backward - we'll restore per-layer to manage memory
    auto initial_checkpoint = mRunState.Stack.checkpoint();
    int last_layer_restored = -1;
    auto clear_shared_grads = [&](int layer_idx) {
        if (!mRunState.large_bwd_temps_on_stack()) {
            return;
        }
        if (layer_idx < 0 || layer_idx >= static_cast<int>(mConfig.NumLayers)) {
            return;
        }
        auto& grads = mRunState.simplified_grads(layer_idx);
        if (grads.d_ln2.Data) {
            fill_zero(grads.d_ln2, mRunState.MainStream);
        }
        if (grads.d_att.Data) {
            fill_zero(grads.d_att, mRunState.MainStream);
        }
        if (grads.d_ln1.Data) {
            fill_zero(grads.d_ln1, mRunState.MainStream);
        }
    };
    auto prune_stack_tensors = [&](int current_layer) {
        for (auto it = mTensorMap.begin(); it != mTensorMap.end(); ) {
            // Skip MoE expert_offsets - needed throughout backward for grouped GEMM ops
            if (mConfig.NumExperts > 0 && it->first == "moe_expert_offsets") {
                ++it;
                continue;
            }
            // Skip cross-layer gradients - these are needed by the previous layer's backward
            // Cross-layer gradients have names like "d_blocks[N].XXX" where N < current_layer
            // They flow from one layer's backward to the previous layer's backward
            if (current_layer >= 0 && it->first.rfind("d_blocks[", 0) == 0) {
                // Parse the layer index from the gradient name
                auto bracket_pos = it->first.find('[');
                auto close_pos = it->first.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string layer_str = it->first.substr(bracket_pos + 1, close_pos - bracket_pos - 1);
                    try {
                        int grad_layer = std::stoi(layer_str);
                        // Preserve gradients for layers below the current one (they'll be needed)
                        if (grad_layer < current_layer) {
                            ++it;
                            continue;
                        }
                    } catch (...) {
                        // If parsing fails, skip this tensor to be safe
                        ++it;
                        continue;
                    }
                }
            }
            // Skip saved tensors for layers below current (needed for their backward)
            // Saved tensors have names like "blocks[N].XXX" where N < current_layer
            if (current_layer >= 0 && it->first.rfind("blocks[", 0) == 0) {
                auto bracket_pos = it->first.find('[');
                auto close_pos = it->first.find(']');
                if (bracket_pos != std::string::npos && close_pos != std::string::npos) {
                    std::string layer_str = it->first.substr(bracket_pos + 1, close_pos - bracket_pos - 1);
                    try {
                        int saved_layer = std::stoi(layer_str);
                        // Preserve saved tensors for layers below the current one
                        if (saved_layer < current_layer) {
                            ++it;
                            continue;
                        }
                    } catch (...) {
                        // If parsing fails, skip this tensor to be safe
                        ++it;
                        continue;
                    }
                }
            }
            if (it->second.Data && mRunState.Stack.owns(it->second.Data) &&
                !mRunState.Stack.is_live(it->second.Data)) {
                it = mTensorMap.erase(it);
            } else {
                ++it;
            }
        }
    };

    // Bind initial gradient tensors (from loss computation)
    // d_logits is stored in the output buffer after loss backward (only when lmhead_chunks == 1)
    auto& output = mRunState.non_block_activations().output;
    if (!output.Data) {
        throw std::runtime_error("CompiledExecutor: output tensor has no data (B=" +
                                std::to_string(mB) + ", T=" + std::to_string(mT) + ")");
    }

    if (mOptions.LMHeadChunks <= 1) {
        Tensor logits_view = view_tensor(output, {mB, mT, static_cast<long>(mConfig.VocabSize)});
        mTensorMap["d_logits"] = logits_view;
        // Also provide flattened version for matmul backward ops
        Tensor logits_flat = view_tensor(output, {mB * mT, static_cast<long>(mConfig.VocabSize)});
        if (logits_flat.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat has wrong rank=" +
                                    std::to_string(logits_flat.Rank) + " expected 2");
        }
        mTensorMap["d_logits_flat"] = logits_flat;
        // Verify the map entry
        auto& check = mTensorMap["d_logits_flat"];
        if (check.Rank != 2) {
            throw std::runtime_error("CompiledExecutor: d_logits_flat in map has wrong rank=" +
                                    std::to_string(check.Rank));
        }
    }

    // Bind gradient output buffers for final layer norm backward
    // DSL-driven: use slot registry to derive all mappings from gradient_of relationships
    Tensor& d_ln_final_buf = mRunState.non_block_gradients().d_ln_final;
    Tensor& d_embeddings_buf = mRunState.non_block_gradients().d_embeddings;

    Tensor d_ln_final_flat = view_tensor(d_ln_final_buf,
                                         {mB * mT, static_cast<long>(mConfig.HiddenSize)});

    // Helper to determine target buffer based on gradient_of field
    auto get_target_buffer = [&](const std::string& grad_of) -> Tensor* {
        // Final norm gradients (xF, ln_final, residual_final)
        if (grad_of == "xF" || grad_of == "ln_final" || grad_of == "xF_flat" ||
            grad_of == "residual_final" || grad_of == "final_residual") {
            return &d_ln_final_buf;
        }
        // Embedding output gradients (x0, encoded)
        if (grad_of == "x0" || grad_of == "encoded" || grad_of == "embeddings") {
            if (!mRunState.is_lora_only_mode()) {
                return &d_embeddings_buf;
            }
        }
        // Note: d_xN, d_residualN don't map to persistent buffers - they're computed on-the-fly
        return nullptr;
    };

    // Bind global gradient tensors - these are always needed regardless of DSL layout
    // The DSL gradient slots declare shape/dtype but the actual buffers come from RunState
    mTensorMap["d_xF_flat"] = d_ln_final_flat;
    mTensorMap["d_xF"] = d_ln_final_buf;
    mTensorMap["d_ln_final"] = d_ln_final_buf;
    mTensorMap["d_ln_final_flat"] = d_ln_final_flat;

    if (!mRunState.is_lora_only_mode()) {
        mTensorMap["d_encoded"] = d_embeddings_buf;
        mTensorMap["d_x0"] = d_embeddings_buf;
    }

    static bool emb_grad_initial_logged = false;
    if (!emb_grad_initial_logged && d_embeddings_buf.Data) {
        if (tensor_sample_has_nan_or_inf(d_embeddings_buf, 3)) {
            fprintf(stderr, "[EMB_DOUT_NAN_INIT] micro_step=%d ptr=%p\n", micro_step, d_embeddings_buf.Data);
            log_tensor_stats_ex("EMB_DOUT_INIT_NAN", -1, "d_embeddings", d_embeddings_buf, 4096, true);
        }
        emb_grad_initial_logged = true;
    }

    // DSL-driven binding for any additional gradient slots declared in the Python model
    if (mSlotRegistry && mSlotRegistry->has_dsl_layout()) {
        mSlotRegistry->for_each([&](const std::string& slot_name,
                                    const TensorSlotRegistry::SlotEntry& entry) {
            if (entry.scope != ActivationScope::GlobalGradient) return;
            // Skip if already bound above
            if (mTensorMap.find(slot_name) != mTensorMap.end()) return;

            Tensor* target_buf = get_target_buffer(entry.gradient_of);
            if (target_buf && target_buf->Data) {
                mTensorMap[slot_name] = *target_buf;
            }
        });
    }

    // Ensure global block outputs (xN/residualN) map to the last block's gradients.
    // These gradients must survive layer-boundary stack restores in recompute mode.
    if (mConfig.NumLayers > 0) {
        const int last_layer = static_cast<int>(mConfig.NumLayers) - 1;
        auto& last_grads = mRunState.simplified_grads(last_layer);
        if (last_grads.d_mlp_down.Data) {
            mTensorMap["d_xN"] = last_grads.d_mlp_down;
        }
        if (last_grads.d_res_att.Data) {
            mTensorMap["d_residualN"] = last_grads.d_res_att;
        }

        // Heuristic aliasing for non-inlined StackedBlocks outputs (e.g., "StackedBlocks_4").
        if (mSaved) {
            std::vector<std::pair<int, std::string>> stacked;
            stacked.reserve(2);
            for (const auto& kv : *mSaved) {
                const std::string& name = kv.first;
                if (name.rfind("StackedBlocks_", 0) != 0) {
                    continue;
                }
                int idx = -1;
                const char* s = name.c_str() + std::strlen("StackedBlocks_");
                if (*s) {
                    char* end = nullptr;
                    long parsed = std::strtol(s, &end, 10);
                    if (end != s) {
                        idx = static_cast<int>(parsed);
                    }
                }
                if (idx >= 0) {
                    stacked.emplace_back(idx, name);
                }
            }
            if (!stacked.empty()) {
                std::sort(stacked.begin(), stacked.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });
                if (stacked.size() == 1) {
                    if (last_grads.d_res_att.Data) {
                        mTensorMap["d_" + stacked[0].second] = last_grads.d_res_att;
                    }
                } else {
                    if (last_grads.d_mlp_down.Data) {
                        mTensorMap["d_" + stacked[0].second] = last_grads.d_mlp_down;
                    }
                    if (last_grads.d_res_att.Data) {
                        mTensorMap["d_" + stacked[1].second] = last_grads.d_res_att;
                    }
                }
            }
        }
    }

    // Bind autodiff-generated gradient names (d_embed_1, etc.) from forward embedding outputs
    // These are dynamically generated and not in the DSL layout
    if (!mRunState.is_lora_only_mode()) {
        for (const auto& emb_out : mEmbeddingOutputs) {
            std::string grad_name = "d_" + emb_out;
            mTensorMap[grad_name] = d_embeddings_buf;
        }
    }

    // Restore MoE expert_offsets from persistent CPU storage
    // This is needed by grouped GEMM backward ops for proper token routing
    if (mConfig.NumExperts > 0 && !mMoEExpertOffsetsData.empty()) {
        // Allocate PERSISTENT GPU buffer for expert_offsets (not stack-allocated)
        // This ensures the memory won't be invalidated by stack restores or temp_free calls
        const int num_elements = static_cast<int>(mMoEExpertOffsetsData.size());
        const size_t needed_bytes = num_elements * sizeof(int);

        // Allocate or resize GPU buffer if needed
        if (mMoEExpertOffsetsGPU == nullptr || mMoEExpertOffsetsGPUSize < needed_bytes) {
            if (mMoEExpertOffsetsGPU) {
                CUDA_CHECK(cudaFree(mMoEExpertOffsetsGPU));
            }
            CUDA_CHECK(cudaMalloc(&mMoEExpertOffsetsGPU, needed_bytes));
            mMoEExpertOffsetsGPUSize = needed_bytes;
        }

        // Copy data from CPU to GPU
        CUDA_CHECK(cudaMemcpyAsync(mMoEExpertOffsetsGPU, mMoEExpertOffsetsData.data(),
                                   needed_bytes, cudaMemcpyHostToDevice, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));

        // Create tensor wrapper pointing to persistent buffer
        Tensor expert_offsets;
        expert_offsets.DType = ETensorDType::INT32;
        expert_offsets.Rank = 1;
        expert_offsets.Sizes[0] = num_elements;
        expert_offsets.Data = static_cast<std::byte*>(mMoEExpertOffsetsGPU);

        mTensorMap["moe_expert_offsets"] = expert_offsets;
        // Note: NOT adding to mTemps since this is persistent memory managed separately
    }

    // Also bind standard inputs that backward ops may reference
    mTensorMap["token_ids"] = mRunState.Inputs;
    mTensorMap["position_ids"] = mRunState.PositionIDs;

    // Build the set of gradients that require accumulation (not the first micro-step).
    // Also bind parameter gradient tensors to mTensorMap so they're used instead of temporaries.
    // This mirrors the logic in graph_executor_backward.cpp (bind_param_grad).
    for (const auto& param_name : mGrads.param_names()) {
        if (param_name.find("rope_freqs") != std::string::npos) {
            continue;
        }
        bool accumulate = false;
        Tensor* grad_tensor = mGrads.get_param_grad(param_name, accumulate);
        if (grad_tensor && grad_tensor->Data) {
            std::string grad_name = "d_" + param_name;
            mTensorMap[grad_name] = *grad_tensor;
            if (accumulate) {
                mAccumulateTensors.insert(grad_name);
            }
        }
    }

    auto is_grad_ref = [](const TensorRef& ref) -> bool {
        if (!ref.name.empty() && ref.name.size() > 2 && ref.name[0] == 'd' && ref.name[1] == '_') {
            return true;
        }
        switch (ref.slot) {
            case TensorSlot::BlockDLN1:
            case TensorSlot::BlockDQKV:
            case TensorSlot::BlockDAtt:
            case TensorSlot::BlockDSwiGLU:
            case TensorSlot::BlockDMLPUp:
            case TensorSlot::BlockDMLPDown:
            case TensorSlot::BlockDLN2:
            case TensorSlot::BlockDResAtt:
            case TensorSlot::BlockDResFFN:
            case TensorSlot::DLoss:
                return true;
            default:
                return false;
        }
    };

    auto ref_layer_idx = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto ref_layer_idx_any = [&](const TensorRef& ref) -> int {
        if (ref.layer_idx >= 0) {
            return ref.layer_idx;
        }
        if (ref.name.empty()) {
            return -1;
        }
        std::string_view name = ref.name;
        if (name.rfind("d_", 0) == 0) {
            name.remove_prefix(2);
        }
        if (name.rfind("saved.", 0) == 0) {
            name.remove_prefix(6);
        }
        int layer_idx = -1;
        std::string field;
        if (parse_block_param(name, layer_idx, field)) {
            return layer_idx;
        }
        return -1;
    };

    auto op_layer_idx = [&](const CompiledOp& op) -> int {
        int detected_non_grad = -1;
        for (const auto& ref : op.inputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        for (const auto& ref : op.outputs) {
            if (!is_grad_ref(ref)) {
                const int layer_idx = ref_layer_idx(ref);
                if (layer_idx >= 0) {
                    detected_non_grad = std::max(detected_non_grad, layer_idx);
                }
            }
        }
        return (detected_non_grad >= 0) ? detected_non_grad : -1;
    };

    auto op_layer_idx_any = [&](const CompiledOp& op) -> int {
        int detected_any = -1;
        for (const auto& ref : op.inputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        for (const auto& ref : op.outputs) {
            const int layer_idx = ref_layer_idx_any(ref);
            if (layer_idx >= 0) {
                detected_any = std::max(detected_any, layer_idx);
            }
        }
        if (op.attrs.layer_idx >= 0) {
            detected_any = std::max(detected_any, op.attrs.layer_idx);
        }
        return detected_any;
    };

    const bool skip_logits_grad = (mOptions.LMHeadChunks > 1);
    auto is_logits_grad_name = [](const std::string& name) {
        return name == "d_logits" || name == "d_logits_flat";
    };
    auto is_logits_grad_op = [&](const CompiledOp& op) {
        for (const auto& ref : op.inputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        for (const auto& ref : op.outputs) {
            if (is_logits_grad_name(ref.name)) return true;
        }
        return false;
    };

    const int num_layers = static_cast<int>(mConfig.NumLayers);
    const bool assert_recompute_a = (std::getenv("SUROGATE_ASSERT_RECOMPUTE_A") != nullptr);
    if (assert_recompute_a) {
        if (mRecomputeSamples.size() != static_cast<std::size_t>(num_layers)) {
            mRecomputeSamples.assign(static_cast<std::size_t>(num_layers), RecomputeSample{});
        }
        for (auto& sample : mRecomputeSamples) {
            sample.micro_step = mMicroStep;
            sample.ln1_valid = false;
            sample.ln2_valid = false;
        }
    }
    auto record_recompute_sample = [&](int layer_idx) {
        if (!assert_recompute_a) return;
        if (layer_idx < 0 || layer_idx >= num_layers) return;
        auto& sample = mRecomputeSamples[static_cast<std::size_t>(layer_idx)];
        sample.micro_step = mMicroStep;
        sample.ln1_valid = false;
        sample.ln2_valid = false;
        const long sample_token = 3;
        std::vector<float> vals;
        auto& acts = mRunState.simplified_acts(layer_idx);
        if (acts.ln1.Data && copy_tensor_token_sample_as_f32(acts.ln1, sample_token, 4, vals)) {
            for (int i = 0; i < 4; ++i) sample.ln1[static_cast<std::size_t>(i)] = vals[i];
            sample.ln1_valid = true;
        }
        if (acts.ln2.Data && copy_tensor_token_sample_as_f32(acts.ln2, sample_token, 4, vals)) {
            for (int i = 0; i < 4; ++i) sample.ln2[static_cast<std::size_t>(i)] = vals[i];
            sample.ln2_valid = true;
        }
    };
    std::vector<std::size_t> layer_start_indices(num_layers, SIZE_MAX);
    std::vector<bool> layer_seen_any(num_layers, false);
    for (const auto& op : graph.ops) {
        if (op.layer_start >= 0 && op.layer_start < num_layers) {
            layer_start_indices[op.layer_start] = &op - graph.ops.data();
        }
    }

    for (std::size_t idx = 0; idx < graph.ops.size(); ++idx) {
        const auto& op = graph.ops[idx];
        const int op_layer_any = op_layer_idx_any(op);
        if (skip_logits_grad && is_logits_grad_op(op)) {
            continue;
        }

        // DEBUG: Trace flash attention backward op order and layer indices.
        static int flash_bwd_trace = 0;
        if (op.type == CompiledOpType::FlashAttentionBackward && flash_bwd_trace < 12) {
            int layer_idx_dbg = -1;
            std::string field_dbg;
            if (!op.inputs.empty()) {
                parse_block_param(op.inputs[1].name, layer_idx_dbg, field_dbg);
            }
            fprintf(stderr,
                    "[EXEC_FLASH_BWD] op=%zu layer_idx=%d in_out=%s in_lse=%s in_qkv=%s out=%s\n",
                    idx,
                    layer_idx_dbg,
                    op.inputs.size() > 1 ? op.inputs[1].name.c_str() : "<none>",
                    op.inputs.size() > 2 ? op.inputs[2].name.c_str() : "<none>",
                    op.inputs.size() > 3 ? op.inputs[3].name.c_str() : "<none>",
                    op.outputs.size() > 0 ? op.outputs[0].name.c_str() : "<none>");
            flash_bwd_trace++;
        }

        if (op.layer_start >= 0) {
            handle_layer_start(op.layer_start);
            // Optional trace: watch a specific layer's res_ffn buffer to catch corruption.
            if (std::getenv("SUROGATE_TRACE_RESFFN_LAYER")) {
                const int target_layer = env_int("SUROGATE_TRACE_RESFFN_LAYER", 27);
                static int resffn_trace_count = 0;
                if (target_layer >= 0 && target_layer < num_layers && resffn_trace_count < 64) {
                    Tensor& res = mRunState.get_residual(target_layer, mRunState.MainStream);
                    fprintf(stderr, "[BWD_RESFFN_TRACE] at_layer=%d target=%d res_ptr=%p\n",
                            op.layer_start, target_layer, res.Data);
                    std::vector<float> vals;
                    const std::size_t total = static_cast<std::size_t>(res.nelem());
                    const std::size_t n = std::min<std::size_t>(4096, total);
                    if (copy_tensor_sample_as_f32(res, n, vals)) {
                        std::size_t nan = 0;
                        std::size_t inf = 0;
                        float min_v = std::numeric_limits<float>::infinity();
                        float max_v = -std::numeric_limits<float>::infinity();
                        double sum_abs = 0.0;
                        for (float v : vals) {
                            if (std::isnan(v)) {
                                nan++;
                                continue;
                            }
                            if (std::isinf(v)) {
                                inf++;
                                continue;
                            }
                            min_v = std::min(min_v, v);
                            max_v = std::max(max_v, v);
                            sum_abs += std::fabs(v);
                        }
                        const double mean_abs = n ? (sum_abs / static_cast<double>(n)) : 0.0;
                        fprintf(stderr,
                                "[BWD_RESFFN_TRACE_STATS] target=%d n=%zu total=%zu nan=%zu inf=%zu min=%.6f max=%.6f mean_abs=%.6f\n",
                                target_layer, n, total, nan, inf,
                                std::isfinite(min_v) ? min_v : 0.0f,
                                std::isfinite(max_v) ? max_v : 0.0f,
                                static_cast<float>(mean_abs));
                    }
                    resffn_trace_count++;
                }
            }
            if (mRecomputeEnabled && mRecomputeFn) {
                const int layer_idx = op.layer_start;
                if (layer_idx >= 0 && layer_idx != mLastRecomputeLayer) {
                if (layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(layer_idx)]) {
                    clear_shared_grads(layer_idx);
                    layer_seen_any[static_cast<std::size_t>(layer_idx)] = true;
                }
                // DEBUG: Check MoE expert gate_up weights before recompute at layer start.
                static int moe_w_pre_recompute_start_trace = 0;
                if (layer_idx == 2 && moe_w_pre_recompute_start_trace < 1) {
                    const std::string w_name = "blocks[2].experts_gate_up";
                    if (mWeights.has(w_name)) {
                        Tensor& w = mWeights.get(w_name);
                        float w_min = 0.0f, w_max = 0.0f;
                        const bool w_nan = tensor_row_has_nan_or_inf(w, 122, &w_min, &w_max);
                        fprintf(stderr,
                                "[MOE_W_PRE_RECOMP_START] layer=%d name=%s ptr=%p device=%d nan=%d min=%.6f max=%.6f\n",
                                layer_idx, w_name.c_str(),
                                w.Data, w.Device,
                                w_nan ? 1 : 0, w_min, w_max);
                    } else {
                        fprintf(stderr, "[MOE_W_PRE_RECOMP_START] layer=%d name=%s missing\n",
                                layer_idx, w_name.c_str());
                    }
                    moe_w_pre_recompute_start_trace++;
                }
                mRecomputeFn(layer_idx, mB, mT, mRecomputeUseGraphs);
                record_recompute_sample(layer_idx);
                mLastRecomputeLayer = layer_idx;
            }
        }
        }

        if (mRecomputeEnabled && mRecomputeFn) {
            const int layer_idx = op_layer_idx(op);
            const int layer_idx_any = op_layer_idx_any(op);
            // Always recompute when switching layers. This is critical because:
            // - Shared buffers (ln1, ln2, qkv, mlp_up, swiglu) contain only ONE layer's data
            // - If the backward graph interleaves ops from different layers, we MUST
            //   recompute to ensure the correct layer's data is in the shared buffers
            // - The old check (missing_start || op_before_start) would skip recomputation
            //   for layer N's late ops if we had already visited layer N earlier, causing
            //   those ops to read stale data from whatever layer was recomputed last
            static int debug_op_count = 0;
            if (debug_op_count < 10) {
                fprintf(stderr, "[DEBUG_LAYER] op=%d layer_idx=%d layer_idx_any=%d mLastRecompute=%d op_type=%s\n",
                        static_cast<int>(idx), layer_idx, layer_idx_any, mLastRecomputeLayer, op_type_to_string(op.type));
                debug_op_count++;
            }
            // Use layer_idx_any as fallback when layer_idx is -1
            const int effective_layer_idx = (layer_idx >= 0) ? layer_idx : layer_idx_any;
            if (effective_layer_idx >= 0 && effective_layer_idx != mLastRecomputeLayer) {
                if (effective_layer_idx < num_layers && !layer_seen_any[static_cast<std::size_t>(effective_layer_idx)]) {
                    clear_shared_grads(effective_layer_idx);
                    layer_seen_any[static_cast<std::size_t>(effective_layer_idx)] = true;
                }
                // DEBUG: Check MoE expert gate_up weights before recompute for layer 2.
                static int moe_w_pre_recompute_trace = 0;
                if (effective_layer_idx == 2 && moe_w_pre_recompute_trace < 1) {
                    const std::string w_name = "blocks[2].experts_gate_up";
                    if (mWeights.has(w_name)) {
                        Tensor& w = mWeights.get(w_name);
                        float w_min = 0.0f, w_max = 0.0f;
                        const bool w_nan = tensor_row_has_nan_or_inf(w, 122, &w_min, &w_max);
                        fprintf(stderr,
                                "[MOE_W_PRE_RECOMP] layer=%d name=%s ptr=%p device=%d nan=%d min=%.6f max=%.6f\n",
                                effective_layer_idx, w_name.c_str(),
                                w.Data, w.Device,
                                w_nan ? 1 : 0, w_min, w_max);
                    } else {
                        fprintf(stderr, "[MOE_W_PRE_RECOMP] layer=%d name=%s missing\n",
                                effective_layer_idx, w_name.c_str());
                    }
                    moe_w_pre_recompute_trace++;
                }
                static int recompute_call_count = 0;
                if (recompute_call_count < 5) {
                    fprintf(stderr, "[CALLING_RECOMPUTE] layer_idx=%d (effective=%d) num_layers=%d\n", layer_idx, effective_layer_idx, num_layers);
                    recompute_call_count++;
                }
                mRecomputeFn(effective_layer_idx, mB, mT, mRecomputeUseGraphs);
                record_recompute_sample(effective_layer_idx);
                mLastRecomputeLayer = effective_layer_idx;
            }
        }

        try {
            switch (op.type) {
                // Explicit backward ops
                case CompiledOpType::ViewBackward:
                    dispatch_view_backward(op);
                    break;
                case CompiledOpType::AddBackward:
                    dispatch_add_backward(op);
                    break;
                case CompiledOpType::CrossEntropyLossBackward:
                    dispatch_cross_entropy_loss_backward(op);
                    break;
                case CompiledOpType::FusedLMHeadLossBackward:
                    dispatch_fused_lm_head_loss_backward(op);
                    break;
                case CompiledOpType::MatmulBackward:
                    dispatch_matmul_backward(op, hook);
                    // After the first matmul_backward (LM-head backward), free the output tensor
                    // to reclaim ~1.2GB of stack memory. The d_logits data has been consumed.
                    if (idx == 1) {
                        mRunState.temp_free(mRunState.non_block_activations().output);
                        mTemps.clear();
                        // Update initial_checkpoint to reflect the freed output tensor
                        // This prevents subsequent checkpoint restores from re-allocating it
                        initial_checkpoint = mRunState.Stack.checkpoint();
                    }
                    break;
                case CompiledOpType::BiasAddBackward:
                    dispatch_bias_add_backward(op);
                    break;
                case CompiledOpType::SwiGLUBackward:
                    dispatch_swiglu_backward(op);
                    break;
                case CompiledOpType::SiluBackward:
                    dispatch_silu_backward(op);
                    break;
                case CompiledOpType::MulBackward:
                    dispatch_mul_backward(op);
                    break;
                case CompiledOpType::MatmulSwiGLUBackward:
                    dispatch_matmul_swiglu_backward(op, hook);
                    break;
                case CompiledOpType::RoPEBackward:
                    dispatch_rope_backward(op);
                    break;
                case CompiledOpType::QKVQKNormRoPEBackward:
                    dispatch_qkv_qk_norm_rope_backward(op);
                    break;
                case CompiledOpType::FlashAttentionBackward:
                    dispatch_flash_attention_backward(op);
                    break;
                case CompiledOpType::ZerosBackward:
                    dispatch_zeros_backward(op);
                    break;
                case CompiledOpType::FusedResidualRMSNormBackward:
                    dispatch_fused_residual_rmsnorm_backward(op);
                    break;
                case CompiledOpType::EmbeddingBackward:
                    dispatch_embedding_backward(op);
                    break;

                // Forward ops that appear in backward graph (autodiff generates these)
                // View/reshape is the same operation in forward and backward - just reshapes gradient
                case CompiledOpType::View:
                    dispatch_view_backward(op);
                    break;
                // "add" ops in the backward graph are gradient-accumulation nodes,
                // so we must execute them as forward add (sum inputs), not add-backward.
                case CompiledOpType::Add:
                    dispatch_add(op);
                    break;
                // Zeros in backward is a no-op
                case CompiledOpType::Zeros:
                    dispatch_zeros_backward(op);
                    break;

                // MoE backward operations
                case CompiledOpType::MoESoftmaxBackward:
                    dispatch_moe_softmax_backward(op);
                    break;
                case CompiledOpType::MoESigmoidBackward:
                    dispatch_moe_sigmoid_backward(op);
                    break;
                case CompiledOpType::MoETopKBackward:
                    dispatch_moe_topk_backward(op);
                    break;
                case CompiledOpType::MoEPermuteBackward:
                    dispatch_moe_permute_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmGateUpBackward:
                    dispatch_moe_grouped_gemm_gate_up_backward(op);
                    break;
                case CompiledOpType::MoEGroupedGemmDownBackward:
                    dispatch_moe_grouped_gemm_down_backward(op);
                    break;
                case CompiledOpType::MoEUnpermuteBackward:
                    dispatch_moe_unpermute_backward(op);
                    break;

                // MoE forward ops that may appear in backward graph
                case CompiledOpType::MoESoftmax:
                case CompiledOpType::MoESigmoid:
                case CompiledOpType::MoETopK:
                case CompiledOpType::MoEPermute:
                case CompiledOpType::MoEGroupedGemmGateUp:
                case CompiledOpType::MoEGroupedGemmDown:
                case CompiledOpType::MoEUnpermute:
                case CompiledOpType::Silu:
                case CompiledOpType::Mul:
                    // These forward MoE ops may appear in backward graph due to autodiff
                    throw std::runtime_error("CompiledExecutor: MoE forward op in backward graph not yet supported");

                default: {
                    std::ostringstream oss;
                    oss << "CompiledExecutor: unsupported backward op type at idx " << idx
                        << " (type=" << op_type_to_string(op.type) << ", id=" << op.op_id << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            // Optional: detect the first op that corrupts a target res_ffn buffer.
            if (std::getenv("SUROGATE_TRACE_RESFFN_WATCH")) {
                const int target_layer = env_int("SUROGATE_TRACE_RESFFN_LAYER", 27);
                const int watch_layer = env_int("SUROGATE_TRACE_RESFFN_WATCH_LAYER", target_layer);
                static bool res_ok = true;
                static int res_watch_count = 0;
                if (op.layer_start >= 0 && op.layer_start == watch_layer) {
                    res_ok = true;
                    res_watch_count = 0;
                }
                const int op_layer = op_layer_idx_any(op);
                if (res_ok && op_layer == watch_layer && target_layer >= 0 && target_layer < num_layers) {
                    Tensor& res = mRunState.get_residual(target_layer, mRunState.MainStream);
                    std::vector<float> vals;
                    const std::size_t total = static_cast<std::size_t>(res.nelem());
                    const std::size_t n = std::min<std::size_t>(4096, total);
                    bool corrupt = false;
                    float max_abs = 0.0f;
                    std::size_t nan = 0;
                    if (copy_tensor_sample_as_f32(res, n, vals)) {
                        for (float v : vals) {
                            if (std::isnan(v) || std::isinf(v)) {
                                nan++;
                                corrupt = true;
                                continue;
                            }
                            max_abs = std::max(max_abs, std::fabs(v));
                        }
                        const float thresh = env_float("SUROGATE_TRACE_RESFFN_THRESH", 1e6f);
                        if (max_abs > thresh) {
                            corrupt = true;
                        }
                    }
                    if (corrupt && res_watch_count < 4) {
                        fprintf(stderr,
                                "[RESFFN_CORRUPT] watch_layer=%d target=%d op=%s op_id=%s max_abs=%.3e nan=%zu\n",
                                watch_layer, target_layer,
                                op_type_to_string(op.type), op.op_id.c_str(),
                                static_cast<double>(max_abs), nan);
                        res_watch_count++;
                        res_ok = false;
                    }
                }
            }

            // Trace first writers for layer-8 residual gradients to pinpoint the spike source.
            static int res_writer_trace_enabled = -1;
            if (res_writer_trace_enabled < 0) {
                res_writer_trace_enabled = (std::getenv("SUROGATE_TRACE_RES_WRITERS") != nullptr) ? 1 : 0;
            }
            if (res_writer_trace_enabled) {
                struct ResWriterState {
                    int micro_step = -1;
                    int res_att_count = 0;
                    int res_ffn_count = 0;
                };
                static ResWriterState res_state;
                if (res_state.micro_step != mMicroStep) {
                    res_state = {};
                    res_state.micro_step = mMicroStep;
                }
                auto log_res_writer = [&](const TensorRef& ref, int& counter) {
                    if (counter >= 12) {
                        return;
                    }
                    Tensor& t = resolve_tensor(ref);
                    fprintf(stderr,
                            "[RES_WRITER] micro_step=%d op_idx=%zu type=%s id=%s name=%s\n",
                            mMicroStep,
                            idx,
                            op_type_to_string(op.type),
                            op.op_id.c_str(),
                            ref.name.c_str());
                    log_tensor_mag_unbounded("RES_WRITER_TENSOR", op_layer_any, ref.name, t, 4096);
                    counter++;
                };
                for (const auto& out_ref : op.outputs) {
                    const std::string base = strip_ssa_suffix(out_ref.name);
                    if (base == "d_blocks[8].res_att") {
                        log_res_writer(out_ref, res_state.res_att_count);
                    } else if (base == "d_blocks[8].res_ffn") {
                        log_res_writer(out_ref, res_state.res_ffn_count);
                    }
                }
            }

            // Track when key gradients first spike or go NaN.
            static std::unordered_map<std::string, int> watch_counts;
            auto log_watch = [&](const TensorRef& ref) {
                if (ref.name.empty()) return;
                const std::string base = strip_ssa_suffix(ref.name);
                if (base != "d_blocks[0].ln1" &&
                    base != "d_blocks[0].ln2" &&
                    base != "d_blocks[0].res_ffn" &&
                    base != "d_blocks[0].mlp_up" &&
                    base != "d_blocks[0].swiglu") {
                    return;
                }
                int& count = watch_counts[base];
                if (count >= 8) {
                    return;
                }
                Tensor& t = resolve_tensor(ref);
                fprintf(stderr,
                        "[BWD_WATCH] op_idx=%zu type=%s id=%s name=%s\n",
                        idx, op_type_to_string(op.type), op.op_id.c_str(), ref.name.c_str());
                log_tensor_stats_ex("BWD_WATCH_OUT", op_layer_any, ref.name, t, 4096, true);
                count++;
            };
            for (const auto& out_ref : op.outputs) {
                log_watch(out_ref);
            }

            static bool emb_nan_logged = false;
            if (!emb_nan_logged && d_embeddings_buf.Data && tensor_sample_has_nan_or_inf(d_embeddings_buf, 3)) {
                fprintf(stderr,
                        "[EMB_DOUT_NAN_AFTER] micro_step=%d op_idx=%zu type=%s id=%s\n",
                        micro_step, idx, op_type_to_string(op.type), op.op_id.c_str());
                log_tensor_stats_ex("EMB_DOUT_NAN_AFTER", -1, "d_embeddings", d_embeddings_buf, 4096, true);
                const long hidden = d_embeddings_buf.Rank > 2 ? d_embeddings_buf.Sizes[2] : 0;
                if (hidden > 0) {
                    const std::size_t off = static_cast<std::size_t>(3) * static_cast<std::size_t>(hidden);
                    log_tensor_sample_stats("EMB_DOUT_NAN_ROW3", d_embeddings_buf, off,
                                            static_cast<std::size_t>(hidden));
                }
                emb_nan_logged = true;
            }

            // Memory management - restore stack checkpoint periodically to free temporaries
            // This prevents memory accumulation during backward pass
            // Option 1: At layer boundaries if annotated
            // TEMPORARILY DISABLED for MoE models due to tensor corruption issues
            // TODO: Fix proper tensor lifetime tracking for MoE backward
            if (mConfig.NumExperts == 0 && op.layer_end >= 0 && op.layer_end != last_layer_restored) {
                // Restore stack and clear temps
                mRunState.Stack.restore(initial_checkpoint);
                mTemps.clear();
                prune_stack_tensors(op.layer_end);
                // Note: cudnn_workspace is persistently allocated, no need to clear
                // Clear stack-allocated tensor pointers in simplified_acts/grads for this layer.
                // These pointers become stale after checkpoint restore.
                if (mRunState.ffn_temps_on_stack()) {
                    auto& acts = mRunState.simplified_acts(op.layer_end);
                    acts.mlp_up.Data = nullptr;
                    acts.swiglu.Data = nullptr;
                }
                if (mRunState.large_bwd_temps_on_stack()) {
                    auto& grads_to_clear = mRunState.simplified_grads(op.layer_end);
                    grads_to_clear.d_qkv.Data = nullptr;
                    grads_to_clear.d_mlp_up.Data = nullptr;
                    grads_to_clear.d_swiglu.Data = nullptr;
                }
                last_layer_restored = op.layer_end;
            }
            // Option 2: Every N ops as fallback (catches non-annotated layers)
            // NOTE: When recompute is disabled, we cannot aggressively prune tensors because
            // the backward graph may reference intermediate tensors (like d_blocks[N].view_K)
            // that were produced earlier but are still needed. The stack restore + prune
            // would remove these tensors from mTensorMap, causing "tensor not found" errors.
            // For now, skip periodic cleanup when recompute is disabled to preserve correctness.
            // Memory usage will be higher but the backward pass will complete successfully.
            // TODO: Implement proper tensor lifetime tracking to enable safe pruning.
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "CompiledExecutor backward op " << idx << " (type=" << op_type_to_string(op.type)
                << ", id=" << op.op_id << "): " << e.what();
            // Add inputs/outputs for debugging
            oss << "\n  inputs: [";
            for (size_t i = 0; i < op.inputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.inputs[i].name << "(slot=" << static_cast<int>(op.inputs[i].slot) << ")";
            }
            oss << "]";
            oss << "\n  outputs: [";
            for (size_t i = 0; i < op.outputs.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.outputs[i].name << "(slot=" << static_cast<int>(op.outputs[i].slot) << ")";
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }

    }

    // Final cleanup - pass -1 to allow full pruning (backward complete)
    mRunState.Stack.restore(initial_checkpoint);
    prune_stack_tensors(-1);
    mTemps.clear();

    if (mWeightManager && (mWeightManager->is_streaming_enabled() || mWeightManager->is_offload_enabled())) {
        mWeightManager->release_final_norm(mRunState.MainStream);
        if (mOptions.LMHeadChunks <= 1) {
            mWeightManager->release_lm_head(mRunState.MainStream);
        }
    }
}

}  // namespace dsl
