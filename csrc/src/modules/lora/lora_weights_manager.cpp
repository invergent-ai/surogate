// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_weights_manager.h"

#include <atomic>
#include <cmath>
#include <fmt/format.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"

namespace modules {

namespace {
inline bool copy_tensor_sample_offset_as_f32(const Tensor& t,
                                             std::size_t elem_offset,
                                             std::size_t count,
                                             std::vector<float>& out) {
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

inline std::size_t tensor_row_width(const Tensor& t) {
    if (t.Rank <= 1) {
        return static_cast<std::size_t>(t.nelem());
    }
    std::size_t row_width = 1;
    for (int i = 1; i < t.Rank; ++i) {
        row_width *= static_cast<std::size_t>(t.Sizes[i]);
    }
    return row_width;
}

inline bool tensor_row_has_nan_or_inf(const Tensor& t, long row_idx, float* out_min, float* out_max) {
    const std::size_t row_width = tensor_row_width(t);
    if (row_width == 0) {
        return false;
    }
    std::vector<float> vals(row_width);
    if (!copy_tensor_sample_offset_as_f32(t, static_cast<std::size_t>(row_idx) * row_width, row_width, vals)) {
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

inline bool find_first_nan_row_local(const Tensor& t, long* out_row, float* out_min, float* out_max) {
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

inline bool lora_sync_nan_trace_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("SUROGATE_LORA_SYNC_NAN_TRACE");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled == 1;
}

inline int lora_sync_nan_trace_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_LORA_SYNC_NAN_TRACE_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

inline int lora_sync_nan_trace_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char* env = std::getenv("SUROGATE_LORA_SYNC_NAN_TRACE_LIMIT");
        limit = env ? std::atoi(env) : 32;
    }
    return limit;
}

inline bool lora_sync_nan_trace_should_log(int layer_idx) {
    if (!lora_sync_nan_trace_enabled()) return false;
    const int target = lora_sync_nan_trace_layer();
    if (target >= 0 && target != layer_idx) return false;
    static std::atomic<int> counter{0};
    const int limit = lora_sync_nan_trace_limit();
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

inline int lora_b_guard_layer() {
    static int layer = -2;
    if (layer == -2) {
        const char* env = std::getenv("SUROGATE_LORA_B_GUARD_LAYER");
        layer = env ? std::atoi(env) : -1;
    }
    return layer;
}

inline int lora_b_guard_limit() {
    static int limit = -1;
    if (limit < 0) {
        const char* env = std::getenv("SUROGATE_LORA_B_GUARD_LIMIT");
        limit = env ? std::atoi(env) : 32;
    }
    return limit;
}

inline bool lora_b_guard_should_log(int layer_idx) {
    if (!lora_b_guard_enabled()) return false;
    const int target = lora_b_guard_layer();
    if (target >= 0 && target != layer_idx) return false;
    static std::atomic<int> counter{0};
    const int limit = lora_b_guard_limit();
    if (limit <= 0) return false;
    const int idx = counter.fetch_add(1);
    return idx < limit;
}

inline bool lora_sync_nan_trace_can_copy(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        return false;
    }
    return status == cudaStreamCaptureStatusNone;
}

inline void log_lora_sync_nan(const Tensor& t, int layer_idx, const char* tag) {
    if (!t.Data) return;
    long row = -1;
    float min_val = 0.0f;
    float max_val = 0.0f;
    if (!find_first_nan_row_local(t, &row, &min_val, &max_val)) {
        return;
    }
    std::cerr << fmt::format("[LORA_SYNC_NAN] layer={} tag={} row={} min={} max={} dtype={} ptr={}\n",
                             layer_idx,
                             tag ? tag : "<unnamed>",
                             row,
                             min_val,
                             max_val,
                             dtype_to_str(t.DType),
                             static_cast<const void*>(t.Data));
}

struct LoRABGuardKey {
    const void* ptr = nullptr;
    int layer = -1;
};

struct LoRABGuardKeyHash {
    std::size_t operator()(const LoRABGuardKey& k) const noexcept {
        const auto h1 = std::hash<const void*>{}(k.ptr);
        const auto h2 = std::hash<int>{}(k.layer);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

struct LoRABGuardKeyEq {
    bool operator()(const LoRABGuardKey& a, const LoRABGuardKey& b) const noexcept {
        return a.ptr == b.ptr && a.layer == b.layer;
    }
};

std::unordered_map<LoRABGuardKey, LoRABGuardSample, LoRABGuardKeyHash, LoRABGuardKeyEq> g_lora_b_guard_samples;
std::mutex g_lora_b_guard_mutex;
}  // namespace

void record_lora_b_guard_sample(const void* ptr,
                                int layer_idx,
                                const char* tag,
                                const std::array<float, 8>& vals) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(g_lora_b_guard_mutex);
    LoRABGuardKey key{ptr, layer_idx};
    auto& entry = g_lora_b_guard_samples[key];
    entry.vals = vals;
    entry.tag = tag ? tag : "";
}

bool fetch_lora_b_guard_sample(const void* ptr,
                               int layer_idx,
                               LoRABGuardSample& out) {
    if (!ptr) return false;
    std::lock_guard<std::mutex> lock(g_lora_b_guard_mutex);
    LoRABGuardKey key{ptr, layer_idx};
    auto it = g_lora_b_guard_samples.find(key);
    if (it == g_lora_b_guard_samples.end()) {
        return false;
    }
    out = it->second;
    return true;
}

ModularLoRAWeightsManager::ModularLoRAWeightsManager(const Config& config, TensorAllocator& allocator)
    : mConfig(config), mAllocator(&allocator) {
    mMaster.config = config.lora_config;
    mWork.config = config.lora_config;

    if (!enabled()) {
        return;
    }

    auto ctx = mAllocator->with_context("Modular_LoRA_Weights");
    mMaster.blocks.resize(config.num_layers);
    mWork.blocks.resize(config.num_layers);
    for (int l = 0; l < config.num_layers; ++l) {
        allocate_block_weights(l);
    }
}

void ModularLoRAWeightsManager::allocate_layer_weights(
    LoRALayerWeights<TensorShard>& shard,
    LoRALayerWeights<Tensor>& work,
    int in_features,
    int out_features,
    const std::string& name) {

    const int r = mConfig.lora_config.rank;
    const ETensorDType master_dtype = mConfig.lora_config.dtype;
    const ETensorDType work_dtype = mConfig.work_dtype;

    // Data-parallel LoRA: replicate weights on all ranks (no sharding yet).
    shard.A = TensorShard(mAllocator->allocate(master_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {r, in_features}));
    shard.B = mAllocator->allocate_shard(master_dtype, /*shard_idx=*/0, /*num_shards=*/1, (name + "_B").c_str(), {out_features, r});

    work.A = mAllocator->allocate(work_dtype, (name + "_A_work").c_str(), EAllocationType::ON_DEVICE, {r, in_features});
    work.B = mAllocator->allocate(work_dtype, (name + "_B_work").c_str(), EAllocationType::ON_DEVICE, {out_features, r});
}

void ModularLoRAWeightsManager::allocate_block_weights(int layer_idx) {
    if (!enabled()) return;

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int Hq = mConfig.num_query_heads;
    const int Hkv = mConfig.num_kv_heads;
    const int Hs = mConfig.head_size;
    const int q_out = Hq * Hs;
    const int kv_out = Hkv * Hs;

    auto& master = mMaster.blocks[layer_idx];
    auto& work = mWork.blocks[layer_idx];

    const std::string prefix = fmt::format("lora_layer_{}", layer_idx);

    if (mConfig.lora_config.applies_to_q()) {
        master.attention.q.emplace();
        work.attention.q.emplace();
        allocate_layer_weights(*master.attention.q, *work.attention.q, /*in=*/C, /*out=*/q_out, prefix + "_q");
    }
    if (mConfig.lora_config.applies_to_k()) {
        master.attention.k.emplace();
        work.attention.k.emplace();
        allocate_layer_weights(*master.attention.k, *work.attention.k, /*in=*/C, /*out=*/kv_out, prefix + "_k");
    }
    if (mConfig.lora_config.applies_to_v()) {
        master.attention.v.emplace();
        work.attention.v.emplace();
        allocate_layer_weights(*master.attention.v, *work.attention.v, /*in=*/C, /*out=*/kv_out, prefix + "_v");
    }
    if (mConfig.lora_config.applies_to_o()) {
        master.attention.o.emplace();
        work.attention.o.emplace();
        allocate_layer_weights(*master.attention.o, *work.attention.o, /*in=*/q_out, /*out=*/C, prefix + "_o");
    }

    // MLP LoRA: For dense models, use standard MLP LoRA. For MoE models, use per-expert LoRA.
    if (mConfig.is_moe && mConfig.num_experts > 0) {
        master.moe.use_grouped = true;
        work.moe.use_grouped = true;
        const bool has_mlp_lora = mConfig.lora_config.applies_to_gate() ||
                                   mConfig.lora_config.applies_to_up() ||
                                   mConfig.lora_config.applies_to_down();
        if (has_mlp_lora) {
            allocate_grouped_moe_weights(master.moe.grouped, work.moe.grouped, layer_idx);
        }

        // Allocate router gate LoRA storage when train_router is enabled
        // Router is a linear layer: (hidden_size -> num_experts), so:
        // - lora_A: (rank, hidden_size) - projects input to low-rank space
        // - lora_B: (num_experts, rank) - projects from low-rank to expert logits
        if (mConfig.train_router) {
            const int E = mConfig.num_experts;
            const std::string router_prefix = fmt::format("lora_layer_{}_router", layer_idx);
            master.router.emplace();
            work.router.emplace();
            allocate_layer_weights(*master.router, *work.router, /*in=*/C, /*out=*/E, router_prefix);
        }
    } else {
        // Dense model: standard MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) {
            master.mlp.gate.emplace();
            work.mlp.gate.emplace();
            allocate_layer_weights(*master.mlp.gate, *work.mlp.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
        }
        if (mConfig.lora_config.applies_to_up()) {
            master.mlp.up.emplace();
            work.mlp.up.emplace();
            allocate_layer_weights(*master.mlp.up, *work.mlp.up, /*in=*/C, /*out=*/D, prefix + "_up");
        }
        if (mConfig.lora_config.applies_to_down()) {
            master.mlp.down.emplace();
            work.mlp.down.emplace();
            allocate_layer_weights(*master.mlp.down, *work.mlp.down, /*in=*/D, /*out=*/C, prefix + "_down");
        }
    }
}

void ModularLoRAWeightsManager::allocate_grouped_moe_weights(
    LoRAGroupedExpertWeights<TensorShard>& master_moe,
    LoRAGroupedExpertWeights<Tensor>& work_moe,
    int layer_idx) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.effective_moe_intermediate();
    const int num_experts = mConfig.num_experts;
    const int r = mConfig.lora_config.rank;
    const ETensorDType master_dtype = mConfig.lora_config.dtype;
    const ETensorDType work_dtype = mConfig.work_dtype;
    const std::string prefix = fmt::format("lora_layer_{}_moe", layer_idx);

    auto allocate_grouped = [&](auto& m_layer, auto& w_layer, int in, int out, const std::string& name) {
        m_layer.emplace();
        w_layer.emplace();

        m_layer->A = TensorShard(mAllocator->allocate(master_dtype, (name + "_A").c_str(), EAllocationType::ON_DEVICE, {num_experts, r, in}));
        m_layer->B = mAllocator->allocate_shard(master_dtype, 0, 1, (name + "_B").c_str(), {num_experts, out, r});

        w_layer->A = mAllocator->allocate(work_dtype, (name + "_A_work").c_str(), EAllocationType::ON_DEVICE, {num_experts, r, in});
        w_layer->B = mAllocator->allocate(work_dtype, (name + "_B_work").c_str(), EAllocationType::ON_DEVICE, {num_experts, out, r});
    };

    if (mConfig.lora_config.applies_to_gate()) {
        allocate_grouped(master_moe.gate, work_moe.gate, C, D, prefix + "_gate");
    }
    if (mConfig.lora_config.applies_to_up()) {
        allocate_grouped(master_moe.up, work_moe.up, C, D, prefix + "_up");
    }
    if (mConfig.lora_config.applies_to_down()) {
        allocate_grouped(master_moe.down, work_moe.down, D, C, prefix + "_down");
    }
}

void ModularLoRAWeightsManager::allocate_expert_weights(
    LoRAExpertWeights<TensorShard>& master_expert,
    LoRAExpertWeights<Tensor>& work_expert,
    int layer_idx, int expert_idx) {

    const int C = mConfig.hidden_size;
    const int D = mConfig.effective_moe_intermediate();
    const std::string prefix = fmt::format("lora_layer_{}_expert_{}", layer_idx, expert_idx);

    if (mConfig.lora_config.applies_to_gate()) {
        master_expert.gate.emplace();
        work_expert.gate.emplace();
        allocate_layer_weights(*master_expert.gate, *work_expert.gate, /*in=*/C, /*out=*/D, prefix + "_gate");
    }
    if (mConfig.lora_config.applies_to_up()) {
        master_expert.up.emplace();
        work_expert.up.emplace();
        allocate_layer_weights(*master_expert.up, *work_expert.up, /*in=*/C, /*out=*/D, prefix + "_up");
    }
    if (mConfig.lora_config.applies_to_down()) {
        master_expert.down.emplace();
        work_expert.down.emplace();
        allocate_layer_weights(*master_expert.down, *work_expert.down, /*in=*/D, /*out=*/C, prefix + "_down");
    }
}

void ModularLoRAWeightsManager::random_init(int seed, NCCLCommunicator& comm) {
    if (!enabled()) return;

    auto init_layer = [&](std::optional<LoRALayerWeights<TensorShard>>& layer,
                          int in_features,
                          unsigned long long subsequence) {
        if (!layer.has_value()) return;
        // std consistent with kaiming_uniform_(a=sqrt(5)) => bound = 1/sqrt(fan_in)
        float std_a = 1.0f / std::sqrt(3.0f * static_cast<float>(in_features));
        fill_normal(layer->A, layer->A.nelem(), 0.0f, std_a, seed, subsequence, nullptr);
        fill_zero(layer->B, nullptr);
    };

    auto init_grouped = [&](std::optional<LoRAGroupedLayerWeights<TensorShard>>& layer,
                            int in_features,
                            unsigned long long subsequence) {
        if (!layer.has_value()) return;
        float std_a = 1.0f / std::sqrt(3.0f * static_cast<float>(in_features));
        fill_normal(layer->A, layer->A.nelem(), 0.0f, std_a, seed, subsequence, nullptr);
        fill_zero(layer->B, nullptr);
    };

    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;
    const int D_moe = mConfig.effective_moe_intermediate();
    const int q_out = mConfig.num_query_heads * mConfig.head_size;
    const int E = mConfig.num_experts;

    for (int l = 0; l < mConfig.num_layers; ++l) {
        auto& b = mMaster.blocks[l];
        unsigned long long base = static_cast<unsigned long long>(l) * 32ULL;
        init_layer(b.attention.q, C, base + 0);
        init_layer(b.attention.k, C, base + 1);
        init_layer(b.attention.v, C, base + 2);
        init_layer(b.attention.o, q_out, base + 3);

        // Dense MLP LoRA
        init_layer(b.mlp.gate, C, base + 4);
        init_layer(b.mlp.up, C, base + 5);
        init_layer(b.mlp.down, D, base + 6);

        // MoE expert LoRA
        if (b.moe.use_grouped) {
            init_grouped(b.moe.grouped.gate, C, base + 8);
            init_grouped(b.moe.grouped.up, C, base + 9);
            init_grouped(b.moe.grouped.down, D_moe, base + 10);
        } else {
            for (int e = 0; e < (int)b.moe.experts.size(); ++e) {
                auto& expert = b.moe.experts[e];
                // Use separate subsequence space for each expert to avoid correlation
                unsigned long long expert_base = base + 8ULL + static_cast<unsigned long long>(e) * 4ULL;
                init_layer(expert.gate, C, expert_base + 0);
                init_layer(expert.up, C, expert_base + 1);
                init_layer(expert.down, D_moe, expert_base + 2);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

void ModularLoRAWeightsManager::import_from_file(const std::string& file_name, NCCLCommunicator& comm) {
    if (!enabled()) return;
    load_safetensors(file_name, *this, /*allow_cast=*/true);
    CUDA_CHECK(cudaDeviceSynchronize());
    comm.barrier();
}

void ModularLoRAWeightsManager::export_to_file(const std::string& file_name, NCCLCommunicator& comm) const {
    if (!enabled()) return;
    if (comm.rank() == 0) {
        write_safetensors(file_name, const_cast<ModularLoRAWeightsManager&>(*this));
    }
    comm.barrier();
}

LoRABlockWeights<Tensor>& ModularLoRAWeightsManager::get_block(int layer_idx, cudaStream_t stream) {
    auto& work = mWork.blocks[layer_idx];
    if (!enabled()) return work;

    auto& master = mMaster.blocks[layer_idx];

    auto sync_tensor = [&](Tensor& dst_t, const TensorShard& src_t, const char* name) {
        if (!dst_t.Data || !src_t.Data) return;
        if (dst_t.nelem() != src_t.nelem()) {
            throw std::logic_error(fmt::format("ModularLoRAWeightsManager::get_block: {} nelem mismatch (dst={}, src={})",
                                               name, dst_t.nelem(), src_t.nelem()));
        }

        if (dst_t.DType == src_t.DType) {
            CUDA_CHECK(cudaMemcpyAsync(dst_t.Data, src_t.Data, dst_t.bytes(), cudaMemcpyDeviceToDevice, stream));
            return;
        }

        if (dst_t.DType == ETensorDType::BF16 && src_t.DType == ETensorDType::FP32) {
            convert_dtype(dst_t.get<nv_bfloat16>(), src_t.get<float>(), dst_t.nelem(), stream);
            return;
        }
        if (dst_t.DType == ETensorDType::FP32 && src_t.DType == ETensorDType::BF16) {
            convert_dtype(dst_t.get<float>(), src_t.get<nv_bfloat16>(), dst_t.nelem(), stream);
            return;
        }

        throw std::logic_error(fmt::format(
            "ModularLoRAWeightsManager::get_block: unsupported dtype cast for {} (src={}, dst={})",
            name, dtype_to_str(src_t.DType), dtype_to_str(dst_t.DType)));
    };

    auto sync_layer = [&](std::optional<LoRALayerWeights<Tensor>>& dst,
                          const std::optional<LoRALayerWeights<TensorShard>>& src,
                          const char* layer_name) {
        if (!dst.has_value() || !src.has_value()) return;
        sync_tensor(dst->A, src->A, (std::string(layer_name) + ".A").c_str());
        sync_tensor(dst->B, src->B, (std::string(layer_name) + ".B").c_str());
    };

    auto sync_grouped = [&](std::optional<LoRAGroupedLayerWeights<Tensor>>& dst,
                            const std::optional<LoRAGroupedLayerWeights<TensorShard>>& src,
                            const char* layer_name) {
        if (!dst.has_value() || !src.has_value()) return;
        sync_tensor(dst->A, src->A, (std::string(layer_name) + ".A").c_str());
        sync_tensor(dst->B, src->B, (std::string(layer_name) + ".B").c_str());
    };

    sync_layer(work.attention.q, master.attention.q, "q_proj");
    sync_layer(work.attention.k, master.attention.k, "k_proj");
    sync_layer(work.attention.v, master.attention.v, "v_proj");
    sync_layer(work.attention.o, master.attention.o, "o_proj");

    // MLP LoRA
    if (work.moe.use_grouped) {
        sync_grouped(work.moe.grouped.gate, master.moe.grouped.gate, "moe_gate_grouped");
        sync_grouped(work.moe.grouped.up, master.moe.grouped.up, "moe_up_grouped");
        sync_grouped(work.moe.grouped.down, master.moe.grouped.down, "moe_down_grouped");
    } else {
        // Dense MLP LoRA
        sync_layer(work.mlp.gate, master.mlp.gate, "gate_proj");
        sync_layer(work.mlp.up, master.mlp.up, "up_proj");
        sync_layer(work.mlp.down, master.mlp.down, "down_proj");

        // MoE expert LoRA
        for (int e = 0; e < (int)master.moe.experts.size(); ++e) {
            auto& master_expert = master.moe.experts[e];
            auto& work_expert = work.moe.experts[e];
            std::string expert_prefix = fmt::format("expert_{}", e);
            sync_layer(work_expert.gate, master_expert.gate, (expert_prefix + "_gate").c_str());
            sync_layer(work_expert.up, master_expert.up, (expert_prefix + "_up").c_str());
            sync_layer(work_expert.down, master_expert.down, (expert_prefix + "_down").c_str());
        }
    }

    // Sync router LoRA (when train_router is enabled)
    sync_layer(work.router, master.router, "router");

    if (lora_b_guard_should_log(layer_idx) && lora_sync_nan_trace_can_copy(stream)) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto record_sample = [&](const Tensor& t, const char* tag) {
            if (!t.Data) return;
            std::vector<float> vals;
            if (!copy_tensor_sample_offset_as_f32(t, 0, 8, vals)) return;
            std::array<float, 8> out{};
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] = i < vals.size() ? vals[i] : 0.0f;
            }
            record_lora_b_guard_sample(t.Data, layer_idx, tag, out);
        };
        if (work.attention.q.has_value()) {
            record_sample(work.attention.q->B, "q_proj.B");
        }
        if (work.attention.k.has_value()) {
            record_sample(work.attention.k->B, "k_proj.B");
        }
        if (work.attention.v.has_value()) {
            record_sample(work.attention.v->B, "v_proj.B");
        }
    }

    if (lora_sync_nan_trace_should_log(layer_idx) && lora_sync_nan_trace_can_copy(stream)) {
        if (work.attention.q.has_value()) {
            log_lora_sync_nan(work.attention.q->A, layer_idx, "q_proj.A");
            log_lora_sync_nan(work.attention.q->B, layer_idx, "q_proj.B");
        }
        if (work.attention.k.has_value()) {
            log_lora_sync_nan(work.attention.k->A, layer_idx, "k_proj.A");
            log_lora_sync_nan(work.attention.k->B, layer_idx, "k_proj.B");
        }
        if (work.attention.v.has_value()) {
            log_lora_sync_nan(work.attention.v->A, layer_idx, "v_proj.A");
            log_lora_sync_nan(work.attention.v->B, layer_idx, "v_proj.B");
        }
    }

    return work;
}

LoRABlockWeights<Tensor>& ModularLoRAWeightsManager::peek_block(int layer_idx) {
    if (!enabled()) {
        return mWork.blocks[layer_idx];
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mWork.blocks.size())) {
        throw std::out_of_range("ModularLoRAWeightsManager::peek_block layer_idx out of range");
    }
    return mWork.blocks[layer_idx];
}

LoRABlockWeights<TensorShard>& ModularLoRAWeightsManager::get_master_block(int layer_idx, cudaStream_t stream) {
    (void)stream;
    return mMaster.blocks[layer_idx];
}

std::size_t ModularLoRAWeightsManager::num_parameters() const {
    if (!enabled()) return 0;

    const std::size_t r = static_cast<std::size_t>(mConfig.lora_config.rank);
    const std::size_t C = static_cast<std::size_t>(mConfig.hidden_size);
    const std::size_t D = static_cast<std::size_t>(mConfig.intermediate_size);
    const std::size_t D_moe = static_cast<std::size_t>(mConfig.effective_moe_intermediate());
    const std::size_t Hq = static_cast<std::size_t>(mConfig.num_query_heads);
    const std::size_t Hkv = static_cast<std::size_t>(mConfig.num_kv_heads);
    const std::size_t Hs = static_cast<std::size_t>(mConfig.head_size);
    const std::size_t q_out = Hq * Hs;
    const std::size_t kv_out = Hkv * Hs;
    const std::size_t E = static_cast<std::size_t>(mConfig.num_experts);

    std::size_t per_layer = 0;

    // Attention LoRA parameters
    if (mConfig.lora_config.applies_to_q()) per_layer += r * C + q_out * r;
    if (mConfig.lora_config.applies_to_k()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_v()) per_layer += r * C + kv_out * r;
    if (mConfig.lora_config.applies_to_o()) per_layer += r * q_out + C * r;

    // MLP LoRA parameters (dense or MoE)
    if (mConfig.is_moe && E > 0) {
        // Per-expert LoRA for MoE models
        std::size_t per_expert = 0;
        if (mConfig.lora_config.applies_to_gate()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_up()) per_expert += r * C + D_moe * r;
        if (mConfig.lora_config.applies_to_down()) per_expert += r * D_moe + C * r;
        per_layer += per_expert * E;

        // Router LoRA parameters (when train_router is enabled)
        // Router shape: (hidden_size -> num_experts), so lora_A: (r, C), lora_B: (E, r)
        if (mConfig.train_router) {
            per_layer += r * C + E * r;
        }
    } else {
        // Dense MLP LoRA
        if (mConfig.lora_config.applies_to_gate()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_up()) per_layer += r * C + D * r;
        if (mConfig.lora_config.applies_to_down()) per_layer += r * D + C * r;
    }

    return per_layer * static_cast<std::size_t>(mConfig.num_layers);
}

void ModularLoRAWeightsManager::iterate_tensors(
    const std::function<void(std::string, const TensorShard&)>& callback) {
    if (!enabled()) return;

    for (int l = 0; l < (int)mMaster.blocks.size(); ++l) {
        std::string prefix = fmt::format("base_model.model.model.layers.{}", l);
        auto& block = mMaster.blocks[l];

        if (block.attention.q.has_value()) {
            callback(prefix + ".self_attn.q_proj.lora_A.weight", block.attention.q->A);
            callback(prefix + ".self_attn.q_proj.lora_B.weight", block.attention.q->B);
        }
        if (block.attention.k.has_value()) {
            callback(prefix + ".self_attn.k_proj.lora_A.weight", block.attention.k->A);
            callback(prefix + ".self_attn.k_proj.lora_B.weight", block.attention.k->B);
        }
        if (block.attention.v.has_value()) {
            callback(prefix + ".self_attn.v_proj.lora_A.weight", block.attention.v->A);
            callback(prefix + ".self_attn.v_proj.lora_B.weight", block.attention.v->B);
        }
        if (block.attention.o.has_value()) {
            callback(prefix + ".self_attn.o_proj.lora_A.weight", block.attention.o->A);
            callback(prefix + ".self_attn.o_proj.lora_B.weight", block.attention.o->B);
        }

        // Dense MLP LoRA
        if (block.mlp.gate.has_value()) {
            callback(prefix + ".mlp.gate_proj.lora_A.weight", block.mlp.gate->A);
            callback(prefix + ".mlp.gate_proj.lora_B.weight", block.mlp.gate->B);
        }
        if (block.mlp.up.has_value()) {
            callback(prefix + ".mlp.up_proj.lora_A.weight", block.mlp.up->A);
            callback(prefix + ".mlp.up_proj.lora_B.weight", block.mlp.up->B);
        }
        if (block.mlp.down.has_value()) {
            callback(prefix + ".mlp.down_proj.lora_A.weight", block.mlp.down->A);
            callback(prefix + ".mlp.down_proj.lora_B.weight", block.mlp.down->B);
        }

        // MoE expert LoRA
        if (block.moe.use_grouped) {
            // Export grouped tensors in per-expert format for PEFT compatibility
            // Grouped tensors have shape [num_experts, ...], slice along dim 0
            auto& g = block.moe.grouped;
            const int num_experts = mConfig.num_experts;

            auto export_grouped_layer = [&](const std::optional<LoRAGroupedLayerWeights<TensorShard>>& layer,
                                            const char* proj_name) {
                if (!layer.has_value() || !layer->has_value()) return;
                for (int e = 0; e < num_experts; ++e) {
                    std::string expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);
                    // Slice out expert e from dim 0: A[e,:,:] and B[e,:,:]
                    TensorShard A_slice = TensorShard(slice(layer->A, 0, e, e + 1));
                    TensorShard B_slice = TensorShard(slice(layer->B, 0, e, e + 1));
                    // Remove the leading dimension of size 1 by adjusting shape
                    // A: [1, rank, in] -> [rank, in], B: [1, out, rank] -> [out, rank]
                    A_slice.Rank = layer->A.Rank - 1;
                    B_slice.Rank = layer->B.Rank - 1;
                    for (int d = 0; d < A_slice.Rank; ++d) A_slice.Sizes[d] = A_slice.Sizes[d + 1];
                    for (int d = 0; d < B_slice.Rank; ++d) B_slice.Sizes[d] = B_slice.Sizes[d + 1];
                    // Update global shape to match local shape (not sharded)
                    std::copy(A_slice.Sizes.begin(), A_slice.Sizes.end(), A_slice.GlobalShape.begin());
                    std::copy(B_slice.Sizes.begin(), B_slice.Sizes.end(), B_slice.GlobalShape.begin());
                    callback(expert_prefix + "." + proj_name + ".lora_A.weight", A_slice);
                    callback(expert_prefix + "." + proj_name + ".lora_B.weight", B_slice);
                }
            };

            export_grouped_layer(g.gate, "gate_proj");
            export_grouped_layer(g.up, "up_proj");
            export_grouped_layer(g.down, "down_proj");
        } else {
            // MoE expert LoRA (HuggingFace naming convention: .mlp.experts.{e}.{proj})
            for (int e = 0; e < (int)block.moe.experts.size(); ++e) {
                auto& expert = block.moe.experts[e];
                std::string expert_prefix = fmt::format("{}.mlp.experts.{}", prefix, e);

                if (expert.gate.has_value()) {
                    callback(expert_prefix + ".gate_proj.lora_A.weight", expert.gate->A);
                    callback(expert_prefix + ".gate_proj.lora_B.weight", expert.gate->B);
                }
                if (expert.up.has_value()) {
                    callback(expert_prefix + ".up_proj.lora_A.weight", expert.up->A);
                    callback(expert_prefix + ".up_proj.lora_B.weight", expert.up->B);
                }
                if (expert.down.has_value()) {
                    callback(expert_prefix + ".down_proj.lora_A.weight", expert.down->A);
                    callback(expert_prefix + ".down_proj.lora_B.weight", expert.down->B);
                }
            }
        }

        // Export router LoRA (if train_router is enabled) - PEFT-compatible format
        if (block.router.has_value() && block.router->has_value()) {
            callback(prefix + ".mlp.gate.lora_A.weight", block.router->A);
            callback(prefix + ".mlp.gate.lora_B.weight", block.router->B);
        }
    }
}

} // namespace modules
