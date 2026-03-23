// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/infer/continuous_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/infer/decode_state.h"
#include "runtime/dsl/dsl_run_state.h"
#include "runtime/dsl/dsl_param_store.h"
#include "runtime/dsl/graph_executor.h"
#include "runtime/core/model_config.h"
#include "runtime/training/runtime_options.h"
#include "kernels/attention_decode.h"
#include "kernels/attention_flat_paged.h"
#include "kernels/sampling.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "utilities/utils.h"

namespace infer {

namespace {

inline bool sanitize_logits_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* env = std::getenv("SUROGATE_SANITIZE_LOGITS");
    cached = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    return cached != 0;
}

inline bool supports_fp8_kv_cache(const cudaDeviceProp& prop) {
    const char* env = std::getenv("SUROGATE_ENABLE_FP8_KV_CACHE");
    if (!env || !*env || *env == '0') return false;
    return (prop.major * 10 + prop.minor) >= 89;
}

std::vector<int> build_graph_buckets(int max_slots) {
    std::vector<int> buckets;
    for (int b = 1; b <= max_slots; b *= 2) {
        buckets.push_back(b);
    }
    if (buckets.empty() || buckets.back() < max_slots) {
        buckets.push_back(max_slots);
    }
    return buckets;
}

inline int static_prefill_t_cap(const dsl::DslRunState& run_state) {
    if (run_state.Inputs.Rank >= 2 && run_state.Inputs.Sizes.size() >= 2) {
        const int static_t_cap = static_cast<int>(run_state.Inputs.Sizes[1]);
        if (static_t_cap > 0) {
            return static_t_cap;
        }
    }
    return 0;
}

inline int resolve_prefill_chunk_size(
        int requested_chunk_size,
        int max_prefill_len,
        const dsl::DslRunState& run_state) {
    if (max_prefill_len <= 0) {
        return 0;
    }
    int chunk = (requested_chunk_size > 0)
        ? std::min(requested_chunk_size, max_prefill_len)
        : max_prefill_len;
    const int static_t_cap = static_prefill_t_cap(run_state);
    if (static_t_cap > 0) {
        chunk = std::min(chunk, static_t_cap);
    }
    return std::max(1, chunk);
}

inline void ensure_device_buffer_capacity(
        void*& ptr,
        std::size_t required_bytes,
        std::size_t& capacity_bytes) {
    if (ptr && capacity_bytes >= required_bytes) {
        return;
    }
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
        capacity_bytes = 0;
    }
    if (required_bytes == 0) {
        return;
    }
    CUDA_CHECK(cudaMalloc(&ptr, required_bytes));
    capacity_bytes = required_bytes;
}

struct EngineStateStorage {
    std::vector<std::unordered_map<int, void*>> slot_recurrent_states;
    std::vector<std::unordered_map<int, std::size_t>> slot_recurrent_state_bytes;
    std::vector<std::unordered_map<int, void*>> slot_conv_states;
    std::vector<std::unordered_map<int, std::size_t>> slot_conv_state_bytes;

    std::unordered_map<int, void*> compact_recurrent_states;
    std::unordered_map<int, std::size_t> compact_recurrent_state_bytes;
    std::unordered_map<int, std::size_t> compact_recurrent_state_capacity_bytes;
    std::unordered_map<int, void*> compact_conv_states;
    std::unordered_map<int, std::size_t> compact_conv_state_bytes;
    std::unordered_map<int, std::size_t> compact_conv_state_capacity_bytes;
};

std::mutex g_engine_state_mu;
std::unordered_map<const ContinuousGenerationEngine*, std::unique_ptr<EngineStateStorage>>
    g_engine_state;

EngineStateStorage& ensure_engine_state_storage(
        const ContinuousGenerationEngine* engine,
        int max_slots) {
    std::lock_guard<std::mutex> lock(g_engine_state_mu);
    auto& ptr = g_engine_state[engine];
    if (!ptr) {
        ptr = std::make_unique<EngineStateStorage>();
    }
    const std::size_t slots = static_cast<std::size_t>(std::max(0, max_slots));
    ptr->slot_recurrent_states.resize(slots);
    ptr->slot_recurrent_state_bytes.resize(slots);
    ptr->slot_conv_states.resize(slots);
    ptr->slot_conv_state_bytes.resize(slots);
    return *ptr;
}

EngineStateStorage* get_engine_state_storage(const ContinuousGenerationEngine* engine) {
    std::lock_guard<std::mutex> lock(g_engine_state_mu);
    auto it = g_engine_state.find(engine);
    if (it == g_engine_state.end()) {
        return nullptr;
    }
    return it->second.get();
}

void erase_engine_state_storage(const ContinuousGenerationEngine* engine) {
    std::lock_guard<std::mutex> lock(g_engine_state_mu);
    g_engine_state.erase(engine);
}

void free_slot_states_impl(EngineStateStorage& state, int slot_id) {
    auto free_state_map = [&](std::vector<std::unordered_map<int, void*>>& slot_states,
                              std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes) {
        const std::size_t idx = static_cast<std::size_t>(slot_id);
        if (idx >= slot_states.size() || idx >= slot_state_bytes.size()) {
            return;
        }
        auto& slot_map = slot_states[idx];
        for (auto& [_, ptr] : slot_map) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
                ptr = nullptr;
            }
        }
        slot_map.clear();
        slot_state_bytes[idx].clear();
    };
    if (slot_id < 0) {
        return;
    }
    free_state_map(state.slot_recurrent_states, state.slot_recurrent_state_bytes);
    free_state_map(state.slot_conv_states, state.slot_conv_state_bytes);
}

void clear_state_storage_impl(EngineStateStorage& state) {
    for (std::size_t slot_id = 0; slot_id < state.slot_recurrent_states.size(); ++slot_id) {
        free_slot_states_impl(state, static_cast<int>(slot_id));
    }
    auto free_compact_map = [](std::unordered_map<int, void*>& compact_states,
                               std::unordered_map<int, std::size_t>& compact_state_bytes,
                               std::unordered_map<int, std::size_t>& compact_state_capacity_bytes) {
        for (auto& [_, ptr] : compact_states) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
                ptr = nullptr;
            }
        }
        compact_states.clear();
        compact_state_bytes.clear();
        compact_state_capacity_bytes.clear();
    };
    free_compact_map(
        state.compact_recurrent_states,
        state.compact_recurrent_state_bytes,
        state.compact_recurrent_state_capacity_bytes);
    free_compact_map(
        state.compact_conv_states,
        state.compact_conv_state_bytes,
        state.compact_conv_state_capacity_bytes);
}

void flush_compact_states_to_slots_impl(
        EngineStateStorage& state,
        dsl::DslRunState* run_state,
        int compact_B,
        const std::vector<int>& active_slot_ids,
        int max_slots,
        const std::vector<SequenceSlot>& slots) {
    if (compact_B <= 0 || active_slot_ids.empty() || !run_state) {
        return;
    }
    cudaStream_t stream = run_state->MainStream;
    const int B = std::min(compact_B, static_cast<int>(active_slot_ids.size()));

    auto flush_state_map = [&](const std::unordered_map<int, void*>& compact_states,
                               const std::unordered_map<int, std::size_t>& compact_state_bytes,
                               std::vector<std::unordered_map<int, void*>>& slot_states,
                               std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes) {
        for (const auto& [layer_idx, src_base_void] : compact_states) {
            if (!src_base_void) continue;
            auto it_b = compact_state_bytes.find(layer_idx);
            if (it_b == compact_state_bytes.end() || it_b->second == 0) continue;
            const std::size_t per_seq_bytes = it_b->second;
            const char* src_base = static_cast<const char*>(src_base_void);
            for (int i = 0; i < B; ++i) {
                const int slot_id = active_slot_ids[static_cast<std::size_t>(i)];
                if (slot_id < 0 || slot_id >= max_slots) continue;
                if (static_cast<std::size_t>(slot_id) >= slots.size()) continue;
                if (!slots[static_cast<std::size_t>(slot_id)].active) continue;
                auto& slot_map = slot_states[static_cast<std::size_t>(slot_id)];
                auto& bytes_map = slot_state_bytes[static_cast<std::size_t>(slot_id)];
                void*& dst_ptr = slot_map[layer_idx];
                auto it_slot_b = bytes_map.find(layer_idx);
                if (it_slot_b != bytes_map.end() && it_slot_b->second != per_seq_bytes) {
                    if (dst_ptr) {
                        CUDA_CHECK(cudaFree(dst_ptr));
                        dst_ptr = nullptr;
                    }
                    bytes_map.erase(it_slot_b);
                }
                if (!dst_ptr) {
                    CUDA_CHECK(cudaMalloc(&dst_ptr, per_seq_bytes));
                }
                bytes_map[layer_idx] = per_seq_bytes;
                const char* src_ptr = src_base + static_cast<std::size_t>(i) * per_seq_bytes;
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, per_seq_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
            }
        }
    };

    flush_state_map(
        state.compact_recurrent_states,
        state.compact_recurrent_state_bytes,
        state.slot_recurrent_states,
        state.slot_recurrent_state_bytes);
    flush_state_map(
        state.compact_conv_states,
        state.compact_conv_state_bytes,
        state.slot_conv_states,
        state.slot_conv_state_bytes);
}

void build_compact_states_from_slots_impl(
        EngineStateStorage& state,
        dsl::DslRunState* run_state,
        int compact_B,
        int compact_B_padded,
        const std::vector<int>& active_slot_ids,
        int max_slots) {
    if (compact_B_padded <= 0 || !run_state) {
        return;
    }
    cudaStream_t stream = run_state->MainStream;
    const int B = compact_B;
    const int Bp = compact_B_padded;

    auto build_state_map = [&](std::vector<std::unordered_map<int, void*>>& slot_states,
                               std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes,
                               std::unordered_map<int, void*>& compact_states,
                               std::unordered_map<int, std::size_t>& compact_state_bytes,
                               std::unordered_map<int, std::size_t>& compact_state_capacity_bytes) {
        std::unordered_map<int, std::size_t> layer_per_seq_bytes;
        for (int i = 0; i < B; ++i) {
            const int slot_id = active_slot_ids[static_cast<std::size_t>(i)];
            if (slot_id < 0 || slot_id >= max_slots) continue;
            const auto& slot_map = slot_states[static_cast<std::size_t>(slot_id)];
            const auto& bytes_map = slot_state_bytes[static_cast<std::size_t>(slot_id)];
            for (const auto& [layer_idx, ptr] : slot_map) {
                if (!ptr) continue;
                auto it_b = bytes_map.find(layer_idx);
                if (it_b == bytes_map.end() || it_b->second == 0) continue;
                const std::size_t per_seq_bytes = it_b->second;
                auto it_existing = layer_per_seq_bytes.find(layer_idx);
                if (it_existing == layer_per_seq_bytes.end()) {
                    layer_per_seq_bytes[layer_idx] = per_seq_bytes;
                } else if (it_existing->second != per_seq_bytes) {
                    throw std::runtime_error(
                        "ContinuousGenerationEngine: per-seq state byte-size mismatch for layer "
                        + std::to_string(layer_idx));
                }
            }
        }

        for (auto it = compact_states.begin(); it != compact_states.end();) {
            const int layer_idx = it->first;
            if (layer_per_seq_bytes.find(layer_idx) != layer_per_seq_bytes.end()) {
                ++it;
                continue;
            }
            if (it->second) {
                CUDA_CHECK(cudaFree(it->second));
            }
            compact_state_bytes.erase(layer_idx);
            compact_state_capacity_bytes.erase(layer_idx);
            it = compact_states.erase(it);
        }

        for (const auto& [layer_idx, per_seq_bytes] : layer_per_seq_bytes) {
            void*& dst_base = compact_states[layer_idx];
            std::size_t& cap_bytes = compact_state_capacity_bytes[layer_idx];
            const std::size_t required_bytes =
                static_cast<std::size_t>(Bp) * per_seq_bytes;
            ensure_device_buffer_capacity(dst_base, required_bytes, cap_bytes);
            compact_state_bytes[layer_idx] = per_seq_bytes;
            CUDA_CHECK(cudaMemsetAsync(dst_base, 0, required_bytes, stream));
        }

        for (int i = 0; i < B; ++i) {
            const int slot_id = active_slot_ids[static_cast<std::size_t>(i)];
            if (slot_id < 0 || slot_id >= max_slots) continue;
            auto& slot_map = slot_states[static_cast<std::size_t>(slot_id)];
            auto& bytes_map = slot_state_bytes[static_cast<std::size_t>(slot_id)];
            for (const auto& [layer_idx, src_ptr] : slot_map) {
                if (!src_ptr) continue;
                auto it_b = bytes_map.find(layer_idx);
                auto it_comp_b = compact_state_bytes.find(layer_idx);
                if (it_b == bytes_map.end() || it_comp_b == compact_state_bytes.end()) continue;
                const std::size_t per_seq_bytes = it_b->second;
                if (it_comp_b->second != per_seq_bytes) continue;
                auto it_dst = compact_states.find(layer_idx);
                if (it_dst == compact_states.end() || !it_dst->second) continue;
                char* dst_ptr = static_cast<char*>(it_dst->second)
                              + static_cast<std::size_t>(i) * per_seq_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_ptr, src_ptr, per_seq_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }

        // NOTE: Do NOT free slot-level state buffers here.
        // Keep them allocated so the next flush (compact→slot) can reuse them
        // without cudaMalloc.  Slot buffers are freed in free_slot_states_impl()
        // when the slot is actually released via release_slot().
    };

    build_state_map(
        state.slot_recurrent_states,
        state.slot_recurrent_state_bytes,
        state.compact_recurrent_states,
        state.compact_recurrent_state_bytes,
        state.compact_recurrent_state_capacity_bytes);
    build_state_map(
        state.slot_conv_states,
        state.slot_conv_state_bytes,
        state.compact_conv_states,
        state.compact_conv_state_bytes,
        state.compact_conv_state_capacity_bytes);
}

}  // namespace

ContinuousGenerationEngine::~ContinuousGenerationEngine() {
    if (full_step_graph_exec_) {
        cudaGraphExecDestroy(full_step_graph_exec_);
        full_step_graph_exec_ = nullptr;
    }
    erase_engine_state_storage(this);
}

void ContinuousGenerationEngine::init(
        int max_slots, int max_seq_len, int total_pages,
        bool use_cuda_graphs,
        dsl::DslRunState& run_state,
        dsl::DslParamStore& weights,
        const modules::ModelConfig& config,
        const RuntimeOptions& options,
        DeviceMemoryStack& arena) {
    if (initialized_) {
        throw std::runtime_error("ContinuousGenerationEngine: already initialized");
    }

    max_slots_ = max_slots;
    max_seq_len_ = max_seq_len;
    max_pages_per_seq_ = (max_seq_len + kPageBlockSize - 1) / kPageBlockSize;
    vocab_size_ = config.VocabSize;
    fallback_token_id_ = 0;
    use_cuda_graphs_ = use_cuda_graphs;
    cuda_graph_primed_ = false;
    run_state_ = &run_state;
    weights_ = &weights;
    config_ = &config;
    options_ = &options;
    arena_ = &arena;

    const int Hkv = config.NumKeyValHeads;
    const int Hs = config.head_size();
    const int num_layers = config.NumLayers;
    const KVDType kv_dtype =
        supports_fp8_kv_cache(run_state.DeviceProp) ? KVDType::FP8_E4M3 : KVDType::BF16;

    page_pool_.init(total_pages, kPageBlockSize, num_layers,
                    Hkv, Hs, kv_dtype, arena);

    slots_.resize(static_cast<std::size_t>(max_slots));
    (void)ensure_engine_state_storage(this, max_slots);
    free_slot_stack_.resize(max_slots);
    for (int i = 0; i < max_slots; ++i) {
        free_slot_stack_[i] = max_slots - 1 - i;
    }

    graph_buckets_ = build_graph_buckets(max_slots);

    const auto S = static_cast<std::size_t>(max_slots);
    const auto V = static_cast<std::size_t>(vocab_size_);
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);

    auto alloc = [&](std::size_t bytes, const char* name) -> std::byte* {
        return arena.allocate(bytes, name);
    };

    last_tokens_gpu_ = reinterpret_cast<int32_t*>(alloc(S * sizeof(int32_t), "ce_last_tokens"));
    seq_lens_gpu_ = reinterpret_cast<int*>(alloc(S * sizeof(int), "ce_seq_lens"));
    finished_gpu_ = reinterpret_cast<int*>(alloc(S * sizeof(int), "ce_finished"));
    position_ids_gpu_ = reinterpret_cast<int32_t*>(alloc(S * sizeof(int32_t), "ce_pos_ids"));
    cu_seqlens_q_gpu_ = reinterpret_cast<int32_t*>(alloc((S + 1) * sizeof(int32_t), "ce_cu_q"));
    seqused_k_gpu_ = reinterpret_cast<int32_t*>(alloc(S * sizeof(int32_t), "ce_seqused_k"));
    block_table_gpu_ = reinterpret_cast<int*>(alloc(S * MPS * sizeof(int), "ce_block_table"));
    logits_gpu_ = reinterpret_cast<float*>(alloc(S * V * sizeof(float), "ce_logits"));
    probs_gpu_ = logits_gpu_;
    sampled_tokens_gpu_ = reinterpret_cast<int32_t*>(alloc(S * sizeof(int32_t), "ce_sampled"));
    temperature_gpu_ = reinterpret_cast<float*>(alloc(S * sizeof(float), "ce_temperature"));
    completion_lens_gpu_ = reinterpret_cast<int32_t*>(alloc(S * sizeof(int32_t), "ce_completion_lens"));
    softmax_ws_bytes_ = S * 256 * 8;
    softmax_ws_ = alloc(softmax_ws_bytes_, "ce_softmax_ws");
    prefill_block_table_gpu_ = reinterpret_cast<int*>(alloc(MPS * sizeof(int), "ce_prefill_bt"));

    // Zero GPU buffers.
    cudaStream_t stream = run_state.MainStream;
    CUDA_CHECK(cudaMemsetAsync(block_table_gpu_, 0, S * MPS * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(finished_gpu_, 0, S * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(seq_lens_gpu_, 0, S * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(completion_lens_gpu_, 0, S * sizeof(int32_t), stream));

    sampled_tokens_host_.resize(S);
    finished_host_.resize(S);

    run_state.ensure_rope_freq_capacity(config, max_seq_len);

    // Pre-allocate flat-step buffers (avoid per-step cudaMallocAsync).
    // max_batched_tokens is the token budget per flat_step.
    // Use the runtime's B*T capacity as the budget.
    max_batched_tokens_ = static_cast<int>(
        run_state.Inputs.Sizes[0] * (run_state.Inputs.Rank >= 2 ? run_state.Inputs.Sizes[1] : 1));
    if (max_batched_tokens_ <= 0) max_batched_tokens_ = 2048;
    const auto MBT = static_cast<std::size_t>(max_batched_tokens_);

    flat_token_to_req_gpu_ = reinterpret_cast<int32_t*>(
        alloc(MBT * sizeof(int32_t), "ce_flat_token_to_req"));
    flat_kv_write_pos_gpu_ = reinterpret_cast<int32_t*>(
        alloc(MBT * sizeof(int32_t), "ce_flat_kv_write_pos"));
    flat_q_indptr_gpu_ = reinterpret_cast<int32_t*>(
        alloc((S + 1) * sizeof(int32_t), "ce_flat_q_indptr"));
    flat_last_token_indices_gpu_ = reinterpret_cast<int32_t*>(
        alloc(S * sizeof(int32_t), "ce_flat_last_token_indices"));
    flat_seq_lens_k_gpu_ = reinterpret_cast<int32_t*>(
        alloc(S * sizeof(int32_t), "ce_flat_seq_lens_k"));
    flat_page_indptr_gpu_ = reinterpret_cast<int32_t*>(
        alloc((S + 1) * sizeof(int32_t), "ce_flat_page_indptr"));
    flat_page_indices_gpu_ = reinterpret_cast<int32_t*>(
        alloc(S * MPS * sizeof(int32_t), "ce_flat_page_indices"));
    flat_last_page_len_gpu_ = reinterpret_cast<int32_t*>(
        alloc(S * sizeof(int32_t), "ce_flat_last_page_len"));
    // Plan workspaces: allocated from CUDA pool (not arena) since they may
    // need to persist across graph capture.
    CUDA_CHECK(cudaMalloc(&flat_plan_int_ws_gpu_, kPlanIntWsSize));
    CUDA_CHECK(cudaMalloc(&flat_plan_float_ws_gpu_, kPlanFloatWsSize));

    batch_dirty_ = true;
    initialized_ = true;
}

int ContinuousGenerationEngine::next_bucket(int B) const {
    for (int b : graph_buckets_) {
        if (b >= B) return b;
    }
    return B;
}

// ---------------------------------------------------------------------------
// warm_prefill_graphs — pre-compile prefill graphs for common T values
// ---------------------------------------------------------------------------
void ContinuousGenerationEngine::warm_prefill_graphs(
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm) {
    if (!initialized_) return;

    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);

    // Minimal DecodeState for compilation — prefill_mode skips KV reads,
    // and logits_out_gpu=nullptr skips the LM head output copy.
    // We still need valid page pool pointers for the KV write path.
    DecodeState warmup_ds{};
    warmup_ds.paged = true;
    warmup_ds.fp8 = fp8_kv;
    warmup_ds.k_pages = page_pool_.k_pages();
    warmup_ds.v_pages = page_pool_.v_pages();
    warmup_ds.k_scales = page_pool_.k_scales();
    warmup_ds.v_scales = page_pool_.v_scales();
    warmup_ds.k_scales_paged_fp8 = page_pool_.k_scales();
    warmup_ds.v_scales_paged_fp8 = page_pool_.v_scales();
    warmup_ds.per_pool_bytes = page_pool_.full_pool_bytes();
    warmup_ds.block_table_gpu = prefill_block_table_gpu_;
    warmup_ds.block_table_stride = max_pages_per_seq_;
    warmup_ds.page_block_size = kPageBlockSize;
    warmup_ds.total_pages = page_pool_.total_pages();
    warmup_ds.num_kv_heads = config_->NumKeyValHeads;
    warmup_ds.head_dim = config_->head_size();
    warmup_ds.logits_out_gpu = nullptr;
    warmup_ds.vocab_size = vocab_size_;
    warmup_ds.prefill_mode = true;
    warmup_ds.prefill_pos_offset = 0;

    // Zero the prefill block table so warmup writes are harmless.
    cudaStream_t stream = run_state_->MainStream;
    CUDA_CHECK(cudaMemsetAsync(prefill_block_table_gpu_, 0,
        static_cast<std::size_t>(max_pages_per_seq_) * sizeof(int), stream));

    const int warmup_t_cap = std::max(32, resolve_prefill_chunk_size(
        /*requested_chunk_size=*/2048, max_seq_len_, *run_state_));
    static constexpr int kWarmupT[] = {32, 64, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048};
    for (int t : kWarmupT) {
        if (t > max_seq_len_ || t > warmup_t_cap) break;
        std::vector<int32_t> dummy_tokens(static_cast<std::size_t>(t), 0);
        std::vector<int32_t> dummy_pos(static_cast<std::size_t>(t));
        for (int i = 0; i < t; ++i) dummy_pos[i] = i;

        graph_executor.execute_prefill(
            1L, static_cast<long>(t),
            dummy_tokens.data(), dummy_pos.data(),
            warmup_ds, comm);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ---------------------------------------------------------------------------
// rebuild_compact_batch — only called when batch_dirty_ is set
// ---------------------------------------------------------------------------
void ContinuousGenerationEngine::rebuild_compact_batch() {
    cudaStream_t stream = run_state_->MainStream;
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);
    auto& state = ensure_engine_state_storage(this, max_slots_);

    // Persist decode-updated compact recurrent/conv state into slot-local storage
    // before rebuilding row order.
    flush_compact_states_to_slots_impl(
        state, run_state_, compact_B_, active_slot_ids_, max_slots_, slots_);

    active_slot_ids_.clear();
    for (int i = 0; i < max_slots_; ++i) {
        auto& s = slots_[static_cast<std::size_t>(i)];
        if (s.active && !s.finished) {
            s.compact_idx = static_cast<int>(active_slot_ids_.size());
            active_slot_ids_.push_back(i);
        } else {
            s.compact_idx = -1;
        }
    }

    compact_B_ = static_cast<int>(active_slot_ids_.size());
    compact_B_padded_ = (compact_B_ > 0 && use_cuda_graphs_)
        ? next_bucket(compact_B_) : compact_B_;

    if (compact_B_ == 0) {
        batch_dirty_ = false;
        return;
    }

    const auto Bp = static_cast<std::size_t>(compact_B_padded_);

    // Build compact host arrays and upload.
    // These are the one-time-per-batch-change uploads.
    std::vector<int32_t> h_last_tokens(Bp, 0);
    std::vector<int> h_seq_lens(Bp, 0);
    std::vector<int> h_finished(Bp, 0);
    std::vector<float> h_temp(Bp, 1.0f);
    std::vector<int> h_block_table(Bp * MPS, 0);

    int max_seqlen_k = 0;
    for (int i = 0; i < compact_B_; ++i) {
        const auto& s = slots_[static_cast<std::size_t>(active_slot_ids_[i])];
        h_last_tokens[i] = s.last_token;
        h_seq_lens[i] = s.seq_len;
        h_finished[i] = 0;
        h_temp[i] = s.temperature;
        max_seqlen_k = std::max(max_seqlen_k, s.seq_len + 1);
        const auto row = static_cast<std::size_t>(i) * MPS;
        for (std::size_t p = 0; p < s.page_ids.size(); ++p) {
            h_block_table[row + p] = s.page_ids[p];
        }
    }
    // Pad [compact_B_, compact_B_padded_) with finished=1.
    for (int i = compact_B_; i < compact_B_padded_; ++i) {
        h_finished[i] = 1;
    }

    CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu_, h_last_tokens.data(),
        Bp * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu_, h_seq_lens.data(),
        Bp * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(finished_gpu_, h_finished.data(),
        Bp * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(temperature_gpu_, h_temp.data(),
        Bp * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, h_block_table.data(),
        Bp * MPS * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(completion_lens_gpu_, 0,
        Bp * sizeof(int32_t), stream));

    // Cache sampling config from first active slot.
    const auto& first = slots_[static_cast<std::size_t>(active_slot_ids_[0])];
    batch_temperature_ = first.temperature;
    batch_top_k_ = first.top_k;
    batch_top_p_ = first.top_p;
    batch_min_p_ = first.min_p;
    greedy_ = (batch_temperature_ <= 0.0f);

    // Build compact recurrent/conv state rows in the same active-slot order.
    build_compact_states_from_slots_impl(
        state, run_state_, compact_B_, compact_B_padded_, active_slot_ids_, max_slots_);

    // Build persistent DecodeState.
    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    decode_state_ = DecodeState{};
    decode_state_.recurrent_states = &state.compact_recurrent_states;
    decode_state_.recurrent_state_bytes = &state.compact_recurrent_state_bytes;
    decode_state_.conv_states = &state.compact_conv_states;
    decode_state_.conv_state_bytes = &state.compact_conv_state_bytes;
    decode_state_.paged = true;
    decode_state_.fp8 = fp8_kv;
    decode_state_.k_pages = page_pool_.k_pages();
    decode_state_.v_pages = page_pool_.v_pages();
    decode_state_.k_scales = page_pool_.k_scales();
    decode_state_.v_scales = page_pool_.v_scales();
    decode_state_.k_scales_paged_fp8 = page_pool_.k_scales();
    decode_state_.v_scales_paged_fp8 = page_pool_.v_scales();
    decode_state_.per_pool_bytes = page_pool_.full_pool_bytes();
    decode_state_.block_table_gpu = block_table_gpu_;
    decode_state_.block_table_stride = max_pages_per_seq_;
    decode_state_.page_block_size = kPageBlockSize;
    decode_state_.total_pages = page_pool_.total_pages();
    decode_state_.seq_lens_gpu = seq_lens_gpu_;
    decode_state_.cu_seqlens_q_gpu = cu_seqlens_q_gpu_;
    decode_state_.cu_seqlens_k_gpu = seqused_k_gpu_;
    decode_state_.num_kv_heads = config_->NumKeyValHeads;
    decode_state_.head_dim = config_->head_size();
    decode_state_.finished_gpu = finished_gpu_;
    decode_state_.logits_out_gpu = logits_gpu_;
    decode_state_.vocab_size = vocab_size_;
    decode_state_.max_seqlen_k = max_seq_len_;  // conservative upper bound

    // Invalidate full-step graph (batch composition changed).
    if (full_step_graph_exec_) {
        CUDA_CHECK(cudaGraphExecDestroy(full_step_graph_exec_));
        full_step_graph_exec_ = nullptr;
    }
    full_step_primed_ = false;

    batch_dirty_ = false;
}

// ---------------------------------------------------------------------------
// run_one_decode_step — GPU-only, no CPU sync
// ---------------------------------------------------------------------------
void ContinuousGenerationEngine::run_one_decode_step(
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {
    cudaStream_t stream = run_state_->MainStream;
    const int Bp = compact_B_padded_;
    const int V = vocab_size_;

    // 1. Prepare per-step metadata (GPU kernels, no H2D).
    fill_decode_cu_seqlens(cu_seqlens_q_gpu_, seqused_k_gpu_,
                           seq_lens_gpu_, Bp, stream);
    CUDA_CHECK(cudaMemcpyAsync(position_ids_gpu_, seq_lens_gpu_,
        static_cast<std::size_t>(Bp) * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, stream));
    mask_finished_tokens(last_tokens_gpu_, finished_gpu_, Bp, stream);

    // 2. Model forward.
    // Use the inner decode CUDA graph when full-step graph is NOT active.
    const bool use_inner_graph = use_cuda_graphs_ && !full_step_primed_;
    graph_executor.execute_decode_step(
        static_cast<long>(Bp), last_tokens_gpu_, position_ids_gpu_,
        decode_state_, comm, hook, use_inner_graph);

    // 3. Sampling.
    if (sanitize_logits_enabled()) {
        sampling_sanitize_logits(logits_gpu_, Bp, V, stream);
    }
    CUDA_CHECK(cudaGetLastError());

    if (greedy_) {
        sampling_argmax(logits_gpu_, sampled_tokens_gpu_, Bp, V, stream);
    } else {
        const float* temp_ptr = (batch_temperature_ == 1.0f)
            ? nullptr : temperature_gpu_;
        sampling_softmax(logits_gpu_, probs_gpu_, temp_ptr,
                         Bp, V, softmax_ws_, softmax_ws_bytes_, stream);

        if (batch_top_k_ > 0 && batch_top_p_ < 1.0f) {
            sampling_top_k_top_p(probs_gpu_, sampled_tokens_gpu_,
                batch_top_k_, batch_top_p_, Bp, V, false, 42, 0, stream);
        } else if (batch_top_k_ > 0) {
            sampling_top_k(probs_gpu_, sampled_tokens_gpu_, batch_top_k_,
                Bp, V, false, 42, 0, stream);
        } else if (batch_top_p_ < 1.0f) {
            sampling_top_p(probs_gpu_, sampled_tokens_gpu_, batch_top_p_,
                Bp, V, false, 42, 0, stream);
        } else if (batch_min_p_ > 0.0f) {
            sampling_min_p(probs_gpu_, sampled_tokens_gpu_, batch_min_p_,
                Bp, V, false, 42, 0, stream);
        } else {
            sampling_from_probs(probs_gpu_, sampled_tokens_gpu_, Bp, V,
                false, 42, 0, stream);
        }
    }

    // 4. Post-sampling: mask finished, sanitize, update state.
    mask_finished_tokens(sampled_tokens_gpu_, finished_gpu_, Bp, stream);
    sampling_sanitize_token_ids(sampled_tokens_gpu_, Bp, V,
        fallback_token_id_, nullptr, stream);

    // Feed sampled tokens back as input for next step (GPU D2D).
    CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu_, sampled_tokens_gpu_,
        static_cast<std::size_t>(Bp) * sizeof(int32_t),
        cudaMemcpyDeviceToDevice, stream));

    // Update seq_lens (++), check EOS on GPU.
    // Use a dummy EOS=-1 here; we'll check per-slot EOS on host after sync.
    update_generation_state(sampled_tokens_gpu_, finished_gpu_, seq_lens_gpu_,
        completion_lens_gpu_, /*eos_token_id=*/-1, Bp, stream);
}

// ---------------------------------------------------------------------------
// add_sequence
// ---------------------------------------------------------------------------
int ContinuousGenerationEngine::add_sequence(
        const std::vector<int32_t>& prompt_ids,
        int max_gen_len, float temperature, int32_t eos_token_id,
        int top_k, float top_p, float min_p,
        int prefill_chunk_size,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {
    if (!initialized_) {
        throw std::runtime_error("ContinuousGenerationEngine: not initialized");
    }
    auto& state = ensure_engine_state_storage(this, max_slots_);
    if (free_slot_stack_.empty()) return -1;

    const int plen = static_cast<int>(prompt_ids.size());
    if (plen <= 0) {
        throw std::invalid_argument("ContinuousGenerationEngine: empty prompt");
    }
    const int total_len = plen + max_gen_len;
    if (total_len > max_seq_len_) {
        throw std::invalid_argument(
            "ContinuousGenerationEngine: total length exceeds max_seq_len");
    }
    const int pages_needed = (total_len + kPageBlockSize - 1) / kPageBlockSize;
    if (page_pool_.num_free() < pages_needed) return -1;

    const int slot_id = free_slot_stack_.back();
    free_slot_stack_.pop_back();

    SequenceSlot& slot = slots_[static_cast<std::size_t>(slot_id)];
    slot = SequenceSlot{};
    slot.active = true;
    slot.max_gen_len = max_gen_len;
    slot.eos_token_id = eos_token_id;
    slot.temperature = temperature;
    slot.top_k = top_k;
    slot.top_p = top_p;
    slot.min_p = min_p;
    slot.rng_offset = 0;
    slot.page_ids = page_pool_.allocate_pages(pages_needed);

    // --- Prefill ---
    std::vector<int> bt_host(static_cast<std::size_t>(max_pages_per_seq_), -1);
    for (int p = 0; p < static_cast<int>(slot.page_ids.size()); ++p) {
        bt_host[p] = slot.page_ids[p];
    }
    cudaStream_t stream = run_state_->MainStream;
    CUDA_CHECK(cudaMemcpyAsync(prefill_block_table_gpu_, bt_host.data(),
        bt_host.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    std::unordered_map<int, void*> prompt_recurrent_states;
    std::unordered_map<int, std::size_t> prompt_recurrent_state_bytes;
    std::unordered_map<int, void*> prompt_conv_states;
    std::unordered_map<int, std::size_t> prompt_conv_state_bytes;
    DecodeState prefill_ds{};
    prefill_ds.paged = true;
    prefill_ds.fp8 = fp8_kv;
    prefill_ds.k_pages = page_pool_.k_pages();
    prefill_ds.v_pages = page_pool_.v_pages();
    prefill_ds.k_scales = page_pool_.k_scales();
    prefill_ds.v_scales = page_pool_.v_scales();
    prefill_ds.k_scales_paged_fp8 = page_pool_.k_scales();
    prefill_ds.v_scales_paged_fp8 = page_pool_.v_scales();
    prefill_ds.per_pool_bytes = page_pool_.full_pool_bytes();
    prefill_ds.block_table_gpu = prefill_block_table_gpu_;
    prefill_ds.block_table_stride = max_pages_per_seq_;
    prefill_ds.page_block_size = kPageBlockSize;
    prefill_ds.total_pages = page_pool_.total_pages();
    prefill_ds.num_kv_heads = config_->NumKeyValHeads;
    prefill_ds.head_dim = config_->head_size();
    prefill_ds.logits_out_gpu = nullptr;
    prefill_ds.vocab_size = vocab_size_;
    prefill_ds.recurrent_states = &prompt_recurrent_states;
    prefill_ds.recurrent_state_bytes = &prompt_recurrent_state_bytes;
    prefill_ds.conv_states = &prompt_conv_states;
    prefill_ds.conv_state_bytes = &prompt_conv_state_bytes;
    prefill_ds.prefill_mode = true;
    prefill_ds.prefill_pos_offset = 0;

    const int prefill_len = plen - 1;
    if (prefill_len > 0) {
        const int chunk = resolve_prefill_chunk_size(
            prefill_chunk_size, prefill_len, *run_state_);
        std::vector<int32_t> tok_buf(static_cast<std::size_t>(chunk));
        std::vector<int32_t> pos_buf(static_cast<std::size_t>(chunk));

        for (int cs = 0; cs < prefill_len; cs += chunk) {
            const int cl = std::min(chunk, prefill_len - cs);
            for (int t = 0; t < cl; ++t) {
                tok_buf[t] = prompt_ids[static_cast<std::size_t>(cs + t)];
                pos_buf[t] = cs + t;
            }
            prefill_ds.prefill_pos_offset = cs;
            graph_executor.execute_prefill(1L, static_cast<long>(cl),
                tok_buf.data(), pos_buf.data(), prefill_ds, comm, hook);
        }
    }

    // Persist prefill recurrent/conv state for this slot.
    auto persist_prompt_states_for_slot =
        [&](const std::unordered_map<int, void*>& prompt_states,
            const std::unordered_map<int, std::size_t>& prompt_state_bytes,
            std::vector<std::unordered_map<int, void*>>& slot_states,
            std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes) {
            for (const auto& [layer_idx, src_ptr] : prompt_states) {
                if (!src_ptr) continue;
                auto it_b = prompt_state_bytes.find(layer_idx);
                if (it_b == prompt_state_bytes.end() || it_b->second == 0) {
                    throw std::runtime_error(
                        "ContinuousGenerationEngine::add_sequence: missing per-seq state bytes for layer "
                        + std::to_string(layer_idx));
                }
                const std::size_t per_seq_bytes = it_b->second;
                auto& slot_map = slot_states[static_cast<std::size_t>(slot_id)];
                auto& bytes_map = slot_state_bytes[static_cast<std::size_t>(slot_id)];
                void*& dst_ptr = slot_map[layer_idx];
                auto it_slot_b = bytes_map.find(layer_idx);
                if (it_slot_b != bytes_map.end() && it_slot_b->second != per_seq_bytes) {
                    if (dst_ptr) {
                        CUDA_CHECK(cudaFree(dst_ptr));
                        dst_ptr = nullptr;
                    }
                    bytes_map.erase(it_slot_b);
                }
                if (!dst_ptr) {
                    CUDA_CHECK(cudaMalloc(&dst_ptr, per_seq_bytes));
                }
                bytes_map[layer_idx] = per_seq_bytes;
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_ptr, src_ptr, per_seq_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        };
    persist_prompt_states_for_slot(
        prompt_recurrent_states,
        prompt_recurrent_state_bytes,
        state.slot_recurrent_states,
        state.slot_recurrent_state_bytes);
    persist_prompt_states_for_slot(
        prompt_conv_states,
        prompt_conv_state_bytes,
        state.slot_conv_states,
        state.slot_conv_state_bytes);

    for (auto& [_, ptr] : prompt_recurrent_states) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    for (auto& [_, ptr] : prompt_conv_states) {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }

    slot.seq_len = prefill_len;
    slot.last_token = prompt_ids.back();
    slot.generated_count = 0;
    slot.finished = false;

    batch_dirty_ = true;  // compact batch must be rebuilt
    // Also invalidate CUDA graph since B may change.
    cuda_graph_primed_ = false;

    return slot_id;
}

// ---------------------------------------------------------------------------
// add_sequences_batch — batch prefill for multiple prompts at once
// ---------------------------------------------------------------------------
std::vector<int> ContinuousGenerationEngine::add_sequences_batch(
        const std::vector<std::vector<int32_t>>& prompts,
        const BatchAddConfig& config,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {
    const int N = static_cast<int>(prompts.size());
    std::vector<int> slot_ids(N, -1);
    if (!initialized_ || N == 0) return slot_ids;
    auto& state = ensure_engine_state_storage(this, max_slots_);

    // --- 1. Allocate slots and pages for all prompts ---
    std::vector<int> valid_indices;  // indices into prompts[] that succeeded
    valid_indices.reserve(N);

    for (int i = 0; i < N; ++i) {
        const int plen = static_cast<int>(prompts[i].size());
        if (plen <= 0) continue;
        const int total_len = plen + config.max_gen_len;
        if (total_len > max_seq_len_) continue;
        const int pages_needed = (total_len + kPageBlockSize - 1) / kPageBlockSize;
        if (free_slot_stack_.empty() || page_pool_.num_free() < pages_needed) continue;

        const int sid = free_slot_stack_.back();
        free_slot_stack_.pop_back();

        auto& slot = slots_[static_cast<std::size_t>(sid)];
        slot = SequenceSlot{};
        slot.active = true;
        slot.max_gen_len = config.max_gen_len;
        slot.eos_token_id = config.eos_token_id;
        slot.temperature = config.temperature;
        slot.top_k = config.top_k;
        slot.top_p = config.top_p;
        slot.min_p = config.min_p;
        slot.rng_offset = 0;
        slot.page_ids = page_pool_.allocate_pages(pages_needed);

        slot_ids[i] = sid;
        valid_indices.push_back(i);
    }

    const int B = static_cast<int>(valid_indices.size());
    if (B == 0) return slot_ids;

    // Pad B to a power-of-2 bucket so the compiled prefill graph
    // can be reused across different batch sizes.
    const int B_padded = next_bucket(B);

    cudaStream_t stream = run_state_->MainStream;
    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);

    // --- 2. Find max prefill length and build padded token/position arrays ---
    int max_prefill_len = 0;
    for (int idx : valid_indices) {
        const int plen = static_cast<int>(prompts[idx].size()) - 1;  // last token is decode input
        max_prefill_len = std::max(max_prefill_len, plen);
    }

    if (max_prefill_len > 0) {
        // Build block table for B_padded sequences (pad rows [B, B_padded) with page 0).
        std::vector<int> bt_host(static_cast<std::size_t>(B_padded) * MPS, 0);
        for (int b = 0; b < B; ++b) {
            const int sid = slot_ids[valid_indices[b]];
            const auto& pages = slots_[static_cast<std::size_t>(sid)].page_ids;
            const auto row = static_cast<std::size_t>(b) * MPS;
            for (std::size_t p = 0; p < pages.size(); ++p) {
                bt_host[row + p] = pages[p];
            }
        }

        // Upload block table (reuse block_table_gpu_ which is sized for max_slots_).
        CUDA_CHECK(cudaMemcpyAsync(
            block_table_gpu_, bt_host.data(),
            static_cast<std::size_t>(B_padded) * MPS * sizeof(int),
            cudaMemcpyHostToDevice, stream));

        DecodeState prefill_ds{};
        std::unordered_map<int, void*> prompt_recurrent_states;
        std::unordered_map<int, std::size_t> prompt_recurrent_state_bytes;
        std::unordered_map<int, void*> prompt_conv_states;
        std::unordered_map<int, std::size_t> prompt_conv_state_bytes;
        prefill_ds.paged = true;
        prefill_ds.fp8 = fp8_kv;
        prefill_ds.k_pages = page_pool_.k_pages();
        prefill_ds.v_pages = page_pool_.v_pages();
        prefill_ds.k_scales = page_pool_.k_scales();
        prefill_ds.v_scales = page_pool_.v_scales();
        prefill_ds.k_scales_paged_fp8 = page_pool_.k_scales();
        prefill_ds.v_scales_paged_fp8 = page_pool_.v_scales();
        prefill_ds.per_pool_bytes = page_pool_.full_pool_bytes();
        prefill_ds.block_table_gpu = block_table_gpu_;
        prefill_ds.block_table_stride = max_pages_per_seq_;
        prefill_ds.page_block_size = kPageBlockSize;
        prefill_ds.total_pages = page_pool_.total_pages();
        prefill_ds.num_kv_heads = config_->NumKeyValHeads;
        prefill_ds.head_dim = config_->head_size();
        prefill_ds.logits_out_gpu = nullptr;
        prefill_ds.vocab_size = vocab_size_;
        prefill_ds.recurrent_states = &prompt_recurrent_states;
        prefill_ds.recurrent_state_bytes = &prompt_recurrent_state_bytes;
        prefill_ds.conv_states = &prompt_conv_states;
        prefill_ds.conv_state_bytes = &prompt_conv_state_bytes;
        prefill_ds.prefill_mode = true;

        // Chunked prefill: process all B_padded sequences together, chunk by chunk.
        // Padding rows [B, B_padded) get token 0 / pos 0 which is harmless.
        const int chunk_size = resolve_prefill_chunk_size(
            config.prefill_chunk_size, max_prefill_len, *run_state_);

        std::vector<int32_t> tok_buf(static_cast<std::size_t>(B_padded) * chunk_size, 0);
        std::vector<int32_t> pos_buf(static_cast<std::size_t>(B_padded) * chunk_size, 0);

        for (int cs = 0; cs < max_prefill_len; cs += chunk_size) {
            const int cl = std::min(chunk_size, max_prefill_len - cs);

            // Fill padded [B_padded, cl] token and position arrays.
            // Real sequences: fill from prompt data.
            // Padding rows [B, B_padded): stay zero (harmless).
            std::fill(tok_buf.begin(), tok_buf.end(), 0);
            std::fill(pos_buf.begin(), pos_buf.end(), 0);
            for (int b = 0; b < B; ++b) {
                const auto& prompt = prompts[valid_indices[b]];
                const int plen = static_cast<int>(prompt.size()) - 1;
                for (int t = 0; t < cl; ++t) {
                    const int abs_t = cs + t;
                    if (abs_t < plen) {
                        tok_buf[static_cast<std::size_t>(b) * cl + t] = prompt[abs_t];
                        pos_buf[static_cast<std::size_t>(b) * cl + t] = abs_t;
                    }
                }
            }
            prefill_ds.prefill_pos_offset = cs;
            graph_executor.execute_prefill(
                static_cast<long>(B_padded), static_cast<long>(cl),
                tok_buf.data(), pos_buf.data(),
                prefill_ds, comm, hook);
        }

        auto scatter_prompt_states_to_slots =
            [&](const std::unordered_map<int, void*>& prompt_states,
                const std::unordered_map<int, std::size_t>& prompt_state_bytes,
                std::vector<std::unordered_map<int, void*>>& slot_states,
                std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes) {
                for (const auto& [layer_idx, src_base_void] : prompt_states) {
                    if (!src_base_void) continue;
                    auto it_b = prompt_state_bytes.find(layer_idx);
                    if (it_b == prompt_state_bytes.end() || it_b->second == 0) {
                        throw std::runtime_error(
                            "ContinuousGenerationEngine::add_sequences_batch: missing per-seq state bytes for layer "
                            + std::to_string(layer_idx));
                    }
                    const std::size_t per_seq_bytes = it_b->second;
                    const char* src_base = static_cast<const char*>(src_base_void);
                    for (int b = 0; b < B; ++b) {
                        const int sid = slot_ids[valid_indices[static_cast<std::size_t>(b)]];
                        if (sid < 0 || sid >= max_slots_) continue;
                        auto& slot_map = slot_states[static_cast<std::size_t>(sid)];
                        auto& bytes_map = slot_state_bytes[static_cast<std::size_t>(sid)];
                        void*& dst_ptr = slot_map[layer_idx];
                        auto it_slot_b = bytes_map.find(layer_idx);
                        if (it_slot_b != bytes_map.end() && it_slot_b->second != per_seq_bytes) {
                            if (dst_ptr) {
                                CUDA_CHECK(cudaFree(dst_ptr));
                                dst_ptr = nullptr;
                            }
                            bytes_map.erase(it_slot_b);
                        }
                        if (!dst_ptr) {
                            CUDA_CHECK(cudaMalloc(&dst_ptr, per_seq_bytes));
                        }
                        bytes_map[layer_idx] = per_seq_bytes;
                        const char* src_ptr = src_base + static_cast<std::size_t>(b) * per_seq_bytes;
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_ptr, src_ptr, per_seq_bytes, cudaMemcpyDeviceToDevice, stream));
                    }
                }
            };
        scatter_prompt_states_to_slots(
            prompt_recurrent_states,
            prompt_recurrent_state_bytes,
            state.slot_recurrent_states,
            state.slot_recurrent_state_bytes);
        scatter_prompt_states_to_slots(
            prompt_conv_states,
            prompt_conv_state_bytes,
            state.slot_conv_states,
            state.slot_conv_state_bytes);

        for (auto& [_, ptr] : prompt_recurrent_states) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
        for (auto& [_, ptr] : prompt_conv_states) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
    }

    // --- 3. Finalize slot state ---
    for (int idx : valid_indices) {
        const int sid = slot_ids[idx];
        auto& slot = slots_[static_cast<std::size_t>(sid)];
        const auto& prompt = prompts[idx];
        slot.seq_len = static_cast<int>(prompt.size()) - 1;
        slot.last_token = prompt.back();
        slot.generated_count = 0;
        slot.finished = false;
    }

    batch_dirty_ = true;
    cuda_graph_primed_ = false;
    return slot_ids;
}

// ---------------------------------------------------------------------------
// step — multi-token inner loop
// ---------------------------------------------------------------------------
ContinuousStepResult ContinuousGenerationEngine::step(
        int max_tokens,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {
    ContinuousStepResult result;
    if (!initialized_) return result;

    if (batch_dirty_) {
        rebuild_compact_batch();
    }
    if (compact_B_ == 0) return result;

    const int B = compact_B_;
    const int Bp = compact_B_padded_;
    const auto Bp_sz = static_cast<std::size_t>(Bp);
    cudaStream_t stream = run_state_->MainStream;

    // Pre-allocate result vectors.
    result.slot_ids.resize(B);
    result.tokens.resize(B);
    result.finished.resize(B, 0);
    result.completion_lens.resize(B, 0);
    for (int i = 0; i < B; ++i) {
        result.slot_ids[i] = active_slot_ids_[i];
        result.tokens[i].reserve(max_tokens);
    }

    // Pre-allocate host readback buffer for multi-step.
    const std::size_t step_host_size = Bp_sz * static_cast<std::size_t>(max_tokens);
    std::vector<int32_t> all_sampled_host(step_host_size, 0);

    // --- Multi-token decode loop (pure GPU, no sync until the end) ---
    // Full-step CUDA graph: captures run_one_decode_step + D2H copy in one graph.
    // Priming: first step runs eagerly to trigger lazy allocations.
    // Capture: second step captures the graph.
    // Replay: subsequent steps replay at near-zero CPU overhead.
    const bool use_full_step_graph = use_cuda_graphs_ && !greedy_;  // greedy has no RNG issues
    // (For greedy, graph is also fine — enable unconditionally when cuda_graphs on.)
    // Disable full-step graph for hybrid models — recurrent/conv state
    // pointers in the DecodeState are accessed via unordered_map which
    // the graph cannot track across replays.
    const bool has_recurrent = !decode_state_.recurrent_states ||
        !decode_state_.recurrent_states->empty();
    const bool has_conv = !decode_state_.conv_states ||
        !decode_state_.conv_states->empty();
    const bool is_hybrid = has_recurrent || has_conv;
    const bool try_full_step_graph = use_cuda_graphs_ && !is_hybrid;

    int executed = 0;
    for (; executed < max_tokens; ++executed) {
        if (try_full_step_graph && full_step_primed_) {
            // Capture or replay the full-step graph.
            dsl::trace_or_execute_cuda_graph_with_stack(
                [&]() {
                    run_one_decode_step(graph_executor, comm, hook);
                },
                stream,
                full_step_graph_exec_,
                /*enabled=*/true,
                run_state_->Stack,
                full_step_graph_checkpoint_);
        } else {
            // Eager execution (first step primes lazy allocations).
            run_one_decode_step(graph_executor, comm, hook);
            if (try_full_step_graph && !full_step_primed_) {
                full_step_primed_ = true;
            }
        }

        // Async D2H of sampled tokens for this step (into contiguous buffer).
        CUDA_CHECK(cudaMemcpyAsync(
            all_sampled_host.data() + static_cast<std::size_t>(executed) * Bp_sz,
            sampled_tokens_gpu_,
            Bp_sz * sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream));
    }

    // Single sync after all steps.
    CUDA_CHECK(cudaMemcpyAsync(finished_host_.data(), finished_gpu_,
        Bp_sz * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- Scatter results back to slots ---
    for (int step = 0; step < executed; ++step) {
        const auto step_off = static_cast<std::size_t>(step) * Bp_sz;
        for (int i = 0; i < B; ++i) {
            auto& s = slots_[static_cast<std::size_t>(active_slot_ids_[i])];
            const int32_t tok = all_sampled_host[step_off + i];

            // Skip tokens generated after this slot was already finished.
            if (s.finished) continue;

            s.last_token = tok;
            s.seq_len += 1;
            s.generated_count += 1;
            s.rng_offset += 1;
            result.tokens[i].push_back(tok);

            if (tok == s.eos_token_id || s.generated_count >= s.max_gen_len) {
                s.finished = true;
            }
        }
    }

    // Fill result metadata.
    bool any_finished = false;
    for (int i = 0; i < B; ++i) {
        const auto& s = slots_[static_cast<std::size_t>(active_slot_ids_[i])];
        result.finished[i] = s.finished ? 1 : 0;
        result.completion_lens[i] = s.generated_count;
        if (s.finished) any_finished = true;
    }

    // Do NOT rebuild compact batch when sequences finish.
    // Finished sequences stay in their compact positions — masked by
    // finished_gpu_ in decode, and their recurrent/conv state slots are
    // isolated (no cross-contamination).  Rebuild only happens on
    // add_sequence() or release_slot() which change compact positions.

    return result;
}

// ---------------------------------------------------------------------------
// release_slot
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// flat_step — prefill + decode in ONE forward pass via flat-token execution
// ---------------------------------------------------------------------------
ContinuousGenerationEngine::FlatStepResult ContinuousGenerationEngine::flat_step(
        const std::vector<std::vector<int32_t>>& new_prompts,
        const FlatStepConfig& config,
        dsl::GraphExecutor& graph_executor,
        NCCLCommunicator& comm,
        const modules::ForwardHook* hook) {
    FlatStepResult result;
    if (!initialized_) return result;
    cudaStream_t stream = run_state_->MainStream;
    auto& state = ensure_engine_state_storage(this, max_slots_);

    // === Phase 1: Allocate slots + pages for new prompts ===
    result.new_slot_ids.resize(new_prompts.size(), -1);
    struct NewSeq {
        int slot_id;
        int prompt_len;       // full prompt length
        int prefill_len;      // prompt_len - 1 (tokens to prefill)
    };
    std::vector<NewSeq> new_seqs;
    for (std::size_t i = 0; i < new_prompts.size(); ++i) {
        const auto& prompt = new_prompts[i];
        const int plen = static_cast<int>(prompt.size());
        if (plen <= 0) continue;
        const int total_len = plen + config.max_gen_len;
        if (total_len > max_seq_len_) continue;
        const int pages_needed = (total_len + kPageBlockSize - 1) / kPageBlockSize;
        if (free_slot_stack_.empty() || page_pool_.num_free() < pages_needed) continue;

        const int sid = free_slot_stack_.back();
        free_slot_stack_.pop_back();
        auto& slot = slots_[static_cast<std::size_t>(sid)];
        slot = SequenceSlot{};
        slot.active = true;
        slot.max_gen_len = config.max_gen_len;
        slot.eos_token_id = config.eos_token_id;
        slot.temperature = config.temperature;
        slot.top_k = config.top_k;
        slot.top_p = config.top_p;
        slot.min_p = config.min_p;
        slot.page_ids = page_pool_.allocate_pages(pages_needed);
        slot.seq_len = 0;
        slot.last_token = prompt.back();
        slot.generated_count = 0;
        slot.finished = false;

        result.new_slot_ids[i] = sid;
        new_seqs.push_back({sid, plen, plen - 1});
    }

    // === Phase 2: Collect all active slots ===
    std::vector<int> active_sids;
    for (int i = 0; i < max_slots_; ++i) {
        if (slots_[i].active && !slots_[i].finished) {
            active_sids.push_back(i);
        }
    }
    const int batch_size = static_cast<int>(active_sids.size());
    if (batch_size == 0) return result;

    // Build compact recurrent/conv state rows in the same active-slot order as this flat batch.
    // Qwen3.5 decode correctness depends on these persistent per-layer states.
    build_compact_states_from_slots_impl(
        state, run_state_, batch_size, batch_size, active_sids, max_slots_);

    // === Phase 3: Build flat token arrays ===
    // For each request: prefill tokens + 1 decode token (or just 1 decode token for existing)
    // New request contributes prompt_len tokens (prefill_len + 1 decode token = prompt_len)
    // Existing request contributes 1 decode token

    // Compute per-request Q lengths and total tokens
    std::vector<int> q_lens(batch_size, 0);
    int total_tokens = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int sid = active_sids[i];
        const auto& slot = slots_[static_cast<std::size_t>(sid)];
        bool is_new = false;
        for (const auto& ns : new_seqs) {
            if (ns.slot_id == sid) { is_new = true; q_lens[i] = ns.prompt_len; break; }
        }
        if (!is_new) {
            q_lens[i] = 1;  // decode: single token
        }
        total_tokens += q_lens[i];
    }

    // Build host arrays
    std::vector<int32_t> h_token_ids(total_tokens);
    std::vector<int32_t> h_position_ids(total_tokens);
    std::vector<int32_t> h_token_to_req(total_tokens);
    std::vector<int32_t> h_kv_write_pos(total_tokens);
    std::vector<int32_t> h_q_indptr(batch_size + 1, 0);
    std::vector<int32_t> h_seq_lens_k(batch_size, 0);

    int tok_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        h_q_indptr[i] = static_cast<int32_t>(tok_offset);
        const int sid = active_sids[i];
        const auto& slot = slots_[static_cast<std::size_t>(sid)];

        // Find if this is a new sequence
        const std::vector<int32_t>* prompt_ptr = nullptr;
        for (std::size_t ni = 0; ni < new_prompts.size(); ++ni) {
            if (result.new_slot_ids[ni] == sid) {
                prompt_ptr = &new_prompts[ni];
                break;
            }
        }

        if (prompt_ptr) {
            // New request: all prompt tokens (prefill + last token for decode)
            const auto& prompt = *prompt_ptr;
            const int plen = static_cast<int>(prompt.size());
            for (int t = 0; t < plen; ++t) {
                h_token_ids[tok_offset + t] = prompt[t];
                h_position_ids[tok_offset + t] = t;
                h_token_to_req[tok_offset + t] = static_cast<int32_t>(i);
                h_kv_write_pos[tok_offset + t] = t;  // write at positions 0..plen-1
            }
            // After this forward pass, seq_len = plen (all tokens written to KV)
            h_seq_lens_k[i] = static_cast<int32_t>(plen);
            tok_offset += plen;
        } else {
            // Existing request: 1 decode token
            h_token_ids[tok_offset] = slot.last_token;
            h_position_ids[tok_offset] = static_cast<int32_t>(slot.seq_len);
            h_token_to_req[tok_offset] = static_cast<int32_t>(i);
            h_kv_write_pos[tok_offset] = static_cast<int32_t>(slot.seq_len);
            h_seq_lens_k[i] = static_cast<int32_t>(slot.seq_len + 1);
            tok_offset += 1;
        }
    }
    h_q_indptr[batch_size] = static_cast<int32_t>(tok_offset);

    // Build block table (compact, batch_size rows)
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);
    std::vector<int> h_block_table(static_cast<std::size_t>(batch_size) * MPS, 0);
    for (int i = 0; i < batch_size; ++i) {
        const auto& pages = slots_[static_cast<std::size_t>(active_sids[i])].page_ids;
        const auto row = static_cast<std::size_t>(i) * MPS;
        for (std::size_t p = 0; p < pages.size(); ++p) {
            h_block_table[row + p] = pages[p];
        }
    }

    // === Phase 4: Upload to GPU ===
    // Allocate GPU buffers from arena (or reuse pre-allocated ones)
    const auto TT = static_cast<std::size_t>(total_tokens);
    const auto BS = static_cast<std::size_t>(batch_size);

    // Use pre-allocated buffers where possible, temp_alloc for variable-size
    CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu_, h_token_ids.data(),
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(position_ids_gpu_, h_position_ids.data(),
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Use pre-allocated flat-step buffers (no per-step cudaMallocAsync).
    int32_t* d_token_to_req = flat_token_to_req_gpu_;
    int32_t* d_kv_write_pos = flat_kv_write_pos_gpu_;
    int32_t* d_q_indptr = flat_q_indptr_gpu_;
    int32_t* d_seq_lens_k = flat_seq_lens_k_gpu_;

    CUDA_CHECK(cudaMemcpyAsync(d_token_to_req, h_token_to_req.data(),
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_kv_write_pos, h_kv_write_pos.data(),
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_q_indptr, h_q_indptr.data(),
        (BS + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_k, h_seq_lens_k.data(),
        BS * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, h_block_table.data(),
        BS * MPS * sizeof(int), cudaMemcpyHostToDevice, stream));

    // === Phase 5: Build DecodeState for flat-token mode ===
    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    DecodeState ds{};
    ds.flat_token_mode = true;
    ds.flat_batch_size = batch_size;
    ds.flat_total_tokens = total_tokens;
    ds.recurrent_states = &state.compact_recurrent_states;
    ds.recurrent_state_bytes = &state.compact_recurrent_state_bytes;
    ds.conv_states = &state.compact_conv_states;
    ds.conv_state_bytes = &state.compact_conv_state_bytes;
    // Compute total Q tiles for FlashInfer grid dispatch.
    // CTA_TILE_Q=16 (fixed, matches dispatch in attention_flat_paged.cu).
    constexpr int kCtaTileQ = 16;
    const int Hq_model = config_->NumQueryHeads;
    const int Hkv_model = config_->NumKeyValHeads;
    const int group_size = (Hkv_model > 0) ? (Hq_model / Hkv_model) : 1;
    int total_tiles = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int q_len = h_q_indptr[i + 1] - h_q_indptr[i];
        const int packed_len = q_len * group_size;
        total_tiles += (packed_len + kCtaTileQ - 1) / kCtaTileQ;
    }
    ds.flat_padded_batch_size = total_tiles;

    // Compute last-token indices for LM head gather optimization.
    std::vector<int32_t> h_last_token_indices(BS);
    for (int i = 0; i < batch_size; ++i) {
        h_last_token_indices[i] = h_q_indptr[i + 1] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(flat_last_token_indices_gpu_, h_last_token_indices.data(),
        BS * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ds.flat_last_token_indices_gpu = flat_last_token_indices_gpu_;

    ds.token_to_req_gpu = d_token_to_req;
    ds.kv_write_pos_gpu = d_kv_write_pos;
    ds.q_indptr_gpu = d_q_indptr;
    ds.seq_lens_gpu = reinterpret_cast<int*>(d_seq_lens_k);
    ds.cu_seqlens_q_gpu = d_q_indptr;
    ds.cu_seqlens_k_gpu = d_seq_lens_k;
    ds.seq_lens_k_gpu = d_seq_lens_k;
    ds.max_seqlen_k = h_seq_lens_k.empty()
        ? 0
        : *std::max_element(h_seq_lens_k.begin(), h_seq_lens_k.end());
    ds.paged = true;
    ds.fp8 = fp8_kv;
    ds.k_pages = page_pool_.k_pages();
    ds.v_pages = page_pool_.v_pages();
    ds.k_scales = page_pool_.k_scales();
    ds.v_scales = page_pool_.v_scales();
    ds.k_scales_paged_fp8 = page_pool_.k_scales();
    ds.v_scales_paged_fp8 = page_pool_.v_scales();
    ds.per_pool_bytes = page_pool_.full_pool_bytes();
    ds.block_table_gpu = block_table_gpu_;
    ds.block_table_stride = max_pages_per_seq_;
    ds.page_block_size = kPageBlockSize;
    ds.total_pages = page_pool_.total_pages();
    ds.num_kv_heads = config_->NumKeyValHeads;
    ds.head_dim = config_->head_size();
    // With the gather optimization in the LM head, logits are only computed for
    // [batch_size, V] (not total_tokens). Use the pre-allocated logits buffer.
    const int V = vocab_size_;
    ds.logits_out_gpu = logits_gpu_;  // pre-allocated [max_slots * V]
    ds.vocab_size = V;

    // === Phase 5b: Run FlashInfer PrefillPlan (once, reused across all layers) ===
    ds.flat_plan_int_ws_gpu = flat_plan_int_ws_gpu_;
    ds.flat_plan_float_ws_gpu = flat_plan_float_ws_gpu_;
    ds.flat_page_indptr_gpu = flat_page_indptr_gpu_;
    ds.flat_page_indices_gpu = flat_page_indices_gpu_;
    ds.flat_last_page_len_gpu = flat_last_page_len_gpu_;

    flat_attention_plan(
        h_q_indptr.data(), h_seq_lens_k.data(), h_block_table.data(),
        max_pages_per_seq_, kPageBlockSize,
        batch_size, total_tokens,
        config_->NumQueryHeads, config_->NumKeyValHeads, config_->head_size(),
        ds.flat_page_indptr_gpu, ds.flat_page_indices_gpu, ds.flat_last_page_len_gpu,
        ds.flat_plan_int_ws_gpu, ds.flat_plan_float_ws_gpu,
        kPlanIntWsSize, kPlanFloatWsSize,
        static_cast<void*>(ds.flat_plan_info_storage),
        stream);
    ds.flat_padded_batch_size = total_tiles;

    // === Phase 6: Execute flat-token forward pass ===
    graph_executor.execute_flat_tokens(
        static_cast<long>(total_tokens),
        last_tokens_gpu_,       // token_ids on GPU
        position_ids_gpu_,      // position_ids on GPU
        ds, comm, hook);

    // Persist decode-updated recurrent/conv compact state back into slot-local storage.
    flush_compact_states_to_slots_impl(
        state, run_state_, batch_size, active_sids, max_slots_, slots_);

    // === Phase 7: Sample from logits ===
    // With the LM head gather optimization, logits_gpu_ already contains
    // [batch_size, V] logits (gathered from last-token hidden states).
    float* compact_logits = logits_gpu_;

    // Sample from compact logits
    if (config.temperature <= 0.0f) {
        sampling_argmax(compact_logits, sampled_tokens_gpu_, batch_size, V, stream);
    } else {
        float* temp_arr = nullptr;
        if (config.temperature != 1.0f) {
            temp_arr = temperature_gpu_;  // pre-allocated [max_slots_]
            std::vector<float> h_temp(BS, config.temperature);
            CUDA_CHECK(cudaMemcpyAsync(temp_arr, h_temp.data(), BS * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }
        sampling_softmax(compact_logits, compact_logits, temp_arr,
                         batch_size, V, softmax_ws_, softmax_ws_bytes_, stream);
        if (config.top_k > 0 && config.top_p < 1.0f) {
            sampling_top_k_top_p(compact_logits, sampled_tokens_gpu_,
                config.top_k, config.top_p, batch_size, V, false, 42, 0, stream);
        } else if (config.top_k > 0) {
            sampling_top_k(compact_logits, sampled_tokens_gpu_, config.top_k,
                batch_size, V, false, 42, 0, stream);
        } else if (config.top_p < 1.0f) {
            sampling_top_p(compact_logits, sampled_tokens_gpu_, config.top_p,
                batch_size, V, false, 42, 0, stream);
        } else if (config.min_p > 0.0f) {
            sampling_min_p(compact_logits, sampled_tokens_gpu_, config.min_p,
                batch_size, V, false, 42, 0, stream);
        } else {
            sampling_from_probs(compact_logits, sampled_tokens_gpu_, batch_size, V,
                false, 42, 0, stream);
        }
    }

    // D2H readback
    std::vector<int32_t> h_sampled(BS);
    CUDA_CHECK(cudaMemcpyAsync(h_sampled.data(), sampled_tokens_gpu_,
        BS * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // All buffers are pre-allocated — no per-step frees needed.

    // === Phase 8: Update slot state ===
    result.active_slot_ids.resize(batch_size);
    result.sampled_tokens.resize(batch_size);
    result.finished.resize(batch_size, 0);
    result.completion_lens.resize(batch_size, 0);

    for (int i = 0; i < batch_size; ++i) {
        const int sid = active_sids[i];
        auto& slot = slots_[static_cast<std::size_t>(sid)];
        const int32_t tok = h_sampled[i];

        // For new sequences: set seq_len to prompt_len (KV fully populated)
        bool is_new = false;
        for (const auto& ns : new_seqs) {
            if (ns.slot_id == sid) {
                slot.seq_len = ns.prompt_len;  // all prefill tokens now in KV
                is_new = true;
                break;
            }
        }

        if (!is_new) {
            slot.seq_len += 1;  // decode appended one token to KV
        }
        slot.last_token = tok;
        slot.generated_count += 1;

        if (tok == slot.eos_token_id || slot.generated_count >= slot.max_gen_len) {
            slot.finished = true;
        }

        result.active_slot_ids[i] = sid;
        result.sampled_tokens[i] = tok;
        result.finished[i] = slot.finished ? 1 : 0;
        result.completion_lens[i] = slot.generated_count;
    }

    return result;
}

void ContinuousGenerationEngine::release_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots_) return;
    auto& s = slots_[static_cast<std::size_t>(slot_id)];
    if (!s.active) return;

    page_pool_.free_pages(s.page_ids);
    if (auto* state = get_engine_state_storage(this)) {
        free_slot_states_impl(*state, slot_id);
    }
    s = SequenceSlot{};
    free_slot_stack_.push_back(slot_id);
    batch_dirty_ = true;
    cuda_graph_primed_ = false;  // B may change
}

int ContinuousGenerationEngine::num_active() const {
    int count = 0;
    for (const auto& s : slots_) {
        if (s.active) ++count;
    }
    return count;
}

int ContinuousGenerationEngine::num_free_slots() const {
    return static_cast<int>(free_slot_stack_.size());
}

int ContinuousGenerationEngine::num_free_pages() const {
    return page_pool_.num_free();
}

void ContinuousGenerationEngine::destroy() {
    for (int i = 0; i < max_slots_; ++i) {
        if (slots_[static_cast<std::size_t>(i)].active) {
            release_slot(i);
        }
    }
    if (auto* state = get_engine_state_storage(this)) {
        clear_state_storage_impl(*state);
    }
    erase_engine_state_storage(this);
    // Free plan workspaces (allocated with cudaMalloc, not arena).
    if (flat_plan_int_ws_gpu_) { cudaFree(flat_plan_int_ws_gpu_); flat_plan_int_ws_gpu_ = nullptr; }
    if (flat_plan_float_ws_gpu_) { cudaFree(flat_plan_float_ws_gpu_); flat_plan_float_ws_gpu_ = nullptr; }
    initialized_ = false;
}

}  // namespace infer
