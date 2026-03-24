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
#include <limits>
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

inline bool flat_cuda_graphs_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* env = std::getenv("SUROGATE_SERVE_FLAT_CUDA_GRAPHS");
    if (!env || env[0] == '\0') {
        cached = 1;  // default on; can be disabled with SUROGATE_SERVE_FLAT_CUDA_GRAPHS=0
        return true;
    }
    cached = (env[0] != '0') ? 1 : 0;
    return cached != 0;
}

inline bool supports_fp8_kv_cache(const cudaDeviceProp& prop) {
    const char* env = std::getenv("SUROGATE_ENABLE_FP8_KV_CACHE");
    if (!env || !*env || *env == '0') return false;
    return (prop.major * 10 + prop.minor) >= 89;
}

inline bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "False") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

std::vector<int> build_graph_buckets(int max_slots) {
    // Match vLLM's capture sizes: powers of 2 up to 16, then linear steps of 8.
    std::vector<int> buckets;
    for (int b = 1; b <= std::min(16, max_slots); b *= 2) {
        buckets.push_back(b);
    }
    for (int b = 24; b <= max_slots; b += 8) {
        buckets.push_back(b);
    }
    if (buckets.empty() || buckets.back() < max_slots) {
        buckets.push_back(max_slots);
    }
    return buckets;
}

std::vector<int> build_prefill_token_buckets(int max_tokens) {
    // Match sglang-style piecewise capture buckets to maximize graph reuse:
    // small tokens use fine granularity, larger tokens use coarser steps.
    std::vector<int> buckets;
    buckets.reserve(128);
    auto append_range = [&](int start, int end, int step) {
        for (int t = start; t <= end && t <= max_tokens; t += step) {
            buckets.push_back(t);
        }
    };
    append_range(4, 32, 4);
    append_range(48, 256, 16);
    append_range(288, 512, 32);
    append_range(576, 1024, 64);
    append_range(1280, 4096, 256);
    append_range(4608, max_tokens, 512);

    if (buckets.empty() || buckets.back() < max_tokens) {
        buckets.push_back(max_tokens);
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

inline int resolve_flat_prefill_chunk_size(
        int requested_chunk_size,
        int token_budget,
        const dsl::DslRunState& run_state) {
    const int budget = std::max(1, token_budget);
    int chunk = (requested_chunk_size <= 0)
        ? budget
        : std::min(requested_chunk_size, budget);
    const int static_t_cap = static_prefill_t_cap(run_state);
    if (static_t_cap > 0) {
        chunk = std::min(chunk, static_t_cap);
    }
    return std::max(1, chunk);
}

inline bool is_valid_token_id(const int32_t tok, const int vocab_size) {
    return tok >= 0 && tok < vocab_size;
}

inline int32_t resolve_fallback_token_id(const int32_t preferred, const int vocab_size) {
    if (is_valid_token_id(preferred, vocab_size)) {
        return preferred;
    }
    return 0;
}

inline void try_free_device_buffer(void*& ptr) {
    if (!ptr) return;
    const cudaError_t free_st = cudaFree(ptr);
    if (free_st != cudaSuccess) {
        (void)cudaGetLastError();
    }
    ptr = nullptr;
}

inline void ensure_device_buffer_capacity(
        void*& ptr,
        std::size_t required_bytes,
        std::size_t& capacity_bytes,
        int expected_device_id) {
    auto is_valid_device_ptr = [expected_device_id](void* p) -> bool {
        if (!p) return false;
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess) {
            (void)cudaGetLastError();
            current_device = -1;
        }
        cudaPointerAttributes attrs{};
        const cudaError_t st = cudaPointerGetAttributes(&attrs, p);
        if (st != cudaSuccess) {
            (void)cudaGetLastError();  // clear "invalid value" from pointer query
            return false;
        }
#if CUDART_VERSION >= 10000
        if (attrs.type == cudaMemoryTypeManaged) {
            return true;
        }
        if (attrs.type != cudaMemoryTypeDevice) {
            return false;
        }
        if (expected_device_id >= 0 && attrs.device >= 0 && attrs.device != expected_device_id) {
            return false;
        }
        if (current_device >= 0 && attrs.device >= 0 && attrs.device != current_device) {
            return false;
        }
        return true;
#else
        if (attrs.memoryType != cudaMemoryTypeDevice) {
            return false;
        }
        return true;
#endif
    };

    if (ptr && capacity_bytes >= required_bytes) {
        if (is_valid_device_ptr(ptr)) {
            return;
        }
        // Stale pointer: drop cached metadata and reallocate below.
        ptr = nullptr;
        capacity_bytes = 0;
    }
    if (ptr) {
        try_free_device_buffer(ptr);
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

    bool pinned = false;  // true after compact buffers are pre-grown to max_slots
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
    const int expected_device = run_state ? run_state->DeviceId : -1;
    auto is_valid_device_ptr = [expected_device](const void* p) -> bool {
        if (!p) return false;
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess) {
            (void)cudaGetLastError();
            current_device = -1;
        }
        cudaPointerAttributes attrs{};
        const cudaError_t st =
            cudaPointerGetAttributes(&attrs, const_cast<void*>(p));
        if (st != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
#if CUDART_VERSION >= 10000
        if (attrs.type == cudaMemoryTypeManaged) {
            return true;
        }
        if (attrs.type != cudaMemoryTypeDevice) {
            return false;
        }
        if (expected_device >= 0 && attrs.device >= 0 && attrs.device != expected_device) {
            return false;
        }
        if (current_device >= 0 && attrs.device >= 0 && attrs.device != current_device) {
            return false;
        }
        return true;
#else
        if (attrs.memoryType != cudaMemoryTypeDevice) {
            return false;
        }
        return true;
#endif
    };

    auto flush_state_map = [&](const std::unordered_map<int, void*>& compact_states,
                               const std::unordered_map<int, std::size_t>& compact_state_bytes,
                               const std::unordered_map<int, std::size_t>& compact_state_capacity_bytes,
                               std::vector<std::unordered_map<int, void*>>& slot_states,
                               std::vector<std::unordered_map<int, std::size_t>>& slot_state_bytes) {
        for (const auto& [layer_idx, src_base_void] : compact_states) {
            if (!src_base_void) continue;
            if (!is_valid_device_ptr(src_base_void)) continue;
            auto it_b = compact_state_bytes.find(layer_idx);
            if (it_b == compact_state_bytes.end() || it_b->second == 0) continue;
            const std::size_t per_seq_bytes = it_b->second;
            auto it_cap = compact_state_capacity_bytes.find(layer_idx);
            if (it_cap == compact_state_capacity_bytes.end() || it_cap->second == 0) continue;
            if (per_seq_bytes > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(std::max(1, B)))) {
                continue;
            }
            const std::size_t required_bytes = static_cast<std::size_t>(B) * per_seq_bytes;
            if (required_bytes > it_cap->second) continue;
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
                if (dst_ptr && !is_valid_device_ptr(dst_ptr)) {
                    dst_ptr = nullptr;
                    bytes_map.erase(layer_idx);
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
        state.compact_recurrent_state_capacity_bytes,
        state.slot_recurrent_states,
        state.slot_recurrent_state_bytes);
    flush_state_map(
        state.compact_conv_states,
        state.compact_conv_state_bytes,
        state.compact_conv_state_capacity_bytes,
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

        // Don't erase compact state entries when all slots are new (empty state).
        // Instead, keep the allocated buffers and just zero them — the recurrent
        // ops need valid pointers to write their initial state into.
        for (auto it = compact_states.begin(); it != compact_states.end(); ++it) {
            const int layer_idx = it->first;
            if (layer_per_seq_bytes.find(layer_idx) == layer_per_seq_bytes.end()) {
                // No slot has state for this layer — keep buffer but ensure it's
                // registered in layer_per_seq_bytes so it gets zeroed below.
                auto it_b = compact_state_bytes.find(layer_idx);
                if (it_b != compact_state_bytes.end() && it_b->second > 0) {
                    layer_per_seq_bytes[layer_idx] = it_b->second;
                }
            }
        }

        for (const auto& [layer_idx, per_seq_bytes] : layer_per_seq_bytes) {
            void*& dst_base = compact_states[layer_idx];
            std::size_t& cap_bytes = compact_state_capacity_bytes[layer_idx];
            if (Bp > 0 && per_seq_bytes > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(Bp))) {
                throw std::runtime_error(
                    "ContinuousGenerationEngine: compact state size overflow for layer "
                    + std::to_string(layer_idx));
            }
            const std::size_t required_bytes =
                static_cast<std::size_t>(Bp) * per_seq_bytes;
            ensure_device_buffer_capacity(
                dst_base,
                required_bytes,
                cap_bytes,
                run_state ? run_state->DeviceId : -1);
            compact_state_bytes[layer_idx] = per_seq_bytes;
            cudaError_t memset_st = cudaMemsetAsync(dst_base, 0, required_bytes, stream);
            if (memset_st == cudaErrorInvalidValue) {
                (void)cudaGetLastError();
                // Cached pointer likely stale/cross-device. Drop and reallocate once.
                dst_base = nullptr;
                cap_bytes = 0;
                ensure_device_buffer_capacity(
                    dst_base,
                    required_bytes,
                    cap_bytes,
                    run_state ? run_state->DeviceId : -1);
                memset_st = cudaMemsetAsync(dst_base, 0, required_bytes, stream);
            }
            if (memset_st != cudaSuccess) {
                throw std::runtime_error(
                    "ContinuousGenerationEngine: compact state memset failed for layer "
                    + std::to_string(layer_idx)
                    + ", bytes=" + std::to_string(required_bytes)
                    + ", error=" + std::string(cudaGetErrorString(memset_st)));
            }
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
    if (pinned_sampled_) {
        cudaFreeHost(pinned_sampled_);
        pinned_sampled_ = nullptr;
    }
    pinned_.free_all();
    if (sampled_ready_event_) {
        cudaEventDestroy(sampled_ready_event_);
        sampled_ready_event_ = nullptr;
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

    // Size token_ids and position_ids for max(slots, batched_tokens) since
    // flat_step uploads total_tokens which can exceed max_slots during prefill.
    const auto token_buf_size = std::max(S, static_cast<std::size_t>(
        run_state.Inputs.Sizes[0] * (run_state.Inputs.Rank >= 2 ? run_state.Inputs.Sizes[1] : 1)));
    last_tokens_gpu_ = reinterpret_cast<int32_t*>(alloc(token_buf_size * sizeof(int32_t), "ce_last_tokens"));
    seq_lens_gpu_ = reinterpret_cast<int*>(alloc(S * sizeof(int), "ce_seq_lens"));
    finished_gpu_ = reinterpret_cast<int*>(alloc(S * sizeof(int), "ce_finished"));
    position_ids_gpu_ = reinterpret_cast<int32_t*>(alloc(token_buf_size * sizeof(int32_t), "ce_pos_ids"));
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

    // Pinned host buffer for async D2H readback (eliminates cudaStreamSynchronize)
    CUDA_CHECK(cudaMallocHost(&pinned_sampled_, S * sizeof(int32_t)));
    CUDA_CHECK(cudaEventCreateWithFlags(&sampled_ready_event_, cudaEventDisableTiming));

    // Pinned staging buffers for H2D copies in flat_step / rebuild_compact_batch.
    // With pageable std::vector memory, cudaMemcpyAsync synchronizes (~631μs/call).
    // With pinned memory, it returns in ~5μs — eliminating ~6s of host overhead.
    {
        const auto TK = token_buf_size;  // max(slots, batched_tokens)
        CUDA_CHECK(cudaMallocHost(&pinned_.token_ids, TK * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.position_ids, TK * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.token_to_req, TK * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.kv_write_pos, TK * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.q_indptr, (S + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.seq_lens_current, S * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&pinned_.seq_lens_k, S * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.finished_flags, S * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&pinned_.block_table, S * MPS * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&pinned_.last_token_indices, S * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.compact_last_tokens, S * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&pinned_.compact_seq_lens, S * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&pinned_.compact_finished, S * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&pinned_.compact_temp, S * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&pinned_.compact_block_table, S * MPS * sizeof(int)));
        pinned_.allocated = true;
    }

    run_state.ensure_rope_freq_capacity(config, max_seq_len);

    // Pre-allocate flat-step buffers (avoid per-step cudaMallocAsync).
    // max_batched_tokens is the token budget per flat_step.
    // Use the runtime's B*T capacity as the budget.
    max_batched_tokens_ = static_cast<int>(
        run_state.Inputs.Sizes[0] * (run_state.Inputs.Rank >= 2 ? run_state.Inputs.Sizes[1] : 1));
    if (max_batched_tokens_ <= 0) max_batched_tokens_ = 2048;
    prefill_token_buckets_ = build_prefill_token_buckets(max_batched_tokens_);
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

int ContinuousGenerationEngine::next_prefill_token_bucket(int total_tokens) const {
    for (int t : prefill_token_buckets_) {
        if (t >= total_tokens) return t;
    }
    return total_tokens;
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

    // Zero GPU buffers so warmup writes are harmless.
    cudaStream_t stream = run_state_->MainStream;
    CUDA_CHECK(cudaMemsetAsync(prefill_block_table_gpu_, 0,
        static_cast<std::size_t>(max_pages_per_seq_) * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(block_table_gpu_, 0,
        static_cast<std::size_t>(max_slots_) * max_pages_per_seq_ * sizeof(int), stream));

    // Warm prefill graphs for common prompt lengths (legacy execute_prefill path).
    static constexpr int kWarmupT[] = {32, 64, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048};
    for (int t : kWarmupT) {
        if (t > max_seq_len_) break;
        std::vector<int32_t> dummy_tokens(static_cast<std::size_t>(t), 0);
        std::vector<int32_t> dummy_pos(static_cast<std::size_t>(t));
        for (int i = 0; i < t; ++i) dummy_pos[i] = i;

        graph_executor.execute_prefill(
            1L, static_cast<long>(t),
            dummy_tokens.data(), dummy_pos.data(),
            warmup_ds, comm);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ========================================================================
    // Warm flat-token path with ALL prefill bucket sizes.
    // This is the path used during actual serving (flat_step → execute_flat_tokens).
    // Without this, every new padded T value triggers ~72 segment CUDA graph
    // captures during inference (~1000 cudaGraphInstantiate + ~4000 cudaMalloc).
    // ========================================================================
    // Flat-token segment graph warmup disabled: each captured CUDA graph
    // consumes ~0.5-1MB GPU memory. Pre-warming 60 buckets × 72 segments
    // ≈ 3GB, which starves KV cache and reduces concurrency.
    // Segment graphs are captured lazily on first use instead.
    if (false && use_cuda_graphs_ && !prefill_token_buckets_.empty()) {
        auto& state = ensure_engine_state_storage(this, max_slots_);
        const int Hq_model = config_->NumQueryHeads;
        const int Hkv_model = config_->NumKeyValHeads;
        const int group_size = (Hkv_model > 0) ? (Hq_model / Hkv_model) : 1;
        constexpr int kCtaTileQ = 16;

        for (int bucket_t : prefill_token_buckets_) {
            if (bucket_t <= 0 || bucket_t > max_batched_tokens_) continue;

            // Build minimal flat-token DecodeState: 1 request with q_len=bucket_t.
            const int batch_size = 1;

            // Zero-fill pinned staging buffers for this warmup call.
            std::memset(pinned_.token_ids, 0, static_cast<std::size_t>(bucket_t) * sizeof(int32_t));
            std::memset(pinned_.position_ids, 0, static_cast<std::size_t>(bucket_t) * sizeof(int32_t));
            std::memset(pinned_.token_to_req, 0, static_cast<std::size_t>(bucket_t) * sizeof(int32_t));
            std::memset(pinned_.kv_write_pos, 0, static_cast<std::size_t>(bucket_t) * sizeof(int32_t));

            // q_indptr: [0, bucket_t]
            pinned_.q_indptr[0] = 0;
            pinned_.q_indptr[1] = static_cast<int32_t>(bucket_t);

            // seq_lens: 0 (no prior KV)
            pinned_.seq_lens_current[0] = 0;
            pinned_.seq_lens_k[0] = static_cast<int32_t>(bucket_t);
            pinned_.finished_flags[0] = 0;
            std::memset(pinned_.block_table, 0, static_cast<std::size_t>(max_pages_per_seq_) * sizeof(int));

            // Upload to GPU
            const auto TT = static_cast<std::size_t>(bucket_t);
            CUDA_CHECK(cudaMemcpyAsync(run_state_->Inputs.Data, pinned_.token_ids,
                TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(run_state_->PositionIDs.Data, pinned_.position_ids,
                TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(flat_token_to_req_gpu_, pinned_.token_to_req,
                TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(flat_kv_write_pos_gpu_, pinned_.kv_write_pos,
                TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(flat_q_indptr_gpu_, pinned_.q_indptr,
                2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu_, pinned_.seq_lens_current,
                sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(flat_seq_lens_k_gpu_, pinned_.seq_lens_k,
                sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, pinned_.block_table,
                static_cast<std::size_t>(max_pages_per_seq_) * sizeof(int),
                cudaMemcpyHostToDevice, stream));

            // Last-token index
            pinned_.last_token_indices[0] = static_cast<int32_t>(bucket_t - 1);
            CUDA_CHECK(cudaMemcpyAsync(flat_last_token_indices_gpu_, pinned_.last_token_indices,
                sizeof(int32_t), cudaMemcpyHostToDevice, stream));

            DecodeState ds{};
            ds.flat_token_mode = true;
            ds.flat_batch_size = batch_size;
            ds.flat_total_tokens = bucket_t;
            ds.q_indptr_host = pinned_.q_indptr;
            ds.flat_max_q_len = bucket_t;
            ds.recurrent_states = &state.compact_recurrent_states;
            ds.recurrent_state_bytes = &state.compact_recurrent_state_bytes;
            ds.conv_states = &state.compact_conv_states;
            ds.conv_state_bytes = &state.compact_conv_state_bytes;
            ds.strict_state_buffers = true;  // enable piecewise graph capture
            const int packed_len = bucket_t * group_size;
            ds.flat_padded_batch_size = (packed_len + kCtaTileQ - 1) / kCtaTileQ;
            ds.flat_last_token_indices_gpu = flat_last_token_indices_gpu_;
            ds.token_to_req_gpu = flat_token_to_req_gpu_;
            ds.kv_write_pos_gpu = flat_kv_write_pos_gpu_;
            ds.q_indptr_gpu = flat_q_indptr_gpu_;
            ds.seq_lens_gpu = seq_lens_gpu_;
            ds.cu_seqlens_q_gpu = flat_q_indptr_gpu_;
            ds.cu_seqlens_k_gpu = flat_seq_lens_k_gpu_;
            ds.seq_lens_k_gpu = flat_seq_lens_k_gpu_;
            ds.max_seqlen_k = bucket_t;
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
            ds.logits_out_gpu = logits_gpu_;
            ds.vocab_size = vocab_size_;
            ds.flat_plan_int_ws_gpu = flat_plan_int_ws_gpu_;
            ds.flat_plan_float_ws_gpu = flat_plan_float_ws_gpu_;
            ds.flat_page_indptr_gpu = flat_page_indptr_gpu_;
            ds.flat_page_indices_gpu = flat_page_indices_gpu_;
            ds.flat_last_page_len_gpu = flat_last_page_len_gpu_;

            graph_executor.execute_flat_tokens(
                static_cast<long>(bucket_t),
                reinterpret_cast<const int32_t*>(run_state_->Inputs.Data),
                reinterpret_cast<const int32_t*>(run_state_->PositionIDs.Data),
                ds, comm, nullptr, /*use_cuda_graph=*/true);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
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

    // Build compact host arrays and upload — use pinned staging buffers.
    int32_t* h_last_tokens = pinned_.compact_last_tokens;
    int*     h_seq_lens = pinned_.compact_seq_lens;
    int*     h_finished = pinned_.compact_finished;
    float*   h_temp = pinned_.compact_temp;
    int*     h_block_table_compact = pinned_.compact_block_table;
    std::memset(h_last_tokens, 0, Bp * sizeof(int32_t));
    std::memset(h_seq_lens, 0, Bp * sizeof(int));
    std::memset(h_finished, 0, Bp * sizeof(int));
    // Init temperature to 1.0f
    for (std::size_t i = 0; i < Bp; ++i) h_temp[i] = 1.0f;
    std::memset(h_block_table_compact, 0, Bp * MPS * sizeof(int));

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
            h_block_table_compact[row + p] = s.page_ids[p];
        }
    }
    // Pad [compact_B_, compact_B_padded_) with finished=1.
    for (int i = compact_B_; i < compact_B_padded_; ++i) {
        h_finished[i] = 1;
    }

    CUDA_CHECK(cudaMemcpyAsync(last_tokens_gpu_, h_last_tokens,
        Bp * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu_, h_seq_lens,
        Bp * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(finished_gpu_, h_finished,
        Bp * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(temperature_gpu_, h_temp,
        Bp * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, h_block_table_compact,
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
    decode_state_.strict_state_buffers = state.pinned;

    // Only invalidate full-step graph if the padded batch SIZE changed.
    // When individual slots change (new request, finished request) but the
    // padded batch size stays the same, the graph is still valid — the H2D
    // uploads above update GPU buffer CONTENTS (seq_lens, block_table, etc.)
    // which are read by kernels at replay time. The CUDA graph captures kernel
    // launch configurations (grid dims, pointers), not data contents.
    // This is the same principle vLLM uses for stable CUDA graph replay.
    if (full_step_graph_exec_ && compact_B_padded_ != prev_compact_B_padded_) {
        CUDA_CHECK(cudaGraphExecDestroy(full_step_graph_exec_));
        full_step_graph_exec_ = nullptr;
        full_step_primed_ = false;
    }
    prev_compact_B_padded_ = compact_B_padded_;

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
    const bool has_recurrent_state =
        decode_state_.recurrent_states && !decode_state_.recurrent_states->empty();
    const bool has_conv_state =
        decode_state_.conv_states && !decode_state_.conv_states->empty();
    const bool is_hybrid_state = has_recurrent_state || has_conv_state;
    const bool use_inner_graph = use_cuda_graphs_ && !full_step_primed_ && !is_hybrid_state;
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
        const bool unconstrained_sampling =
            (batch_top_k_ <= 0) && (batch_top_p_ >= 1.0f) && (batch_min_p_ <= 0.0f);
        if (unconstrained_sampling && batch_temperature_ == 1.0f) {
            // Fuse softmax + categorical sampling for the common serving default path.
            sampling_from_logits(logits_gpu_, sampled_tokens_gpu_, Bp, V,
                false, 42, 0, stream);
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
    const bool has_recurrent = decode_state_.recurrent_states &&
        !decode_state_.recurrent_states->empty();
    const bool has_conv = decode_state_.conv_states &&
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
                full_step_graph_checkpoint_,
                "full_step");
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
    const int V = vocab_size_;
    fallback_token_id_ = resolve_fallback_token_id(config.eos_token_id, V);
    bool compact_batch_needs_rebuild = false;

    // CPU-side phase timing
    static const bool flat_step_profile = std::getenv("SUROGATE_FLAT_STEP_PROFILE") != nullptr;
    static int flat_step_profile_count = 0;
    // Track inter-step gap (time between end of previous flat_step and start of this one)
    static auto last_flat_step_end = std::chrono::high_resolution_clock::now();
    long inter_step_gap_us = 0;
    if (flat_step_profile) {
        inter_step_gap_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - last_flat_step_end).count();
    }
    using hrc = std::chrono::high_resolution_clock;
    auto t_start = hrc::now();
    auto t_phase = t_start;
    auto us_since = [](auto start) {
        return std::chrono::duration_cast<std::chrono::microseconds>(hrc::now() - start).count();
    };
    long p1_us = 0, p2_us = 0, p3_us = 0, p4_us = 0, p5_us = 0, p5b_us = 0, p6_us = 0, p7_us = 0, p8_us = 0;

    // === Phase 1: Allocate slots + pages for new prompts ===
    result.new_slot_ids.resize(new_prompts.size(), -1);
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
        slot.last_token = 0;
        slot.generated_count = 0;
        slot.finished = false;
        slot.prompt = prompt;           // store full prompt for chunked prefill
        slot.prefill_progress = 0;      // nothing prefilled yet

        result.new_slot_ids[i] = sid;
        compact_batch_needs_rebuild = true;
    }

    if (flat_step_profile) { p1_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 2: Select active slots under token budget ===
    // Always schedule all decode slots (1 token each), then add prefill slots
    // up to the flat-step token budget. Prefill admission is sequential and
    // token-budgeted (mini-sglang style), not round-robin micro-chunking.
    const int chunk_size = resolve_flat_prefill_chunk_size(
        config.prefill_chunk_size, max_batched_tokens_, *run_state_);
    const int token_budget = std::max(1, max_batched_tokens_);

    std::vector<int> decode_sids;
    std::vector<int> prefill_sids;
    decode_sids.reserve(static_cast<std::size_t>(max_slots_));
    prefill_sids.reserve(static_cast<std::size_t>(max_slots_));
    for (int i = 0; i < max_slots_; ++i) {
        if (!slots_[i].active || slots_[i].finished) continue;
        if (slots_[i].prefilling()) {
            prefill_sids.push_back(i);
        } else {
            decode_sids.push_back(i);
        }
    }

    std::vector<int> active_sids;
    std::vector<int> prefill_q_lens;
    active_sids.reserve(decode_sids.size() + prefill_sids.size());
    for (int sid : decode_sids) {
        if (static_cast<int>(active_sids.size()) >= token_budget) break;
        active_sids.push_back(sid);
    }

    int remaining_budget = token_budget - static_cast<int>(active_sids.size());
    if (remaining_budget > 0 && !prefill_sids.empty()) {
        for (int sid : prefill_sids) {
            if (remaining_budget <= 0) {
                break;
            }
            const auto& slot = slots_[static_cast<std::size_t>(sid)];
            const int remaining_prompt =
                static_cast<int>(slot.prompt.size()) - slot.prefill_progress;
            if (remaining_prompt <= 0) {
                continue;
            }
            const int desired = std::min(remaining_prompt, chunk_size);
            const int admitted = std::min(desired, remaining_budget);
            if (admitted <= 0) {
                continue;
            }
            active_sids.push_back(sid);
            prefill_q_lens.push_back(admitted);
            remaining_budget -= admitted;
        }
    }

    const int batch_size = static_cast<int>(active_sids.size());
    if (batch_size == 0) return result;
    const bool selected_pure_decode = prefill_q_lens.empty();
    const bool use_bucketed_decode_batch = selected_pure_decode && use_cuda_graphs_;
    const int decode_exec_batch = use_bucketed_decode_batch ? next_bucket(batch_size) : batch_size;
    const bool model_has_stateful_ssm =
        config_ && (config_->MambaNumHeads > 0
                 || config_->MambaSsmStateSize > 0
                 || config_->MambaChunkSize > 0);

    // Build compact recurrent/conv state rows in the same active-slot order as this flat batch.
    // For pure-decode CUDA-graph mode, pad state rows to the decode graph bucket.
    build_compact_states_from_slots_impl(
        state,
        run_state_,
        batch_size,
        use_bucketed_decode_batch ? decode_exec_batch : batch_size,
        active_sids,
        max_slots_);

    if (flat_step_profile) { p2_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 3: Build flat token arrays ===
    // Each decode slot contributes 1 token.
    // Each selected prefill slot contributes its assigned q_len from prefill_q_lens.
    std::vector<int> q_lens(batch_size, 1);
    int prefill_cursor = 0;
    int total_tokens = 0;
    bool has_prefill_slots = false;
    for (int i = 0; i < batch_size; ++i) {
        const int sid = active_sids[static_cast<std::size_t>(i)];
        const auto& slot = slots_[static_cast<std::size_t>(sid)];
        if (slot.prefilling()) {
            has_prefill_slots = true;
            const int q = (prefill_cursor < static_cast<int>(prefill_q_lens.size()))
                ? prefill_q_lens[static_cast<std::size_t>(prefill_cursor)]
                : 1;
            q_lens[i] = std::max(1, q);
            ++prefill_cursor;
        } else {
            q_lens[i] = 1;
        }
        total_tokens += q_lens[i];
    }

    const int static_t_cap = static_prefill_t_cap(*run_state_);
    if (has_prefill_slots && static_t_cap > 0) {
        for (int q : q_lens) {
            if (q > static_t_cap) {
                throw std::runtime_error(
                    "ContinuousGenerationEngine::flat_step: prefill q_len exceeds static runtime T cap");
            }
        }
    }

    if (total_tokens > token_budget) {
        throw std::runtime_error(
            "ContinuousGenerationEngine::flat_step: selected tokens exceed flat-step budget");
    }

    // Keep both logical token count and padded execution token count.
    // Logical count drives flat-token metadata (q_indptr, KV writes, FlashInfer plan).
    // Padded count drives compiled graph shape (B=1, T=padded_total_tokens).
    const int actual_total_tokens = total_tokens;
    int padded_total_tokens = actual_total_tokens;
    const bool has_prefill = has_prefill_slots;
    const bool has_recurrent_state = !state.compact_recurrent_states.empty();
    const bool has_conv_state = !state.compact_conv_states.empty();
    // Piecewise flat graphs: capture non-attention, non-SSM segments as CUDA
    // graphs while running attention and GDN ops eagerly.  Previously disabled
    // for stateful SSM models (Qwen3.5) entirely — now enabled since the
    // graph-breaking ops (FlashAttention, ChunkGatedDeltaRule, Qwen3_5Decay)
    // are handled by the piecewise segment system.
    // Disable piecewise CUDA graphs for steps with prefill. Prefill is
    // compute-bound (large matmuls, long attention) so kernel launch overhead
    // is negligible, but graph compilation for each unique padded T costs
    // 90-267ms. vLLM similarly uses graphs only for decode, not prefill.
    const bool use_flat_graphs = use_cuda_graphs_
        && flat_cuda_graphs_enabled()
        && state.pinned
        && !has_prefill;
    if (has_prefill) {
        // Always snap to a bucket to maximize graph cache reuse.
        // The bucket list is finite (~60 entries) and fully warmed at startup.
        padded_total_tokens = next_prefill_token_bucket(actual_total_tokens);
        padded_total_tokens = std::min(padded_total_tokens, max_batched_tokens_);
    } else if (use_bucketed_decode_batch) {
        // Decode-only CUDA graph mode: pad to the decode graph batch bucket.
        padded_total_tokens = decode_exec_batch;
    }
    const bool pure_decode = !has_prefill && (actual_total_tokens == batch_size);
    const int exec_batch_size = pure_decode ? padded_total_tokens : batch_size;

    // Build host arrays using pinned staging buffers (zero-copy cudaMemcpyAsync).
    // Pageable std::vector memory forces cudaMemcpyAsync to synchronize (~631μs/call).
    // Pinned memory makes it truly async (~5μs/call).
    int32_t* h_token_ids = pinned_.token_ids;
    int32_t* h_position_ids = pinned_.position_ids;
    int32_t* h_token_to_req = pinned_.token_to_req;
    int32_t* h_kv_write_pos = pinned_.kv_write_pos;
    int32_t* h_q_indptr = pinned_.q_indptr;
    int*     h_seq_lens_current = pinned_.seq_lens_current;
    int32_t* h_seq_lens_k = pinned_.seq_lens_k;
    int*     h_finished_flags = pinned_.finished_flags;

    // Zero-initialize (pinned memory persists across calls)
    std::memset(h_token_ids, 0, padded_total_tokens * sizeof(int32_t));
    std::memset(h_position_ids, 0, padded_total_tokens * sizeof(int32_t));
    std::memset(h_token_to_req, 0, padded_total_tokens * sizeof(int32_t));
    std::memset(h_kv_write_pos, 0, padded_total_tokens * sizeof(int32_t));
    std::memset(h_q_indptr, 0, (batch_size + 1) * sizeof(int32_t));
    std::memset(h_seq_lens_current, 0, exec_batch_size * sizeof(int));
    std::memset(h_seq_lens_k, 0, exec_batch_size * sizeof(int32_t));
    std::memset(h_finished_flags, 0, exec_batch_size * sizeof(int));

    int tok_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        h_q_indptr[i] = static_cast<int32_t>(tok_offset);
        const int sid = active_sids[i];
        auto& slot = slots_[static_cast<std::size_t>(sid)];

        if (slot.prefilling()) {
            // Chunked prefill: emit tokens [prefill_progress .. prefill_progress + q_len)
            const int start = slot.prefill_progress;
            const int q_len = q_lens[i];
            for (int t = 0; t < q_len; ++t) {
                const int32_t tok = slot.prompt[start + t];
                h_token_ids[tok_offset + t] = is_valid_token_id(tok, V)
                    ? tok
                    : fallback_token_id_;
                h_position_ids[tok_offset + t] = start + t;
                h_token_to_req[tok_offset + t] = static_cast<int32_t>(i);
                h_kv_write_pos[tok_offset + t] = start + t;
            }
            // seq_lens_k = tokens already in KV + tokens we're adding now
            h_seq_lens_current[i] = slot.seq_len;
            h_seq_lens_k[i] = static_cast<int32_t>(slot.seq_len + q_len);
            tok_offset += q_len;
        } else {
            // Decode: 1 token
            int32_t decode_tok = slot.last_token;
            if (!is_valid_token_id(decode_tok, V)) {
                decode_tok = fallback_token_id_;
                slot.last_token = decode_tok;
            }
            h_token_ids[tok_offset] = decode_tok;
            h_position_ids[tok_offset] = static_cast<int32_t>(slot.seq_len);
            h_token_to_req[tok_offset] = static_cast<int32_t>(i);
            h_kv_write_pos[tok_offset] = static_cast<int32_t>(slot.seq_len);
            h_seq_lens_current[i] = slot.seq_len;
            h_seq_lens_k[i] = static_cast<int32_t>(slot.seq_len + 1);
            tok_offset += 1;
        }
    }
    h_q_indptr[batch_size] = static_cast<int32_t>(tok_offset);
    if (tok_offset != actual_total_tokens) {
        throw std::runtime_error(
            "ContinuousGenerationEngine::flat_step: token packing mismatch in flat batch");
    }
    if (pure_decode && exec_batch_size > batch_size) {
        // Padded decode rows: keep seq_lens at 0, but set seqused_k to 1 and
        // mark finished so decode rope skips appending any KV for padded rows.
        for (int i = batch_size; i < exec_batch_size; ++i) {
            h_seq_lens_k[static_cast<std::size_t>(i)] = 1;
            h_finished_flags[static_cast<std::size_t>(i)] = 1;
        }
    }

    // Build block table (compact, batch_size rows) — use pinned staging
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);
    int* h_block_table = pinned_.block_table;
    std::memset(h_block_table, 0, static_cast<std::size_t>(exec_batch_size) * MPS * sizeof(int));
    for (int i = 0; i < batch_size; ++i) {
        const auto& pages = slots_[static_cast<std::size_t>(active_sids[i])].page_ids;
        const auto row = static_cast<std::size_t>(i) * MPS;
        for (std::size_t p = 0; p < pages.size(); ++p) {
            h_block_table[row + p] = pages[p];
        }
    }

    if (flat_step_profile) { p3_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 4: Upload to GPU ===
    // Allocate GPU buffers from arena (or reuse pre-allocated ones)
    const auto TT = static_cast<std::size_t>(padded_total_tokens);
    const auto BS = static_cast<std::size_t>(batch_size);
    const auto BS_exec = static_cast<std::size_t>(exec_batch_size);
    const std::size_t token_bytes = TT * sizeof(int32_t);

    // Upload flat token/position IDs directly into run-state inputs consumed by
    // GraphExecutor::execute_flat_tokens. This avoids an extra D2D staging copy.
    if (!run_state_->Inputs.Data || token_bytes > run_state_->Inputs.bytes()) {
        throw std::runtime_error("ContinuousGenerationEngine::flat_step: input token buffer exceeds run-state capacity");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        run_state_->Inputs.Data,
        h_token_ids,
        token_bytes,
        cudaMemcpyHostToDevice,
        stream));

    if (!run_state_->PositionIDs.Data) {
        throw std::runtime_error("ContinuousGenerationEngine::flat_step: position_ids buffer is null");
    }
    if (run_state_->PositionIDs.Rank == 3 && run_state_->PositionIDs.Sizes[0] == 3) {
        const std::size_t plane_elems =
            static_cast<std::size_t>(run_state_->PositionIDs.Sizes[1]) *
            static_cast<std::size_t>(run_state_->PositionIDs.Sizes[2]);
        const std::size_t plane_bytes = plane_elems * sizeof(int32_t);
        if (token_bytes > plane_bytes) {
            throw std::runtime_error("ContinuousGenerationEngine::flat_step: position_ids exceed mRoPE plane capacity");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            run_state_->PositionIDs.Data,
            h_position_ids,
            token_bytes,
            cudaMemcpyHostToDevice,
            stream));
    } else {
        if (token_bytes > run_state_->PositionIDs.bytes()) {
            throw std::runtime_error("ContinuousGenerationEngine::flat_step: position_ids exceed run-state capacity");
        }
        CUDA_CHECK(cudaMemcpyAsync(
            run_state_->PositionIDs.Data,
            h_position_ids,
            token_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }

    // Use pre-allocated flat-step buffers (no per-step cudaMallocAsync).
    int32_t* d_token_to_req = flat_token_to_req_gpu_;
    int32_t* d_kv_write_pos = flat_kv_write_pos_gpu_;
    int32_t* d_q_indptr = flat_q_indptr_gpu_;
    int32_t* d_seq_lens_k = flat_seq_lens_k_gpu_;

    CUDA_CHECK(cudaMemcpyAsync(d_token_to_req, h_token_to_req,
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_kv_write_pos, h_kv_write_pos,
        TT * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_q_indptr, h_q_indptr,
        (BS + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(seq_lens_gpu_, h_seq_lens_current,
        BS_exec * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens_k, h_seq_lens_k,
        BS_exec * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    if (pure_decode) {
        CUDA_CHECK(cudaMemcpyAsync(finished_gpu_, h_finished_flags,
            BS_exec * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(block_table_gpu_, h_block_table,
        BS_exec * MPS * sizeof(int), cudaMemcpyHostToDevice, stream));

    if (flat_step_profile) { p4_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 5: Build DecodeState for flat-token mode ===
    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    DecodeState ds{};
    ds.flat_token_mode = true;
    ds.flat_batch_size = batch_size;
    ds.flat_total_tokens = actual_total_tokens;
    ds.q_indptr_host = h_q_indptr;
    // Compute max Q length across requests (for recurrent op padding).
    {
        int mql = 0;
        for (int i = 0; i < batch_size; ++i) {
            mql = std::max(mql, static_cast<int>(h_q_indptr[i + 1] - h_q_indptr[i]));
        }
        ds.flat_max_q_len = mql;
    }
    ds.recurrent_states = &state.compact_recurrent_states;
    ds.recurrent_state_bytes = &state.compact_recurrent_state_bytes;
    ds.conv_states = &state.compact_conv_states;
    ds.conv_state_bytes = &state.compact_conv_state_bytes;
    ds.strict_state_buffers = state.pinned;
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
    int32_t* h_last_token_indices = pinned_.last_token_indices;
    for (int i = 0; i < batch_size; ++i) {
        h_last_token_indices[i] = h_q_indptr[i + 1] - 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(flat_last_token_indices_gpu_, h_last_token_indices,
        BS * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ds.flat_last_token_indices_gpu = flat_last_token_indices_gpu_;

    ds.token_to_req_gpu = d_token_to_req;
    ds.kv_write_pos_gpu = d_kv_write_pos;
    ds.q_indptr_gpu = d_q_indptr;
    ds.seq_lens_gpu = seq_lens_gpu_;
    ds.cu_seqlens_q_gpu = d_q_indptr;
    ds.cu_seqlens_k_gpu = d_seq_lens_k;
    ds.seq_lens_k_gpu = d_seq_lens_k;
    ds.max_seqlen_k = (exec_batch_size == 0)
        ? 0
        : *std::max_element(h_seq_lens_k, h_seq_lens_k + exec_batch_size);
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
    ds.logits_out_gpu = logits_gpu_;  // pre-allocated [max_slots * V]
    ds.vocab_size = V;

    if (flat_step_profile) { p5_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 5b: Run FlashInfer PrefillPlan (once, reused across all layers) ===
    ds.flat_plan_int_ws_gpu = flat_plan_int_ws_gpu_;
    ds.flat_plan_float_ws_gpu = flat_plan_float_ws_gpu_;
    ds.flat_page_indptr_gpu = flat_page_indptr_gpu_;
    ds.flat_page_indices_gpu = flat_page_indices_gpu_;
    ds.flat_last_page_len_gpu = flat_last_page_len_gpu_;

    if (!pure_decode) {
        flat_attention_plan(
            h_q_indptr, h_seq_lens_k, h_block_table,
            max_pages_per_seq_, kPageBlockSize,
            batch_size, actual_total_tokens,
            config_->NumQueryHeads, config_->NumKeyValHeads, config_->head_size(),
            ds.flat_page_indptr_gpu, ds.flat_page_indices_gpu, ds.flat_last_page_len_gpu,
            ds.flat_plan_int_ws_gpu, ds.flat_plan_float_ws_gpu,
            kPlanIntWsSize, kPlanFloatWsSize,
            static_cast<void*>(ds.flat_plan_info_storage),
            stream);
        ds.flat_padded_batch_size = total_tiles;
    }

    if (flat_step_profile) { p5b_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 6: Execute forward pass ===
    // For pure-decode steps (all q_lens=1), use execute_decode_step which
    // supports CUDA graph capture for much faster repeated execution.
    // For mixed steps with prefill tokens, use flat-token eager execution.
    const bool use_decode_path = pure_decode && use_cuda_graphs_;
    if (use_decode_path) {
        // Pure decode: use execute_decode_step with bucketed B to maximize
        // graph-cache reuse (mini-sglang style).
        ds.flat_token_mode = false;
        ds.prefill_mode = false;
        ds.logits_out_gpu = logits_gpu_;
        ds.finished_gpu = finished_gpu_;
        // Keep decode graph signature stable across token steps.
        ds.max_seqlen_k = max_seq_len_;
        fill_decode_cu_seqlens(
            cu_seqlens_q_gpu_,
            seqused_k_gpu_,
            seq_lens_gpu_,
            exec_batch_size,
            stream);
        ds.cu_seqlens_q_gpu = cu_seqlens_q_gpu_;
        ds.cu_seqlens_k_gpu = seqused_k_gpu_;
        ds.seq_lens_k_gpu = seqused_k_gpu_;
        // With pinned buffers, the recurrent/conv state addresses are stable —
        // safe for CUDA graph capture. Set strict_state_buffers so ops don't
        // try to allocate (they must use the pre-allocated buffers).
        ds.flat_batch_size = batch_size;
        ds.strict_state_buffers = state.pinned;
        graph_executor.execute_decode_step(
            static_cast<long>(exec_batch_size),
            reinterpret_cast<const int32_t*>(run_state_->Inputs.Data),
            reinterpret_cast<const int32_t*>(run_state_->PositionIDs.Data),
            ds, comm, hook, state.pinned ? use_cuda_graphs_ : false);
    } else {
        // Mixed prefill+decode: flat-token execution with piecewise CUDA graphs.
        // Non-attention segments are captured/replayed; attention runs eagerly.
        graph_executor.execute_flat_tokens(
            static_cast<long>(padded_total_tokens),
            reinterpret_cast<const int32_t*>(run_state_->Inputs.Data),
            reinterpret_cast<const int32_t*>(run_state_->PositionIDs.Data),
            ds, comm, hook, use_flat_graphs);
    }

    // After the first forward pass, pre-grow all compact state buffers to
    // max_slots_ capacity. This ensures the decode path (with CUDA graphs)
    // never needs to allocate — all buffers are pre-allocated at fixed addresses.
    if (!state.pinned) {
        auto pin_buffers = [&](std::unordered_map<int, void*>& compact_states,
                               std::unordered_map<int, std::size_t>& compact_state_bytes,
                               std::unordered_map<int, std::size_t>& compact_state_capacity_bytes) {
            const std::size_t max_B = static_cast<std::size_t>(max_slots_);
            for (auto& [layer_idx, per_seq_bytes] : compact_state_bytes) {
                if (per_seq_bytes == 0) continue;
                const std::size_t max_bytes = max_B * per_seq_bytes;
                auto& cap = compact_state_capacity_bytes[layer_idx];
                if (cap < max_bytes) {
                    void*& ptr = compact_states[layer_idx];
                    if (ptr) {
                        CUDA_CHECK(cudaFree(ptr));
                        ptr = nullptr;
                    }
                    CUDA_CHECK(cudaMalloc(&ptr, max_bytes));
                    CUDA_CHECK(cudaMemsetAsync(ptr, 0, max_bytes, stream));
                    cap = max_bytes;
                }
            }
        };
        pin_buffers(state.compact_recurrent_states,
                    state.compact_recurrent_state_bytes,
                    state.compact_recurrent_state_capacity_bytes);
        pin_buffers(state.compact_conv_states,
                    state.compact_conv_state_bytes,
                    state.compact_conv_state_capacity_bytes);
        state.pinned = true;
    }

    // Persist decode-updated recurrent/conv compact state back into slot-local storage.
    flush_compact_states_to_slots_impl(
        state, run_state_, batch_size, active_sids, max_slots_, slots_);

    if (flat_step_profile) { p6_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 7: Sample from logits ===
    // With the LM head gather optimization, logits_gpu_ already contains
    // [batch_size, V] logits (gathered from last-token hidden states).
    float* compact_logits = logits_gpu_;
    if (sanitize_logits_enabled()) {
        sampling_sanitize_logits(compact_logits, batch_size, V, stream);
    }

    // Sample from compact logits
    if (config.temperature <= 0.0f) {
        sampling_argmax(compact_logits, sampled_tokens_gpu_, batch_size, V, stream);
    } else {
        const bool unconstrained_sampling =
            (config.top_k <= 0) && (config.top_p >= 1.0f) && (config.min_p <= 0.0f);
        if (unconstrained_sampling && config.temperature == 1.0f) {
            // Fuse softmax + categorical sampling for default decode settings.
            sampling_from_logits(compact_logits, sampled_tokens_gpu_, batch_size, V,
                false, 42, 0, stream);
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
    }
    sampling_sanitize_token_ids(sampled_tokens_gpu_, batch_size, V,
        fallback_token_id_, nullptr, stream);

    // === D2H readback ===
    long p7_launch_us = 0;
    if (flat_step_profile) { p7_launch_us = us_since(t_phase); }
    CUDA_CHECK(cudaMemcpyAsync(pinned_sampled_, sampled_tokens_gpu_,
        BS * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(sampled_ready_event_, stream));
    CUDA_CHECK(cudaEventSynchronize(sampled_ready_event_));

    if (flat_step_profile) { p7_us = us_since(t_phase); t_phase = hrc::now(); }
    // === Phase 8: Update slot state from pinned readback ===
    result.active_slot_ids.resize(batch_size);
    result.sampled_tokens.resize(batch_size);
    result.finished.resize(batch_size, 0);
    result.completion_lens.resize(batch_size, 0);

    for (int i = 0; i < batch_size; ++i) {
        const int sid = active_sids[i];
        auto& slot = slots_[static_cast<std::size_t>(sid)];
        const bool was_prefilling = slot.prefilling();
        int32_t tok = pinned_sampled_[i];
        if (!is_valid_token_id(tok, V)) {
            tok = fallback_token_id_;
        }

        if (was_prefilling) {
            slot.prefill_progress += q_lens[i];
            slot.seq_len = slot.prefill_progress;

            if (!slot.prefilling()) {
                slot.last_token = tok;
                slot.generated_count = 1;
                slot.prompt.clear();
                slot.prompt.shrink_to_fit();
                compact_batch_needs_rebuild = true;

                if (tok == slot.eos_token_id || slot.generated_count >= slot.max_gen_len) {
                    slot.finished = true;
                }
            } else {
                slot.generated_count = 0;
            }
        } else {
            slot.seq_len += 1;
            slot.last_token = tok;
            slot.generated_count += 1;

            if (tok == slot.eos_token_id || slot.generated_count >= slot.max_gen_len) {
                slot.finished = true;
            }
        }

        result.active_slot_ids[i] = sid;
        result.sampled_tokens[i] = tok;
        result.finished[i] = slot.finished ? 1 : 0;
        result.completion_lens[i] = slot.generated_count;
    }

    if (compact_batch_needs_rebuild) {
        batch_dirty_ = true;
    }

    if (flat_step_profile) {
        p8_us = us_since(t_phase);
        const long total_us = us_since(t_start);
        if (flat_step_profile_count < 50) {
            std::fprintf(stderr,
                "[FLAT_STEP] #%d total=%ldus gap=%ldus "
                "P1_alloc=%ld P2_select=%ld P3_build=%ld P4_upload=%ld "
                "P5_state=%ld P5b_plan=%ld P6_forward=%ld P7_launch=%ld P7_sync=%ld P8_update=%ld\n",
                flat_step_profile_count, total_us, inter_step_gap_us,
                p1_us, p2_us, p3_us, p4_us, p5_us, p5b_us, p6_us, p7_launch_us, p7_us, p8_us);
        }
        flat_step_profile_count++;
        last_flat_step_end = std::chrono::high_resolution_clock::now();
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
