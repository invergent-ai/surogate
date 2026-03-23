// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/infer/continuous_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

}  // namespace

ContinuousGenerationEngine::~ContinuousGenerationEngine() {
    if (full_step_graph_exec_) {
        cudaGraphExecDestroy(full_step_graph_exec_);
        full_step_graph_exec_ = nullptr;
    }
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
}

// ---------------------------------------------------------------------------
// rebuild_compact_batch — only called when batch_dirty_ is set
// ---------------------------------------------------------------------------
void ContinuousGenerationEngine::rebuild_compact_batch() {
    cudaStream_t stream = run_state_->MainStream;
    const auto MPS = static_cast<std::size_t>(max_pages_per_seq_);

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

    // Build persistent DecodeState.
    const bool fp8_kv = (page_pool_.dtype() == KVDType::FP8_E4M3);
    decode_state_ = DecodeState{};
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
    // When full-step graph capture is active, the inner decode graph is subsumed —
    // pass use_cuda_graph=false to avoid nested graph capture.
    graph_executor.execute_decode_step(
        static_cast<long>(Bp), last_tokens_gpu_, position_ids_gpu_,
        decode_state_, comm, hook, /*use_cuda_graph=*/false);

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
    prefill_ds.prefill_mode = true;
    prefill_ds.prefill_pos_offset = 0;

    const int prefill_len = plen - 1;
    if (prefill_len > 0) {
        const int chunk = (prefill_chunk_size > 0)
            ? std::min(prefill_chunk_size, prefill_len) : prefill_len;
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
        prefill_ds.prefill_mode = true;

        // Chunked prefill: process all B_padded sequences together, chunk by chunk.
        // Padding rows [B, B_padded) get token 0 / pos 0 which is harmless.
        const int chunk_size = (config.prefill_chunk_size > 0)
            ? std::min(config.prefill_chunk_size, max_prefill_len) : max_prefill_len;

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
    const bool try_full_step_graph = use_cuda_graphs_;

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

    // If any sequence finished, the compact batch is stale.
    if (any_finished) {
        batch_dirty_ = true;
        cuda_graph_primed_ = false;  // B may change
    }

    return result;
}

// ---------------------------------------------------------------------------
// release_slot
// ---------------------------------------------------------------------------
void ContinuousGenerationEngine::release_slot(int slot_id) {
    if (slot_id < 0 || slot_id >= max_slots_) return;
    auto& s = slots_[static_cast<std::size_t>(slot_id)];
    if (!s.active) return;

    page_pool_.free_pages(s.page_ids);
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
    initialized_ = false;
}

}  // namespace infer
