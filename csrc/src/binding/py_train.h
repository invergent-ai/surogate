// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_BINDING_PY_TRAIN_H
#define SUROGATE_SRC_BINDING_PY_TRAIN_H

#include <memory>
#include <string>
#include <utility>
#include <thread>
#include <functional>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "config/pretrained_config.h"
#include "runtime/training/runtime_options.h"
#include "config/lora_adapter_config.h"
#include "runtime/qlora/qlora_config.h"
#include "runtime/qlora/dsl_qlora_pipeline.h"
#include "runtime/optimizers/optimizer_config.h"
#include "runtime/dsl/dsl_debug.h"

class DataLoader;
class IModel;
class IGPUUtilTracker;
struct GPUUtilInfo;
struct sSegmentMemory;
class CommunicatorThreadsPack;
class NCCLCommunicator;

//! \brief A multi-GPU trainer wrapper to be used for python bindings
//! \details When wrapping the C++ Surogate core for Python, the  main source of difficulty is handling
//! multi-GPU support. The C++ version supports both multi-process and multi-thread, with
//! multi-thread being the more interesting (due to cudaMemcpy) option.
//! However, mapping multi-threading to python is problematic due to GIL (maybe that will be better once
//! free-threaded python is widely used); hence, this wrapper is used to hide all worker threads
//! from the python interface.
//!
//! Internally, we start up one thread per GPU, and keep track of its training state (`sThreadContext`).
//! Each interface function wraps the desired model call into a std::function that gets sent to the thread
//! context. Each thread runs an infinite loop, and picks up the work it has been sent. Interface functions
//! only return once the work is done. If the work function does not synchronize with the GPU, "done" in this
//! case means that the CPU execution has finished, but the GPU might still be busy. This allows overlap of
//! python execution with GPU execution.
//!
//! As a consequence of this implementation strategy, data loading in python will be slightly different than in the
//! C++ implementation. For C++, each thread has its own DataLoader, providing `B*T` tokens each step. For python,
//! we have only one interface-visible thread, which gets `nGPU*B*T` tokens per step, and splits them into `B*T`-sized
//! chunks for each GPU.
class MultiGPUPyTrainer {
public:
    //! Single-node constructor (original)
    MultiGPUPyTrainer(int ngpus,
                      const PretrainedConfig& config,
                      RuntimeOptions options,
                      int batch_size,
                      int seq_len,
                      int grad_accum,
                      bool memcpy_all_gather,
                      bool memcpy_send_recv,
                      std::optional<LoRAAdapterConfig> lora_config = std::nullopt,
                      std::optional<modules::QLoRAConfig> qlora_config = std::nullopt);

    //! Multi-node constructor (for Ray distributed training)
    MultiGPUPyTrainer(int ngpus,
                      int node_rank,
                      int num_nodes,
                      const void* nccl_id,
                      const PretrainedConfig& config,
                      RuntimeOptions options,
                      int batch_size,
                      int seq_len,
                      int grad_accum,
                      bool memcpy_all_gather,
                      bool memcpy_send_recv,
                      std::optional<LoRAAdapterConfig> lora_config = std::nullopt,
                      std::optional<modules::QLoRAConfig> qlora_config = std::nullopt);

    ~MultiGPUPyTrainer();

    void set_adapter_path(std::string path);
    void import_weights(std::string path);
    void import_weights_from_external(std::string safetensors_path,
                                      std::vector<std::vector<qlora::ExternalWeight>> per_gpu_weights);
    void export_model(std::string path);
    void export_adapter(std::string path, std::string base_model_path = "");
    void init_weights();
    void load_checkpoint(std::string directory, int step);
    void save_checkpoint(std::string directory, int step);
    void step(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids = nullptr);

    /// Chunked-sequence step (KV-checkpointed chunks): forward KV sweep over
    /// chunks 0..N-1, then reverse-order re-forward + backward with exact
    /// dK/dV accumulation. Input arrays are [rows, B, N*T].
    void step_chunked(const std::int32_t* inputs,
                      const std::int32_t* targets,
                      const std::int32_t* position_ids,
                      int seq_chunks);
    float validate(const std::int32_t* inputs, const std::int32_t* targets, const std::int32_t* position_ids = nullptr);

    /// Chunked-sequence eval: forward-only chunk sweep with per-chunk losses
    /// combined weighted by valid-token counts. Arrays are [rows, B, N*T].
    float validate_chunked(const std::int32_t* inputs,
                           const std::int32_t* targets,
                           const std::int32_t* position_ids,
                           int seq_chunks);
    std::pair<float, float> update_with_config(const optimizers::OptimizerConfig& config, int step);
    std::pair<float, float> train_step_graphed(const std::int32_t* inputs,
                                               const std::int32_t* targets,
                                               const std::int32_t* position_ids,
                                               const optimizers::OptimizerConfig& config,
                                               int step);
    void stop();

    std::vector<GPUUtilInfo> get_gpu_info();

    // MoE stats (returns aux/z/load/router diagnostics plus valid flag)
    // Returns zeros with valid=false for non-MoE models
    std::tuple<float, float, float, float, float, float, float, float, float, float, bool> get_moe_stats();

    int world_size() const;
    int local_world_size() const {
        return static_cast<int>(mContexts.size());
    }
    int batch_size() const {
        return B;
    }
    int seq_length() const {
        return T;
    }

    /// Sequence length step() arrays must carry: the graph T times the
    /// chunked-sequence factor (equal to seq_length() when chunking is off).
    int step_seq_length() const {
        return seq_length() * std::max(1, mOptions.SequenceChunks);
    }
    int grad_accumulation() const {
        return mGradAccumulation;
    }
    void set_grad_accumulation(int n) {
        mGradAccumulation = n;
        mTrainMicroStep = 0;
    }
    const PretrainedConfig& config() const {
        return *mConfig;
    }
    const RuntimeOptions& options() const {
        return mOptions;
    }
    bool is_qlora() const {
        return mLoRAConfig.has_value() && mQLoRAConfig.has_value() && mQLoRAConfig->is_quantized();
    }

    std::vector<std::pair<std::string, sSegmentMemory>> get_allocations(int gpu_id);
    std::vector<std::pair<std::string, long>> get_stack_info(int gpu_id);

    // ============================================================================
    // Phase-tree / region / layout introspection (design/buffer-runtime-v4.md
    // Phase 4 debug surface). All getters run a read-only work item on rank 0
    // and return structured data for the `surogate debug tensor-*` subcommands.
    // ============================================================================

    //! Per-tid layout across forward + backward graphs.
    std::vector<dsl::DebugTensorEntry> get_debug_tensor_layout();

    //! Arena sizes + per-graph coverage / per-region counts.
    dsl::DebugArenaSummary get_debug_arena_summary();

    //! Descriptor/capability counts across forward + backward graphs.
    dsl::DebugDescriptorSummary get_debug_descriptor_summary();

    //! Deterministic fusion rewrite preview across forward + backward graphs.
    dsl::DebugFusionPreview get_debug_fusion_preview();

    //! BufferPlan schema/allocation diagnostics.
    dsl::DebugBufferPlanSummary get_debug_buffer_plan_summary();

    //! Phase tree + flattened instruction stream for one graph.
    dsl::DebugPhaseTree get_debug_phase_tree(bool is_backward);

    //! Tids whose arena byte ranges overlap within the same coloring bucket.
    //! Empty under correct compilation (modulo intentional `alias_of`).
    std::vector<dsl::DebugAliasingPair> get_debug_static_aliasing();

    //! Single-tensor provenance lookup. If `name` is empty, uses `tid`;
    //! otherwise resolves via the graph's name→tid map.
    dsl::DebugTensorResolution get_debug_tensor_resolution(const std::string& name, int tid, bool is_backward);

    /// Shrink the DSL stack on every rank to the measured high-water mark plus
    /// `safety_bytes`, provided the savings exceed `min_savings_bytes`.
    /// Intended to be called exactly once by the trainer after the first full
    /// training step has returned — at that point the stack is empty and
    /// `max_utilization` reflects the true runtime peak for this (B, T).
    ///
    /// Returns per-rank (new_size, old_size) pairs. (new_size == 0) means
    /// the rank did not resize (either not worth it or stack unused).
    std::vector<std::pair<long, long>> shrink_stack_after_warmup(long safety_bytes, long min_savings_bytes);
    std::vector<std::pair<std::string, Tensor>> get_gradients(int gpu_id);

    // ---- Debug-only dispatch-PP sub-range parity (GPU 0, resident weights) -
    std::vector<float> dispatch_pp_forward_hidden(const std::int32_t* inputs);
    std::vector<float> dispatch_pp_forward_subranges(const std::int32_t* inputs, int split_after_block);
    std::vector<float> dispatch_pp_grad_norms_whole(const std::int32_t* inputs, const std::int32_t* targets);
    std::vector<float>
    dispatch_pp_grad_norms_subranges(const std::int32_t* inputs, const std::int32_t* targets, int split_after_block);
    // Round-robin forward dispatch of contiguous block stages [los[i]..his[i]]
    // across the GPU pool (stage i -> GPU i % ngpu), handing the boundary residual
    // GPU->host->GPU between stages. Returns the final hidden state as flat f32.
    std::vector<float> dispatch_pp_forward_hidden_multigpu(const std::int32_t* inputs,
                                                           const std::vector<int>& los,
                                                           const std::vector<int>& his);
    // Round-robin backward dispatch (reverse stage order) across the GPU pool,
    // handing the boundary gradients GPU->host->GPU. Returns per-block weight-grad
    // L2 norms collected from whichever GPU computed each block.
    std::vector<float> dispatch_pp_grad_norms_multigpu(const std::int32_t* inputs,
                                                       const std::int32_t* targets,
                                                       const std::vector<int>& los,
                                                       const std::vector<int>& his);
    // Per-GPU weight-residency snapshot (GPU 0; the pool is homogeneous): total
    // device-resident weight bytes, the streaming block double-buffer footprint,
    // and the slot count. Proves the dispatch-PP memory invariant — GPU weight
    // residency is bounded by the slot count, not the layer count.
    std::unordered_map<std::string, std::size_t> dispatch_pp_weight_residency();
    // One full single-GPU dispatch-PP training step (forward -> loss, backward ->
    // grads, optimizer update) through the sub-range executor. Returns the loss.
    float dispatch_pp_train_step(const std::int32_t* inputs,
                                 const std::int32_t* targets,
                                 const optimizers::OptimizerConfig& opt_config,
                                 int step_idx);
    // One full multi-GPU dispatch-PP training step: round-robin backward dispatch
    // with cross-GPU boundary handoff -> collect per-stage grads -> optimizer on the
    // master replica -> broadcast updated weights to every GPU. Returns the loss.
    // stale=true defers the optimizer update by one step (the previous step's grads
    // are applied while this step trains on weights one update behind) — the
    // RoundPipe one-step staleness. Call dispatch_pp_flush_pending at the end to
    // apply the last deferred grads.
    float dispatch_pp_train_step_multigpu(const std::int32_t* inputs,
                                          const std::int32_t* targets,
                                          const std::vector<int>& los,
                                          const std::vector<int>& his,
                                          const optimizers::OptimizerConfig& opt_config,
                                          int step_idx,
                                          bool stale,
                                          int num_microbatches = 1);
    // Grad norm computed by the last dispatch_pp optimizer apply (for the loss display).
    float dispatch_pp_last_grad_norm() const {
        return mDispatchPpLastGradNorm;
    }
    // Apply the last deferred (stale) gradients, if any.
    void dispatch_pp_flush_pending(const optimizers::OptimizerConfig& opt_config);
    std::vector<std::pair<std::string, Tensor>> get_lora_gradients(int gpu_id);
    std::vector<std::pair<std::string, Tensor>> get_lora_weights(int gpu_id);
    int get_valid_token_count(int gpu_id);
    void set_visual_inputs(const std::int32_t* visual_pos_masks,
                           const float* visual_embeds,
                           const std::vector<const float*>& deepstack_visual_embeds);

    // Compute per-token log-probabilities for a batch [B, T].
    // use_lora=true applies LoRA (policy model); use_lora=false skips LoRA (reference model).
    // position_ids: optional [B, T] position IDs for packed sequences (nullptr = sequential).
    // Returns B*T float log-probs; masked positions (target==-100) receive 0.
    std::vector<float> compute_logprobs(const std::int32_t* input_ids,
                                        const std::int32_t* targets,
                                        int B,
                                        int T,
                                        bool use_lora,
                                        const std::int32_t* position_ids = nullptr,
                                        const float* temperatures = nullptr);

    // GRPO: run one training micro-step with externally-computed per-token gradient multipliers.
    // per_token_grads[b*T + t] = dL_GRPO/d(log_prob_policy)[b, t].
    // Replaces the standard d_loss=1.0 seeding; call update_with_config() after grad_accum steps.
    void step_with_custom_loss(const std::int32_t* inputs,
                               const std::int32_t* targets,
                               const float* per_token_grads,
                               const std::int32_t* position_ids = nullptr,
                               const float* temperatures = nullptr);

    // GRPO single-pass: training forward that saves activations AND returns logprobs.
    // Returns B*T float logprobs extracted from the loss buffer (logprob = -loss).
    // Call backward_grpo() after computing per-token grads from these logprobs.
    std::vector<float> forward_for_grpo(const std::int32_t* inputs,
                                        const std::int32_t* targets,
                                        const std::int32_t* position_ids = nullptr,
                                        const float* temperatures = nullptr);

    // GRPO backward pass using activations saved by forward_for_grpo().
    // Inputs/targets/position_ids are reused from forward_for_grpo (already in GPU buffers).
    void backward_grpo(const float* per_token_grads);

    void step_grpo_native(const std::int32_t* inputs,
                          const std::int32_t* targets,
                          const float* inference_logprobs,
                          const float* advantages,
                          const std::uint8_t* loss_mask,
                          const std::int32_t* sample_starts,
                          const std::int32_t* sample_ends,
                          int sample_count,
                          const std::int32_t* position_ids,
                          const float* temperatures,
                          const float* teacher_logprobs,
                          float loss_scale,
                          float ipo_mask_low,
                          float ipo_mask_high,
                          float adv_tau,
                          float teacher_tau,
                          float kl_tau);
    std::unordered_map<std::string, float> get_grpo_native_metrics();

    // Knowledge-distillation training micro-step: standard SFT forward/backward
    // with a top-K teacher signal. kd_ids/kd_logprobs are host arrays of shape
    // [ngpu*B, T, top_k], sliced per GPU like inputs/targets.
    void step_with_kd(const std::int32_t* inputs,
                      const std::int32_t* targets,
                      const std::int32_t* kd_ids,
                      const float* kd_logprobs,
                      const std::int32_t* position_ids,
                      int top_k,
                      float temperature,
                      float kd_weight,
                      float ce_weight);

    // Mean KD loss per valid token accumulated since the last call (rank-0
    // local, mirroring get_grpo_native_metrics). Consumes the accumulator on
    // every rank.
    float get_kd_loss();
    // Host-array layout for step_dpo_native. Per-token arrays and sample/pair
    // arrays are either shared by every GPU row (rows == 1) or carry one row
    // per host batch row. When sample_rows > 1, each sample/pair row is padded
    // with -1 to its configured capacity.
    struct DpoHostLayout {
        int token_rows = 1;
        int sample_rows = 1;
        long token_len = 0;
        long input_rows = 0;
        long input_cols = 0;
    };

    // Offline DPO micro-step. Pair arrays shard together with sample arrays.
    void step_dpo_native(const std::int32_t* inputs,
                         const std::int32_t* targets,
                         const float* ref_logprobs,
                         const std::uint8_t* loss_mask,
                         const std::int32_t* sample_starts,
                         const std::int32_t* sample_ends,
                         int sample_count,
                         const std::int32_t* pair_chosen,
                         const std::int32_t* pair_rejected,
                         int pair_count,
                         const std::int32_t* position_ids,
                         float loss_scale,
                         float beta,
                         int length_norm,
                         DpoHostLayout layout);
    std::unordered_map<std::string, float> get_dpo_native_metrics();

    // DPO reference log-probs via the fused-loss forward (fp8/multi-GPU-safe).
    // `inputs`/`targets`/`position_ids` are [host_rows*B, T] (one B-block per host
    // batch row). Returns logprobs for ALL host rows, [host_rows*B*T] in row order.
    std::vector<float> compute_ref_logprobs_dpo(const std::int32_t* inputs,
                                                const std::int32_t* targets,
                                                const std::int32_t* position_ids,
                                                int input_rows);

private:
    std::unique_ptr<PretrainedConfig> mConfig;  // unique_ptr to preserve polymorphism
    RuntimeOptions mOptions;
    std::optional<LoRAAdapterConfig> mLoRAConfig;
    std::optional<modules::QLoRAConfig> mQLoRAConfig;
    int B;
    int T;

    int mTrainMicroStep = 0;
    int mEvalStep = 0;
    int mGradAccumulation = 1;
    // Controls metric aggregation after a DPO step with distinct host rows.
    bool mDpoShardedRows = false;

    std::unique_ptr<CommunicatorThreadsPack> mThreads;
    struct sFullStepGraphState {
        Tensor chunk_pos_scratch;  ///< pinned [planes, B, chunk_T] staging for chunked pos ids
        cudaGraphExec_t graph_exec = nullptr;
        bool captured = false;
        int captured_B = 0;
        int captured_T = 0;
        int captured_grad_accum = 0;
        bool has_stack_checkpoint = false;
        std::byte* stack_top = nullptr;
        std::size_t stack_alloc_count = 0;
        int captured_doc_cap = 0;
        Tensor opt_params;
        Tensor opt_step;
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        std::vector<Tensor> position_ids;

        /// Destroy the captured graph and clear capture/stack-checkpoint
        /// state. Leaves the pinned I/O buffers (`opt_params`, `opt_step`,
        /// `inputs`, `targets`, `position_ids`) intact — those are still
        /// valid after a stack resize since they live in the allocator.
        void reset_capture() {
            if (graph_exec) {
                (void)cudaGraphExecDestroy(graph_exec);
                graph_exec = nullptr;
            }
            captured = false;
            has_stack_checkpoint = false;
            stack_top = nullptr;
            stack_alloc_count = 0;
            captured_doc_cap = 0;
        }
    };
    struct sThreadContext {
        NCCLCommunicator* Communicator;
        std::unique_ptr<IModel> Model;
        std::unique_ptr<IGPUUtilTracker> GPUUtil;
        std::unique_ptr<sFullStepGraphState> FullStepGraph;
        std::function<void(sThreadContext& ctx)> Work;
    };
    std::vector<sThreadContext> mContexts;
    std::mutex mGlobalMutex;
    std::atomic<bool> mIsRunning = false;
    std::atomic<bool> mHasCrashed = false;
    std::atomic<int> mIsReady = 0;
    std::atomic<int> mWorkDone = 0;

    // dispatch-PP async per-GPU dispatch: launch work on a single GPU without the
    // global barrier of run_work, and wait per-GPU later. This lets the stage
    // scheduler run different stages/microbatches on different GPUs concurrently.
    // Per-GPU monotonic counters: dispatch bumps Pending, the worker bumps Done after
    // it finishes a work item; the GPU is free when Done==Pending. The synchronous
    // run_work path (and mWorkDone) is unchanged.
    std::unique_ptr<std::atomic<int>[]> mCtxPending;
    std::unique_ptr<std::atomic<int>[]> mCtxDone;
    void init_async_slots(std::size_t n);
    void dispatch_async(std::function<void(sThreadContext& ctx)> work, int gpu);
    void wait_gpu(int gpu);

    std::function<void(sThreadContext& ctx)> fetch_work(sThreadContext& ctx);
    void run_work(std::function<void(sThreadContext& ctx)> work, int idx = -1);
    void main_loop(NCCLCommunicator& comm);
    void print_timing_breakdown(int step, int micro_steps);

    // dispatch-PP: apply collected grads on the master GPU (optimizer) and broadcast
    // the updated weights to every replica. opt_step_1based is the Adam step index.
    void dispatch_pp_apply_grads_(const std::vector<std::pair<std::string, std::vector<std::byte>>>& collected,
                                  const optimizers::OptimizerConfig& opt_config,
                                  int opt_step_1based,
                                  int valid_tokens);
    // dispatch-PP one-step-stale state: gradients deferred from the previous step,
    // and the count of optimizer updates applied so far (1-based Adam step).
    std::vector<std::pair<std::string, std::vector<std::byte>>> mDispatchPpPendingGrads;
    int mDispatchPpAppliedStep = 0;
    int mDispatchPpPendingValidTokens = 0;  // valid-token count for the deferred (stale) grads
    float mDispatchPpLastGradNorm = 0.0f;
};

#endif  //SUROGATE_SRC_BINDING_PY_TRAIN_H
