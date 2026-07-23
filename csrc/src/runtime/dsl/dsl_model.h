// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_H

#include <memory>
#include <optional>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>

#include "runtime/dsl/buffer_plan.h"
#include "runtime/dsl/ir.h"
#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/hook_registry.h"
#include "runtime/core/model_config.h"
#include "runtime/lora/lora_config.h"
#include "runtime/lora/lora_grads_manager.h"
#include "runtime/lora/lora_optimizer_state.h"
#include "runtime/lora/lora_run_state.h"
#include "runtime/lora/lora_weights_manager.h"
#include "runtime/qlora/qlora_config.h"
#include "runtime/qlora/dsl_qlora_pipeline.h"
#include "runtime/optimizers/optimizer.h"
#include "runtime/training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"
#include "runtime/core/qlora_provider.h"
#include "runtime/dsl/mapping_spec.h"

namespace modules {
struct HfMapping;
}  // namespace modules

namespace optimizers {
class AdamWOptimizer;
class AdamW8BitOptimizer;
class NorMuonOptimizer;
}  // namespace optimizers

namespace dsl {

class IGraphExecutor;
class GraphExecutor;
class DslParamStore;
class DslGradStore;
class DslRunState;
class DslWeightManager;

struct GrpoNativeLossConfig {
    float loss_scale = 1.0f;
    float ipo_mask_low = 0.2f;
    float ipo_mask_high = 0.2f;
    float adv_tau = 1.0f;
    float teacher_tau = 0.0f;
    float kl_tau = 1.0e-3f;
};

struct GrpoNativeMetrics {
    float policy_loss = 0.0f;
    float mismatch_kl = 0.0f;
    float masked_mismatch_kl = 0.0f;
    float unmasked_mismatch_kl = 0.0f;
    float is_masked = 0.0f;
    float is_masked_low = 0.0f;
    float is_masked_high = 0.0f;
    float teacher_kl = 0.0f;
    float keep_tokens = 0.0f;
    float total_tokens = 0.0f;
};

struct DpoNativeLossConfig {
    float loss_scale = 1.0f;
    float beta = 0.1f;
    // Divide each sequence's response-token logprob sum by its response length
    // (SimPO-style); off for minimal pairs where chosen/rejected lengths match.
    bool length_norm = false;
};

struct DpoNativeMetrics {
    float loss = 0.0f;      ///< mean -log sigmoid(beta * margin) over pairs
    float accuracy = 0.0f;  ///< fraction of pairs with margin > 0
    float margin = 0.0f;    ///< mean margin
    /// Raw pair count the means were normalized by (before the max(.,1) clamp);
    /// lets callers re-weight when combining across data-parallel ranks.
    float pair_count = 0.0f;
};

/// Configuration for the offline knowledge-distillation step.
/// Total loss: ce_weight * CE + kd_weight * tau^2 * KL(teacher_topk || student).
struct KdLossConfig {
    int top_k = 32;
    float temperature = 1.0f;
    float kd_weight = 0.5f;
    float ce_weight = 0.5f;
};

class EmptyTensorContainer final : public ITensorContainer {
public:
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>&) override {
    }
};

class DslModel final : public IModel {
public:
    // Concrete optimizer implementations access DslModel's private state
    // (params, grads, weight manager, LoRA hooks) through friendship.
    friend class ::optimizers::AdamWOptimizer;
    friend class ::optimizers::AdamW8BitOptimizer;
    friend class ::optimizers::NorMuonOptimizer;

    DslModel(const PretrainedConfig& config,
             const RuntimeOptions& options,
             const std::string& ir_json,
             const std::shared_ptr<TensorAllocator>& allocator,
             const std::optional<modules::ModularLoRAConfig>& lora_config = std::nullopt,
             const modules::QLoRAConfig& qlora_config = modules::QLoRAConfig{},
             int shard_idx = 0,
             int num_shards = 1);
    ~DslModel() override;

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;

    // ---- Chunked-sequence training (KV-checkpointed chunks) ----
    void set_sequence_chunk(int idx, int count, const ChunkPackMeta* pack = nullptr) override;
    void zero_sequence_chunk_dkv() override;
    void forward_no_save(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;

    // ---- Debug-only dispatch-PP sub-range parity (BF16 full-FT, resident) --
    // Whole-graph forward; returns the final hidden state flattened to host f32.
    std::vector<float> dispatch_pp_forward_hidden(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm);
    // Forward as two contiguous block sub-ranges [0..split] then [split+1..last],
    // the boundary residual round-tripped through host; returns final hidden f32.
    std::vector<float>
    dispatch_pp_forward_subranges(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int split_after_block);
    // Run one forward stage (blocks [lo..hi]) eagerly, leaving state resident.
    // When inject_layer >= 0, inject ``inject_host`` into get_residual(inject_layer)
    // first (the cross-GPU activation handoff). Read the result via the executor's
    // debug readers (debug_read_residual_bytes / last_block_hidden_f32).
    void dispatch_pp_forward_stage(Tensor inputs,
                                   Tensor position_ids,
                                   NCCLCommunicator& comm,
                                   int lo,
                                   int hi,
                                   std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named,
                                   bool preserve_output);
    // Run one backward stage (blocks [lo..hi]) on this GPU. Forward only this
    // stage's blocks from fwd_inject (block lo-1's residual, captured in the
    // forward pass; empty for lo==0, which forwards from the embedding), then run
    // the bounded backward for [lo..hi]. Forwarding just the stage -- instead of
    // the whole model -- bounds resident activations to one stage, so deep
    // linear-attention models don't overflow the compute stack at longer seq.
    // is_loss_stage (the stage owning the last block) backpropagates from the
    // loss; otherwise inject the incoming boundary gradients (inject_named:
    // d_blocks[hi].res_att / .mlp_down). Read results via the executor readers
    // (block_grad_norms, read_named_bytes for d_blocks[lo-1].*).
    void dispatch_pp_backward_stage(Tensor inputs,
                                    Tensor targets,
                                    Tensor position_ids,
                                    NCCLCommunicator& comm,
                                    int lo,
                                    int hi,
                                    bool is_loss_stage,
                                    std::vector<std::pair<std::string, std::vector<std::byte>>> fwd_inject,
                                    std::vector<std::pair<std::string, std::vector<std::byte>>> inject_named,
                                    int micro_step = 0,
                                    int total_micro = 1);
    // Whole-graph backward; returns per-block weight-grad L2 norms (block order).
    std::vector<float>
    dispatch_pp_grad_norms_whole(Tensor inputs, Tensor targets, Tensor position_ids, NCCLCommunicator& comm);
    // One full dispatch-PP training step through the forced-eager sub-range
    // executor: forward (computes the loss), backward (grads to the store), then
    // the optimizer update. Returns the step's mean loss. Single-GPU end-to-end
    // convergence keystone (the cross-GPU stage handoff is validated separately).
    float dispatch_pp_train_step(Tensor inputs,
                                 Tensor targets,
                                 Tensor position_ids,
                                 NCCLCommunicator& comm,
                                 const optimizers::OptimizerConfig& opt_config,
                                 int step_idx);
    // --- Multi-GPU dispatch training-step host-transfer primitives ---
    // Read the weight gradients for the parameters of blocks [lo..hi] (and, when
    // include_nonblock, the non-block params: lm_head / final norm) to host,
    // keyed by parameter name. The cross-GPU grad collection for the fused step.
    std::vector<std::pair<std::string, std::vector<std::byte>>>
    dispatch_pp_read_block_grads(int lo, int hi, bool include_head, bool include_embed);
    // Write gradients from host into the grad store by name (host -> device).
    void dispatch_pp_write_grads(const std::vector<std::pair<std::string, std::vector<std::byte>>>& items);
    // Zero the grad accumulators before a backward wavefront. The diagonal schedule
    // hits a stage's microbatches out of order, so grads are pre-zeroed once and every
    // backward_stage accumulates (rather than per-task zeroing).
    void dispatch_pp_zero_grads();
    // Read / write every parameter weight to / from host (the master broadcast).
    std::vector<std::pair<std::string, std::vector<std::byte>>> dispatch_pp_read_weights();
    void dispatch_pp_write_weights(const std::vector<std::pair<std::string, std::vector<std::byte>>>& items);
    // Run the optimizer over the (collected) grad store on this GPU. Uses
    // total-token normalization (ValidTokenCount is not populated on the collecting
    // GPU since the dispatch backward skips the DP loss reduce). step_idx is 1-based.
    float
    dispatch_pp_apply_optimizer(NCCLCommunicator& comm, const optimizers::OptimizerConfig& opt_config, int step_idx);
    // Raw (summed) loss from the last forward — readable without ValidTokenCount,
    // which the dispatch backward leaves unset. Monotone with the mean loss, so it
    // tracks convergence.
    [[nodiscard]] float dispatch_pp_raw_loss() const;
    // Valid (non-pad) tokens the loss stage counted for the last microbatch; the multi-GPU
    // trainer sums these across microbatches and publishes the total via the setter below so
    // dispatch_pp_apply_optimizer (which runs on the master GPU, not the loss GPU) can scale
    // the grad norm per valid token instead of per total (padded) token.
    [[nodiscard]] int dispatch_pp_loss_valid_tokens() const {
        return mDispatchPpLossValidTokens;
    }
    void set_dispatch_pp_valid_tokens(int n) {
        mDispatchPpValidTokens = n;
    }
    // Backward as two contiguous block sub-ranges (high range first, boundary
    // grad round-tripped through host); returns per-block grad norms.
    std::vector<float> dispatch_pp_grad_norms_subranges(Tensor inputs,
                                                        Tensor targets,
                                                        Tensor position_ids,
                                                        NCCLCommunicator& comm,
                                                        int split_after_block);
    void update(NCCLCommunicator& comm,
                float learning_rate,
                float beta_1,
                float beta_2,
                int t,
                float epsilon,
                float weight_decay,
                float grad_clip) override;
    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override;
    void update_with_graph_params(NCCLCommunicator& comm,
                                  const optimizers::OptimizerConfig& config,
                                  const float* opt_params,
                                  const int* opt_step);
    void prepare_optimizer_state_for_graph(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config);
    void zero_grads(cudaStream_t stream);
    void set_internal_graphs_enabled(bool enabled);
    [[nodiscard]] bool internal_graphs_enabled() const;
    [[nodiscard]] bool has_capture_unsafe_ops() const;
    void prepare_bwd_cross_layer_for_capture();

    /// Enable cap-and-pad for outer-capture safety. See
    /// `IGraphExecutor::enable_doc_masking_pad_to_max`.
    void enable_doc_masking_pad_to_max(int max_num_docs, int num_micro_steps);

    /// Stage cu_seqlens for a specific micro-step before cudaGraphLaunch.
    /// Called by the trainer to update each ms's slice in the pinned host
    /// buffer; the captured H2D memcpy reads from a stable per-ms address.
    void stage_cu_seqlens_for_micro_step(int micro_step, const std::int32_t* cu_seqlens_cpu, int count, int total_q);

    /// Re-issue cu_seqlens H2D memcpy for the given micro-step. Called
    /// from DslModel::backward in cap-and-pad mode so each ms's backward
    /// kernels see its own cu_seqlens (the captured graph contains all
    /// forwards before any backward, and the GPU buffer is shared, so
    /// without re-issue every backward sees the LAST forward's data).
    void reissue_cu_seqlens_for_micro_step(int micro_step);

    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override;
    ITensorContainer& opt_momentum_scales() override;
    ITensorContainer& opt_variance() override;
    ITensorContainer& opt_variance_scales() override;

    // LoRA adapter API (no-op for non-LoRA models).
    void export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path = "");
    void import_adapter(const std::string& file_name, NCCLCommunicator& comm);
    void save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;
    void load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;

    [[nodiscard]] bool lora_enabled() const override {
        return mLoRAConfig.has_value() && mLoRAConfig->enabled();
    }
    [[nodiscard]] bool qlora_enabled() const {
        return lora_enabled() && mQLoRAConfig.is_quantized();
    }
    [[nodiscard]] bool is_moe_model() const {
        return mIsMoEModel;
    }
    [[nodiscard]] std::size_t lora_num_parameters() const {
        return mLoRAWeights ? mLoRAWeights->num_parameters() : 0;
    }
    [[nodiscard]] std::size_t qlora_quantized_weights_bytes() const;
    [[nodiscard]] float qlora_memory_savings_ratio() const;
    [[nodiscard]] std::size_t saved_buffers_total_bytes() const;
    [[nodiscard]] int saved_buffers_count() const;
    [[nodiscard]] const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const;
    DslModel& base_model() {
        return *this;
    }
    [[nodiscard]] const modules::ModelConfig& config() const {
        return mModelConfig;
    }

    // Weight streaming/sharding support
    [[nodiscard]] bool is_weight_streaming_enabled() const;
    [[nodiscard]] DslWeightManager* weight_manager() {
        return mWeightManager.get();
    }

    /// Debuggability (design/buffer-runtime-v4.md Phase 4 debug surface):
    /// read-only access to the concrete graph executor so tooling can reach
    /// the compiled forward/backward graphs + phase arenas without friending.
    /// Returns nullptr if no executor has been constructed yet.
    [[nodiscard]] const GraphExecutor* graph_executor() const;
    [[nodiscard]] GraphExecutor* graph_executor();

    modules::ModularLoRAWeightsManager& lora_weights();
    modules::ModularLoRAGradsManager& lora_grads();
    modules::LoRARunState& lora_run_state();
    [[nodiscard]] const modules::ModularLoRAConfig& lora_config() const {
        return *mLoRAConfig;
    }
    const DslGradStore& grads() const;
    [[nodiscard]] const HookRegistry& hook_registry() const {
        return mHookRegistry;
    }

    std::vector<std::byte> rng_state() const override;
    void set_rng_state(const std::vector<std::byte>& state) override;

    /// Set the path to a PEFT adapter to merge into base weights during import.
    /// Must be called before import_weights().
    void set_adapter_path(const std::string& path) {
        mAdapterPath = path;
    }

    /// Compute per-token log-probabilities for a batch of sequences.
    ///
    /// input_ids: CPU int32 token IDs, shape [B, T] (row-major).
    /// targets:   CPU int32 target IDs, shape [B, T]; -100 = masked positions.
    /// B, T:      Batch size and sequence length.
    /// use_lora:  If true and LoRA is enabled, apply LoRA adapters (policy model).
    ///            If false, skip LoRA (reference model).
    ///
    /// Returns a vector of B*T float log-probabilities; masked positions receive 0.
    /// position_ids: Optional CPU int32 position IDs, shape [B, T].
    ///               If nullptr, sequential [0..T-1] per row is used.
    ///               Provide explicit position_ids for packed sequences.
    /// temperatures: Optional CPU float per-token temperatures, shape [B, T].
    ///               If nullptr, uses temperature=1.0 for all tokens.
    std::vector<float> compute_logprobs(const std::int32_t* input_ids,
                                        const std::int32_t* targets,
                                        int B,
                                        int T,
                                        bool use_lora,
                                        NCCLCommunicator& comm,
                                        const std::int32_t* position_ids = nullptr,
                                        const float* temperatures = nullptr);

    /// Run one training micro-step with externally-computed per-token gradient multipliers.
    ///
    /// Equivalent to forward() + backward() but replaces the standard d_loss=1.0 seeding
    /// with the provided per-token gradient values.  Used by GRPO to feed
    /// dL_GRPO/d(log_prob_policy)[t] directly through the LM-head backward.
    ///
    /// inputs:              Forward input tensor (from get_input_buffer()).
    /// position_ids:        Position IDs tensor (from get_position_ids_buffer()).
    /// targets:             Target token IDs tensor (from get_target_buffer()).
    /// per_token_grads_cpu: CPU float32 per-token gradient multipliers, shape [B*T].
    ///   per_token_grads_cpu[b*T + t] = dL_GRPO / d(log_prob_policy)[b, t].
    ///   Masked positions (target==-100) should have 0 gradient.
    /// temperatures: Optional CPU float per-token temperatures, shape [B*T].
    ///   If nullptr, uses temperature=1.0 for all tokens.
    /// grad_accum_steps:    Total gradient accumulation steps.
    /// micro_step:          Current micro-step index within grad_accum_steps.
    void step_with_custom_loss(Tensor inputs,
                               Tensor position_ids,
                               Tensor targets,
                               const float* per_token_grads_cpu,
                               int grad_accum_steps,
                               int micro_step,
                               NCCLCommunicator& comm,
                               const float* temperatures = nullptr);

    /// GRPO single-pass forward: runs training forward (saves activations) AND
    /// returns per-token log-probabilities extracted from the loss buffer.
    /// The caller should compute per-token GRPO gradients from the returned logprobs
    /// and then call backward_grpo() to complete the micro-step.
    ///
    /// logprob[t] = -loss[t] where loss[t] = logsumexp - logit[target[t]].
    /// Masked positions (target == -100) receive 0.
    std::vector<float> forward_for_grpo(Tensor inputs,
                                        Tensor position_ids,
                                        Tensor targets,
                                        int grad_accum_steps,
                                        int micro_step,
                                        NCCLCommunicator& comm,
                                        const float* temperatures = nullptr);

    /// GRPO backward pass using activations saved by forward_for_grpo().
    /// Must be called after forward_for_grpo() in the same micro-step.
    void backward_grpo(Tensor inputs,
                       Tensor targets,
                       const float* per_token_grads_cpu,
                       int grad_accum_steps,
                       int micro_step,
                       NCCLCommunicator& comm);

    void step_grpo_native(Tensor inputs,
                          Tensor position_ids,
                          Tensor targets,
                          const float* inference_logprobs_cpu,
                          const float* advantages_cpu,
                          const std::uint8_t* loss_mask_cpu,
                          const std::int32_t* sample_starts_cpu,
                          const std::int32_t* sample_ends_cpu,
                          int sample_count,
                          int grad_accum_steps,
                          int micro_step,
                          NCCLCommunicator& comm,
                          const GrpoNativeLossConfig& loss_config,
                          const float* temperatures_cpu = nullptr,
                          const float* teacher_logprobs_cpu = nullptr);
    GrpoNativeMetrics consume_grpo_native_metrics();

    /// Knowledge-distillation training step: standard SFT forward + backward
    /// with a top-K teacher signal injected into the fused LM-head loss.
    /// Unlike the GRPO steps this keeps standard loss semantics: CE losses and
    /// ValidTokenCount accumulate across micro-steps, gradients are normalized
    /// by 1/valid_token_count (mUseTokenScale), and get_loss() reports mean CE.
    ///
    /// kd_ids_cpu / kd_logprobs_cpu: CPU arrays of shape [B*T*top_k] holding
    /// the teacher's top-K token ids / raw logprobs, row i aligned with
    /// targets[i]. The KD loss metric accumulates on device; read it once per
    /// optimizer step via consume_kd_loss_sum().
    void step_with_kd(Tensor inputs,
                      Tensor position_ids,
                      Tensor targets,
                      const std::int32_t* kd_ids_cpu,
                      const float* kd_logprobs_cpu,
                      int grad_accum_steps,
                      int micro_step,
                      NCCLCommunicator& comm,
                      const KdLossConfig& kd_config);

    /// Rank-local sum of the KD loss over all micro-steps since the last call
    /// (synchronous D2H read; zeroes the accumulator).
    float consume_kd_loss_sum();

    /// Offline DPO micro-step: forward over a packed batch of (chosen, rejected)
    /// sequences, compute the per-pair sigmoid-DPO gradient against precomputed
    /// reference log-probs, and backward through the LM head. Reuses the GRPO
    /// native scratch (the inference_logprobs buffer carries ref_logprobs).
    ///
    /// ref_logprobs_cpu: CPU FP32 [B*T] reference per-token logprob in the shifted
    ///   layout (ref[out_idx] = logprob of logical token out_idx+1; masked = any).
    /// sample_starts/ends_cpu: token ranges [start,end) of each packed sequence.
    /// pair_chosen/pair_rejected_cpu: per pair, the chosen/rejected SAMPLE index.
    void step_dpo_native(Tensor inputs,
                         Tensor position_ids,
                         Tensor targets,
                         const float* ref_logprobs_cpu,
                         const std::uint8_t* loss_mask_cpu,
                         const std::int32_t* sample_starts_cpu,
                         const std::int32_t* sample_ends_cpu,
                         int sample_count,
                         const std::int32_t* pair_chosen_cpu,
                         const std::int32_t* pair_rejected_cpu,
                         int pair_count,
                         int grad_accum_steps,
                         int micro_step,
                         NCCLCommunicator& comm,
                         const DpoNativeLossConfig& loss_config);
    DpoNativeMetrics consume_dpo_native_metrics();

    /// Per-token reference log-probs for DPO, via the SAME fused-loss forward
    /// step_dpo_native uses (fp8- and multi-GPU-safe), with NO backward. Returns
    /// logprob[i] = -loss[i] in the shifted layout (logprob of logical token i+1),
    /// matching the ref_logprobs the dpo_dloss kernel expects. Call at init: LoRA B
    /// is zero so the forward equals the start checkpoint = π_ref. Unlike
    /// compute_logprobs (forward-only + full logits) this neither trips the GDN
    /// forward-save path nor the fp8 full-logits lm-head GEMM.
    std::vector<float> compute_ref_logprobs(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm);

    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;

    /// Import model weights from external GPU pointers (zero-copy from vLLM).
    ///
    /// Quantized weights are borrowed from the external source (no disk I/O).
    /// Non-quantized weights (norms, biases, embeddings) are still loaded from SafeTensors.
    ///
    /// @param safetensors_path  Path to HF SafeTensors (for non-quantized weights).
    /// @param external_weights  Externally-owned quantized weight descriptors.
    /// @param comm              NCCL communicator.
    void import_weights_from_external(const std::string& safetensors_path,
                                      const std::vector<qlora::ExternalWeight>& external_weights,
                                      NCCLCommunicator& comm);
    void on_restore_checkpoint(NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;
    void prepare_optimizer_for_checkpoint_load() override;

    void allocate_run_state(const RuntimeOptions& options,
                            NCCLCommunicator& comm,
                            int B,
                            int T,
                            bool allocate_optimizer) override;

    /// Schedule deferred QLoRA offloading auto-tune.  The actual tuning
    /// happens at the start of step 1 (inside invalidate_cache) after all
    /// lazy runtime allocations from step 0 are settled.
    void auto_tune_offloading();

    /// Destroy every CUDA-graph capture (whole-graph, per-layer forward/
    /// backward, split-attention segment graphs) and clear the stack
    /// checkpoints that referenced them. Intended to be called after the
    /// DSL stack buffer has been swapped — any captured tensor addresses
    /// from before the swap are invalid.
    void invalidate_cuda_graphs();

    float get_loss() const override;
    float get_accuracy() const override;

    std::string_view model_type() const override;
    IRunState& get_run_state() const override;

    // Type alias for backward compatibility — canonical definition is dsl::MappingSpec.
    using MappingSpec = ::dsl::MappingSpec;

private:
    void validate_ir();
    const Module& pick_model_module(const IRFile& ir) const;
    void validate_config_mapping(const Module& module) const;
    void validate_param_shapes(const Module& module) const;
    void calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream, bool grads_reduced);
    void allocate_lora_run_state(NCCLCommunicator& comm, int B, int T);
    void ensure_lora_run_state(NCCLCommunicator& comm, int B, int T);
    void update_lora_adamw(NCCLCommunicator& comm,
                           float learning_rate,
                           float beta_1,
                           float beta_2,
                           int t,
                           float epsilon,
                           float weight_decay,
                           float grad_clip);
    void update_lora_adamw_graph(NCCLCommunicator& comm, float grad_clip, const float* opt_params, const int* opt_step);
    void update_lora_adamw_8bit(NCCLCommunicator& comm,
                                float learning_rate,
                                float beta_1,
                                float beta_2,
                                int t,
                                float epsilon,
                                float weight_decay,
                                float grad_clip);
    void
    update_lora_adamw_8bit_graph(NCCLCommunicator& comm, float grad_clip, const float* opt_params, const int* opt_step);
    void update_lora_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step);
    void calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip);
    void populate_lora_norm_pointers(NCCLCommunicator& comm, cudaStream_t stream);
    void initialize_lora_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream);
    void initialize_lora_adamw_state(NCCLCommunicator& comm, cudaStream_t stream);
    void update_lora_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream);
    void update_lora_adamw_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream);

    std::unique_ptr<PretrainedConfig> mConfig;
    modules::ModelConfig mModelConfig;
    RuntimeOptions mOptions{};
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unique_ptr<DslRunState> mRunState;
    IRFile mIr;
    const Module* mModule = nullptr;
    DslRuntimeConfig mRuntimeConfig;
    std::vector<BlockSchemaPlanRecord> mBlockSchemaPlanRecords;
    HookRegistry mHookRegistry;
    std::unique_ptr<DslParamStore> mParams;
    std::unique_ptr<DslGradStore> mGrads;
    std::unique_ptr<DslWeightManager> mWeightManager;  // Optional - for streaming/sharding
    EmptyTensorContainer mEmpty;
    std::unique_ptr<optimizers::Optimizer> mOptimizer;
    std::unique_ptr<IGraphExecutor> mExecutor;
    std::vector<std::byte> mRngState;

    // LoRA state (optional)
    std::optional<modules::ModularLoRAConfig> mLoRAConfig;
    std::unique_ptr<modules::ModularLoRAWeightsManager> mLoRAWeights;
    std::unique_ptr<modules::ModularLoRAGradsManager> mLoRAGrads;
    std::unique_ptr<modules::LoRARunState> mLoRARunState;
    std::unique_ptr<modules::LoRAAdamWState> mLoRAAdamWState;
    std::unique_ptr<modules::LoRAAdamW8BitState> mLoRAAdamW8BitState;
    std::unique_ptr<modules::LoRANorMuonState> mLoRANorMuonState;
    bool mIsMoEModel = false;
    bool mUseTokenScale = true;  // apply 1/valid_token_count in global_norm_sqrt
    // dispatch-PP: grads are collected complete onto this GPU by hand (not via DDP),
    // so the optimizer must skip the cross-GPU grad/norm all-reduce that would
    // deadlock waiting for the idle pool.
    bool mDispatchPpLocalGrads = false;
    float mDispatchPpLastLoss = 0.0f;         // mean loss stashed by the dispatch backward's forward
    int mDispatchPpLossValidTokens = 0;       // valid (non-pad) target tokens the loss stage counted
                                              // for the current microbatch (read back per microbatch)
    int mDispatchPpValidTokens = 0;           // valid tokens summed over the step's microbatches,
                                              // published onto the optimizer GPU for valid-token
                                              // grad-norm scaling (the dispatch path skips reduce_loss)
    bool mDocMaskingActive = false;           // set by forward(), cleared by backward()
    bool mSequenceChunkActive = false;        // chunked-sequence mode (doc masking handled per chunk)
    float* mGrpoInvTemperatureGpu = nullptr;  // persists from forward_for_grpo() to backward_grpo()

    // Adapter merge state (optional — stacked LoRA)
    std::string mAdapterPath;

    // QLoRA state (optional)
    modules::QLoRAConfig mQLoRAConfig;
    int mShardIdx = 0;
    int mNumShards = 1;
    std::unique_ptr<QLoRAWeightProvider> mQLoRAProvider;

    std::unordered_map<std::string, MappingSpec> mHfMapping;
    std::unordered_map<std::string, MappingSpec> mHfExport;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_MODEL_H
