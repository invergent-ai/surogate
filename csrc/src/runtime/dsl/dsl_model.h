// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_H

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>

#include "runtime/dsl/ir.h"
#include "runtime/dsl/dsl_runtime_config.h"
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
class DslParamStore;
class DslGradStore;
class DslRunState;
class DslWeightManager;

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

    modules::ModularLoRAWeightsManager& lora_weights();
    modules::ModularLoRAGradsManager& lora_grads();
    modules::LoRARunState& lora_run_state();
    [[nodiscard]] const modules::ModularLoRAConfig& lora_config() const {
        return *mLoRAConfig;
    }
    const DslGradStore& grads() const;

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
    bool mUseTokenScale = true;               // apply 1/valid_token_count in global_norm_sqrt
    bool mDocMaskingActive = false;           // set by forward(), cleared by backward()
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
