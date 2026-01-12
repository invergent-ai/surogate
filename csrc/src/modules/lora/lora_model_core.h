// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_CORE_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_CORE_H

#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

#include "lora_config.h"
#include "lora_weights.h"
#include "lora_run_state.h"
#include "lora_optimizer_state.h"
#include "modules/model/modular_model.h"
#include "modules/qlora/qlora_config.h"
#include "modules/qlora/fp8_weight_provider.h"
#include "modules/qlora/fp4_weight_provider.h"
#include "modules/qlora/bnb_weight_provider.h"
#include "training/model.h"
#include "training/runtime_options.h"
#include "utilities/allocator.h"
#include "utilities/comm.h"

namespace modules {

template<typename Block>
class ModularLoRAModel final : public IModel {
public:
    ModularLoRAModel(std::unique_ptr<ModularTransformerModel<Block>> base_model,
                     const ModularLoRAConfig& lora_config,
                     const RuntimeOptions& options,
                     NCCLCommunicator& comm,
                     const std::shared_ptr<TensorAllocator>& allocator,
                     const QLoRAConfig& qlora_config = QLoRAConfig{});

    // IModel
    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T, bool allocate_optimizer) override;
    void init_weights(NCCLCommunicator& comm) override;
    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override;
    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override;
    void on_restore_checkpoint(NCCLCommunicator& comm) override { mBaseModel->on_restore_checkpoint(comm); }
    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;
    
    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2,
                int t, float epsilon, float weight_decay, float grad_clip) override;
    
    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override;

    float get_loss() const override { return mBaseModel->get_loss(); }

    ITensorContainer& weights() override;
    ITensorContainer& opt_momentum() override;
    ITensorContainer& opt_momentum_scales() override;
    ITensorContainer& opt_variance() override;
    ITensorContainer& opt_variance_scales() override;

    std::vector<std::byte> rng_state() const override { return mBaseModel->rng_state(); }
    void set_rng_state(const std::vector<std::byte>& state) override { mBaseModel->set_rng_state(state); }
    std::string_view model_type() const override { return mBaseModel->model_type(); }
    IRunState& get_run_state() const override { return mBaseModel->get_run_state(); }

    // LoRA API
    [[nodiscard]] bool lora_enabled() const { return mLoRAConfig.enabled(); }
    [[nodiscard]] bool qlora_enabled() const { return mQLoRAConfig.is_quantized(); }
    [[nodiscard]] bool is_moe_model() const { return mIsMoEModel; }
    [[nodiscard]] std::size_t lora_num_parameters() const { return mLoRAWeights ? mLoRAWeights->num_parameters() : 0; }

    ModularTransformerModel<Block>& base_model() { return *mBaseModel; }
    const ModularLoRAConfig& lora_config() const { return mLoRAConfig; }
    const QLoRAConfig& qlora_config() const { return mQLoRAConfig; }
    ModularLoRAWeightsManager& lora_weights() { return *mLoRAWeights; }
    ModularLoRAGradsManager& lora_grads() { return *mLoRAGrads; }

    // QLoRA memory stats
    [[nodiscard]] std::size_t qlora_quantized_weights_bytes() const;
    [[nodiscard]] float qlora_memory_savings_ratio() const;

    void export_adapter(const std::string& directory, NCCLCommunicator& comm, const std::string& base_model_path = "");
    void import_adapter(const std::string& file_name, NCCLCommunicator& comm);
    void save_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;
    void load_lora_checkpoint(const std::string& checkpoint_dir, NCCLCommunicator& comm) override;

    // Debug API
    float debug_get_grad_scale() const;
    int debug_get_valid_tokens() const;
    std::vector<float> debug_get_grad_norms_by_layer(NCCLCommunicator& comm) const;
    std::vector<std::pair<std::string, float>> debug_get_grad_norms_by_module(int layer_idx, NCCLCommunicator& comm) const;

private:
    std::unique_ptr<ModularTransformerModel<Block>> mBaseModel;
    ModularLoRAConfig mLoRAConfig;
    QLoRAConfig mQLoRAConfig;
    RuntimeOptions mOptions;
    std::shared_ptr<TensorAllocator> mAllocator;

    std::unique_ptr<ModularLoRAWeightsManager> mLoRAWeights;
    std::unique_ptr<ModularLoRAGradsManager> mLoRAGrads;
    std::unique_ptr<LoRAAdamW8BitState> mLoRAAdamW8BitState;
    std::unique_ptr<LoRANorMuonState> mLoRANorMuonState;
    std::unique_ptr<LoRARunState> mLoRARunState;

    bool mIsMoEModel = false;

    std::unique_ptr<FP8WeightProvider<Block>> mFP8WeightProvider;
    std::unique_ptr<FP4WeightProvider<Block>> mFP4WeightProvider;
    std::unique_ptr<BnBWeightProvider<Block>> mBnBWeightProvider;

    std::minstd_rand mLoRAOptimizerRNG;

    // Private helper methods
    void allocate_lora_run_state(NCCLCommunicator& comm, int B, int T);
    void ensure_lora_run_state(NCCLCommunicator& comm, int B, int T);
    Tensor recompute_rmsnorm(const Tensor& residual, const Tensor& ln_weight, float epsilon, int B, int T, int C, cudaStream_t stream);

    // QLoRA loading
    void import_weights_qlora(const std::string& file_name, NCCLCommunicator& comm);
    void import_weights_fp4_qlora(const std::string& file_name, NCCLCommunicator& comm);
    void import_weights_bnb_qlora(const std::string& file_name, NCCLCommunicator& comm);

    // Backward helpers
    void backward_lora_qkv(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream);
    void backward_lora_attn_out(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream);
    void backward_lora_mlp_up(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream);
    void backward_lora_mlp_down(int layer_idx, int B, int T, bool accumulate, NCCLCommunicator& comm, cudaStream_t stream);
    void calculate_lora_gradient_norm(NCCLCommunicator& comm, float grad_clip);

    // Optimizer helpers
    void update_adamw_8bit(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon, float weight_decay, float grad_clip);
    void update_normuon(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step);
    void initialize_multi_tensor_state(NCCLCommunicator& comm, cudaStream_t stream);
    void update_grad_pointers(NCCLCommunicator& comm, cudaStream_t stream);
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_CORE_H
