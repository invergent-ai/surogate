// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BasePreTrainedModel - Base class for pretrained transformer models
//
// This file defines the model class inheritance hierarchy that mirrors the
// config class hierarchy from pretrained_config.h:
//
//   BasePreTrainedModel (abstract)
//   └── LlamaModel
//       └── Qwen2Model
//           └── Qwen3Model
//               └── Qwen3MoEModel
//
// Each model class is a thin wrapper around ModularTransformerModel<Block>
// that specifies the correct block type based on the model architecture.
// The heavy lifting is done by ModularTransformerModel template; model classes
// primarily serve as:
//   1. Type aliases for correct Block selection
//   2. Architecture-specific configuration handling
//   3. Model identification for weight I/O mapping
//

#ifndef SUROGATE_SRC_MODULES_MODEL_BASE_MODEL_H
#define SUROGATE_SRC_MODULES_MODEL_BASE_MODEL_H

#include <memory>
#include <string_view>

#include "modular_model.h"
#include "training/model.h"
#include "config/pretrained_config.h"

namespace modules {

// Forward declarations
class NCCLCommunicator;
class TensorAllocator;

/**
 * @brief Abstract base class for pretrained transformer models
 *
 * Provides the common interface and default implementations for all
 * pretrained models. Derived classes (LlamaModel, Qwen2Model, etc.)
 * specify their block types and architecture-specific behavior.
 *
 * This class exists to:
 * 1. Establish the model inheritance hierarchy parallel to config hierarchy
 * 2. Provide type-erased access to model instances through IModel interface
 * 3. Enable runtime polymorphism for model factory pattern
 *
 * Note: The actual implementation is delegated to ModularTransformerModel<Block>.
 * BasePreTrainedModel subclasses are thin wrappers that specify Block type.
 */
class BasePreTrainedModel : public IModel {
public:
    virtual ~BasePreTrainedModel() = default;

    // ========================================================================
    // Model identification - used for weight I/O mapping
    // ========================================================================

    /**
     * @brief Check if this is an MoE model
     */
    [[nodiscard]] virtual bool is_moe_model() const { return false; }

    /**
     * @brief Get the config inheritance type
     *
     * Returns the most specific config type that this model uses.
     */
    [[nodiscard]] virtual PretrainedConfig::ArchitectureId architecture_id() const = 0;

protected:
    BasePreTrainedModel() = default;
};

/**
 * @brief Concept for models that provide architecture metadata
 *
 * Used by ModelFactory to extract architecture info from model instances.
 */
template<typename T>
concept HasArchitectureInfo = requires(const T& model) {
    { model.model_type() } -> std::convertible_to<std::string_view>;
    { model.architecture_id() } -> std::same_as<PretrainedConfig::ArchitectureId>;
    { model.is_moe_model() } -> std::same_as<bool>;
};

/**
 * @brief CRTP mixin to add BasePreTrainedModel interface to ModularTransformerModel
 *
 * This template allows ModularTransformerModel<Block> specializations to
 * expose BasePreTrainedModel's virtual methods without modifying the base template.
 *
 * @tparam Derived The final model class (CRTP pattern)
 * @tparam Block The transformer block type
 */
template<typename Derived, typename Block>
class PreTrainedModelMixin : public ModularTransformerModel<Block>, public BasePreTrainedModel {
public:
    using Base = ModularTransformerModel<Block>;

    // Forward constructor to ModularTransformerModel
    using Base::Base;

    // ========================================================================
    // IModel interface - delegate to ModularTransformerModel
    // ========================================================================

    void init_weights(NCCLCommunicator& comm) override {
        Base::init_weights(comm);
    }

    void import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) override {
        Base::import_weights(file_name, allow_cast, comm);
    }

    void export_weights(const std::string& file_name, NCCLCommunicator& comm) override {
        Base::export_weights(file_name, comm);
    }

    void on_restore_checkpoint(NCCLCommunicator& comm) override {
        Base::on_restore_checkpoint(comm);
    }

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override {
        Base::forward(inputs, position_ids, comm, micro_step);
    }

    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override {
        return Base::validate(inputs, position_ids, targets, comm, micro_step);
    }

    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override {
        Base::backward(inputs, targets, comm, grad_accum_steps, micro_step);
    }

    void update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t,
                float epsilon, float weight_decay, float grad_clip) override {
        Base::update(comm, learning_rate, beta_1, beta_2, t, epsilon, weight_decay, grad_clip);
    }

    void update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) override {
        Base::update_with_config(comm, config, step);
    }

    float get_loss() const override {
        return Base::get_loss();
    }

    ITensorContainer& weights() override {
        return Base::weights();
    }

    ITensorContainer& opt_momentum() override {
        return Base::opt_momentum();
    }

    ITensorContainer& opt_momentum_scales() override {
        return Base::opt_momentum_scales();
    }

    ITensorContainer& opt_variance() override {
        return Base::opt_variance();
    }

    ITensorContainer& opt_variance_scales() override {
        return Base::opt_variance_scales();
    }

    std::vector<std::byte> rng_state() const override {
        return Base::rng_state();
    }

    void set_rng_state(const std::vector<std::byte>& state) override {
        Base::set_rng_state(state);
    }

    std::string_view model_type() const override {
        return Base::model_type();
    }

    IRunState& get_run_state() const override {
        return Base::get_run_state();
    }

    void allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm,
                            int B, int T, bool allocate_optimizer) override {
        Base::allocate_run_state(options, comm, B, T, allocate_optimizer);
    }

    // ========================================================================
    // BasePreTrainedModel interface - implemented by derived class via CRTP
    // ========================================================================

    [[nodiscard]] bool is_moe_model() const override {
        return static_cast<const Derived*>(this)->is_moe_model_impl();
    }

    [[nodiscard]] PretrainedConfig::ArchitectureId architecture_id() const override {
        return static_cast<const Derived*>(this)->architecture_id_impl();
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MODEL_BASE_MODEL_H
