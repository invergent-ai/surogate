// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Internal helper declarations for DSL model implementation.

#ifndef SUROGATE_SRC_DSL_DSL_MODEL_INTERNAL_H
#define SUROGATE_SRC_DSL_DSL_MODEL_INTERNAL_H

#include <optional>
#include <string>
#include <vector>
#include <functional>

#include "dsl/dsl_model.h"
#include "dsl/ir.h"
#include "modules/lora/lora_optimizer_state.h"
#include "utilities/tensor_container.h"

namespace dsl {
namespace internal {

// Environment variable helpers
float env_float(const char* name, float fallback);
int env_int(const char* name, int fallback);

// CUDA stream capture utilities
bool stream_is_capturing(cudaStream_t stream);
void wait_event_if_not_capturing(cudaStream_t stream, cudaEvent_t event);
void record_event_if_not_capturing(cudaEvent_t event, cudaStream_t stream);

// HuggingFace config value type
struct HfValue {
    enum class Kind { Int, Float, Bool };
    Kind kind;
    long i = 0;
    double f = 0.0;
    bool b = false;
};

// Tensor container for LoRA optimizer states
class LoRAAdamW8BitStateContainer final : public ITensorContainer {
public:
    explicit LoRAAdamW8BitStateContainer(modules::LoRAAdamW8BitState* state);
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    modules::LoRAAdamW8BitState* mState = nullptr;
};

class LoRANorMuonStateContainer final : public ITensorContainer {
public:
    explicit LoRANorMuonStateContainer(modules::LoRANorMuonState* state);
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    modules::LoRANorMuonState* mState = nullptr;
};

// Config/attribute value helpers
std::optional<HfValue> get_hf_value(const PretrainedConfig& cfg, const std::string& key);
std::optional<HfValue> attr_to_value(const AttrValue& value);
bool values_match(const HfValue& expected, const HfValue& actual);

const AttrMap* as_map(const AttrValue& value);
const AttrList* as_list(const AttrValue& value);
std::optional<std::string> as_string(const AttrValue& value);
std::optional<long> as_int(const AttrValue& value);
std::optional<bool> as_bool(const AttrValue& value);
const AttrValue* find_key(const AttrMap* map, const std::string& key);

// String utilities
bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);
std::string to_lower(std::string s);
bool contains_ci(const std::string& haystack, const std::string& needle);
void replace_all(std::string& str, std::string_view from, std::string_view to);
std::string format_hf_name(std::string templ, int layer_idx, int expert_idx = -1);

// HuggingFace mapping helpers
DslModel::MappingSpec parse_mapping_spec(const AttrValue& value);
const DslModel::MappingSpec* find_mapping_spec(
    const std::unordered_map<std::string, DslModel::MappingSpec>& mapping,
    const std::string& internal_name,
    int& layer_idx);

// Tensor utilities
Tensor slice_dim0(const Tensor& base, long offset, long length);
bool is_norm_param_name(const std::string& name);
bool is_bias_param_name(const std::string& name);
std::vector<long> infer_fuse_slices(const std::string& name, const PretrainedConfig& cfg, int num_sources);

// QLoRA provider factory (DSL)
std::unique_ptr<QLoRAWeightProvider> create_dsl_qlora_provider(
    const modules::ModelConfig& model_cfg,
    const RuntimeOptions& options,
    const modules::ModularLoRAConfig& lora_cfg,
    const modules::QLoRAConfig& qlora_cfg,
    const std::shared_ptr<TensorAllocator>& allocator,
    const modules::HfMapping* hf_mapping);

}  // namespace internal
}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_MODEL_INTERNAL_H
