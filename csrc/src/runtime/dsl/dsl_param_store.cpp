// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL parameter store implementation.

#include "runtime/dsl/dsl_param_store.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "runtime/dsl/ir.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/tensor_role.h"
#include "runtime/training/runtime_options.h"
#include "runtime/training/model.h"
#include "runtime/lora/lora_config.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

bool is_rope_param(const std::string& name) {
    return tensor_role_is_rope_name(name);
}

bool is_router_param(const std::string& name) {
    return tensor_role_is_router_name(name);
}

void augment_shape_env(ShapeEnv& env, const AttrMap& config) {
    auto get_long = [&](std::string_view key) -> std::optional<long> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::int64_t>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        if (auto v = std::get_if<double>(&it->second.value)) {
            return static_cast<long>(*v);
        }
        return std::nullopt;
    };
    auto get_string = [&](std::string_view key) -> std::optional<std::string> {
        auto it = config.find(std::string(key));
        if (it == config.end()) return std::nullopt;
        if (auto v = std::get_if<std::string>(&it->second.value)) {
            return *v;
        }
        return std::nullopt;
    };

    auto d_model = get_long("d_model");
    if (!d_model) {
        d_model = get_long("hidden_size");
    }
    auto num_q = get_long("num_query_heads");
    if (!num_q) {
        num_q = get_long("num_attention_heads");
    }
    auto num_kv = get_long("num_kv_heads");
    if (!num_kv) {
        num_kv = get_long("num_key_value_heads");
    }
    auto head_size = get_long("head_size");
    if (!head_size) {
        head_size = get_long("head_dim");
    }
    auto d_ff = get_long("d_ff");
    if (!d_ff) {
        d_ff = get_long("intermediate_size");
    }
    auto mlp_activation = get_string("mlp_activation");
    if (!mlp_activation) mlp_activation = get_string("mlp_hidden_act");
    if (!mlp_activation) mlp_activation = get_string("activation");
    int up_factor = 2;
    if (mlp_activation) {
        std::string act = *mlp_activation;
        std::transform(act.begin(), act.end(), act.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (act == "swiglu" || act == "geglu") {
            up_factor = 2;
        } else if (act == "relu" || act == "relu2" || act == "gelu" || act == "gelu_new" || act == "gelu_fast" ||
                   act == "silu" || act == "swish") {
            up_factor = 1;
        }
    }
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }
    auto max_seq = get_long("max_seq");
    if (!max_seq) {
        max_seq = get_long("max_position_embeddings");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
    }
    if (max_seq) {
        env.values.emplace("MaxSeq", *max_seq);
    }
    if (num_q) {
        env.values.emplace("Hq", *num_q);
    }
    if (num_kv) {
        env.values.emplace("Hkv", *num_kv);
    } else if (num_q) {
        env.values.emplace("Hkv", *num_q);
    }
    long Hq = env.values.count("Hq") ? env.values.at("Hq") : 0;
    long Hkv = env.values.count("Hkv") ? env.values.at("Hkv") : 0;
    long C = env.values.count("C") ? env.values.at("C") : 0;
    if (!head_size && Hq > 0 && C > 0) {
        head_size = C / Hq;
    }
    if (head_size) {
        env.values.emplace("D", *head_size);
    }
    if (d_ff) {
        env.values.emplace("M", *d_ff);
        env.values.emplace("MUp", up_factor * (*d_ff));
    }
    if (vocab) {
        env.values.emplace("V", *vocab);
    }
    if (Hq > 0 && head_size) {
        env.values.emplace("AttnDim", Hq * (*head_size));
    }
    if (head_size && Hq > 0 && Hkv > 0) {
        env.values.emplace("QKV", (Hq + 2 * Hkv) * (*head_size));
    }

    // MoE dimensions
    auto num_experts = get_long("num_experts");
    auto num_experts_per_tok = get_long("num_experts_per_tok");
    if (!num_experts_per_tok) num_experts_per_tok = get_long("num_selected_experts");
    auto shared_expert_intermediate = get_long("shared_expert_intermediate");
    if (!shared_expert_intermediate) shared_expert_intermediate = get_long("shared_expert_intermediate_size");

    if (num_experts) {
        env.values.emplace("E", *num_experts);
    }
    if (num_experts_per_tok) {
        env.values.emplace("K", *num_experts_per_tok);
    }
    if (shared_expert_intermediate && *shared_expert_intermediate > 0) {
        env.values.emplace("SharedM", *shared_expert_intermediate);
        env.values.emplace("SharedMUp", up_factor * (*shared_expert_intermediate));
    } else if (d_ff) {
        // Default shared expert size to regular intermediate size if not specified
        env.values.emplace("SharedM", *d_ff);
        env.values.emplace("SharedMUp", up_factor * (*d_ff));
    }
}

}  // namespace

DslParamStore::DslParamStore(const Module& module,
                             const Graph& graph,
                             const RuntimeOptions& options,
                             const PretrainedConfig& config,
                             const std::shared_ptr<TensorAllocator>& allocator,
                             const modules::ModularLoRAConfig* lora_config,
                             const std::unordered_set<std::string>* external_params,
                             bool use_weight_manager)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslParamStore: allocator is null");
    }

    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

    const bool freeze_base = lora_config && lora_config->enabled();
    const bool train_router = freeze_base && lora_config->train_router;

    if (external_params) {
        mExternalParams = *external_params;
    }

    mUsesWeightManager = use_weight_manager;

    for (const auto& kv : graph.params) {
        const std::string& name = kv.first;
        const TensorInfo& info = kv.second;

        if (is_rope_param(name)) {
            // RoPE frequencies are provided by the run state.
            continue;
        }

        ETensorDType dtype = info.dtype.value_or(config.DType);
        std::vector<long> shape = resolve_shape(info.shape, env);

        Entry entry;
        entry.external = mExternalParams.find(name) != mExternalParams.end();
        entry.managed_by_weight_manager = (!entry.external && mUsesWeightManager);
        entry.storage_alias = freeze_base && config.TiedWordEmbeddings && name == "lm_head" &&
            graph.params.contains("embedding") && !entry.external && !mUsesWeightManager;
        if (entry.external || entry.managed_by_weight_manager || entry.storage_alias) {
            entry.tensor = Tensor::empty(dtype, shape);
        } else {
            entry.tensor = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
        }
        entry.trainable = !info.frozen && !is_rope_param(name);
        if (freeze_base) {
            entry.trainable = !info.frozen && train_router && is_router_param(name);
        }

        mParams.emplace(name, entry);
        mParamOrder.push_back(name);
    }

    if (mParams.contains("lm_head") && mParams.at("lm_head").storage_alias) {
        auto& head = mParams.at("lm_head").tensor;
        const auto& embedding = mParams.at("embedding").tensor;
        if (head.nelem() != embedding.nelem() || head.DType != embedding.DType) {
            throw std::runtime_error("tied frozen head and embedding must have matching shapes and dtype");
        }
        head = embedding;
    }

    // Deterministic ordering for optimizer updates/checkpointing.
    std::sort(mParamOrder.begin(), mParamOrder.end());
}

Tensor& DslParamStore::get(const std::string& name) {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        std::cerr << "[ERROR] DslParamStore::get: parameter '" << name << "' not found. Available params: ";
        std::size_t printed = 0;
        for (auto& p : mParams) {
            if (printed++ >= 32) break;
            std::cerr << p.first << ", ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    if (it->second.external) {
        if (!mQLoRAProvider) {
            throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
        }
        return mQLoRAProvider->resolve_param(name, mDefaultStream);
    }
    if (it->second.managed_by_weight_manager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for parameter " + name);
        }
        return mWeightManager->get(name);
    }
    return it->second.tensor;
}

const Tensor& DslParamStore::get(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    if (it->second.external) {
        if (!mQLoRAProvider) {
            throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
        }
        return mQLoRAProvider->resolve_param(name, mDefaultStream);
    }
    if (it->second.managed_by_weight_manager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for parameter " + name);
        }
        return mWeightManager->get(name);
    }
    return it->second.tensor;
}

bool DslParamStore::has(const std::string& name) const {
    return mParams.find(name) != mParams.end();
}

bool DslParamStore::is_trainable(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return false;
    return it->second.trainable;
}

bool DslParamStore::is_external(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return false;
    return it->second.external;
}

bool DslParamStore::work_is_transient(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return false;
    if (!it->second.managed_by_weight_manager || !mWeightManager) return false;
    return mWeightManager->work_is_transient(name);
}

Tensor* DslParamStore::master_tensor(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) return nullptr;
    if (!it->second.managed_by_weight_manager || !mWeightManager || !mWeightManager->has(name)) return nullptr;
    Tensor& master = mWeightManager->get_master(name);
    return master.Data ? &master : nullptr;
}

const Tensor& DslParamStore::template_tensor(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    return it->second.tensor;
}

std::size_t DslParamStore::rebindable_persistent_bytes(const CompiledGraph& graph) const {
    // Layout assigns offsets to every ForwardParam tid in the graph — the
    // Persistent arena has to be sized to the maximum of (offset + bytes)
    // across the tids whose backing storage the arena actually replaces.
    // If ANY param is provider-resolved (QLoRA external) or managed by a
    // weight manager, that set intersperses offsets with the locally-
    // allocated set, so clamping to only the local-set high-water mark
    // isn't sound — the compiler placed local tids at their global offsets,
    // which include the unused external ranges. Whole-or-nothing: return 0
    // as soon as any param is external or weight-manager-managed. Those
    // non-local paths have their own arena-backed storage (handled by
    // `QLoRAWeightProvider` / `DslWeightManager` / the LoRA manager).
    for (const auto& kv : mParams) {
        const Entry& entry = kv.second;
        if (entry.external || entry.managed_by_weight_manager || entry.storage_alias) {
            return 0;
        }
    }
    std::size_t high_water = 0;
    for (const auto& kv : mParams) {
        const Entry& entry = kv.second;
        if (entry.tensor.Data == nullptr && !entry.arena_pending) continue;
        const int tid = graph.find_tensor_id(kv.first);
        if (tid < 0) continue;
        const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
        if (meta.region != RegionKind::Persistent || meta.offset == SIZE_MAX) continue;
        const std::size_t tensor_bytes = entry.tensor.bytes();
        if (tensor_bytes == 0 || meta.bytes < tensor_bytes) continue;
        high_water = std::max(high_water, meta.offset + tensor_bytes);
    }
    return high_water;
}

std::optional<std::size_t> DslParamStore::persistent_arena_offset(const CompiledGraph& graph,
                                                                  const std::string& name,
                                                                  const Entry& entry,
                                                                  std::size_t arena_bytes) const {
    if (entry.external || entry.managed_by_weight_manager) return std::nullopt;
    if (entry.tensor.Data == nullptr && !entry.arena_pending) return std::nullopt;
    const int tid = graph.find_tensor_id(name);
    if (tid < 0) return std::nullopt;
    const auto& meta = graph.tensor_meta[static_cast<std::size_t>(tid)];
    if (meta.region != RegionKind::Persistent || meta.offset == SIZE_MAX) return std::nullopt;
    const std::size_t tensor_bytes = entry.tensor.bytes();
    if (tensor_bytes == 0 || meta.bytes < tensor_bytes || meta.offset + tensor_bytes > arena_bytes) {
        return std::nullopt;
    }
    return meta.offset;
}

std::size_t DslParamStore::release_storage_for_persistent_arena(const CompiledGraph& graph, std::size_t arena_bytes) {
    if (mContentsValid) return 0;  // weights already loaded: the rebind copies them
    if (mParams.contains("lm_head") && mParams.at("lm_head").storage_alias) return 0;
    std::size_t released = 0;
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        Entry& entry = it->second;
        if (entry.arena_pending || !persistent_arena_offset(graph, name, entry, arena_bytes)) continue;
        released += entry.tensor.bytes();
        float* preserved_stats = entry.tensor.Stats;
        const int device = entry.tensor.Device;
        mAllocator->free(entry.tensor);  // clears Data
        entry.tensor.Device = device;
        entry.tensor.Stats = preserved_stats;
        entry.arena_pending = true;
    }
    return released;
}

void DslParamStore::restore_released_storage() {
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end() || !it->second.arena_pending) continue;
        Entry& entry = it->second;
        std::vector<long> shape(entry.tensor.Sizes.begin(), entry.tensor.Sizes.begin() + entry.tensor.Rank);
        float* preserved_stats = entry.tensor.Stats;
        entry.tensor = mAllocator->allocate(entry.tensor.DType, name.c_str(), EAllocationType::ON_DEVICE, shape);
        entry.tensor.Stats = preserved_stats;
        entry.arena_pending = false;
    }
}

void DslParamStore::rebind_to_persistent_arena(const CompiledGraph& graph,
                                               const PhaseArenas& arenas,
                                               cudaStream_t stream) {
    if (mParams.contains("lm_head") && mParams.at("lm_head").storage_alias) return;
    if (!arenas.allocated || arenas.persistent_ptr == nullptr || arenas.persistent_bytes == 0) {
        restore_released_storage();
        return;
    }

    std::size_t rebound = 0;
    std::size_t bound_in_place = 0;
    std::size_t skipped_external = 0;
    std::size_t skipped_managed = 0;
    std::size_t skipped_other = 0;

    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        Entry& entry = it->second;
        if (entry.external) {
            ++skipped_external;
            continue;
        }
        if (entry.managed_by_weight_manager) {
            ++skipped_managed;
            continue;
        }
        const auto offset = persistent_arena_offset(graph, name, entry, arenas.persistent_bytes);
        if (!offset) {
            if (entry.arena_pending) {
                throw std::logic_error("DslParamStore: released parameter " + name + " has no persistent arena slot");
            }
            ++skipped_other;
            continue;
        }

        std::byte* arena_ptr = arenas.persistent_ptr + *offset;
        if (entry.arena_pending) {
            // Released before the arena existed, with nothing written yet: bind in place.
            entry.tensor.Data = arena_ptr;
            entry.arena_pending = false;
            ++bound_in_place;
            continue;
        }
        CUDA_CHECK(
            cudaMemcpyAsync(arena_ptr, entry.tensor.Data, entry.tensor.bytes(), cudaMemcpyDeviceToDevice, stream));

        float* preserved_stats = entry.tensor.Stats;
        const int device = entry.tensor.Device;
        mAllocator->free(entry.tensor);
        entry.tensor.Data = arena_ptr;
        entry.tensor.Device = device;
        entry.tensor.Stats = preserved_stats;
        ++rebound;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (const char* dbg = std::getenv("SUROGATE_DEBUG_ARENA_CONSUME")) {
        if (std::string(dbg) == "1") {
            std::cerr << "[arena-consume persistent] rebound=" << rebound << " bound_in_place=" << bound_in_place
                      << " skipped_external=" << skipped_external << " skipped_managed=" << skipped_managed
                      << " skipped_other=" << skipped_other << " arena_bytes=" << arenas.persistent_bytes << "\n";
        }
    }
}

void DslParamStore::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    mContentsValid = true;  // the callback may write (load_safetensors)
    if (mUsesWeightManager) {
        if (!mWeightManager) {
            throw std::runtime_error("DslParamStore: weight manager not set for iterate_tensors");
        }
        mWeightManager->iterate_tensors(callback);
        return;
    }
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        if (it->second.external) {
            if (!mQLoRAProvider) {
                throw std::runtime_error("DslParamStore: external parameter requested without QLoRA provider: " + name);
            }
            callback(name, TensorShard(mQLoRAProvider->resolve_param(name, mDefaultStream)));
        } else {
            callback(name, TensorShard(it->second.tensor));
        }
    }
}

}  // namespace dsl
