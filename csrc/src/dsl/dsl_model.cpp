// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL model wrapper (IR validation + execution).

#include "dsl/dsl_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <nlohmann/json.hpp>

#include "dsl/graph_executor.h"
#include "dsl/dsl_runtime.h"
#include "kernels/kernels.h"
#include "modules/model_config.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"

namespace dsl {
namespace {

struct HfValue {
    enum class Kind { Int, Float, Bool };
    Kind kind;
    long i = 0;
    double f = 0.0;
    bool b = false;
};

std::optional<HfValue> get_hf_value(const PretrainedConfig& cfg, const std::string& key) {
    if (key == "hidden_size") return HfValue{HfValue::Kind::Int, cfg.HiddenSize, 0.0, false};
    if (key == "intermediate_size") return HfValue{HfValue::Kind::Int, cfg.IntermediateSize, 0.0, false};
    if (key == "vocab_size") return HfValue{HfValue::Kind::Int, cfg.VocabSize, 0.0, false};
    if (key == "num_attention_heads") return HfValue{HfValue::Kind::Int, cfg.NumQueryHeads, 0.0, false};
    if (key == "num_key_value_heads") return HfValue{HfValue::Kind::Int, cfg.NumKeyValHeads, 0.0, false};
    if (key == "num_hidden_layers") return HfValue{HfValue::Kind::Int, cfg.NumLayers, 0.0, false};
    if (key == "max_position_embeddings") return HfValue{HfValue::Kind::Int, cfg.MaxPositionEmbeddings, 0.0, false};
    if (key == "head_dim") return HfValue{HfValue::Kind::Int, cfg.head_size(), 0.0, false};
    if (key == "attention_bias") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.UseQKVBias};
    if (key == "use_qk_norm") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.UseQKNorm};
    if (key == "rms_norm_eps") return HfValue{HfValue::Kind::Float, 0, cfg.RmsNormEps, false};
    if (key == "rope_theta") return HfValue{HfValue::Kind::Float, 0, cfg.RopeTheta, false};
    if (key == "tie_word_embeddings") return HfValue{HfValue::Kind::Bool, 0, 0.0, cfg.TiedWordEmbeddings};
    return std::nullopt;
}

std::optional<HfValue> attr_to_value(const AttrValue& value) {
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return HfValue{HfValue::Kind::Int, static_cast<long>(*v), 0.0, false};
    }
    if (auto v = std::get_if<double>(&value.value)) {
        return HfValue{HfValue::Kind::Float, 0, *v, false};
    }
    if (auto v = std::get_if<bool>(&value.value)) {
        return HfValue{HfValue::Kind::Bool, 0, 0.0, *v};
    }
    return std::nullopt;
}

bool values_match(const HfValue& expected, const HfValue& actual) {
    if (expected.kind == HfValue::Kind::Bool || actual.kind == HfValue::Kind::Bool) {
        if (expected.kind != HfValue::Kind::Bool || actual.kind != HfValue::Kind::Bool) {
            return false;
        }
        return expected.b == actual.b;
    }
    const double lhs = (expected.kind == HfValue::Kind::Int) ? static_cast<double>(expected.i) : expected.f;
    const double rhs = (actual.kind == HfValue::Kind::Int) ? static_cast<double>(actual.i) : actual.f;
    return std::abs(lhs - rhs) <= 1e-6;
}

const AttrMap* as_map(const AttrValue& value) {
    if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&value.value)) {
        if (*map_ptr) return map_ptr->get();
    }
    return nullptr;
}

const AttrList* as_list(const AttrValue& value) {
    if (const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value)) {
        if (*list_ptr) return list_ptr->get();
    }
    return nullptr;
}

std::optional<std::string> as_string(const AttrValue& value) {
    if (const auto* str = std::get_if<std::string>(&value.value)) {
        return *str;
    }
    return std::nullopt;
}

std::optional<long> as_int(const AttrValue& value) {
    if (const auto* i64 = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<long>(*i64);
    }
    if (const auto* f64 = std::get_if<double>(&value.value)) {
        return static_cast<long>(*f64);
    }
    return std::nullopt;
}

std::optional<bool> as_bool(const AttrValue& value) {
    if (const auto* b = std::get_if<bool>(&value.value)) {
        return *b;
    }
    return std::nullopt;
}

const AttrValue* find_key(const AttrMap* map, const std::string& key) {
    if (!map) return nullptr;
    auto it = map->find(key);
    if (it == map->end()) return nullptr;
    return &it->second;
}

bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    if (prefix.find("blocks[") == 0) {
        auto close = prefix.find(']');
        if (close == std::string_view::npos) return false;
        auto idx_str = prefix.substr(7, close - 7);
        try {
            layer_idx = std::stoi(std::string(idx_str));
        } catch (...) {
            return false;
        }
        param_name = std::string(rest);
        return true;
    }

    if (prefix == "blocks") {
        auto idx_str = name.substr(dot + 1);
        auto dot2 = idx_str.find('.');
        if (dot2 == std::string_view::npos) return false;
        try {
            layer_idx = std::stoi(std::string(idx_str.substr(0, dot2)));
        } catch (...) {
            return false;
        }
        param_name = std::string(idx_str.substr(dot2 + 1));
        return true;
    }

    return false;
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool contains_ci(const std::string& haystack, const std::string& needle) {
    auto h = to_lower(haystack);
    auto n = to_lower(needle);
    return h.find(n) != std::string::npos;
}

void replace_all(std::string& str, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::size_t start = 0;
    while ((start = str.find(from, start)) != std::string::npos) {
        str.replace(start, from.size(), to);
        start += to.size();
    }
}

std::string format_hf_name(std::string templ, int layer_idx, int expert_idx = -1) {
    if (templ.find("{layer}") != std::string::npos) {
        if (layer_idx < 0) {
            throw std::runtime_error("DSL model: HF mapping uses {layer} but no layer index available");
        }
        replace_all(templ, "{layer}", std::to_string(layer_idx));
    }
    if (templ.find("{expert}") != std::string::npos) {
        if (expert_idx < 0) {
            throw std::runtime_error("DSL model: HF mapping uses {expert} but no expert index available");
        }
        replace_all(templ, "{expert}", std::to_string(expert_idx));
    }
    return templ;
}

DslModel::MappingSpec parse_mapping_spec(const AttrValue& value) {
    DslModel::MappingSpec spec;

    if (auto direct = as_string(value)) {
        spec.kind = DslModel::MappingSpec::Kind::Direct;
        spec.source = *direct;
        return spec;
    }

    const AttrMap* map = as_map(value);
    if (!map) {
        return spec;
    }

    if (const auto* opt_val = find_key(map, "optional")) {
        if (auto opt = as_bool(*opt_val)) {
            spec.optional = *opt;
        }
    }

    std::string type;
    if (const auto* type_val = find_key(map, "type")) {
        if (auto t = as_string(*type_val)) {
            type = *t;
        }
    }

    auto get_source = [&]() -> std::string {
        if (const auto* src_val = find_key(map, "source")) {
            if (auto src = as_string(*src_val)) {
                return *src;
            }
        }
        if (const auto* path_val = find_key(map, "path")) {
            if (auto path = as_string(*path_val)) {
                return *path;
            }
        }
        return {};
    };

    if (type.empty() || type == "direct") {
        spec.kind = DslModel::MappingSpec::Kind::Direct;
        spec.source = get_source();
        return spec;
    }

    if (type == "fuse") {
        spec.kind = DslModel::MappingSpec::Kind::Fuse;
        if (const auto* dim_val = find_key(map, "dim")) {
            if (auto dim = as_int(*dim_val)) {
                spec.dim = static_cast<int>(*dim);
            }
        }
        if (const auto* list_val = find_key(map, "sources")) {
            if (const auto* list = as_list(*list_val)) {
                for (const auto& item : *list) {
                    if (auto src = as_string(item)) {
                        spec.sources.push_back(*src);
                    }
                }
            }
        }
        return spec;
    }

    if (type == "split") {
        spec.kind = DslModel::MappingSpec::Kind::Split;
        spec.source = get_source();
        if (const auto* dim_val = find_key(map, "dim")) {
            if (auto dim = as_int(*dim_val)) {
                spec.dim = static_cast<int>(*dim);
            }
        }
        if (const auto* list_val = find_key(map, "ranges")) {
            if (const auto* list = as_list(*list_val)) {
                for (const auto& item : *list) {
                    if (const auto* pair_list = as_list(item)) {
                        if (pair_list->size() >= 2) {
                            auto start = as_int(pair_list->at(0));
                            auto end = as_int(pair_list->at(1));
                            if (start && end) {
                                spec.ranges.emplace_back(*start, *end);
                            }
                        }
                    }
                }
            }
        }
        return spec;
    }

    if (type == "transform") {
        spec.kind = DslModel::MappingSpec::Kind::Transform;
        spec.source = get_source();
        if (const auto* fn_val = find_key(map, "fn")) {
            if (auto fn = as_string(*fn_val)) {
                spec.fn = *fn;
            }
        }
        return spec;
    }

    if (type == "tied_to") {
        spec.kind = DslModel::MappingSpec::Kind::TiedTo;
        if (const auto* tgt_val = find_key(map, "target")) {
            if (auto tgt = as_string(*tgt_val)) {
                spec.target = *tgt;
            }
        }
        return spec;
    }

    return spec;
}

const DslModel::MappingSpec* find_mapping_spec(
    const std::unordered_map<std::string, DslModel::MappingSpec>& mapping,
    const std::string& internal_name,
    int& layer_idx) {
    layer_idx = -1;
    auto it = mapping.find(internal_name);
    if (it != mapping.end()) {
        return &it->second;
    }

    std::string base;
    if (parse_block_param(internal_name, layer_idx, base)) {
        std::string placeholder = std::string("blocks[{layer}].") + base;
        it = mapping.find(placeholder);
        if (it != mapping.end()) {
            return &it->second;
        }
        it = mapping.find(base);
        if (it != mapping.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

Tensor slice_dim0(const Tensor& base, long offset, long length) {
    Tensor slice = base;
    if (slice.Rank < 1) {
        throw std::runtime_error("DSL model: cannot slice rank-0 tensor");
    }
    long stride = 1;
    for (int i = 1; i < slice.Rank; ++i) {
        stride *= slice.Sizes[i];
    }
    const std::size_t elem_size = get_dtype_size(slice.DType);
    const std::size_t byte_offset = static_cast<std::size_t>(offset) * static_cast<std::size_t>(stride) * elem_size;
    slice.Data = static_cast<std::byte*>(slice.Data) + byte_offset;
    slice.Sizes[0] = length;
    return slice;
}

bool is_norm_param_name(const std::string& name) {
    auto lower = to_lower(name);
    return lower.find("norm") != std::string::npos || lower.find("ln1") != std::string::npos || lower.find("ln2") != std::string::npos;
}

bool is_bias_param_name(const std::string& name) {
    return contains_ci(name, "bias");
}

std::vector<long> infer_fuse_slices(const std::string& name, const PretrainedConfig& cfg, int num_sources) {
    if (contains_ci(name, "qkv")) {
        const long hs = cfg.head_size();
        const long q_rows = static_cast<long>(cfg.NumQueryHeads) * hs;
        const long kv_rows = static_cast<long>(cfg.NumKeyValHeads) * hs;
        return {q_rows, kv_rows, kv_rows};
    }
    if (contains_ci(name, "mlp_up") || contains_ci(name, "gate_up")) {
        const long m = cfg.IntermediateSize;
        return std::vector<long>(num_sources, m);
    }
    return {};
}

} // namespace

DslModel::DslModel(const PretrainedConfig& config,
                   const RuntimeOptions& options,
                   const std::string& ir_json,
                   const std::shared_ptr<TensorAllocator>& allocator)
    : mConfig(config.clone()),
      mAllocator(allocator ? allocator : std::make_shared<TensorAllocator>()),
      mOptions(options),
      mModelConfig(modules::ModelConfig::from_pretrained_config(config)) {
    if (ir_json.empty()) {
        throw std::runtime_error("DSL model: IR JSON is empty");
    }
    nlohmann::json root = nlohmann::json::parse(ir_json);
    mIr = load_ir_from_json(root);
    if (!mIr.success) {
        throw std::runtime_error("DSL model: IR JSON indicates compilation failure");
    }
    mModule = &pick_model_module(mIr);
    validate_ir();

    if (!mModule->forward.has_value()) {
        throw std::runtime_error("DSL model: module missing forward graph");
    }

    mParams = std::make_unique<DslParamStore>(*mModule, mModule->forward.value(),
                                              options, *mConfig, mAllocator);
    mGrads = std::make_unique<DslGradStore>(*mParams, mAllocator);

    for (const auto& kv : mModule->hf_mapping) {
        mHfMapping.emplace(kv.first, parse_mapping_spec(kv.second));
    }
    for (const auto& kv : mModule->hf_export) {
        mHfExport.emplace(kv.first, parse_mapping_spec(kv.second));
    }
}

DslModel::~DslModel() = default;

void DslModel::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::forward called before allocate_run_state()");
    }
    mExecutor->forward(inputs, position_ids, comm, micro_step);
}

float DslModel::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::validate called before allocate_run_state()");
    }
    return mExecutor->validate(inputs, position_ids, targets, comm, micro_step);
}

void DslModel::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    if (!mExecutor) {
        throw std::logic_error("DslModel::backward called before allocate_run_state()");
    }
    mExecutor->backward(inputs, targets, comm, grad_accum_steps, micro_step);
}

void DslModel::update(NCCLCommunicator& comm, float learning_rate, float beta_1, float beta_2, int t, float epsilon,
                      float weight_decay, float grad_clip) {
    if (!mRunState || !mParams || !mGrads) {
        throw std::logic_error("DslModel::update called before allocate_run_state()");
    }
    if (!mAdamW8BitState) {
        throw std::logic_error("DslModel::update: optimizer state not allocated");
    }

    auto& rs = *mRunState;
    cudaStream_t stream = rs.MainStream;
    CUDA_CHECK(cudaStreamWaitEvent(stream, rs.BackwardDone, 0));

    const bool grads_reduced = comm.world_size() > 1;
    if (grads_reduced) {
        mGrads->reduce_all(comm, stream);
    }

    calculate_gradient_norm(comm, grad_clip, stream, grads_reduced);
    const float* grad_scale = rs.scratch().norm_buffer.template get<float>() + 1;

    if (std::getenv("SUROGATE_DEBUG_NAN")) {
        Tensor tmp = rs.temp_alloc(ETensorDType::FP32, {static_cast<long>(rs.scratch().norm_buffer.nelem())});
        for (const auto& name : mGrads->param_names()) {
            bool accumulate = false;
            Tensor* grad = mGrads->get_param_grad(name, accumulate);
            if (!grad || !grad->Data || grad->nelem() == 0) {
                continue;
            }
            fill_zero(tmp, stream);
            global_norm_squared(tmp, *grad, grad->nelem(), rs.DeviceProp, stream);
            deterministic_sum(tmp.template get<float>(), tmp.template get<float>(), tmp.nelem(), stream);
            CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, tmp.Data, sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float gnorm = *rs.NormHost;
            if (!std::isfinite(gnorm)) {
                fprintf(stderr, "[DSL DEBUG] Non-finite grad norm for %s (norm=%f)\n",
                        name.c_str(), gnorm);
                break;
            }
        }
        rs.temp_free(tmp);
    }

    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }

    auto& state = *mAdamW8BitState;
    constexpr size_t BLOCK_SIZE = 2048;  // Must match ADAMW8BIT_BLOCK_SIZE in adamw8bit.cu
    size_t state_offset = 0;

    for (const auto& name : mGrads->param_names()) {
        Tensor& val = mParams->get(name);
        bool accumulate = false;
        Tensor* grad = mGrads->get_param_grad(name, accumulate);
        (void)accumulate;
        if (!grad) {
            continue;
        }

        float wd = weight_decay;
        if (is_norm_param_name(name) || is_bias_param_name(name)) {
            wd = 0.f;
        }

        state_offset = (state_offset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        const size_t n = val.nelem();
        if (!val.Data || !grad->Data) {
            state_offset += n;
            continue;
        }
        const size_t block_offset = state_offset / BLOCK_SIZE;

        unsigned char* s1 = reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()) + state_offset;
        unsigned char* s2 = reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()) + state_offset;
        float* am1 = state.absmax1.template get<float>() + block_offset;
        float* am2 = state.absmax2.template get<float>() + block_offset;
        float* q1 = state.quantiles1.template get<float>();
        float* q2 = state.quantiles2.template get<float>();

        if (val.DType == ETensorDType::FP32) {
            if (grad->DType == ETensorDType::FP32) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<float>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, stream
                );
            } else if (grad->DType == ETensorDType::BF16) {
                adamw_update_8bit(
                    val.template get<float>(),
                    grad->template get<nv_bfloat16>(),
                    s1, s2, n,
                    learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                    q1, q2, am1, am2, stream
                );
            } else {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
        } else if (val.DType == ETensorDType::BF16) {
            if (grad->DType != ETensorDType::BF16) {
                throw std::runtime_error("DslModel::update: unsupported grad dtype for " + name);
            }
            adamw_update_8bit(
                val.template get<nv_bfloat16>(),
                grad->template get<nv_bfloat16>(),
                s1, s2, n,
                learning_rate, beta_1, beta_2, t, epsilon, wd, grad_scale,
                q1, q2, am1, am2, stream
            );
        } else {
            throw std::runtime_error("DslModel::update: unsupported param dtype for " + name);
        }

        state_offset += n;
        if (state_offset > state.total_state_elems) {
            throw std::runtime_error("DslModel::update: state buffer overflow");
        }
    }

    CUDA_CHECK(cudaEventRecord(rs.OptimizerDone, stream));
}

void DslModel::update_with_config(NCCLCommunicator& comm, const optimizers::OptimizerConfig& config, int step) {
    switch (config.type) {
        case optimizers::OptimizerType::ADAMW_8BIT:
            update(comm, config.learning_rate, config.adamw_beta1, config.adamw_beta2,
                   step, config.adamw_epsilon, config.weight_decay, config.grad_clip);
            break;
        default:
            throw std::logic_error("DslModel::update_with_config: unsupported optimizer type");
    }
}

ITensorContainer& DslModel::weights() {
    return mParams ? static_cast<ITensorContainer&>(*mParams) : mEmpty;
}

std::vector<std::byte> DslModel::rng_state() const {
    if (mExecutor) {
        return mExecutor->rng_state();
    }
    return mRngState;
}

void DslModel::set_rng_state(const std::vector<std::byte>& state) {
    mRngState = state;
    if (mExecutor) {
        mExecutor->set_rng_state(state);
    }
}

void DslModel::init_weights(NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::init_weights called before parameters are initialized");
    }

    const float scale = 0.02f;
    const float residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(mConfig->NumLayers));
    unsigned long long seed = 42ULL;
    unsigned long long subseq = 0ULL;

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        if (is_bias_param_name(name)) {
            fill_zero(param, nullptr);
            continue;
        }
        if (is_norm_param_name(name)) {
            fill_constant(param, 1.f, param.nelem(), nullptr);
            continue;
        }
        float stddev = scale;
        if (contains_ci(name, "out_weight") || contains_ci(name, "mlp_down_weight") || contains_ci(name, "down_proj")) {
            stddev *= residual_scale;
        }
        fill_normal(param, param.nelem(), 0.f, stddev, seed, subseq++, nullptr);
    }

    comm.barrier();
}

void DslModel::import_weights(const std::string& file_name, bool allow_cast, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::import_weights called before parameters are initialized");
    }

    SafeTensorsReader reader(file_name);
    std::vector<std::pair<std::string, std::string>> tied_params;

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        int layer_idx = -1;
        const MappingSpec* spec = find_mapping_spec(mHfMapping, name, layer_idx);
        MappingSpec direct_fallback;
        if (!spec) {
            direct_fallback.kind = MappingSpec::Kind::Direct;
            direct_fallback.source = name;
            spec = &direct_fallback;
        }

        if (spec->kind == MappingSpec::Kind::TiedTo) {
            tied_params.emplace_back(name, spec->target);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            entry.read_tensor(param, allow_cast);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse mapping only supports dim=0 for " + name);
            }
            long offset = 0;
            for (const auto& src : spec->sources) {
                const std::string hf_name = format_hf_name(src, layer_idx);
                const auto& entry = reader.find_entry(hf_name);
                if (entry.shape().empty()) {
                    throw std::runtime_error("DSL model: empty shape for " + hf_name);
                }
                if (static_cast<int>(entry.shape().size()) != param.Rank) {
                    throw std::runtime_error("DSL model: rank mismatch for " + hf_name);
                }
                for (int i = 1; i < param.Rank; ++i) {
                    if (entry.shape().at(i) != param.Sizes[i]) {
                        throw std::runtime_error("DSL model: shape mismatch for " + hf_name);
                    }
                }
                const long slice_len = entry.shape().at(0);
                Tensor slice = slice_dim0(param, offset, slice_len);
                entry.read_raw(slice, 0, slice.nelem(), allow_cast);
                offset += slice_len;
            }
            if (offset != param.Sizes[0]) {
                throw std::runtime_error("DSL model: fuse mapping size mismatch for " + name);
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Split) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: split mapping only supports dim=0 for " + name);
            }
            if (spec->ranges.empty()) {
                throw std::runtime_error("DSL model: split mapping missing ranges for " + name);
            }
            auto [start, end] = spec->ranges.front();
            if (start < 0 || end <= start) {
                throw std::runtime_error("DSL model: unsupported split range for " + name);
            }
            const long expected = end - start;
            if (param.Sizes[0] != expected) {
                throw std::runtime_error("DSL model: split range size mismatch for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            long stride = 1;
            for (int i = 1; i < param.Rank; ++i) {
                stride *= param.Sizes[i];
            }
            const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(start) * stride;
            entry.read_raw(param, offset, param.nelem(), allow_cast);
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported transform '" + spec->fn + "' for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            const auto& entry = reader.find_entry(hf_name);
            if (entry.shape().size() != 2 || param.Rank != 2) {
                throw std::runtime_error("DSL model: transpose expects 2D tensors for " + name);
            }
            Tensor tmp = mAllocator->allocate(param.DType, ("hf_tmp_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {entry.shape().at(0), entry.shape().at(1)});
            entry.read_tensor(tmp, allow_cast);
            cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
            transpose(param, tmp, static_cast<int>(entry.shape().at(0)),
                      static_cast<int>(entry.shape().at(1)), stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF mapping for " + name);
    }

    for (const auto& tie : tied_params) {
        Tensor& dst = mParams->get(tie.first);
        Tensor& src = mParams->get(tie.second);
        CUDA_CHECK(cudaMemcpy(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToDevice));
    }

    comm.barrier();
}

void DslModel::on_restore_checkpoint(NCCLCommunicator& comm) {
    (void)comm;
    if (mAdamW8BitState && mAdamW8BitState->state1.Data) {
        mAdamW8BitState->initialized = true;
        mAdamWMomentumContainer.update_pointers(&mAdamW8BitState->state1, &mAdamW8BitState->absmax1);
        mAdamWVarianceContainer.update_pointers(&mAdamW8BitState->state2, &mAdamW8BitState->absmax2);
    }
}

void DslModel::prepare_optimizer_for_checkpoint_load() {
    if (!mAdamW8BitState) {
        mAdamW8BitState = std::make_unique<AdamW8BitState>();
    }
    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    if (!mAdamW8BitState->initialized) {
        init_optimizer_state(stream);
    }
}

void DslModel::export_weights(const std::string& file_name, NCCLCommunicator& comm) {
    if (!mParams) {
        throw std::logic_error("DslModel::export_weights called before parameters are initialized");
    }

    const auto& mapping = !mHfExport.empty() ? mHfExport : mHfMapping;
    SafeTensorWriter writer(file_name);

    struct ExportEntry {
        std::string name;
        Tensor tensor;
        bool needs_transpose = false;
        Tensor source;
    };

    std::vector<ExportEntry> exports;
    exports.reserve(mParams->param_names().size());

    for (const auto& name : mParams->param_names()) {
        Tensor& param = mParams->get(name);
        int layer_idx = -1;
        const MappingSpec* spec = find_mapping_spec(mapping, name, layer_idx);
        if (!spec) {
            MappingSpec fallback;
            fallback.kind = MappingSpec::Kind::Direct;
            fallback.source = name;
            spec = &fallback;
        }

        if (spec->kind == MappingSpec::Kind::Direct) {
            const std::string hf_name = format_hf_name(
                spec->source.empty() ? name : spec->source, layer_idx);
            exports.push_back({hf_name, param, false, {}});
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Fuse) {
            if (spec->dim != 0) {
                throw std::runtime_error("DSL model: fuse export only supports dim=0 for " + name);
            }
            std::vector<long> slice_sizes = infer_fuse_slices(name, *mConfig, static_cast<int>(spec->sources.size()));
            if (slice_sizes.empty()) {
                if (param.Sizes[0] % static_cast<long>(spec->sources.size()) == 0) {
                    const long chunk = param.Sizes[0] / static_cast<long>(spec->sources.size());
                    slice_sizes.assign(spec->sources.size(), chunk);
                } else {
                    throw std::runtime_error("DSL model: cannot infer fuse slices for " + name);
                }
            } else if (slice_sizes.size() != spec->sources.size()) {
                throw std::runtime_error("DSL model: fuse slice count mismatch for " + name);
            }
            long offset = 0;
            for (std::size_t i = 0; i < spec->sources.size(); ++i) {
                const auto& src = spec->sources[i];
                const std::string hf_name = format_hf_name(src, layer_idx);
                const long slice_len = slice_sizes.at(i);
                if (slice_len <= 0) {
                    throw std::runtime_error("DSL model: invalid fuse slice for " + name);
                }
                Tensor slice = slice_dim0(param, offset, slice_len);
                exports.push_back({hf_name, slice, false, {}});
                offset += slice_len;
            }
            if (offset != param.Sizes[0]) {
                throw std::runtime_error("DSL model: fuse slices do not cover full tensor for " + name);
            }
            continue;
        }

        if (spec->kind == MappingSpec::Kind::Transform) {
            if (spec->fn != "transpose") {
                throw std::runtime_error("DSL model: unsupported export transform '" + spec->fn + "' for " + name);
            }
            if (param.Rank != 2) {
                throw std::runtime_error("DSL model: transpose export expects 2D tensor for " + name);
            }
            const std::string hf_name = format_hf_name(spec->source, layer_idx);
            Tensor tmp = mAllocator->allocate(param.DType, ("export_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {param.Sizes[1], param.Sizes[0]});
            exports.push_back({hf_name, tmp, true, param});
            continue;
        }

        throw std::runtime_error("DSL model: unsupported HF export mapping for " + name);
    }

    for (const auto& entry : exports) {
        writer.register_tensor(entry.name, TensorShard(entry.tensor));
    }
    writer.prepare_metadata(&comm);

    cudaStream_t stream = mRunState ? mRunState->MainStream : cudaStreamDefault;
    for (auto& entry : exports) {
        if (entry.needs_transpose) {
            transpose(entry.tensor, entry.source,
                      static_cast<int>(entry.source.Sizes[0]),
                      static_cast<int>(entry.source.Sizes[1]),
                      stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        writer.write_tensor(entry.name, TensorShard(entry.tensor), &comm);
    }

    writer.finalize(&comm);
}

float DslModel::get_loss() const {
    if (!mRunState) {
        return 0.0f;
    }
    float raw_loss = mRunState->get_loss();
    int valid_tokens = 0;
    CUDA_CHECK(cudaMemcpy(&valid_tokens, mRunState->ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost));
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, mRunState->WorldSize));
        return raw_loss / avg_valid;
    }
    return 0.0f;
}

float DslModel::get_accuracy() const {
    return IModel::get_accuracy();
}

void DslModel::allocate_run_state(const RuntimeOptions& options, NCCLCommunicator& comm, int B, int T,
                                  bool allocate_optimizer) {
    if (!mAllocator) {
        mAllocator = std::make_shared<TensorAllocator>();
    }
    mOptions = options;
    mRunState = std::make_unique<DslRunState>(*mConfig, options, B, T, mAllocator);
    mRunState->WorldSize = comm.world_size();

    GraphExecutorOptions exec_opts;
    exec_opts.auto_backward = true;
    exec_opts.debug_print_backward = false;
    mExecutor = std::make_unique<GraphExecutor>(*mModule, *mRunState, *mParams, *mGrads, mModelConfig, options, exec_opts);
    if (!mRngState.empty()) {
        mExecutor->set_rng_state(mRngState);
    }

    if (allocate_optimizer) {
        if (!mAdamW8BitState) {
            mAdamW8BitState = std::make_unique<AdamW8BitState>();
        }
        if (!mAdamW8BitState->quantiles1.Data) {
            mAdamW8BitState->quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
            mAdamW8BitState->quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
            std::vector<float> h_q1(256), h_q2(256);
            create_adamw8bit_quantiles1(h_q1.data());
            create_adamw8bit_quantiles2(h_q2.data());
            CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mAdamW8BitState->quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

std::string_view DslModel::model_type() const {
    return mConfig ? mConfig->model_name() : "DSL";
}

IRunState& DslModel::get_run_state() const {
    if (!mRunState) {
        throw std::logic_error("DslModel::get_run_state() called before allocate_run_state()");
    }
    return *mRunState;
}

void DslModel::init_optimizer_state(cudaStream_t stream) {
    if (!mAdamW8BitState) {
        throw std::runtime_error("DslModel::init_optimizer_state: optimizer state not allocated");
    }
    auto& state = *mAdamW8BitState;
    if (state.initialized) {
        return;
    }

    if (!state.quantiles1.Data) {
        state.quantiles1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles1", {256});
        state.quantiles2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_quantiles2", {256});
        std::vector<float> h_q1(256), h_q2(256);
        create_adamw8bit_quantiles1(h_q1.data());
        create_adamw8bit_quantiles2(h_q2.data());
        CUDA_CHECK(cudaMemcpy(state.quantiles1.Data, h_q1.data(), h_q1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.quantiles2.Data, h_q2.data(), h_q2.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    constexpr size_t BLOCK_SIZE = 2048;
    size_t total_params = 0;
    size_t state_elems = 0;
    auto add_tensor = [&](size_t n) {
        total_params += n;
        state_elems = (state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        state_elems += n;
    };

    for (const auto& name : mGrads->param_names()) {
        Tensor& param = mParams->get(name);
        add_tensor(param.nelem());
    }

    state.total_params = total_params;
    state.total_state_elems = state_elems;
    state.num_blocks = (state.total_state_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;

    state.state1 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state1", {(long)state.total_state_elems});
    state.state2 = mAllocator->allocate(ETensorDType::BYTE, "adamw8bit_state2", {(long)state.total_state_elems});
    state.absmax1 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax1", {(long)state.num_blocks});
    state.absmax2 = mAllocator->allocate(ETensorDType::FP32, "adamw8bit_absmax2", {(long)state.num_blocks});

    init_adamw8bit_state(reinterpret_cast<unsigned char*>(state.state1.template get<std::byte>()),
                         reinterpret_cast<unsigned char*>(state.state2.template get<std::byte>()),
                         state.absmax1.template get<float>(),
                         state.absmax2.template get<float>(),
                         state.total_state_elems, stream);

    state.initialized = true;
    mAdamWMomentumContainer.update_pointers(&state.state1, &state.absmax1);
    mAdamWVarianceContainer.update_pointers(&state.state2, &state.absmax2);
}

void DslModel::calculate_gradient_norm(NCCLCommunicator& comm, float grad_clip, cudaStream_t stream, bool grads_reduced) {
    auto& rs = *mRunState;

    fill_zero(rs.scratch().norm_buffer, stream);
    for (const auto& kv : mGrads->grads()) {
        const Tensor& grad = kv.second;
        if (!grad.Data || grad.nelem() == 0) continue;
        global_norm_squared(rs.scratch().norm_buffer, grad, grad.nelem(), rs.DeviceProp, stream);
    }

    deterministic_sum(rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.template get<float>(),
                      rs.scratch().norm_buffer.nelem(),
                      stream);

    if (!grads_reduced && comm.world_size() > 1) {
        comm.reduce_norm(rs.scratch().norm_buffer.template get<float>(), stream);
    }

    float total_tokens = static_cast<float>(rs.B) * static_cast<float>(rs.T)
                       * static_cast<float>(std::max(1, rs.GradAccumSteps))
                       * static_cast<float>(std::max(1, comm.world_size()));
    global_norm_sqrt(rs.scratch().norm_buffer.template get<float>(), rs.NormHost, grad_clip,
                     rs.ValidTokenCount.template get<int>(), total_tokens,
                     rs.DeviceProp, stream);
    CUDA_CHECK(cudaEventRecord(rs.NormDone, stream));
}

void DslModel::validate_ir() {
    if (!mModule) {
        throw std::runtime_error("DSL model: no module selected");
    }
    validate_config_mapping(*mModule);
    validate_param_shapes(*mModule);
}

const Module& DslModel::pick_model_module(const IRFile& ir) const {
    const Module* candidate = nullptr;
    for (const auto& mod : ir.modules) {
        if (mod.kind != "model") {
            continue;
        }
        if (candidate) {
            throw std::runtime_error("DSL model: multiple model modules in IR");
        }
        candidate = &mod;
    }
    if (!candidate) {
        throw std::runtime_error("DSL model: no model module in IR");
    }
    return *candidate;
}

void DslModel::validate_config_mapping(const Module& module) const {
    const AttrMap* mapping = nullptr;
    auto it = module.hf_config.find("param_mapping");
    if (it != module.hf_config.end()) {
        if (const auto* map_ptr = std::get_if<AttrValue::MapPtr>(&it->second.value)) {
            if (*map_ptr) {
                mapping = map_ptr->get();
            }
        }
    }
    if (!mapping) {
        mapping = &module.hf_config;
    }

    for (const auto& kv : *mapping) {
        const auto* hf_key = std::get_if<std::string>(&kv.second.value);
        if (!hf_key) {
            continue;
        }
        auto expected = get_hf_value(*mConfig, *hf_key);
        if (!expected) {
            continue;
        }
        auto it = module.config.find(kv.first);
        if (it == module.config.end()) {
            throw std::runtime_error("DSL model: missing module config param " + kv.first);
        }
        auto actual = attr_to_value(it->second);
        if (!actual || !values_match(*expected, *actual)) {
            throw std::runtime_error("DSL model: config mismatch for param " + kv.first);
        }
    }
}

void DslModel::validate_param_shapes(const Module& module) const {
    auto env = make_shape_env(module, /*B=*/1, /*T=*/1);
    for (const auto& kv : module.params) {
        const auto& info = kv.second;
        if (info.shape.empty()) {
            continue;
        }
        auto resolved = resolve_shape(info.shape, env);
        for (const auto dim : resolved) {
            if (dim <= 0) {
                throw std::runtime_error("DSL model: invalid shape for param " + kv.first);
            }
        }
    }
}

} // namespace dsl
