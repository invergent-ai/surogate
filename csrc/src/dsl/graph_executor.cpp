// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (Qwen3-first).

#include "dsl/graph_executor.h"
#include "dsl/autodiff.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "modules/model/modular_model.h"
#include "modules/composite/transformer_block_impl.h"
#include "modules/gradient_manager.h"
#include "modules/lora/lora_model_core.h"
#include "modules/lora/lora_model_state_management.h"
#include "modules/lora/lora_model_utils.h"
#include "modules/lora/lora_utils.h"
#include "modules/model/modular_model_fwd.h"
#include "modules/run_state.h"
#include "modules/weights/weight_manager.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"

namespace dsl {
namespace {

using Block = modules::Qwen3TransformerBlock;
using Model = modules::ModularTransformerModel<Block>;
using RunState = modules::ModularRunState<Block>;
using WeightManager = modules::ModularWeightManager<Block>;
using GradManager = modules::ModularGradientManager<Block>;

constexpr std::string_view kSavedPrefix = "saved.";

struct ExecState {
    Model& model;
    RunState& rs;
    WeightManager& weights;
    GradManager& grads;
    const modules::ModelConfig& config;
    const modules::ModelOptions& options;
    recipes::Recipe& recipe;
    NCCLCommunicator& comm;
    long B = 0;
    long T = 0;
    ShapeEnv shape_env{};

    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_set<std::string> zero_tensors;
    std::vector<Tensor> temps;
    bool lm_head_accumulate = false;
};

template<typename Gradients>
Tensor* get_ln1_weight_grad(Gradients& grads) {
    if constexpr (requires(Gradients g) { g.ln1_grads.d_weight; }) {
        return &grads.ln1_grads.d_weight;
    } else if constexpr (requires(Gradients g) { g.ln1.d_weight; }) {
        return &grads.ln1.d_weight;
    } else {
        return nullptr;
    }
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

const AttrValue* find_attr(const AttrMap& attrs, std::string_view key) {
    auto it = attrs.find(std::string(key));
    if (it == attrs.end()) {
        return nullptr;
    }
    return &it->second;
}

std::optional<std::string> attr_string(const AttrValue& value) {
    if (auto v = std::get_if<std::string>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

[[maybe_unused]] std::optional<long> attr_int(const AttrValue& value) {
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<long>(*v);
    }
    if (auto v = std::get_if<double>(&value.value)) {
        return static_cast<long>(*v);
    }
    return std::nullopt;
}

std::optional<double> attr_double(const AttrValue& value) {
    if (auto v = std::get_if<double>(&value.value)) {
        return *v;
    }
    if (auto v = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*v);
    }
    return std::nullopt;
}

[[maybe_unused]] std::optional<bool> attr_bool(const AttrValue& value) {
    if (auto v = std::get_if<bool>(&value.value)) {
        return *v;
    }
    return std::nullopt;
}

std::vector<long> resolve_attr_shape(const AttrValue& value, const ShapeEnv& env) {
    const auto* list_ptr = std::get_if<AttrValue::ListPtr>(&value.value);
    if (!list_ptr || !*list_ptr) {
        throw std::runtime_error("DSL graph executor: shape attr is not a list");
    }
    std::vector<long> shape;
    shape.reserve((*list_ptr)->size());
    for (const auto& item : **list_ptr) {
        if (auto v = std::get_if<std::int64_t>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<double>(&item.value)) {
            shape.push_back(static_cast<long>(*v));
            continue;
        }
        if (auto v = std::get_if<std::string>(&item.value)) {
            shape.push_back(resolve_dim(Dim::computed(*v), env));
            continue;
        }
        throw std::runtime_error("DSL graph executor: unsupported shape attr item");
    }
    return shape;
}

[[maybe_unused]] ETensorDType resolve_dtype(const TensorInfo& info, ETensorDType fallback) {
    if (info.dtype.has_value()) {
        return info.dtype.value();
    }
    return fallback;
}

Tensor view_tensor(const Tensor& src, const std::vector<long>& shape) {
    if (shape.size() > MAX_TENSOR_DIM) {
        throw std::runtime_error("DSL graph executor: view rank too large");
    }
    Tensor out = src;
    out.Rank = static_cast<int>(shape.size());
    for (int i = 0; i < out.Rank; ++i) {
        out.Sizes[i] = shape[i];
    }
    for (int i = out.Rank; i < MAX_TENSOR_DIM; ++i) {
        out.Sizes[i] = 1;
    }
    return out;
}

Tensor& ensure_tensor(ExecState& st, const std::string& name, ETensorDType dtype, const std::vector<long>& shape) {
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return it->second;
    }
    Tensor t = st.rs.temp_alloc(dtype, shape);
    st.temps.push_back(t);
    auto [ins_it, inserted] = st.tensors.emplace(name, t);
    (void)inserted;
    return ins_it->second;
}

Tensor& get_tensor(ExecState& st, const std::string& name, const std::unordered_map<std::string, Tensor>& saved) {
    // Check for explicit "saved." prefix first
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        if (it == saved.end()) {
            throw std::runtime_error("DSL graph executor: missing saved tensor " + key);
        }
        return const_cast<Tensor&>(it->second);
    }
    // Check current tensors (params, intermediates, gradients)
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return it->second;
    }
    // Fall back to saved tensors (for forward intermediates used in backward)
    auto sit = saved.find(name);
    if (sit != saved.end()) {
        return const_cast<Tensor&>(sit->second);
    }
    throw std::runtime_error("DSL graph executor: unknown tensor " + name);
}

// Try to get a tensor by name, returning nullptr if not found (no throw)
Tensor* try_get_tensor(ExecState& st, const std::string& name, std::unordered_map<std::string, Tensor>& saved) {
    if (starts_with(name, kSavedPrefix)) {
        std::string key = std::string(name.substr(kSavedPrefix.size()));
        auto it = saved.find(key);
        return it != saved.end() ? &it->second : nullptr;
    }
    auto it = st.tensors.find(name);
    if (it != st.tensors.end()) {
        return &it->second;
    }
    auto sit = saved.find(name);
    if (sit != saved.end()) {
        return &sit->second;
    }
    return nullptr;
}

// Resolve view shape from either "shape" or "shape_like" attribute
std::vector<long> resolve_view_shape(
    const Operation& op,
    const ShapeEnv& env,
    ExecState& st,
    std::unordered_map<std::string, Tensor>& saved) {
    // Check for shape_like attribute (used by autodiff to reference tensor shape)
    auto* shape_like_attr = find_attr(op.attrs, "shape_like");
    if (shape_like_attr) {
        if (auto ref_name = attr_string(*shape_like_attr)) {
            // Try to get shape from referenced tensor (may be saved or intermediate)
            Tensor* ref = try_get_tensor(st, *ref_name, saved);
            if (ref) {
                return std::vector<long>(ref->Sizes.begin(), ref->Sizes.begin() + ref->Rank);
            }
            // If the reference tensor isn't available, try to infer shape from input tensor
            // This handles cases like backward view for logits where logits_flat isn't saved
            // For view backward: d_logits [B,T,V] -> d_logits_flat [B*T,V]
            // The input (d_logits) shape can be flattened to get output shape
            if (!op.inputs.empty()) {
                Tensor* input = try_get_tensor(st, op.inputs[0], saved);
                if (input && input->Rank >= 2) {
                    // Assume flattening first (Rank-1) dimensions and keeping last
                    // This is a heuristic for the common [B,T,V] -> [B*T,V] case
                    long flat_dim = 1;
                    for (int i = 0; i < input->Rank - 1; ++i) {
                        flat_dim *= input->Sizes[i];
                    }
                    return {flat_dim, input->Sizes[input->Rank - 1]};
                }
            }
            throw std::runtime_error("DSL graph executor: view shape_like reference not found: " + *ref_name);
        }
        throw std::runtime_error("DSL graph executor: view shape_like attr must be a string");
    }

    auto* shape_attr = find_attr(op.attrs, "shape");
    if (!shape_attr) {
        throw std::runtime_error("DSL graph executor: view missing shape or shape_like attr");
    }
    return resolve_attr_shape(*shape_attr, env);
}

EMMTranspose parse_transpose(const AttrMap& attrs) {
    auto attr = find_attr(attrs, "transpose");
    if (!attr) {
        return EMMTranspose::NN;
    }
    if (auto s = attr_string(*attr)) {
        if (*s == "NN") return EMMTranspose::NN;
        if (*s == "NT") return EMMTranspose::NT;
        if (*s == "TN") return EMMTranspose::TN;
        if (*s == "TT") return EMMTranspose::TT;
    }
    throw std::runtime_error("DSL graph executor: invalid transpose attr");
}

EMMTranspose swap_transpose(EMMTranspose mode) {
    // Row-major GEMM mapping to column-major: swap A/B, swap M/N, and swap transpose flags.
    // NN -> NN, NT -> TN, TN -> NT, TT -> TT.
    switch (mode) {
        case EMMTranspose::NN:
            return EMMTranspose::NN;
        case EMMTranspose::NT:
            return EMMTranspose::TN;
        case EMMTranspose::TN:
            return EMMTranspose::NT;
        case EMMTranspose::TT:
            return EMMTranspose::TT;
    }
    return EMMTranspose::NN;
}

void matmul_dims(const Tensor& a, const Tensor& b, EMMTranspose mode, int& M, int& N, int& K) {
    if (a.Rank != 2 || b.Rank != 2) {
        throw std::runtime_error("DSL graph executor: matmul expects rank-2 tensors");
    }
    const long a0 = a.Sizes[0];
    const long a1 = a.Sizes[1];
    const long b0 = b.Sizes[0];
    const long b1 = b.Sizes[1];
    const bool transA = (mode == EMMTranspose::TN || mode == EMMTranspose::TT);
    const bool transB = (mode == EMMTranspose::NT || mode == EMMTranspose::TT);
    const long a_rows = transA ? a1 : a0;
    const long a_cols = transA ? a0 : a1;
    const long b_rows = transB ? b1 : b0;
    const long b_cols = transB ? b0 : b1;
    if (a_cols != b_rows) {
        throw std::runtime_error("DSL graph executor: matmul dimension mismatch");
    }
    M = static_cast<int>(a_rows);
    N = static_cast<int>(b_cols);
    K = static_cast<int>(a_cols);
}

bool is_required_op(const Operation& op, const std::unordered_set<std::string>& needed) {
    for (const auto& out : op.outputs) {
        if (needed.count(out) > 0) {
            return true;
        }
    }
    return false;
}

std::vector<char> compute_required_ops(const Graph& graph, const std::vector<std::string>& outputs) {
    std::unordered_set<std::string> needed(outputs.begin(), outputs.end());
    std::vector<char> required(graph.operations.size(), 0);
    for (std::size_t idx = graph.operations.size(); idx-- > 0;) {
        const auto& op = graph.operations[idx];
        if (!is_required_op(op, needed)) {
            continue;
        }
        required[idx] = 1;
        for (const auto& inp : op.inputs) {
            needed.insert(inp);
        }
    }
    return required;
}

void free_temps(ExecState& st) {
    for (auto it = st.temps.rbegin(); it != st.temps.rend(); ++it) {
        st.rs.temp_free(*it);
    }
    st.temps.clear();
}

void reduce_loss(RunState& rs, long B, long T, NCCLCommunicator& comm) {
    deterministic_sum(rs.Losses.template get<float>(), rs.Losses.template get<float>(), B * T, rs.MainStream);
    comm.reduce_loss(rs.Losses.template get<float>(), rs.MainStream);
    CUDA_CHECK(cudaMemcpyAsync(rs.LossHost, rs.Losses.template get<float>(), sizeof(float), cudaMemcpyDeviceToHost, rs.MainStream));
}

Tensor recompute_lora_rmsnorm(modules::LoRARunState& lora_rs, const Tensor& residual, const Tensor& weight,
                              float eps, int B, int T, int C, cudaStream_t stream) {
    if (!lora_rs.recompute_ln.Data || !lora_rs.recompute_rstd.Data) {
        throw std::runtime_error("DSL graph executor: LoRA recompute buffers not allocated");
    }
    rmsnorm_forward(lora_rs.recompute_ln, lora_rs.recompute_rstd,
                    residual, weight, nullptr, eps, B, T, C, stream);
    return lora_rs.recompute_ln;
}

modules::ForwardHook make_lora_forward_hook(Model& model,
                                           modules::ModularLoRAModel<Block>& lora,
                                           bool is_training) {
    lora.lora_run_state().is_training = is_training;

    return [&model, &lora](int layer_idx, cudaStream_t stream,
                                                       modules::ForwardHookPoint point, void* context) {
        (void)context;
        auto& lora_cfg = lora.lora_config();
        auto& lora_weights = lora.lora_weights();
        auto& lora_rs = lora.lora_run_state();
        const auto& cfg = model.config();
        auto& rs = model.run_state();
        const int B = (int)rs.B;
        const int T = (int)rs.T;
        const int C = (int)cfg.HiddenSize;
        const int D = (int)cfg.IntermediateSize;
        const int Hq = (int)cfg.NumQueryHeads;
        const int Hkv = (int)cfg.NumKeyValHeads;
        const int Hs = (int)cfg.head_size();
        const int rank = lora_cfg.rank;
        const float scaling = lora_cfg.scaling();
        const float dropout = lora_cfg.dropout;
        const bool training = lora_rs.is_training;

        auto get_dropout_seed = [&](int proj_type) -> unsigned int {
            return lora_rs.dropout_base_seed
                   + static_cast<unsigned int>(layer_idx) * 1000000u
                   + static_cast<unsigned int>(proj_type) * 100000u
                   + static_cast<unsigned int>(lora_rs.micro_step) * 10000u;
        };

        auto& acts = rs.simplified_acts(layer_idx);
        auto& lora_block = lora_weights.get_block(layer_idx, stream);
        const int BT = B * T;

        switch (point) {
            case modules::ForwardHookPoint::AfterQKVProjection: {
                if (lora_block.attention.q.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, 0, acts.ln1, lora_block.attention.q.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(0), training,
                                                             BT, C, Hq * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.k.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, Hq * Hs, acts.ln1, lora_block.attention.k.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(1), training,
                                                             BT, C, Hkv * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.attention.v.has_value()) {
                    modules::detail::apply_lora_contribution(acts.qkv, (Hq + Hkv) * Hs, acts.ln1, lora_block.attention.v.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(2), training,
                                                             BT, C, Hkv * Hs, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterAttnOutProjection: {
                if (lora_block.attention.o.has_value()) {
                    modules::detail::apply_lora_contribution(acts.att_out, 0, acts.att, lora_block.attention.o.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(3), training,
                                                             BT, Hq * Hs, C, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPUpProjection: {
                if (lora_block.mlp.up.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, 0, acts.ln2, lora_block.mlp.up.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(4), training,
                                                             BT, C, D, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
                if (lora_block.mlp.gate.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_up, D, acts.ln2, lora_block.mlp.gate.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(5), training,
                                                             BT, C, D, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            case modules::ForwardHookPoint::AfterMLPDownProjection: {
                if (lora_block.mlp.down.has_value()) {
                    modules::detail::apply_lora_contribution(acts.mlp_down, 0, acts.swiglu, lora_block.mlp.down.value(),
                                                             lora_rs.intermediate, lora_rs.slice,
                                                             scaling, dropout, get_dropout_seed(6), training,
                                                             BT, D, C, rank,
                                                             rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                }
            } break;
            default:
                break;
        }
    };
}

}  // namespace

GraphExecutor::GraphExecutor(const Module& module, Model& backend)
    : mModule(module),
      mBackend(backend),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    init(GraphExecutorOptions{});
}

GraphExecutor::GraphExecutor(const Module& module, Model& backend, const GraphExecutorOptions& options)
    : mModule(module),
      mBackend(backend),
      mForward(module.forward ? &module.forward.value() : nullptr),
      mBackward(nullptr) {
    init(options);
}

void GraphExecutor::set_lora_model(modules::ModularLoRAModel<modules::Qwen3TransformerBlock>* lora_model) {
    mLoRA = lora_model;
}

void GraphExecutor::init(const GraphExecutorOptions& options) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: module missing forward graph");
    }

    // Check if module has explicit backward graph
    if (mModule.backward.has_value()) {
        mBackward = &mModule.backward.value();
    } else if (options.auto_backward) {
        // Derive backward graph automatically using autodiff
        DeriveBackwardOptions derive_opts;
        derive_opts.loss_name = options.loss_name;
        derive_opts.auto_save = true;
        derive_opts.accumulate_grads = true;

        try {
            mDerivedBackward = derive_backward_graph(*mForward, derive_opts);
            mBackward = &mDerivedBackward.value();

            // Merge auto-computed saves with forward.save, but filter out:
            // 1. Graph outputs (they're re-computed during backward by classifier)
            // 2. Tensors produced by ops that depend on lm_head (not available in full=false mode)
            std::unordered_set<std::string> save_set(mForward->save.begin(), mForward->save.end());
            for (const auto& s : mDerivedBackward->save) {
                save_set.insert(s);
            }
            // Remove graph outputs - they're handled by the classifier
            for (const auto& [name, _] : mForward->outputs) {
                save_set.erase(name);
            }
            // Also remove tensors that are produced by ops that come after the last save-able tensor
            // (i.e., tensors that depend on lm_head which isn't available during training forward)
            // For now, we specifically exclude "logits_flat" as it's produced by the lm_head matmul
            save_set.erase("logits_flat");
            save_set.erase("logits");
            mSaveList.assign(save_set.begin(), save_set.end());

            if (options.debug_print_backward) {
                std::cerr << "[Autodiff] Derived backward graph with "
                          << mDerivedBackward->operations.size() << " operations\n";
                for (const auto& op : mDerivedBackward->operations) {
                    std::cerr << "  " << op.kernel_type << ": [";
                    for (size_t i = 0; i < op.inputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.inputs[i];
                    }
                    std::cerr << "] -> [";
                    for (size_t i = 0; i < op.outputs.size(); ++i) {
                        if (i > 0) std::cerr << ", ";
                        std::cerr << op.outputs[i];
                    }
                    std::cerr << "]\n";
                }
                std::cerr << "[Autodiff] Auto-computed saves: ";
                for (const auto& s : mSaveList) {
                    std::cerr << s << " ";
                }
                std::cerr << "\n";
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("DSL graph executor: autodiff failed: ") + e.what());
        }
    }

    if (!mBackward) {
        throw std::runtime_error(
            "DSL graph executor: module missing backward graph (set auto_backward=true to derive automatically)");
    }

    // If we didn't derive backward (using explicit backward from module), use forward.save
    if (mSaveList.empty()) {
        mSaveList = mForward->save;
    }
}

unsigned int GraphExecutor::next_rng_seed() {
    auto state = mBackend.rng_state();
    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(state.data()), static_cast<std::streamsize>(state.size()));
    std::minstd_rand rng;
    ss >> rng;
    unsigned int seed = static_cast<unsigned int>(rng());
    std::stringstream out;
    out << rng;
    auto view = out.rdbuf()->view();
    std::vector<std::byte> updated;
    updated.reserve(view.size());
    std::transform(view.begin(), view.end(), std::back_inserter(updated),
                   [](char c) { return static_cast<std::byte>(c); });
    mBackend.set_rng_state(updated);
    return seed;
}

void GraphExecutor::forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    mSaved.clear();

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        weights.invalidate();
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(rs.MainStream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(rs.MainStream);
        }
        rs.reset_moe_stats();
    }

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    modules::ForwardHook lora_hook;
    const modules::ForwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        if (mLoRA->qlora_enabled() && micro_step == 0) {
            mLoRA->invalidate_qlora_cache();
        }
        auto& lora_rs = mLoRA->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_hook = make_lora_forward_hook(model, *mLoRA, /*is_training=*/true);
        hook_ptr = &lora_hook;
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.MainStream));
    }

    execute_forward_graph(B, T, comm, /*full=*/false, hook_ptr);

    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, rs.MainStream));
}

float GraphExecutor::validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    mSaved.clear();

    if (micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.OptimizerDone, 0));
        weights.invalidate();
        if (rs.has_fp8_delayed_scaling()) {
            auto& fp8_state = rs.fp8_scaling_state();
            if (!mFP8ScalingInitialized) {
                fp8_state.reset(rs.MainStream);
                mFP8ScalingInitialized = true;
            }
            fp8_state.zero_recorded_amaxes(rs.MainStream);
        }
        rs.reset_moe_stats();
    }
    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];

    modules::ForwardHook lora_hook;
    const modules::ForwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        auto& lora_rs = mLoRA->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_hook = make_lora_forward_hook(model, *mLoRA, /*is_training=*/false);
        hook_ptr = &lora_hook;
    }

    // Copy inputs and position ids to device.
    {
        const std::size_t input_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(inputs.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Inputs.Data, inputs.Data, input_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        const std::size_t pos_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(position_ids.DType);
        if (position_ids.Device == -1) {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyHostToDevice, rs.MainStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(rs.PositionIDs.Data, position_ids.Data, pos_bytes, cudaMemcpyDeviceToDevice, rs.MainStream));
        }
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.MainStream));
    }

    execute_forward_graph(B, T, comm, /*full=*/false, hook_ptr);

    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
    CUDA_CHECK(cudaEventRecord(rs.ForwardDone, rs.MainStream));

    fill_zero(rs.Losses, rs.MainStream);
    fill_zero(rs.ValidTokenCount, rs.MainStream);
    fill_zero(rs.CorrectCount, rs.MainStream);

    const std::size_t target_bytes = static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
    if (targets.Device == -1) {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyDeviceToDevice));
    }

    run_classifier(B, T, comm, /*grad_accum_steps=*/1, micro_step, /*compute_accuracy=*/true);

    reduce_loss(rs, B, T, comm);
    comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    comm.all_reduce_sum_int(rs.CorrectCount.template get<int>(), /*n=*/1, rs.MainStream);

    CUDA_CHECK(cudaMemcpyAsync(rs.NormHost, rs.ValidTokenCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaMemcpyAsync(rs.AccuracyHost, rs.CorrectCount.Data, sizeof(int), cudaMemcpyDeviceToHost, rs.MainStream));
    CUDA_CHECK(cudaDeviceSynchronize());

    int valid_tokens = *reinterpret_cast<int*>(rs.NormHost);
    int correct_tokens = *reinterpret_cast<int*>(rs.AccuracyHost);
    if (valid_tokens > 0) {
        float avg_valid = static_cast<float>(valid_tokens) / static_cast<float>(std::max(1, comm.world_size()));
        *rs.LossHost /= avg_valid;
        *rs.AccuracyHost = (static_cast<float>(correct_tokens) / static_cast<float>(valid_tokens)) * 100.0f;
    } else {
        *rs.LossHost = 0.0f;
        *rs.AccuracyHost = 0.0f;
    }

    if (mLmHeadCached) {
        model.weights_manager().release_lm_head(rs.MainStream);
        mLmHeadCached = false;
    }
    rs.temp_free(rs.non_block_activations().output);

    return *rs.LossHost;
}

void GraphExecutor::backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& grads = model.grads();
    const auto& config = model.config();
    rs.GradAccumSteps = std::max(1, grad_accum_steps);
    rs.WorldSize = std::max(1, comm.world_size());

    const long B = inputs.Sizes[0];
    const long T = inputs.Sizes[1];
    mLastInputsCpu = inputs;

    modules::BackwardHook lora_hook;
    const modules::BackwardHook* hook_ptr = nullptr;
    if (mLoRA && mLoRA->lora_enabled()) {
        auto* lora = mLoRA;
        lora->ensure_lora_state(comm, static_cast<int>(B), static_cast<int>(T));
        auto& lora_rs = lora->lora_run_state();
        lora_rs.micro_step = micro_step;
        lora_rs.is_training = true;
        lora->lora_grads().start_micro_step(rs.MainStream, micro_step, grad_accum_steps);

        lora_hook = [&, lora](int layer_idx, bool accumulate, cudaStream_t stream,
                        modules::BackwardHookPoint point, void* context) {
            (void)context;
            auto& lora_cfg = lora->lora_config();
            auto& lora_weights = lora->lora_weights();
            auto& lora_grads_mgr = lora->lora_grads();
            auto& lora_rs = lora->lora_run_state();
            const auto& cfg = model.config();
            auto& rs = model.run_state();
            const int B = (int)rs.B;
            const int T = (int)rs.T;
            const int rank = lora_cfg.rank;
            const float dropout = lora_cfg.dropout;
            const bool is_training = lora_rs.is_training;
            const int micro_step = lora_rs.micro_step;

            auto get_dropout_seed = [&](int proj_type) -> unsigned int {
                return lora_rs.dropout_base_seed
                       + static_cast<unsigned int>(layer_idx) * 1000000u
                       + static_cast<unsigned int>(proj_type) * 100000u
                       + static_cast<unsigned int>(micro_step) * 10000u;
            };

            auto& a = rs.simplified_acts(layer_idx);
            auto& da = rs.simplified_grads(layer_idx);
            auto& lora_block = lora_weights.get_block(layer_idx, stream);

            switch (point) {
                case modules::BackwardHookPoint::AfterQKVBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int Hq = (int)cfg.NumQueryHeads;
                    const int Hkv = (int)cfg.NumKeyValHeads;
                    const int Hs = (int)cfg.head_size();

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;

                    Tensor ln1_input;
                    if (rs.config().recompute_lora) {
                        Tensor& residual = rs.get_residual(layer_idx, stream);
                        auto& block_weights = model.weights_manager().get_block(layer_idx, stream);
                        ln1_input = recompute_lora_rmsnorm(lora_rs, residual, block_weights.ln1.weight,
                                                           cfg.RmsNormEps, B, T, C, stream);
                    } else {
                        ln1_input = a.ln1;
                    }

                    Tensor dA_q{}, dB_q{}, dA_k{}, dB_k{}, dA_v{}, dB_v{};
                    modules::LoRALayerWeights<Tensor> lora_q{}, lora_k{}, lora_v{};

                    if (lora_block.attention.q.has_value() && lora_grads.attention.q.has_value()) {
                        dA_q = lora_grads.attention.q->A;
                        dB_q = lora_grads.attention.q->B;
                        lora_q = *lora_block.attention.q;
                    }
                    if (lora_block.attention.k.has_value() && lora_grads.attention.k.has_value()) {
                        dA_k = lora_grads.attention.k->A;
                        dB_k = lora_grads.attention.k->B;
                        lora_k = *lora_block.attention.k;
                    }
                    if (lora_block.attention.v.has_value() && lora_grads.attention.v.has_value()) {
                        dA_v = lora_grads.attention.v->A;
                        dB_v = lora_grads.attention.v->B;
                        lora_v = *lora_block.attention.v;
                    }

                    modules::detail::backward_lora_qkv_fused(
                        dA_q, dB_q,
                        dA_k, dB_k,
                        dA_v, dB_v,
                        da.d_ln1,
                        da.d_qkv,
                        ln1_input,
                        lora_q, lora_k, lora_v,
                        lora_cfg.scaling(),
                        dropout,
                        get_dropout_seed(0), get_dropout_seed(1), get_dropout_seed(2), is_training,
                        B * T,
                        C,
                        Hq * Hs,
                        Hkv * Hs,
                        rank,
                        lora_accum,
                        lora_rs.intermediate,
                        lora_rs.intermediate2,
                        lora_rs.slice,
                        rs.CublasLtHandle,
                        rs.CuBlasWorkspace,
                        stream);

                    lora_grads_mgr.notify_block(layer_idx, stream, comm);
                } break;
                case modules::BackwardHookPoint::AfterAttnOutBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int Hq = (int)cfg.NumQueryHeads;
                    const int Hs = (int)cfg.head_size();

                    if (!lora_block.attention.o.has_value()) {
                        break;
                    }

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.attention.o.has_value()) {
                        break;
                    }

                    const unsigned int dropout_seed =
                        lora_rs.dropout_base_seed
                        + static_cast<unsigned int>(layer_idx) * 1000000u
                        + 3u * 100000u
                        + static_cast<unsigned int>(micro_step) * 10000u;

                    Tensor x = a.att;
                    Tensor dL_dy = da.d_res_att;

                    modules::detail::backward_lora_layer(
                        lora_grads.attention.o->A, lora_grads.attention.o->B,
                        da.d_att,
                        dL_dy, 0,
                        x,
                        lora_block.attention.o->A, lora_block.attention.o->B,
                        lora_cfg.scaling(),
                        dropout, dropout_seed, is_training,
                        lora_rs.intermediate, lora_rs.slice,
                        B * T, Hq * Hs, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                case modules::BackwardHookPoint::AfterMLPUpBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int D = (int)cfg.IntermediateSize;

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;

                    Tensor ln2_input;
                    if (rs.config().recompute_lora) {
                        auto& block_weights = model.weights_manager().get_block(layer_idx, stream);
                        ln2_input = recompute_lora_rmsnorm(lora_rs, a.residual_att, block_weights.ln2.weight,
                                                           cfg.RmsNormEps, B, T, C, stream);
                    } else {
                        ln2_input = a.ln2;
                    }

                    Tensor dA_up{}, dB_up{}, dA_gate{}, dB_gate{};
                    modules::LoRALayerWeights<Tensor> lora_up{}, lora_gate{};

                    if (lora_block.mlp.up.has_value() && lora_grads.mlp.up.has_value()) {
                        dA_up = lora_grads.mlp.up->A;
                        dB_up = lora_grads.mlp.up->B;
                        lora_up = *lora_block.mlp.up;
                    }
                    if (lora_block.mlp.gate.has_value() && lora_grads.mlp.gate.has_value()) {
                        dA_gate = lora_grads.mlp.gate->A;
                        dB_gate = lora_grads.mlp.gate->B;
                        lora_gate = *lora_block.mlp.gate;
                    }

                    modules::detail::backward_lora_mlp_up_gate_fused(
                        dA_up, dB_up,
                        dA_gate, dB_gate,
                        da.d_ln2,
                        da.d_mlp_up,
                        ln2_input,
                        lora_up, lora_gate,
                        lora_cfg.scaling(),
                        dropout,
                        get_dropout_seed(4), get_dropout_seed(5), is_training,
                        B * T,
                        C,
                        D,
                        rank,
                        lora_accum,
                        lora_rs.intermediate,
                        lora_rs.intermediate2,
                        lora_rs.slice,
                        rs.CublasLtHandle,
                        rs.CuBlasWorkspace,
                        stream);
                } break;
                case modules::BackwardHookPoint::AfterMLPDownBackward: {
                    const int C = (int)cfg.HiddenSize;
                    const int D = (int)cfg.IntermediateSize;

                    if (!lora_block.mlp.down.has_value()) {
                        break;
                    }

                    bool lora_accum = false;
                    auto& lora_grads = lora_grads_mgr.get_block_full(layer_idx, stream, comm, lora_accum);
                    lora_accum = lora_accum || accumulate;
                    if (!lora_grads.mlp.down.has_value()) {
                        break;
                    }

                    const unsigned int dropout_seed =
                        lora_rs.dropout_base_seed
                        + static_cast<unsigned int>(layer_idx) * 1000000u
                        + 6u * 100000u
                        + static_cast<unsigned int>(micro_step) * 10000u;

                    Tensor x = a.swiglu;
                    Tensor dL_dy = da.d_res_ffn;

                    modules::detail::backward_lora_layer(
                        lora_grads.mlp.down->A, lora_grads.mlp.down->B,
                        da.d_swiglu,
                        dL_dy, 0,
                        x,
                        lora_block.mlp.down->A, lora_block.mlp.down->B,
                        lora_cfg.scaling(),
                        dropout, dropout_seed, is_training,
                        lora_rs.intermediate, lora_rs.slice,
                        B * T, D, C, rank, lora_accum,
                        rs.CublasLtHandle, rs.CuBlasWorkspace, stream);
                } break;
                default:
                    break;
            }
        };

        hook_ptr = &lora_hook;
    }

    // Copy targets to device (side stream).
    {
        CUDA_CHECK(cudaStreamWaitEvent(rs.side_stream(), rs.BackwardDone, 0));
        const std::size_t target_bytes =
            static_cast<std::size_t>(B) * static_cast<std::size_t>(T) * get_dtype_size(targets.DType);
        CUDA_CHECK(cudaMemcpyAsync(rs.Targets.Data, targets.Data, target_bytes, cudaMemcpyHostToDevice, rs.side_stream()));
        CUDA_CHECK(cudaEventRecord(rs.TransferDone, rs.side_stream()));
    }

    if (micro_step == 0) {
        fill_zero(rs.Losses, rs.MainStream);
        fill_zero(rs.ValidTokenCount, rs.MainStream);
        grads.start_micro_step(rs.side_stream(), micro_step, grad_accum_steps);
        CUDA_CHECK(cudaEventRecord(rs.side_stream_event(), rs.side_stream()));
    } else {
        grads.start_micro_step(rs.MainStream, micro_step, grad_accum_steps);
    }

    // Zero d_ln_final and last layer residual gradient.
    fill_zero(rs.non_block_gradients().d_ln_final, rs.MainStream);
    fill_zero(rs.simplified_grads(static_cast<int>(config.NumLayers) - 1).d_res_ffn, rs.MainStream);

    run_classifier(B, T, comm, grad_accum_steps, micro_step, /*compute_accuracy=*/false);

    const bool last_step = micro_step == grad_accum_steps - 1;
    if (last_step) {
        reduce_loss(rs, B, T, comm);
        comm.all_reduce_sum_int(rs.ValidTokenCount.template get<int>(), /*n=*/1, rs.MainStream);
    }

    execute_backward_graph(B, T, comm, grad_accum_steps, micro_step, hook_ptr);

    grads.end_micro_step(rs.MainStream, comm);
    if (mLoRA && mLoRA->lora_enabled()) {
        mLoRA->lora_grads().end_micro_step(rs.MainStream, comm);
    }
    CUDA_CHECK(cudaEventRecord(rs.BackwardDone, rs.MainStream));
    CUDA_CHECK(cudaEventSynchronize(rs.TransferDone));
}

void GraphExecutor::run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy) {
    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    const auto& options = model.options();
    const auto& config = model.config();
    if (options.lmhead_chunks != 1) {
        throw std::runtime_error("DSL graph executor: lmhead_chunks > 1 not supported yet");
    }

    const size_t V = config.VocabSize;
    const size_t Vp = config.VocabSize;
    const float d_loss = compute_accuracy ? 1.0f : (1.0f / static_cast<float>(B * T * grad_accum_steps));

    if (!mLmHeadCached) {
        weights.gather_lm_head(comm, rs.side_stream());
        mLmHeadCached = true;
    }
    rs.temp_acquire(rs.non_block_activations().output);

    // Ensure targets and gradient zeroing are visible on main stream.
    CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.TransferDone, 0));
    if (!compute_accuracy && micro_step == 0) {
        CUDA_CHECK(cudaStreamWaitEvent(rs.MainStream, rs.side_stream_event(), 0));
    }

    Tensor lnf_flat = view_tensor(rs.non_block_activations().ln_final, {B * T, config.HiddenSize});
    Tensor logits = rs.non_block_activations().output;

    // Row-major logits = lnf @ lm_head.T -> map to column-major backend by swapping A/B,
    // swapping M/N, and swapping transpose flags.
    matmul(logits, weights.get_lm_head(rs.MainStream), lnf_flat,
           std::nullopt, nullptr, nullptr, rs.CublasLtHandle, rs.CuBlasWorkspace,
           static_cast<int>(V), static_cast<int>(B * T), static_cast<int>(config.HiddenSize),
           swap_transpose(EMMTranspose::NT), false, rs.MainStream);

    Tensor tgt = rs.Targets;
    tgt.Sizes[0] = static_cast<long>(B * T);
    tgt.Rank = 1;

    Tensor losses = rs.Losses;
    losses.Sizes[0] = static_cast<long>(B * T);
    losses.Rank = 1;

    if (compute_accuracy) {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount, &rs.CorrectCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    } else {
        fused_classifier(logits, losses, d_loss, tgt,
                         &rs.ValidTokenCount,
                         static_cast<int>(B * T), static_cast<int>(V), static_cast<int>(Vp), true, rs.MainStream);
    }
}

void GraphExecutor::execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full,
                                          const modules::ForwardHook* hook) {
    if (!mForward) {
        throw std::runtime_error("DSL graph executor: missing forward graph");
    }

    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    auto& grads = model.grads();
    auto& recipe = *model.recipe();
    const auto& config = model.config();
    const auto& options = model.options();

    ExecState st{
        .model = model,
        .rs = rs,
        .weights = weights,
        .grads = grads,
        .config = config,
        .options = options,
        .recipe = recipe,
        .comm = comm,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
    };

    // Bind known inputs.
    st.tensors.emplace("token_ids", rs.Inputs);
    st.tensors.emplace("position_ids", rs.PositionIDs);
    st.tensors.emplace("x0", rs.non_block_activations().encoded);

    if (full) {
        weights.gather_lm_head(comm, rs.side_stream());
        st.tensors.emplace("lm_head", weights.get_lm_head(rs.MainStream));
    }

    std::vector<char> required;
    if (!full) {
        required = compute_required_ops(*mForward, mSaveList);
    }

    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!full && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;

        if (op_type == "embedding") {
            Tensor& token_ids = get_tensor(st, op.inputs.at(0), mSaved);
            weights.gather_embeddings(comm, rs.side_stream());
            Tensor& emb = weights.get_embeddings(rs.MainStream);
            const std::string& out_name = op.outputs.at(0);
            if (st.tensors.find(out_name) == st.tensors.end()) {
                st.tensors.emplace(out_name, rs.non_block_activations().encoded);
            }
            Tensor& out = st.tensors.at(out_name);
            encoder_forward(out, token_ids, emb, std::nullopt,
                            static_cast<int>(B), static_cast<int>(T),
                            config.HiddenSize, config.VocabSize, rs.MainStream);
            weights.release_embeddings(rs.MainStream);
            continue;
        }

        if (op_type == "zeros") {
            auto* shape_attr = find_attr(op.attrs, "shape");
            if (!shape_attr) {
                throw std::runtime_error("DSL graph executor: zeros missing shape attr");
            }
            auto shape = resolve_attr_shape(*shape_attr, st.shape_env);
            ETensorDType dtype = ETensorDType::BF16;
            if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                if (auto s = attr_string(*dtype_attr)) {
                    dtype = dtype_from_str(*s);
                }
            }
            Tensor& out = ensure_tensor(st, op.outputs.at(0), dtype, shape);
            fill_zero(out, rs.MainStream);
            st.zero_tensors.insert(op.outputs.at(0));
            continue;
        }

        if (op.name == "StackedBlocks") {
            // Execute the transformer block stack using modular path.
            if (options.use_cuda_graphs) {
                rs.configure_forward_graphs(hook != nullptr);
            }
            weights.gather_block(0, comm, rs.side_stream());
            for (int l = 0; l < config.NumLayers; ++l) {
                const int skip_first = std::max(0, options.skip_quant_first_layers);
                const int skip_last = std::max(0, options.skip_quant_last_layers);
                const bool in_skip_range = (l < skip_first) || (l >= config.NumLayers - skip_last);
                const bool allow_quant_layer = !in_skip_range;

                if (l != config.NumLayers - 1) {
                    weights.gather_block(l + 1, comm, rs.side_stream());
                }

                auto& block_weights = weights.get_block(l, rs.MainStream);
                auto& acts = rs.simplified_acts(l);
                auto& q = rs.simplified_quant_acts(l);
                Tensor& residual = (l == 0) ? rs.non_block_activations().encoded : rs.get_residual(l - 1, rs.MainStream);

                const modules::ForwardHook* hook_ptr = hook;
                modules::detail::trace_or_execute_cuda_graph_with_stack([&]() {
                    Block::template forward_block_modular<Block>(
                        recipe, rs, block_weights, acts, q, residual, l,
                        config, options, rs.MainStream,
                        rs.has_fp8_forward() ? &rs.fp8_forward_quants() : nullptr,
                        rs.has_fp4_forward() ? &rs.fp4_forward_quants() : nullptr,
                        &weights, allow_quant_layer, hook_ptr);
                }, rs.MainStream, rs.forward_block_graph(l), options.use_cuda_graphs,
                   rs.Stack, rs.forward_block_stack_checkpoint(l));

                weights.release_block(l, rs.MainStream);
                if (l > 0 && options.offload_residuals) {
                    rs.put_residual(l - 1, rs.side_stream());
                }
            }

            // Expose outputs for subsequent ops.
            auto& last = rs.simplified_acts(config.NumLayers - 1);
            st.tensors.emplace(op.outputs.at(0), last.mlp_down);
            st.tensors.emplace(op.outputs.at(1), last.residual_att);
            continue;
        }

        if (op_type == "fused_residual_rmsnorm") {
            Tensor& residual_in = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& input = get_tensor(st, op.inputs.at(1), mSaved);

            weights.gather_final_norm(comm, rs.side_stream());
            Tensor& weight = weights.get_final_norm(rs.MainStream);

            Tensor& residual_out = rs.get_final_residual();
            Tensor& y = rs.non_block_activations().ln_final;
            Tensor& rstd = rs.non_block_activations().ln_final_rstd;

            double eps = config.RmsNormEps;
            if (auto* eps_attr = find_attr(op.attrs, "eps")) {
                if (auto v = attr_double(*eps_attr)) {
                    eps = *v;
                }
            }
            fused_residual_rmsnorm_forward(residual_out, y, rstd, residual_in, input, weight, nullptr,
                                           static_cast<float>(eps), static_cast<int>(B * T),
                                           config.HiddenSize, rs.MainStream);
            weights.release_final_norm(rs.MainStream);

            st.tensors.emplace(op.outputs.at(0), residual_out);
            st.tensors.emplace(op.outputs.at(1), y);
            st.tensors.emplace(op.outputs.at(2), rstd);
            continue;
        }

        if (op_type == "view") {
            Tensor& src = get_tensor(st, op.inputs.at(0), mSaved);
            auto shape = resolve_view_shape(op, st.shape_env, st, mSaved);
            Tensor view = view_tensor(src, shape);
            st.tensors.emplace(op.outputs.at(0), view);
            continue;
        }

        if (op_type == "matmul") {
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
            // swapping M/N, and swapping transpose flags.
            EMMTranspose mode_col = swap_transpose(mode);
            matmul(out, b, a, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K, mode_col, false, rs.MainStream);
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported forward op " + op.name);
    }

    // Save requested tensors for backward (uses auto-computed list if autodiff was used).
    for (const auto& name : mSaveList) {
        auto it = st.tensors.find(name);
        if (it != st.tensors.end()) {
            mSaved[name] = it->second;
        } else if (name == "token_ids") {
            mSaved[name] = rs.Inputs;
        } else {
            throw std::runtime_error("DSL graph executor: missing save tensor " + name);
        }
    }

    free_temps(st);
    if (full) {
        weights.release_lm_head(rs.MainStream);
    }
}

void GraphExecutor::execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step,
                                           const modules::BackwardHook* hook) {
    if (!mBackward) {
        throw std::runtime_error("DSL graph executor: missing backward graph");
    }

    auto& model = mBackend;
    auto& rs = model.run_state();
    auto& weights = model.weights_manager();
    auto& grads = model.grads();
    auto& recipe = *model.recipe();
    const auto& config = model.config();
    const auto& options = model.options();
    const bool lora_only = rs.is_lora_only_mode();

    ExecState st{
        .model = model,
        .rs = rs,
        .weights = weights,
        .grads = grads,
        .config = config,
        .options = options,
        .recipe = recipe,
        .comm = comm,
        .B = B,
        .T = T,
        .shape_env = make_shape_env(mModule, B, T),
    };

    // Bind d_logits (produced by classifier) as [B, T, V] view.
    Tensor logits_view = view_tensor(rs.non_block_activations().output, {B, T, config.VocabSize});
    st.tensors.emplace("d_logits", logits_view);

    // Bind views for d_xF_flat / d_xF to write into d_ln_final.
    Tensor d_ln_final_flat = view_tensor(rs.non_block_gradients().d_ln_final, {B * T, config.HiddenSize});
    st.tensors.emplace("d_xF_flat", d_ln_final_flat);
    st.tensors.emplace("d_xF", rs.non_block_gradients().d_ln_final);

    // Bind gradient outputs for parameters.
    bool lm_head_accumulate = false;
    if (!lora_only) {
        Tensor& d_lm_head = grads.get_lm_head_full(rs.MainStream, comm, lm_head_accumulate);
        st.lm_head_accumulate = lm_head_accumulate;
        st.tensors.emplace("d_lm_head", d_lm_head);
    }
    bool final_norm_accumulate = false;
    Tensor& d_final_norm = grads.get_final_norm_full(rs.MainStream, comm, final_norm_accumulate);
    st.tensors.emplace("d_final_norm", d_final_norm);
    (void)final_norm_accumulate;

    if (!mLmHeadCached) {
        weights.gather_lm_head(comm, rs.side_stream());
        mLmHeadCached = true;
    }
    st.tensors.emplace("lm_head", weights.get_lm_head(rs.MainStream));

    for (const auto& op : mBackward->operations) {
        const std::string& op_type = op.kernel_type.empty() ? op.name : op.kernel_type;

        if (op_type == "view") {
            Tensor& src = get_tensor(st, op.inputs.at(0), mSaved);
            auto shape = resolve_view_shape(op, st.shape_env, st, mSaved);
            Tensor view = view_tensor(src, shape);
            st.tensors.emplace(op.outputs.at(0), view);
            continue;
        }

        if (op_type == "matmul") {
            if (op.outputs.at(0) == "d_lm_head" && lora_only) {
                continue;
            }
            Tensor& a = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& b = get_tensor(st, op.inputs.at(1), mSaved);
            EMMTranspose mode = parse_transpose(op.attrs);
            int M = 0, N = 0, K = 0;
            matmul_dims(a, b, mode, M, N, K);
            std::vector<long> shape{M, N};
            Tensor& out = ensure_tensor(st, op.outputs.at(0), a.DType, shape);
            bool do_accumulate = false;
            if (op.outputs.at(0) == "d_lm_head") {
                do_accumulate = st.lm_head_accumulate;
            }
            // DSL matmul uses row-major semantics; map to column-major backend by swapping A/B,
            // swapping M/N, and swapping transpose flags.
            EMMTranspose mode_col = swap_transpose(mode);
            matmul(out, b, a, std::nullopt, nullptr, nullptr,
                   rs.CublasLtHandle, rs.CuBlasWorkspace,
                   N, M, K, mode_col, do_accumulate, rs.MainStream);
            if (op.outputs.at(0) == "d_lm_head") {
                grads.notify_lm_head(rs.MainStream, comm);
            }
            continue;
        }

        if (op_type == "zeros") {
            auto* shape_attr = find_attr(op.attrs, "shape");
            if (!shape_attr) {
                throw std::runtime_error("DSL graph executor: zeros missing shape attr");
            }
            auto shape = resolve_attr_shape(*shape_attr, st.shape_env);
            ETensorDType dtype = ETensorDType::BF16;
            if (auto* dtype_attr = find_attr(op.attrs, "dtype")) {
                if (auto s = attr_string(*dtype_attr)) {
                    dtype = dtype_from_str(*s);
                }
            }
            Tensor& out = ensure_tensor(st, op.outputs.at(0), dtype, shape);
            fill_zero(out, rs.MainStream);
            st.zero_tensors.insert(op.outputs.at(0));
            continue;
        }

        if (op_type == "fused_residual_rmsnorm_backward" || op.name == "fused_residual_rmsnorm_backward") {
            Tensor& d_y = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& residual_out = get_tensor(st, op.inputs.at(2), mSaved);
            Tensor* d_residual_next = nullptr;
            if (!op.inputs.at(1).empty()) {
                d_residual_next = &get_tensor(st, op.inputs.at(1), mSaved);
            }

            weights.gather_final_norm(comm, rs.side_stream());
            Tensor& weight = weights.get_final_norm(rs.MainStream);

            Tensor& d_input = ensure_tensor(st, op.outputs.at(1), d_y.DType, {B, T, config.HiddenSize});
            st.tensors[op.outputs.at(0)] = d_input;

            Tensor* d_residual = d_residual_next;
            if (!d_residual) {
                std::string zero_name = op.id + "_dres_zero";
                Tensor& d_residual_zero = ensure_tensor(st, zero_name, d_y.DType, {B, T, config.HiddenSize});
                fill_zero(d_residual_zero, rs.MainStream);
                d_residual = &d_residual_zero;
            }

            rmsnorm_backward(d_input, d_final_norm, rs.scratch().rmsnorm_scratch,
                             *d_residual, d_y, residual_out, weight,
                             get_tensor(st, op.inputs.at(4), mSaved),
                             rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             rs.DeviceProp, rs.MainStream,
                             rs.is_lora_only_mode());

            weights.release_final_norm(rs.MainStream);
            grads.notify_final_norm(rs.MainStream, comm);
            continue;
        }

        if (op.name == "StackedBlocksBackward") {
            Tensor& d_xN = get_tensor(st, op.inputs.at(0), mSaved);
            Tensor& d_residualN = get_tensor(st, op.inputs.at(1), mSaved);

            // Initialize last layer gradient with d_xN (d_residualN should match for final norm).
            Tensor& d_res_ffn = rs.simplified_grads(config.NumLayers - 1).d_res_ffn;
            const long N = static_cast<long>(B) * static_cast<long>(T) * static_cast<long>(config.HiddenSize);
            // vector_add_sr computes scale * (left + right); scale=0.5 copies when left==right.
            vector_add_sr(d_res_ffn, d_xN, d_xN, 0.5f, N, /*seed=*/0, rs.MainStream);
            (void)d_residualN;

            if (options.use_cuda_graphs) {
                rs.configure_backward_graphs(hook != nullptr);
            }

            cudaStream_t fetch_stream = rs.side_stream();
            if (options.offload_residuals && config.NumLayers > 1) {
                rs.fetch_residual(config.NumLayers - 2, fetch_stream);
            }
            weights.gather_block(config.NumLayers - 1, comm, fetch_stream);

            for (int l = config.NumLayers - 1; l >= 0; --l) {
                if (l > 0) {
                    if (l > 1) {
                        rs.fetch_residual(l - 2, fetch_stream);
                    }
                    weights.gather_block(l - 1, comm, fetch_stream);
                }

                bool accumulate = false;
                auto& block_weights = weights.get_block(l, rs.MainStream);
                auto& block_grads = grads.get_block_full(l, rs.MainStream, comm, accumulate);
                const int skip_first = std::max(0, options.skip_quant_first_layers);
                const int skip_last = std::max(0, options.skip_quant_last_layers);
                const bool in_skip_range = (l < skip_first) || (l >= config.NumLayers - skip_last);
                const bool allow_quant_layer = !in_skip_range;

                Tensor& residual = (l == 0) ? rs.non_block_activations().encoded : rs.get_residual(l - 1, rs.MainStream);

                const modules::BackwardHook* hook_ptr = hook;
                modules::detail::trace_or_execute_cuda_graph_with_stack([&]() {
                    if (options.recompute_rmsnorm || options.recompute_qkv || options.recompute_attention ||
                        options.recompute_ffn || options.recompute_swiglu || options.recompute_block) {
                        Block::template recompute_block_modular<Block>(
                            recipe, rs, block_weights, rs.simplified_acts(l), rs.simplified_quant_acts(l),
                            residual, l, config, options, rs.MainStream,
                            rs.has_fp8_forward() ? &rs.fp8_forward_quants() : nullptr,
                            rs.has_fp4_forward() ? &rs.fp4_forward_quants() : nullptr,
                            &weights, allow_quant_layer);
                    }
                    Block::template backward_block_modular<Block>(
                        recipe, rs, block_weights, block_grads,
                        rs.simplified_acts(l), rs.simplified_grads(l),
                        rs.simplified_quant_acts(l), rs.simplified_quant_grads(),
                        l, config, options, accumulate, rs.MainStream,
                        allow_quant_layer, hook_ptr);
                }, rs.MainStream, rs.backward_block_graph(l, accumulate), options.use_cuda_graphs,
                   rs.Stack, rs.backward_block_stack_checkpoint(l, accumulate));

                // LN1 backward (matches modular model)
                {
                    auto& a = rs.simplified_acts(l);
                    auto& da = rs.simplified_grads(l);
                    Tensor* d_ln1_w = get_ln1_weight_grad(block_grads);
                    if (!d_ln1_w) {
                        throw std::logic_error("DSL graph executor: LN1 weight gradients unavailable");
                    }
                    if (l > 0) {
                        auto& prev_da = rs.simplified_grads(l - 1);
                        rmsnorm_backward(prev_da.d_res_ffn,
                                         *d_ln1_w,
                                         rs.scratch().rmsnorm_scratch,
                                         da.d_res_att,
                                         da.d_ln1,
                                         residual,
                                         block_weights.ln1.weight,
                                         a.ln1_rstd,
                                         rs.has_grad_quants() ? rs.simplified_quant_grads().d_res_ffn.abs_max() : nullptr,
                                         static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                                         rs.DeviceProp,
                                         rs.MainStream,
                                         rs.is_lora_only_mode());
                    } else {
                        rmsnorm_backward(rs.non_block_gradients().d_embeddings,
                                         *d_ln1_w,
                                         rs.scratch().rmsnorm_scratch,
                                         da.d_res_att,
                                         da.d_ln1,
                                         residual,
                                         block_weights.ln1.weight,
                                         a.ln1_rstd,
                                         nullptr,
                                         static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                                         rs.DeviceProp,
                                         rs.MainStream,
                                         rs.is_lora_only_mode());
                    }
                }

                weights.release_block(l, rs.MainStream);
                grads.notify_block(l, rs.MainStream, comm);

                if (l > 0) {
                    rs.release_residual(l - 1, rs.MainStream);
                }
            }

            st.tensors.emplace(op.outputs.at(0), rs.non_block_gradients().d_embeddings);
            st.tensors.emplace(op.outputs.at(1), rs.non_block_gradients().d_embeddings);
            continue;
        }

        if (op_type == "embedding_backward" || op.name == "embedding_backward") {
            if (lora_only) {
                continue;
            }
            Tensor& d_out = get_tensor(st, op.inputs.at(0), mSaved);
            bool accumulate = false;
            Tensor& d_emb = grads.get_embeddings_full(rs.MainStream, comm, accumulate);
            encoder_backward(d_emb,
                             rs.scratch().encoder_bwd_scratch,
                             rs.scratch().encoder_bwd_indices,
                             rs.scratch().encoder_bwd_info,
                             d_out,
                             rs.Inputs,
                             mLastInputsCpu,
                             static_cast<int>(B), static_cast<int>(T), config.HiddenSize,
                             next_rng_seed(),
                             rs.MainStream,
                             rs.side_stream_event(),
                             rs.side_stream());
            grads.notify_embeddings(rs.MainStream, comm);
            continue;
        }

        throw std::runtime_error("DSL graph executor: unsupported backward op " + op.name);
    }

    free_temps(st);
    if (mLmHeadCached) {
        weights.release_lm_head(rs.MainStream);
        mLmHeadCached = false;
    }
    rs.temp_free(rs.non_block_activations().output);
}

}  // namespace dsl
