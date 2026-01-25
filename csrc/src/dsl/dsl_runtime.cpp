// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime components implementation.

#include "dsl/dsl_runtime.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "kernels/kernels.h"
#include "training/runtime_options.h"
#include "modules/lora/lora_config.h"
#include "modules/fp8_run_state.h"
#include "modules/fp8_scaling_config.h"
#include "modules/fp8_scaling_state.h"
#include "modules/matmul_context.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

bool is_rope_param(const std::string& name) {
    return name.find("rope_freqs") != std::string::npos;
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
    auto vocab = get_long("vocab_size");
    if (!vocab) {
        vocab = get_long("vocab");
    }

    if (d_model) {
        env.values.emplace("C", *d_model);
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
        env.values.emplace("MUp", 2 * (*d_ff));
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
}

} // namespace

DslParamStore::DslParamStore(const Module& module,
                             const Graph& graph,
                             const RuntimeOptions& options,
                             const PretrainedConfig& config,
                             const std::shared_ptr<TensorAllocator>& allocator,
                             const modules::ModularLoRAConfig* lora_config)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslParamStore: allocator is null");
    }

    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

    const bool freeze_base = lora_config && lora_config->enabled();
    const bool train_router = freeze_base && lora_config->train_router;
    auto is_router_param = [&](const std::string& name) -> bool {
        return name.find("router") != std::string::npos;
    };

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
        entry.tensor = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
        entry.trainable = !is_rope_param(name);
        if (freeze_base) {
            entry.trainable = train_router && is_router_param(name);
        }

        mParams.emplace(name, entry);
        mParamOrder.push_back(name);
    }

    // Deterministic ordering for optimizer updates/checkpointing.
    std::sort(mParamOrder.begin(), mParamOrder.end());
}

Tensor& DslParamStore::get(const std::string& name) {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
    }
    return it->second.tensor;
}

const Tensor& DslParamStore::get(const std::string& name) const {
    auto it = mParams.find(name);
    if (it == mParams.end()) {
        throw std::runtime_error("DslParamStore: missing parameter " + name);
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

void DslParamStore::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    for (const auto& name : mParamOrder) {
        auto it = mParams.find(name);
        if (it == mParams.end()) continue;
        callback(name, TensorShard(it->second.tensor));
    }
}

DslGradStore::DslGradStore(const DslParamStore& params,
                           const std::shared_ptr<TensorAllocator>& allocator)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslGradStore: allocator is null");
    }

    for (const auto& name : params.param_names()) {
        if (!params.is_trainable(name)) {
            continue;
        }
        const Tensor& weight = params.get(name);
        std::vector<long> shape(weight.Sizes.begin(), weight.Sizes.begin() + weight.Rank);
        Tensor grad = mAllocator->allocate(weight.DType, ("d_" + name).c_str(), EAllocationType::ON_DEVICE, shape);
        mGrads.emplace(name, grad);
        mParamOrder.push_back(name);
    }
    std::sort(mParamOrder.begin(), mParamOrder.end());

    // Initialize double-buffer block states
    mBlockStates[0] = {-1, false, nullptr};
    mBlockStates[1] = {-1, false, nullptr};
}

void DslGradStore::configure(const DslGradStoreConfig& config) {
    mConfig = config;
    // Early exit if no gradients to manage (e.g., LoRA-only training)
    if (mParamOrder.empty()) {
        return;
    }
    if (mConfig.num_layers > 0) {
        build_layer_grad_map();
        // Only create events if we have layer gradients to reduce
        if (mHasLayerGrads) {
            create_layer_events(mConfig.num_layers);
        }
    }
}

void DslGradStore::build_layer_grad_map() {
    mLayerGradNames.clear();
    mLayerGradNames.resize(mConfig.num_layers);
    mHasLayerGrads = false;

    // Parse gradient names to determine which layer they belong to.
    // Naming convention: "blocks.{layer_idx}.{component}" or "layers.{layer_idx}.{component}"
    for (const auto& name : mParamOrder) {
        int layer_idx = -1;

        // Check for "blocks.N." or "layers.N." pattern
        auto extract_layer = [&](const std::string& prefix) -> bool {
            auto pos = name.find(prefix);
            if (pos != std::string::npos) {
                pos += prefix.size();
                auto dot_pos = name.find('.', pos);
                if (dot_pos != std::string::npos) {
                    try {
                        layer_idx = std::stoi(name.substr(pos, dot_pos - pos));
                        return true;
                    } catch (...) {}
                }
            }
            return false;
        };

        if (extract_layer("blocks.") || extract_layer("layers.") ||
            extract_layer("model.layers.") || extract_layer("model.blocks.")) {
            if (layer_idx >= 0 && layer_idx < mConfig.num_layers) {
                mLayerGradNames[layer_idx].push_back(name);
                mHasLayerGrads = true;
            }
        }
        // Non-layer params (embeddings, lm_head, final_norm) are not tracked per-layer
    }
}

void DslGradStore::create_layer_events(int num_layers) {
    destroy_layer_events();
    mLayerReduceEvents.resize(static_cast<std::size_t>(num_layers), nullptr);
    for (auto& ev : mLayerReduceEvents) {
        CUDA_CHECK(cudaEventCreate(&ev));
    }

    // Create events for double-buffer states
    for (auto& state : mBlockStates) {
        if (!state.Event) {
            CUDA_CHECK(cudaEventCreate(&state.Event));
        }
    }
}

void DslGradStore::destroy_layer_events() noexcept {
    for (auto& ev : mLayerReduceEvents) {
        if (ev) {
            cudaEventDestroy(ev);
            ev = nullptr;
        }
    }
    mLayerReduceEvents.clear();

    for (auto& state : mBlockStates) {
        if (state.Event) {
            cudaEventDestroy(state.Event);
            state.Event = nullptr;
        }
    }
}

void DslGradStore::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    mMicroStep = micro_step;
    mIsLastMicroStep = (micro_step == total_steps - 1);
    mAccumulate = micro_step > 0;
    if (!mAccumulate) {
        zero_all(stream);
    }

    // Reset block states for new micro-step (only needed when overlapped reduction is active)
    if (mHasLayerGrads) {
        for (auto& state : mBlockStates) {
            state.LayerIdx = -1;
            state.NeedsAccumulation = false;
        }
    }
}

void DslGradStore::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    (void)stream;
    (void)comm;
    // Note: For ZeRO-2, any pending accumulations would be handled here.
    // Currently we wait for all reductions inline via wait_for_block_reduce.
}

Tensor* DslGradStore::get_param_grad(const std::string& name, bool& accumulate) {
    auto it = mGrads.find(name);
    if (it == mGrads.end()) {
        return nullptr;
    }
    accumulate = mAccumulate;
    return &it->second;
}

void DslGradStore::zero_all(cudaStream_t stream) {
    for (auto& kv : mGrads) {
        fill_zero(kv.second, stream);
    }
}

void DslGradStore::reduce_all(NCCLCommunicator& comm, cudaStream_t stream) {
    for (auto& kv : mGrads) {
        comm.all_reduce_avg(kv.second, stream);
    }
    mReducePending = false;
}

void DslGradStore::reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event) {
    // If overlapped reduction is enabled and we have layer gradients that were already reduced,
    // only reduce non-layer gradients (embeddings, lm_head, final_norm)
    if (is_overlapped_enabled() && mHasLayerGrads) {
        // Collect non-layer gradient names (those not in any layer)
        std::unordered_set<std::string> layer_grads;
        for (const auto& layer_names : mLayerGradNames) {
            for (const auto& name : layer_names) {
                layer_grads.insert(name);
            }
        }

        // Reduce non-layer gradients
        for (auto& kv : mGrads) {
            if (layer_grads.find(kv.first) == layer_grads.end()) {
                comm.all_reduce_avg(kv.second, stream);
            }
        }
    } else {
        // Fallback: reduce all gradients (original behavior)
        for (auto& kv : mGrads) {
            comm.all_reduce_avg(kv.second, stream);
        }
    }

    // Record completion event so optimizer can wait on it
    if (done_event) {
        CUDA_CHECK(cudaEventRecord(done_event, stream));
    }
    mReducePending = true;
}

void DslGradStore::notify_block(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    // Single-GPU: nothing to reduce
    if (mConfig.num_shards == 1) return;

    // No layer gradient map: can't do per-layer reduction
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;
    if (mLayerGradNames[layer_idx].empty()) return;

    if (!mConfig.shard_gradients) {
        // ZeRO-1: reduce-scatter once per optimizer step (on the last micro-step)
        if (!mIsLastMicroStep) return;
        scatter_reduce_layer(layer_idx, stream, comm);
        return;
    }

    // ZeRO-2: reduce-scatter on every micro-step
    // Use double-buffering to overlap reduction with next layer's compute
    auto& state = mBlockStates[layer_idx % 2];

    // Wait for previous layer using this buffer slot to finish its reduction
    if (state.NeedsAccumulation && state.Event) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, state.Event, 0));
        state.NeedsAccumulation = false;
    }

    state.LayerIdx = layer_idx;
    scatter_reduce_layer(layer_idx, stream, comm);
    state.NeedsAccumulation = true;
}

void DslGradStore::wait_for_block_reduce(int layer_idx, cudaStream_t stream) {
    if (mConfig.num_shards == 1) return;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerReduceEvents.size())) return;

    cudaEvent_t ev = mLayerReduceEvents[layer_idx];
    if (ev) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, ev, 0));
    }
}

std::vector<Tensor*> DslGradStore::get_layer_grads(int layer_idx) {
    std::vector<Tensor*> result;
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) {
        return result;
    }

    for (const auto& name : mLayerGradNames[layer_idx]) {
        auto it = mGrads.find(name);
        if (it != mGrads.end()) {
            result.push_back(&it->second);
        }
    }
    return result;
}

void DslGradStore::scatter_reduce_layer(int layer_idx, cudaStream_t stream, NCCLCommunicator& comm) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(mLayerGradNames.size())) return;

    const auto& grad_names = mLayerGradNames[layer_idx];
    if (grad_names.empty()) return;

    cudaEvent_t ev = (layer_idx < static_cast<int>(mLayerReduceEvents.size()))
                         ? mLayerReduceEvents[layer_idx]
                         : nullptr;

    if (mConfig.shard_gradients) {
        // ZeRO-2: Use reduce-scatter (batched transaction)
        comm.begin_transaction(stream);
        for (const auto& name : grad_names) {
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                comm.schedule_reduce_scatter(it->second);
            }
        }
        comm.execute_transaction(ev);
    } else {
        // DDP/ZeRO-1: Use all-reduce (one per tensor, as the API requires)
        for (const auto& name : grad_names) {
            auto it = mGrads.find(name);
            if (it != mGrads.end() && it->second.Data && it->second.nelem() > 0) {
                comm.all_reduce_avg(it->second, stream);
            }
        }
        // Record event after all reductions
        if (ev) {
            CUDA_CHECK(cudaEventRecord(ev, stream));
        }
    }
}

DslRunState::DslRunState(const PretrainedConfig& config,
                         const RuntimeOptions& options,
                         int B, int T,
                         const std::shared_ptr<TensorAllocator>& allocator,
                         bool lora_only_mode,
                         std::size_t stack_bytes,
                         bool allocate_stack)
    : IRunState(config.clone(), B, T, allocator),
      mAllocator(allocator),
      mRecomputeBlock(options.RecomputeBlock),
      mRecomputeLoRA(options.RecomputeLoRA),
      mLoraOnlyMode(lora_only_mode),
      mNumLayers(config.NumLayers),
      mPerLayerGraphsEnabled(options.UseCudaGraphs) {
    if (!mAllocator) {
        throw std::runtime_error("DslRunState: allocator is null");
    }

    mActivationDtype = options.ModelType.value_or(config.DType);
    if (is_fp8_dtype(mActivationDtype)) {
        mActivationDtype = ETensorDType::BF16;
    }
    mGradDtype = mActivationDtype;
    mMatmulDtype = options.MatmulType.value_or(options.ModelType.value_or(config.DType));
    if (options.TrainingRecipe && options.TrainingRecipe->is_fp8_hybrid()) {
        mGradQuantDtype = ETensorDType::FP8_E5M2;
    } else {
        mGradQuantDtype = options.GradientType.value_or(mMatmulDtype);
    }
    mEnableFp8Forward = options.fp8_forward_enabled();
    mStackSimulate = !allocate_stack;

    const std::size_t stack_capacity = (stack_bytes > 0) ? stack_bytes : kDefaultStackBytes;
    if (allocate_stack) {
        // Allocate stack memory (heuristic size).
        mStackBuffer = mAllocator->allocate(
            ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE,
            {static_cast<long>(stack_capacity)});
        Stack = DeviceMemoryStack(mStackBuffer.Data, stack_capacity, DeviceId);
    } else {
        // Dummy stack for sizing pass (no device allocation).
        Stack = DeviceMemoryStack(nullptr, stack_capacity, DeviceId);
    }

    create_cuda_resources();
    allocate_non_block_state(config);
    allocate_simplified_activations(config);
    allocate_simplified_gradients(config);
    allocate_simplified_quant_buffers(config, options);
    allocate_residual_buffers(config, options.OffloadResidual);
    allocate_scratch_buffers(config);

    // Allocate per-layer CUDA graph arrays
    allocate_graph_arrays(config.NumLayers);
}

DslRunState::~DslRunState() {
    destroy_cuda_graphs();
    release_cuda_resources();
}

void DslRunState::set_stack_buffer(Tensor buffer, const DeviceMemoryStack::AllocationList& high_mark) {
    if (!buffer.Data || buffer.bytes() == 0) {
        throw std::runtime_error("DslRunState::set_stack_buffer: invalid stack buffer");
    }
    mStackBuffer = std::move(buffer);
    Stack = DeviceMemoryStack(mStackBuffer.Data, static_cast<std::size_t>(mStackBuffer.bytes()), DeviceId);
    if (!high_mark.empty()) {
        Stack.set_high_mark(high_mark);
    }
    mStackSimulate = false;
}

Tensor& DslRunState::get_residual(int layer_idx, cudaStream_t stream) {
    if (!mResidualManager) {
        throw std::runtime_error("DslRunState: residual manager not initialized");
    }
    return mResidualManager->get_residual(layer_idx, stream);
}

Tensor& DslRunState::get_final_residual() {
    if (!mResidualManager) {
        throw std::runtime_error("DslRunState: residual manager not initialized");
    }
    return mResidualManager->get_final_residual();
}

void DslRunState::allocate_non_block_state(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long V = cfg.VocabSize;
    const auto dtype = mActivationDtype;

    mNonBlockActivations.encoded = mAllocator->allocate(dtype, "encoded", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final = mAllocator->allocate(dtype, "ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final_rstd = mAllocator->allocate(ETensorDType::FP32, "ln_final_rstd", EAllocationType::ON_DEVICE, {B, T});

    // Output buffer (persistent; avoids large stack pressure for full fine-tuning).
    mNonBlockActivations.output = mAllocator->allocate(dtype, "output", EAllocationType::ON_DEVICE, {B * T, V});

    // RoPE frequencies (if not using fused RoPE).
    const int max_seq_len = std::min(static_cast<int>(T), cfg.MaxPositionEmbeddings);
    if (max_seq_len > 0) {
        const int head_size = cfg.head_size();
        if (dtype == ETensorDType::BF16) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, cfg.RopeTheta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            precompute_freqs_cis(freq_cpu.data(), head_size, max_seq_len, cfg.RopeTheta);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // Default: allocate in model dtype and leave zeroed.
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            fill_zero(mNonBlockActivations.freq_cis, MainStream);
        }
    }

    mNonBlockGradients.d_ln_final = mAllocator->allocate(mGradDtype, "d_ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockGradients.d_embeddings = mAllocator->allocate(mGradDtype, "d_embeddings", EAllocationType::ON_DEVICE, {B, T, C});
}

void DslRunState::allocate_simplified_activations(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;
    const bool use_qk_norm = cfg.UseQKNorm;

    const auto dtype = mActivationDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    const bool lora_only = mLoraOnlyMode;
    const bool lora_can_share_ln = !lora_only || mRecomputeLoRA;
    const bool lora_can_share_att = !lora_only;
    const bool lora_can_share_qkv = !lora_only;
    const bool lora_can_share_mlp_up = !lora_only;
    const bool lora_can_share_swiglu = !lora_only || mRecomputeLoRA;
    const bool share_ln1 = mRecomputeBlock && lora_can_share_ln;
    const bool share_ln2 = mRecomputeBlock && lora_can_share_ln;
    const bool share_qkv = mRecomputeBlock && lora_can_share_qkv;
    const bool share_att = mRecomputeBlock && lora_can_share_att;
    const bool share_mlp_up = mRecomputeBlock && lora_can_share_mlp_up;
    const bool share_swiglu = mRecomputeBlock && lora_can_share_swiglu;
    // Keep per-layer residual_att and mlp_down to preserve per-layer inputs for recompute.
    const bool share_residual = false;
    const bool ffn_temps_on_stack = mRecomputeBlock && lora_can_share_mlp_up && lora_can_share_swiglu;
    mFfnTempsOnStack = ffn_temps_on_stack;
    if (mStackSimulate && ffn_temps_on_stack) {
        const long mlp_up_bytes = B * T * MUp * get_dtype_size(dtype);
        const long swiglu_bytes = B * T * M * get_dtype_size(dtype);
        auto* sim_mlp_up = Stack.allocate(static_cast<std::size_t>(mlp_up_bytes), "mlp_up_simulate");
        auto* sim_swiglu = Stack.allocate(static_cast<std::size_t>(swiglu_bytes), "swiglu_simulate");
        Stack.free(sim_swiglu);
        Stack.free(sim_mlp_up);
    }

    Tensor shared_ln1{}, shared_ln2{}, shared_qkv{}, shared_att{}, shared_att_out{};
    Tensor shared_mlp_up{}, shared_swiglu{}, shared_residual_att{}, shared_mlp_down{};

    if (share_ln1) shared_ln1 = mAllocator->allocate(dtype, "ln1_shared", kind, {B, T, C});
    if (share_ln2) shared_ln2 = mAllocator->allocate(dtype, "ln2_shared", kind, {B, T, C});
    if (share_qkv) shared_qkv = mAllocator->allocate(dtype, "qkv_shared", kind, {B, T, QKV});
    if (share_att) {
        shared_att = mAllocator->allocate(dtype, "att_shared", kind, {B, T, AttnDim});
        shared_att_out = mAllocator->allocate(dtype, "att_out_shared", kind, {B, T, C});
    }
    if (share_mlp_up && !ffn_temps_on_stack) shared_mlp_up = mAllocator->allocate(dtype, "mlp_up_shared", kind, {B, T, MUp});
    if (share_swiglu && !ffn_temps_on_stack) shared_swiglu = mAllocator->allocate(dtype, "swiglu_shared", kind, {B, T, M});
    if (share_residual) {
        shared_residual_att = mAllocator->allocate(dtype, "residual_att_shared", kind, {B, T, C});
        shared_mlp_down = mAllocator->allocate(dtype, "mlp_down_shared", kind, {B, T, C});
    }

    mSimplifiedActivations.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& acts = mSimplifiedActivations[i];
        acts.ln1_rstd = mAllocator->allocate(ETensorDType::FP32, "ln1_rstd", kind, {B, T});
        acts.ln1 = share_ln1 ? shared_ln1 : mAllocator->allocate(dtype, "ln1", kind, {B, T, C});

        acts.ln2_rstd = mAllocator->allocate(ETensorDType::FP32, "ln2_rstd", kind, {B, T});
        acts.ln2 = share_ln2 ? shared_ln2 : mAllocator->allocate(dtype, "ln2", kind, {B, T, C});

        if (use_qk_norm) {
            acts.q_rstd = mAllocator->allocate(ETensorDType::FP32, "q_rstd", kind, {B, T, Hq});
            acts.k_rstd = mAllocator->allocate(ETensorDType::FP32, "k_rstd", kind, {B, T, Hkv});
        } else {
            acts.q_rstd = {};
            acts.k_rstd = {};
        }

        acts.qkv = share_qkv ? shared_qkv : mAllocator->allocate(dtype, "qkv", kind, {B, T, QKV});
        acts.qkv_rope = {};

        acts.lse = mAllocator->allocate(ETensorDType::FP32, "lse", kind, {B, T, Hq});
        acts.att = share_att ? shared_att : mAllocator->allocate(dtype, "att", kind, {B, T, AttnDim});
        acts.att_out = share_att ? shared_att_out : mAllocator->allocate(dtype, "att_out", kind, {B, T, C});

        if (share_residual) {
            acts.residual_att = shared_residual_att;
        } else {
            acts.residual_att = mAllocator->allocate(dtype, "residual_att", kind, {B, T, C});
        }

        if (ffn_temps_on_stack) {
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
        } else {
            acts.mlp_up = share_mlp_up ? shared_mlp_up : mAllocator->allocate(dtype, "mlp_up", kind, {B, T, MUp});
            acts.swiglu = share_swiglu ? shared_swiglu : mAllocator->allocate(dtype, "swiglu", kind, {B, T, M});
        }

        if (share_residual) {
            acts.mlp_down = shared_mlp_down;
        } else {
            acts.mlp_down = mAllocator->allocate(dtype, "mlp_down", kind, {B, T, C});
        }
    }
}

void DslRunState::allocate_simplified_gradients(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;

    const auto dtype = mGradDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    const bool large_bwd_temps_on_stack = mRecomputeBlock;
    if (mStackSimulate && large_bwd_temps_on_stack) {
        const long d_qkv_bytes = B * T * QKV * get_dtype_size(dtype);
        const long d_mlp_up_bytes = B * T * MUp * get_dtype_size(dtype);
        const long d_swiglu_bytes = B * T * M * get_dtype_size(dtype);
        const long d_up_bytes = B * T * MUp * get_dtype_size(dtype);
        auto* sim_d_qkv = Stack.allocate(static_cast<std::size_t>(d_qkv_bytes), "d_qkv_simulate");
        auto* sim_d_mlp_up = Stack.allocate(static_cast<std::size_t>(d_mlp_up_bytes), "d_mlp_up_simulate");
        auto* sim_d_swiglu = Stack.allocate(static_cast<std::size_t>(d_swiglu_bytes), "d_swiglu_simulate");
        auto* sim_d_up = Stack.allocate(static_cast<std::size_t>(d_up_bytes), "d_up_simulate");
        Stack.free(sim_d_up);
        Stack.free(sim_d_swiglu);
        Stack.free(sim_d_mlp_up);
        Stack.free(sim_d_qkv);
    }

    mSimplifiedGradients.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& g = mSimplifiedGradients[i];
        g.d_res_ffn = mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        g.d_ln2 = mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});

        if (large_bwd_temps_on_stack) {
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, QKV});
        } else {
            g.d_mlp_up = mAllocator->allocate(dtype, "d_mlp_up", kind, {B, T, MUp});
            g.d_swiglu = mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, M});
            g.d_qkv = mAllocator->allocate(dtype, "d_qkv", kind, {B, T, QKV});
        }

        g.d_mlp_down = mAllocator->allocate(dtype, "d_mlp_down", kind, {B, T, C});
        g.d_att = mAllocator->allocate(dtype, "d_att", kind, {B, T, Hq * D});
        g.d_ln1 = mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
    }

    // Preserve the original buffer pointers so we can restore them if the
    // compiled executor temporarily aliases gradients to stack-backed temps.
    mSimplifiedGradientsBase = mSimplifiedGradients;
}

void DslRunState::reset_simplified_gradients() {
    if (mSimplifiedGradientsBase.size() != mSimplifiedGradients.size()) {
        return;
    }
    for (std::size_t i = 0; i < mSimplifiedGradients.size(); ++i) {
        auto& dst = mSimplifiedGradients[i];
        const auto& src = mSimplifiedGradientsBase[i];

        dst.d_res_ffn.Data = src.d_res_ffn.Data;
        dst.d_res_att.Data = src.d_res_att.Data;
        dst.d_ln2.Data = src.d_ln2.Data;
        dst.d_mlp_up.Data = src.d_mlp_up.Data;
        dst.d_swiglu.Data = src.d_swiglu.Data;
        dst.d_mlp_down.Data = src.d_mlp_down.Data;
        dst.d_att.Data = src.d_att.Data;
        dst.d_qkv.Data = src.d_qkv.Data;
        dst.d_ln1.Data = src.d_ln1.Data;

        dst.d_mamba_normed.Data = src.d_mamba_normed.Data;
        dst.d_mamba_gated.Data = src.d_mamba_gated.Data;
        dst.d_mamba_scan_out.Data = src.d_mamba_scan_out.Data;
        dst.d_mamba_u.Data = src.d_mamba_u.Data;
        dst.d_mamba_delta.Data = src.d_mamba_delta.Data;
        dst.d_mamba_B.Data = src.d_mamba_B.Data;
        dst.d_mamba_C.Data = src.d_mamba_C.Data;
        dst.d_mamba_conv_out.Data = src.d_mamba_conv_out.Data;
    }
}

void DslRunState::allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;

    if (mEnableFp8Forward) {
        modules::allocate_fp8_forward_buffers(
            mFP8ForwardQuants, mFP8ForwardStats, *mAllocator,
            B, T, C, M, AttnDim, options.forward_matmul_dtype());
    }

    if (options.fp8_hybrid_enabled()) {
        modules::FP8ScalingConfig fp8_cfg{};
        fp8_cfg.amax_history_len = options.RecipeOptions.fp8_amax_history_len;
        fp8_cfg.margin = static_cast<float>(options.RecipeOptions.fp8_margin);
        mFP8ScalingState = std::make_unique<modules::FP8ScalingState>(
            fp8_cfg, mAllocator, DeviceId, cfg.NumLayers);
    }

    if (mGradQuantDtype == mGradDtype) {
        const std::array<long, 3> ln_shape{B, T, C};
        const std::array<long, 3> mlp_up_shape{B, T, MUp};
        const std::array<long, 3> qkv_shape{B, T, QKV};
        mSimplifiedQuantGrads.d_res_ffn = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, ln_shape);
        mSimplifiedQuantGrads.d_res_att = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, ln_shape);
        mSimplifiedQuantGrads.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, mlp_up_shape);
        mSimplifiedQuantGrads.d_qkv = Tensor::from_pointer(nullptr, DeviceId, mGradQuantDtype, qkv_shape);
        return;
    }

    mGradQuantStats = mAllocator->allocate(ETensorDType::FP32, "dsl_grad_quant_stats",
                                           EAllocationType::ON_DEVICE, {8L});
    float* stats = mGradQuantStats.get<float>();

    auto alloc = [&](ETensorDType dtype, const std::string& name, const std::vector<long>& shape) -> Tensor {
        return mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
    };

    mSimplifiedQuantGrads.d_res_ffn = alloc(mGradQuantDtype, "dsl_d_res_ffn_q", {B, T, C});
    mSimplifiedQuantGrads.d_res_ffn.Stats = stats + 0;
    mSimplifiedQuantGrads.d_res_att = alloc(mGradQuantDtype, "dsl_d_res_att_q", {B, T, C});
    mSimplifiedQuantGrads.d_res_att.Stats = stats + 2;
    mSimplifiedQuantGrads.d_mlp_up = alloc(mGradQuantDtype, "dsl_d_mlp_up_q", {B, T, MUp});
    mSimplifiedQuantGrads.d_mlp_up.Stats = stats + 4;
    mSimplifiedQuantGrads.d_qkv = alloc(mGradQuantDtype, "dsl_d_qkv_q", {B, T, QKV});
    mSimplifiedQuantGrads.d_qkv.Stats = stats + 6;
}

void DslRunState::allocate_scratch_buffers(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    const long QKV = D * (Hq + 2 * Hkv);

    const long rmsnorm_scratch_bytes = static_cast<long>(get_rmsnorm_backward_scratch_size(static_cast<int>(C), DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(
        ETensorDType::BYTE, "rmsnorm_scratch", EAllocationType::ON_DEVICE, {rmsnorm_scratch_bytes});

    const long M = cfg.IntermediateSize;
    const long MUp = 2 * M;
    const long V = cfg.VocabSize;
    const long max_bias_channels = std::max<long>(QKV, std::max<long>(C, std::max<long>(MUp, V)));
    const long bias_scratch_bytes =
        static_cast<long>(get_bias_backward_scratch_size(mGradDtype, static_cast<int>(max_bias_channels), DeviceProp));
    mScratch.matmul_bias_scratch = mAllocator->allocate(
        ETensorDType::FP32, "bias_scratch", EAllocationType::ON_DEVICE, {bias_scratch_bytes / static_cast<long>(sizeof(float))});

    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(DeviceProp)));
    mScratch.norm_buffer = mAllocator->allocate(
        ETensorDType::FP32, "norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums});

    mScratch.matmul_scales = mAllocator->allocate(
        ETensorDType::FP32, "matmul_scales", EAllocationType::ON_DEVICE, {2L});

    const long group_width = static_cast<long>(16 / get_dtype_size(mGradDtype) * 32);
    const long num_c_groups = (C + group_width - 1) / group_width;
    mScratch.encoder_bwd_scratch = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_scratch", EAllocationType::ON_DEVICE, {B, T, num_c_groups * 5});
    mScratch.encoder_bwd_indices = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_indices", EAllocationType::PINNED, {B, T, num_c_groups});
    mScratch.encoder_bwd_info = mAllocator->allocate(
        ETensorDType::INT32, "encoder_bwd_info", EAllocationType::PINNED, {B, T, 4 * num_c_groups});

    const long cudnn_ws_size = static_cast<long>(
        cudnn_get_workspace_size(static_cast<int>(B), static_cast<int>(T), static_cast<int>(Hq),
                                 static_cast<int>(Hkv), static_cast<int>(D), CudnnHandle));
    mScratch.cudnn_workspace = Tensor{ETensorDType::BYTE, {cudnn_ws_size}, nullptr, nullptr, 1, DeviceId};

    if (mStackSimulate) {
        if (mRecomputeBlock) {
            const long d_qkv_bytes = B * T * QKV * get_dtype_size(mGradDtype);
            auto* simulated_d_qkv = Stack.allocate(static_cast<std::size_t>(d_qkv_bytes), "d_qkv_simulate");
            auto* simulated_ws = Stack.allocate(static_cast<std::size_t>(mScratch.cudnn_workspace.bytes()), "workspace");
            Stack.free(simulated_ws);
            Stack.free(simulated_d_qkv);
        } else {
            auto* simulated_ws = Stack.allocate(static_cast<std::size_t>(mScratch.cudnn_workspace.bytes()), "workspace");
            Stack.free(simulated_ws);
        }
    }
}

Tensor* DslRunState::get_fp8_forward_buffer(int op) {
    if (!has_fp8_forward()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV:
            return &mFP8ForwardQuants.ln1;
        case modules::MatmulOp::MLPUp:
            return &mFP8ForwardQuants.ln2;
        case modules::MatmulOp::AttnOut:
            return &mFP8ForwardQuants.att;
        case modules::MatmulOp::MLPDown:
            return &mFP8ForwardQuants.swiglu;
        default:
            return nullptr;
    }
}

Tensor* DslRunState::get_gradient_quant_buffer(int op) {
    if (!has_grad_quants()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV:
            return &mSimplifiedQuantGrads.d_qkv;
        case modules::MatmulOp::MLPUp:
            return &mSimplifiedQuantGrads.d_mlp_up;
        case modules::MatmulOp::AttnOut:
            return &mSimplifiedQuantGrads.d_res_att;
        case modules::MatmulOp::MLPDown:
            return &mSimplifiedQuantGrads.d_res_ffn;
        default:
            return nullptr;
    }
}

void DslRunState::allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals) {
    mOffloadResiduals = offload_residuals;
    mResidualManager = std::make_unique<modules::ResidualManager>(
        mAllocator,
        cfg.NumLayers,
        static_cast<int>(B),
        static_cast<int>(T),
        cfg.HiddenSize,
        cfg.DType,
        offload_residuals,
        /*num_residual_buffers=*/2,
        MainStream);
}

void DslRunState::fetch_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->fetch_residual(layer_idx, stream);
    }
}

void DslRunState::put_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->put_residual(layer_idx, stream);
    }
}

void DslRunState::mark_residual_ready(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->mark_residual_ready(layer_idx, stream);
    }
}

void DslRunState::release_residual(int layer_idx, cudaStream_t stream) {
    if (mResidualManager) {
        mResidualManager->release_residual(layer_idx, stream);
    }
}

void DslRunState::create_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&mSideStream));
    CUDA_CHECK(cudaEventCreate(&mSideStreamEvent));
    CUDA_CHECK(cudaEventCreate(&mAllReduceDone));
}

void DslRunState::release_cuda_resources() noexcept {
    if (mAllReduceDone) {
        cudaEventDestroy(mAllReduceDone);
        mAllReduceDone = nullptr;
    }
    if (mSideStreamEvent) {
        cudaEventDestroy(mSideStreamEvent);
        mSideStreamEvent = nullptr;
    }
    if (mSideStream) {
        cudaStreamDestroy(mSideStream);
        mSideStream = nullptr;
    }
}

void DslRunState::allocate_graph_arrays(int num_layers) {
    mForwardBlockGraphs.resize(static_cast<std::size_t>(num_layers), nullptr);
    mBackwardBlockGraphs.resize(static_cast<std::size_t>(num_layers), {nullptr, nullptr});
    mForwardBlockStackCheckpoints.resize(static_cast<std::size_t>(num_layers));
    mBackwardBlockStackCheckpoints.resize(static_cast<std::size_t>(num_layers));
}

void DslRunState::destroy_cuda_graphs() noexcept {
    for (auto& g : mForwardBlockGraphs) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
    for (auto& arr : mBackwardBlockGraphs) {
        for (auto& g : arr) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }
}

void DslRunState::reset_cuda_graphs() {
    destroy_cuda_graphs();
    // Reset checkpoints to default
    for (auto& cp : mForwardBlockStackCheckpoints) {
        cp = DeviceMemoryStack::Checkpoint{};
    }
    for (auto& arr : mBackwardBlockStackCheckpoints) {
        arr[0] = DeviceMemoryStack::Checkpoint{};
        arr[1] = DeviceMemoryStack::Checkpoint{};
    }
}

void DslRunState::configure_forward_graphs(bool hooked) {
    if (mForwardGraphsHooked == hooked) {
        return;
    }
    // Graph topology changes when hooks are added/removed - must re-capture
    for (auto& g : mForwardBlockGraphs) {
        if (g) {
            (void)cudaGraphExecDestroy(g);
            g = nullptr;
        }
    }
    mForwardGraphsHooked = hooked;
}

void DslRunState::configure_backward_graphs(bool hooked) {
    if (mBackwardGraphsHooked == hooked) {
        return;
    }
    // Graph topology changes when hooks are added/removed - must re-capture
    for (auto& arr : mBackwardBlockGraphs) {
        for (auto& g : arr) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }
    mBackwardGraphsHooked = hooked;
}

} // namespace dsl
