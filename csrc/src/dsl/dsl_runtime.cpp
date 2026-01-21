// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime components implementation.

#include "dsl/dsl_runtime.h"

#include <algorithm>
#include <stdexcept>

#include "kernels/kernels.h"
#include "training/runtime_options.h"
#include "utilities/comm.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

constexpr std::size_t kDefaultStackBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2GB

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
                             const std::shared_ptr<TensorAllocator>& allocator)
    : mAllocator(allocator) {
    if (!mAllocator) {
        throw std::runtime_error("DslParamStore: allocator is null");
    }

    ShapeEnv env = make_shape_env(module, /*B=*/1, /*T=*/1);
    augment_shape_env(env, module.config);

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
}

void DslGradStore::start_micro_step(cudaStream_t stream, int micro_step, int total_steps) {
    (void)total_steps;
    mAccumulate = micro_step > 0;
    if (!mAccumulate) {
        zero_all(stream);
    }
}

void DslGradStore::end_micro_step(cudaStream_t stream, NCCLCommunicator& comm) {
    (void)stream;
    (void)comm;
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
}

DslRunState::DslRunState(const PretrainedConfig& config,
                         const RuntimeOptions& options,
                         int B, int T,
                         const std::shared_ptr<TensorAllocator>& allocator)
    : IRunState(config.clone(), B, T, allocator),
      mAllocator(allocator),
      mRecomputeBlock(false) {
    if (!mAllocator) {
        throw std::runtime_error("DslRunState: allocator is null");
    }

    mActivationDtype = options.ModelType.value_or(config.DType);
    mGradDtype = mActivationDtype;

    // Allocate stack memory (heuristic size).
    mStackBuffer = mAllocator->allocate(
        ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE,
        {static_cast<long>(kDefaultStackBytes)});
    Stack = DeviceMemoryStack(mStackBuffer.Data, kDefaultStackBytes, DeviceId);

    create_cuda_resources();
    allocate_non_block_state(config);
    allocate_simplified_activations(config);
    allocate_simplified_gradients(config);
    allocate_residual_buffers(config);
    allocate_scratch_buffers(config);
}

DslRunState::~DslRunState() {
    release_cuda_resources();
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

    // Output buffer (allocated on stack when needed).
    mNonBlockActivations.output = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B * T, V});

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

    const bool share_ln1 = mRecomputeBlock || false;
    const bool share_ln2 = mRecomputeBlock || false;
    const bool share_qkv = mRecomputeBlock || false;
    const bool share_att = mRecomputeBlock || false;
    const bool share_mlp_up = mRecomputeBlock || false;
    const bool share_swiglu = mRecomputeBlock || false;
    const bool share_residual = mRecomputeBlock;
    const bool ffn_temps_on_stack = mRecomputeBlock;

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

        g.d_att = mAllocator->allocate(dtype, "d_att", kind, {B, T, Hq * D});
        g.d_ln1 = mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
    }
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

    const long bias_scratch_bytes =
        static_cast<long>(get_bias_backward_scratch_size(mGradDtype, static_cast<int>(QKV), DeviceProp));
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
}

void DslRunState::allocate_residual_buffers(const PretrainedConfig& cfg) {
    mResidualManager = std::make_unique<modules::ResidualManager>(
        mAllocator,
        cfg.NumLayers,
        static_cast<int>(B),
        static_cast<int>(T),
        cfg.HiddenSize,
        cfg.DType,
        /*offload_residuals=*/false,
        /*num_residual_buffers=*/2,
        MainStream);
}

void DslRunState::create_cuda_resources() {
    CUDA_CHECK(cudaStreamCreate(&mSideStream));
    CUDA_CHECK(cudaEventCreate(&mSideStreamEvent));
}

void DslRunState::release_cuda_resources() noexcept {
    if (mSideStreamEvent) {
        cudaEventDestroy(mSideStreamEvent);
        mSideStreamEvent = nullptr;
    }
    if (mSideStream) {
        cudaStreamDestroy(mSideStream);
        mSideStream = nullptr;
    }
}

} // namespace dsl
