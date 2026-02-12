// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL run state implementation.

#include "runtime/dsl/dsl_run_state.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "runtime/training/runtime_options.h"
#include "runtime/core/fp8_run_state.h"
#include "runtime/core/fp8_scaling_config.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace dsl {

namespace {
int resolve_mlp_up_factor(const PretrainedConfig& cfg) {
    if (auto* model_cfg = dynamic_cast<const modules::ModelConfig*>(&cfg)) {
        return model_cfg->mlp_up_factor();
    }
    return 2;
}

bool is_hybrid_architecture(const PretrainedConfig& cfg) {
    if (auto* model_cfg = dynamic_cast<const modules::ModelConfig*>(&cfg)) {
        return model_cfg->architecture == modules::ArchitectureType::Hybrid;
    }
    return false;
}

constexpr double kPi = 3.14159265358979323846;

struct RopeInvFreq {
    std::vector<float> inv_freq;
    float attention_scale = 1.0f;
    int dim = 0;
};

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

inline float get_mscale(float scale, float mscale = 1.0f) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * std::log(scale) + 1.0f;
}

std::string to_lower(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

RopeInvFreq compute_rope_inv_freq(const PretrainedConfig& cfg, int head_size, int seq_len) {
    RopeInvFreq out;
    const auto& rope = cfg.Rope;
    int dim = rope.rotary_dim(head_size);
    dim = (dim / 2) * 2;
    out.dim = dim;
    if (dim <= 0) return out;

    const int half = dim / 2;
    out.inv_freq.resize(static_cast<std::size_t>(half), 0.0f);

    const std::string rope_type = rope.rope_type.empty() ? "default" : to_lower(rope.rope_type);
    const double base = static_cast<double>(rope.theta);
    const double factor = static_cast<double>(rope.scaling_factor);

    auto compute_default = [&](double base_val) {
        for (int i = 0; i < half; ++i) {
            const double exponent = (2.0 * i) / static_cast<double>(dim);
            out.inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(1.0 / std::pow(base_val, exponent));
        }
    };

    if (rope_type == "linear") {
        compute_default(base);
        if (factor != 0.0) {
            for (auto& v : out.inv_freq) v = static_cast<float>(v / factor);
        }
        return out;
    }

    if (rope_type == "dynamic") {
        const double max_pos = static_cast<double>(cfg.MaxPositionEmbeddings);
        const double seq = static_cast<double>(std::max(seq_len, cfg.MaxPositionEmbeddings));
        if (dim > 2 && max_pos > 0.0 && factor > 0.0) {
            const double term = (factor * seq / max_pos) - (factor - 1.0);
            if (term > 0.0) {
                const double power = static_cast<double>(dim) / static_cast<double>(dim - 2);
                const double scaled_base = base * std::pow(term, power);
                compute_default(scaled_base);
                return out;
            }
        }
        compute_default(base);
        return out;
    }

    if (rope_type == "yarn") {
        const double max_pos = static_cast<double>(
            rope.original_max_position_embeddings.value_or(cfg.MaxPositionEmbeddings));
        const double beta_fast = static_cast<double>(rope.beta_fast.value_or(32.0f));
        const double beta_slow = static_cast<double>(rope.beta_slow.value_or(1.0f));
        const bool truncate = rope.truncate.value_or(true);

        if (rope.attention_factor) {
            out.attention_scale = *rope.attention_factor;
        } else if (rope.mscale && rope.mscale_all_dim) {
            out.attention_scale = get_mscale(static_cast<float>(factor), *rope.mscale) /
                                  get_mscale(static_cast<float>(factor), *rope.mscale_all_dim);
        } else {
            out.attention_scale = get_mscale(static_cast<float>(factor));
        }

        std::vector<double> pos_freqs(half);
        for (int i = 0; i < half; ++i) {
            const double exponent = (2.0 * i) / static_cast<double>(dim);
            pos_freqs[static_cast<std::size_t>(i)] = std::pow(base, exponent);
        }
        std::vector<double> inv_freq_extrapolation(half);
        std::vector<double> inv_freq_interpolation(half);
        for (int i = 0; i < half; ++i) {
            const double pf = pos_freqs[static_cast<std::size_t>(i)];
            inv_freq_extrapolation[static_cast<std::size_t>(i)] = 1.0 / pf;
            inv_freq_interpolation[static_cast<std::size_t>(i)] = (factor > 0.0) ? (1.0 / (factor * pf)) : (1.0 / pf);
        }

        auto find_correction_dim = [&](double num_rot, double dim_val, double base_val, double max_pos_val) {
            return (dim_val * std::log(max_pos_val / (num_rot * 2.0 * kPi))) / (2.0 * std::log(base_val));
        };
        auto find_correction_range = [&](double low_rot, double high_rot, double dim_val,
                                         double base_val, double max_pos_val, bool truncate_val) {
            double low = find_correction_dim(low_rot, dim_val, base_val, max_pos_val);
            double high = find_correction_dim(high_rot, dim_val, base_val, max_pos_val);
            if (truncate_val) {
                low = std::floor(low);
                high = std::ceil(high);
            }
            low = std::max(low, 0.0);
            high = std::min(high, dim_val - 1.0);
            return std::pair<double, double>(low, high);
        };

        auto [low, high] = find_correction_range(beta_fast, beta_slow, static_cast<double>(dim), base, max_pos, truncate);
        if (low == high) {
            high += 0.001;
        }

        for (int i = 0; i < half; ++i) {
            const double linear = (static_cast<double>(i) - low) / (high - low);
            const double ramp = clampf(static_cast<float>(linear), 0.0f, 1.0f);
            const double extrap_factor = 1.0 - ramp;
            const double inv_val = inv_freq_interpolation[static_cast<std::size_t>(i)] * (1.0 - extrap_factor) +
                                   inv_freq_extrapolation[static_cast<std::size_t>(i)] * extrap_factor;
            out.inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(inv_val);
        }
        return out;
    }

    if (rope_type == "longrope") {
        const int original_max = rope.original_max_position_embeddings_config.value_or(cfg.MaxPositionEmbeddings);
        double attention_factor = rope.attention_factor.value_or(0.0f);
        double factor_for_attn = factor;
        if (original_max > 0 && rope.original_max_position_embeddings_config) {
            factor_for_attn = static_cast<double>(cfg.MaxPositionEmbeddings) / static_cast<double>(original_max);
        }
        if (attention_factor <= 0.0) {
            if (factor_for_attn <= 1.0) {
                attention_factor = 1.0;
            } else if (original_max > 0) {
                attention_factor = std::sqrt(1.0 + std::log(factor_for_attn) / std::log(static_cast<double>(original_max)));
            } else {
                attention_factor = 1.0;
            }
        }
        out.attention_scale = static_cast<float>(attention_factor);

        const bool use_long = (seq_len > original_max);
        const auto& factors = use_long ? rope.long_factor : rope.short_factor;
        std::vector<float> ext_factors;
        if (factors.size() == static_cast<std::size_t>(half)) {
            ext_factors.assign(factors.begin(), factors.end());
        } else {
            ext_factors.assign(static_cast<std::size_t>(half), 1.0f);
        }

        for (int i = 0; i < half; ++i) {
            const double exponent = (2.0 * i) / static_cast<double>(dim);
            const double pf = std::pow(base, exponent);
            const double ext = static_cast<double>(ext_factors[static_cast<std::size_t>(i)]);
            out.inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(1.0 / (ext * pf));
        }
        return out;
    }

    if (rope_type == "llama3") {
        if (!rope.low_freq_factor || !rope.high_freq_factor) {
            compute_default(base);
            return out;
        }
        const double factor_llama = factor;
        const double low_freq_factor = static_cast<double>(*rope.low_freq_factor);
        const double high_freq_factor = static_cast<double>(*rope.high_freq_factor);
        const int old_ctx = rope.original_max_position_embeddings.value_or(cfg.MaxPositionEmbeddings);
        if (old_ctx <= 0 || factor_llama <= 0.0 || high_freq_factor == low_freq_factor) {
            compute_default(base);
            return out;
        }

        compute_default(base);
        const double low_freq_wavelen = static_cast<double>(old_ctx) / low_freq_factor;
        const double high_freq_wavelen = static_cast<double>(old_ctx) / high_freq_factor;

        for (int i = 0; i < half; ++i) {
            const double inv = static_cast<double>(out.inv_freq[static_cast<std::size_t>(i)]);
            const double wavelen = 2.0 * kPi / inv;
            double inv_llama = (wavelen > low_freq_wavelen) ? (inv / factor_llama) : inv;
            const double smooth_factor = (static_cast<double>(old_ctx) / wavelen - low_freq_factor) /
                                         (high_freq_factor - low_freq_factor);
            const double smoothed = (1.0 - smooth_factor) * inv_llama / factor_llama + smooth_factor * inv_llama;
            const bool is_medium = !(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen);
            const double final_inv = is_medium ? smoothed : inv_llama;
            out.inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(final_inv);
        }
        return out;
    }

    // default
    compute_default(base);
    return out;
}

template<typename T>
inline T rope_cast(float v) {
    return static_cast<T>(v);
}

template<>
inline nv_bfloat16 rope_cast<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<typename T>
void fill_rope_freqs(std::vector<T>& out, const RopeInvFreq& params, int head_size, int max_seq_len) {
    if (params.dim <= 0 || params.inv_freq.empty()) return;
    const int dim = params.dim;
    const int half = dim / 2;
    const std::size_t stride = static_cast<std::size_t>(dim);
    std::fill(out.begin(), out.end(), T{});
    for (int t = 0; t < max_seq_len; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * stride;
        for (int i = 0; i < half; ++i) {
            const float angle = static_cast<float>(t) * params.inv_freq[static_cast<std::size_t>(i)];
            const float c = std::cos(angle) * params.attention_scale;
            const float s = std::sin(angle) * params.attention_scale;
            out[base + static_cast<std::size_t>(2 * i)] = rope_cast<T>(c);
            out[base + static_cast<std::size_t>(2 * i + 1)] = rope_cast<T>(s);
        }
    }
}
}  // namespace

DslRunState::DslRunState(const PretrainedConfig& config,
                         const DslRuntimeConfig& runtime_config,
                         const RuntimeOptions& options,
                         int B, int T,
                         const std::shared_ptr<TensorAllocator>& allocator,
                         bool lora_only_mode,
                         bool prequantized,
                         std::size_t stack_bytes,
                         bool allocate_stack,
                         const ActivationLayoutIR* activation_layout)
    : IRunState(config.clone(), B, T, allocator),
      mAllocator(allocator),
      mRuntimeConfig(runtime_config),
      mRecomputeLevel(options.Recompute),
      mLoraOnlyMode(lora_only_mode),
      mPrequantized(prequantized),
      mNumLayers(config.NumLayers),
      mPerLayerGraphsEnabled(options.UseCudaGraphs) {
    if (!mAllocator) {
        throw std::runtime_error("DslRunState: allocator is null");
    }
    if (activation_layout) {
        mSlotRegistry.init_from_layout(*activation_layout);
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
    if (options.LMHeadChunks < 1) {
        throw std::runtime_error("lmhead_chunks must be >= 1");
    }
    if (options.AttBwdChunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    mLMHeadChunks = options.LMHeadChunks;
    mAttnBwdChunks = options.AttBwdChunks;
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
    build_activation_grad_zero_segments();
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
    const long lmhead_chunks = static_cast<long>(mLMHeadChunks);
    const long out_size = (B * T) / lmhead_chunks;
    mNonBlockActivations.output = mAllocator->allocate(dtype, "output", EAllocationType::ON_DEVICE, {out_size, V});

    // RoPE frequencies (if not using fused RoPE).
    const int max_seq_len = static_cast<int>(T);
    if (max_seq_len > 0) {
        const int head_size = cfg.head_size();
        const RopeInvFreq rope_params = compute_rope_inv_freq(cfg, head_size, max_seq_len);
        if (dtype == ETensorDType::BF16) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            fill_rope_freqs(freq_cpu, rope_params, head_size, max_seq_len);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data, freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            mNonBlockActivations.freq_cis = mAllocator->allocate(
                dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            fill_rope_freqs(freq_cpu, rope_params, head_size, max_seq_len);
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
    // Skip d_embeddings allocation in LoRA-only mode - embedding backward is skipped entirely
    if (!mLoraOnlyMode) {
        mNonBlockGradients.d_embeddings = mAllocator->allocate(mGradDtype, "d_embeddings", EAllocationType::ON_DEVICE, {B, T, C});
    }
}

void DslRunState::allocate_simplified_activations(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    // Derive attention output channels directly from Hq * head_size to avoid
    // config mismatches (e.g. attn_out_channels not reflecting GQA layout).
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;
    const long NumExperts = mRuntimeConfig.num_experts;
    const long TopK = (mRuntimeConfig.num_experts_per_tok > 0) ? mRuntimeConfig.num_experts_per_tok : 1;
    const long MoeM = (mRuntimeConfig.moe_intermediate_size > 0)
        ? mRuntimeConfig.moe_intermediate_size
        : cfg.IntermediateSize;
    const long MoeMUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * MoeM;
    const bool use_qk_norm = mRuntimeConfig.use_qk_norm;

    const auto dtype = mActivationDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    // =========================================================================
    // DSL-driven activation sharing
    // =========================================================================
    // The sharing policy for each slot is now defined in the Python DSL
    // (e.g., surogate/dsl/blocks/qwen3.py) via the share_policy attribute.
    // This eliminates hardcoded rules and makes the DSL the single source of truth.
    //
    // The TensorSlotRegistry.should_share() method evaluates the share_policy:
    // - PerLayer: Never share (always allocate per-layer)
    // - WhenRecomputed: Share when recompute is enabled and will_recompute() is true
    // - AlwaysShare: Always share across layers
    // - FFTShare: Share only in FFT mode (not LoRA)
    // - LoRAShare: Share only in LoRA mode (not FFT)
    //
    const bool recompute_enabled = mRecomputeLevel >= RecomputeLevel::Enabled;
    const bool lora_only = mLoraOnlyMode;
    const bool has_layout = mSlotRegistry.has_dsl_layout();

    // Helper to determine if a slot should be shared
    auto should_share_slot = [&](const char* name) -> bool {
        if (has_layout) {
            return mSlotRegistry.should_share(name, lora_only, recompute_enabled);
        }
        // Legacy fallback when no DSL layout is available
        return recompute_enabled && mSlotRegistry.will_recompute(name, lora_only);
    };

    // Query the DSL for sharing decisions
    const bool share_ln1 = should_share_slot("ln1");
    const bool share_ln2 = should_share_slot("ln2");
    const bool share_qkv = should_share_slot("qkv");
    const bool share_att = should_share_slot("att");
    const bool share_att_out = should_share_slot("att_out");
    const bool share_mlp_up = should_share_slot("mlp_up");
    const bool share_swiglu = should_share_slot("swiglu");
    const bool share_residual_intermediates = should_share_slot("res_att");
    const bool share_mlp_down = should_share_slot("mlp_down");
    const bool share_qk_rstd = use_qk_norm && should_share_slot("q_rstd");

    // FFN temps: Use stack-backed temps only in LoRA mode with recompute enabled.
    // This keeps FFT behavior unchanged while enabling QLoRA memory savings.
    const bool ffn_temps_on_stack = recompute_enabled && lora_only;
    mFfnTempsOnStack = ffn_temps_on_stack;
    if (mStackSimulate && ffn_temps_on_stack) {
        const long mlp_up_bytes = B * T * MUp * get_dtype_size(dtype);
        const long swiglu_bytes = B * T * M * get_dtype_size(dtype);
        auto* sim_mlp_up = Stack.allocate(static_cast<std::size_t>(mlp_up_bytes), "mlp_up_simulate");
        auto* sim_swiglu = Stack.allocate(static_cast<std::size_t>(swiglu_bytes), "swiglu_simulate");
        Stack.free(sim_swiglu);
        Stack.free(sim_mlp_up);
    }

    Tensor shared_ln1{}, shared_ln2{}, shared_qkv{}, shared_att{}, shared_att_out{}, shared_lse{};
    Tensor shared_q_rstd{}, shared_k_rstd{};
    Tensor shared_mlp_up{}, shared_swiglu{}, shared_residual_att{}, shared_mlp_down{};

    if (share_ln1) shared_ln1 = mAllocator->allocate(dtype, "ln1_shared", kind, {B, T, C});
    if (share_ln2) shared_ln2 = mAllocator->allocate(dtype, "ln2_shared", kind, {B, T, C});
    if (share_qkv) {
        shared_qkv = mAllocator->allocate(dtype, "qkv_shared", kind, {B, T, QKV});
    }
    if (share_qk_rstd && use_qk_norm) {
        shared_q_rstd = mAllocator->allocate(ETensorDType::FP32, "q_rstd_shared", kind, {B, T, Hq});
        shared_k_rstd = mAllocator->allocate(ETensorDType::FP32, "k_rstd_shared", kind, {B, T, Hkv});
    }
    // LSE sharing: only in lora_only mode. FFT needs per-layer LSE for bit-exact gradients.
    if (share_att) shared_lse = mAllocator->allocate(ETensorDType::FP32, "lse_shared", kind, {B, Hq, T});
    if (share_att) {
        shared_att = mAllocator->allocate(dtype, "att_shared", kind, {B, T, AttnDim});
    }
    if (share_att_out) {
        shared_att_out = mAllocator->allocate(dtype, "att_out_shared", kind, {B, T, C});
    }
    // att_out sharing is handled by share_att when recompute is enabled.
    if (share_mlp_up && !ffn_temps_on_stack) shared_mlp_up = mAllocator->allocate(dtype, "mlp_up_shared", kind, {B, T, MUp});
    if (share_swiglu && !ffn_temps_on_stack) shared_swiglu = mAllocator->allocate(dtype, "swiglu_shared", kind, {B, T, M});
    if (share_residual_intermediates) {
        shared_residual_att = mAllocator->allocate(dtype, "residual_att_shared", kind, {B, T, C});
    }
    if (share_mlp_down) {
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
            acts.q_rstd = share_qk_rstd ? shared_q_rstd
                                        : mAllocator->allocate(ETensorDType::FP32, "q_rstd", kind, {B, T, Hq});
            acts.k_rstd = share_qk_rstd ? shared_k_rstd
                                        : mAllocator->allocate(ETensorDType::FP32, "k_rstd", kind, {B, T, Hkv});
        } else {
            acts.q_rstd = {};
            acts.k_rstd = {};
        }

        acts.qkv = share_qkv ? shared_qkv : mAllocator->allocate(dtype, "qkv", kind, {B, T, QKV});
        // In FFT mode with recompute enabled, we need BOTH qkv (original) and qkv_rope (transformed)
        // because the qk_norm_rope backward kernel expects qkv_rope (the transformed output).
        // The kernel applies inverse RoPE internally to compute the gradient.
        // Without separate buffers, the in-place forward transform overwrites the qkv buffer,
        // so we can't distinguish between original qkv and transformed qkv_rope.
        // In LoRA mode, qkv_rope has lora_only policy so it gets recomputed during backward.
        // In no-recompute mode, the qkv buffer contains qkv_rope after forward (in-place transform).
        const bool need_separate_qkv_rope = recompute_enabled && !lora_only && use_qk_norm;
        acts.qkv_rope = need_separate_qkv_rope
            ? mAllocator->allocate(dtype, "qkv_rope", kind, {B, T, QKV})
            : Tensor{};

        acts.lse = share_att ? shared_lse
                             : mAllocator->allocate(ETensorDType::FP32, "lse", kind, {B, Hq, T});
        acts.att = share_att ? shared_att : mAllocator->allocate(dtype, "att", kind, {B, T, AttnDim});
        // When recompute is enabled, att_out can be shared across layers.
        acts.att_out = share_att_out ? shared_att_out : mAllocator->allocate(dtype, "att_out", kind, {B, T, C});

        // residual_att can be shared when recompute_block=true
        acts.residual_att = share_residual_intermediates ? shared_residual_att
                                                          : mAllocator->allocate(dtype, "residual_att", kind, {B, T, C});

        if (ffn_temps_on_stack) {
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
        } else {
            acts.mlp_up = share_mlp_up ? shared_mlp_up : mAllocator->allocate(dtype, "mlp_up", kind, {B, T, MUp});
            acts.swiglu = share_swiglu ? shared_swiglu : mAllocator->allocate(dtype, "swiglu", kind, {B, T, M});
        }

        acts.mlp_down = share_mlp_down ? shared_mlp_down
                                       : mAllocator->allocate(dtype, "mlp_down", kind, {B, T, C});

        if (NumExperts > 0) {
            const long num_tokens = B * T;
            const long total_tokens = num_tokens * TopK;
            acts.router_logits = mAllocator->allocate(dtype, "router_logits", kind, {num_tokens, NumExperts});
            acts.router_probs = mAllocator->allocate(dtype, "router_probs", kind, {num_tokens, NumExperts});
            acts.routing_weights = mAllocator->allocate(dtype, "routing_weights", kind, {num_tokens, TopK});
            acts.routing_indices = mAllocator->allocate(ETensorDType::INT32, "routing_indices", kind, {num_tokens, TopK});
            acts.permuted_input = mAllocator->allocate(dtype, "permuted_input", kind, {total_tokens, C});
            acts.scatter_indices = mAllocator->allocate(ETensorDType::INT32, "scatter_indices", kind, {total_tokens});
            acts.expert_gate_up = mAllocator->allocate(dtype, "expert_gate_up", kind, {total_tokens, MoeMUp});
            acts.expert_act = mAllocator->allocate(dtype, "expert_act", kind, {total_tokens, MoeM});
            acts.expert_down = mAllocator->allocate(dtype, "expert_down", kind, {total_tokens, C});
            acts.moe_out = view_tensor(acts.mlp_down, {num_tokens, C});
        } else {
            acts.router_logits = {};
            acts.router_probs = {};
            acts.routing_weights = {};
            acts.routing_indices = {};
            acts.permuted_input = {};
            acts.scatter_indices = {};
            acts.expert_gate_up = {};
            acts.expert_act = {};
            acts.expert_down = {};
            acts.moe_out = {};
        }
    }

    // Allocate temporary buffers for recomputation
    // This prevents overwriting saved values when recomputing forward activations
    if (recompute_enabled) {
        mRecomputeRstd = mAllocator->allocate(ETensorDType::FP32, "recompute_rstd", kind, {B, T});
        // LSE buffer for attention recomputation - same shape as acts.lse [B, Hq, T]
        mRecomputeLSE = mAllocator->allocate(ETensorDType::FP32, "recompute_lse", kind, {B, Hq, T});
    }
}

void DslRunState::allocate_simplified_gradients(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    // Derive attention output channels directly from Hq * head_size to avoid
    // config mismatches (e.g. attn_out_channels not reflecting GQA layout).
    const long AttnDim = Hq * D;
    const long QKV = D * (Hq + 2 * Hkv);
    const long M = cfg.IntermediateSize;
    const long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;

    const auto dtype = mGradDtype;
    const auto kind = EAllocationType::ON_DEVICE;

    // =========================================================================
    // Gradient buffer sharing
    // =========================================================================
    // Unlike activations, gradient sharing is more complex due to:
    // 1. Backward hooks in LoRA mode that rely on per-layer gradients
    // 2. Different backward graph structures in FFT vs LoRA modes
    // 3. Risk of gradient corruption when sharing across layers
    //
    // For now, we keep gradient sharing disabled (per-layer allocation).
    // In the future, this could be made DSL-driven via Gradient share_policy,
    // but the benefit is smaller than activation sharing (gradients are typically
    // computed and consumed immediately, not stored across layers).
    //
    const bool recompute_enabled = mRecomputeLevel >= RecomputeLevel::Enabled;
    const bool share_grads = false;
    const bool share_res_ffn = false;
    const bool share_mlp_down = false;
    const bool large_bwd_temps_on_stack = recompute_enabled;

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

    // Allocate shared gradient buffers if recompute_block is enabled
    if (share_grads && !mSharedDResAtt.Data) {
        if (share_res_ffn) {
            mSharedDResFFN[0] = mAllocator->allocate(dtype, "d_res_ffn_a", kind, {B, T, C});
            mSharedDResFFN[1] = mAllocator->allocate(dtype, "d_res_ffn_b", kind, {B, T, C});
        }
        if (share_mlp_down) {
            mSharedDMlpDown[0] = mAllocator->allocate(dtype, "d_mlp_down_a", kind, {B, T, C});
            mSharedDMlpDown[1] = mAllocator->allocate(dtype, "d_mlp_down_b", kind, {B, T, C});
        }
        mSharedDResAtt = mAllocator->allocate(dtype, "d_res_att_shared", kind, {B, T, C});
        if (is_hybrid_architecture(cfg)) {
            mSharedDAttOut = mAllocator->allocate(dtype, "d_att_out_shared", kind, {B, T, C});
        } else {
            mSharedDAttOut = mSharedDResAtt;
        }
        mSharedDLn2 = mAllocator->allocate(dtype, "d_ln2_shared", kind, {B, T, C});
        mSharedDAtt = mAllocator->allocate(dtype, "d_att_shared", kind, {B, T, AttnDim});
        mSharedDLn1 = mAllocator->allocate(dtype, "d_ln1_shared", kind, {B, T, C});
    }

    const bool hybrid = is_hybrid_architecture(cfg);

    mSimplifiedGradients.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& g = mSimplifiedGradients[i];

        // d_res_ffn uses alternating buffers (i % 2) to avoid aliasing in adjacent layers
        g.d_res_ffn = share_res_ffn ? mSharedDResFFN[static_cast<std::size_t>(i % 2)]
                                    : mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = share_grads ? mSharedDResAtt : mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        // In standard transformers res_att = x + att_out, so the gradients are equal.
        // Only hybrid architectures (Nemotron-H) need separate storage.
        g.d_att_out = hybrid ? (share_grads ? mSharedDAttOut : mAllocator->allocate(dtype, "d_att_out", kind, {B, T, C}))
                             : g.d_res_att;
        g.d_ln2 = share_grads ? mSharedDLn2 : mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});

        if (large_bwd_temps_on_stack) {
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, MUp});
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, M});
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, QKV});
        } else {
            g.d_mlp_up = mAllocator->allocate(dtype, "d_mlp_up", kind, {B, T, MUp});
            g.d_swiglu = mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, M});
            g.d_qkv = mAllocator->allocate(dtype, "d_qkv", kind, {B, T, QKV});
        }

        // d_mlp_down uses alternating buffers (i % 2) to avoid aliasing in adjacent layers
        g.d_mlp_down = share_mlp_down ? mSharedDMlpDown[static_cast<std::size_t>(i % 2)]
                                      : mAllocator->allocate(dtype, "d_mlp_down", kind, {B, T, C});
        g.d_att = share_grads ? mSharedDAtt : mAllocator->allocate(dtype, "d_att", kind, {B, T, AttnDim});
        g.d_ln1 = share_grads ? mSharedDLn1 : mAllocator->allocate(dtype, "d_ln1", kind, {B, T, C});
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
        dst.d_att_out.Data = src.d_att_out.Data;
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

void DslRunState::zero_activation_gradients(cudaStream_t stream) {
    // Zero activation gradient buffers to prevent stale gradients from accumulating.
    // Use a single kernel launch over a precomputed (ptr, bytes) list to reduce graph-node
    // overhead vs many separate cudaMemsetAsync calls.
    if (mActGradZeroCount > 0 && mActGradZeroPtrs.Data && mActGradZeroSizes.Data) {
        zero_device_segments(reinterpret_cast<const std::uint64_t*>(mActGradZeroPtrs.Data),
                             reinterpret_cast<const std::uint64_t*>(mActGradZeroSizes.Data),
                             mActGradZeroCount,
                             stream);
        return;
    }

    // Fallback: should not normally happen.
    for (std::size_t i = 0; i < mSimplifiedGradients.size(); ++i) {
        auto& g = mSimplifiedGradients[i];
        if (i < mSimplifiedGradients.size() - 1 && g.d_res_ffn.Data) fill_zero(g.d_res_ffn, stream);
        if (g.d_res_att.Data) fill_zero(g.d_res_att, stream);
        if (g.d_att_out.Data) fill_zero(g.d_att_out, stream);
    }
}

void DslRunState::build_activation_grad_zero_segments() {
    mActGradZeroPtrs = {};
    mActGradZeroSizes = {};
    mActGradZeroCount = 0;

    if (!mAllocator) {
        return;
    }
    if (mSimplifiedGradientsBase.empty()) {
        return;
    }

    std::vector<std::uint64_t> ptrs;
    std::vector<std::uint64_t> sizes;
    ptrs.reserve(mSimplifiedGradientsBase.size() * 8);
    sizes.reserve(mSimplifiedGradientsBase.size() * 8);

    std::unordered_set<std::byte*> seen;
    seen.reserve(mSimplifiedGradientsBase.size() * 8);

    auto add = [&](const Tensor& t) {
        if (!t.Data) return;
        const std::size_t bytes = static_cast<std::size_t>(t.bytes());
        if (bytes == 0) return;
        if (!seen.insert(t.Data).second) return;
        ptrs.push_back(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t.Data)));
        sizes.push_back(static_cast<std::uint64_t>(bytes));
    };

    const std::size_t n_layers = mSimplifiedGradientsBase.size();
    for (std::size_t i = 0; i < n_layers; ++i) {
        const auto& g = mSimplifiedGradientsBase[i];
        // d_res_ffn for the last layer is zeroed separately (it receives the loss gradient).
        if (i + 1 < n_layers) {
            add(g.d_res_ffn);
        }
        // Residual gradients can be used as accumulation targets (multiple branches).
        // Other activation gradients are expected to be overwritten (beta=0 / memcpy) within
        // the backward graph and don't need blanket zeroing.
        add(g.d_res_att);
        add(g.d_att_out);
    }

    mActGradZeroCount = static_cast<int>(ptrs.size());
    if (mActGradZeroCount <= 0) {
        return;
    }

    const long bytes = static_cast<long>(static_cast<std::size_t>(mActGradZeroCount) * sizeof(std::uint64_t));
    mActGradZeroPtrs = mAllocator->allocate(ETensorDType::BYTE, "dsl_act_grad_zero_ptrs",
                                            EAllocationType::ON_DEVICE, {bytes});
    mActGradZeroSizes = mAllocator->allocate(ETensorDType::BYTE, "dsl_act_grad_zero_sizes",
                                             EAllocationType::ON_DEVICE, {bytes});

    CUDA_CHECK(cudaMemcpy(mActGradZeroPtrs.Data, ptrs.data(),
                          static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mActGradZeroSizes.Data, sizes.data(),
                          static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
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
    const long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;

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
    const long C_attn = static_cast<long>(cfg.attn_out_channels());

    const long rmsnorm_scratch_bytes = static_cast<long>(get_rmsnorm_backward_scratch_size(static_cast<int>(C), DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(
        ETensorDType::BYTE, "rmsnorm_scratch", EAllocationType::ON_DEVICE, {rmsnorm_scratch_bytes});

    const long M = cfg.IntermediateSize;
    const long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;
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

    const long BT = B * T;
    mScratch.cross_entropy_dloss = mAllocator->allocate(
        ETensorDType::FP32, "cross_entropy_dloss", EAllocationType::ON_DEVICE, {BT});
    mScratch.cross_entropy_logsumexp = mAllocator->allocate(
        ETensorDType::FP32, "cross_entropy_logsumexp", EAllocationType::ON_DEVICE, {BT});
    const int n_chunks = static_cast<int>(
        (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE);
    if (n_chunks > 1) {
        mScratch.cross_entropy_chunk_logsumexp = mAllocator->allocate(
            ETensorDType::FP32, "cross_entropy_chunk_logsumexp", EAllocationType::ON_DEVICE,
            {BT, n_chunks});
    }

    // Encoder backward scratch buffers - skip in LoRA-only mode since embedding backward is skipped entirely
    if (!mLoraOnlyMode) {
        const long group_width = static_cast<long>(16 / get_dtype_size(mGradDtype) * 32);
        const long num_c_groups = (C + group_width - 1) / group_width;
        mScratch.encoder_bwd_scratch = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_scratch", EAllocationType::ON_DEVICE, {B, T, num_c_groups * 5});
        mScratch.encoder_bwd_indices = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_indices", EAllocationType::PINNED, {B, T, num_c_groups});
        mScratch.encoder_bwd_info = mAllocator->allocate(
            ETensorDType::INT32, "encoder_bwd_info", EAllocationType::PINNED, {B, T, 4 * num_c_groups});
    }

    const int attn_chunks = mAttnBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const long attn_ws_batch_size =
        (attn_chunks == 1) ? B : div_exact(B, static_cast<long>(attn_chunks));
    const bool cudnn_ok = (D > 0 && Hq > 0 && Hkv > 0 && (D % 8 == 0) && D <= 128);
    if (cudnn_ok) {
        const long cudnn_ws_size = static_cast<long>(
            cudnn_get_workspace_size(static_cast<int>(attn_ws_batch_size), static_cast<int>(T),
                                     static_cast<int>(Hq), static_cast<int>(Hkv),
                                     static_cast<int>(D), CudnnHandle));
        // Pre-allocate cudnn_workspace using the persistent allocator to avoid overlap with
        // stack-allocated gradient buffers. The workspace is large (~192MB) and if allocated
        // from the temp stack, checkpoint restores during backward can cause it to be reallocated
        // in a region that overlaps with gradient buffers.
        mScratch.cudnn_workspace = mAllocator->allocate(
            ETensorDType::BYTE, "cudnn_workspace", EAllocationType::ON_DEVICE, {cudnn_ws_size});
    } else {
        // Leave an empty descriptor; attention ops will fail later if invoked with invalid head size.
        mScratch.cudnn_workspace = Tensor::empty(ETensorDType::BYTE, {0});
    }

    // Note: Stack simulation no longer needed for workspace since it's persistently allocated
    if (mStackSimulate) {
        if (mRecomputeLevel >= RecomputeLevel::Enabled) {
            const long d_qkv_bytes = B * T * QKV * get_dtype_size(mGradDtype);
            auto* simulated_d_qkv = Stack.allocate(static_cast<std::size_t>(d_qkv_bytes), "d_qkv_simulate");
            Stack.free(simulated_d_qkv);
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
    CUBLAS_CHECK(cublasCreate(&mCublasHandle));
    CUBLAS_CHECK(cublasSetMathMode(mCublasHandle, CUBLAS_TF32_TENSOR_OP_MATH));
}

void DslRunState::release_cuda_resources() noexcept {
    if (mCublasHandle) {
        cublasDestroy(mCublasHandle);
        mCublasHandle = nullptr;
    }
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
