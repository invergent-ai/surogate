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
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "runtime/attention/attention_backend.h"
#include "runtime/executor/graph_executor_utils.h"
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

RopeInvFreq compute_rope_inv_freq(const PretrainedConfig& cfg, const RoPEConfig& rope, int head_size, int seq_len);

RopeInvFreq compute_rope_inv_freq(const PretrainedConfig& cfg, int head_size, int seq_len) {
    return compute_rope_inv_freq(cfg, cfg.Rope, head_size, seq_len);
}

RopeInvFreq compute_rope_inv_freq(const PretrainedConfig& cfg, const RoPEConfig& rope, int head_size, int seq_len) {
    RopeInvFreq out;
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
            for (auto& v : out.inv_freq)
                v = static_cast<float>(v / factor);
        }
        return out;
    }

    if (rope_type == "proportional") {
        const double rope_proportion = static_cast<double>(rope.partial_factor);
        const int rope_angles = static_cast<int>((rope_proportion * static_cast<double>(head_size)) / 2.0);
        // HF Gemma4 proportional RoPE returns a full head-width frequency
        // vector with trailing zero frequencies, then applies rotate_half()
        // across the full head. This is not equivalent to rotating a
        // contiguous prefix only.
        out.dim = head_size;
        out.inv_freq.assign(static_cast<std::size_t>(std::max(0, head_size / 2)), 0.0f);
        for (int i = 0; i < rope_angles; ++i) {
            const double exponent = (2.0 * i) / static_cast<double>(head_size);
            out.inv_freq[static_cast<std::size_t>(i)] = static_cast<float>(1.0 / std::pow(base, exponent));
        }
        if (factor != 0.0) {
            for (auto& v : out.inv_freq)
                v = static_cast<float>(v / factor);
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
        const double max_pos =
            static_cast<double>(rope.original_max_position_embeddings.value_or(cfg.MaxPositionEmbeddings));
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
        auto find_correction_range = [&](double low_rot,
                                         double high_rot,
                                         double dim_val,
                                         double base_val,
                                         double max_pos_val,
                                         bool truncate_val) {
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

        auto [low, high] =
            find_correction_range(beta_fast, beta_slow, static_cast<double>(dim), base, max_pos, truncate);
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
                attention_factor =
                    std::sqrt(1.0 + std::log(factor_for_attn) / std::log(static_cast<double>(original_max)));
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
            const double smooth_factor =
                (static_cast<double>(old_ctx) / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
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

template <typename T>
inline T rope_cast(float v) {
    return static_cast<T>(v);
}

template <>
inline nv_bfloat16 rope_cast<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
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
                         int B,
                         int T,
                         const std::shared_ptr<TensorAllocator>& allocator,
                         bool lora_only_mode,
                         bool prequantized,
                         std::size_t stack_bytes,
                         const ActivationLayoutIR* activation_layout)
    : IRunState(config.clone(), B, T, allocator),
      mAllocator(allocator),
      mRuntimeConfig(runtime_config),
      mRecomputeLevel(options.Recompute),
      mLoraOnlyMode(lora_only_mode),
      mPrequantized(prequantized),
      mCpuTraining(options.CpuTraining),
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

    // Stack is always device-backed now; `dsl::required_stack_bytes` is the
    // single source of truth for the size (see dsl_model.cpp). Historically
    // a sizing pass ran first with a null-backed dummy stack — that pass is
    // gone, replaced by `BufferPlan::plan_stack_peak_bytes`.
    const std::size_t stack_capacity = (stack_bytes > 0) ? stack_bytes : kDefaultStackBytes;
    mStackBuffer = mAllocator->allocate(ETensorDType::BYTE,
                                        "dsl_stack",
                                        EAllocationType::ON_DEVICE,
                                        {static_cast<long>(stack_capacity)});
    Stack = DeviceMemoryStack(mStackBuffer.Data, stack_capacity, DeviceId);

    create_cuda_resources();

    // Build the buffer plan *before* any allocation: captures all sharing,
    // sizing, and stack-temp decisions in one place so allocate_simplified_*
    // can be a mechanical walk over the plan.
    mBufferPlan = BufferPlan::build(config,
                                    mRuntimeConfig,
                                    options,
                                    mSlotRegistry,
                                    mLoraOnlyMode,
                                    static_cast<long>(B),
                                    static_cast<long>(T),
                                    mActivationDtype,
                                    mGradDtype);

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
    if (mMoEStatsDevice) {
        (void)cudaFree(mMoEStatsDevice);
        mMoEStatsDevice = nullptr;
    }
    if (mMoEStatsHost) {
        (void)cudaFreeHost(mMoEStatsHost);
        mMoEStatsHost = nullptr;
    }
    // Free any phase-arena Stack buffer we adopted ownership of.
    if (mOwnedExternalStack) {
        (void)cudaFree(mOwnedExternalStack);
        mOwnedExternalStack = nullptr;
        mOwnedExternalStackBytes = 0;
    }
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
}

void DslRunState::rebase_stack_to_external(std::byte* ptr, std::size_t bytes) {
    if (!ptr || bytes == 0) {
        throw std::runtime_error("DslRunState::rebase_stack_to_external: invalid buffer");
    }
    if (Stack.bytes_used() != 0) {
        throw std::runtime_error("DslRunState::rebase_stack_to_external: Stack is not empty");
    }
    Stack = DeviceMemoryStack(ptr, bytes, DeviceId);
}

void DslRunState::unbind_external_stack() {
    if (Stack.bytes_used() != 0) {
        throw std::runtime_error("DslRunState::unbind_external_stack: Stack is not empty");
    }
    if (!mStackBuffer.Data) {
        throw std::runtime_error("DslRunState::unbind_external_stack: no original stack buffer");
    }
    Stack = DeviceMemoryStack(mStackBuffer.Data, static_cast<std::size_t>(mStackBuffer.bytes()), DeviceId);
}

void DslRunState::free_allocator_stack_buffer() {
    if (!mStackBuffer.Data) return;
    mAllocator->free(mStackBuffer);
    mStackBuffer = Tensor{};
}

void DslRunState::adopt_external_stack(std::byte* ptr, std::size_t bytes) {
    if (!ptr || bytes == 0) {
        throw std::runtime_error("DslRunState::adopt_external_stack: invalid buffer");
    }
    // Free any previously-adopted buffer first.
    if (mOwnedExternalStack) {
        cudaFree(mOwnedExternalStack);
        mOwnedExternalStack = nullptr;
        mOwnedExternalStackBytes = 0;
    }
    mOwnedExternalStack = ptr;
    mOwnedExternalStackBytes = bytes;
}

void DslRunState::resize_stack_to(long new_size_bytes) {
    if (new_size_bytes <= 0) {
        throw std::runtime_error("DslRunState::resize_stack_to: non-positive size");
    }
    // Free the old buffer *before* requesting the new one. The TensorAllocator
    // retains tracked allocations until its destructor runs, so a naive
    // allocate+swap would leak the pre-resize buffer to the end of the run
    // and briefly hold 2x VRAM on tight-memory setups.
    mAllocator->free(mStackBuffer);
    Stack = DeviceMemoryStack();
    Tensor new_stack =
        mAllocator->allocate(ETensorDType::BYTE, "dsl_stack", EAllocationType::ON_DEVICE, {new_size_bytes});
    set_stack_buffer(std::move(new_stack));
}

long DslRunState::shrink_stack_to_high_water_mark(long safety_bytes, long min_savings_bytes) {
    const long peak = static_cast<long>(Stack.max_utilization());
    if (peak <= 0) {
        // Stack has never seen an allocation — nothing to measure.
        return 0;
    }
    const long current = static_cast<long>(mStackBuffer.bytes());
    const long target = peak + safety_bytes;
    if (current - target < min_savings_bytes) {
        return 0;
    }
    resize_stack_to(target);
    return target;
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

Tensor& DslRunState::rope_freqs(std::string_view name) {
    int layer_idx = -1;
    std::string field;
    if (!mPerLayerRopeFreqs.empty() && parse_block_param(name, layer_idx, field) && layer_idx >= 0 &&
        static_cast<std::size_t>(layer_idx) < mPerLayerRopeFreqs.size() &&
        mPerLayerRopeFreqs[static_cast<std::size_t>(layer_idx)].Data) {
        return mPerLayerRopeFreqs[static_cast<std::size_t>(layer_idx)];
    }
    if (!mNonBlockActivations.freq_cis.Data) {
        throw std::runtime_error("DslRunState: RoPE frequencies not allocated");
    }
    return mNonBlockActivations.freq_cis;
}

const Tensor& DslRunState::rope_freqs(std::string_view name) const {
    return const_cast<DslRunState*>(this)->rope_freqs(name);
}

void DslRunState::allocate_non_block_state(const PretrainedConfig& cfg) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long V = cfg.VocabSize;
    const auto dtype = mActivationDtype;

    mNonBlockActivations.encoded = mAllocator->allocate(dtype, "encoded", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final = mAllocator->allocate(dtype, "ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    mNonBlockActivations.ln_final_rstd =
        mAllocator->allocate(ETensorDType::FP32, "ln_final_rstd", EAllocationType::ON_DEVICE, {B, T});

    // Output buffer (persistent; avoids large stack pressure for full fine-tuning).
    const long lmhead_chunks = static_cast<long>(mLMHeadChunks);
    const long out_size = (B * T) / lmhead_chunks;
    mNonBlockActivations.output = mAllocator->allocate(dtype, "output", EAllocationType::ON_DEVICE, {out_size, V});

    // RoPE frequencies (if not using fused RoPE).
    // Two sources of position IDs that exceed the training sequence length T:
    //   1. MRoPE / multimodal: vision tokens reference spatial positions.
    //   2. sample_packing: packed documents carry their original (pre-truncation)
    //      per-document positions, so a mid-document slice of length T can have
    //      positions well above T (observed up to ~680 for T=512 on ro_gsm8k).
    // Indexing freqs_cis past the populated region reads uninitialized memory
    // and produces ~1e38 garbage in RoPE output — catastrophic NaN downstream.
    // So we extend the freqs table whenever the model can natively see positions
    // past T. Cap at 4x T by default (enough headroom for typical packing runs
    // while keeping the allocation small); SUROGATE_ROPE_MAX_SEQ overrides.
    // Examples: Qwen3.5 MaxPositionEmbeddings=262144 → 256 MiB for T=2048 at
    // full cap, so the 4x cap is essential for memory. For sample_packing runs
    // that exceed 4x T, set SUROGATE_ROPE_MAX_SEQ accordingly.
    int max_seq_len = static_cast<int>(T);
    if (cfg.MaxPositionEmbeddings > max_seq_len) {
        const char* rope_max_env = std::getenv("SUROGATE_ROPE_MAX_SEQ");
        if (rope_max_env) {
            max_seq_len = std::max(max_seq_len, static_cast<int>(std::strtol(rope_max_env, nullptr, 10)));
        } else {
            // Cap at 4x training sequence length — sufficient for text packing
            // and minor MRoPE spatial offsets.
            max_seq_len = std::min(cfg.MaxPositionEmbeddings, max_seq_len * 4);
        }
    }
    if (max_seq_len > 0) {
        const int head_size = cfg.head_size();
        const RopeInvFreq rope_params = compute_rope_inv_freq(cfg, head_size, max_seq_len);
        if (dtype == ETensorDType::BF16) {
            mNonBlockActivations.freq_cis =
                mAllocator->allocate(dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            fill_rope_freqs(freq_cpu, rope_params, head_size, max_seq_len);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data,
                                  freq_cpu.data(),
                                  freq_cpu.size() * sizeof(nv_bfloat16),
                                  cudaMemcpyHostToDevice));
        } else if (dtype == ETensorDType::FP32) {
            mNonBlockActivations.freq_cis =
                mAllocator->allocate(dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) * 2 * head_size);
            fill_rope_freqs(freq_cpu, rope_params, head_size, max_seq_len);
            CUDA_CHECK(cudaMemcpy(mNonBlockActivations.freq_cis.Data,
                                  freq_cpu.data(),
                                  freq_cpu.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
        } else {
            // Default: allocate in model dtype and leave zeroed.
            mNonBlockActivations.freq_cis =
                mAllocator->allocate(dtype, "freq_cis", EAllocationType::ON_DEVICE, {max_seq_len, 2 * head_size});
            fill_zero(mNonBlockActivations.freq_cis, MainStream);
        }

        if (mRuntimeConfig.has_per_layer_rope()) {
            mPerLayerRopeFreqs.resize(mRuntimeConfig.per_layer_rope.size());
            for (std::size_t i = 0; i < mRuntimeConfig.per_layer_rope.size(); ++i) {
                const auto& layer_rope = mRuntimeConfig.per_layer_rope[i];
                const RopeInvFreq layer_params =
                    compute_rope_inv_freq(cfg, layer_rope.rope, layer_rope.head_size, max_seq_len);
                if (layer_params.dim <= 0) {
                    continue;
                }

                const std::vector<long> shape = {max_seq_len, layer_params.dim / 2, 2};
                const std::string name = "rope_freqs_layer" + std::to_string(i);
                if (dtype == ETensorDType::BF16) {
                    Tensor freqs = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
                    std::vector<nv_bfloat16> freq_cpu(static_cast<std::size_t>(max_seq_len) *
                                                      static_cast<std::size_t>(layer_params.dim));
                    fill_rope_freqs(freq_cpu, layer_params, static_cast<int>(layer_rope.head_size), max_seq_len);
                    CUDA_CHECK(cudaMemcpy(freqs.Data,
                                          freq_cpu.data(),
                                          freq_cpu.size() * sizeof(nv_bfloat16),
                                          cudaMemcpyHostToDevice));
                    mPerLayerRopeFreqs[i] = freqs;
                } else if (dtype == ETensorDType::FP32) {
                    Tensor freqs = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
                    std::vector<float> freq_cpu(static_cast<std::size_t>(max_seq_len) *
                                                static_cast<std::size_t>(layer_params.dim));
                    fill_rope_freqs(freq_cpu, layer_params, static_cast<int>(layer_rope.head_size), max_seq_len);
                    CUDA_CHECK(cudaMemcpy(freqs.Data,
                                          freq_cpu.data(),
                                          freq_cpu.size() * sizeof(float),
                                          cudaMemcpyHostToDevice));
                    mPerLayerRopeFreqs[i] = freqs;
                } else {
                    Tensor freqs = mAllocator->allocate(dtype, name.c_str(), EAllocationType::ON_DEVICE, shape);
                    fill_zero(freqs, MainStream);
                    mPerLayerRopeFreqs[i] = freqs;
                }
            }
        }
    }

    mNonBlockGradients.d_ln_final =
        mAllocator->allocate(mGradDtype, "d_ln_final", EAllocationType::ON_DEVICE, {B, T, C});
    // Always allocate d_embeddings even in LoRA-only mode. While embedding backward
    // is skipped in LoRA mode, the autodiff graph still produces d_embed_1 as an
    // intermediate. Without a persistent buffer, ensure_output_tensor allocates it on
    // the stack where it blocks can_restore_stack for the entire backward pass (its
    // last_use is the final embedding_backward op), preventing per-layer stack restores
    // and causing stack OOM on MoE models with many layers.
    mNonBlockGradients.d_embeddings =
        mAllocator->allocate(mGradDtype, "d_embeddings", EAllocationType::ON_DEVICE, {B, T, C});
}

void DslRunState::allocate_simplified_activations(const PretrainedConfig& cfg) {
    // All sharing / sizing / stack-temp decisions come from the plan built
    // in the constructor — see buffer_plan.{h,cpp}. This function is a
    // mechanical walk over the plan: any NEW sharing or sizing decision
    // belongs in `BufferPlan::build`, not here. Re-reading `mRecomputeLevel`,
    // `mLoraOnlyMode`, `mRuntimeConfig`, or `mSlotRegistry` directly inside
    // the allocator would re-introduce the policy/mechanism split the plan
    // was introduced to eliminate.
    const auto& plan = mBufferPlan;
    const long B = plan.B;
    const long T = plan.T;
    const long C = plan.C;
    const long Hq = plan.Hq;
    const long Hkv = plan.Hkv;
    const long AttnDim = plan.AttnDim;  // max across hybrid layers
    const long QKV = plan.QKV;          // max across hybrid layers
    const long M = plan.M;              // max across hybrid layers
    const long MUp = plan.MUp;          // max across hybrid layers
    const long NumExperts = plan.NumExperts;
    const long TopK = plan.TopK;
    const long MoeM = plan.MoeM;
    const long MoeMUp = plan.MoeMUp;
    const bool use_qk_norm = plan.use_qk_norm;
    const bool recompute_enabled = plan.recompute_enabled;

    const auto dtype = plan.act_dtype;
    const auto kind = EAllocationType::ON_DEVICE;

    // Phase 4 M5 preamble: remove upfront `shared_*` buffer allocations.
    // The legacy share pattern coalesced all layers' <slot> onto one
    // buffer under recompute, relying on forward replay to regenerate
    // per-layer values before each backward reads. That broke the LoRA
    // dispatch's captured input pointers when Phase A's per-layer
    // Stack migration changed the storage. Always allocate per-layer
    // now; LoRA naturally sees per-layer pointers and Phase A
    // migration can proceed slot-by-slot without special-casing share.
    // Memory cost: (NumLayers - 1) * <slot size> per shared slot.

    mSimplifiedActivations.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& acts = mSimplifiedActivations[i];

        // Per-layer dimensions for hybrid models — accessors fall back to the
        // shared (max) value for homogeneous models.
        const long lQKV = plan.layer_qkv(i);
        const long lAttnDim = plan.layer_attn_dim(i);
        const long lMUp = plan.layer_mlp_up(i);
        const long lM = plan.layer_intermediate(i);

        // Per-layer allocation tag = canonical slot name; shared buffers are
        // already created above with the `<slot>_shared` tag.
        auto tag = [](TensorSlot s) {
            return builtin_slot_name(s);
        };

        const bool rstd_on_stack = []() {
            const char* e = std::getenv("SUROGATE_RSTD_ON_STACK");
            return e && std::string(e) == "1";
        }();
        if (rstd_on_stack) {
            acts.ln1_rstd = Tensor::from_pointer(nullptr, DeviceId, ETensorDType::FP32, std::vector<long>{B, T});
        } else {
            acts.ln1_rstd = mAllocator->allocate(ETensorDType::FP32, tag(TensorSlot::BlockLN1RSTD), kind, {B, T});
        }
        if (rstd_on_stack) {
            acts.ln1 = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, C});
        } else {
            acts.ln1 = mAllocator->allocate(dtype, tag(TensorSlot::BlockLN1), kind, {B, T, C});
        }

        if (rstd_on_stack) {
            acts.ln2_rstd = Tensor::from_pointer(nullptr, DeviceId, ETensorDType::FP32, std::vector<long>{B, T});
        } else {
            acts.ln2_rstd = mAllocator->allocate(ETensorDType::FP32, tag(TensorSlot::BlockLN2RSTD), kind, {B, T});
        }
        if (rstd_on_stack) {
            acts.ln2 = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, C});
        } else {
            acts.ln2 = mAllocator->allocate(dtype, tag(TensorSlot::BlockLN2), kind, {B, T, C});
        }

        if (use_qk_norm) {
            if (rstd_on_stack) {
                acts.q_rstd = Tensor::from_pointer(nullptr, DeviceId, ETensorDType::FP32, std::vector<long>{B, T, Hq});
                acts.k_rstd = Tensor::from_pointer(nullptr, DeviceId, ETensorDType::FP32, std::vector<long>{B, T, Hkv});
            } else {
                acts.q_rstd = mAllocator->allocate(ETensorDType::FP32, tag(TensorSlot::BlockQRSTD), kind, {B, T, Hq});
                acts.k_rstd = mAllocator->allocate(ETensorDType::FP32, tag(TensorSlot::BlockKRSTD), kind, {B, T, Hkv});
            }
        } else {
            acts.q_rstd = {};
            acts.k_rstd = {};
        }

        acts.qkv = mAllocator->allocate(dtype, tag(TensorSlot::BlockQKV), kind, {B, T, lQKV});
        if (plan.need_separate_qkv_rope) {
            acts.qkv_rope = mAllocator->allocate(dtype, tag(TensorSlot::BlockQKVRoPE), kind, {B, T, lQKV});
        } else {
            acts.qkv_rope = {};
        }

        if (rstd_on_stack) {
            acts.lse = Tensor::from_pointer(nullptr, DeviceId, ETensorDType::FP32, std::vector<long>{B, Hq, T});
        } else {
            acts.lse = mAllocator->allocate(ETensorDType::FP32, tag(TensorSlot::BlockLSE), kind, {B, Hq, T});
        }
        acts.att = mAllocator->allocate(dtype, tag(TensorSlot::BlockAtt), kind, {B, T, lAttnDim});
        if (rstd_on_stack) {
            acts.att_out = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, C});
        } else {
            acts.att_out = mAllocator->allocate(dtype, tag(TensorSlot::BlockAttOut), kind, {B, T, C});
        }

        acts.residual_att = mAllocator->allocate(dtype, tag(TensorSlot::BlockResidualAtt), kind, {B, T, C});

        // Skip mlp_up/swiglu allocation when the DSL layout doesn't define these slots
        // (e.g., GatedMLP which uses stack-based temps instead of pre-allocated buffers).
        if (plan.ffn_temps_on_stack || !plan.has_mlp_up_slot) {
            acts.mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, lMUp});
        } else {
            acts.mlp_up = mAllocator->allocate(dtype, tag(TensorSlot::BlockMLPUp), kind, {B, T, lMUp});
        }
        if (plan.ffn_temps_on_stack || !plan.has_swiglu_slot) {
            acts.swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, lM});
        } else {
            acts.swiglu = mAllocator->allocate(dtype, tag(TensorSlot::BlockSwiGLU), kind, {B, T, lM});
        }

        acts.mlp_down = mAllocator->allocate(dtype, tag(TensorSlot::BlockMLPDown), kind, {B, T, C});
        // Dedicated per-layer block output slot used by Gemma4 (keeps the
        // block's final h_out separate from the MLP's mlp_down so the
        // autodiff's produced_by map doesn't lose the MLP → post_ff_ln
        // dependency that drives MLP LoRA gradients).
        if (rstd_on_stack) {
            acts.h_out = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, C});
        } else {
            acts.h_out = mAllocator->allocate(dtype, tag(TensorSlot::BlockHOut), kind, {B, T, C});
        }

        if (NumExperts > 0) {
            const long num_tokens = B * T;
            const long total_tokens = num_tokens * TopK;
            // MoE slot tags now come from the TensorSlot enum — same reverse
            // mapping used for every other block slot. Adding a new MoE slot
            // is a one-line change in `kSlotMappings`/`slot_to_name`.
            acts.router_logits =
                mAllocator->allocate(dtype, tag(TensorSlot::BlockRouterLogits), kind, {num_tokens, NumExperts});
            acts.router_probs =
                mAllocator->allocate(dtype, tag(TensorSlot::BlockRouterProbs), kind, {num_tokens, NumExperts});
            acts.routing_weights =
                mAllocator->allocate(dtype, tag(TensorSlot::BlockRoutingWeights), kind, {num_tokens, TopK});
            acts.routing_indices = mAllocator->allocate(ETensorDType::INT32,
                                                        tag(TensorSlot::BlockRoutingIndices),
                                                        kind,
                                                        {num_tokens, TopK});
            acts.permuted_input =
                mAllocator->allocate(dtype, tag(TensorSlot::BlockPermutedInput), kind, {total_tokens, C});
            acts.scatter_indices =
                mAllocator->allocate(ETensorDType::INT32, tag(TensorSlot::BlockScatterIndices), kind, {total_tokens});
            acts.expert_gate_up =
                mAllocator->allocate(dtype, tag(TensorSlot::BlockExpertGateUp), kind, {total_tokens, MoeMUp});
            acts.expert_act = mAllocator->allocate(dtype, tag(TensorSlot::BlockExpertAct), kind, {total_tokens, MoeM});
            acts.expert_down = mAllocator->allocate(dtype, tag(TensorSlot::BlockExpertDown), kind, {total_tokens, C});
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
}

void DslRunState::allocate_simplified_gradients(const PretrainedConfig& cfg) {
    // Like allocate_simplified_activations, this is a mechanical walk over
    // the plan — new decisions belong in BufferPlan::build, not here.
    const auto& plan = mBufferPlan;
    const long B = plan.B;
    const long T = plan.T;
    const long C = plan.C;

    const auto dtype = plan.grad_dtype;
    const auto kind = EAllocationType::ON_DEVICE;

    mSimplifiedGradients.resize(cfg.NumLayers);
    for (int i = 0; i < cfg.NumLayers; ++i) {
        auto& g = mSimplifiedGradients[i];

        const long lQKV = plan.layer_qkv(i);
        const long lAttnDim = plan.layer_attn_dim(i);
        const long lMUp = plan.layer_mlp_up(i);
        const long lM = plan.layer_intermediate(i);

        g.d_res_ffn = mAllocator->allocate(dtype, "d_res_ffn", kind, {B, T, C});
        g.d_res_att = mAllocator->allocate(dtype, "d_res_att", kind, {B, T, C});
        g.d_att_out = plan.is_hybrid ? mAllocator->allocate(dtype, "d_att_out", kind, {B, T, C}) : g.d_res_att;
        g.d_ln2 = mAllocator->allocate(dtype, "d_ln2", kind, {B, T, C});

        if (plan.large_bwd_temps_on_stack || !plan.has_mlp_up_slot) {
            g.d_mlp_up = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, lMUp});
        } else {
            g.d_mlp_up = mAllocator->allocate(dtype, "d_mlp_up", kind, {B, T, lMUp});
        }
        if (plan.large_bwd_temps_on_stack || !plan.has_swiglu_slot) {
            g.d_swiglu = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, lM});
        } else {
            g.d_swiglu = mAllocator->allocate(dtype, "d_swiglu", kind, {B, T, lM});
        }
        if (plan.large_bwd_temps_on_stack) {
            g.d_qkv = Tensor::from_pointer(nullptr, DeviceId, dtype, std::vector<long>{B, T, lQKV});
        } else {
            g.d_qkv = mAllocator->allocate(dtype, "d_qkv", kind, {B, T, lQKV});
        }

        g.d_mlp_down = mAllocator->allocate(dtype, "d_mlp_down", kind, {B, T, C});
        // Per-layer h_out gradient (Gemma4: block's final-output grad,
        // separate from MLP's d_mlp_down so the backward chain doesn't
        // collide across the MLP / _finalize split).
        g.d_h_out = mAllocator->allocate(dtype, "d_h_out", kind, {B, T, C});
        g.d_att = mAllocator->allocate(dtype, "d_att", kind, {B, T, lAttnDim});
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
        dst.d_att_out.Data = src.d_att_out.Data;
        dst.d_ln2.Data = src.d_ln2.Data;
        dst.d_mlp_up.Data = src.d_mlp_up.Data;
        dst.d_swiglu.Data = src.d_swiglu.Data;
        dst.d_mlp_down.Data = src.d_mlp_down.Data;
        dst.d_att.Data = src.d_att.Data;
        dst.d_qkv.Data = src.d_qkv.Data;
        dst.d_ln1.Data = src.d_ln1.Data;
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
        // d_h_out is an accumulation target for the next block's backward
        // (gemma4 routes block output through a dedicated h_out slot to
        // avoid mlp_down collision). Without zeroing, garbage in d_h_out
        // leaks into the first backward that reads it (→ NaN cascade).
        add(g.d_h_out);
    }

    mActGradZeroCount = static_cast<int>(ptrs.size());
    if (mActGradZeroCount <= 0) {
        return;
    }

    const long bytes = static_cast<long>(static_cast<std::size_t>(mActGradZeroCount) * sizeof(std::uint64_t));
    mActGradZeroPtrs =
        mAllocator->allocate(ETensorDType::BYTE, "dsl_act_grad_zero_ptrs", EAllocationType::ON_DEVICE, {bytes});
    mActGradZeroSizes =
        mAllocator->allocate(ETensorDType::BYTE, "dsl_act_grad_zero_sizes", EAllocationType::ON_DEVICE, {bytes});

    CUDA_CHECK(cudaMemcpy(mActGradZeroPtrs.Data, ptrs.data(), static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(mActGradZeroSizes.Data, sizes.data(), static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice));
}

void DslRunState::allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options) {
    const long B = this->B;
    const long T = this->T;
    const long C = cfg.HiddenSize;
    const long D = cfg.head_size();
    const long Hq = cfg.NumQueryHeads;
    const long Hkv = cfg.NumKeyValHeads;
    long AttnDim = Hq * D;
    long QKV = D * (Hq + 2 * Hkv);
    long M = cfg.IntermediateSize;
    long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;

    // Hybrid-architecture safety: on models like Gemma4 where different layer
    // types carry different attention dims (full uses global_head_dim, sliding
    // uses head_size) or MLP intermediate widths (e.g. shared-KV blocks with
    // `use_double_wide_mlp`), the shared FP8 forward/quant buffers must be
    // sized to the MAX across all layers. Without this, a layer with larger
    // dims writes past the end of the buffer and triggers
    // `cudaErrorIllegalAddress` inside `quantize_with_delayed_scale` or the
    // quantized-grad matmul_backward.
    if (mRuntimeConfig.has_per_layer_dims()) {
        for (const auto& pld : mRuntimeConfig.per_layer_dims) {
            AttnDim = std::max(AttnDim, static_cast<long>(pld.attn_dim));
            QKV = std::max(QKV, static_cast<long>(pld.qkv_channels));
            M = std::max(M, static_cast<long>(pld.intermediate));
            MUp = std::max(MUp, static_cast<long>(pld.mlp_up));
        }
    }

    if (mEnableFp8Forward) {
        modules::allocate_fp8_forward_buffers(mFP8ForwardQuants,
                                              mFP8ForwardStats,
                                              *mAllocator,
                                              B,
                                              T,
                                              C,
                                              M,
                                              AttnDim,
                                              options.forward_matmul_dtype());
    }

    if (options.fp8_hybrid_enabled()) {
        modules::FP8ScalingConfig fp8_cfg{};
        fp8_cfg.amax_history_len = options.RecipeOptions.fp8_amax_history_len;
        fp8_cfg.margin = static_cast<float>(options.RecipeOptions.fp8_margin);
        mFP8ScalingState = std::make_unique<modules::FP8ScalingState>(fp8_cfg, mAllocator, DeviceId, cfg.NumLayers);
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

    mGradQuantStats =
        mAllocator->allocate(ETensorDType::FP32, "dsl_grad_quant_stats", EAllocationType::ON_DEVICE, {8L});
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

    const long rmsnorm_scratch_bytes =
        static_cast<long>(get_rmsnorm_backward_scratch_size(static_cast<int>(C), DeviceProp));
    mScratch.rmsnorm_scratch = mAllocator->allocate(ETensorDType::BYTE,
                                                    "rmsnorm_scratch",
                                                    EAllocationType::ON_DEVICE,
                                                    {rmsnorm_scratch_bytes});

    const long M = cfg.IntermediateSize;
    const long MUp = static_cast<long>(resolve_mlp_up_factor(cfg)) * M;
    const long V = cfg.VocabSize;
    const long max_bias_channels = std::max<long>(QKV, std::max<long>(C, std::max<long>(MUp, V)));
    const long bias_scratch_bytes =
        static_cast<long>(get_bias_backward_scratch_size(mGradDtype, static_cast<int>(max_bias_channels), DeviceProp));
    mScratch.matmul_bias_scratch = mAllocator->allocate(ETensorDType::FP32,
                                                        "bias_scratch",
                                                        EAllocationType::ON_DEVICE,
                                                        {bias_scratch_bytes / static_cast<long>(sizeof(float))});

    const long num_block_sums = std::max<long>(2, static_cast<long>(get_max_num_block_sums(DeviceProp)));
    mScratch.norm_buffer =
        mAllocator->allocate(ETensorDType::FP32, "norm_buffer", EAllocationType::ON_DEVICE, {num_block_sums});

    mScratch.matmul_scales =
        mAllocator->allocate(ETensorDType::FP32, "matmul_scales", EAllocationType::ON_DEVICE, {2L});

    const long BT = B * T;
    mScratch.cross_entropy_dloss =
        mAllocator->allocate(ETensorDType::FP32, "cross_entropy_dloss", EAllocationType::ON_DEVICE, {BT});
    mScratch.cross_entropy_logsumexp =
        mAllocator->allocate(ETensorDType::FP32, "cross_entropy_logsumexp", EAllocationType::ON_DEVICE, {BT});
    const int n_chunks = static_cast<int>((V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE);
    if (n_chunks > 1) {
        mScratch.cross_entropy_chunk_logsumexp = mAllocator->allocate(ETensorDType::FP32,
                                                                      "cross_entropy_chunk_logsumexp",
                                                                      EAllocationType::ON_DEVICE,
                                                                      {BT, n_chunks});
    }

    // Encoder backward scratch buffers - skip in LoRA-only mode since embedding backward is skipped entirely
    if (!mLoraOnlyMode) {
        const long group_width = static_cast<long>(16 / get_dtype_size(mGradDtype) * 32);
        const long num_c_groups = (C + group_width - 1) / group_width;
        mScratch.encoder_bwd_scratch = mAllocator->allocate(ETensorDType::INT32,
                                                            "encoder_bwd_scratch",
                                                            EAllocationType::ON_DEVICE,
                                                            {B, T, num_c_groups * 5});
        mScratch.encoder_bwd_indices = mAllocator->allocate(ETensorDType::INT32,
                                                            "encoder_bwd_indices",
                                                            EAllocationType::PINNED,
                                                            {B, T, num_c_groups});
        mScratch.encoder_bwd_info = mAllocator->allocate(ETensorDType::INT32,
                                                         "encoder_bwd_info",
                                                         EAllocationType::PINNED,
                                                         {B, T, 4 * num_c_groups});
    }

    const int attn_chunks = mAttnBwdChunks;
    if (attn_chunks < 1) {
        throw std::runtime_error("attn_bwd_chunks must be >= 1");
    }
    const long attn_ws_batch_size = (attn_chunks == 1) ? B : div_exact(B, static_cast<long>(attn_chunks));
    // For hybrid models, use max head_size for cuDNN workspace sizing.
    long max_D = D;
    if (mRuntimeConfig.has_per_layer_dims()) {
        for (const auto& pld : mRuntimeConfig.per_layer_dims) {
            max_D = std::max(max_D, pld.head_size);
        }
    }
    // Delegate workspace sizing to the attention-backend registry.
    // Only cuDNN has a persistent workspace; other backends report 0.
    const long cudnn_ws_size =
        static_cast<long>(AttentionBackendRegistry::instance().max_workspace_bytes(static_cast<int>(attn_ws_batch_size),
                                                                                   static_cast<int>(T),
                                                                                   static_cast<int>(Hq),
                                                                                   static_cast<int>(Hkv),
                                                                                   static_cast<int>(max_D),
                                                                                   CudnnHandle,
                                                                                   cublas_handle()));
    if (cudnn_ws_size > 0) {
        // Pre-allocate cudnn_workspace using the persistent allocator to avoid overlap with
        // stack-allocated gradient buffers. The workspace is large (~192MB) and if allocated
        // from the temp stack, checkpoint restores during backward can cause it to be reallocated
        // in a region that overlaps with gradient buffers.
        mScratch.cudnn_workspace =
            mAllocator->allocate(ETensorDType::BYTE, "cudnn_workspace", EAllocationType::ON_DEVICE, {cudnn_ws_size});
    } else {
        // Leave an empty descriptor; attention ops will fail later if invoked with invalid head size.
        mScratch.cudnn_workspace = Tensor::empty(ETensorDType::BYTE, {0});
    }
}

Tensor* DslRunState::get_fp8_forward_buffer(int op) {
    if (!has_fp8_forward()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV: return &mFP8ForwardQuants.ln1;
        case modules::MatmulOp::MLPUp: return &mFP8ForwardQuants.ln2;
        case modules::MatmulOp::AttnOut: return &mFP8ForwardQuants.att;
        case modules::MatmulOp::MLPDown: return &mFP8ForwardQuants.swiglu;
        default: return nullptr;
    }
}

Tensor* DslRunState::get_gradient_quant_buffer(int op) {
    if (!has_grad_quants()) return nullptr;
    auto matmul_op = static_cast<modules::MatmulOp>(op);
    switch (matmul_op) {
        case modules::MatmulOp::QKV: return &mSimplifiedQuantGrads.d_qkv;
        case modules::MatmulOp::MLPUp: return &mSimplifiedQuantGrads.d_mlp_up;
        case modules::MatmulOp::AttnOut: return &mSimplifiedQuantGrads.d_res_att;
        case modules::MatmulOp::MLPDown: return &mSimplifiedQuantGrads.d_res_ffn;
        default: return nullptr;
    }
}

void DslRunState::allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals) {
    mOffloadResiduals = offload_residuals;
    mResidualManager = std::make_unique<modules::ResidualManager>(mAllocator,
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
    // Must be initialized before any CUDA graph capture; otherwise first
    // fallback GEMM can call cublasCreate inside capture and fail.
    init_cublas_fallback_handle();
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

void DslRunState::set_moe_config(int num_experts, float aux_loss_coef) {
    if (num_experts <= 0) return;
    mNumMoEExperts = num_experts;
    mMoEAuxLossCoef = aux_loss_coef;
    if (!mMoEStatsDevice) {
        CUDA_CHECK(cudaMalloc(&mMoEStatsDevice, kMoEStatsSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(mMoEStatsDevice, 0, kMoEStatsSize * sizeof(float)));
    }
    if (!mMoEStatsHost) {
        CUDA_CHECK(cudaMallocHost(&mMoEStatsHost, kMoEStatsSize * sizeof(float)));
        std::memset(mMoEStatsHost, 0, kMoEStatsSize * sizeof(float));
    }
}

IRunState::MoEStats DslRunState::get_moe_stats() const {
    MoEStats stats;
    if (!mMoEStatsDevice || mNumMoEExperts <= 0) {
        return stats;
    }
    // Copy accumulated stats from device to host (sync — called after forward is complete)
    CUDA_CHECK(cudaMemcpy(mMoEStatsHost, mMoEStatsDevice, kMoEStatsSize * sizeof(float), cudaMemcpyDeviceToHost));
    const int num_layers = static_cast<int>(mMoEStatsHost[4]);
    if (num_layers <= 0) {
        return stats;
    }
    stats.aux_loss = mMoEStatsHost[0];                         // summed across layers
    stats.z_loss = mMoEStatsHost[1];                           // summed across layers
    stats.expert_utilization = mMoEStatsHost[2] / num_layers;  // average
    stats.load_imbalance = mMoEStatsHost[3] / num_layers;      // average
    stats.num_layers = num_layers;
    stats.valid = true;
    return stats;
}

void DslRunState::reset_moe_stats() {
    if (mMoEStatsDevice) {
        CUDA_CHECK(cudaMemsetAsync(mMoEStatsDevice, 0, kMoEStatsSize * sizeof(float), MainStream));
    }
}

}  // namespace dsl
