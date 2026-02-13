// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP8/FP4 weight caching for DSL Graph executor.

#include "runtime/dsl/graph_executor.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/dsl_runtime.h"
#include "runtime/dsl/dsl_weight_manager.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "runtime/core/fp8_scaling_state.h"
#include "runtime/core/matmul_context.h"
#include "runtime/core/model_config.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "runtime/qlora/generic_qlora_provider.h"
#include "runtime/training/runtime_options.h"
#include "utilities/tensor.h"

namespace dsl {
namespace {

// Helper functions for weight caching
bool allow_quant_layer(const RuntimeOptions& options, const modules::ModelConfig& config, int layer_idx) {
    if (layer_idx < 0) return false;
    const int skip_first = options.RecipeOptions.skip_quant_first_layers;
    const int skip_last = options.RecipeOptions.skip_quant_last_layers;
    if (skip_first > 0 && layer_idx < skip_first) return false;
    if (skip_last > 0 && layer_idx >= static_cast<int>(config.NumLayers) - skip_last) return false;
    return true;
}

Tensor* fp8_forward_buffer(DslRunState& rs, modules::MatmulOp op) {
    if (!rs.has_fp8_forward()) return nullptr;
    auto& q = rs.fp8_forward_quants();
    switch (op) {
        case modules::MatmulOp::QKV:
            return &q.ln1;
        case modules::MatmulOp::MLPUp:
            return &q.ln2;
        case modules::MatmulOp::AttnOut:
            return &q.att;
        case modules::MatmulOp::MLPDown:
            return &q.swiglu;
        default:
            return nullptr;
    }
}

Tensor* fp8_grad_buffer(DslRunState& rs, modules::MatmulOp op) {
    if (!rs.has_fp8_hybrid_backward()) return nullptr;
    auto& q = rs.simplified_quant_grads();
    switch (op) {
        case modules::MatmulOp::QKV:
            return &q.d_qkv;
        case modules::MatmulOp::MLPUp:
            return &q.d_mlp_up;
        case modules::MatmulOp::AttnOut:
            return &q.d_res_att;
        case modules::MatmulOp::MLPDown:
            return &q.d_res_ffn;
        default:
            return nullptr;
    }
}

int fp8_quantizer_index(const DslRunState& rs, modules::MatmulOp op, int layer_idx) {
    if (!rs.has_fp8_delayed_scaling()) return -1;
    switch (op) {
        case modules::MatmulOp::QKV:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN1);
        case modules::MatmulOp::MLPUp:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_LN2);
        case modules::MatmulOp::AttnOut:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_ATT);
        case modules::MatmulOp::MLPDown:
            return modules::get_quantizer_index(layer_idx, modules::QuantizerIndex::FWD_SWIGLU);
        default:
            return -1;
    }
}

}  // namespace

void GraphExecutor::prime_fp8_weight_cache(const std::vector<char>& required) {
    if (!mOptions.TrainingRecipe || !mRunState.has_fp8_forward()) {
        return;
    }
    if (!mForward) {
        return;
    }
    const auto& config = mConfig;
    for (std::size_t idx = 0; idx < mForward->operations.size(); ++idx) {
        if (!required.empty() && !required[idx]) {
            continue;
        }
        const auto& op = mForward->operations[idx];
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        if (op_type != "matmul" && op_type != "matmul_bias") {
            continue;
        }
        if (op.inputs.size() < 2) {
            continue;
        }
        int layer_idx = -1;
        auto op_kind = matmul_op_from_weight(op.inputs.at(1), layer_idx);
        if (!op_kind.has_value()) {
            continue;
        }
        if (!allow_quant_layer(mOptions, config, layer_idx)) {
            continue;
        }
        const std::string& weight_name = op.inputs.at(1);
        if (!mWeights.has(weight_name)) {
            continue;
        }
        Tensor& weight = mWeights.get(weight_name);
        (void)get_fp8_cached_weight(weight_name, weight, mRunState.MainStream);
    }
}

const Tensor* GraphExecutor::get_fp8_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream) {
    if (!mRunState.has_fp8_forward()) {
        return nullptr;
    }
    if (weight.DType == ETensorDType::FP8_E4M3) {
        return &weight;
    }
    if (!mWeights.has(name) || mWeights.is_trainable(name)) {
        return nullptr;
    }
    auto it = mFP8WeightCache.find(name);
    if (it == mFP8WeightCache.end()) {
        FP8WeightCacheEntry entry{};
        std::vector<long> shape(weight.Sizes.begin(), weight.Sizes.begin() + weight.Rank);
        entry.weight = mRunState.Allocator->allocate(ETensorDType::FP8_E4M3,
                                                     ("fp8_cache_" + name).c_str(),
                                                     EAllocationType::ON_DEVICE,
                                                     shape);
        entry.stats = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                    ("fp8_cache_" + name + "_stats").c_str(),
                                                    EAllocationType::ON_DEVICE,
                                                    {2L});
        entry.weight.Stats = entry.stats.get<float>();
        auto [insert_it, _] = mFP8WeightCache.emplace(name, std::move(entry));
        it = insert_it;
    }

    // Quantize BF16/FP32 weight to FP8 cache once (static weights only).
    if (!it->second.initialized) {
        if (weight.DType == ETensorDType::BF16 || weight.DType == ETensorDType::FP32) {
            const long N = static_cast<long>(weight.nelem());
            if (N > 0) {
                abs_max(it->second.weight.abs_max(), weight, N, mRunState.DeviceProp, stream);
                quantize_with_abs_max(it->second.weight, it->second.weight.scale(),
                                      weight, it->second.weight.abs_max(),
                                      N, mRunState.DeviceProp, stream);
            }
        }
        it->second.initialized = true;
    }

    return &it->second.weight;
}

// ============================================================================
// FP8 Transposed Weight Caching (for FP8 hybrid backward dinp path)
// ============================================================================

void GraphExecutor::prime_fp8_weight_cache_transposed(const std::vector<char>& required) {
    if (!mOptions.TrainingRecipe || !mRunState.has_fp8_hybrid_backward()) {
        return;
    }
    if (!mBackward) {
        return;
    }

    const auto& config = mConfig;
    for (std::size_t idx = 0; idx < mBackward->operations.size(); ++idx) {
        if (!required.empty() && !required[idx]) {
            continue;
        }
        const auto& op = mBackward->operations[idx];
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        if (op_type != "matmul_backward" && op_type != "matmul_swiglu_backward") {
            continue;
        }
        if (op.inputs.size() < 3) {
            continue;
        }
        const std::string& weight_name = op.inputs.at(2);

        int layer_idx = -1;
        auto op_kind = matmul_op_from_weight(weight_name, layer_idx);
        if (!op_kind.has_value()) {
            continue;
        }
        if (!allow_quant_layer(mOptions, config, layer_idx)) {
            continue;
        }

        if (!mWeights.has(weight_name)) {
            continue;
        }
        Tensor& weight = mWeights.get(weight_name);
        (void)get_fp8_cached_weight_transposed(weight_name, weight, mRunState.MainStream);
    }
}

const Tensor* GraphExecutor::get_fp8_cached_weight_transposed(const std::string& name,
                                                              Tensor& weight,
                                                              cudaStream_t stream) {
    if (!mRunState.has_fp8_hybrid_backward()) {
        return nullptr;
    }
    if (!mWeights.has(name) || mWeights.is_trainable(name)) {
        return nullptr;
    }

    // Only cache rank-2 weights.
    if (weight.Rank != 2) {
        return nullptr;
    }

    const int N = static_cast<int>(weight.Sizes[0]);
    const int K = static_cast<int>(weight.Sizes[1]);
    if (N <= 0 || K <= 0) {
        return nullptr;
    }

    // Cached tensor is the transposed view: (K, N).
    auto it = mFP8WeightCacheT.find(name);
    if (it == mFP8WeightCacheT.end()) {
        FP8WeightCacheEntry entry{};
        entry.weight = mRunState.Allocator->allocate(ETensorDType::FP8_E4M3,
                                                     ("fp8_cacheT_" + name).c_str(),
                                                     EAllocationType::ON_DEVICE,
                                                     {static_cast<long>(K), static_cast<long>(N)});
        entry.stats = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                    ("fp8_cacheT_" + name + "_stats").c_str(),
                                                    EAllocationType::ON_DEVICE,
                                                    {2L});
        entry.weight.Stats = entry.stats.get<float>();
        auto [insert_it, _] = mFP8WeightCacheT.emplace(name, std::move(entry));
        it = insert_it;
    }

    if (!it->second.initialized) {
        if (weight.DType == ETensorDType::BF16 || weight.DType == ETensorDType::FP32) {
            const long numel = static_cast<long>(N) * K;
            abs_max(it->second.weight.abs_max(), weight, numel, mRunState.DeviceProp, stream);
            // Quantize+transpose into the cached tensor (K, N).
            quantize_and_transpose_with_abs_max(it->second.weight, it->second.weight.scale(),
                                                weight, it->second.weight.abs_max(),
                                                /*rows=*/N, /*cols=*/K,
                                                mRunState.DeviceProp, stream);
        } else if (weight.DType == ETensorDType::FP8_E4M3) {
            // Weight already FP8: just transpose and keep the same scale.
            transpose(it->second.weight, weight, N, K, stream);
            if (!weight.scale()) {
                throw std::runtime_error("DSL FP8 cacheT: FP8 weight missing scale Stats");
            }
            CUDA_CHECK(cudaMemcpyAsync(it->second.weight.scale(), weight.scale(), sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        } else {
            return nullptr;
        }
        it->second.initialized = true;
    }

    return &it->second.weight;
}

// ============================================================================
// FP4 Weight Caching (for NVFP4 recipe on Blackwell+)
// ============================================================================

void GraphExecutor::prime_fp4_weight_cache(const std::vector<char>& required) {
    if (!mOptions.TrainingRecipe || !mOptions.fp4_enabled()) {
        return;
    }
    if (!mForward) {
        return;
    }

    int primed_fwd = 0, skipped_fwd = 0, failed_fwd = 0;

    // Pre-quantize static weights for all matmul operations that allow FP4 (forward pass)
    for (std::size_t i = 0; i < mForward->operations.size(); ++i) {
        if (!required.empty() && required[i] == 0) {
            continue;
        }
        const auto& op = mForward->operations[i];
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        if (op_type != "matmul" && op_type != "matmul_bias" && op_type != "matmul_swiglu") {
            continue;
        }
        if (op.inputs.size() < 2) {
            continue;
        }

        const std::string& weight_name = op.inputs.at(1);
        if (!mWeights.has(weight_name)) {
            continue;
        }

        // Check if this layer allows FP4 quantization
        int layer_idx = -1;
        auto op_kind = matmul_op_from_weight(weight_name, layer_idx);
        if (!op_kind.has_value() || !allow_quant_layer(mOptions, mConfig, layer_idx)) {
            continue;
        }

        Tensor& weight = mWeights.get(weight_name);
        auto* entry = get_fp4_cached_weight(weight_name, weight, mRunState.MainStream);
        if (entry) {
            primed_fwd++;
        } else {
            failed_fwd++;
            std::cerr << "[FP4 cache] FAILED to prime forward weight: " << weight_name
                      << " (trainable=" << mWeights.is_trainable(weight_name)
                      << ", dtype=" << static_cast<int>(weight.DType)
                      << ", qlora=" << (mWeights.qlora_provider() != nullptr)
                      << ")" << std::endl;
        }
    }

    int primed_bwd = 0, failed_bwd = 0;

    // Also prime transposed FP4 cache for backward pass (matmul_backward dgrad)
    if (mBackward) {
        for (const auto& op : mBackward->operations) {
            const std::string& op_type =
                (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
            if (op_type != "matmul_backward" && op_type != "matmul_swiglu_backward") {
                continue;
            }
            if (op.inputs.size() < 3) {
                continue;
            }

            // Weight is the third input for matmul_backward and matmul_swiglu_backward
            const std::string& weight_name = op.inputs.at(2);
            if (!mWeights.has(weight_name)) {
                continue;
            }

            // Check if this layer allows FP4 quantization
            int layer_idx = -1;
            auto op_kind = matmul_op_from_weight(weight_name, layer_idx);
            if (!op_kind.has_value() || !allow_quant_layer(mOptions, mConfig, layer_idx)) {
                continue;
            }

            Tensor& weight = mWeights.get(weight_name);
            auto* entry = get_fp4_cached_weight_transposed(weight_name, weight, mRunState.MainStream);
            if (entry) {
                primed_bwd++;
            } else {
                failed_bwd++;
                std::cerr << "[FP4 cache] FAILED to prime backward weight: " << weight_name
                          << " (trainable=" << mWeights.is_trainable(weight_name)
                          << ", dtype=" << static_cast<int>(weight.DType)
                          << ")" << std::endl;
            }
        }
    }

    std::cerr << "[FP4 cache] Primed: " << primed_fwd << " fwd + " << primed_bwd << " bwd"
              << " | Failed: " << failed_fwd << " fwd + " << failed_bwd << " bwd"
              << " | Cache sizes: fwd=" << mFP4WeightCache.size()
              << " bwd=" << mFP4WeightCacheT.size() << std::endl;
}

const FP4WeightCacheEntry* GraphExecutor::get_fp4_cached_weight(
    const std::string& name, Tensor& weight, cudaStream_t stream) {

    // Check if FP4 is enabled
    if (!mOptions.fp4_enabled()) {
        return nullptr;
    }

    // Only cache static (non-trainable) weights
    if (!mWeights.has(name) || mWeights.is_trainable(name)) {
        return nullptr;
    }

    // Return early if already cached
    auto it = mFP4WeightCache.find(name);
    if (it != mFP4WeightCache.end() && it->second.initialized) {
        return &it->second;
    }

    // -------------------------------------------------------------------------
    // Direct QLoRA FP4 → CUTLASS FP4 path (avoids BF16 dequant+requant).
    //
    // QLoRA's FP4_BLOCK_2D packed data is identical to CUTLASS row-major FP4,
    // and the F8_128x4 scale layout is algebraically identical to CUTLASS
    // Sm1xxBlkScaled. Only the global scale representation differs:
    //   QLoRA: global_scale = amax / (FP8_MAX * FP4_MAX)
    //   CUTLASS cache: stores raw amax
    //
    // IMPORTANT: This path is only valid when Four Over Six (4/6) adaptive
    // block scaling is DISABLED. QLoRA FP4 uses standard quantization
    // (tensor_scale = amax / (448*6) = amax/2688), but 4/6 alpha computation
    // expects tensor_scale = amax / (384*4) = amax/1536. When 4/6 is enabled,
    // we must fall through to the BF16 → FP4 path which properly re-quantizes
    // with 4/6 adaptive scaling.
    // -------------------------------------------------------------------------
    bool uses_4o6 = false;
    if (mOptions.TrainingRecipe) {
        if (auto* nvfp4 = dynamic_cast<const recipes::NVFP4Recipe*>(mOptions.TrainingRecipe.get())) {
            uses_4o6 = nvfp4->uses_four_over_six();
        }
    }

    if (!uses_4o6) {
    if (auto* provider = dynamic_cast<qlora::GenericQLoRAProvider*>(mWeights.qlora_provider())) {
        if (auto* wm = provider->weight_manager()) {
            if (auto* qt = wm->get_quantized(name)) {
                if (qt->format == qlora::QuantFormat::FP4_BLOCK_2D &&
                    !qt->data.is_null() && !qt->scales.is_null() &&
                    !qt->is_on_host()) {
                    const int N = qt->M;
                    const int K = qt->K;
                    if (N <= 0 || K <= 0 || K % 2 != 0) {
                        return nullptr;
                    }

                    // Allocate cache entry if needed
                    if (it == mFP4WeightCache.end()) {
                        FP4WeightCacheEntry entry{};
                        entry.data = mRunState.Allocator->allocate(ETensorDType::BYTE,
                            ("fp4_cache_" + name + "_data").c_str(),
                            EAllocationType::ON_DEVICE,
                            {static_cast<long>(N), static_cast<long>(K / 2)});
                        const std::size_t ss = compute_nvfp4_cutlass_scale_size(N, K);
                        entry.scales = mRunState.Allocator->allocate(ETensorDType::BYTE,
                            ("fp4_cache_" + name + "_scales").c_str(),
                            EAllocationType::ON_DEVICE,
                            {static_cast<long>(ss)});
                        entry.amax = mRunState.Allocator->allocate(ETensorDType::FP32,
                            ("fp4_cache_" + name + "_amax").c_str(),
                            EAllocationType::ON_DEVICE,
                            {1L});
                        auto [ins_it, _] = mFP4WeightCache.emplace(name, std::move(entry));
                        it = ins_it;
                    }

                    // Copy packed FP4 data directly (identical byte layout)
                    const size_t data_bytes = static_cast<size_t>(N) * (K / 2);
                    CUDA_CHECK(cudaMemcpyAsync(
                        it->second.data.Data, qt->data.Data,
                        data_bytes, cudaMemcpyDeviceToDevice, stream));

                    // Copy FP8 scales directly (F8_128x4 == CUTLASS Sm1xxBlkScaled layout)
                    const std::size_t scale_bytes = compute_nvfp4_cutlass_scale_size(N, K);
                    CUDA_CHECK(cudaMemcpyAsync(
                        it->second.scales.Data, qt->scales.Data,
                        scale_bytes, cudaMemcpyDeviceToDevice, stream));

                    // Convert global_scale to amax: amax = global_scale * FP8_MAX * FP4_MAX
                    constexpr float kFP8Max = 448.0f;
                    constexpr float kFP4Max = 6.0f;
                    float amax = qt->global_scale * kFP8Max * kFP4Max;
                    CUDA_CHECK(cudaMemcpy(
                        it->second.amax.Data, &amax,
                        sizeof(float), cudaMemcpyHostToDevice));

                    it->second.initialized = true;
                    return &it->second;
                }
            }
        }
    }
    }  // !uses_4o6

    // -------------------------------------------------------------------------
    // Fallback: BF16 → FP4 quantization path
    // -------------------------------------------------------------------------
    if (weight.DType != ETensorDType::BF16) {
        return nullptr;
    }

    if (it == mFP4WeightCache.end()) {
        // Weight shape: (N, K) = (C_out, C_in)
        if (weight.Rank != 2) {
            return nullptr;
        }

        const int N = static_cast<int>(weight.Sizes[0]);
        const int K = static_cast<int>(weight.Sizes[1]);

        // K must be even for FP4 packing
        if (K % 2 != 0) {
            return nullptr;
        }

        FP4WeightCacheEntry entry{};

        // Allocate FP4 data: (N, K/2) bytes
        entry.data = mRunState.Allocator->allocate(ETensorDType::BYTE,
                                                   ("fp4_cache_" + name + "_data").c_str(),
                                                   EAllocationType::ON_DEVICE,
                                                   {static_cast<long>(N), static_cast<long>(K / 2)});

        // Allocate FP4 scales in CUTLASS layout
        const std::size_t scale_size = compute_nvfp4_cutlass_scale_size(N, K);
        entry.scales = mRunState.Allocator->allocate(ETensorDType::BYTE,
                                                     ("fp4_cache_" + name + "_scales").c_str(),
                                                     EAllocationType::ON_DEVICE,
                                                     {static_cast<long>(scale_size)});

        // Allocate global amax (single FP32 value)
        entry.amax = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                   ("fp4_cache_" + name + "_amax").c_str(),
                                                   EAllocationType::ON_DEVICE,
                                                   {1L});

        auto [insert_it, _] = mFP4WeightCache.emplace(name, std::move(entry));
        it = insert_it;
    }

    // Quantize weight to FP4 cache once (static weights only)
    if (!it->second.initialized) {
        const int N = static_cast<int>(weight.Sizes[0]);
        const int K = static_cast<int>(weight.Sizes[1]);

        // Use 4/6 quantization if enabled in recipe config
        // Check if recipe is NVFP4 and get 4/6 config from it
        bool use_4o6 = false;
        recipes::FourOverSixErrorMetric four_over_six_metric = recipes::FourOverSixErrorMetric::MSE;
        if (mOptions.TrainingRecipe) {
            if (auto* nvfp4 = dynamic_cast<const recipes::NVFP4Recipe*>(mOptions.TrainingRecipe.get())) {
                use_4o6 = nvfp4->uses_four_over_six();
                four_over_six_metric = nvfp4->four_over_six_metric();
            }
        }

        if (use_4o6) {
            quantize_nvfp4_4o6_cutlass_auto_scale(
                it->second.data.get<uint8_t>(),
                it->second.scales.get<uint8_t>(),
                it->second.amax.get<float>(),
                weight.get<nv_bfloat16>(),
                N, K,
                four_over_six_metric,
                mRunState.DeviceProp, stream);
        } else {
            quantize_nvfp4_weight_cutlass_auto_scale(
                it->second.data.get<uint8_t>(),
                it->second.scales.get<uint8_t>(),
                it->second.amax.get<float>(),
                weight.get<nv_bfloat16>(),
                N, K,
                mRunState.DeviceProp, stream);
        }

        it->second.initialized = true;
    }

    return &it->second;
}

const FP4WeightCacheEntry* GraphExecutor::get_fp4_cached_weight_transposed(
    const std::string& name, Tensor& weight, cudaStream_t stream) {
    // Check if FP4 is enabled
    if (!mOptions.fp4_enabled()) {
        return nullptr;
    }

    // Only cache static (non-trainable) weights
    if (!mWeights.has(name) || mWeights.is_trainable(name)) {
        return nullptr;
    }

    // Only support BF16 weights for now
    if (weight.DType != ETensorDType::BF16) {
        return nullptr;
    }

    auto it = mFP4WeightCacheT.find(name);
    if (it == mFP4WeightCacheT.end()) {
        // Original weight shape: (N, K) = (C_out, C_in)
        // Transposed shape for dgrad: (K, N) = (C_in, C_out)
        if (weight.Rank != 2) {
            return nullptr;
        }

        const int N = static_cast<int>(weight.Sizes[0]);  // C_out
        const int K = static_cast<int>(weight.Sizes[1]);  // C_in

        // N must be even for FP4 packing of transposed weight (K, N/2)
        if (N % 2 != 0) {
            return nullptr;
        }

        FP4WeightCacheEntry entry{};

        // Allocate transposed FP4 data: (K, N/2) bytes
        entry.data = mRunState.Allocator->allocate(ETensorDType::BYTE,
                                                   ("fp4_cacheT_" + name + "_data").c_str(),
                                                   EAllocationType::ON_DEVICE,
                                                   {static_cast<long>(K), static_cast<long>(N / 2)});

        // Allocate FP4 scales in CUTLASS layout for transposed shape
        const std::size_t scale_size = compute_nvfp4_cutlass_scale_size(K, N);
        entry.scales = mRunState.Allocator->allocate(ETensorDType::BYTE,
                                                     ("fp4_cacheT_" + name + "_scales").c_str(),
                                                     EAllocationType::ON_DEVICE,
                                                     {static_cast<long>(scale_size)});

        // Allocate global amax (single FP32 value)
        entry.amax = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                   ("fp4_cacheT_" + name + "_amax").c_str(),
                                                   EAllocationType::ON_DEVICE,
                                                   {1L});

        auto [insert_it, _] = mFP4WeightCacheT.emplace(name, std::move(entry));
        it = insert_it;
    }

    // Quantize weight to transposed FP4 cache once (static weights only)
    if (!it->second.initialized) {
        const int N = static_cast<int>(weight.Sizes[0]);
        const int K = static_cast<int>(weight.Sizes[1]);
        
        // Use transpose quantization for dgrad
        // Note: No 4/6 variant for weight transpose quantization yet
        quantize_nvfp4_weight_cutlass_transpose_auto_scale(
            it->second.data.get<uint8_t>(),
            it->second.scales.get<uint8_t>(),
            it->second.amax.get<float>(),
            weight.get<nv_bfloat16>(),
            N, K,
            mRunState.DeviceProp, stream);

        it->second.initialized = true;
    }

    return &it->second;
}

void GraphExecutor::build_layer_weight_map() {
    if (!mForward || !mLayerWeightNames.empty()) {
        return;
    }

    const int num_layers = mConfig.NumLayers;
    mLayerWeightNames.resize(num_layers);

    // Scan forward graph for matmul operations and map weight names to layers
    for (const auto& op : mForward->operations) {
        const std::string& op_type =
            (op.kernel_type.empty() || op.kernel_type == "custom") ? op.name : op.kernel_type;
        if (op_type != "matmul" && op_type != "matmul_bias") {
            continue;
        }
        if (op.inputs.size() < 2) {
            continue;
        }

        const std::string& weight_name = op.inputs.at(1);
        int layer_idx = -1;
        std::string param_name;
        if (!parse_block_param(weight_name, layer_idx, param_name)) {
            continue;
        }
        if (layer_idx < 0 || layer_idx >= num_layers) {
            continue;
        }

        // Check if this weight is a candidate for FP8 caching
        if (!mWeights.has(weight_name) || mWeights.is_trainable(weight_name)) {
            continue;
        }

        // Avoid duplicates
        auto& names = mLayerWeightNames[layer_idx];
        if (std::find(names.begin(), names.end(), weight_name) == names.end()) {
            names.push_back(weight_name);
        }
    }

    // Enable prefetching if we have:
    // - FP8 forward (need async quantization)
    // - FP4 forward (need async quantization)
    // - Weight streaming/sharding enabled (need async gather)
    // - QLoRA offloading enabled (need async CPU→GPU transfer)
    // - Multi-layer model (enables overlap of weight access with compute)
    const bool has_weight_streaming = mWeightManager && mWeightManager->is_streaming_enabled();
    const bool has_quantized_forward = mRunState.has_fp8_forward() || mRunState.has_fp4_forward();
    const bool has_qlora_offloading = mWeights.qlora_provider() &&
                                      mWeights.qlora_provider()->has_offloading();
    mPrefetchEnabled = num_layers > 1 &&
                       (has_quantized_forward || has_weight_streaming || has_qlora_offloading);
    if (mPrefetchEnabled && !mPrefetchEvent) {
        CUDA_CHECK(cudaEventCreate(&mPrefetchEvent));
    }
}

void GraphExecutor::build_layer_boundaries() {
    if (!mForward || !mLayerBoundaries.empty()) {
        return;
    }

    const int num_layers = mConfig.NumLayers;
    if (num_layers <= 0) {
        return;
    }

    // Temporary map: layer_idx -> [start_idx, end_idx]
    std::unordered_map<int, std::pair<std::size_t, std::size_t>> layer_ranges;

    // Scan all operations to find layer boundaries
    for (std::size_t i = 0; i < mForward->operations.size(); ++i) {
        const auto& op = mForward->operations[i];

        // Check all outputs for layer indices
        for (const auto& out_name : op.outputs) {
            int layer_idx = -1;
            std::string field;
            if (parse_block_param(out_name, layer_idx, field)) {
                if (layer_idx >= 0 && layer_idx < num_layers) {
                    auto it = layer_ranges.find(layer_idx);
                    if (it == layer_ranges.end()) {
                        // First operation for this layer
                        layer_ranges[layer_idx] = {i, i + 1};
                    } else {
                        // Update end index
                        it->second.second = i + 1;
                    }
                }
                break;  // Only need one match per operation
            }
        }
    }

    // Convert to sorted vector
    mLayerBoundaries.reserve(layer_ranges.size());
    for (const auto& [layer_idx, range] : layer_ranges) {
        LayerBoundary boundary;
        boundary.layer_idx = layer_idx;
        boundary.start_op_idx = range.first;
        boundary.end_op_idx = range.second;
        mLayerBoundaries.push_back(boundary);
    }

    // Sort by start_op_idx for O(1) lookup during execution
    std::sort(mLayerBoundaries.begin(), mLayerBoundaries.end(),
              [](const LayerBoundary& a, const LayerBoundary& b) {
                  return a.start_op_idx < b.start_op_idx;
              });
}

void GraphExecutor::prefetch_layer_weights(int layer_idx, cudaStream_t stream) {
    if (!mPrefetchEnabled || layer_idx < 0 || layer_idx >= static_cast<int>(mLayerWeightNames.size())) {
        return;
    }

    // Use weight manager for streaming/sharding if available
    // This enables prefetching for BF16/FP4 with weight streaming (ZeRO-3 style)
    if (mWeightManager && mWeightManager->is_streaming_enabled()) {
        // Note: gather_block needs NCCLCommunicator, which we don't have here.
        // The weight manager will be called from the forward/backward with comm.
        // Here we just record that this layer should be prefetched.
        // The actual prefetch will happen via DslWeightManager::gather_block
        // when called from execute_forward_graph/execute_backward_graph.
    }

    // FP8 weight prefetching (async quantization on prefetch stream)
    if (mRunState.has_fp8_forward()) {
        const auto& weight_names = mLayerWeightNames[layer_idx];
        for (const auto& name : weight_names) {
            if (!mWeights.has(name)) {
                continue;
            }
            Tensor& weight = mWeights.get(name);
            if (weight.DType == ETensorDType::FP8_E4M3) {
                continue;  // Already quantized
            }

            // Ensure cache entry exists
            auto it = mFP8WeightCache.find(name);
            if (it == mFP8WeightCache.end()) {
                FP8WeightCacheEntry entry{};
                std::vector<long> shape(weight.Sizes.begin(), weight.Sizes.begin() + weight.Rank);
                entry.weight = mRunState.Allocator->allocate(ETensorDType::FP8_E4M3,
                                                             ("fp8_cache_" + name).c_str(),
                                                             EAllocationType::ON_DEVICE,
                                                             shape);
                entry.stats = mRunState.Allocator->allocate(ETensorDType::FP32,
                                                            ("fp8_cache_" + name + "_stats").c_str(),
                                                            EAllocationType::ON_DEVICE,
                                                            {2L});
                entry.weight.Stats = entry.stats.get<float>();
                auto [insert_it, _] = mFP8WeightCache.emplace(name, std::move(entry));
                it = insert_it;
            }

            // Quantize on the prefetch stream if not already done
            if (!it->second.initialized) {
                const long N = static_cast<long>(weight.nelem());
                if (N > 0) {
                    abs_max(it->second.weight.abs_max(), weight, N, mRunState.DeviceProp, stream);
                    quantize_with_abs_max(it->second.weight, it->second.weight.scale(),
                                          weight, it->second.weight.abs_max(),
                                          N, mRunState.DeviceProp, stream);
                }
                it->second.initialized = true;
            }
        }
    }

    // FP4 weight prefetching (async quantization on prefetch stream)
    if (mRunState.has_fp4_forward() && mOptions.fp4_enabled()) {
        const auto& weight_names = mLayerWeightNames[layer_idx];
        for (const auto& name : weight_names) {
            if (!mWeights.has(name) || mWeights.is_trainable(name)) {
                continue;
            }
            Tensor& weight = mWeights.get(name);
            if (weight.DType != ETensorDType::BF16) {
                continue;  // FP4 cache only supports BF16 source weights
            }

            // Prefetch forward FP4 cache
            (void)get_fp4_cached_weight(name, weight, stream);

            // Also prefetch transposed FP4 cache for backward dgrad
            (void)get_fp4_cached_weight_transposed(name, weight, stream);
        }
    }

    // QLoRA offload prefetching (async CPU→GPU transfer of quantized weights)
    if (auto* provider = mWeights.qlora_provider()) {
        provider->prefetch_for_layer(layer_idx, stream);
    }

    mPrefetchedLayer = layer_idx;
    if (mPrefetchEvent) {
        CUDA_CHECK(cudaEventRecord(mPrefetchEvent, stream));
    }
}

void GraphExecutor::wait_for_prefetch(int layer_idx, cudaStream_t stream) {
    if (!mPrefetchEnabled || mPrefetchedLayer != layer_idx || !mPrefetchEvent) {
        return;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, mPrefetchEvent, 0));
}

}  // namespace dsl
