// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// GenericQLoRAProvider: QLoRAWeightProvider backed by GenericWeightManager.

#include "runtime/qlora/generic_qlora_provider.h"

#include <stdexcept>

#include <fmt/format.h>

#include "config/pretrained_config.h"
#include "utilities/utils.h"

namespace qlora {

namespace {

/// Parse a block parameter name to extract the layer index.
/// Supports: "blocks[N].param", "mamba_blocks[N].param",
///           "moe_blocks[N].param", "attn_blocks[N].param", etc.
/// Returns -1 if the name doesn't match any block pattern.
int parse_layer_index(std::string_view name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return -1;
    auto prefix = name.substr(0, dot);

    auto bracket = prefix.find("blocks[");
    if (bracket == std::string_view::npos) return -1;

    auto close = prefix.find(']', bracket);
    if (close == std::string_view::npos) return -1;

    auto idx_start = bracket + 7;  // length of "blocks["
    auto idx_str = prefix.substr(idx_start, close - idx_start);
    try {
        return std::stoi(std::string(idx_str));
    } catch (...) {
        return -1;
    }
}

}  // anonymous namespace

// =============================================================================
// Construction
// =============================================================================

GenericQLoRAProvider::GenericQLoRAProvider(
    std::unique_ptr<GenericWeightManager> weight_mgr)
    : mWeightMgr(std::move(weight_mgr))
{
    // Compute total BF16 bytes for memory savings calculation
    mTotalBF16Bytes = 0;
    for (const auto& name : mWeightMgr->weight_names()) {
        const auto* qt = mWeightMgr->get_quantized(name);
        if (qt) {
            mTotalBF16Bytes += static_cast<size_t>(qt->nelem()) * 2;  // BF16 = 2 bytes
        }
    }

    build_layer_offload_map();

    // QLoRA base weights are frozen
    mWeightMgr->set_frozen(true);
}

GenericQLoRAProvider::GenericQLoRAProvider(
    DslQLoRAPipelineConfig config,
    const PretrainedConfig& pt_config,
    std::shared_ptr<TensorAllocator> allocator)
    : mDeferredConfig(std::make_unique<DslQLoRAPipelineConfig>(std::move(config)))
    , mPtConfig(&pt_config)
    , mAllocator(std::move(allocator))
    , mEPSize(mDeferredConfig->ep_size)
{
}

GenericQLoRAProvider::~GenericQLoRAProvider() = default;

// =============================================================================
// QLoRAWeightProvider interface
// =============================================================================

bool GenericQLoRAProvider::handles_param(std::string_view name) const {
    if (!mWeightMgr) {
        return false;
    }
    return mWeightMgr->has_weight(std::string(name));
}

Tensor& GenericQLoRAProvider::resolve_param(std::string_view name, cudaStream_t stream) {
    if (!mWeightMgr) {
        throw std::runtime_error(
            "GenericQLoRAProvider: weights not initialized "
            "(import_and_quantize not called)");
    }
    return mWeightMgr->get(std::string(name), stream);
}

void GenericQLoRAProvider::import_and_quantize(
    const std::string& file_name,
    NCCLCommunicator& /*comm*/,
    cudaStream_t stream) {

    if (mWeightMgr) {
        throw std::runtime_error(
            "GenericQLoRAProvider: weights already imported");
    }

    if (!mDeferredConfig) {
        throw std::runtime_error(
            "GenericQLoRAProvider: no deferred config (constructed with "
            "pre-built weight manager?)");
    }

    if (mDeferredConfig->prequantized) {
        mWeightMgr = import_prequantized_weights(
            file_name,
            *mDeferredConfig,
            *mPtConfig,
            mAllocator,
            stream);
    } else {
        mWeightMgr = import_and_quantize_weights(
            file_name,
            *mDeferredConfig,
            *mPtConfig,
            mAllocator,
            stream);
    }

    // Compute total BF16 bytes for memory savings
    mTotalBF16Bytes = 0;
    for (const auto& name : mWeightMgr->weight_names()) {
        const auto* qt = mWeightMgr->get_quantized(name);
        if (qt) {
            mTotalBF16Bytes += static_cast<size_t>(qt->nelem()) * 2;
        }
    }

    // Build layer → offload group mapping
    build_layer_offload_map();

    // QLoRA base weights are frozen — mark them so that new_step() skips
    // dequant cache invalidation. This avoids redundant FP4/FP8→BF16
    // dequantization every step (the dominant overhead on multi-GPU FP4).
    mWeightMgr->set_frozen(true);

    // Release deferred config (no longer needed)
    mDeferredConfig.reset();
    mPtConfig = nullptr;
}

void GenericQLoRAProvider::invalidate_cache() {
    if (mWeightMgr) {
        mWeightMgr->new_step();
    }

    // Deferred auto-tune: after the first training step completes, all lazy
    // runtime allocations (dequant pool, NCCL buffers, cuDNN workspace, EP
    // persistent state, MoE saved buffers) are settled.  At the start of
    // step 1, measure actual free GPU memory and maximize resident groups.
    if (mAutoTunePending && mStepCount > 0) {
        auto_tune_offloading();
        mAutoTunePending = false;
    }
    mStepCount++;
}

bool GenericQLoRAProvider::refresh_moe_experts(
    int /*layer_idx*/,
    const modules::SelectiveExpertInfo& /*selection*/,
    cudaStream_t /*stream*/) {
    // The generic system handles expert offloading through the OffloadManager
    // which is group-based. Selective expert dequantization is not supported
    // in the generic path - all experts in a group are loaded/unloaded together.
    return false;
}

void GenericQLoRAProvider::prefetch_for_layer(int layer_idx, cudaStream_t stream) {
    if (!mWeightMgr || !mHasOffloading) {
        return;
    }

    auto it = mLayerOffloadGroups.find(layer_idx);
    if (it == mLayerOffloadGroups.end()) {
        return;
    }

    // Prefetch all offload groups that have weights in this layer
    for (int group_id : it->second) {
        mWeightMgr->prefetch_group(group_id, stream);
    }
}

bool GenericQLoRAProvider::has_offloading() const {
    return mHasOffloading;
}

std::size_t GenericQLoRAProvider::quantized_weights_bytes() const {
    return mWeightMgr ? mWeightMgr->quantized_bytes() : 0;
}

float GenericQLoRAProvider::memory_savings_ratio() const {
    if (!mWeightMgr || mTotalBF16Bytes == 0) {
        return 1.0f;
    }
    return static_cast<float>(mWeightMgr->quantized_bytes()) /
           static_cast<float>(mTotalBF16Bytes);
}

// =============================================================================
// Quantized data access (for EP quantized weight transfer)
// =============================================================================

const qlora::QuantizedTensor* GenericQLoRAProvider::try_get_quantized(std::string_view name) const {
    if (!mWeightMgr) return nullptr;
    return mWeightMgr->get_quantized(std::string(name));
}

qlora::IQuantizer* GenericQLoRAProvider::get_quantizer() const {
    if (!mWeightMgr) return nullptr;
    return mWeightMgr->quantizer();
}

void GenericQLoRAProvider::auto_tune_offloading() {
    if (!mWeightMgr) return;
    auto* om = mWeightMgr->offload_manager();
    if (!om || om->num_groups() == 0 || om->max_resident_groups() == 0) return;

    // If called before the first training step, defer to after step 0
    // when all lazy runtime allocations are settled.
    if (mStepCount == 0) {
        mAutoTunePending = true;
        return;
    }

    size_t gpu_free = 0, gpu_total = 0;
    CUDA_CHECK(cudaMemGetInfo(&gpu_free, &gpu_total));

    const size_t max_grp = om->max_group_bytes();
    const int num_grp = om->num_groups();
    // Called after step 0: all runtime buffers are allocated (dequant pool,
    // NCCL, cuDNN, EP persistent state, MoE saved buffers). Free memory
    // reflects actual steady-state availability.  Reserve a small margin
    // (1 GB or 5% of total) for runtime variance and fragmentation.
    // When EP is active, LLEP transfers foreign expert weights as BF16 (up to
    // 2× the quantized group size), so reserve extra to avoid OOM during forward.
    size_t reserve = std::max(static_cast<size_t>(1ULL * 1024 * 1024 * 1024), gpu_total / 20);
    if (mEPSize > 1 && max_grp > 0) {
        reserve += max_grp * 6;  // LLEP foreign BF16 weights + NCCL wt-transfer buffers + dequant pool growth + fragmentation
    }
    const size_t available = (gpu_free > reserve) ? (gpu_free - reserve) : 0;
    int new_max = (max_grp > 0)
        ? static_cast<int>(available / max_grp) : om->max_resident_groups();
    new_max = std::max(2, std::min(new_max, num_grp));

    fprintf(stderr, "[QLoRA] Offload auto-tune: gpu_free=%.1f GB, reserve=%.1f GB, "
            "max_group=%.1f MB, %d groups -> max_resident: %d -> %d%s\n",
            static_cast<double>(gpu_free) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(reserve) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(max_grp) / (1024.0 * 1024.0),
            num_grp, om->max_resident_groups(), new_max,
            (new_max >= num_grp) ? " (all fit)" : "");
    om->set_max_resident_groups(new_max);
}

// =============================================================================
// Internal helpers
// =============================================================================

void GenericQLoRAProvider::build_layer_offload_map() {
    mLayerOffloadGroups.clear();
    mHasOffloading = false;

    if (!mWeightMgr) {
        return;
    }

    for (const auto& name : mWeightMgr->weight_names()) {
        int offload_group = mWeightMgr->get_offload_group(name);
        if (offload_group < 0) {
            continue;
        }

        mHasOffloading = true;

        int layer_idx = parse_layer_index(name);
        if (layer_idx >= 0) {
            mLayerOffloadGroups[layer_idx].insert(offload_group);
        }
    }
}

}  // namespace qlora
