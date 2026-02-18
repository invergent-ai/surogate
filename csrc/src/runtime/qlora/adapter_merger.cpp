// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/qlora/adapter_merger.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <cuda_bf16.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "utilities/safetensors.h"
#include "utilities/tensor.h"

namespace fs = std::filesystem;

namespace qlora {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

AdapterMerger::AdapterMerger(
    const std::string& adapter_dir,
    const dsl::MappingTable& hf_mapping,
    SafeTensorsReader& base_reader,
    dsl::ShardConfig shard)
    : mAdapterReader(
          (fs::path(adapter_dir) / "adapter_model.safetensors").string())
    , mMapping(hf_mapping)
    , mBaseReader(base_reader)
    , mShard(shard) {

    // ---- Parse adapter_config.json ----
    const auto config_path = fs::path(adapter_dir) / "adapter_config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error(
            "AdapterMerger: adapter_config.json not found in " + adapter_dir);
    }

    nlohmann::json cfg;
    {
        std::ifstream f(config_path);
        f >> cfg;
    }

    mRank = cfg.at("r").get<int>();
    const float alpha = cfg.at("lora_alpha").get<float>();
    const bool use_rslora = cfg.value("use_rslora", false);

    if (use_rslora) {
        mScaling = alpha / std::sqrt(static_cast<float>(mRank));
    } else {
        mScaling = alpha / static_cast<float>(mRank);
    }

    // ---- Build adapter target index from safetensors entries ----
    // PEFT naming: base_model.model.{hf_base_name}.lora_A.weight
    //              base_model.model.{hf_base_name}.lora_B.weight
    // We scan for lora_A entries and derive lora_B keys.

    for (const auto& entry : mAdapterReader.entries()) {
        const auto& name = entry.name();

        // Look for ".lora_A.weight" suffix
        const std::string lora_a_suffix = ".lora_A.weight";
        if (name.size() <= lora_a_suffix.size()) continue;
        if (name.compare(name.size() - lora_a_suffix.size(),
                         lora_a_suffix.size(), lora_a_suffix) != 0) {
            continue;
        }

        // Extract base name by stripping prefix and suffix.
        // Try "base_model.model." prefix first, fall back to raw name.
        std::string base_name;
        const std::string peft_prefix = "base_model.model.";
        if (name.compare(0, peft_prefix.size(), peft_prefix) == 0) {
            base_name = name.substr(
                peft_prefix.size(),
                name.size() - peft_prefix.size() - lora_a_suffix.size());
        } else {
            base_name = name.substr(
                0, name.size() - lora_a_suffix.size());
        }

        // Construct lora_B key
        std::string lora_b_key;
        if (name.compare(0, peft_prefix.size(), peft_prefix) == 0) {
            lora_b_key = peft_prefix + base_name + ".lora_B.weight";
        } else {
            lora_b_key = base_name + ".lora_B.weight";
        }

        mAdapterTargets[base_name] = LoRAPair{name, lora_b_key};
    }

    if (mAdapterTargets.empty()) {
        throw std::runtime_error(
            "AdapterMerger: no LoRA adapter weights found in " + adapter_dir);
    }

    // ---- Create cuBLAS handle ----
    auto status = cublasCreate(&mCublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("AdapterMerger: cublasCreate failed");
    }

    fmt::print(stderr,
               "[AdapterMerger] Loaded adapter from {}: rank={}, alpha={}, "
               "scaling={:.4f}, {} target weights\n",
               adapter_dir, mRank, alpha, mScaling,
               mAdapterTargets.size());
}

AdapterMerger::~AdapterMerger() {
    if (mLoraABuf) cudaFree(mLoraABuf);
    if (mLoraBBuf) cudaFree(mLoraBBuf);
    if (mCublasHandle) cublasDestroy(mCublasHandle);
}

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

void AdapterMerger::ensure_buf(void*& buf, size_t& current_size, size_t needed) {
    if (needed <= current_size) return;
    if (buf) cudaFree(buf);
    auto err = cudaMalloc(&buf, needed);
    if (err != cudaSuccess) {
        buf = nullptr;
        current_size = 0;
        throw std::runtime_error(
            fmt::format("AdapterMerger: cudaMalloc({}) failed: {}",
                        needed, cudaGetErrorString(err)));
    }
    current_size = needed;
}

// ---------------------------------------------------------------------------
// Mapping resolution helpers
// ---------------------------------------------------------------------------

const dsl::MappingSpec* AdapterMerger::find_spec(
    const std::string& dsl_name, int& layer_idx) const {
    // Try exact match first.
    auto it = mMapping.find(dsl_name);
    if (it != mMapping.end()) {
        return &it->second;
    }

    // Try pattern match: "blocks[N].param" → "blocks[{layer}].param"
    std::string param_name;
    if (dsl::DslWeightLoader::parse_block_param(dsl_name, layer_idx, param_name)) {
        // Try several common block-type prefix patterns
        for (const char* prefix : {"blocks[{layer}].",
                                    "attn_blocks[{layer}].",
                                    "mamba_blocks[{layer}].",
                                    "mlp_blocks[{layer}].",
                                    "moe_blocks[{layer}]."}) {
            std::string pattern = std::string(prefix) + param_name;
            it = mMapping.find(pattern);
            if (it != mMapping.end()) {
                return &it->second;
            }
        }
    }

    return nullptr;
}

std::string AdapterMerger::strip_weight_suffix(const std::string& hf_name) {
    const std::string suffix = ".weight";
    if (hf_name.size() > suffix.size() &&
        hf_name.compare(hf_name.size() - suffix.size(),
                        suffix.size(), suffix) == 0) {
        return hf_name.substr(0, hf_name.size() - suffix.size());
    }
    return hf_name;
}

// ---------------------------------------------------------------------------
// Core merge: apply adapter delta for one HF component
// ---------------------------------------------------------------------------

void AdapterMerger::merge_component(
    const std::string& hf_base_name,
    Tensor& bf16_weight,
    long row_offset,
    long component_rows,
    cudaStream_t stream) {

    auto target_it = mAdapterTargets.find(hf_base_name);
    if (target_it == mAdapterTargets.end()) return;

    const auto& pair = target_it->second;

    // Load lora_A and lora_B shapes from the adapter safetensors.
    const auto& entry_a = mAdapterReader.find_entry(pair.lora_a_key);
    const auto& entry_b = mAdapterReader.find_entry(pair.lora_b_key);

    // lora_A: [rank, in_features], lora_B: [out_features, rank]
    const long R = entry_a.shape()[0];  // rank
    const long K = entry_a.shape()[1];  // in_features
    const long M = entry_b.shape()[0];  // out_features

    // For sharded weights: determine which slice of lora_B to use.
    long shard_row_start = 0;
    long shard_row_end = M;
    if (mShard.num_shards > 1 && component_rows > 0 && component_rows < M) {
        // Weight is row-sharded: this GPU owns rows [shard_start, shard_end)
        const long shard_size = M / mShard.num_shards;
        shard_row_start = shard_size * mShard.shard_idx;
        shard_row_end = shard_row_start + shard_size;
    }
    const long local_M = shard_row_end - shard_row_start;

    // Allocate/grow GPU scratch for lora_A [R, K] and lora_B [local_M, R]
    const size_t a_bytes = R * K * sizeof(nv_bfloat16);
    const size_t b_bytes = local_M * R * sizeof(nv_bfloat16);
    ensure_buf(mLoraABuf, mLoraABufBytes, a_bytes);
    ensure_buf(mLoraBBuf, mLoraBBufBytes, b_bytes);

    // Create Tensor views for the scratch buffers
    Tensor lora_a = Tensor::from_pointer(
        static_cast<std::byte*>(mLoraABuf), 0, ETensorDType::BF16,
        std::vector<long>{R, K});
    Tensor lora_b = Tensor::from_pointer(
        static_cast<std::byte*>(mLoraBBuf), 0, ETensorDType::BF16,
        std::vector<long>{local_M, R});

    // Load from adapter safetensors to GPU.
    if (shard_row_start == 0 && shard_row_end == M) {
        // No sharding: load full tensors
        entry_a.read_tensor(lora_a, true);
        entry_b.read_tensor(lora_b, true);
    } else {
        // Sharded: load full A, load slice of B
        entry_a.read_tensor(lora_a, true);
        entry_b.read_raw(lora_b, shard_row_start * R, local_M * R, true);
    }

    // cuBLAS GEMM: W[row_offset : row_offset+local_M, :K] += scaling * B @ A
    //
    // All matrices are row-major BF16. cuBLAS uses column-major, so:
    //   Row-major W[M,K] == column-major W^T[K,M]
    //   Row-major B[M,R] == column-major B^T[R,M]
    //   Row-major A[R,K] == column-major A^T[K,R]
    //
    // We want: W += scaling * B @ A  (row-major)
    // In column-major: W^T = W^T + scaling * A^T @ B^T
    //   = cublasGemmEx(N, N, K, local_M, R, &scaling, A^T, K, B^T, R, &one, W^T, lda)
    //
    // lda for W is the leading dimension of the full weight (= K = number of columns in row-major).

    cublasSetStream(mCublasHandle, stream);

    const float alpha_val = mScaling;
    const float beta_val = 1.0f;

    const long lda_W = bf16_weight.Rank >= 2 ? bf16_weight.Sizes[bf16_weight.Rank - 1] : 1;
    auto* W_ptr = reinterpret_cast<nv_bfloat16*>(bf16_weight.Data)
                  + row_offset * lda_W;
    auto* A_ptr = reinterpret_cast<nv_bfloat16*>(mLoraABuf);
    auto* B_ptr = reinterpret_cast<nv_bfloat16*>(mLoraBBuf);

    auto gemm_status = cublasGemmEx(
        mCublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(K),       // m (columns of C in col-major)
        static_cast<int>(local_M), // n (rows of C in col-major)
        static_cast<int>(R),       // k (inner dimension)
        &alpha_val,
        A_ptr, CUDA_R_16BF, static_cast<int>(K),       // A^T [K, R]
        B_ptr, CUDA_R_16BF, static_cast<int>(R),       // B^T [R, M]
        &beta_val,
        W_ptr, CUDA_R_16BF, static_cast<int>(lda_W),   // W^T [K, M]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(fmt::format(
            "AdapterMerger: cuBLAS GEMM failed for '{}' (status={})",
            hf_base_name, static_cast<int>(gemm_status)));
    }
}

// ---------------------------------------------------------------------------
// Public: apply adapter to a DSL parameter
// ---------------------------------------------------------------------------

void AdapterMerger::apply(
    const std::string& dsl_param_name,
    Tensor& bf16_weight,
    cudaStream_t stream) {

    int layer_idx = -1;
    const auto* spec = find_spec(dsl_param_name, layer_idx);
    if (!spec) return;  // No mapping → nothing to merge

    switch (spec->kind) {
        case dsl::MappingSpec::Kind::Direct: {
            // Single HF tensor → single merge
            std::string hf_name = dsl::DslWeightLoader::format_hf_name(
                spec->source, layer_idx);
            std::string base_name = strip_weight_suffix(hf_name);
            const long M = bf16_weight.Sizes[0];
            merge_component(base_name, bf16_weight, 0, M, stream);
            break;
        }

        case dsl::MappingSpec::Kind::Fuse: {
            // Multiple HF tensors concatenated along dim 0.
            // Apply each component's delta at the correct row offset.
            long offset = 0;
            for (const auto& src : spec->sources) {
                std::string hf_name = dsl::DslWeightLoader::format_hf_name(
                    src, layer_idx);
                std::string base_name = strip_weight_suffix(hf_name);

                // Determine this component's row count from the base model.
                long component_rows = 0;
                try {
                    const auto& base_entry = mBaseReader.find_entry(hf_name);
                    component_rows = base_entry.shape()[0];
                    // Handle sharding: if weight is sharded, component_rows is the global size
                    if (mShard.num_shards > 1) {
                        component_rows = component_rows / mShard.num_shards;
                    }
                } catch (const std::out_of_range&) {
                    // Component not in base model (optional) — skip
                    continue;
                }

                merge_component(base_name, bf16_weight, offset,
                                component_rows, stream);
                offset += component_rows;
            }
            break;
        }

        case dsl::MappingSpec::Kind::Transform: {
            // Transform mapping: merge before transform would be applied.
            // The weight is already loaded and transformed, so we apply the
            // delta to the transformed tensor. For transpose transforms,
            // this means the delta needs to match the transformed shape.
            // For now, treat like Direct (most transforms are simple transpose).
            std::string hf_name = dsl::DslWeightLoader::format_hf_name(
                spec->source, layer_idx);
            std::string base_name = strip_weight_suffix(hf_name);
            const long M = bf16_weight.Sizes[0];
            merge_component(base_name, bf16_weight, 0, M, stream);
            break;
        }

        case dsl::MappingSpec::Kind::Split: {
            // Split mapping: one HF tensor sliced into multiple DSL params.
            // The adapter targets the full HF weight; we need to apply only
            // the slice that corresponds to this param.
            std::string hf_name = dsl::DslWeightLoader::format_hf_name(
                spec->source, layer_idx);
            std::string base_name = strip_weight_suffix(hf_name);

            auto target_it = mAdapterTargets.find(base_name);
            if (target_it == mAdapterTargets.end()) break;

            // For Split, we need the full delta then slice.
            // But this is rare — skip for now with a warning.
            fmt::print(stderr,
                       "[AdapterMerger] Warning: Split mapping for '{}' "
                       "not supported yet, skipping adapter merge\n",
                       dsl_param_name);
            break;
        }

        case dsl::MappingSpec::Kind::TiedTo:
            // TiedTo params are copies of other params that are already merged.
            break;

        case dsl::MappingSpec::Kind::StackExperts:
            // Expert weights handled by apply_expert() in the per-expert loop.
            break;

        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Public: apply adapter to a single expert
// ---------------------------------------------------------------------------

void AdapterMerger::apply_expert(
    const std::string& dsl_param_name,
    int expert_idx,
    Tensor& bf16_expert,
    cudaStream_t stream) {

    int layer_idx = -1;
    const auto* spec = find_spec(dsl_param_name, layer_idx);
    if (!spec) return;

    if (spec->kind != dsl::MappingSpec::Kind::StackExperts) return;

    // For StackExperts, the source template contains {expert} placeholder.
    // Resolve both {layer} and {expert} to get the concrete HF name.

    if (spec->fuse_gate_up) {
        // Fused gate_up: two HF components per expert.
        // Source has pattern like "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"
        // and there's a corresponding gate_proj source.
        // The fuse_gate_up pattern stores the up_proj source; gate_proj is derived.
        std::string up_hf = dsl::DslWeightLoader::format_hf_name(
            spec->source, layer_idx, expert_idx);
        std::string up_base = strip_weight_suffix(up_hf);

        // Derive gate_proj name from up_proj
        std::string gate_hf = up_hf;
        auto pos = gate_hf.find("up_proj");
        if (pos != std::string::npos) {
            gate_hf.replace(pos, 7, "gate_proj");
        }
        std::string gate_base = strip_weight_suffix(gate_hf);

        // Expert buffer is [2*intermediate, hidden] with up in first half, gate in second.
        const long half_rows = bf16_expert.Sizes[0] / 2;
        merge_component(up_base, bf16_expert, 0, half_rows, stream);
        merge_component(gate_base, bf16_expert, half_rows, half_rows, stream);
    } else {
        // Simple: one HF tensor per expert.
        std::string hf_name = dsl::DslWeightLoader::format_hf_name(
            spec->source, layer_idx, expert_idx);
        std::string base_name = strip_weight_suffix(hf_name);
        const long M = bf16_expert.Sizes[0];
        merge_component(base_name, bf16_expert, 0, M, stream);
    }
}

}  // namespace qlora
