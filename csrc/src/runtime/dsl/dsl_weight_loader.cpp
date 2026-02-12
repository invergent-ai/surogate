// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DslWeightLoader implementation.
//
// Handles all weight loading logic: Direct, Fuse, Split, Transform,
// TiedTo, and StackExperts mapping kinds. Each handler is extracted
// from the original DslModel::import_weights() into a self-contained
// method that can operate without DslModel state.

#include "runtime/dsl/dsl_weight_loader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "config/pretrained_config.h"
#include "kernels/kernels.h"
#include "utilities/allocator.h"
#include "utilities/dtype.h"
#include "utilities/safetensors.h"
#include "utilities/utils.h"

namespace dsl {

// ============================================================================
// Construction / Destruction
// ============================================================================

DslWeightLoader::DslWeightLoader(SafeTensorsReader& reader,
                                 const MappingTable& mapping,
                                 const PretrainedConfig& config,
                                 TensorAllocator& allocator,
                                 ShardConfig shard,
                                 MoEWeightConfig moe_config)
    : mReader(reader)
    , mMapping(mapping)
    , mConfig(config)
    , mAllocator(allocator)
    , mShard(shard)
    , mMoEConfig(moe_config) {}

DslWeightLoader::~DslWeightLoader() = default;

// ============================================================================
// Public API
// ============================================================================

bool DslWeightLoader::load_param(const std::string& name,
                                 Tensor& target,
                                 bool allow_cast,
                                 bool param_sharded,
                                 const Tensor* global_template,
                                 cudaStream_t stream) {
    int layer_idx = -1;
    const MappingSpec* spec = find_mapping_spec(name, layer_idx);

    // No explicit mapping: fall back to Direct with the internal name as HF path.
    MappingSpec direct_fallback;
    if (!spec) {
        direct_fallback.kind = MappingSpec::Kind::Direct;
        direct_fallback.source = name;
        spec = &direct_fallback;
    }

    switch (spec->kind) {
        case MappingSpec::Kind::TiedTo:
            mTiedParams.emplace_back(name, spec->target);
            return true;

        case MappingSpec::Kind::Direct:
            return load_direct(*spec, name, target, layer_idx, allow_cast, param_sharded, global_template);

        case MappingSpec::Kind::Fuse:
            return load_fused(*spec, name, target, layer_idx, allow_cast, param_sharded, global_template);

        case MappingSpec::Kind::Split:
            return load_split(*spec, name, target, layer_idx, allow_cast, param_sharded, global_template);

        case MappingSpec::Kind::Transform:
            return load_transform(*spec, name, target, layer_idx, allow_cast, param_sharded, global_template, stream);

        case MappingSpec::Kind::StackExperts:
            return load_stack_experts(*spec, name, target, layer_idx, allow_cast, param_sharded, stream);

        case MappingSpec::Kind::Unknown:
            throw std::runtime_error("DslWeightLoader: unsupported mapping kind for '" + name + "'");
    }

    return false;
}

bool DslWeightLoader::try_multiple(const std::vector<std::pair<std::string, MappingSpec>>& options,
                                   const std::string& name,
                                   Tensor& target,
                                   bool allow_cast,
                                   int layer_idx,
                                   bool param_sharded,
                                   const Tensor* global_template,
                                   cudaStream_t stream) {
    bool any_required = false;

    for (const auto& [key, spec] : options) {
        if (!spec.optional) {
            any_required = true;
        }

        // Create a temporarily-optional version of the spec for trial loading.
        MappingSpec trial = spec;
        trial.optional = true;

        bool success = false;
        switch (trial.kind) {
            case MappingSpec::Kind::Direct:
                success = load_direct(trial, name, target, layer_idx, allow_cast, param_sharded, global_template);
                break;
            case MappingSpec::Kind::Fuse:
                success = load_fused(trial, name, target, layer_idx, allow_cast, param_sharded, global_template);
                break;
            case MappingSpec::Kind::Split:
                success = load_split(trial, name, target, layer_idx, allow_cast, param_sharded, global_template);
                break;
            case MappingSpec::Kind::Transform:
                success = load_transform(trial, name, target, layer_idx, allow_cast, param_sharded, global_template, stream);
                break;
            case MappingSpec::Kind::StackExperts:
                success = load_stack_experts(trial, name, target, layer_idx, allow_cast, param_sharded, stream);
                break;
            case MappingSpec::Kind::TiedTo:
                mTiedParams.emplace_back(name, trial.target);
                return true;
            case MappingSpec::Kind::Unknown:
                break;
        }

        if (success) {
            return true;
        }
    }

    if (any_required) {
        throw std::runtime_error("DslWeightLoader: all options failed for '" + name + "'");
    }
    return false;
}

bool DslWeightLoader::load_expert(const std::string& name, int expert_idx,
                                   Tensor& target, bool allow_cast, cudaStream_t stream) {
    int layer_idx = -1;
    const MappingSpec* spec = find_mapping_spec(name, layer_idx);

    if (!spec) {
        throw std::runtime_error("DslWeightLoader: load_expert missing mapping for '" + name + "'");
    }

    const std::size_t elem_size = get_dtype_size(target.DType);
    if (!stream) {
        stream = cudaStreamDefault;
    }
    auto get_expert_scratch = [&](ETensorDType dtype, const std::vector<long>& shape) -> Tensor {
        std::size_t elems = 1;
        for (long dim : shape) {
            elems *= static_cast<std::size_t>(dim);
        }
        const std::size_t bytes = elems * get_dtype_size(dtype);
        if (!mExpertScratch.Data || mExpertScratchBytes < bytes || mExpertScratch.DType != dtype) {
            mExpertScratch = mAllocator.allocate(dtype, "wl_tmp_expert",
                                                 EAllocationType::ON_DEVICE, shape);
            mExpertScratchBytes = mExpertScratch.bytes();
        }
        Tensor scratch = mExpertScratch;
        scratch.DType = dtype;
        scratch.Rank = static_cast<int>(shape.size());
        for (int i = 0; i < scratch.Rank; ++i) {
            scratch.Sizes[i] = shape[i];
        }
        for (int i = scratch.Rank; i < MAX_TENSOR_DIM; ++i) {
            scratch.Sizes[i] = 1;
        }
        return scratch;
    };

    if (spec->kind == MappingSpec::Kind::StackExperts && spec->fuse_gate_up) {
        // Fused gate_up: target is [2*D, C], load up_proj into first D rows, gate into next D.
        std::string gate_pattern = spec->source;
        std::string up_pattern = spec->source;
        std::size_t pos = up_pattern.find("gate_proj");
        if (pos != std::string::npos) {
            up_pattern.replace(pos, 9, "up_proj");
        }

        const long fused_rows = target.Sizes[0];  // 2*D
        const long D = fused_rows / 2;
        const long C = target.Rank >= 2 ? target.Sizes[1] : 1;
        const long sub_expert_elems = D * C;

        // Load up_proj into first D rows.
        std::string up_hf = format_hf_name(up_pattern, layer_idx, expert_idx);
        const auto& up_entry = mReader.find_entry(up_hf);
        if (up_entry.shape().empty()) {
            throw std::runtime_error("DslWeightLoader: load_expert missing up_proj '" + up_hf + "'");
        }
        Tensor up_slice = target;
        up_slice.Sizes[0] = D;
        up_entry.read_tensor(up_slice, allow_cast);

        // Load gate_proj into second D rows.
        std::string gate_hf = format_hf_name(gate_pattern, layer_idx, expert_idx);
        const auto& gate_entry = mReader.find_entry(gate_hf);
        if (gate_entry.shape().empty()) {
            throw std::runtime_error("DslWeightLoader: load_expert missing gate_proj '" + gate_hf + "'");
        }
        Tensor gate_slice = target;
        gate_slice.Sizes[0] = D;
        gate_slice.Data = static_cast<std::byte*>(target.Data) + sub_expert_elems * elem_size;
        gate_entry.read_tensor(gate_slice, allow_cast);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }

    if (spec->kind == MappingSpec::Kind::StackExperts) {
        // Load single expert tensor from per-expert HF entries.
        std::string hf_name = format_hf_name(spec->source, layer_idx, expert_idx);
        const auto& entry = mReader.find_entry(hf_name);
        if (entry.shape().empty()) {
            throw std::runtime_error("DslWeightLoader: load_expert missing '" + hf_name + "'");
        }
        entry.read_tensor(target, allow_cast);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }

    if (spec->kind != MappingSpec::Kind::Direct && spec->kind != MappingSpec::Kind::Transform) {
        throw std::runtime_error(
            "DslWeightLoader: load_expert unsupported mapping kind for '" + name + "'");
    }
    if (spec->kind == MappingSpec::Kind::Transform && spec->fn != "transpose") {
        throw std::runtime_error("DslWeightLoader: load_expert unsupported transform '" + spec->fn
                                 + "' for '" + name + "'");
    }

    const bool has_expert_placeholder =
        spec->source.find("{expert}") != std::string::npos;
    std::string hf_name = format_hf_name(
        spec->source.empty() ? name : spec->source,
        layer_idx,
        has_expert_placeholder ? expert_idx : -1);

    const SafeTensorEntry* entry_ptr = nullptr;
    try {
        entry_ptr = &mReader.find_entry(hf_name);
    } catch (const std::out_of_range&) {
        if (spec->optional) {
            return false;
        }
        throw std::runtime_error("DslWeightLoader: load_expert missing '" + hf_name + "'");
    }
    const auto& entry = *entry_ptr;
    const auto& shape = entry.shape();
    if (shape.empty()) {
        if (spec->optional) {
            return false;
        }
        throw std::runtime_error("DslWeightLoader: load_expert missing '" + hf_name + "'");
    }

    // Per-expert tensors with {expert} in the name.
    if (has_expert_placeholder) {
        if (spec->kind == MappingSpec::Kind::Direct) {
            entry.read_tensor(target, allow_cast);
        } else {
            if (shape.size() != 2 || target.Rank != 2) {
                throw std::runtime_error("DslWeightLoader: load_expert transpose expects 2D tensor for '" + name + "'");
            }
            Tensor tmp = get_expert_scratch(target.DType, {shape.at(0), shape.at(1)});
            entry.read_tensor(tmp, allow_cast);
            transpose(target, tmp, static_cast<int>(shape.at(0)),
                      static_cast<int>(shape.at(1)), stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }

    // Batched expert tensors [E, ...] without {expert} placeholder.
    if (shape.size() < 2) {
        throw std::runtime_error("DslWeightLoader: load_expert expects expert-batched tensor for '" + name + "'");
    }
    if (expert_idx < 0 || expert_idx >= shape.at(0)) {
        throw std::runtime_error("DslWeightLoader: load_expert index out of range for '" + name + "'");
    }

    long per_expert_elems = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        per_expert_elems *= shape[i];
    }
    if (static_cast<long>(target.nelem()) != per_expert_elems) {
        throw std::runtime_error("DslWeightLoader: load_expert size mismatch for '" + name + "'");
    }
    const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(expert_idx) * per_expert_elems;

    if (spec->kind == MappingSpec::Kind::Direct) {
        entry.read_raw(target, offset, per_expert_elems, allow_cast);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }

    // Transform transpose for batched [E, A, B] -> [B, A]
    if (shape.size() != 3 || target.Rank != 2) {
        throw std::runtime_error("DslWeightLoader: load_expert transpose expects 3D source for '" + name + "'");
    }
    const long A = shape[1];
    const long B = shape[2];
    if (target.Sizes[0] != B || target.Sizes[1] != A) {
        throw std::runtime_error("DslWeightLoader: load_expert transpose shape mismatch for '" + name + "'");
    }
    Tensor tmp = get_expert_scratch(target.DType, {A, B});
    entry.read_raw(tmp, offset, per_expert_elems, allow_cast);
    transpose(target, tmp, static_cast<int>(A), static_cast<int>(B), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

void DslWeightLoader::resolve_tied_params(const std::function<Tensor&(const std::string&)>& get_tensor) {
    for (const auto& [dst_name, src_name] : mTiedParams) {
        Tensor& dst = get_tensor(dst_name);
        Tensor& src = get_tensor(src_name);
        if (dst.bytes() != src.bytes()) {
            throw std::runtime_error("DslWeightLoader: tied param size mismatch: '"
                                     + dst_name + "' vs '" + src_name + "'");
        }
        CUDA_CHECK(cudaMemcpy(dst.Data, src.Data, src.bytes(), cudaMemcpyDeviceToDevice));
    }
}

bool DslWeightLoader::has_entry(const std::string& hf_name) const {
    try {
        mReader.find_entry(hf_name);
        return true;
    } catch (const std::out_of_range&) {
        return false;
    }
}

// ============================================================================
// Direct Loading
// ============================================================================

bool DslWeightLoader::load_direct(const MappingSpec& spec, const std::string& name,
                                  Tensor& target, int layer_idx, bool allow_cast,
                                  bool param_sharded, const Tensor* global_template) {
    const std::string hf_name = format_hf_name(
        spec.source.empty() ? name : spec.source, layer_idx);

    // Look up the HF tensor entry.
    const SafeTensorEntry* entry_ptr = nullptr;
    try {
        entry_ptr = &mReader.find_entry(hf_name);
    } catch (const std::out_of_range&) {
        // Handle tied embeddings fallback: if lm_head.weight is missing, try embed_tokens.
        if (mConfig.TiedWordEmbeddings && hf_name == "lm_head.weight") {
            std::vector<std::string> candidates;
            int emb_layer_idx = -1;
            if (const auto* emb_spec = find_mapping_spec("embedding", emb_layer_idx)) {
                if (emb_spec->kind == MappingSpec::Kind::Direct && !emb_spec->source.empty()) {
                    candidates.push_back(format_hf_name(emb_spec->source, emb_layer_idx));
                }
            }
            candidates.push_back("model.embed_tokens.weight");
            candidates.push_back("model.language_model.embed_tokens.weight");

            for (const auto& candidate : candidates) {
                if (candidate.empty()) {
                    continue;
                }
                try {
                    entry_ptr = &mReader.find_entry(candidate);
                    break;
                } catch (const std::out_of_range&) {
                    continue;
                }
            }

            if (!entry_ptr) {
                if (spec.optional) return false;
                throw std::runtime_error("DslWeightLoader: missing HF tensor '" + hf_name
                                         + "' (and tied fallback) for param '" + name + "'");
            }
        } else {
            if (spec.optional) return false;
            throw std::runtime_error("DslWeightLoader: missing HF tensor '" + hf_name
                                     + "' for param '" + name + "'");
        }
    }
    const auto& entry = *entry_ptr;

    if (!param_sharded) {
        // Handle squeezable shape mismatch (e.g. HF Conv1d weight [D,1,K] → DSL [D,K]).
        // When the file has extra size-1 dimensions, element counts match and read_raw is safe.
        const auto& file_shape = entry.shape();

        if (static_cast<int>(file_shape.size()) != target.Rank) {
            // Check if squeezing size-1 dims yields the target shape.
            std::vector<long> squeezed;
            for (long d : file_shape) {
                if (d != 1) squeezed.push_back(d);
            }
            bool match = (static_cast<int>(squeezed.size()) == target.Rank);
            for (int i = 0; match && i < target.Rank; ++i) {
                if (squeezed[i] != target.Sizes[i]) match = false;
            }
            if (!match) {
                throw std::runtime_error("DslWeightLoader: shape mismatch for '" + hf_name
                    + "': file shape cannot be squeezed to target shape for param '" + name + "'");
            }
            entry.read_raw(target, 0, target.nelem(), allow_cast);
        } else {
            entry.read_tensor(target, allow_cast);
        }
    } else {
        if (!global_template) {
            throw std::runtime_error("DslWeightLoader: sharded param '" + name
                                     + "' requires global_template");
        }
        const long global_rows = global_template->Sizes[0];
        auto [start, end] = shard_range(global_rows, true);
        const long stride = row_stride(entry.shape());
        (void)end;
        entry.read_raw(target, static_cast<std::ptrdiff_t>(start) * stride, target.nelem(), allow_cast);
    }
    return true;
}

// ============================================================================
// Fuse Loading (concatenate multiple HF tensors along dim0)
// ============================================================================

bool DslWeightLoader::load_fused(const MappingSpec& spec, const std::string& name,
                                 Tensor& target, int layer_idx, bool allow_cast,
                                 bool param_sharded, const Tensor* global_template) {
    if (spec.dim != 0) {
        throw std::runtime_error("DslWeightLoader: fuse mapping only supports dim=0 for '" + name + "'");
    }

    // Infer slice sizes for the fused tensor.
    std::vector<long> slice_sizes = infer_fuse_slices(name, static_cast<int>(spec.sources.size()));
    if (slice_sizes.empty()) {
        // Fall back to equal-sized chunks.
        const Tensor& shape_ref = global_template ? *global_template : target;
        if (shape_ref.Sizes[0] % static_cast<long>(spec.sources.size()) == 0) {
            const long chunk = shape_ref.Sizes[0] / static_cast<long>(spec.sources.size());
            slice_sizes.assign(spec.sources.size(), chunk);
        } else {
            if (spec.optional) return false;
            throw std::runtime_error("DslWeightLoader: cannot infer fuse slices for '" + name + "'");
        }
    } else if (slice_sizes.size() != spec.sources.size()) {
        throw std::runtime_error("DslWeightLoader: fuse slice count mismatch for '" + name + "'");
    }

    // Check that at least one source exists (for optional specs).
    if (spec.optional) {
        const std::string first_hf = format_hf_name(spec.sources.front(), layer_idx);
        if (!has_entry(first_hf)) {
            return false;
        }
    }

    const Tensor& global = global_template ? *global_template : target;
    const long global_rows = global.Sizes[0];
    auto [shard_start, shard_end] = shard_range(global_rows, param_sharded);

    long offset = 0;
    for (std::size_t i = 0; i < spec.sources.size(); ++i) {
        const std::string hf_name = format_hf_name(spec.sources[i], layer_idx);
        const auto& entry = mReader.find_entry(hf_name);

        if (entry.shape().empty()) {
            throw std::runtime_error("DslWeightLoader: empty shape for '" + hf_name + "'");
        }
        if (static_cast<int>(entry.shape().size()) != target.Rank) {
            throw std::runtime_error("DslWeightLoader: rank mismatch for '" + hf_name + "'");
        }
        for (int j = 1; j < target.Rank; ++j) {
            if (entry.shape().at(j) != global.Sizes[j]) {
                throw std::runtime_error("DslWeightLoader: shape mismatch for '" + hf_name + "'");
            }
        }

        const long slice_len = slice_sizes.at(i);

        if (!param_sharded) {
            Tensor slice = slice_dim0(target, offset, slice_len);
            entry.read_raw(slice, 0, slice.nelem(), allow_cast);
            offset += slice_len;
            continue;
        }

        // Sharded: compute overlap between this source slice and our shard range.
        const long src_begin = offset;
        const long src_end = offset + slice_len;
        const long overlap_begin = std::max(src_begin, shard_start);
        const long overlap_end = std::min(src_end, shard_end);
        if (overlap_begin < overlap_end) {
            const long rows = overlap_end - overlap_begin;
            const long dst_row_offset = overlap_begin - shard_start;
            const long src_row_offset = overlap_begin - src_begin;
            Tensor slice = slice_dim0(target, dst_row_offset, rows);
            const long stride = row_stride(entry.shape());
            entry.read_raw(slice, static_cast<std::ptrdiff_t>(src_row_offset) * stride,
                           slice.nelem(), allow_cast);
        }
        offset += slice_len;
    }

    if (offset != global_rows) {
        throw std::runtime_error("DslWeightLoader: fuse mapping size mismatch for '" + name
                                 + "' (got " + std::to_string(offset) + " rows, expected "
                                 + std::to_string(global_rows) + ")");
    }
    return true;
}

// ============================================================================
// Split Loading (extract sub-range from one HF tensor)
// ============================================================================

bool DslWeightLoader::load_split(const MappingSpec& spec, const std::string& name,
                                 Tensor& target, int layer_idx, bool allow_cast,
                                 bool param_sharded, const Tensor* global_template) {
    if (spec.dim != 0) {
        throw std::runtime_error("DslWeightLoader: split mapping only supports dim=0 for '" + name + "'");
    }
    if (spec.ranges.empty()) {
        throw std::runtime_error("DslWeightLoader: split mapping missing ranges for '" + name + "'");
    }

    auto [start, end] = spec.ranges.front();
    if (start < 0 || end <= start) {
        throw std::runtime_error("DslWeightLoader: unsupported split range for '" + name + "'");
    }

    const long expected = end - start;
    const std::string hf_name = format_hf_name(spec.source, layer_idx);

    const SafeTensorEntry* entry_ptr = nullptr;
    try {
        entry_ptr = &mReader.find_entry(hf_name);
    } catch (const std::out_of_range&) {
        if (spec.optional) return false;
        throw std::runtime_error("DslWeightLoader: missing HF tensor '" + hf_name
                                 + "' for split param '" + name + "'");
    }
    const auto& entry = *entry_ptr;

    if (!param_sharded) {
        if (target.Sizes[0] != expected) {
            throw std::runtime_error("DslWeightLoader: split range size mismatch for '" + name + "'");
        }
        long stride = 1;
        for (int i = 1; i < target.Rank; ++i) {
            stride *= target.Sizes[i];
        }
        const std::ptrdiff_t byte_offset = static_cast<std::ptrdiff_t>(start) * stride;
        entry.read_raw(target, byte_offset, target.nelem(), allow_cast);
    } else {
        const long shard_rows = expected / mShard.num_shards;
        if (expected % mShard.num_shards != 0 || target.Sizes[0] != shard_rows) {
            throw std::runtime_error("DslWeightLoader: split shard size mismatch for '" + name + "'");
        }
        const long local_start = start + shard_rows * mShard.shard_idx;
        const long stride = row_stride(entry.shape());
        entry.read_raw(target, static_cast<std::ptrdiff_t>(local_start) * stride,
                       target.nelem(), allow_cast);
    }
    return true;
}

// ============================================================================
// Transform Loading (load + apply function, e.g. transpose)
// ============================================================================

bool DslWeightLoader::load_transform(const MappingSpec& spec, const std::string& name,
                                     Tensor& target, int layer_idx, bool allow_cast,
                                     bool param_sharded, const Tensor* global_template,
                                     cudaStream_t stream) {
    if (spec.fn != "transpose") {
        throw std::runtime_error("DslWeightLoader: unsupported transform '" + spec.fn
                                 + "' for '" + name + "'");
    }

    const std::string hf_name = format_hf_name(spec.source, layer_idx);

    const SafeTensorEntry* entry_ptr = nullptr;
    try {
        entry_ptr = &mReader.find_entry(hf_name);
    } catch (const std::out_of_range&) {
        if (spec.optional) return false;
        throw std::runtime_error("DslWeightLoader: missing HF tensor '" + hf_name
                                 + "' for transform param '" + name + "'");
    }
    const auto& entry = *entry_ptr;

    if (entry.shape().size() != 2 || target.Rank != 2) {
        throw std::runtime_error("DslWeightLoader: transpose expects 2D tensors for '" + name + "'");
    }

    if (!stream) {
        stream = cudaStreamDefault;
    }

    if (!param_sharded) {
        // Allocate temporary for the untransposed data.
        Tensor tmp = mAllocator.allocate(target.DType, ("wl_tmp_" + name).c_str(),
                                         EAllocationType::ON_DEVICE,
                                         {entry.shape().at(0), entry.shape().at(1)});
        entry.read_tensor(tmp, allow_cast);
        transpose(target, tmp, static_cast<int>(entry.shape().at(0)),
                  static_cast<int>(entry.shape().at(1)), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        if (!global_template) {
            throw std::runtime_error("DslWeightLoader: sharded transform param '" + name
                                     + "' requires global_template");
        }
        const long global_rows = global_template->Sizes[0];
        auto [shard_start, shard_end] = shard_range(global_rows, true);

        if (target.Sizes[0] != (shard_end - shard_start)) {
            throw std::runtime_error("DslWeightLoader: transpose shard size mismatch for '" + name + "'");
        }

        // Allocate temporary for source (untransposed) and full transposed result.
        Tensor tmp_src = mAllocator.allocate(target.DType, ("wl_tmp_src_" + name).c_str(),
                                             EAllocationType::ON_DEVICE,
                                             {entry.shape().at(0), entry.shape().at(1)});
        Tensor tmp_full = mAllocator.allocate(target.DType, ("wl_tmp_full_" + name).c_str(),
                                              EAllocationType::ON_DEVICE,
                                              {global_template->Sizes[0], global_template->Sizes[1]});
        entry.read_tensor(tmp_src, allow_cast);
        transpose(tmp_full, tmp_src, static_cast<int>(entry.shape().at(0)),
                  static_cast<int>(entry.shape().at(1)), stream);

        // Copy our shard slice from the full transposed result.
        Tensor full_slice = slice_dim0(tmp_full, shard_start, shard_end - shard_start);
        CUDA_CHECK(cudaMemcpyAsync(target.Data, full_slice.Data, target.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return true;
}

// ============================================================================
// StackExperts Loading (per-expert HF tensors -> batched [E, ...])
// ============================================================================

bool DslWeightLoader::load_stack_experts(const MappingSpec& spec, const std::string& name,
                                         Tensor& target, int layer_idx, bool allow_cast,
                                         bool param_sharded, cudaStream_t stream) {
    if (spec.source.empty()) {
        throw std::runtime_error("DslWeightLoader: stack_experts missing pattern for '" + name + "'");
    }

    const int num_experts = resolve_num_experts(spec);
    if (num_experts <= 0) {
        throw std::runtime_error("DslWeightLoader: stack_experts cannot determine num_experts for '" + name + "'");
    }

    if (target.Rank < 1 || target.Sizes[0] < num_experts) {
        throw std::runtime_error("DslWeightLoader: stack_experts param rank/size mismatch for '" + name + "'");
    }

    // Check that at least the first expert exists (for optional specs).
    if (spec.optional) {
        const std::string first_hf = format_hf_name(spec.source, layer_idx, 0);
        if (!has_entry(first_hf)) {
            return false;
        }
    }

    const long expert_size = target.nelem() / target.Sizes[0];
    const std::size_t elem_size = get_dtype_size(target.DType);

    if (!stream) {
        stream = cudaStreamDefault;
    }

    // Determine which experts this shard owns.
    const int experts_per_shard = param_sharded ? (num_experts / mShard.num_shards) : num_experts;
    const int expert_start = param_sharded ? (mShard.shard_idx * experts_per_shard) : 0;
    const int expert_end = param_sharded ? (expert_start + experts_per_shard) : num_experts;

    if (spec.fuse_gate_up) {
        // Fused gate_up layout: [E, 2*D, C] where first D rows are up, next D rows are gate.
        std::string gate_pattern = spec.source;
        std::string up_pattern = spec.source;
        std::size_t pos = up_pattern.find("gate_proj");
        if (pos != std::string::npos) {
            up_pattern.replace(pos, 9, "up_proj");
        }

        const long fused_rows = target.Rank >= 2 ? target.Sizes[1] : 1;
        const long D = fused_rows / 2;
        const long C = target.Rank >= 3 ? target.Sizes[2] : 1;
        const long sub_expert_elems = D * C;

        for (int e = expert_start; e < expert_end; ++e) {
            const int local_e = e - expert_start;
            const std::size_t base_offset = static_cast<std::size_t>(local_e) * fused_rows * C * elem_size;

            // Load up_proj into first D rows.
            std::string up_hf = format_hf_name(up_pattern, layer_idx, e);
            const auto& up_entry = mReader.find_entry(up_hf);
            if (up_entry.shape().empty()) {
                throw std::runtime_error("DslWeightLoader: stack_experts missing up_proj '" + up_hf + "'");
            }
            Tensor up_slice = target;
            up_slice.Rank = 2;
            up_slice.Sizes[0] = D;
            up_slice.Sizes[1] = C;
            up_slice.Data = static_cast<std::byte*>(target.Data) + base_offset;
            up_entry.read_tensor(up_slice, allow_cast);

            // Load gate_proj into second D rows.
            std::string gate_hf = format_hf_name(gate_pattern, layer_idx, e);
            const auto& gate_entry = mReader.find_entry(gate_hf);
            if (gate_entry.shape().empty()) {
                throw std::runtime_error("DslWeightLoader: stack_experts missing gate_proj '" + gate_hf + "'");
            }
            Tensor gate_slice = target;
            gate_slice.Rank = 2;
            gate_slice.Sizes[0] = D;
            gate_slice.Sizes[1] = C;
            gate_slice.Data = static_cast<std::byte*>(target.Data) + base_offset + sub_expert_elems * elem_size;
            gate_entry.read_tensor(gate_slice, allow_cast);
        }
    } else {
        // Load single tensor per expert (e.g. down_proj).
        for (int e = expert_start; e < expert_end; ++e) {
            const int local_e = e - expert_start;
            std::string hf_name = format_hf_name(spec.source, layer_idx, e);
            const auto& entry = mReader.find_entry(hf_name);
            if (entry.shape().empty()) {
                throw std::runtime_error("DslWeightLoader: stack_experts missing '" + hf_name + "'");
            }

            // Create a view for this expert's slice.
            Tensor slice = target;
            slice.Rank = target.Rank - 1;
            for (int d = 0; d < slice.Rank; ++d) {
                slice.Sizes[d] = target.Sizes[d + 1];
            }
            slice.Data = static_cast<std::byte*>(target.Data)
                       + static_cast<std::size_t>(local_e) * expert_size * elem_size;
            entry.read_tensor(slice, allow_cast);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

// ============================================================================
// Utility Methods
// ============================================================================

std::pair<long, long> DslWeightLoader::shard_range(long global_rows, bool param_sharded) const {
    if (!param_sharded || mShard.num_shards <= 1) {
        return {0, global_rows};
    }
    if (global_rows % mShard.num_shards != 0) {
        throw std::runtime_error("DslWeightLoader: sharded load requires dim0 ("
                                 + std::to_string(global_rows) + ") divisible by num_shards ("
                                 + std::to_string(mShard.num_shards) + ")");
    }
    const long shard_rows = global_rows / mShard.num_shards;
    const long start = shard_rows * mShard.shard_idx;
    return {start, start + shard_rows};
}

long DslWeightLoader::row_stride(const std::vector<long>& shape) {
    long stride = 1;
    for (std::size_t i = 1; i < shape.size(); ++i) {
        stride *= shape[i];
    }
    return stride;
}

Tensor DslWeightLoader::slice_dim0(const Tensor& base, long offset, long length) {
    Tensor slice = base;
    if (slice.Rank < 1) {
        throw std::runtime_error("DslWeightLoader: cannot slice rank-0 tensor");
    }
    long stride = 1;
    for (int i = 1; i < slice.Rank; ++i) {
        stride *= slice.Sizes[i];
    }
    const std::size_t elem_size = get_dtype_size(slice.DType);
    const std::size_t byte_offset = static_cast<std::size_t>(offset)
                                  * static_cast<std::size_t>(stride) * elem_size;
    slice.Data = static_cast<std::byte*>(slice.Data) + byte_offset;
    slice.Sizes[0] = length;
    return slice;
}

std::vector<long> DslWeightLoader::infer_fuse_slices(const std::string& name, int num_sources) const {
    // Case-insensitive name matching for common fuse patterns.
    auto lower = [](const std::string& s) {
        std::string r = s;
        for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return r;
    };
    const std::string lname = lower(name);

    if (lname.find("qkv") != std::string::npos) {
        const long hs = mConfig.head_size();
        const long q_rows = static_cast<long>(mConfig.NumQueryHeads) * hs;
        const long kv_rows = static_cast<long>(mConfig.NumKeyValHeads) * hs;
        return {q_rows, kv_rows, kv_rows};
    }

    if (lname.find("mlp_up") != std::string::npos || lname.find("gate_up") != std::string::npos) {
        const long m = mConfig.IntermediateSize;
        return std::vector<long>(num_sources, m);
    }

    return {};
}

std::string DslWeightLoader::format_hf_name(std::string templ, int layer_idx, int expert_idx) {
    // Replace {layer} placeholder.
    {
        const std::string placeholder = "{layer}";
        std::size_t pos = templ.find(placeholder);
        while (pos != std::string::npos) {
            if (layer_idx < 0) {
                throw std::runtime_error("DslWeightLoader: HF mapping uses {layer} but no layer index available");
            }
            templ.replace(pos, placeholder.size(), std::to_string(layer_idx));
            pos = templ.find(placeholder, pos);
        }
    }

    // Replace {expert} placeholder.
    {
        const std::string placeholder = "{expert}";
        std::size_t pos = templ.find(placeholder);
        while (pos != std::string::npos) {
            if (expert_idx < 0) {
                throw std::runtime_error("DslWeightLoader: HF mapping uses {expert} but no expert index available");
            }
            templ.replace(pos, placeholder.size(), std::to_string(expert_idx));
            pos = templ.find(placeholder, pos);
        }
    }

    return templ;
}

bool DslWeightLoader::parse_block_param(std::string_view name, int& layer_idx, std::string& param_name) {
    auto dot = name.find('.');
    if (dot == std::string_view::npos) return false;
    auto prefix = name.substr(0, dot);
    auto rest = name.substr(dot + 1);

    // Match patterns: blocks[N], mamba_blocks[N], attn_blocks[N], mlp_blocks[N], moe_blocks[N].
    auto bracket = prefix.find("blocks[");
    if (bracket != std::string_view::npos) {
        auto close = prefix.find(']', bracket);
        if (close == std::string_view::npos) return false;
        auto idx_start = bracket + 7;  // length of "blocks["
        auto idx_str = prefix.substr(idx_start, close - idx_start);
        try {
            layer_idx = std::stoi(std::string(idx_str));
        } catch (...) {
            return false;
        }
        param_name = std::string(rest);
        return true;
    }

    // Match legacy format: blocks.<idx>.field.
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

const MappingSpec* DslWeightLoader::find_mapping_spec(const std::string& internal_name,
                                                      int& layer_idx) const {
    layer_idx = -1;

    // Try exact match first.
    auto it = mMapping.find(internal_name);
    if (it != mMapping.end()) {
        return &it->second;
    }

    // Try block parameter pattern: "blocks[N].param" -> "blocks[{layer}].param".
    std::string base;
    if (parse_block_param(internal_name, layer_idx, base)) {
        std::string placeholder = std::string("blocks[{layer}].") + base;
        it = mMapping.find(placeholder);
        if (it != mMapping.end()) {
            return &it->second;
        }
        // Try just the base parameter name.
        it = mMapping.find(base);
        if (it != mMapping.end()) {
            return &it->second;
        }
    }

    return nullptr;
}

int DslWeightLoader::resolve_num_experts(const MappingSpec& spec) const {
    if (spec.num_experts > 0) {
        return spec.num_experts;
    }
    if (mMoEConfig.num_experts > 0) {
        return mMoEConfig.num_experts;
    }
    return 0;
}

}  // namespace dsl
