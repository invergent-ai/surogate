// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// MappingSpec: Standalone struct describing how to load an internal
// parameter from a HuggingFace SafeTensors checkpoint.
//
// This is extracted from DslModel::MappingSpec so that DslWeightLoader
// can consume it without depending on the full DslModel header.
// DslModel retains a type alias for backward compatibility.

#ifndef SUROGATE_SRC_RUNTIME_DSL_MAPPING_SPEC_H
#define SUROGATE_SRC_RUNTIME_DSL_MAPPING_SPEC_H

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsl {

/// Describes one weight-mapping rule from HF checkpoint to internal parameter.
///
/// Each internal parameter name (e.g. "blocks[{layer}].qkv_weight") maps
/// to exactly one MappingSpec that tells the weight loader how to obtain
/// the tensor data from the HuggingFace SafeTensors file(s).
struct MappingSpec {
    enum class Kind {
        Direct,         ///< 1-to-1: single HF tensor -> internal param
        Fuse,           ///< Concatenate multiple HF tensors along `dim`
        Split,          ///< Extract a sub-range from one HF tensor
        Transform,      ///< Load then apply a function (e.g. transpose)
        TiedTo,         ///< This param is a copy of another internal param
        StackExperts,   ///< Stack per-expert HF tensors into [E, ...] format
        Unknown,
    };

    Kind kind = Kind::Unknown;

    /// For Direct / Transform / Split / StackExperts:
    /// the HF tensor path template, may contain {layer} and {expert} placeholders.
    std::string source;

    /// For Fuse: ordered list of HF source tensor paths to concatenate.
    std::vector<std::string> sources;

    /// For Split: [start, end) ranges in elements along `dim`.
    std::vector<std::pair<long, long>> ranges;

    /// For Transform: the transform function name (e.g. "transpose").
    std::string fn;

    /// For TiedTo: the internal parameter name to copy from after all loads.
    std::string target;

    /// For Fuse / Split: the dimension along which to concatenate/split.
    int dim = 0;

    /// If true, a missing HF tensor is not an error (skip silently).
    bool optional = false;

    /// For StackExperts: fuse gate+up projections into interleaved gate_up format.
    bool fuse_gate_up = false;

    /// For StackExperts: number of experts (0 = auto-detect from config).
    int num_experts = 0;
};

/// A table mapping internal parameter names (or patterns with {layer}) to specs.
using MappingTable = std::unordered_map<std::string, MappingSpec>;

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_MAPPING_SPEC_H
