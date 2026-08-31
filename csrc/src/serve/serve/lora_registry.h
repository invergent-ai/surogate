#pragma once

// Registered LoRA adapters, loaded from PEFT checkpoints (`--enable-lora`).
//
// An adapter is a directory holding `adapter_config.json` and
// `adapter_model.safetensors`, which is what `surogate sft` and PEFT both write.
// The registry reads them at startup rather than on first use: an adapter whose
// rank exceeds the deployment's bound, or whose tensors do not match the served
// model's geometry, is a configuration error, and a server that starts and then
// fails every request naming that adapter is worse than one that refuses to
// start.
//
// Nothing here touches the device. Loading the tensors onto the GPU belongs to
// the target that will apply them, because only it knows which projections it
// fuses and therefore which of the adapter's modules it can honour.

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace sinfer::serve {

/// One `lora_A`/`lora_B` pair, named by the base module it adapts.
struct LoraTensorPair {
    std::string module;       ///< e.g. "model.layers.3.self_attn.q_proj"
    std::int32_t rank   = 0;  ///< rows of A / columns of B
    std::int32_t in_dim = 0;  ///< columns of A: the projection's k
    std::int32_t out_dim = 0; ///< rows of B: the projection's n
    std::uint64_t a_offset = 0; ///< byte offsets into the safetensors payload
    std::uint64_t a_bytes  = 0;
    std::uint64_t b_offset = 0;
    std::uint64_t b_bytes  = 0;
    bool a_is_bf16 = true;
    bool b_is_bf16 = true;
};

/// One adapter as it sits on disk, validated but not yet resident.
struct LoraAdapter {
    std::string name;
    std::filesystem::path directory;
    std::filesystem::path weights_file;
    std::int32_t rank      = 0;
    double alpha           = 0.0;
    /// PEFT's `lora_alpha / r`, the factor the delta is multiplied by. Folded into
    /// A when the tensors are made resident, so it is carried here only to be
    /// reported and applied once.
    double scale           = 1.0;
    std::vector<std::string> target_modules;
    std::vector<LoraTensorPair> pairs;
};

/// Reads and validates the adapters named on the command line.
///
/// Throws std::invalid_argument naming the adapter and the reason when a
/// directory is missing a file, the config is not a LoRA config, the rank
/// exceeds `max_rank`, or the safetensors header disagrees with the config.
class LoraRegistry {
public:
    LoraRegistry() = default;

    void load(const std::vector<std::pair<std::string, std::string>>& modules,
              std::uint32_t max_rank);

    [[nodiscard]] bool empty() const noexcept { return adapters_.empty(); }
    [[nodiscard]] const std::map<std::string, LoraAdapter>& adapters() const noexcept {
        return adapters_;
    }
    /// The adapter a request named, or nullptr when the name is the base model.
    [[nodiscard]] const LoraAdapter* find(const std::string& name) const;

private:
    std::map<std::string, LoraAdapter> adapters_;
};

} // namespace sinfer::serve
