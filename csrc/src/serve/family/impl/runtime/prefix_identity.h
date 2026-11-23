#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <api/family/prepared_prompt.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::family::detail {

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    /// `lora_generation` is the slot's KV generation the values are computed under
    /// (ops::lora_slot_kv_generation; 0 for the base model).
    void assign(const PreparedPromptData& prompt, std::int32_t lora_slot, std::uint64_t lora_generation);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count,
                               std::int32_t requester_lora_slot,
                               std::uint64_t requester_lora_generation) const;
    /// Whether every input this identity licenses is also the first `size()` of `longer`'s:
    /// the same adapter, token types, positions and the media items it covers.
    [[nodiscard]] bool is_prefix_of(const ResidentPrefixIdentity& longer) const;
    [[nodiscard]] std::int32_t lora_slot() const noexcept { return lora_slot_; }
    [[nodiscard]] std::uint64_t lora_generation() const noexcept { return lora_generation_; }
    /// Text tokens only, at their plain positions' token types: no media items.
    [[nodiscard]] bool text_only() const noexcept {
        return vision_items_.empty() &&
               std::all_of(token_types_.begin(), token_types_.end(), [](std::uint8_t type) { return type == 0; });
    }

private:
    // The adapter the resident values were computed under (-1 for the base
    // model). KV computed with adapter deltas in q/k/v is that adapter's KV, so
    // token identity alone is not license to reuse it -- vLLM keys its prefix
    // blocks by (tokens, lora id) for the same reason. Without this, a request
    // for adapter b could resume on a's cache and answer from a blend of the two.
    std::int32_t lora_slot_ = -1;
    // Which bytes that adapter slot held (LoraStore::slot_kv_generation). An in-place swap
    // keeps the slot and changes its weights: KV computed under the old version is not the
    // new version's either, and serving it answered a request under the new adapter from the
    // old one's prompt state (an in-place swap scored a prompt exactly as before it, 11-10).
    std::uint64_t lora_generation_ = 0;
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count, std::int32_t requester_lora_slot,
                                  std::uint64_t requester_lora_generation);

} // namespace sinfer::family::detail
