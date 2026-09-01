#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <api/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sinfer::targets::qwen3_6::detail {

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt, std::int32_t lora_slot);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count,
                               std::int32_t requester_lora_slot) const;

private:
    // The adapter the resident values were computed under (-1 for the base
    // model). KV computed with adapter deltas in q/k/v is that adapter's KV, so
    // token identity alone is not license to reuse it -- vLLM keys its prefix
    // blocks by (tokens, lora id) for the same reason. Without this, a request
    // for adapter b could resume on a's cache and answer from a blend of the two.
    std::int32_t lora_slot_ = -1;
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count, std::int32_t requester_lora_slot);

} // namespace sinfer::targets::qwen3_6::detail
