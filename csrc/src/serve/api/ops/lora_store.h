#pragma once

// The resident adapter: PEFT tensors uploaded once, keyed by the base weight
// they adapt.
//
// The variant methods that run a projection do not take a layer index -- they
// take the layer's weights -- so the store is keyed by the base weight's device
// pointer. Each layer's projection has its own, which makes the pointer a
// sufficient identity and leaves every projection signature unchanged.
//
// One adapter is active for the whole server. Per-request adapters need the
// scheduler to group lanes by adapter, which is a different piece of work; a
// deployment that wants two adapters runs two servers, and one that names two
// here is refused at startup rather than served the wrong one.

#include "api/ops/lora.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sinfer::ops {

class LoraStore {
public:
    /// Uploads one A/B pair for the projection whose base weight is `base_key`.
    ///
    /// `a` is [rank, in] and `b` is [out, rank] in row-major host memory, already
    /// converted to BF16. `scale` (PEFT's alpha/r) is folded into A here, once,
    /// rather than multiplied over the delta on every call.
    void add(const void* base_key, const std::vector<std::uint16_t>& a,
             const std::vector<std::uint16_t>& b, std::int32_t rank, std::int32_t in_dim,
             std::int32_t out_dim, float scale);

    /// The adapter for a projection, or nullptr when it has none. Hot path.
    [[nodiscard]] const LoraWeights* find(const void* base_key) const noexcept {
        const auto found = entries_.find(base_key);
        return found == entries_.end() ? nullptr : &found->second;
    }

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    /// Largest rank and output width held, for sizing the delta scratch once.
    [[nodiscard]] std::int32_t max_rank() const noexcept { return max_rank_; }
    [[nodiscard]] std::int32_t max_out_dim() const noexcept { return max_out_dim_; }

    ~LoraStore();
    LoraStore()                            = default;
    LoraStore(const LoraStore&)            = delete;
    LoraStore& operator=(const LoraStore&) = delete;

private:
    std::unordered_map<const void*, LoraWeights> entries_;
    std::vector<void*> owned_;
    std::int32_t max_rank_    = 0;
    std::int32_t max_out_dim_ = 0;
};

/// The active store for the current device; empty unless an adapter was loaded.
[[nodiscard]] LoraStore& lora_store_for_current_device();

/// True when any adapter is resident on the current device, so the projection
/// hooks can skip the lookup entirely in the common case.
[[nodiscard]] bool lora_active();

} // namespace sinfer::ops
