#pragma once

// Single source of truth for the decode-batch width that ops must admit.
//
// Ops cannot see the serving API's kMaximumConcurrency, so every batched op
// used to carry its own literal mirror of it. Those mirrors drift: raising
// the concurrency ceiling once meant finding eleven of them by running into
// their errors one at a time, and the mirrors that clamp instead of throwing
// (PATCHES.md #35) corrupt the token stream instead of failing. Ops now spell
// the bound as kMaximumBatchColumns and a static assertion at the engine
// binds it to kMaximumConcurrency, so the ceiling moves in one place.

#include <cstdint>

namespace sinfer {

inline constexpr std::int32_t kMaximumBatchColumns = 128;

} // namespace sinfer
