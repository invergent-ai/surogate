#pragma once

// Maximum lanes in one GPU decode round. Active requests are stored separately
// and scheduled in bounded batches, so this does not limit engine concurrency.

#include <cstdint>

namespace sinfer {

inline constexpr std::int32_t kMaximumBatchColumns = 128;

} // namespace sinfer
