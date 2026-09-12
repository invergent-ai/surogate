#pragma once

#include "runtime/contract/types.h"

#include <cstddef>

namespace sinfer::runtime {

/// Memory an expert cache must leave for the requested KV policy, including automatic
/// headroom. Explicit capacities reserve all requested pages, not just one full sequence.
[[nodiscard]] std::size_t minimum_kv_reservation_bytes(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve);

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

} // namespace sinfer::runtime
