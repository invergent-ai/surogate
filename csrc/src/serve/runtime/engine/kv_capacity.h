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

/// The free device memory a target sizes its KV capacity against, read once before and once
/// after the engine's adapter banks were allocated (the same reading when it carries none).
///
/// An automatic capacity without a draft head is sized after the banks. Sized before them, it
/// spent them a second time: one rank-16 slot on every expert of a 35B-A3B is 2.7 GiB against
/// 1 GiB of headroom, and the engine refused to start where the shortfall surfaced (its
/// conversation checkpoints, 43 lanes, no MTP).
///
/// Everything else is sized before them, as it always was. An explicit capacity is the
/// operator's own. A speculative plan reserves a graph allowance its graphs stay far inside
/// (MTP: 82 MiB a lane per topology class; the 35B-A3B armed for adapters captured 289 MiB at
/// 40,960 tokens and 301 MiB at 262,144 against 2.56 GiB at 32 lanes) and that room has held
/// the banks; counting them on top refused plans that serve (32 lanes, MTP 2: 6.21 GB of minimum
/// reservation and 1 GiB of headroom against 6.57 GB).
[[nodiscard]] std::size_t kv_sizing_free_bytes(const KvCapacityPolicy& policy,
                                               SpeculativeBackend speculative,
                                               std::size_t free_before_adapter_banks,
                                               std::size_t free_after_adapter_banks) noexcept;

} // namespace sinfer::runtime
