#pragma once

// The round lifecycle every serving target implements.
//
// ConcurrentExecutor is templated on the target's Instance, so the interface
// between engine and target has always been duck-typed: a target that misses a
// method discovers it as a template error somewhere inside the executor. That
// is workable for one architecture family and hostile for several. The concept
// below states the contract in one place, so a new architecture — Qwen MoE,
// Gemma, GLM, Kimi — gets a direct diagnostic naming the method it owes, and
// so the scheduling policy has a documented surface to sit on.
//
// The lifecycle is deliberately split into launch and consume. A synchronous
// decode_batch (enqueue, synchronize, read egress, book-keep) forces the host
// serial path — sync, egress read, book-keeping, staging, launch — to sit
// between every pair of rounds with the device idle underneath it; that idle
// measured 11-12% of device time under 100-user load. Splitting the round lets
// the executor launch round N+1 on chained device state before it consumes
// round N's egress, which is the overlap vLLM gets from its async scheduler.
// The pipelining then lives in the executor and is written once, rather than
// being re-implemented per target: a target supplies a forward pass, not a
// scheduler.

#include "runtime/contract/types.h"

#include <concepts>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

// Opaque ticket for an in-flight round. Targets may encode whatever they need
// (frame index, graph slot, event id) as long as it round-trips by value.
struct RoundHandle {
    std::uint64_t id       = 0;
    std::uint32_t rows     = 0;
    bool valid() const noexcept { return id != 0; }
};

// A target's decode surface. `launch_decode_round` must enqueue the round's
// work and its egress copy on the target's stream WITHOUT synchronizing, and
// must leave every device-side input for the following round derivable on
// device (chained tokens and positions), so the executor can launch again
// before consuming. `consume_decode_round` synchronizes that round, reads its
// egress, and performs the per-lane book-keeping.
//
// A target that cannot overlap may implement launch as enqueue-and-sync and
// consume as a pure read; the executor stays correct and simply gains nothing.
template <class Program>
concept RoundLifecycle = requires(Program& program, RoundHandle handle,
                                  std::span<const std::uint32_t> lanes,
                                  std::span<const RoundBudget> budgets) {
    { program.launch_decode_round(lanes, budgets) } -> std::same_as<RoundHandle>;
    { program.consume_decode_round(handle) } -> std::same_as<BatchedGeneratedRound>;
    // Depth the target can keep in flight; 1 means no overlap is available.
    { program.maximum_rounds_in_flight() } -> std::convertible_to<std::uint32_t>;
};

} // namespace ninfer::runtime
