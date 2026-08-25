#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

enum class BackfillClass : std::uint8_t {
    None,
    Persistent,
    Temporal,
};

struct ActiveAdmissionSnapshot {
    std::uint64_t request_id = 0;
    AdmissionResources resources;
    std::uint64_t remaining_work_quanta = 0;
    std::uint64_t backfill_epoch        = 0;
    BackfillClass backfill_class        = BackfillClass::None;
};

enum class ProtectionPhase : std::uint8_t {
    Open,
    Drain,
};

struct AdmissionProtection {
    std::uint64_t epoch_id        = 0;
    std::uint64_t head_request_id = 0;
    AdmissionResources head_resources;
    std::array<std::uint64_t, kMaximumConcurrency> incumbent_ids{};
    std::array<std::uint64_t, kMaximumConcurrency> donor_ids{};
    std::size_t incumbent_count   = 0;
    std::size_t donor_count       = 0;
    std::uint64_t temporal_credit = 0;
    ProtectionPhase phase         = ProtectionPhase::Open;
};

[[nodiscard]] bool admission_resources_fit(const AdmissionResources& used,
                                           const AdmissionResources& capacity) noexcept;

// Freezes the currently active requests and selects the earliest projected completion prefix
// whose release makes the protected head componentwise feasible.
[[nodiscard]] AdmissionProtection make_admission_protection(
    std::uint64_t epoch_id, std::uint64_t head_request_id, const AdmissionResources& head_resources,
    std::span<const ActiveAdmissionSnapshot> active, const AdmissionResources& capacity);

// Tests the cumulative future-frontier invariant, including every still-active persistent
// backfill from this epoch and the proposed candidate.
[[nodiscard]] bool persistent_backfill_is_safe(const AdmissionProtection& protection,
                                               std::span<const ActiveAdmissionSnapshot> active,
                                               const AdmissionResources& candidate,
                                               const AdmissionResources& capacity) noexcept;

// Projected distance to the last still-active frozen donor. Later admissions never contribute.
[[nodiscard]] std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept;

// True once the head would fit if current-epoch temporal borrowers were absent. This recognizes
// both the frozen donor frontier and an earlier opportunity created by any incumbent release.
[[nodiscard]] bool
protected_head_safe_without_temporal(const AdmissionProtection& protection,
                                     std::span<const ActiveAdmissionSnapshot> active,
                                     const AdmissionResources& capacity) noexcept;

} // namespace ninfer::runtime
