#include "runtime/engine/admission_policy.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

struct ResourceTotals {
    std::uint64_t active_lanes     = 0;
    std::uint64_t main_kv_pages    = 0;
    std::uint64_t backend_kv_pages = 0;
};

void add(ResourceTotals& total, const AdmissionResources& resources) noexcept {
    total.active_lanes += resources.active_lanes;
    total.main_kv_pages += resources.main_kv_pages;
    total.backend_kv_pages += resources.backend_kv_pages;
}

void subtract(ResourceTotals& total, const AdmissionResources& resources) {
    if (resources.active_lanes > total.active_lanes ||
        resources.main_kv_pages > total.main_kv_pages ||
        resources.backend_kv_pages > total.backend_kv_pages) {
        throw std::logic_error("admission resource subtraction underflow");
    }
    total.active_lanes -= resources.active_lanes;
    total.main_kv_pages -= resources.main_kv_pages;
    total.backend_kv_pages -= resources.backend_kv_pages;
}

bool fits(const ResourceTotals& used, const AdmissionResources& capacity) noexcept {
    return used.active_lanes <= capacity.active_lanes &&
           used.main_kv_pages <= capacity.main_kv_pages &&
           used.backend_kv_pages <= capacity.backend_kv_pages;
}

bool contains(std::span<const std::uint64_t> ids, std::uint64_t id) noexcept {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

bool is_incumbent(const AdmissionProtection& protection, std::uint64_t id) noexcept {
    return contains(
        std::span<const std::uint64_t>(protection.incumbent_ids.data(), protection.incumbent_count),
        id);
}

bool is_donor(const AdmissionProtection& protection, std::uint64_t id) noexcept {
    return contains(
        std::span<const std::uint64_t>(protection.donor_ids.data(), protection.donor_count), id);
}

} // namespace

bool admission_resources_fit(const AdmissionResources& used,
                             const AdmissionResources& capacity) noexcept {
    ResourceTotals total;
    add(total, used);
    return fits(total, capacity);
}

AdmissionProtection make_admission_protection(std::uint64_t epoch_id, std::uint64_t head_request_id,
                                              const AdmissionResources& head_resources,
                                              std::span<const ActiveAdmissionSnapshot> active,
                                              const AdmissionResources& capacity) {
    if (epoch_id == 0 || head_request_id == 0 || active.empty() ||
        active.size() > kMaximumConcurrency || head_resources.active_lanes == 0 ||
        !admission_resources_fit(head_resources, capacity)) {
        throw std::invalid_argument("invalid protected-admission frontier");
    }

    AdmissionProtection out;
    out.epoch_id        = epoch_id;
    out.head_request_id = head_request_id;
    out.head_resources  = head_resources;
    out.incumbent_count = active.size();

    ResourceTotals survivors;
    for (std::size_t i = 0; i < active.size(); ++i) {
        if (active[i].request_id == 0 || active[i].remaining_work_quanta == 0) {
            throw std::invalid_argument("protected incumbent has invalid progress state");
        }
        out.incumbent_ids[i] = active[i].request_id;
        add(survivors, active[i].resources);
    }
    add(survivors, head_resources);
    if (fits(survivors, capacity)) {
        throw std::invalid_argument("protected head is not blocked by frozen incumbents");
    }

    std::array<std::size_t, kMaximumConcurrency> order{};
    for (std::size_t i = 0; i < active.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(active.size()),
              [&](std::size_t lhs, std::size_t rhs) {
                  if (active[lhs].remaining_work_quanta != active[rhs].remaining_work_quanta) {
                      return active[lhs].remaining_work_quanta < active[rhs].remaining_work_quanta;
                  }
                  return active[lhs].request_id < active[rhs].request_id;
              });

    for (std::size_t i = 0; i < active.size(); ++i) {
        const ActiveAdmissionSnapshot& donor = active[order[i]];
        subtract(survivors, donor.resources);
        out.donor_ids[out.donor_count++] = donor.request_id;
        out.temporal_credit              = donor.remaining_work_quanta;
        if (fits(survivors, capacity)) { return out; }
    }
    throw std::logic_error("exclusive-feasible head has no releasing incumbent frontier");
}

bool persistent_backfill_is_safe(const AdmissionProtection& protection,
                                 std::span<const ActiveAdmissionSnapshot> active,
                                 const AdmissionResources& candidate,
                                 const AdmissionResources& capacity) noexcept {
    ResourceTotals future;
    add(future, protection.head_resources);
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_incumbent(protection, request.request_id)) {
            if (!is_donor(protection, request.request_id)) { add(future, request.resources); }
        } else if (request.backfill_epoch == protection.epoch_id &&
                   request.backfill_class == BackfillClass::Persistent) {
            add(future, request.resources);
        }
    }
    add(future, candidate);
    return fits(future, capacity);
}

std::uint64_t
protection_frontier_distance(const AdmissionProtection& protection,
                             std::span<const ActiveAdmissionSnapshot> active) noexcept {
    std::uint64_t distance = 0;
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_donor(protection, request.request_id)) {
            distance = std::max(distance, request.remaining_work_quanta);
        }
    }
    return distance;
}

bool protected_head_safe_without_temporal(const AdmissionProtection& protection,
                                          std::span<const ActiveAdmissionSnapshot> active,
                                          const AdmissionResources& capacity) noexcept {
    ResourceTotals used;
    add(used, protection.head_resources);
    for (const ActiveAdmissionSnapshot& request : active) {
        if (is_incumbent(protection, request.request_id) ||
            request.backfill_epoch != protection.epoch_id ||
            request.backfill_class != BackfillClass::Temporal) {
            add(used, request.resources);
        }
    }
    return fits(used, capacity);
}

} // namespace ninfer::runtime
