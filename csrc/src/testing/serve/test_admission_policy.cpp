#include "runtime/engine/admission_policy.h"

#include <array>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::runtime::ActiveAdmissionSnapshot;
    using ninfer::runtime::AdmissionResources;
    using ninfer::runtime::BackfillClass;

    int failures = 0;
    const AdmissionResources capacity{
        .active_lanes     = 4,
        .main_kv_pages    = 160,
        .backend_kv_pages = 128,
    };
    const AdmissionResources head{
        .active_lanes     = 1,
        .main_kv_pages    = 64,
        .backend_kv_pages = 48,
    };
    std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{
            .request_id            = 1,
            .resources             = {1, 64, 32},
            .remaining_work_quanta = 100,
        },
        ActiveAdmissionSnapshot{
            .request_id            = 2,
            .resources             = {1, 48, 64},
            .remaining_work_quanta = 20,
        },
    };

    const auto protection = ninfer::runtime::make_admission_protection(
        7, 10, head, std::span<const ActiveAdmissionSnapshot>(incumbents), capacity);
    failures += check(protection.donor_count == 1 && protection.donor_ids[0] == 2 &&
                          protection.temporal_credit == 20,
                      "release frontier did not select the earliest sufficient incumbent");
    failures += check(ninfer::runtime::protection_frontier_distance(protection, incumbents) == 20,
                      "frontier distance did not follow the frozen donor");

    const AdmissionResources persistent_candidate{1, 24, 40};
    failures += check(ninfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                   persistent_candidate, capacity),
                      "future resource surplus rejected a persistent-safe backfill");
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, incumbents, AdmissionResources{1, 40, 60}, capacity),
                      "persistent backfill borrowed protected future capacity");

    std::array<ActiveAdmissionSnapshot, 3> with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 3,
            .resources             = persistent_candidate,
            .remaining_work_quanta = 50,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Persistent,
        },
    };
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, with_persistent, AdmissionResources{1, 9, 9}, capacity),
                      "persistent ledger failed to accumulate earlier backfills");

    std::array<ActiveAdmissionSnapshot, 2> after_donor{
        incumbents[0],
        ActiveAdmissionSnapshot{
            .request_id            = 4,
            .resources             = {1, 32, 64},
            .remaining_work_quanta = 8,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Temporal,
        },
    };
    failures += check(ninfer::runtime::protection_frontier_distance(protection, after_donor) == 0,
                      "later temporal work changed the frozen frontier");
    failures += check(
        ninfer::runtime::protected_head_safe_without_temporal(protection, after_donor, capacity),
        "released frontier did not mature behind a temporal borrower");

    failures += check(
        !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 161, 1}, capacity) &&
            !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 1, 129}, capacity),
        "independent KV pools were incorrectly treated as interchangeable capacity");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
