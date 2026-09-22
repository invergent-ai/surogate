#include "runtime/engine/admission_policy.h"

#include <array>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using sinfer::runtime::ActiveAdmissionSnapshot;
    using sinfer::runtime::AdmissionResources;
    using sinfer::runtime::BackfillClass;

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

    // `std::optional<AdmissionProtection>` also binds a bare AdmissionProtection, so this
    // file compiles unchanged against the signature that threw for an unblocked head; the
    // regression block at the end is what separates the two.
    const std::optional<sinfer::runtime::AdmissionProtection> protection_holder =
        sinfer::runtime::make_admission_protection(
            7, 10, head, std::span<const ActiveAdmissionSnapshot>(incumbents), capacity);
    if (!protection_holder) {
        std::cerr << "a blocked protected head produced no protection frontier\n";
        return 1;
    }
    const sinfer::runtime::AdmissionProtection& protection = *protection_holder;
    failures += check(protection.donor_count == 1 && protection.donor_ids[0] == 2 &&
                          protection.temporal_credit == 20,
                      "release frontier did not select the earliest sufficient incumbent");
    failures += check(sinfer::runtime::protection_frontier_distance(protection, incumbents) == 20,
                      "frontier distance did not follow the frozen donor");

    const AdmissionResources persistent_candidate{1, 24, 40};
    failures += check(sinfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                   persistent_candidate, capacity),
                      "future resource surplus rejected a persistent-safe backfill");
    failures += check(!sinfer::runtime::persistent_backfill_is_safe(
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
    failures += check(!sinfer::runtime::persistent_backfill_is_safe(
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
    failures += check(sinfer::runtime::protection_frontier_distance(protection, after_donor) == 0,
                      "later temporal work changed the frozen frontier");
    failures += check(
        sinfer::runtime::protected_head_safe_without_temporal(protection, after_donor, capacity),
        "released frontier did not mature behind a temporal borrower");

    failures += check(
        !sinfer::runtime::admission_resources_fit(AdmissionResources{1, 161, 1}, capacity) &&
            !sinfer::runtime::admission_resources_fit(AdmissionResources{1, 1, 129}, capacity),
        "independent KV pools were incorrectly treated as interchangeable capacity");

    // A protected request can wait behind more incumbents than fit in one GPU round.
    std::vector<ActiveAdmissionSnapshot> many(513);
    for (std::size_t i = 0; i < many.size(); ++i) {
        many[i] = {.request_id = i + 1, .resources = {1, 1, 1},
                   .remaining_work_quanta = many.size() - i};
    }
    const std::optional<sinfer::runtime::AdmissionProtection> large_holder =
        sinfer::runtime::make_admission_protection(9, 1000, AdmissionResources{1, 1, 1}, many,
                                                   AdmissionResources{513, 513, 513});
    if (!large_holder) {
        std::cerr << "a blocked protected head produced no protection frontier\n";
        return 1;
    }
    const sinfer::runtime::AdmissionProtection& large = *large_holder;
    failures += check(large.incumbent_count == 513 && large.incumbent_ids.back() == 513 &&
                          large.donor_count == 1 && large.donor_ids[0] == 513,
                      "protected admission lost an incumbent above lane 127");
    many.pop_back();
    failures += check(sinfer::runtime::protected_head_safe_without_temporal(
                          large, many, AdmissionResources{513, 513, 513}),
                      "releasing a high-numbered incumbent did not admit the protected head");


    // Regression, engine-admission-fix-v1. The decisions endpoint pins a shared state prefix
    // with save_gpu_prefix; the KV pages that prefix keeps stay committed to the pool while
    // belonging to no lane and to no active request (family/impl/runtime/prefix_cache_impl.h,
    // capture_gpu_prefix moves the lane's bundle into the image and clears `retained`). This
    // ledger therefore sums less than the pool holds, so a head the pool has no room for looks
    // unblocked here. That is an observation about resources, not a broken precondition: it
    // must come back as "no protection", because the engine worker loop is the caller and an
    // exception out of here kills it for the lifetime of the process.
    //
    // The numbers are the incident's: --kv-capacity 32768 over 64-token pages is 512 pages and
    // 8 lanes; two concurrently pinned decision prefixes hold 4 of those pages, a 31,680-token
    // incumbent holds 495, and a 1,024-token head needs 14 more pages than the 13 the pool has
    // left -- while this ledger sees 495 + 16 = 511 of 512 and calls the head unblocked.
    {
        const AdmissionResources pool{
            .active_lanes = 8, .main_kv_pages = 512, .backend_kv_pages = 0};
        const AdmissionResources head_1k{
            .active_lanes = 1, .main_kv_pages = 16, .backend_kv_pages = 0};
        std::array<ActiveAdmissionSnapshot, 1> unblocking{ActiveAdmissionSnapshot{
            .request_id = 1, .resources = {1, 495, 0}, .remaining_work_quanta = 12}};

        bool threw = false;
        std::optional<sinfer::runtime::AdmissionProtection> nothing_to_protect;
        try {
            nothing_to_protect = sinfer::runtime::make_admission_protection(
                3, 42, head_1k, std::span<const ActiveAdmissionSnapshot>(unblocking), pool);
        } catch (const std::exception&) { threw = true; }
        failures += check(!threw, "an unblocked protected head threw into the engine worker loop");
        failures += check(threw || !nothing_to_protect.has_value(),
                          "an unblocked protected head produced a protection frontier");

        // Two pages further along, the same head is genuinely blocked and is still protected.
        std::array<ActiveAdmissionSnapshot, 1> blocking{ActiveAdmissionSnapshot{
            .request_id = 1, .resources = {1, 497, 0}, .remaining_work_quanta = 12}};
        const std::optional<sinfer::runtime::AdmissionProtection> blocked =
            sinfer::runtime::make_admission_protection(
                4, 42, head_1k, std::span<const ActiveAdmissionSnapshot>(blocking), pool);
        failures += check(blocked.has_value() && blocked->donor_count == 1 &&
                              blocked->donor_ids[0] == 1 && blocked->temporal_credit == 12,
                          "a genuinely blocked head lost its donor frontier");

        // The broken preconditions stay programming errors: they are the caller's own checks,
        // and the admission path has already refused the request before it can reach them.
        const auto rejects = [&](std::uint64_t epoch, const AdmissionResources& head_resources,
                                 std::span<const ActiveAdmissionSnapshot> active) {
            try {
                (void)sinfer::runtime::make_admission_protection(epoch, 42, head_resources,
                                                                 active, pool);
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        std::array<ActiveAdmissionSnapshot, 1> stalled{ActiveAdmissionSnapshot{
            .request_id = 1, .resources = {1, 497, 0}, .remaining_work_quanta = 0}};
        failures += check(rejects(0, head_1k, blocking), "protection epoch 0 was accepted");
        failures += check(rejects(4, AdmissionResources{1, 513, 0}, blocking),
                          "a head larger than capacity was accepted for protection");
        failures += check(rejects(4, AdmissionResources{0, 16, 0}, blocking),
                          "a head reserving no lane was accepted for protection");
        failures += check(rejects(4, head_1k, std::span<const ActiveAdmissionSnapshot>()),
                          "an empty incumbent set was accepted for protection");
        failures += check(rejects(4, head_1k, stalled),
                          "an incumbent with no progress state was accepted");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
