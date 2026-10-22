#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const sinfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        sinfer::runtime::resolve_kv_capacity(sinfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        sinfer::runtime::resolve_kv_capacity(sinfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = sinfer::runtime::resolve_kv_capacity(
        sinfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    failures += check(sinfer::runtime::minimum_kv_reservation_bytes(
                          sinfer::KvCapacityPolicy::automatic(50), curve) == 1050,
                      "expert cache floor must leave automatic KV headroom");
    failures += check(sinfer::runtime::minimum_kv_reservation_bytes(
                          sinfer::KvCapacityPolicy::explicit_capacity(129), curve) == 1128,
                      "expert cache floor must reserve every explicit KV page");
    failures += check(sinfer::runtime::minimum_kv_reservation_bytes(
                          sinfer::KvCapacityPolicy::explicit_capacity(384), curve) == 1512,
                      "expert cache floor must leave the whole pipeline KV capacity");

    for (const auto tokens : {1U, 64U, 65U, 128U, 384U, 385U, std::numeric_limits<std::uint32_t>::max()}) {
        const auto policy = sinfer::KvCapacityPolicy::explicit_capacity(tokens);
        const auto result = sinfer::runtime::resolve_kv_capacity(policy, curve, 1512);
        const bool floor = tokens <= 128;
        failures += check(result.main_page_groups == (floor ? 2U : 6U) &&
                          result.resolved_tokens == (floor ? 128U : 384U) &&
                          result.runtime_reservation_bytes == (floor ? 1000U : 1512U) &&
                          sinfer::runtime::minimum_kv_reservation_bytes(policy, curve) == result.runtime_reservation_bytes,
                          "explicit KV capacity or expert staging floor escaped the curve domain");
    }
    const sinfer::runtime::SequenceCapacityCurve lane_floor{
        .main_page_tokens=64, .minimum_main_page_groups=64, .maximum_main_page_groups=128,
        .minimum_device_reservation_bytes=1000, .bytes_per_additional_main_page_group=128};
    const auto lanes = sinfer::runtime::resolve_kv_capacity(
        sinfer::KvCapacityPolicy::explicit_capacity(2048), lane_floor, 1000);
    failures += check(lanes.main_page_groups == 64 && lanes.resolved_tokens == 4096,
                      "explicit context capacity did not reserve one page per concurrent lane");
    for (const auto tokens : {1U, 385U}) {
        bool rejected = false;
        try { (void)sinfer::runtime::resolve_kv_capacity(sinfer::KvCapacityPolicy::explicit_capacity(tokens),
                                                         curve, tokens == 1 ? 999 : 1511); }
        catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "clamped explicit KV capacity ignored available memory");
    }

    bool insufficient_rejected = false;
    try {
        (void)sinfer::runtime::resolve_kv_capacity(sinfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // Recurrent-only pipeline stages keep page ledgers without any KV payload.
    auto metadata_only = curve;
    metadata_only.bytes_per_additional_main_page_group = 0;
    for (const auto policy : {sinfer::KvCapacityPolicy::automatic(50),
                              sinfer::KvCapacityPolicy::explicit_capacity(384)}) {
        const auto result = sinfer::runtime::resolve_kv_capacity(policy, metadata_only, 1050);
        failures += check(result.main_page_groups == 6 && result.runtime_reservation_bytes == 1000,
                          "metadata-only stage did not expand its page ledger at constant cost");
        bool refused = false;
        try { (void)sinfer::runtime::resolve_kv_capacity(policy, metadata_only, 999); }
        catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, "metadata-only stage ignored its fixed memory requirement");
    }

    {
        // Adapter banks: an automatic capacity without a draft head is sized after them, every
        // other policy before them (as it was before they were allocated first).
        using sinfer::SpeculativeBackend;
        const auto automatic_policy = sinfer::KvCapacityPolicy::automatic(50);
        const auto explicit_policy  = sinfer::KvCapacityPolicy::explicit_capacity(129);
        failures += check(sinfer::runtime::kv_sizing_free_bytes(automatic_policy, SpeculativeBackend::None,
                                                                1360, 1000) == 1000,
                          "automatic KV capacity without a draft head must count the adapter banks");
        failures += check(sinfer::runtime::kv_sizing_free_bytes(automatic_policy, SpeculativeBackend::Mtp,
                                                                1360, 1000) == 1360 &&
                              sinfer::runtime::kv_sizing_free_bytes(automatic_policy,
                                                                    SpeculativeBackend::DFlash, 1360,
                                                                    1000) == 1360,
                          "speculative automatic KV capacity must be sized as before the banks");
        failures += check(sinfer::runtime::kv_sizing_free_bytes(explicit_policy, SpeculativeBackend::None,
                                                                1360, 1000) == 1360,
                          "explicit KV capacity must be checked as before the banks");
        failures += check(sinfer::runtime::kv_sizing_free_bytes(automatic_policy, SpeculativeBackend::None,
                                                                1360, 1360) == 1360,
                          "an engine without adapters must size as it always did");
        // The regression in miniature: the curve above, 1600 bytes free after weights, 300 bytes
        // of banks. Sized before them the automatic pool takes all 6 page groups (1512 bytes) and
        // leaves 88, less than the banks; sized after them it takes the 3 that fit beside them.
        const auto before = sinfer::runtime::resolve_kv_capacity(automatic_policy, curve, 1600);
        const auto after  = sinfer::runtime::resolve_kv_capacity(
            automatic_policy, curve,
            sinfer::runtime::kv_sizing_free_bytes(automatic_policy, SpeculativeBackend::None, 1600, 1300));
        failures += check(before.main_page_groups == 6 && before.planned_slack_bytes == 88 &&
                              after.main_page_groups == 3 &&
                              after.runtime_reservation_bytes + 50 <= 1300,
                          "automatic KV capacity sized after the banks must leave them and the headroom");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
