// Expert slot cache: resolve (directory, LRU, active-round protection, miss list) and gather
// (four W8 planes from a pinned host bank into the device pool) on a small synthetic geometry.
#include "api/ops/expert_slot_cache.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

// hidden 64, 8 experts, 2 per token, intermediate 32: gate/up rows 64 × k 64, down rows 64 × k 32.
constexpr ops::SparseMoeGeometry kGeometry{64, 8, 2, 32};
constexpr std::int32_t kLayers = 2;
constexpr std::int32_t kSlots  = 4;

struct HostBank {
    void* pinned = nullptr;
    std::size_t gate_codes = 0, gate_scales = 0, down_codes = 0, down_scales = 0; // per expert
    std::byte* base() const { return static_cast<std::byte*>(pinned); }
    std::byte* gate_codes_plane() const { return base(); }
    std::byte* gate_scales_plane() const { return base() + gate_codes * kGeometry.experts; }
    std::byte* down_codes_plane() const {
        return gate_scales_plane() + gate_scales * kGeometry.experts;
    }
    std::byte* down_scales_plane() const {
        return down_codes_plane() + down_codes * kGeometry.experts;
    }
    std::size_t total() const {
        return (gate_codes + gate_scales + down_codes + down_scales) * kGeometry.experts;
    }
};

// Every byte of expert e's planes is a function of (plane, e, offset) so a wrong row or plane
// shows up as a mismatch.
std::uint8_t pattern(int plane, int expert, std::size_t offset) {
    return static_cast<std::uint8_t>(37 * plane + 11 * expert + (offset % 251));
}

HostBank make_bank() {
    HostBank bank;
    bank.gate_codes  = static_cast<std::size_t>(kGeometry.expert_rows()) * kGeometry.hidden;
    bank.gate_scales = static_cast<std::size_t>(kGeometry.expert_rows()) * (kGeometry.hidden / 32) * 2;
    bank.down_codes  = static_cast<std::size_t>(kGeometry.hidden) * kGeometry.intermediate;
    bank.down_scales = static_cast<std::size_t>(kGeometry.hidden) * (kGeometry.intermediate / 32) * 2;
    cuda_check(cudaHostAlloc(&bank.pinned, bank.total(), cudaHostAllocMapped | cudaHostAllocPortable),
               "cudaHostAlloc");
    const std::byte* planes[4]  = {bank.gate_codes_plane(), bank.gate_scales_plane(),
                                   bank.down_codes_plane(), bank.down_scales_plane()};
    const std::size_t sizes[4] = {bank.gate_codes, bank.gate_scales, bank.down_codes,
                                  bank.down_scales};
    for (int plane = 0; plane < 4; ++plane) {
        auto* p = const_cast<std::byte*>(planes[plane]);
        for (int expert = 0; expert < kGeometry.experts; ++expert) {
            for (std::size_t i = 0; i < sizes[plane]; ++i) {
                p[expert * sizes[plane] + i] = static_cast<std::byte>(pattern(plane, expert, i));
            }
        }
    }
    return bank;
}

ops::Weight host_weight(std::byte* codes, std::byte* scales, std::int32_t rows, std::int32_t k) {
    ops::Weight w{};
    w.payload       = codes;
    w.qtype         = QType::W8G32_F16S;
    w.layout        = QuantLayout::RowSplit;
    w.group_size    = 32;
    w.qdata         = codes;
    w.scales        = scales;
    w.n             = rows;
    w.k             = k;
    w.group         = 32;
    w.scale_dtype   = DType::FP16;
    w.ndim          = 2;
    w.shape[0]      = rows;
    w.shape[1]      = k;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = k;
    return w;
}

struct Fixture {
    HostBank bank;
    ops::ExpertHostBank host;
    GuardedDeviceBuffer pool_memory;
    GuardedDeviceBuffer directory_memory;
    GuardedDeviceBuffer miss_memory;
    GuardedDeviceBuffer ids_memory;
    ops::ExpertSlotPool pool;
    ops::ExpertSlotDirectory directory;
    ops::ExpertMissList misses;

    Fixture()
        : bank(make_bank()),
          pool_memory(ops::expert_slot_pool_bytes(kGeometry, kSlots)),
          directory_memory(ops::expert_slot_directory_bytes(kLayers, kGeometry.experts, kSlots)),
          miss_memory(ops::expert_miss_list_bytes(kGeometry.experts)),
          ids_memory(64 * sizeof(int)) {
        void* mapped = nullptr;
        cuda_check(cudaHostGetDevicePointer(&mapped, bank.pinned, 0), "cudaHostGetDevicePointer");
        const std::ptrdiff_t shift = static_cast<std::byte*>(mapped) - bank.base();
        const ops::Weight gate_up = host_weight(bank.gate_codes_plane() + shift,
                                                bank.gate_scales_plane() + shift,
                                                kGeometry.routed_gate_rows(), kGeometry.hidden);
        const ops::Weight down    = host_weight(bank.down_codes_plane() + shift,
                                                bank.down_scales_plane() + shift,
                                                kGeometry.routed_down_rows(), kGeometry.intermediate);
        host      = ops::expert_host_bank(kGeometry, gate_up, down);
        pool      = ops::create_expert_slot_pool(kGeometry, kSlots, pool_memory.data());
        directory = ops::create_expert_slot_directory(kLayers, kGeometry.experts, kSlots,
                                                      directory_memory.data(), nullptr);
        misses    = ops::create_expert_miss_list(kGeometry.experts, miss_memory.data());
        cuda_synchronize();
    }
    ~Fixture() { cudaFreeHost(bank.pinned); }

    // One round: resolve + gather; returns the miss count and the layer's slot table.
    struct Round {
        long long misses = 0;
        std::vector<int> slot_of_expert;
        std::vector<int> miss_slots, miss_experts;
    };
    Round round(std::int32_t layer, const std::vector<int>& ids) {
        cuda_check(cudaMemcpy(ids_memory.data(), ids.data(), ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "ids upload");
        Tensor ids_tensor(ids_memory.data(), DType::I32, {static_cast<std::int32_t>(ids.size())});
        ops::expert_slot_resolve(ids_tensor, layer, directory, misses, nullptr);
        ops::expert_slot_gather(host, misses, pool, nullptr);
        cuda_synchronize();
        Round out;
        out.misses = from_device<long long>(misses.count.data, 1)[0];
        const std::vector<int> table =
            from_device<int>(directory.slot_of_expert.data, static_cast<std::size_t>(kLayers) * kGeometry.experts);
        out.slot_of_expert.assign(table.begin() + layer * kGeometry.experts,
                                  table.begin() + (layer + 1) * kGeometry.experts);
        out.miss_slots   = from_device<int>(misses.slots.data, static_cast<std::size_t>(out.misses));
        out.miss_experts = from_device<int>(misses.experts.data, static_cast<std::size_t>(out.misses));
        return out;
    }

    // Verifies the pool slot holds expert `expert` in all four planes.
    int verify_slot(const std::string& label, int slot, int expert) {
        int failures = 0;
        const std::byte* planes[4]  = {pool.gate_up_codes, pool.gate_up_scales, pool.down_codes,
                                       pool.down_scales};
        const std::size_t sizes[4] = {bank.gate_codes, bank.gate_scales, bank.down_codes,
                                      bank.down_scales};
        for (int plane = 0; plane < 4; ++plane) {
            const std::vector<std::uint8_t> got =
                from_device<std::uint8_t>(planes[plane] + slot * sizes[plane], sizes[plane]);
            std::vector<std::uint8_t> want(sizes[plane]);
            for (std::size_t i = 0; i < sizes[plane]; ++i) { want[i] = pattern(plane, expert, i); }
            failures += verify_exact((label + " plane " + std::to_string(plane)).c_str(), got, want);
        }
        return failures;
    }
};

int expect(bool ok, const std::string& what) {
    if (!ok) { std::cout << "FAIL " << what << "\n"; }
    return ok ? 0 : 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    Fixture f;

    // Round 1, layer 0: three distinct experts (one duplicated) → three misses, each gathered.
    auto r1 = f.round(0, {0, 1, 2, 1});
    failures += expect(r1.misses == 3, "round 1 misses == 3 (got " + std::to_string(r1.misses) + ")");
    for (int e = 0; e < 3; ++e) {
        failures += expect(r1.slot_of_expert[e] >= 0 && r1.slot_of_expert[e] < kSlots,
                           "round 1 expert " + std::to_string(e) + " mapped");
        if (r1.slot_of_expert[e] >= 0) {
            failures += f.verify_slot("round 1 expert " + std::to_string(e), r1.slot_of_expert[e], e);
        }
    }
    failures += expect(r1.slot_of_expert[3] == -1, "round 1 expert 3 unmapped");

    // Round 2, layer 0: two hits and one new expert → one miss; the hits keep their slots.
    auto r2 = f.round(0, {1, 2, 3, 3});
    failures += expect(r2.misses == 1, "round 2 misses == 1 (got " + std::to_string(r2.misses) + ")");
    failures += expect(r2.slot_of_expert[1] == r1.slot_of_expert[1] &&
                           r2.slot_of_expert[2] == r1.slot_of_expert[2],
                       "round 2 hits keep their slots");
    failures += expect(r2.slot_of_expert[3] >= 0, "round 2 expert 3 mapped");
    if (r2.slot_of_expert[3] >= 0) { failures += f.verify_slot("round 2 expert 3", r2.slot_of_expert[3], 3); }

    // Round 3, layer 1: four new experts fill the pool, evicting layer 0's entries; the
    // active-round guard keeps this round's own assignments from evicting each other.
    auto r3 = f.round(1, {4, 5, 6, 7});
    failures += expect(r3.misses == 4, "round 3 misses == 4 (got " + std::to_string(r3.misses) + ")");
    {
        std::vector<int> used(kSlots, 0);
        for (int e = 4; e < 8; ++e) {
            const int s = r3.slot_of_expert[e];
            failures += expect(s >= 0 && s < kSlots, "round 3 expert " + std::to_string(e) + " mapped");
            if (s >= 0 && s < kSlots) { used[s] += 1; failures += f.verify_slot("round 3 expert " + std::to_string(e), s, e); }
        }
        for (int s = 0; s < kSlots; ++s) { failures += expect(used[s] == 1, "round 3 slot " + std::to_string(s) + " used once"); }
        const std::vector<int> table = from_device<int>(f.directory.slot_of_expert.data,
                                                        static_cast<std::size_t>(kLayers) * kGeometry.experts);
        for (int e = 0; e < kGeometry.experts; ++e) {
            failures += expect(table[e] == -1, "round 3 evicted layer-0 expert " + std::to_string(e));
        }
    }

    // Round 4, layer 0 again: everything is a miss again and the gathered bytes are right.
    auto r4 = f.round(0, {0, 7});
    failures += expect(r4.misses == 2, "round 4 misses == 2 (got " + std::to_string(r4.misses) + ")");
    for (int e : {0, 7}) {
        if (r4.slot_of_expert[e] >= 0) { failures += f.verify_slot("round 4 expert " + std::to_string(e), r4.slot_of_expert[e], e); }
        else { failures += expect(false, "round 4 expert mapped"); }
    }
    failures += f.pool_memory.verify_guards("pool");
    failures += f.directory_memory.verify_guards("directory");
    failures += f.miss_memory.verify_guards("miss list");
    std::cout << (failures ? "FAIL" : "OK") << " expert_slot_cache\n";
    return failures ? 1 : 0;
}
