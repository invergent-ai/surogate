// Expert slot cache: resolve (directory, LRU, active-round protection, miss list) and gather
// (four W8 planes from a pinned host bank into the device pool) on a small synthetic geometry.
#include "api/ops/cpu_expert_compute.h"
#include "api/ops/expert_slot_cache.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <cmath>
#include <random>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

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

Weight host_weight(std::byte* codes, std::byte* scales, std::int32_t rows, std::int32_t k) {
    Weight w{};
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
        const Weight gate_up = host_weight(bank.gate_codes_plane() + shift,
                                                bank.gate_scales_plane() + shift,
                                                kGeometry.routed_gate_rows(), kGeometry.hidden);
        const Weight down    = host_weight(bank.down_codes_plane() + shift,
                                                bank.down_scales_plane() + shift,
                                                kGeometry.routed_down_rows(), kGeometry.intermediate);
        host      = ops::expert_host_bank(kGeometry, gate_up, down);
        pool      = ops::create_expert_slot_pool(kGeometry, kSlots, pool_memory.data());
        directory = ops::create_expert_slot_directory(kLayers, kGeometry.experts, kSlots, 0,
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

    // --- Q4G32AM bank: the gather unpacks nibbles + affine grid into W8 pool planes. ---
    {
        std::mt19937 rng(20260829U);
        // A numeric W8 source (sane fp16 scales), requantised to a pinned q4 object per matrix.
        const std::size_t gate_rows = static_cast<std::size_t>(kGeometry.experts) *
                                      static_cast<std::size_t>(kGeometry.expert_rows());
        const std::size_t down_rows = static_cast<std::size_t>(kGeometry.experts) *
                                      static_cast<std::size_t>(kGeometry.hidden);
        auto fp16 = [](float v) {
            // encode via the op itself: requantise a 1-value... simpler: bit round-trip through
            // __fp16 is unavailable portably here, so use a coarse encoder: scale values are
            // chosen as exact powers of two, whose fp16 bits are exact.
            std::uint16_t bits = 0;
            if (v == 0.5F) { bits = 0x3800; }
            if (v == 0.25F) { bits = 0x3400; }
            if (v == 0.125F) { bits = 0x3000; }
            if (v == 0.0078125F) { bits = 0x2000; }
            return bits;
        };
        std::uniform_int_distribution<int> code(-127, 127);
        const std::uint16_t scale_choices[4] = {fp16(0.5F), fp16(0.25F), fp16(0.125F),
                                                fp16(0.0078125F)};
        std::vector<std::int8_t> gate_codes(gate_rows * kGeometry.hidden);
        std::vector<std::uint16_t> gate_scales(gate_rows * (kGeometry.hidden / 32));
        std::vector<std::int8_t> down_codes(down_rows * kGeometry.intermediate);
        std::vector<std::uint16_t> down_scales(down_rows * (kGeometry.intermediate / 32));
        for (auto& c : gate_codes) { c = static_cast<std::int8_t>(code(rng)); }
        for (auto& c : down_codes) { c = static_cast<std::int8_t>(code(rng)); }
        for (auto& v : gate_scales) { v = scale_choices[rng() % 4]; }
        for (auto& v : down_scales) { v = scale_choices[rng() % 4]; }

        const ops::Q4BankPlanes gate_planes =
            ops::q4_bank_planes(static_cast<std::int64_t>(gate_rows), kGeometry.hidden);
        const ops::Q4BankPlanes down_planes =
            ops::q4_bank_planes(static_cast<std::int64_t>(down_rows), kGeometry.intermediate);
        void* gate_pinned = nullptr;
        void* down_pinned = nullptr;
        cuda_check(cudaHostAlloc(&gate_pinned, gate_planes.total_bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable),
                   "q4 gate alloc");
        cuda_check(cudaHostAlloc(&down_pinned, down_planes.total_bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable),
                   "q4 down alloc");
        auto* gb = static_cast<std::byte*>(gate_pinned);
        auto* db = static_cast<std::byte*>(down_pinned);
        ops::requantise_w8_expert_groups_to_q4(
            gate_codes.data(), gate_scales.data(), static_cast<std::int64_t>(gate_planes.groups),
            reinterpret_cast<std::uint8_t*>(gb),
            reinterpret_cast<std::uint16_t*>(gb + gate_planes.scales_offset),
            reinterpret_cast<std::uint16_t*>(gb + gate_planes.mins_offset));
        ops::requantise_w8_expert_groups_to_q4(
            down_codes.data(), down_scales.data(), static_cast<std::int64_t>(down_planes.groups),
            reinterpret_cast<std::uint8_t*>(db),
            reinterpret_cast<std::uint16_t*>(db + down_planes.scales_offset),
            reinterpret_cast<std::uint16_t*>(db + down_planes.mins_offset));
        void* gate_mapped = nullptr;
        void* down_mapped = nullptr;
        cuda_check(cudaHostGetDevicePointer(&gate_mapped, gate_pinned, 0), "q4 gate map");
        cuda_check(cudaHostGetDevicePointer(&down_mapped, down_pinned, 0), "q4 down map");
        const ops::ExpertHostBank q4 =
            ops::expert_host_bank_q4(kGeometry, gate_mapped, down_mapped);

        GuardedDeviceBuffer pool_memory(ops::expert_slot_pool_bytes(kGeometry, kSlots));
        GuardedDeviceBuffer directory_memory(
            ops::expert_slot_directory_bytes(kLayers, kGeometry.experts, kSlots));
        GuardedDeviceBuffer miss_memory(ops::expert_miss_list_bytes(kGeometry.experts));
        GuardedDeviceBuffer ids_memory(8 * sizeof(int));
        ops::ExpertSlotPool pool = ops::create_expert_slot_pool(kGeometry, kSlots, pool_memory.data());
        ops::ExpertSlotDirectory directory = ops::create_expert_slot_directory(
            kLayers, kGeometry.experts, kSlots, 0, directory_memory.data(), nullptr);
        ops::ExpertMissList misses = ops::create_expert_miss_list(kGeometry.experts, miss_memory.data());
        const std::vector<int> ids = {0, 5, 3};
        cuda_check(cudaMemcpy(ids_memory.data(), ids.data(), ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "q4 ids upload");
        Tensor ids_tensor(ids_memory.data(), DType::I32, {3});
        ops::expert_slot_resolve(ids_tensor, 0, directory, misses, nullptr);
        ops::expert_slot_gather(q4, misses, pool, nullptr);
        cuda_synchronize();
        const auto miss_count   = from_device<long long>(misses.count.data, 1)[0];
        const auto miss_slots   = from_device<int>(misses.slots.data, static_cast<std::size_t>(miss_count));
        const auto miss_experts = from_device<int>(misses.experts.data, static_cast<std::size_t>(miss_count));
        failures += expect(miss_count == 3, "q4 gather misses == 3");

        // Expected pool planes: decode the q4 grid, requantise symmetric int8 with the same
        // float math as the kernel — byte-exact.
        auto fp16f = [](std::uint16_t h) {
            const std::uint32_t sign = (h & 0x8000U) << 16;
            std::uint32_t exp        = (h >> 10) & 0x1FU;
            std::uint32_t mant       = h & 0x3FFU;
            std::uint32_t f;
            if (exp == 0) {
                if (mant == 0) { f = sign; }
                else {
                    exp = 127 - 15 + 1;
                    while ((mant & 0x400U) == 0) { mant <<= 1; --exp; }
                    mant &= 0x3FFU;
                    f = sign | (exp << 23) | (mant << 13);
                }
            } else if (exp == 31) { f = sign | 0x7F800000U | (mant << 13); }
            else { f = sign | ((exp - 15 + 127) << 23) | (mant << 13); }
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return out;
        };
        for (std::size_t m = 0; m < static_cast<std::size_t>(miss_count); ++m) {
            const int slot   = miss_slots[m];
            const int expert = miss_experts[m];
            struct Matrix {
                const std::byte* src;
                const ops::Q4BankPlanes* planes;
                const std::byte* pool_codes;
                const std::byte* pool_scales;
                std::size_t rows_per_expert;
                int k;
            } matrices[2] = {
                {gb, &gate_planes, pool.gate_up_codes, pool.gate_up_scales,
                 static_cast<std::size_t>(kGeometry.expert_rows()), kGeometry.hidden},
                {db, &down_planes, pool.down_codes, pool.down_scales,
                 static_cast<std::size_t>(kGeometry.hidden), kGeometry.intermediate},
            };
            for (int which = 0; which < 2; ++which) {
                const Matrix& mx     = matrices[which];
                const std::size_t groups_per_expert = mx.rows_per_expert * mx.k / 32;
                const std::size_t codes_bytes       = groups_per_expert * 32;
                const std::vector<std::int8_t> got_codes = from_device<std::int8_t>(
                    mx.pool_codes + static_cast<std::size_t>(slot) * codes_bytes, codes_bytes);
                const std::vector<std::uint16_t> got_scales = from_device<std::uint16_t>(
                    mx.pool_scales + static_cast<std::size_t>(slot) * groups_per_expert * 2,
                    groups_per_expert);
                const auto* q4c = reinterpret_cast<const std::uint8_t*>(mx.src);
                const auto* q4s = reinterpret_cast<const std::uint16_t*>(mx.src + mx.planes->scales_offset);
                const auto* q4m = reinterpret_cast<const std::uint16_t*>(mx.src + mx.planes->mins_offset);
                int bad = 0;
                for (std::size_t g = 0; g < groups_per_expert && bad < 4; ++g) {
                    const std::size_t src_g = static_cast<std::size_t>(expert) * groups_per_expert + g;
                    const float step = fp16f(q4s[src_g]);
                    const float lo   = fp16f(q4m[src_g]);
                    float value[32];
                    float amax = 0.0F;
                    for (int i = 0; i < 32; ++i) {
                        const int q = (q4c[src_g * 16 + i / 2] >> (4 * (i % 2))) & 0xF;
                        value[i]     = step * static_cast<float>(q) + lo;
                        amax         = std::max(amax, std::fabs(value[i]));
                    }
                    const float scale = amax / 127.0F;
                    const float inv   = scale > 0.0F ? 1.0F / scale : 0.0F;
                    for (int i = 0; i < 32; ++i) {
                        const int want = static_cast<int>(std::nearbyint(value[i] * inv));
                        if (static_cast<int>(got_codes[g * 32 + i]) != want) {
                            std::cout << "FAIL q4 gather codes matrix " << which << " group " << g
                                      << " lane " << i << " got " << int(got_codes[g * 32 + i])
                                      << " want " << want << "\n";
                            ++bad;
                        }
                    }
                    // The kernel's __float2half_rn(scale): compare through the decoded value.
                    const float got_scale = fp16f(got_scales[g]);
                    if (std::fabs(got_scale - scale) > std::max(1e-3F * scale, 1e-8F)) {
                        std::cout << "FAIL q4 gather scale matrix " << which << " group " << g
                                  << " got " << got_scale << " want " << scale << "\n";
                        ++bad;
                    }
                }
                failures += bad != 0;
            }
        }
        cudaFreeHost(gate_pinned);
        cudaFreeHost(down_pinned);
    }
    // --- Scan ring: wide (scan) resolves cycle in the trailing ring; the LRU region and its
    // resident decode set survive a full expert sweep. ---
    {
        constexpr std::int32_t kRingSlots = 20; // 12 LRU + ring of 8 (= experts)
        GuardedDeviceBuffer pool_memory(ops::expert_slot_pool_bytes(kGeometry, kRingSlots));
        GuardedDeviceBuffer directory_memory(
            ops::expert_slot_directory_bytes(kLayers, kGeometry.experts, kRingSlots));
        GuardedDeviceBuffer miss_memory(ops::expert_miss_list_bytes(kGeometry.experts));
        GuardedDeviceBuffer ids_memory(16 * sizeof(int));
        ops::ExpertSlotPool pool =
            ops::create_expert_slot_pool(kGeometry, kRingSlots, pool_memory.data());
        ops::ExpertSlotDirectory directory = ops::create_expert_slot_directory(
            kLayers, kGeometry.experts, kRingSlots, kGeometry.experts, directory_memory.data(),
            nullptr);
        ops::ExpertMissList misses = ops::create_expert_miss_list(kGeometry.experts, miss_memory.data());
        HostBank ring_bank = make_bank();
        void* mapped       = nullptr;
        cuda_check(cudaHostGetDevicePointer(&mapped, ring_bank.pinned, 0), "ring bank map");
        const std::ptrdiff_t shift = static_cast<std::byte*>(mapped) - ring_bank.base();
        const Weight gate_up = host_weight(ring_bank.gate_codes_plane() + shift,
                                           ring_bank.gate_scales_plane() + shift,
                                           kGeometry.routed_gate_rows(), kGeometry.hidden);
        const Weight down = host_weight(ring_bank.down_codes_plane() + shift,
                                        ring_bank.down_scales_plane() + shift,
                                        kGeometry.routed_down_rows(), kGeometry.intermediate);
        const ops::ExpertHostBank host = ops::expert_host_bank(kGeometry, gate_up, down);

        const auto resolve = [&](const std::vector<int>& ids, int layer, bool scan) {
            cuda_check(cudaMemcpy(ids_memory.data(), ids.data(), ids.size() * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       "ring ids upload");
            Tensor t(ids_memory.data(), DType::I32, {static_cast<std::int32_t>(ids.size())});
            ops::expert_slot_resolve(t, layer, directory, misses, nullptr, scan);
            ops::expert_slot_gather(host, misses, pool, nullptr);
            cuda_synchronize();
            return from_device<int>(directory.slot_of_expert.data,
                                    static_cast<std::size_t>(kLayers) * kGeometry.experts);
        };

        // Decode-style resolves park experts 0..3 of layer 0 in the LRU region [0, 12).
        auto table = resolve({0, 1, 2, 3}, 0, false);
        int lru_ok = 1;
        std::array<int, 4> resident{};
        for (int e = 0; e < 4; ++e) {
            resident[static_cast<std::size_t>(e)] = table[e];
            lru_ok &= table[e] >= 0 && table[e] < 12;
        }
        failures += expect(lru_ok == 1, "ring: decode set placed in the LRU region");

        // A full scan of layer 1 (all 8 experts) lands entirely in the ring [12, 20) and the
        // decode set keeps its slots.
        table = resolve({0, 1, 2, 3, 4, 5, 6, 7}, 1, true);
        int ring_ok = 1;
        for (int e = 0; e < kGeometry.experts; ++e) {
            const int slot = table[kGeometry.experts + e];
            ring_ok &= slot >= 12 && slot < 20;
        }
        failures += expect(ring_ok == 1, "ring: scan misses placed in the ring only");
        int kept = 1;
        for (int e = 0; e < 4; ++e) { kept &= table[e] == resident[static_cast<std::size_t>(e)]; }
        failures += expect(kept == 1, "ring: decode set survives a full scan");

        // A second full scan (layer 0's other experts) recycles the ring, evicting layer 1's
        // scan entries but still never touching the LRU region.
        table = resolve({4, 5, 6, 7, 0, 1, 2, 3}, 0, true);
        int recycled = 1;
        for (int e = 4; e < 8; ++e) { recycled &= table[e] >= 12 && table[e] < 20; }
        failures += expect(recycled == 1, "ring: second scan recycles the ring");
        kept = 1;
        for (int e = 0; e < 4; ++e) { kept &= table[e] == resident[static_cast<std::size_t>(e)]; }
        failures += expect(kept == 1, "ring: decode set survives ring recycling (hits keep slots)");
        cudaFreeHost(ring_bank.pinned);
        failures += pool_memory.verify_guards("ring pool");
        failures += directory_memory.verify_guards("ring directory");
    }
    // --- CPU split job list: a job's token comes from the stated paths-per-token count, for
    // a flat [assignments] view (the prefill path) and a [paths, tokens] view alike. Reading
    // it off the leading extent put every job of a flat round on token 0 (2026-08-29). ---
    {
        constexpr std::int32_t kPaths  = 2;
        constexpr std::int32_t kTokens = 4;
        GuardedDeviceBuffer directory_memory(
            ops::expert_slot_directory_bytes(kLayers, kGeometry.experts, kSlots));
        GuardedDeviceBuffer miss_memory(ops::expert_miss_list_bytes(kGeometry.experts));
        GuardedDeviceBuffer jobs_memory(ops::expert_cpu_job_list_bytes(kPaths * kTokens));
        GuardedDeviceBuffer ids_memory(kPaths * kTokens * sizeof(int));
        GuardedDeviceBuffer alpha_memory(kPaths * kTokens * sizeof(float));
        ops::ExpertSlotDirectory directory = ops::create_expert_slot_directory(
            kLayers, kGeometry.experts, kSlots, 0, directory_memory.data(), nullptr);
        ops::ExpertMissList misses = ops::create_expert_miss_list(kGeometry.experts, miss_memory.data());
        ops::ExpertCpuJobList jobs = ops::create_expert_cpu_job_list(kPaths * kTokens, jobs_memory.data());
        // Token-major: token t routes to experts (2t, 2t+1) with weights 0.1 (t+1), 0.2 (t+1).
        std::vector<int> ids;
        std::vector<float> alpha;
        for (int t = 0; t < kTokens; ++t) {
            for (int path = 0; path < kPaths; ++path) {
                ids.push_back(2 * t + path);
                alpha.push_back(0.1F * static_cast<float>(path + 1) * static_cast<float>(t + 1));
            }
        }
        cuda_check(cudaMemcpy(ids_memory.data(), ids.data(), ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice), "split ids upload");
        cuda_check(cudaMemcpy(alpha_memory.data(), alpha.data(), alpha.size() * sizeof(float),
                              cudaMemcpyHostToDevice), "split alpha upload");
        for (const bool flat : {true, false}) {
            ops::expert_slot_directory_reset(directory, nullptr);
            const Tensor t = flat ? Tensor(ids_memory.data(), DType::I32, {kPaths * kTokens})
                                  : Tensor(ids_memory.data(), DType::I32, {kPaths, kTokens});
            const Tensor a = flat ? Tensor(alpha_memory.data(), DType::FP32, {kPaths * kTokens})
                                  : Tensor(alpha_memory.data(), DType::FP32, {kPaths, kTokens});
            // Every miss goes to the host: share 1.0.
            ops::expert_slot_resolve(t, a, 0, directory, misses, &jobs, 65536U, kPaths, nullptr,
                                     false);
            cuda_synchronize();
            const auto count   = from_device<long long>(jobs.count.data, 1)[0];
            const auto tokens  = from_device<int>(jobs.tokens.data, static_cast<std::size_t>(kPaths * kTokens));
            const auto experts = from_device<int>(jobs.experts.data, static_cast<std::size_t>(kPaths * kTokens));
            const auto weights = from_device<float>(jobs.weights.data, static_cast<std::size_t>(kPaths * kTokens));
            const std::string shape = flat ? "flat" : "2-d";
            failures += expect(count == kPaths * kTokens,
                               "split " + shape + ": every path became a job (got " +
                                   std::to_string(count) + ")");
            int consistent = 1;
            for (long long j = 0; j < std::min<long long>(count, kPaths * kTokens); ++j) {
                const int expert = experts[static_cast<std::size_t>(j)];
                const int token  = tokens[static_cast<std::size_t>(j)];
                // expert 2t+path belongs to token t and carries alpha[2t+path].
                consistent &= expert >= 0 && expert < kPaths * kTokens && token == expert / kPaths &&
                              weights[static_cast<std::size_t>(j)] == alpha[static_cast<std::size_t>(expert)];
            }
            failures += expect(consistent == 1, "split " + shape + ": jobs carry their own token and weight");
        }
    }

    std::cout << (failures ? "FAIL" : "OK") << " expert_slot_cache\n";
    return failures ? 1 : 0;
}
