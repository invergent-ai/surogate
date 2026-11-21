#pragma once

// The visit schedule of Inkling-2-Mini as constexpr data (DESIGN-target.md 1.2): for each of the 48 visits
// (prelude 0-5, core 6-17 in passes 0..2, coda 18-23) its physical layer and pass, which AttnRes stack slots
// it reads, where its running partial block lives, where a block-boundary ("first") layer starts the next
// partial, and which two slots hold the previous pass's bank. Derived from `loop.py` Inkling2LoopModel
// .forward (nb sequence, pass restarts) with the stack stored as 7 slots per column:
//   0 = normed embedding, 1 = prelude block, 2-3 = the blocks of passes 0 and 2 (B6, P12), 4-5 = those of
//   pass 1, 6 = the coda's partial. Pass p >= 1 reads its messages from the other pair (no bank copy).

#include "targets/inkling2/impl/config.h"

#include <array>
#include <cstdint>

namespace sinfer::targets::inkling2::detail {

struct VisitInfo {
    std::int8_t layer       = 0;  ///< physical layer 0..23
    std::int8_t pass        = 0;  ///< 0..2 (prelude and coda run as pass 0)
    std::int8_t stack_count = 0;  ///< completed-block sources read before the partial
    std::array<std::int8_t, 4> stack{-1, -1, -1, -1};
    std::int8_t partial     = 0;  ///< slot of the running partial at the visit's start
    std::int8_t next_partial = 0; ///< slot the MLP conv writes (== partial unless `first`)
    std::array<std::int8_t, 2> bank{-1, -1}; ///< previous pass's (B6, P12) slots for a looped visit
    std::int8_t feature     = -1; ///< DSpark feature index captured at the visit's start (0..2) or -1
    bool first   = false;         ///< layer % 6 == 0: the MLP input closes a block
    bool global  = false;
    bool core    = false;
    bool looped  = false;         ///< core and pass >= 1: messages blend into every input
};

namespace visits_detail {

constexpr VisitInfo make(int layer, int pass) {
    using C = TextConfig;
    VisitInfo v;
    v.layer  = static_cast<std::int8_t>(layer);
    v.pass   = static_cast<std::int8_t>(pass);
    v.first  = layer % C::block_layers == 0;
    v.global = C::is_global(layer);
    v.core   = layer >= C::core_start && layer < C::core_end;
    v.looped = v.core && pass >= 1;
    const auto set_stack = [&](std::initializer_list<int> slots) {
        v.stack_count = 0;
        for (int s : slots) { v.stack[v.stack_count++] = static_cast<std::int8_t>(s); }
    };
    if (layer < C::core_start) {
        if (layer == 0) {
            set_stack({});
            v.partial      = 0;
            v.next_partial = 1;
        } else {
            set_stack({0});
            v.partial = v.next_partial = 1;
        }
    } else if (v.core) {
        const int base = pass % 2 == 0 ? 2 : 4; // this pass's (B6, P12) pair
        if (layer < 12) {
            if (layer == 6) {
                set_stack({0});
                v.partial      = 1;
                v.next_partial = static_cast<std::int8_t>(base);
            } else {
                set_stack({0, 1});
                v.partial = v.next_partial = static_cast<std::int8_t>(base);
            }
        } else {
            if (layer == 12) {
                set_stack({0, 1});
                v.partial      = static_cast<std::int8_t>(base);
                v.next_partial = static_cast<std::int8_t>(base + 1);
            } else {
                set_stack({0, 1, base});
                v.partial = v.next_partial = static_cast<std::int8_t>(base + 1);
            }
        }
        if (pass >= 1) {
            const int previous = (pass - 1) % 2 == 0 ? 2 : 4;
            v.bank = {static_cast<std::int8_t>(previous), static_cast<std::int8_t>(previous + 1)};
        }
    } else {
        // Coda: continues pass 2 (even, so blocks in slots 2-3).
        if (layer == C::core_end) {
            set_stack({0, 1, 2});
            v.partial      = 3;
            v.next_partial = 6;
        } else {
            set_stack({0, 1, 2, 3});
            v.partial = v.next_partial = 6;
        }
    }
    return v;
}

constexpr std::array<VisitInfo, TextConfig::layers> build() {
    std::array<VisitInfo, TextConfig::layers> out{};
    int v = 0;
    for (int i = 0; i < TextConfig::core_start; ++i) { out[v++] = make(i, 0); }
    for (int p = 0; p < TextConfig::passes; ++p) {
        for (int i = TextConfig::core_start; i < TextConfig::core_end; ++i) { out[v++] = make(i, p); }
    }
    for (int i = TextConfig::core_end; i < TextConfig::physical_layers; ++i) { out[v++] = make(i, 0); }
    out[5].feature  = 0;
    out[23].feature = 1;
    out[41].feature = 2;
    return out;
}

} // namespace visits_detail

inline constexpr std::array<VisitInfo, TextConfig::layers> kVisits = visits_detail::build();

/// The final output mix: stack [0, 1, 2, 3] + the coda partial (slot 6) with the output query and the final norm.
inline constexpr std::array<std::int8_t, 4> kFinalStack{0, 1, 2, 3};
inline constexpr std::int8_t kFinalPartial = 6;

/// visit = layer (prelude), layer + 12 * pass (core), layer + 24 (coda) -- OPS.md section 1.
constexpr int visit_of(int layer, int pass) {
    if (layer < TextConfig::core_start) { return layer; }
    if (layer >= TextConfig::core_end) { return layer + (TextConfig::passes - 1) * (TextConfig::core_end - TextConfig::core_start); }
    return layer + pass * (TextConfig::core_end - TextConfig::core_start);
}

// The table of DESIGN-target.md 1.2, checked at compile time.
static_assert(kVisits[0].partial == 0 && kVisits[0].next_partial == 1 && kVisits[0].stack_count == 0);
static_assert(kVisits[5].layer == 5 && kVisits[5].global && kVisits[5].partial == 1 && kVisits[5].feature == 0);
static_assert(kVisits[6].layer == 6 && kVisits[6].first && kVisits[6].next_partial == 2 && kVisits[6].stack_count == 1);
static_assert(kVisits[12].next_partial == 3 && kVisits[13].stack_count == 3 && kVisits[13].stack[2] == 2);
static_assert(kVisits[18].layer == 6 && kVisits[18].pass == 1 && kVisits[18].looped && kVisits[18].partial == 1 &&
              kVisits[18].next_partial == 4 && kVisits[18].bank[0] == 2 && kVisits[18].bank[1] == 3);
static_assert(kVisits[23].layer == 11 && kVisits[23].partial == 4 && kVisits[23].feature == 1 && kVisits[23].global);
static_assert(kVisits[24].next_partial == 5 && kVisits[25].stack[2] == 4 && kVisits[25].partial == 5);
static_assert(kVisits[30].pass == 2 && kVisits[30].next_partial == 2 && kVisits[30].bank[0] == 4 && kVisits[30].bank[1] == 5);
static_assert(kVisits[41].layer == 17 && kVisits[41].partial == 3 && kVisits[41].feature == 2);
static_assert(kVisits[42].layer == 18 && !kVisits[42].core && kVisits[42].stack_count == 3 && kVisits[42].partial == 3 &&
              kVisits[42].next_partial == 6);
static_assert(kVisits[47].layer == 23 && kVisits[47].global && kVisits[47].stack_count == 4 && kVisits[47].partial == 6);
static_assert(visit_of(11, 1) == 23 && visit_of(17, 2) == 41 && visit_of(23, 0) == 47);
// Global visits 5, 11, 17, 23, 29, 35, 41, 47 (OPS.md 1).
static_assert(kVisits[11].global && kVisits[17].global && kVisits[29].global && kVisits[35].global && !kVisits[10].global);

// The whole table against a replay of the reference's own stack (`loop.py` Inkling2LoopModel.forward): every
// stack entry is named by what it holds -- 1 = the embedding, 2 = the prelude block, 10 + 2p / 11 + 2p = pass
// p's blocks of layers 6-11 / 12-17, 20 = the coda's -- and for every visit the sources, the partial, the
// slot the MLP writes and the message bank must name the same entries in the reference and in the slots.
namespace visits_detail {

constexpr int entry_of(int layer, int pass) {
    if (layer < TextConfig::core_start) { return 2; }
    if (layer >= TextConfig::core_end) { return 20; }
    return (layer < 12 ? 10 : 11) + 2 * pass;
}

constexpr bool replay_matches() {
    int ref[8]{1, 0, 0, 0, 0, 0, 0, 0}; // the reference's stack (entries)
    int nb = 0, prelude_blocks = -1;
    int bank0 = 0, bank1 = 0;
    int slot[TextConfig::stack_slots]{1, 0, 0, 0, 0, 0, 0}; // ours
    for (int v = 0; v < TextConfig::layers; ++v) {
        const VisitInfo& vi = kVisits[v];
        const int i = vi.layer, p = vi.pass;
        if (i == TextConfig::core_start && p == 0) {
            prelude_blocks = nb;
        } else if (i == TextConfig::core_start) {
            bank0 = ref[prelude_blocks + 1];
            bank1 = ref[prelude_blocks + 2];
            nb    = prelude_blocks;
            ref[nb] = 2; // stack[nb].copy_(prelude_partial)
        }
        // sources, partial, bank
        if (vi.stack_count != nb) { return false; }
        for (int k = 0; k < nb; ++k) {
            if (slot[vi.stack[k]] != ref[k]) { return false; }
        }
        if (slot[vi.partial] != ref[nb]) { return false; }
        if (vi.looped && (slot[vi.bank[0]] != bank0 || slot[vi.bank[1]] != bank1)) { return false; }
        if (vi.first != (i % TextConfig::block_layers == 0)) { return false; }
        // the MLP conv: a first layer closes a block (nb + 1) and starts the next partial
        if (vi.first) { ++nb; }
        ref[nb]               = entry_of(i, p);
        slot[vi.next_partial] = entry_of(i, p);
        if (!vi.first && vi.next_partial != vi.partial) { return false; }
    }
    // the final output mix reads the reference's stack[:nb] + stack[nb]
    if (nb != static_cast<int>(kFinalStack.size())) { return false; }
    for (int k = 0; k < nb; ++k) {
        if (slot[kFinalStack[k]] != ref[k]) { return false; }
    }
    return slot[kFinalPartial] == ref[nb];
}

} // namespace visits_detail

static_assert(visits_detail::replay_matches(), "the visit table disagrees with the reference's AttnRes stack");

} // namespace sinfer::targets::inkling2::detail
