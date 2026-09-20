// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

// Hard gold-token targets only; sidecar logprobs are ignored. RPS follows the
// semantic ordinal order of non-padding IDs, never token-ID order.
enum class CandidateObjective : int { CrossEntropy = 0, Brier = 1, Rps = 2 };

inline CandidateObjective parse_candidate_objective(std::string_view name) {
    if (name == "cross_entropy") return CandidateObjective::CrossEntropy;
    if (name == "brier") return CandidateObjective::Brier;
    if (name == "rps") return CandidateObjective::Rps;
    throw std::invalid_argument("candidate_objective must be cross_entropy, brier, or rps");
}

inline void validate_candidate_objective(CandidateObjective objective) {
    switch (objective) {
        case CandidateObjective::CrossEntropy:
        case CandidateObjective::Brier:
        case CandidateObjective::Rps: return;
    }
    throw std::invalid_argument("Invalid candidate objective enum");
}

// Called on all rank slices before any worker/collective is launched.
inline void validate_candidate_rows(const int* targets, const int* ids,
                                    std::size_t rows, int slots, int vocab,
                                    CandidateObjective objective) {
    validate_candidate_objective(objective);
    if (!targets || !ids || slots < 2 || slots > 1024 || vocab < 2) {
        throw std::invalid_argument("candidate_only requires valid buffers and K in [2,1024]");
    }
    for (std::size_t row = 0; row < rows; ++row) {
        if (targets[row] == -100) continue;
        int count = 0;
        bool gold_found = false;
        for (int k = 0; k < slots; ++k) {
            const int id = ids[row * slots + k];
            if (id == -1) continue;
            if (id < 0 || id >= vocab) {
                throw std::invalid_argument("candidate_only: candidate token outside vocabulary");
            }
            for (int prior = 0; prior < k; ++prior) {
                if (ids[row * slots + prior] == id) {
                    throw std::invalid_argument("candidate_only: duplicate candidate token");
                }
            }
            ++count;
            gold_found = gold_found || id == targets[row];
        }
        if (count < 2 || !gold_found ||
            (objective != CandidateObjective::CrossEntropy && count > 255)) {
            throw std::invalid_argument("candidate_only: supervised rows need 2..255 allowed candidates including gold for Brier/RPS (CE permits 1024)");
        }
    }
}
