// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_UTILS_PHILOX_H
#define SUROGATE_SRC_UTILS_PHILOX_H

#include <array>
#include <cstdint>
#include <utility>

/**
 * @brief Philox 4x32 counter-based pseudo-random number generator.
 *
 * Implements a Philox-like bijective mixing function operating on a 128-bit
 * counter state split into four 32-bit lanes, producing 4x 32-bit outputs per
 * invocation.
 *
 * This implementation uses a 64-bit seed expanded into a 2x32-bit key.
 */
class Philox4x32 {
private:
    static constexpr std::uint32_t MC[4] = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
    static constexpr int ROUNDS = 10;

    uint32_t key[2];

    /**
     * @brief Multiply two 32-bit values and return the high/low 32-bit halves.
     *
     * @param a First multiplicand.
     * @param b Second multiplicand.
     * @return Pair {high32, low32} of the 64-bit product a*b.
     */
    static std::pair<std::uint32_t, std::uint32_t> mul_hilo(std::uint32_t a, std::uint32_t b) {
        std::uint64_t ab = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
        return {static_cast<std::uint32_t>(ab >> 32ull), static_cast<std::uint32_t>(ab & 0x00000000ffffffffull)};
    }

public:
    /**
     * @brief Construct a generator using a 64-bit seed.
     *
     * The seed is split into two 32-bit words:
     * - key[0] = low 32 bits of @p seed
     * - key[1] = high 32 bits of @p seed
     *
     * @param seed 64-bit seed value used to initialize the 2x32-bit key.
     */
    explicit Philox4x32(uint64_t seed = 1) {
        key[0] = static_cast<uint32_t>(seed);
        key[1] = static_cast<uint32_t>(seed >> 32);
    }

    /**
     * @brief Construct a generator using two 32-bit seed words.
     *
     * @param seed_a First 32-bit seed word (stored as key[0]).
     * @param seed_b Second 32-bit seed word (stored as key[1]).
     */
    Philox4x32(uint32_t seed_a, std::uint32_t seed_b)
        : key{seed_a, seed_b} {
    }

    /**
     * @brief Generate 4 pseudo-random 32-bit values for a given counter.
     *
     * Interprets @p x and @p y as the low 64 bits of the counter. The remaining
     * two 32-bit counter lanes are set to zero in this implementation.
     *
     * @param x Counter word 0 (lane R0).
     * @param y Counter word 1 (lane L0).
     * @return Array of four 32-bit outputs in the order {R0, L0, R1, L1}.
     */
    std::array<uint32_t, 4> generate(std::uint32_t x, std::uint32_t y) {
        // Extract counter parts
        uint32_t R0 = x;
        uint32_t L0 = y;

        uint32_t R1 = 0;  // High part of counter for 4x32
        uint32_t L1 = 0;

        // Initialize keys
        uint32_t K0 = key[0];
        uint32_t K1 = key[1];

        // Perform rounds
        for (int i = 0; i < ROUNDS; ++i) {
            auto [hi0, lo0] = mul_hilo(R0, MC[0]);
            auto [hi1, lo1] = mul_hilo(R1, MC[2]);

            R0 = hi1 ^ L0 ^ K0;
            L0 = lo1;
            R1 = hi0 ^ L1 ^ K1;
            L1 = lo0;

            K0 = (K0 + MC[1]) & 0xFFFFFFFF;
            K1 = (K1 + MC[3]) & 0xFFFFFFFF;
        }

        return {R0, L0, R1, L1};
    }
};

#endif  //SUROGATE_SRC_UTILS_PHILOX_H
