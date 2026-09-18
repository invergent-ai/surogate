// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Exact arithmetic and layout checks for CPU speech operations, without models.
#include "cpu_ops.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <stdexcept>

using at::Tensor;
using namespace sinfer::speech;

static void equal(const Tensor& expected, const Tensor& actual, const char* operation) {
    if (!at::equal(expected.contiguous().view(at::kInt), actual.contiguous().view(at::kInt)))
        throw std::runtime_error(std::string(operation) + " differs from reference FP32 arithmetic");
}

int main() {
    try {
        c10::InferenceMode inference;
        at::manual_seed(104729);
        int cases = 0;
        for (int batch : {1, 2}) {
            for (int heads : {1, 8}) {
                for (int queries : {1, 2, 7, 14, 31, 64}) {
                    for (int cache : {0, 1, 70}) {
                        int keys = queries + cache, positions = 2 * keys - 1;
                        // A nonzero storage offset checks that views retain their origin.
                        auto storage = at::randn({2, batch, heads, queries, positions});
                        auto scores = storage[1];
                        auto expected = at::constant_pad_nd(scores, {1, 0}, 0)
                                            .view({batch, heads, -1, queries})
                                            .slice(2, 1)
                                            .reshape({batch, heads, queries, positions})
                                            .slice(3, 0, keys);
                        auto actual = cpu::relative_shift(scores, keys);
                        equal(expected, actual, "relative shift");
                        if (!actual.is_alias_of(scores)) throw std::runtime_error("Relative shift allocated storage");
                        auto other = at::randn(actual.sizes());
                        equal(expected + other, actual + other, "strided attention addition");
                        ++cases;
                    }
                }
            }
        }
        auto weight = at::randn({512, 2048});
        auto bias = at::randn({512});
        cpu::PackedLinear cache;
        for (int threads : {1, 2, 3, 4, 6, 8, 16}) {
            at::set_num_threads(threads);
            for (int rows : {1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 25, 31, 32, 61, 127}) {
                for (bool have_bias : {false, true}) {
                    std::optional<Tensor> b = have_bias ? std::optional<Tensor>(bias) : std::nullopt;
                    for (int repeat = 0; repeat < 5; ++repeat) {
                        auto storage = at::randn({1, rows, 4096});
                        auto input = storage.slice(2, 0, 2048);
                        if (repeat != 4) input = input.contiguous();
                        auto before = input.clone();
                        try {
                            equal(at::linear(input, weight, b), cache.run(input, weight, b), "packed linear");
                        } catch (...) {
                            std::cerr << "rows=" << rows << " threads=" << threads << " bias=" << have_bias
                                      << " contiguous=" << input.is_contiguous() << '\n';
                            throw;
                        }
                        equal(before, input, "input preservation");
                        ++cases;
                    }
                }
            }
        }
        at::set_num_threads(4);
        auto changed_weight = weight + 1;
        auto input = at::randn({1, 14, 2048});
        for (int repeat = 0; repeat < 4; ++repeat) {
            equal(at::linear(input, changed_weight, bias),
                  cache.run(input, changed_weight, bias),
                  "weight replacement");
            ++cases;
        }
        std::cout << cases << " CPU speech operation cases matched exactly\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
