#include "runtime/engine/pipeline_partition.h"

#include <cassert>
#include <iostream>

int main() {
    const std::vector<std::uint64_t> weights{1, 1, 1, 1, 1, 1};
    const std::vector<int> independent{0, 1, 2, 3, 4, 5};
    const auto balanced = sinfer::runtime::pipeline_partition(weights, independent, 3);
    assert((balanced == std::vector<int>{0, 2, 4, 6}));
    // Layers 4/5 share layer 2/3's KV: neither dependency may cross a stage.
    const std::vector<int> shared{0, 1, 2, 3, 2, 3};
    const auto grouped = sinfer::runtime::pipeline_partition(weights, shared, 3);
    assert((grouped == std::vector<int>{0, 1, 2, 6}));
    bool rejected = false;
    try {
        (void)sinfer::runtime::pipeline_partition(weights, shared, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    const std::vector<std::uint64_t> zero(6, 0);
    assert(sinfer::runtime::pipeline_partition(zero, independent, 6) == (std::vector<int>{0, 1, 2, 3, 4, 5, 6}));
    std::cout << "pipeline partition preserves shared-KV dependencies\n";
}
