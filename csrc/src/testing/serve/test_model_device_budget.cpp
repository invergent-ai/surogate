#include "serve/model_device_budget.h"

#include <cassert>
#include <limits>

int main() {
    using sinfer::serve::model_memory_shortfall;
    const sinfer::serve::DeviceBytes budgets{{0, 100}, {1, 10000}};
    const sinfer::serve::DeviceBytes needed{{0, 90}, {1, 30}};
    assert(model_memory_shortfall(needed, {{0, 20}}, budgets) == std::vector<int>{0});
    // Evicting a model on GPU 1 cannot make room on GPU 0.
    assert(model_memory_shortfall(needed, {{0, 20}, {1, 100}}, budgets) == std::vector<int>{0});
    assert(model_memory_shortfall(needed, {{0, 10}, {1, 100}}, budgets).empty());
    assert(model_memory_shortfall({{1, 30}}, {{0, 100}}, budgets).empty());
    assert(model_memory_shortfall({{0, 101}}, {}, budgets) == std::vector<int>{0});
    const auto maximum = std::numeric_limits<std::size_t>::max();
    assert(model_memory_shortfall({{0, maximum}}, {{0, 1}}, {{0, maximum}}) == std::vector<int>{0});
}
