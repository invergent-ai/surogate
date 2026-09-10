#pragma once

#include <cstddef>
#include <map>
#include <vector>

namespace sinfer::serve {

using DeviceBytes = std::map<int, std::size_t>;

inline std::vector<int> model_memory_shortfall(const DeviceBytes& needed,
                                               const DeviceBytes& resident,
                                               const DeviceBytes& budgets) {
    std::vector<int> shortfall;
    for (const auto& [device, bytes] : needed) {
        const auto budget = budgets.at(device);
        const auto found  = resident.find(device);
        const auto used   = found == resident.end() ? 0 : found->second;
        if (bytes > budget || used > budget - bytes) { shortfall.push_back(device); }
    }
    return shortfall;
}

} // namespace sinfer::serve
