#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace sinfer::runtime {

// A layer and the layer whose KV it consumes must remain on the same stage.
// Coalesce those dependencies before balancing the weights across devices.
inline std::vector<int> pipeline_partition(std::span<const std::uint64_t> weights,
                                           std::span<const int> kv_sources, int stages) {
    const int layers = static_cast<int>(weights.size());
    if (layers == 0 || kv_sources.size() != weights.size() || stages <= 0) {
        throw std::invalid_argument("invalid pipeline partition inputs");
    }
    std::vector<bool> cuts(weights.size() + 1, true);
    for (int layer = 0; layer < layers; ++layer) {
        if (kv_sources[layer] < 0 || kv_sources[layer] > layer) {
            throw std::invalid_argument("invalid pipeline KV dependency");
        }
        for (int cut = kv_sources[layer] + 1; cut <= layer; ++cut) { cuts[cut] = false; }
    }
    std::vector<int> boundaries{0};
    std::vector<std::uint64_t> groups;
    std::uint64_t sum = 0;
    for (int layer = 0; layer < layers; ++layer) {
        sum += weights[layer];
        if (cuts[layer + 1]) {
            boundaries.push_back(layer + 1);
            groups.push_back(sum);
            sum = 0;
        }
    }
    if (stages > static_cast<int>(groups.size())) {
        throw std::invalid_argument("too many pipeline GPUs to keep shared-KV layers together");
    }
    std::uint64_t total = 0;
    for (auto& weight : groups) {
        weight = std::max<std::uint64_t>(weight, 1);
        total += weight;
    }
    std::uint64_t low = *std::max_element(groups.begin(), groups.end()), high = total;
    while (low < high) {
        const auto mid     = low + (high - low) / 2;
        int required       = 1;
        std::uint64_t used = 0;
        for (auto weight : groups) {
            if (used != 0 && used + weight > mid) {
                ++required;
                used = 0;
            }
            used += weight;
        }
        if (required <= stages) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    std::vector<int> out{0};
    std::uint64_t used = 0;
    for (std::size_t group = 0; group < groups.size(); ++group) {
        const auto remaining = stages - static_cast<int>(out.size());
        if (remaining > 0 && used != 0 &&
            (groups.size() - group <= static_cast<std::size_t>(remaining) ||
             used + groups[group] > low)) {
            out.push_back(boundaries[group]);
            used = 0;
        }
        used += groups[group];
    }
    out.push_back(layers);
    return out;
}

} // namespace sinfer::runtime
