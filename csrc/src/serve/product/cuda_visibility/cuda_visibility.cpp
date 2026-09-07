#include "product/cuda_visibility/cuda_visibility.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::product {

namespace {

std::vector<std::string> split_entries(const std::string& value) {
    std::vector<std::string> entries;
    std::size_t start = 0;
    while (start <= value.size()) {
        const std::size_t comma = value.find(',', start);
        entries.push_back(value.substr(start, comma == std::string::npos ? std::string::npos
                                                                         : comma - start));
        if (comma == std::string::npos) { break; }
        start = comma + 1;
    }
    return entries;
}

} // namespace

CudaVisibilityPlan plan_cuda_visibility(const std::string* existing,
                                        const std::vector<int>& requested) {
    CudaVisibilityPlan plan;
    if (requested.empty()) { return plan; }
    std::vector<std::string> entries;
    if (existing != nullptr) { entries = split_entries(*existing); }
    std::vector<std::string> chosen;
    for (const int index : requested) {
        if (index < 0) { throw std::invalid_argument("a CUDA device index is negative"); }
        std::string entry;
        if (existing != nullptr) {
            if (static_cast<std::size_t>(index) >= entries.size()) {
                throw std::invalid_argument(
                    "device " + std::to_string(index) + " is outside CUDA_VISIBLE_DEVICES=" +
                    *existing);
            }
            entry = entries[static_cast<std::size_t>(index)];
        } else {
            entry = std::to_string(index);
        }
        std::size_t position = 0;
        while (position < chosen.size() && chosen[position] != entry) { ++position; }
        if (position == chosen.size()) { chosen.push_back(entry); }
        plan.renumbered.push_back(static_cast<int>(position));
    }
    for (std::size_t i = 0; i < chosen.size(); ++i) {
        if (i != 0) { plan.visible += ','; }
        plan.visible += chosen[i];
    }
    plan.changed = existing == nullptr || *existing != plan.visible;
    return plan;
}

std::string narrow_cuda_visible_devices(int& device, std::vector<int>& devices) {
    const char* raw = std::getenv("CUDA_VISIBLE_DEVICES");
    const std::string existing = raw != nullptr ? raw : "";
    const std::vector<int> requested = devices.empty() ? std::vector<int>{device} : devices;
    const CudaVisibilityPlan plan =
        plan_cuda_visibility(raw != nullptr ? &existing : nullptr, requested);
    if (plan.renumbered.empty()) { return {}; }
    if (setenv("CUDA_VISIBLE_DEVICES", plan.visible.c_str(), 1) != 0) {
        throw std::runtime_error("could not set CUDA_VISIBLE_DEVICES");
    }
    std::string asked;
    for (std::size_t i = 0; i < requested.size(); ++i) {
        if (i != 0) { asked += ','; }
        asked += std::to_string(requested[i]);
    }
    if (devices.empty()) {
        device = plan.renumbered.front();
    } else {
        devices = plan.renumbered;
        device  = devices.front();
    }
    if (!plan.changed && plan.renumbered == requested) { return {}; }
    return "cuda: this process sees only " + (raw != nullptr ? "entries " + asked + " of CUDA_VISIBLE_DEVICES=" + existing : "device(s) " + asked) +
           ", set as CUDA_VISIBLE_DEVICES=" + plan.visible + " and numbered 0.." +
           std::to_string(plan.visible.empty() ? 0 : static_cast<int>(split_entries(plan.visible).size()) - 1) +
           " from here";
}

} // namespace sinfer::product
