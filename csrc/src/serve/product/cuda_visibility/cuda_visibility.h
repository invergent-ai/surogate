#pragma once

// The cards a process may touch. CUDA reads CUDA_VISIBLE_DEVICES once, at its first call, and
// from then on pinned host memory -- every staging buffer, every bank, every boundary -- is
// mapped into every visible device, which stands up a context on cards this engine was never
// asked to run on: 500 MiB on each, on a shared host someone else's. So the product entry
// points narrow the variable to the cards they were given before anything touches CUDA, and
// number their devices within that set.

#include <string>
#include <vector>

namespace sinfer::product {

struct CudaVisibilityPlan {
    /// The value CUDA_VISIBLE_DEVICES takes: the requested cards in first-appearance order,
    /// each translated through the variable's existing entries when it was already set.
    std::string visible;
    /// The requested indices, renumbered into that enumeration (0..n-1), in request order.
    std::vector<int> renumbered;
    bool changed = false;
};

/// Pure: what narrowing `requested` (physical indices, or indices into `existing`'s entries
/// when that is the variable's current value) would set and renumber. Throws when a requested
/// index selects outside an existing list.
[[nodiscard]] CudaVisibilityPlan plan_cuda_visibility(const std::string* existing,
                                                      const std::vector<int>& requested);

/// Applies the plan for `device` (a single card) or `devices` (pipeline stages, in order):
/// sets CUDA_VISIBLE_DEVICES and rewrites both to the narrowed numbering. Returns the line to
/// log, empty when the variable already said exactly this. Call before the first CUDA call.
std::string narrow_cuda_visible_devices(int& device, std::vector<int>& devices);

} // namespace sinfer::product
