#pragma once

// NUMA placement for the host buffers of an offloaded model.
//
// On a multi-socket box the placement of pinned host memory decides how far the host expert
// path is from the cores and from the device: the expert bank of a host-offloaded MoE is read
// by every core, while the staging buffers the GPU DMAs through belong on the node its root
// complex hangs off. This box has two nodes (8x RTX 5090; GPUs 0-3 on node 0, 4-7 on node 1).
//
// The policy belongs in the engine rather than in a wrapper script, because `numactl
// --interleave=all` in front of the binary is easy to forget and leaves no symptom other than
// a slower run. (It is *not* what produced the 23-vs-46 GB/s gather difference seen on
// 2026-08-30: that was PCIe link width — GPUs 0/1/4/6 are x16 and 2/3/5/7 are x8 on this
// host, and the offloaded model wants an x16 slot. The two effects are independent.)
//
// What the helpers do:
//   * `numa_topology()` — node count and the node each CUDA device sits on (from the device's
//     PCI address through sysfs). Single-node or libnuma-less hosts report one node and every
//     helper becomes a no-op.
//   * `ScopedMemoryPolicy::interleaved()` — set the calling thread's allocation policy to
//     interleave over every node, restored on scope exit. Wrap the allocation of buffers that
//     all cores read (the expert bank).
//   * `ScopedMemoryPolicy::bound_to(node)` — bind allocations to one node, restored on scope
//     exit. Wrap the allocation of buffers a single device DMAs through.
//
// The policy must be set *before* the allocation: it applies at page-fault time, and pinned
// pages cannot be migrated afterwards, so `mbind()` on an already-registered region is not a
// substitute.
//
// `SUROGATE_SERVE_NUMA` overrides the default: `auto` (the behaviour above), `interleave`
// (interleave everything, the old `numactl --interleave=all`), `local` (bind everything to the
// device's node) or `off` (no policy, first touch).

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer {

enum class NumaPolicy { Auto, Interleave, Local, Off };

struct NumaTopology {
    int nodes = 1;                    ///< configured nodes; 1 when there is nothing to place
    bool available = false;           ///< libnuma present and the host has more than one node
    std::vector<int> device_node;     ///< node per CUDA device, -1 when unknown
    [[nodiscard]] int node_of_device(int device) const noexcept {
        return device >= 0 && device < static_cast<int>(device_node.size()) ? device_node[static_cast<std::size_t>(device)]
                                                                           : -1;
    }
};

/// Detected once and cached. Safe to call before or after CUDA initialisation.
const NumaTopology& numa_topology();

/// The policy the engine should apply, from `SUROGATE_SERVE_NUMA` (default `auto`).
NumaPolicy numa_policy();

/// One line describing what will be done, for the startup log; empty when there is nothing to say.
std::string numa_policy_description();

/// Sets the calling thread's memory policy for the lifetime of the object and restores it after.
/// Every constructor is a no-op when the host has one node, libnuma is missing, or the policy is
/// `off`; failures are reported through `applied()` and never throw.
class ScopedMemoryPolicy {
public:
    /// Interleave over all nodes — for buffers every core reads (the expert bank).
    static ScopedMemoryPolicy interleaved();
    /// Bind to `node` — for buffers one device DMAs through. `node < 0` is a no-op.
    static ScopedMemoryPolicy bound_to(int node);
    /// Bind to the node of `device`, or interleave when that node is unknown.
    static ScopedMemoryPolicy for_device(int device);

    ScopedMemoryPolicy() = default;
    ~ScopedMemoryPolicy();
    ScopedMemoryPolicy(ScopedMemoryPolicy&& other) noexcept;
    ScopedMemoryPolicy& operator=(ScopedMemoryPolicy&&) = delete;
    ScopedMemoryPolicy(const ScopedMemoryPolicy&) = delete;
    ScopedMemoryPolicy& operator=(const ScopedMemoryPolicy&) = delete;

    /// True when a policy was actually installed (and will be restored).
    [[nodiscard]] bool applied() const noexcept { return applied_; }
    /// Short description of what was installed, for logs ("interleaved", "node 1", "").
    [[nodiscard]] const std::string& what() const noexcept { return what_; }

private:
    bool applied_ = false;
    std::string what_;
};

} // namespace ninfer
