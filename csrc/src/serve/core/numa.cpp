#include "core/numa.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#if defined(__linux__) && __has_include(<numa.h>) && __has_include(<numaif.h>)
#include <numa.h>
#include <numaif.h>
#define NINFER_HAVE_NUMA 1
#else
#define NINFER_HAVE_NUMA 0
#endif

namespace ninfer {
namespace {

#if NINFER_HAVE_NUMA
/// The node a CUDA device hangs off, read from sysfs via its PCI address. -1 when unknown
/// (containers often hide the file, and some firmware reports -1 for every device).
int device_node_from_sysfs(int device) {
    char bus_id[32] = {};
    if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), device) != cudaSuccess) {
        return -1;
    }
    // CUDA reports "00000000:01:00.0"; sysfs uses the 4-digit domain "0000:01:00.0".
    std::string address(bus_id);
    for (char& c : address) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
    const std::size_t colon = address.find(':');
    if (colon != std::string::npos && colon > 4) { address.erase(0, colon - 4); }
    std::ifstream file("/sys/bus/pci/devices/" + address + "/numa_node");
    if (!file) { return -1; }
    int node = -1;
    file >> node;
    return node;
}

struct nodemask_deleter {
    void operator()(struct bitmask* mask) const noexcept {
        if (mask != nullptr) { numa_free_nodemask(mask); }
    }
};
#endif

NumaTopology detect() {
    NumaTopology topology;
#if NINFER_HAVE_NUMA
    if (numa_available() < 0) { return topology; }
    topology.nodes = numa_num_configured_nodes();
    if (topology.nodes <= 1) { return topology; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess) { devices = 0; }
    topology.device_node.assign(static_cast<std::size_t>(std::max(devices, 0)), -1);
    for (int device = 0; device < devices; ++device) {
        topology.device_node[static_cast<std::size_t>(device)] = device_node_from_sysfs(device);
    }
    topology.available = true;
#endif
    return topology;
}

NumaPolicy parse_policy() {
    const char* raw = std::getenv("SUROGATE_SERVE_NUMA");
    if (raw == nullptr || *raw == '\0') { return NumaPolicy::Auto; }
    const std::string value(raw);
    if (value == "auto") { return NumaPolicy::Auto; }
    if (value == "interleave") { return NumaPolicy::Interleave; }
    if (value == "local") { return NumaPolicy::Local; }
    if (value == "off" || value == "none" || value == "0") { return NumaPolicy::Off; }
    std::fprintf(stderr,
                 "SUROGATE_SERVE_NUMA: unknown value '%s' (auto|interleave|local|off); using auto\n",
                 raw);
    return NumaPolicy::Auto;
}

#if NINFER_HAVE_NUMA
/// Installs an interleave policy over every node. Returns true when the kernel accepted it.
bool set_interleave_all() {
    struct bitmask* mask = numa_allocate_nodemask();
    if (mask == nullptr) { return false; }
    copy_bitmask_to_bitmask(numa_all_nodes_ptr, mask);
    const bool ok = set_mempolicy(MPOL_INTERLEAVE, mask->maskp, mask->size + 1) == 0;
    numa_free_nodemask(mask);
    return ok;
}

bool set_bind_node(int node) {
    struct bitmask* mask = numa_allocate_nodemask();
    if (mask == nullptr) { return false; }
    numa_bitmask_clearall(mask);
    numa_bitmask_setbit(mask, static_cast<unsigned>(node));
    const bool ok = set_mempolicy(MPOL_BIND, mask->maskp, mask->size + 1) == 0;
    numa_free_nodemask(mask);
    return ok;
}

void clear_policy() { (void)set_mempolicy(MPOL_DEFAULT, nullptr, 0); }
#endif

} // namespace

const NumaTopology& numa_topology() {
    static const NumaTopology topology = detect();
    return topology;
}

NumaPolicy numa_policy() {
    static const NumaPolicy policy = parse_policy();
    return policy;
}

std::string numa_policy_description() {
    const NumaTopology& topology = numa_topology();
    if (!topology.available) {
#if NINFER_HAVE_NUMA
        return {};
#else
        return "NUMA placement unavailable (built without libnuma); run under "
               "`numactl --interleave=all` on a multi-socket host";
#endif
    }
    switch (numa_policy()) {
    case NumaPolicy::Off:
        return "NUMA placement off (SUROGATE_SERVE_NUMA=off): host buffers land where they are "
               "first touched";
    case NumaPolicy::Interleave:
        return "NUMA placement: every host buffer interleaved over " + std::to_string(topology.nodes) +
               " nodes";
    case NumaPolicy::Local:
        return "NUMA placement: every host buffer bound to the node of the device that uses it";
    case NumaPolicy::Auto:
        break;
    }
    return "NUMA placement: shared banks interleaved over " + std::to_string(topology.nodes) +
           " nodes, per-device staging bound to the device's node";
}

ScopedMemoryPolicy ScopedMemoryPolicy::interleaved() {
    ScopedMemoryPolicy scope;
#if NINFER_HAVE_NUMA
    const NumaTopology& topology = numa_topology();
    if (!topology.available || numa_policy() == NumaPolicy::Off) { return scope; }
    if (set_interleave_all()) {
        scope.applied_ = true;
        scope.what_    = "interleaved";
    }
#endif
    return scope;
}

ScopedMemoryPolicy ScopedMemoryPolicy::bound_to(int node) {
    ScopedMemoryPolicy scope;
#if NINFER_HAVE_NUMA
    const NumaTopology& topology = numa_topology();
    if (!topology.available || numa_policy() == NumaPolicy::Off || node < 0 ||
        node >= topology.nodes) {
        return scope;
    }
    if (numa_policy() == NumaPolicy::Interleave) { return ScopedMemoryPolicy::interleaved(); }
    if (set_bind_node(node)) {
        scope.applied_ = true;
        scope.what_    = "node " + std::to_string(node);
    }
#endif
    return scope;
}

ScopedMemoryPolicy ScopedMemoryPolicy::for_device(int device) {
    const NumaTopology& topology = numa_topology();
    const int node = topology.node_of_device(device);
    // Unknown node: interleaving is the safe half-local placement, and it is what the wrapper
    // script did for every buffer before this existed.
    return node >= 0 ? ScopedMemoryPolicy::bound_to(node) : ScopedMemoryPolicy::interleaved();
}

ScopedMemoryPolicy::ScopedMemoryPolicy(ScopedMemoryPolicy&& other) noexcept
    : applied_(other.applied_), what_(std::move(other.what_)) {
    other.applied_ = false;
}

ScopedMemoryPolicy::~ScopedMemoryPolicy() {
#if NINFER_HAVE_NUMA
    if (applied_) { clear_policy(); }
#endif
}

} // namespace ninfer
