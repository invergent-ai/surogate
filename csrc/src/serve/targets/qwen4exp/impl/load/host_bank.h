#pragma once

// Pinned, device-mapped host memory for the objects that never become device resident: the
// routed expert banks of every layer and the n-gram embedding table. Kernels read them
// zero-copy over PCIe through the mapped device pointer (phase 1); later phases add a device
// slot cache and CPU expert compute in front of the same bank.

#include "artifact/binder.h"
#include "artifact/reader.h"

#include <cstddef>
#include <memory>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen4exp::detail {

struct HostObjectPlan {
    artifact::ObjectHandle handle;
    std::span<const std::byte> payload; ///< the artifact mapping; valid while the reader lives
    std::string name;
};

struct HostBankPlan {
    std::vector<HostObjectPlan> objects;
    [[nodiscard]] std::size_t total_bytes() const noexcept;
};

/// One pinned allocation per host object; `device_pointer` is the mapped alias.
struct HostObject {
    void* host             = nullptr;
    const void* device     = nullptr;
    std::size_t bytes      = 0;
    std::string name;
};

class HostBank {
public:
    HostBank() = default;
    explicit HostBank(const HostBankPlan& plan);
    ~HostBank();

    HostBank(const HostBank&)            = delete;
    HostBank& operator=(const HostBank&) = delete;
    HostBank(HostBank&&)                 = delete;
    HostBank& operator=(HostBank&&)      = delete;

    [[nodiscard]] const HostObject& object(artifact::ObjectHandle handle) const;
    [[nodiscard]] std::size_t total_bytes() const noexcept { return total_bytes_; }

    /// The process-wide bank for this plan: pipeline stages of one model in one process share
    /// the pinned experts instead of pinning them once per device (keyed by the objects' names
    /// and sizes, so the same artifact loaded for another stage reuses the live bank).
    [[nodiscard]] static std::shared_ptr<HostBank> shared(const HostBankPlan& plan);

private:
    std::vector<std::pair<std::size_t, HostObject>> objects_;
    std::size_t total_bytes_ = 0;
};

} // namespace ninfer::targets::qwen4exp::detail
