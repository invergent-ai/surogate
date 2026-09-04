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

namespace sinfer::targets::qwen4exp::detail {

struct HostObjectPlan {
    artifact::ObjectHandle handle;
    /// The artifact mapping, valid while the reader lives. An object read in place from a GGUF
    /// is assembled from several runs -- a fused expert parent is one per source tensor per row
    /// block -- so the bank concatenates `parts` into pinned memory. `payload` is the single
    /// span an object stored in the artifact itself has, and is empty when `parts` is used.
    std::span<const std::byte> payload;
    std::string name;
    std::vector<std::span<const std::byte>> parts;
    // Non-zero: requantise this W8 row-split object to Q4G32AM while it is copied into pinned
    // memory (`q4_rows x q4_k` weights; `q4_w8_scale_offset` locates the source scales plane).
    std::int64_t q4_rows           = 0;
    std::int32_t q4_k              = 0;
    std::size_t q4_w8_scale_offset = 0;
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
    /// True when `host` is an mmap'd region pinned with cudaHostRegister (the fast path:
    /// threaded first-touch then register, ~10 GB/s against cudaHostAlloc's 1.8); false when it
    /// came from the cudaHostAlloc fallback. Decides the release path.
    bool registered        = false;
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

} // namespace sinfer::targets::qwen4exp::detail
