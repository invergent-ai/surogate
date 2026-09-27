#pragma once

#include "api/types.h"
#include "runtime/contract/transient_region.h"

#include <cstddef>
#include <memory>

namespace sinfer {

struct DeviceContext;

namespace runtime {

// Owns the startup-frozen request transient pool. Each active lane owns a disjoint region; no
// request-time device allocation or replacement is permitted.
class RequestMemory {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    RequestMemory(DeviceContext& device, std::size_t frozen_capacity_bytes);
    ~RequestMemory();

    RequestMemory(const RequestMemory&)            = delete;
    RequestMemory& operator=(const RequestMemory&) = delete;
    RequestMemory(RequestMemory&&)                 = delete;
    RequestMemory& operator=(RequestMemory&&)      = delete;

    void activate(std::size_t bytes, std::size_t alignment);
    void deactivate() noexcept;
    bool can_activate_lane(std::uint32_t lane, std::size_t bytes, std::size_t alignment) const noexcept;
    void activate_lane(std::uint32_t lane, std::size_t bytes, std::size_t alignment);
    void deactivate_lane(std::uint32_t lane) noexcept;

    [[nodiscard]] TransientRegion region(std::uint32_t lane = 0) const noexcept;
    [[nodiscard]] ArenaMemorySummary summary() const noexcept;
    void reset_peak() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace sinfer
