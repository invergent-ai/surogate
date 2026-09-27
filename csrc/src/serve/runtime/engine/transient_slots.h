#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>

namespace sinfer::runtime {

// Host bookkeeping for a frozen device allocation. Regions never move while active.
class TransientSlots {
public:
    struct Region { std::size_t offset, bytes, alignment; };
    explicit TransientSlots(std::size_t capacity) : capacity_(capacity) {}

    std::optional<std::size_t> fit(std::uint32_t lane, std::size_t bytes,
                                   std::size_t alignment) const noexcept {
        if (!alignment || (alignment & (alignment - 1)) || alignment > 256 ||
            bytes > capacity_) { return {}; }
        std::size_t offset = 0;
        for (;;) {
            bool moved = false;
            for (const auto& [owner, r] : regions_) {
                if (owner == lane || offset >= r.offset + r.bytes || offset + bytes <= r.offset) { continue; }
                const auto end = r.offset + r.bytes;
                const auto padding = (alignment - (end & (alignment - 1))) & (alignment - 1);
                if (end > capacity_ || padding > capacity_ - end) { return {}; }
                offset = end + padding;
                if (offset > capacity_ || bytes > capacity_ - offset) { return {}; }
                moved = true;
                break;
            }
            if (!moved) { return offset; }
        }
    }

    void activate(std::uint32_t lane, std::size_t bytes, std::size_t alignment) {
        if (!bytes) {
            if (alignment != 1) { throw std::invalid_argument("empty transient alignment must be one"); }
            release(lane);
            return;
        }
        const auto offset = fit(lane, bytes, alignment);
        if (!offset) { throw std::invalid_argument("request transient exceeds available frozen capacity"); }
        regions_.insert_or_assign(lane, Region{*offset, bytes, alignment});
        peak_ = std::max(peak_, used());
    }
    void release(std::uint32_t lane) noexcept { regions_.erase(lane); }
    void clear() noexcept { regions_.clear(); }
    std::optional<Region> region(std::uint32_t lane) const noexcept {
        const auto it = regions_.find(lane);
        return it == regions_.end() ? std::nullopt : std::optional<Region>(it->second);
    }
    std::size_t used() const noexcept {
        std::size_t total = 0;
        for (const auto& [lane, r] : regions_) { total += r.bytes; }
        return total;
    }
    std::size_t peak() const noexcept { return peak_; }
    void reset_peak() noexcept { peak_ = used(); }
private:
    std::size_t capacity_, peak_ = 0;
    std::map<std::uint32_t, Region> regions_;
};

} // namespace sinfer::runtime
