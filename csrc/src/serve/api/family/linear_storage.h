#pragma once

#include <cstdint>
#include <vector>

namespace sinfer {
enum class QType : std::uint16_t;
}

namespace sinfer::family {
/// Matrix dimensions and stored formats, without pointers to loaded device memory.
/// A segmented parent lists every format its row views can dispatch.
struct LinearStorage {
    std::int32_t rows = 0;
    std::int32_t columns = 0;
    std::vector<QType> formats;

    bool operator==(const LinearStorage&) const = default;
};
} // namespace sinfer::family
