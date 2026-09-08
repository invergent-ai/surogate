#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sinfer {

/// Immutable contiguous BF16 storage borrowed from an in-process trainer.
/// The caller must retain the owner until the engine is destroyed.
struct BorrowedTensor {
    std::string name;
    void* data = nullptr;
    std::vector<std::uint64_t> shape;
    std::uint64_t bytes = 0;
    int device = -1;
};

struct DeviceAdapterModule {
    std::int32_t layer = 0;
    std::string module;
    const void* a = nullptr;
    const void* b = nullptr;
    std::int32_t rank = 0;
    std::int32_t in = 0;
    std::int32_t out = 0;
    float scale = 1.0F;
};

} // namespace sinfer
