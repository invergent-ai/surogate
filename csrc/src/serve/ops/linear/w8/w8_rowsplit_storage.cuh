#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail {

// A16 W8 GEMMs multiply the same BF16-rounded weights at every batch width.
// Keep the FP16 scale exact until after multiplication by the signed code.
__device__ __forceinline__ float w8_a16_weight(std::int8_t code, float scale) {
    return __bfloat162float(__float2bfloat16_rn(static_cast<float>(code) * scale));
}

struct W8RowSplitStorage {
    static constexpr int kGroupK             = 32;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 0;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct W8ScalarDecodeAtom {
    static constexpr int kGroupK = W8RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        if (lane >= kGroupK / 2) {
            w0 = 0.0f;
            w1 = 0.0f;
            return;
        }
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint8_t* packed = codes + group_index * W8RowSplitStorage::kCodeBytesPerGroup +
                                     static_cast<std::int64_t>(lane) * 2;
        w0 = w8_a16_weight(static_cast<std::int8_t>(packed[0]), scale);
        w1 = w8_a16_weight(static_cast<std::int8_t>(packed[1]), scale);
    }
};

} // namespace sinfer::ops::detail
