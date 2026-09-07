#include "ops/linear/ggml/ggml_repack.h"

#include "core/device.h"
#include "ops/linear/ggml/ggml_blocks.h"

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// The row-split geometry for an 8-bit format at group 32, spelled here rather than pulled from
// the artifact layer: 32 code bytes per group in the first plane, two scale bytes per group in
// the second, the second starting at the next 256-byte boundary. `k` is a multiple of 128 for
// every weight this runs on, so there are no padding groups to fill.
constexpr std::size_t kPlaneAlignment = 256;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// One warp per block: 32 lanes, one code byte each, and lane zero carries the scale across.
__global__ void q8_0_to_w8_rowsplit_kernel(const block_q8_0* __restrict__ blocks,
                                           std::uint8_t* __restrict__ codes,
                                           std::uint8_t* __restrict__ scales,
                                           const std::int32_t* __restrict__ group_map,
                                           const std::int32_t groups_per_row,
                                           const std::int64_t groups) {
    const std::int64_t group =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (group >= groups) { return; }
    // The map is per row and the same for every row, so the source block is the destination's
    // row with the map applied to its position within that row.
    const std::int64_t source_group =
        group_map == nullptr
            ? group
            : (group / groups_per_row) * groups_per_row + group_map[group % groups_per_row];
    const block_q8_0& source = blocks[source_group];
    const int lane           = static_cast<int>(threadIdx.x);
    codes[group * QK8_0 + lane] = static_cast<std::uint8_t>(source.qs[lane]);
    if (lane == 0) {
        *reinterpret_cast<__half*>(scales + group * 2) = source.d;
    }
}

} // namespace

std::size_t q8_0_source_bytes(std::int32_t rows, std::int32_t k) {
    if (rows <= 0 || k <= 0 || (k % QK8_0) != 0) {
        throw std::invalid_argument("q8_0 repack: k must be a multiple of 32");
    }
    return static_cast<std::size_t>(rows) * (static_cast<std::size_t>(k) / QK8_0) *
           sizeof(block_q8_0);
}

void q8_0_to_w8_rowsplit_launch(const void* blocks, void* out, std::int32_t rows, std::int32_t k,
                                std::size_t out_bytes, const std::int32_t* group_map,
                                cudaStream_t stream) {
    if (blocks == nullptr || out == nullptr || rows <= 0 || k <= 0 || (k % QK8_0) != 0) {
        throw std::invalid_argument("q8_0 repack: [rows, k] with k a multiple of 32");
    }
    const std::size_t groups_per_row = static_cast<std::size_t>(k) / QK8_0;
    const std::size_t groups         = static_cast<std::size_t>(rows) * groups_per_row;
    const std::size_t code_bytes     = groups * QK8_0;
    const std::size_t scale_offset   = align_up(code_bytes, kPlaneAlignment);
    if (out_bytes < scale_offset + groups * 2) {
        throw std::invalid_argument("q8_0 repack: destination is too small for the planes");
    }
    // The gap the plane alignment leaves is never read, but zeroing it keeps a materialised
    // tensor byte-identical to one the converter wrote, which is what the oracle compares.
    CUDA_CHECK(cudaMemsetAsync(out, 0, out_bytes, stream));
    auto* bytes = static_cast<std::uint8_t*>(out);
    const dim3 block(QK8_0, 8);
    const dim3 grid(static_cast<unsigned>((groups + 7) / 8));
    q8_0_to_w8_rowsplit_kernel<<<grid, block, 0, stream>>>(
        static_cast<const block_q8_0*>(blocks), bytes, bytes + scale_offset, group_map,
        static_cast<std::int32_t>(groups_per_row), static_cast<std::int64_t>(groups));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail::ggml
