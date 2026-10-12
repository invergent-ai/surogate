#pragma once

#include <cuda_runtime.h>

namespace sinfer::ops::detail::ggml {

/// The (row tile, column tile) a block of a (row tiles, column tiles) grid computes.
///
/// Blocks are issued in x-fastest order, so mapping blockIdx.x to the row tile sweeps every row
/// tile -- the whole weight -- once per column tile: a prompt chunk of 4,096 columns in 64-wide
/// tiles reads a weight sixty-four times, and one larger than L2 (a GDN input projection, ~21 MB
/// in Q6_K) comes back from DRAM each time. Bands of `Group` row tiles instead run across every
/// column tile before the next band starts, so the band's weight rows stay in L2 while its
/// columns pass, and the activations -- re-read once per band -- are the smaller operand. With
/// one column tile (a decode round) this is the plain mapping. Each tile is still one block's,
/// with the same arithmetic, so the results do not change.
template <int Group>
__device__ __forceinline__ int2 grouped_tile() {
    const int row_tiles = static_cast<int>(gridDim.x);
    const int col_tiles = static_cast<int>(gridDim.y);
    const int linear    = static_cast<int>(blockIdx.y) * row_tiles + static_cast<int>(blockIdx.x);
    const int band      = linear / (Group * col_tiles);
    const int first     = band * Group;
    const int height    = min(Group, row_tiles - first);
    const int within    = linear - band * Group * col_tiles;
    return make_int2(first + within % height, within / height);
}

} // namespace sinfer::ops::detail::ggml
