// DeepGEMM sm90 1D2D tiles, list 1 of 4 (fp8_block_sm90_deepgemm_tiles.h): a translation unit of
// their own so the tiles compile in parallel.

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm90::dg {

SINFER_DG_TILES_1(SINFER_DG_DEFINE)

} // namespace sinfer::ops::detail::fp8_block::sm90::dg
