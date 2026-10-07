// The build's architecture set has no 90a, so it carries no wgmma W8 kernel: the batch-consistent
// route keeps its mma.sync kernels.

#include "ops/linear/w8/w8_rowsplit_wgmma_sm90.h"

namespace sinfer::ops::detail {

bool w8_wgmma_available() noexcept { return false; }

std::int32_t w8_wgmma_min_columns() noexcept { return 33; }

bool w8_wgmma_consistent(const W8WgmmaProblem&, cudaStream_t) { return false; }

} // namespace sinfer::ops::detail
