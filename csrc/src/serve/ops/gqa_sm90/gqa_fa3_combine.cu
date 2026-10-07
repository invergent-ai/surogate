// sinfer::ops - FlashAttention-3's split-KV combine (see gqa_fa3_launch.h): the instantiations
// upstream's run_mha_fwd_combine_ picks for a BF16 output and FP32 partials, varlen, sm90 only.
#include "ops/gqa_sm90/gqa_fa3_launch.h"

#include "flash_fwd_combine_launch_template.h"

#include <stdexcept>

namespace sinfer::ops::detail::gqa_fa3 {
namespace {

// Upstream's choice: the smallest row tile that keeps 256 threads busy reading kBlockK partial
// values, and the smallest power of two of splits that covers the launch (a row tile of 8 needs
// at least 32). Every head dim served here (64, 128, 256) is a whole number of kBlockK columns.
template <int kBlockK>
void combine_for(Flash_fwd_params& params, cudaStream_t stream, bool pdl) {
    constexpr int kBlockM = kBlockK % 128 == 0 ? 8 : 16;
    using T = cutlass::bfloat16_t;
    if constexpr (kBlockM >= 16) {
        if (params.num_splits <= 16) {
            return run_flash_fwd_combine<90, kBlockM, kBlockK, 4, true, true, T, float>(params, stream, pdl);
        }
    }
    if (params.num_splits <= 32) {
        return run_flash_fwd_combine<90, kBlockM, kBlockK, 5, true, true, T, float>(params, stream, pdl);
    }
    run_flash_fwd_combine<90, kBlockM, kBlockK, 6, true, true, T, float>(params, stream, pdl);
}

} // namespace

void combine(Flash_fwd_params& params, cudaStream_t stream, bool pdl) {
    if (params.num_splits < 2 || params.num_splits > 64) {
        throw std::invalid_argument("gqa_fa3: combine takes 2 to 64 splits");
    }
    if (params.dv <= 64) { return combine_for<64>(params, stream, pdl); }
    combine_for<128>(params, stream, pdl);
}

} // namespace sinfer::ops::detail::gqa_fa3
