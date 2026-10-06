// sinfer::ops - FlashAttention-3's FP8 forward over the paged e4m3 KV cache (see gqa_fa3.h).
// The BF16 instantiation's twin: e4m3 queries, keys and values, BF16 out. Kept in its own
// translation unit so the two instantiations compile in parallel.
#include "ops/gqa_sm90/gqa_fa3.h"

#include "flash_fwd_launch_template.h"

namespace sinfer::ops::detail::gqa_fa3 {

void launch_e4m3(Flash_fwd_params& params, cudaStream_t stream) {
    run_flash_fwd</*Arch=*/90, kHeadDim, kHeadDim, 1, cutlass::float_e4m3_t, cutlass::bfloat16_t,
                  /*Is_causal=*/true, /*Is_local=*/false, /*Has_softcap=*/false, /*Varlen=*/true,
                  /*PagedKVNonTMA=*/true, /*AppendKV=*/false, /*HasQv=*/false,
                  /*PackGQA=*/true, /*Split=*/false, /*V_colmajor=*/false>(params, stream);
}

} // namespace sinfer::ops::detail::gqa_fa3
