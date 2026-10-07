#pragma once
// sinfer::ops - the definition of gqa_fa3::launch (gqa_fa3_launch.h), included only by the
// translation units that instantiate it.

#include "ops/gqa_sm90/gqa_fa3_launch.h"

#include "flash_fwd_launch_template.h"

#include <type_traits>

namespace sinfer::ops::detail::gqa_fa3 {

template <int HeadDim, bool Fp8, bool Local, bool Split, bool OneMmaWg>
void launch(Flash_fwd_params& params, cudaStream_t stream) {
    using Element = std::conditional_t<Fp8, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
    run_flash_fwd</*Arch=*/90, HeadDim, HeadDim, /*ClusterM=*/1, Element, cutlass::bfloat16_t,
                  /*Is_causal=*/!Local, /*Is_local=*/Local, /*Has_softcap=*/false, /*Varlen=*/true,
                  /*PagedKVNonTMA=*/true, /*AppendKV=*/false, /*HasQv=*/false,
                  /*PackGQA=*/true, Split, /*V_colmajor=*/false, OneMmaWg>(params, stream);
}

} // namespace sinfer::ops::detail::gqa_fa3
