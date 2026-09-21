#pragma once
#include "utilities/dtype.h"
namespace dsl {
inline bool disable_frozen_moe_qkv_recipe(bool qkv, bool frozen, int experts,
                                       ETensorDType weight_dtype, bool fp8_hybrid_backward) {
    const bool streamed_fp8 = weight_dtype == ETensorDType::FP8_E4M3 && fp8_hybrid_backward;
    return qkv && frozen && experts > 0 && !streamed_fp8;
}
}
