#pragma once
#include "runtime/dsl/tensor_slot.h"
namespace dsl {
// A fused add+norm may be any graph normalization, including parallel branch
// joins. Only a declared block-input LN1 can replay from ResidualManager.
inline bool is_block_input_norm(TensorSlot normalized, TensorSlot rstd) {
    return normalized == TensorSlot::BlockLN1 && rstd == TensorSlot::BlockLN1RSTD;
}
inline bool is_block_mlp_norm(TensorSlot normalized, TensorSlot rstd) {
    return normalized == TensorSlot::BlockLN2 && rstd == TensorSlot::BlockLN2RSTD;
}
}
