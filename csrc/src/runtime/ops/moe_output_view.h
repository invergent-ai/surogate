#pragma once
#include <stdexcept>
#include "utilities/tensor.h"
namespace dsl {
// The graph owns output identity/lifetime. Parallel dense+MoE branches must
// never acquire an unrelated dense-MLP slot here; a serial graph may explicitly
// assign that same storage when its liveness plan permits it.
inline Tensor declared_moe_output_view(const Tensor& declared, int tokens, int hidden,
                                      ETensorDType dtype) {
    if (tokens <= 0 || hidden <= 0 || !declared.Data || declared.Rank < 2 ||
        declared.DType != dtype || declared.Sizes[declared.Rank - 1] != hidden ||
        declared.nelem() != static_cast<std::size_t>(tokens) * hidden)
        throw std::runtime_error("moe_unpermute: declared output storage/shape/dtype mismatch");
    Tensor out=declared;
    out.Rank=2;out.Sizes={tokens,hidden,1,1,1};return out;
}
}
