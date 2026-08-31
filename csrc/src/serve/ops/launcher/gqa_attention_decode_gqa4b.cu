// One head geometry's small-T decode instantiations. Split out so the five
// geometries compile in parallel rather than serially in one translation unit.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace ninfer::ops::detail {

template void gqa_attention_small_t_launch_for<Gqa4BGeometry, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

template void gqa_attention_small_t_launch_for<Gqa4BGeometry, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace ninfer::ops::detail
