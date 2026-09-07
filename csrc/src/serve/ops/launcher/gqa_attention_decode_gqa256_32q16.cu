// One head geometry's small-T decode instantiations. Split out so the registered
// geometries compile in parallel rather than serially in one translation unit.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace sinfer::ops::detail {

template void gqa_attention_small_t_launch_for<Gqa256_32q16, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

template void gqa_attention_small_t_launch_for<Gqa256_32q16, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace sinfer::ops::detail
