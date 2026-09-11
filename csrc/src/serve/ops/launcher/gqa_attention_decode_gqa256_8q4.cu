// One translation unit per geometry keeps the kernel instantiations parallel.
#include "ops/launcher/gqa_attention_decode_launch.cuh"

namespace sinfer::ops::detail {

template void gqa_attention_small_t_launch_for<Gqa256_8q4, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

template void gqa_attention_small_t_launch_for<Gqa256_8q4, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace sinfer::ops::detail
