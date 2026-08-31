#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

/// The fused [query | key | gate | value] row layout of one attention projection. The 27B
/// (24 x 256 query heads, 4 x 256 key) and the 4B (16 x 256, 4 x 256) store the halves in this
/// order; only the sizes differ.
template <std::int32_t kQueryRows, std::int32_t kKeyRows>
struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        constexpr std::int32_t kGateRows   = kQueryRows;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result_bf16;
        } else {
            value[parent_row - kValueBegin] = result_bf16;
        }
    }
};

} // namespace

template <class Geometry, std::int32_t kQueryRows, std::int32_t kKeyRows>
void launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
            cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    static_assert((kQueryRows % 128) == 0);
    static_assert((kKeyRows % 128) == 0);
    static_assert(2 * kQueryRows + 2 * kKeyRows == Geometry::kOutputRows);

    const Nvfp4AttentionInputOutput<kQueryRows, kKeyRows> output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (resolve_nvfp4_gemv_only_problem(weight.n, weight.k) ==
        Nvfp4GemvOnlyProblem::AttnInput2560) {
        launch<Nvfp4AttnInput2560Geometry, 4096, 1024>(x, weight, q, gate, k, v, stream);
        return;
    }
    launch<Nvfp4AttnInputGeometry, 6144, 1024>(x, weight, q, gate, k, v, stream);
}

} // namespace sinfer::ops::detail
