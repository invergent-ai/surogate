#pragma once
// TensorRT-LLM's cutlass fused MoE runner (vendored under csrc/src/third_party/trtllm_moe, the
// kernel FlashInfer builds for vLLM's FLASHINFER_CUTLASS backend) behind a plain-pointer contract
// for the routed-NVFP4 experts. The runner quantises the BF16 rows to e2m1 itself (W4A4), runs
// the two grouped block-scaled GEMMs with the SwiGLU fused between them, and reduces the top-k
// expert outputs with the router's weights. Everything else in the round — the router, the
// shared expert, the residual combine — stays ours.

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail::trtllm_moe {

struct Geometry {
    std::int32_t hidden            = 0;
    std::int32_t experts           = 0;
    std::int32_t experts_per_token = 0;
    std::int32_t intermediate      = 0;
};

/// One layer's routed experts as the runner reads them. Every pointer is device memory.
struct Nvfp4RoutedExperts {
    /// fc1: [experts][2 * intermediate][hidden / 2] e2m1 code pairs, the rows of an expert
    /// ordered **[up; gate]** (the runner reads `linear` from the first half and `gate` from the
    /// second; the groupwise-int profile's [gate; up] is the opposite).
    const void* gate_up_codes = nullptr;
    /// e4m3 block scales, one per 16 values, in the BlockScaleK16M128x4 swizzle over the stacked
    /// rows (an expert is a whole number of 128-row tiles, so stacking commutes with the swizzle).
    const void* gate_up_block_scales = nullptr;
    /// [experts]: the checkpoint's `input_global_scale` (6 * 448 / amax of the calibrated input).
    /// The runner multiplies x by it before the e2m1 rounding, so it is the *large* multiplier.
    const float* gate_up_act_scale = nullptr;
    /// [experts]: 1 / (gate_up_act_scale[e] * weight_global_scale[e]), the fc1 epilogue alpha
    /// that undoes both global scales after the block-scaled MMA.
    const float* gate_up_alpha = nullptr;
    /// fc2: [experts][hidden][intermediate / 2] with the same conventions.
    const void* down_codes        = nullptr;
    const void* down_block_scales = nullptr;
    const float* down_act_scale   = nullptr;
    const float* down_alpha       = nullptr;
};

/// True when the runner is compiled in (sm_120a builds). The stub returns false and every other
/// entry point throws std::runtime_error naming the missing library.
[[nodiscard]] bool available() noexcept;

/// The width bucket a round of `tokens` rows is tuned and run as: FlashInfer's hybrid ladder —
/// powers of two up to 256, steps of 256 up to 2,048, steps of 512 up to 4,096 (the widest
/// slice the prefill family runs). `tokens` above 4,096 throws.
[[nodiscard]] std::int32_t bucket_of(std::int32_t tokens);

/// Scratch `run` needs for rounds of up to `max_tokens` rows, 256-byte aligned. Covers the
/// runner's own workspace, the BF16 staging row block and the permutation map.
[[nodiscard]] std::size_t workspace_bytes(const Geometry& geometry, std::int32_t max_tokens);

/// Chooses the two grouped GEMMs' tactics for every bucket up to `max_tokens` by timing the
/// candidates on `sample` (uniform random routing), and persists the choice under
/// $XDG_CACHE_HOME/surogate or ~/.cache/surogate so later loads skip the timing. Must run outside
/// any stream capture; a no-op for buckets already tuned in this process.
void prepare(const Geometry& geometry, const Nvfp4RoutedExperts& sample, std::int32_t max_tokens,
             cudaStream_t stream);

/// routed_sum[t][:] = sum_k final_scales[t][k] * expert_{ids[t][k]}(x[t]), written as FP32
/// row-major [tokens][hidden]. `x` is BF16 [tokens][hidden]; `ids` int32 and `final_scales`
/// FP32, both [tokens][experts_per_token]. `workspace` holds at least
/// workspace_bytes(geometry, tokens). A bucket that was never prepared is tuned here when the
/// stream is not capturing; when it is capturing this throws std::logic_error, so a captured
/// graph never freezes an untuned tactic.
void run(const Geometry& geometry, const __nv_bfloat16* x, std::int32_t tokens,
         const std::int32_t* ids, const float* final_scales, const Nvfp4RoutedExperts& experts,
         void* workspace, std::size_t workspace_capacity, float* routed_sum, cudaStream_t stream);

} // namespace ninfer::ops::detail::trtllm_moe
