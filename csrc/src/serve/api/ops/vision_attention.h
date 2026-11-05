#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * Returns the caller-owned transient capacity for every legal patch/segment pair in the supplied
 * inclusive envelope. A pair is legal when 1 <= segments <= patches. An envelope with no legal
 * pair throws; a legal uniform single-segment envelope returns zero.
 */
[[nodiscard]] std::size_t vision_attention_workspace_capacity_bytes(std::int32_t min_patches,
                                                                    std::int32_t max_patches,
                                                                    std::int32_t min_segments,
                                                                    std::int32_t max_segments);

/**
 * Packed, non-causal multi-head attention. For each segment [begin,end), head h, and
 * t in that segment:
 *
 *   score[j]    = dot(q[:,h,t], k[:,h,j]) / sqrt(D), begin <= j < end
 *   ideal[:,h,t] = sum_j softmax(score)[j] * v[:,h,j].
 *
 * q/k/v are BF16 [D,H,P], with D=64, D=72 or D=96 and H>0, with contiguous feature and head dimensions; token strides may be
 * padded. out is contiguous BF16 [D,H,P]. cu_seqlens is contiguous I32 [S+1], begins at 0,
 * ends at P, and is strictly increasing. The oracle evaluates `ideal` naively in FP64 from the
 * represented inputs. The BF16 out is promoted and compared directly with that result; output
 * storage rounding belongs to the Op's numerical criterion, not the oracle. Inputs and output are
 * mutually non-overlapping.
 *
 * The Op allocates required opaque tile descriptors from `workspace` for the duration of the call;
 * no capacity is consumed for a single segment. There is no persistent state side effect.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& cu_seqlens,
                      WorkspaceArena& workspace, Tensor& out, cudaStream_t stream, float scale = 0.0F);

/**
 * Equal-segment overload of the same packed, non-causal attention operation. P must be divisible
 * by `segment_length`; consecutive ranges `[s*segment_length,(s+1)*segment_length)` are the
 * independent segments. The q/k/v/out shape, layout, dtype, alias, and numerical semantics are
 * identical to the cu_seqlens overload. Segment tiles are derived directly and no workspace or
 * descriptor-setup launch is used.
 */
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      std::int32_t segment_length, Tensor& out, cudaStream_t stream, float scale = 0.0F);

/**
 * The same packed, non-causal attention with S = Q K^T and O += P V on the e4m3 tensor cores
 * (--vision-fp8): Q and K rounded to e4m3 with one scale per (token, head), V with one per
 * (head, channel), P scaled by 448 before its rounding (ops/kernel/vision_attention_fp8.cuh).
 * `scratch` holds the prepared operands, vision_attention_fp8_scratch_bytes(D, H, P) bytes,
 * 16-byte aligned, and must not overlap the inputs or out. Returns false, having enqueued
 * nothing, where the route does not run: a device other than compute capability 12.x, D other
 * than 64 or 72, misaligned operands, or too small a scratch. Throws as the BF16 overload does
 * on an invalid call.
 */
[[nodiscard]] bool vision_attention_fp8(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& cu_seqlens, WorkspaceArena& workspace,
                                        DeviceSpan scratch, Tensor& out, cudaStream_t stream,
                                        float scale = 0.0F);
/**
 * vision_attention_fp8 with the tower's rotary embedding folded in: q and k are the unrotated
 * projections, `positions` the [P, 2] I32 rotary positions ops::rope takes, and the rotation is
 * applied on the way into e4m3 -- the same rotation, rounded to BF16 as ops::rope rounds it, so
 * the result is bit for bit ops::rope followed by vision_attention_fp8. q and k are only read.
 * Returns false, having enqueued nothing (q and k unrotated), wherever vision_attention_fp8 would
 * or ops::rope would not take its fixed vision kernel (16 heads, two axes, rotary_dim == D, theta
 * 10000), where bit-for-bit agreement is not on offer.
 */
[[nodiscard]] bool vision_attention_fp8_rope(const Tensor& q, const Tensor& k, const Tensor& v,
                                             const Tensor& positions, std::int32_t rotary_dim,
                                             float theta, const Tensor& cu_seqlens,
                                             WorkspaceArena& workspace, DeviceSpan scratch,
                                             Tensor& out, cudaStream_t stream, float scale = 0.0F);
/// Scratch bytes vision_attention_fp8 needs for D x H x P; 0 for an unsupported D.
[[nodiscard]] std::size_t vision_attention_fp8_scratch_bytes(std::int32_t head_dim,
                                                             std::int32_t heads,
                                                             std::int32_t patches);
} // namespace sinfer::ops
