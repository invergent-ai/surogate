#pragma once

// MoE Marlin for the routed gate/up projection.
//
// The routed experts are the 35B's whole cost - the profile puts the Q4 gate/up
// kernel at 53 % of a MoE layer and the Q5 down at 31 % - and that Q4 kernel is
// issue-bound, not bandwidth-bound: sm__throughput 60 %, dram 23 %, occupancy
// capped at 3 blocks/SM by shared memory. vLLM's MoE Marlin is the tuned kernel
// for exactly this shape, and our Q4G64 residency is offset-8 4-bit with per-64
// scales, which is bit-for-bit what Marlin calls kU4B8 with group 64. So the
// weights repack rather than requantise: no quality is traded for this.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

// Rows of the padded token table Marlin indexes, given the assignment count.
[[nodiscard]] std::int32_t marlin_moe_padded_rows(std::int32_t assignments,
                                                  std::int32_t num_experts,
                                                  std::int32_t block_size) noexcept;

// Bytes of B tiles and scales one expert's [n, k] Q4G64 matrix repacks into.
[[nodiscard]] std::size_t marlin_moe_b_bytes(std::int32_t n, std::int32_t k) noexcept;
[[nodiscard]] std::size_t marlin_moe_scale_bytes(std::int32_t n, std::int32_t k) noexcept;

// Repack `experts` matrices of [n, k] Q4G64_F16S residency (codes row-major, 32 bytes per
// 64-wide group; scales FP16 per group) into Marlin B tiles and BF16 scales. `gptq_tmp`
// holds (k/8)*n uint32 for one expert and is reused across them.
void marlin_moe_repack_q4g64(const void* codes, const void* scales_f16, std::int32_t experts,
                             std::int32_t n, std::int32_t k, void* gptq_tmp, void* b_out,
                             void* scales_out, cudaStream_t stream);

// The routing arrays Marlin indexes A and C with, built from our per-expert offsets:
// sorted_token_ids [padded_rows] holds the gathered-row index for each padded slot and
// `assignments` for padding, expert_ids [padded_rows / block] the expert owning each block,
// and num_tokens_past_padded [1] the padded row count.
void marlin_moe_build_routing(const std::int32_t* expert_offsets, std::int32_t num_experts,
                              std::int32_t assignments, std::int32_t block_size,
                              std::int32_t* sorted_token_ids, std::int32_t* expert_ids,
                              std::int32_t* num_tokens_past_padded, cudaStream_t stream);

// C[row, n] = sum_k A[row, k] * dequant(B[expert(row), n, k]) over the gathered rows.
// A is BF16 [assignments, k] already gathered per expert; C is BF16 [assignments, n].
void marlin_moe_gemm_q4g64_bf16(const void* a, const void* b_tiles, const void* scales,
                                void* c, void* c_tmp, const std::int32_t* sorted_token_ids,
                                const std::int32_t* expert_ids,
                                const std::int32_t* num_tokens_past_padded,
                                std::int32_t assignments, std::int32_t n, std::int32_t k,
                                std::int32_t num_experts, std::int32_t block_size,
                                std::int32_t* locks, cudaStream_t stream);

// silu(gate) * up over Marlin's raw [rows, 2*intermediate] product, into [rows, intermediate].
void marlin_moe_silu_mul(const void* product, void* out, std::int32_t rows,
                         std::int32_t intermediate, cudaStream_t stream);

// Workspace the GEMM needs beside its operands: fp32 reduce buffer and lock words.
[[nodiscard]] std::size_t marlin_moe_c_tmp_bytes(std::int32_t assignments,
                                                 std::int32_t n) noexcept;
[[nodiscard]] std::size_t marlin_moe_lock_bytes() noexcept;

// Whether the route is enabled (SUROGATE_SERVE_MOE_MARLIN, default off while it proves out).
[[nodiscard]] bool marlin_moe_route_enabled();

} // namespace sinfer::ops::detail
