// Load-time repack of the routed Q4G64 expert residency into Marlin B tiles.
//
// Our Q4G64_F16S codes are 32 bytes per 64-wide K group, low nibble first, each
// value offset by 8 - which is bit-for-bit GPTQ's kU4B8 packing, so the code
// path here is a regroup rather than a requantisation. The tiling and scale
// permutation come from the same vendored vLLM repack kernel the linear Marlin
// path uses.

#include "ops/sparse_moe/marlin/marlin_moe_gemm.h"

#include "core/device.h"

#include "ops/linear/marlin/marlin_repack.h"
#include "ops/linear/marlin/vendor/marlin.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

// Our codes: row-major [n, k/2], byte b of a row holds values 2b (low nibble) and 2b+1
// (high nibble). GPTQ wants [k/8, n] uint32 with 8 consecutive k-values per word, value i
// in bits [4i, 4i+4), so one uint32 is exactly four of our bytes.
//
// The one re-encode is the bias. Our nibble is a sign-extended 4-bit value - the kernels read
// it as `(x ^ 8) - 8`, so 0..7 are 0..7 and 8..15 are -8..-1 - while kU4B8 means `x - 8`.
// Those differ by a rotation of half the range, and for 4 bits `(x + 8) mod 16 == x ^ 8`, so
// flipping the high bit of each nibble converts one convention to the other exactly. (The W8
// packer does the same thing with ^0x80 for kU8B128.)
__global__ void pack_q4_to_gptq_kernel(const std::uint8_t* __restrict__ codes,
                                       std::uint32_t* __restrict__ qweight, int n, int k) {
    const int col  = blockIdx.x * blockDim.x + threadIdx.x; // n index
    const int krow = blockIdx.y * blockDim.y + threadIdx.y; // k/8 index
    if (col >= n || krow >= k / 8) { return; }
    const std::uint8_t* row = codes + static_cast<std::size_t>(col) * (k / 2) + krow * 4;
    std::uint32_t packed    = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= static_cast<std::uint32_t>(row[i] ^ 0x88u) << (8 * i);
    }
    qweight[static_cast<std::size_t>(krow) * n + col] = packed;
}

// scales [n, k/64] FP16 -> Marlin's [k/64, n] BF16 with the 64-wide interleave.
__constant__ int kMoeScalePerm64[64];

__global__ void permute_moe_scales_kernel(const __half* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst, int n, int groups) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(groups) * n;
    if (index >= total) { return; }
    const int group  = static_cast<int>(index / n);
    const int column = static_cast<int>(index - static_cast<std::size_t>(group) * n);
    const int chunk  = column / 64;
    const int within = column % 64;
    const int source = chunk * 64 + kMoeScalePerm64[within];
    const float value =
        __half2float(src[static_cast<std::size_t>(source) * groups + group]);
    dst[index] = __float2bfloat16(value);
}

} // namespace

void marlin_moe_repack_q4g64(const void* codes, const void* scales_f16, std::int32_t experts,
                             std::int32_t n, std::int32_t k, void* gptq_tmp, void* b_out,
                             void* scales_out, cudaStream_t stream) {
    if (experts <= 0 || n <= 0 || k <= 0 || (k % 64) != 0 ||
        n % MARLIN_NAMESPACE_NAME::tile_n_size != 0 ||
        k % MARLIN_NAMESPACE_NAME::tile_k_size != 0) {
        throw std::invalid_argument("marlin moe repack: shape is not tileable");
    }
    {
        int host_perm[64];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) { host_perm[i * 8 + j] = i + 8 * j; }
        }
        CUDA_CHECK(cudaMemcpyToSymbolAsync(kMoeScalePerm64, host_perm, sizeof(host_perm), 0,
                                           cudaMemcpyHostToDevice, stream));
    }

    const std::size_t code_stride  = static_cast<std::size_t>(n) * (k / 2);
    const std::size_t scale_stride = static_cast<std::size_t>(n) * (k / 64);
    const std::size_t b_stride     = marlin_moe_b_bytes(n, k);
    const std::size_t s_stride     = marlin_moe_scale_bytes(n, k);

    for (std::int32_t expert = 0; expert < experts; ++expert) {
        const auto* expert_codes = static_cast<const std::uint8_t*>(codes) + expert * code_stride;
        const auto* expert_scales =
            static_cast<const __half*>(scales_f16) + expert * scale_stride;
        auto* expert_b = static_cast<std::uint8_t*>(b_out) + expert * b_stride;
        auto* expert_s = static_cast<std::uint8_t*>(scales_out) + expert * s_stride;
        {
            dim3 block(32, 8);
            dim3 grid((n + 31) / 32, (k / 8 + 7) / 8);
            pack_q4_to_gptq_kernel<<<grid, block, 0, stream>>>(
                expert_codes, static_cast<std::uint32_t*>(gptq_tmp), n, k);
            CUDA_CHECK(cudaGetLastError());
        }
        {
            const int groups        = k / 64;
            const std::size_t total = static_cast<std::size_t>(groups) * n;
            const int threads       = 256;
            const int blocks        = static_cast<int>((total + threads - 1) / threads);
            permute_moe_scales_kernel<<<blocks, threads, 0, stream>>>(
                expert_scales, reinterpret_cast<__nv_bfloat16*>(expert_s), n, groups);
            CUDA_CHECK(cudaGetLastError());
        }
        marlin_repack_tiles_q4(gptq_tmp, expert_b, n, k, stream);
    }
}

} // namespace sinfer::ops::detail

namespace sinfer::ops::detail {
namespace {

// moe_align_block_size in one block: our gather already sorts rows by expert, so the padded
// table is a prefix scan over the per-expert runs. Padding slots carry `assignments`, which
// the kernel treats as out of range, and every block gets the expert that owns it.
__global__ void build_routing_kernel(const std::int32_t* __restrict__ expert_offsets,
                                     std::int32_t num_experts, std::int32_t assignments,
                                     std::int32_t block_size,
                                     std::int32_t* __restrict__ sorted_token_ids,
                                     std::int32_t* __restrict__ expert_ids,
                                     std::int32_t* __restrict__ num_tokens_past_padded,
                                     std::int32_t padded_capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }
    std::int32_t out = 0;
    for (std::int32_t expert = 0; expert < num_experts; ++expert) {
        const std::int32_t begin = expert_offsets[expert];
        const std::int32_t end   = expert_offsets[expert + 1];
        const std::int32_t rows  = end - begin;
        if (rows <= 0) { continue; }
        const std::int32_t blocks = (rows + block_size - 1) / block_size;
        for (std::int32_t block = 0; block < blocks; ++block) {
            expert_ids[out / block_size] = expert;
            for (std::int32_t slot = 0; slot < block_size; ++slot) {
                const std::int32_t row = begin + block * block_size + slot;
                sorted_token_ids[out + slot] = row < end ? row : assignments;
            }
            out += block_size;
        }
    }
    // The kernel reads sorted ids as int4, so the tail must be padded too.
    for (std::int32_t slot = out; slot < padded_capacity; ++slot) {
        sorted_token_ids[slot] = assignments;
    }
    num_tokens_past_padded[0] = out;
}

} // namespace

void marlin_moe_build_routing(const std::int32_t* expert_offsets, std::int32_t num_experts,
                              std::int32_t assignments, std::int32_t block_size,
                              std::int32_t* sorted_token_ids, std::int32_t* expert_ids,
                              std::int32_t* num_tokens_past_padded, cudaStream_t stream) {
    const std::int32_t padded = marlin_moe_padded_rows(assignments, num_experts, block_size);
    build_routing_kernel<<<1, 1, 0, stream>>>(expert_offsets, num_experts, assignments,
                                              block_size, sorted_token_ids, expert_ids,
                                              num_tokens_past_padded, padded);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail

namespace sinfer::ops::detail {
namespace {

// Marlin writes the raw [rows, 2*intermediate] gate/up product; our pipeline wants
// silu(gate) * up folded to [rows, intermediate], which our own kernel produces inline.
__global__ void moe_silu_mul_kernel(const __nv_bfloat16* __restrict__ product,
                                    __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                    std::int32_t intermediate) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(rows) * intermediate;
    if (index >= total) { return; }
    const std::int64_t row    = index / intermediate;
    const std::int64_t column = index - row * intermediate;
    const std::int64_t base   = row * (2 * static_cast<std::int64_t>(intermediate));
    const float gate          = __bfloat162float(product[base + column]);
    const float up            = __bfloat162float(product[base + intermediate + column]);
    const float activated     = gate / (1.0F + __expf(-gate));
    out[index]                = __float2bfloat16(activated * up);
}

} // namespace

void marlin_moe_silu_mul(const void* product, void* out, std::int32_t rows,
                         std::int32_t intermediate, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(rows) * intermediate;
    const int threads        = 256;
    const int blocks         = static_cast<int>((total + threads - 1) / threads);
    moe_silu_mul_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(product), static_cast<__nv_bfloat16*>(out), rows,
        intermediate);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
