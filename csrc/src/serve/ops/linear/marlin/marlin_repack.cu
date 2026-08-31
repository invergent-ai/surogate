// Load-time repack of the serve W8G32 weight residency into the Marlin B
// tile format. The repack kernel is vendored from vLLM's
// gptq_marlin_repack.cu (Apache-2.0); the packing and scale-permutation
// kernels adapt the serve layout (codes [N,K] int8 row-major, scales
// [N,K/32] FP16 row-major) to the GPTQ/Marlin expectations.

#include "ops/linear/marlin/marlin_repack.h"

#include "vendor/marlin.cuh"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace MARLIN_NAMESPACE_NAME {


template <int const num_threads, int const num_bits, bool const has_perm,
          bool is_a_8bit>
__global__ void gptq_marlin_repack_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr, uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = target_tile_k_size / 4;

  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = target_tile_k_size / pack_factor;

  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * target_tile_k_size) / 4;

    int4 const* perm_int4_ptr = reinterpret_cast<int4 const*>(perm_ptr);

    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        uint32_t const* sh_perm_int_ptr =
            reinterpret_cast<uint32_t const*>(sh_perm_ptr);

        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;

        cp_async4(
            &sh_ptr[k_id * stage_n_threads + n_id],
            reinterpret_cast<int4 const*>(&(
                b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
      }

    } else {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        int first_k = k_tile_id * target_tile_k_size;
        int first_k_packed = first_k / pack_factor;

        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<int4 const*>(
                      &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                       first_n + (n_id * 4)])));
      }
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;

    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

    uint32_t vals[8];

    if constexpr (has_perm) {
      static_assert(!is_a_8bit);
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];

        uint32_t src_k = sh_perm_int_ptr[k_idx];
        uint32_t src_k_pos = src_k % pack_factor;

        uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
        uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;

        uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
        uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;

        vals[i] = b1_cur_val;
        vals[4 + i] = b2_cur_val;
      }

    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

#pragma unroll
      for (int i = 0; i < tile_ints; i++) {
        if constexpr (is_a_8bit) {
          b1_vals[i] =
              sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
        } else {
          b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
          b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
        }
      }

#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cur_elem = tc_row + (is_a_8bit ? i : tc_offsets[i]);
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;

        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        if constexpr (is_a_8bit)
          vals[4 + i] =
              (b1_vals[cur_int + tile_ints / 2] >> (cur_pos * num_bits)) & mask;
        else
          vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      }
    }

    constexpr int tile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    // Result of:
    // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
    if constexpr (!is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else if constexpr (is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ii = is_a_8bit ? i : pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}


}  // namespace MARLIN_NAMESPACE_NAME

namespace sinfer::ops::detail {
namespace {

// codes [N, K] int8 (bias-128 symmetric under u8b128 after xor) to the GPTQ
// qweight layout [K/4, N] uint32, 4 consecutive k codes per word.
__global__ void pack_w8_to_gptq_kernel(const std::uint8_t* __restrict__ codes,
                                       std::uint32_t* __restrict__ qweight, int n, int k) {
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;  // n index
    const int krow = blockIdx.y * blockDim.y + threadIdx.y;  // k/4 index
    if (col >= n || krow >= k / 4) { return; }
    const std::uint8_t* row = codes + static_cast<std::size_t>(col) * k + krow * 4;
    std::uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        packed |= static_cast<std::uint32_t>(row[i] ^ 0x80u) << (8 * i);
    }
    qweight[static_cast<std::size_t>(krow) * n + col] = packed;
}

// scales [N, K/32] FP16 to the Marlin layout: transpose to [K/32, N] BF16,
// then permute each 64-wide chunk with the grouped-scale interleave
// (vLLM marlin_permute_scales, group_size < K path).
__constant__ int kScalePerm64[64];

__global__ void permute_scales_kernel(const __half* __restrict__ scales_f16,
                                      __nv_bfloat16* __restrict__ out, int n, int groups) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(groups) * n;
    if (idx >= total) { return; }
    const std::size_t base   = idx / 64 * 64;
    const int lane           = static_cast<int>(idx % 64);
    const std::size_t source = base + kScalePerm64[lane];
    const int g              = static_cast<int>(source / n);
    const int col            = static_cast<int>(source % n);
    const float value = __half2float(scales_f16[static_cast<std::size_t>(col) * groups + g]);
    out[idx] = __float2bfloat16(value);
}

// FP8 codes carry no bias: the dequant reads the raw e4m3 byte, so packing
// is the plain [N,K] -> [K/4, N] u32 regroup.
__global__ void pack_fp8_to_gptq_kernel(const std::uint8_t* __restrict__ codes,
                                        std::uint32_t* __restrict__ qweight, int n, int k) {
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;
    const int krow = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || krow >= k / 4) { return; }
    const std::uint8_t* row = codes + static_cast<std::size_t>(col) * k + krow * 4;
    std::uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { packed |= static_cast<std::uint32_t>(row[i]) << (8 * i); }
    qweight[static_cast<std::size_t>(krow) * n + col] = packed;
}

// Channelwise scales use the 32-wide single-group permutation, and Marlin's
// FP8 dequant expects the exponent bias folded in: e4m3 has a 4-bit
// exponent against BF16's 8, so the scale carries 2^(2^7 - 2^3) = 2^120.
__constant__ int kScalePermSingle32[32];

__global__ void permute_row_scales_kernel(const __nv_bfloat16* __restrict__ scales_in,
                                          __nv_bfloat16* __restrict__ out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    const int base   = idx / 32 * 32;
    const int source = base + kScalePermSingle32[idx % 32];
    const float value = __bfloat162float(scales_in[source]);
    out[idx] = __float2bfloat16(value * 1.329227996e36F); // 2^120
}

} // namespace

void marlin_repack_fp8_row(const void* codes, const void* row_scales_bf16, int n, int k,
                           void* gptq_tmp, void* b_out, void* scales_out, cudaStream_t stream) {
    if (n % MARLIN_NAMESPACE_NAME::tile_n_size != 0 ||
        k % MARLIN_NAMESPACE_NAME::tile_k_size != 0 || (n % 32) != 0 || (k % 4) != 0) {
        throw std::invalid_argument("marlin fp8 repack: shape is not tileable");
    }
    {
        dim3 block(32, 8);
        dim3 grid((n + 31) / 32, (k / 4 + 7) / 8);
        pack_fp8_to_gptq_kernel<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(codes), static_cast<std::uint32_t*>(gptq_tmp), n, k);
    }
    {
        int host_perm[32];
        int at = 0;
        for (int i = 0; i < 4; ++i) {
            const int lanes[8] = {0, 1, 8, 9, 16, 17, 24, 25};
            for (int j = 0; j < 8; ++j) { host_perm[at++] = 2 * i + lanes[j]; }
        }
        cudaMemcpyToSymbolAsync(kScalePermSingle32, host_perm, sizeof(host_perm), 0,
                                cudaMemcpyHostToDevice, stream);
        const int threads = 256;
        const int blocks  = (n + threads - 1) / threads;
        permute_row_scales_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(row_scales_bf16),
            static_cast<__nv_bfloat16*>(scales_out), n);
    }
    {
        int device = 0;
        cudaGetDevice(&device);
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        int max_shared_mem = 0;
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        auto kernel = MARLIN_NAMESPACE_NAME::gptq_marlin_repack_kernel<
            MARLIN_NAMESPACE_NAME::repack_threads, 8, false, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
        kernel<<<sms, MARLIN_NAMESPACE_NAME::repack_threads, max_shared_mem, stream>>>(
            static_cast<const std::uint32_t*>(gptq_tmp), nullptr,
            static_cast<std::uint32_t*>(b_out), k, n);
    }
}

void marlin_repack_w8g32(const void* codes, const void* scales_f16, int n, int k,
                         void* gptq_tmp, void* b_out, void* scales_out, cudaStream_t stream) {
    if (n % MARLIN_NAMESPACE_NAME::tile_n_size != 0 ||
        k % MARLIN_NAMESPACE_NAME::tile_k_size != 0 || k % 32 != 0) {
        throw std::invalid_argument("marlin repack: shape is not tileable");
    }
    {
        dim3 block(32, 8);
        dim3 grid((n + 31) / 32, (k / 4 + 7) / 8);
        pack_w8_to_gptq_kernel<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(codes), static_cast<std::uint32_t*>(gptq_tmp), n,
            k);
    }
    {
        int host_perm[64];
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) { host_perm[i * 8 + j] = i + 8 * j; }
        }
        cudaMemcpyToSymbolAsync(kScalePerm64, host_perm, sizeof(host_perm), 0,
                                cudaMemcpyHostToDevice, stream);
        const int groups        = k / 32;
        const std::size_t total = static_cast<std::size_t>(groups) * n;
        const int threads       = 256;
        const int blocks        = static_cast<int>((total + threads - 1) / threads);
        permute_scales_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __half*>(scales_f16), static_cast<__nv_bfloat16*>(scales_out), n,
            groups);
    }
    {
        int device = 0;
        cudaGetDevice(&device);
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        int max_shared_mem = 0;
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device);
        auto kernel = MARLIN_NAMESPACE_NAME::gptq_marlin_repack_kernel<
            MARLIN_NAMESPACE_NAME::repack_threads, 8, false, false>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             max_shared_mem);
        kernel<<<sms, MARLIN_NAMESPACE_NAME::repack_threads, max_shared_mem, stream>>>(
            static_cast<const std::uint32_t*>(gptq_tmp), nullptr,
            static_cast<std::uint32_t*>(b_out), k, n);
    }
}

// The 4-bit tile repack, for the MoE path: the vendored kernel is instantiated here
// because this is the translation unit that owns it (#89).
void marlin_repack_tiles_q4(const void* gptq_tmp, void* b_out, int n, int k,
                            cudaStream_t stream) {
    int device = 0;
    cudaGetDevice(&device);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    auto kernel = MARLIN_NAMESPACE_NAME::gptq_marlin_repack_kernel<
        MARLIN_NAMESPACE_NAME::repack_threads, 4, false, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    kernel<<<sms, MARLIN_NAMESPACE_NAME::repack_threads, max_shared_mem, stream>>>(
        static_cast<const std::uint32_t*>(gptq_tmp), nullptr,
        static_cast<std::uint32_t*>(b_out), k, n);
}

std::size_t marlin_b_out_words(int n, int k) {
    return static_cast<std::size_t>(k / MARLIN_NAMESPACE_NAME::tile_size) *
           (static_cast<std::size_t>(n) * MARLIN_NAMESPACE_NAME::tile_size / 4);
}

} // namespace sinfer::ops::detail
