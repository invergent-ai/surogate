// Weight-only NVFP4 Marlin residency and dense GEMM (see marlin_nvfp4.h).

#include "ops/linear/marlin/marlin_nvfp4.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/engine_context.h"
#include "ops/linear/marlin/marlin_gemm.h"
#include "ops/linear/marlin/marlin_repack.h"
#include "ops/linear/marlin/vendor/marlin.cuh"
#include "ops/linear/plane_storage.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops::detail {
namespace {

// The widest MoE block the route runs (vLLM's block_size_m tops out at 64).
constexpr int kWidestMoeBlock = 64;
// Guard bytes in front of the locks, as the W8 scratch keeps them.
constexpr std::size_t kLockGuardBytes = 256;

struct DeviceState {
    std::mutex mutex;
    std::unique_ptr<DeviceBuffer> scratch;
    Nvfp4MarlinScratch view;
    /// The global-scale arrays adopted weights point at; they live as long as the engine.
    std::vector<std::unique_ptr<DeviceBuffer>> global_scales;
};

DeviceState& device_state() { return engine_slot<DevicePlaneStates<DeviceState>>().current(); }

void ensure_scratch(DeviceState& state, cudaStream_t stream) {
    if (state.scratch) { return; }
    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    // The MoE reduce buffer bound vLLM allocates (sms * 4 blocks of the widest block size and
    // tile), which also covers the dense GEMM's sms * 64 rows of the widest tile.
    const std::size_t c_tmp_bytes = static_cast<std::size_t>(sms) * 4 * kWidestMoeBlock *
                                    MARLIN_NAMESPACE_NAME::max_thread_n * sizeof(float);
    const std::size_t lock_count  = std::max<std::size_t>(static_cast<std::size_t>(sms) * 4, 1024);
    const std::size_t lock_bytes  = lock_count * sizeof(int) + kLockGuardBytes;
    auto buffer = std::make_unique<DeviceBuffer>(c_tmp_bytes + lock_bytes);
    auto* base  = static_cast<std::byte*>(buffer->p);
    CUDA_CHECK(cudaMemsetAsync(base + c_tmp_bytes, 0, lock_bytes, stream));
    state.view = Nvfp4MarlinScratch{
        .c_tmp       = base,
        .c_tmp_bytes = c_tmp_bytes,
        .locks       = reinterpret_cast<int*>(base + c_tmp_bytes + kLockGuardBytes),
        .sm_count    = sms,
    };
    state.scratch = std::move(buffer);
}

// The largest block scale of a weight. E4M3 words of one sign order as their values, and the
// stored scales are nonnegative, so the largest word is the largest scale.
__global__ void scale_word_max_kernel(const std::uint8_t* __restrict__ scales, std::size_t count,
                                      unsigned* __restrict__ out) {
    unsigned local = 0;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local = max(local, static_cast<unsigned>(scales[i] & 0x7Fu));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = max(local, __shfl_down_sync(0xffffffffu, local, offset));
    }
    if ((threadIdx.x & 31) == 0) { atomicMax(out, local); }
}

__global__ void global_scale_kernel(const float* __restrict__ multipliers, int stride, int blocks,
                                    float factor, float* __restrict__ out) {
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block < blocks) { out[block] = multipliers[static_cast<std::size_t>(block) * stride] * factor; }
}

// Codes [n, k/2] (even column in the low nibble) -> GPTQ [k/8, n] u32: one word is exactly
// four bytes of a row, eight consecutive values with value i in bits [4i, 4i + 4). E2M1 codes
// carry no bias, so unlike the Q4 packer nothing is flipped.
__global__ void pack_fp4_to_gptq_kernel(const std::uint8_t* __restrict__ codes,
                                        std::uint32_t* __restrict__ qweight, int n, int k) {
    const int col  = blockIdx.x * blockDim.x + threadIdx.x;
    const int krow = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || krow >= k / 8) { return; }
    const auto* row = reinterpret_cast<const std::uint32_t*>(
        codes + static_cast<std::size_t>(col) * (k / 2));
    qweight[static_cast<std::size_t>(krow) * n + col] = row[krow];
}

__device__ __forceinline__ int scale_perm64(int lane) { return lane / 8 + 8 * (lane % 8); }

// One block's E4M3 scales in the stored tiling -> Marlin's encoding, [k/16, rows] flattened.
// Output position o reads the [0, 2, 1, 3]-reordered, 64-permuted element of the transposed
// matrix (vLLM: marlin_permute_scales, then nvfp4_marlin_process_scales).
__global__ void marlin_fp4_scales_kernel(const std::uint8_t* __restrict__ stored,
                                         std::int64_t row0, int rows, int k, float sf,
                                         std::uint8_t* __restrict__ out) {
    const std::int64_t o     = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int groups         = k / 16;
    const std::int64_t total = static_cast<std::int64_t>(groups) * rows;
    if (o >= total) { return; }
    constexpr int kReorder[4] = {0, 2, 1, 3};
    const std::int64_t i = (o & ~std::int64_t{3}) | kReorder[o & 3];
    const std::int64_t j = (i & ~std::int64_t{63}) + scale_perm64(static_cast<int>(i & 63));
    const int group      = static_cast<int>(j / rows);
    const std::int64_t row = row0 + (j - static_cast<std::int64_t>(group) * rows);
    // The registered block-scale tiling (layouts.py swizzle_nvfp4_scales).
    const int k_tiles = k / 64;
    const std::int64_t offset = ((row / 128) * k_tiles + group / 4) * 512 + (row % 32) * 16 +
                                ((row % 128) / 32) * 4 + group % 4;
    __nv_fp8_e4m3 word;
    word.__x    = stored[offset];
    const float value = static_cast<float>(word) * sf * 128.0F;
    std::uint8_t encoded = 0;
    if (value >= 2.0F) {
        encoded = static_cast<std::uint8_t>((__half_as_ushort(__float2half_rn(value)) >> 7) & 0xFFu);
    }
    out[o] = encoded;
}

float host_scale_factor(unsigned max_word) {
    if (max_word == 0) { return 1.0F; }
    __nv_fp8_e4m3 word;
    word.__x = static_cast<__nv_fp8_storage_t>(max_word);
    const float largest = static_cast<float>(word) * 128.0F;
    const float top     = 448.0F * 128.0F;
    // vLLM's _nvfp4_compute_scale_factor: the power of two that brings the largest nonzero
    // scale (times 2^7) closest to, without passing, 448 * 2^7.
    return largest < top ? std::exp2(std::floor(std::log2(top / largest))) : 1.0F;
}

} // namespace

bool nvfp4_marlin_ready(const Weight& weight) noexcept {
    return weight.qtype == QType::NVFP4_A16 && weight.layout == QuantLayout::MarlinTiles &&
           weight.marlin_global_scale != nullptr;
}

Nvfp4MarlinScratch nvfp4_marlin_scratch() {
    DeviceState& state = device_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.scratch) {
        throw std::logic_error("nvfp4 marlin: no weight on this device was adopted at load");
    }
    return state.view;
}

void nvfp4_marlin_adopt(Weight& weight, std::int32_t blocks, const float* multipliers,
                        std::int32_t stride, cudaStream_t stream) {
    if (nvfp4_marlin_ready(weight)) { return; }
    if (weight.qtype != QType::NVFP4_A16 || weight.layout != QuantLayout::BlockScaleK16M128x4 ||
        weight.qdata == nullptr || weight.scales == nullptr || blocks <= 0 ||
        weight.n % blocks != 0) {
        throw std::invalid_argument("nvfp4 marlin adopt: expected a stored NVFP4_A16 weight");
    }
    const int rows = weight.n / blocks;
    const int k    = weight.k;
    // Marlin's tiles and its smallest thread tile; the stored tiling needs 128-row blocks.
    if (rows % 128 != 0 || k % 128 != 0) {
        throw std::invalid_argument("nvfp4 marlin adopt: rows per block must be a multiple of "
                                    "128 and K of 128, got " + std::to_string(rows) + " x " +
                                    std::to_string(k));
    }
    DeviceState& state = device_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    ensure_scratch(state, stream);

    auto* codes                    = static_cast<std::uint8_t*>(const_cast<void*>(weight.qdata));
    auto* scales                   = static_cast<std::uint8_t*>(const_cast<void*>(weight.scales));
    const std::size_t scale_bytes  = static_cast<std::size_t>(weight.n) * (k / 16);
    const std::size_t block_codes  = static_cast<std::size_t>(rows) * (k / 2);
    const std::size_t block_scales = static_cast<std::size_t>(rows) * (k / 16);
    // A dense matrix's own second level is the payload's trailing word, right after its scales.
    const float* second = multipliers != nullptr
                              ? multipliers
                              : reinterpret_cast<const float*>(scales + scale_bytes);
    if (multipliers == nullptr && blocks != 1) {
        throw std::invalid_argument("nvfp4 marlin adopt: a bank needs its per-block multipliers");
    }

    DeviceBuffer max_word(sizeof(unsigned));
    CUDA_CHECK(cudaMemsetAsync(max_word.p, 0, sizeof(unsigned), stream));
    scale_word_max_kernel<<<512, 256, 0, stream>>>(scales, scale_bytes,
                                                   static_cast<unsigned*>(max_word.p));
    CUDA_CHECK(cudaGetLastError());
    unsigned largest = 0;
    CUDA_CHECK(cudaMemcpyAsync(&largest, max_word.p, sizeof(largest), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const float sf = host_scale_factor(largest);

    auto globals = std::make_unique<DeviceBuffer>(static_cast<std::size_t>(blocks) * sizeof(float));
    global_scale_kernel<<<(blocks + 255) / 256, 256, 0, stream>>>(
        second, stride, blocks, std::exp2(119.0F) / sf, static_cast<float*>(globals->p));
    CUDA_CHECK(cudaGetLastError());

    DeviceBuffer gptq(block_codes);
    DeviceBuffer staged(block_scales);
    for (std::int32_t block = 0; block < blocks; ++block) {
        std::uint8_t* block_code  = codes + static_cast<std::size_t>(block) * block_codes;
        std::uint8_t* block_scale = scales + static_cast<std::size_t>(block) * block_scales;
        {
            const dim3 threads(32, 8);
            const dim3 grid((rows + 31) / 32, (k / 8 + 7) / 8);
            pack_fp4_to_gptq_kernel<<<grid, threads, 0, stream>>>(
                block_code, static_cast<std::uint32_t*>(gptq.p), rows, k);
            CUDA_CHECK(cudaGetLastError());
        }
        // The tile repack reads only the GPTQ staging, so it writes the codes' own bytes.
        marlin_repack_tiles_q4(gptq.p, block_code, rows, k, stream);
        CUDA_CHECK(cudaGetLastError());
        {
            const std::int64_t total = static_cast<std::int64_t>(rows) * (k / 16);
            marlin_fp4_scales_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
                scales, static_cast<std::int64_t>(block) * rows, rows, k, sf,
                static_cast<std::uint8_t*>(staged.p));
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpyAsync(block_scale, staged.p, block_scales, cudaMemcpyDeviceToDevice,
                                   stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout              = QuantLayout::MarlinTiles;
    weight.marlin_global_scale = static_cast<const float*>(globals->p);
    state.global_scales.push_back(std::move(globals));
}

void nvfp4_marlin_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (!nvfp4_marlin_ready(weight)) {
        throw std::invalid_argument(
            "nvfp4 marlin linear: an NVFP4_A16 weight is served from Marlin tiles only, and this "
            "one was not adopted at load");
    }
    const Nvfp4MarlinScratch scratch = nvfp4_marlin_scratch();
    marlin_gemm_nvfp4_bf16(x.data, weight.qdata, weight.scales, weight.marlin_global_scale,
                           out.data, scratch.c_tmp, scratch.locks, x.ne[1], weight.n, weight.k,
                           scratch.sm_count, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
