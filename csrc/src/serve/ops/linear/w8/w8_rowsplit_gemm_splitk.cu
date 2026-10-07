#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_pipelined.cuh"
#include "ops/linear/w8/w8_rowsplit_wgmma_sm90.h"
#include "ops/linear/w8/w8_launch.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sinfer::ops::detail {

void w8_rowsplit_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);

namespace {

constexpr int kRows           = 2048;
constexpr int kHidden         = 16384;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using ExactTLauncher          = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols  = ActiveCols <= 8    ? 8
                              : ActiveCols <= 16 ? 16
                              : ActiveCols <= 24 ? 24
                              : ActiveCols <= 32 ? 32
                              : ActiveCols <= 40 ? 40
                                                 : 48;
    constexpr int KWarps    = ActiveCols <= 36 ? 16 : 8;
    constexpr int MinBlocks = KWarps == 16 ? 1 : 2;
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto ActivationCache = ActiveCols <= 36 || ActiveCols == 48 ? Cache::cg : Cache::ca;
    using Geometry                 = W8LinearGeometry<kRows, kHidden>;
    using Schedule = W8SmallTMmaSchedule<KWarps, TileCols, MinBlocks, ScaleAccess, ActivationCache>;
    static_assert((kRows % kRowsPerCta) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ExactTLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

void require_problem(const Tensor& x, const Weight& w, const Tensor& out) {
    if (x.ne[0] != kHidden || out.ne[0] != kRows || out.ne[1] != x.ne[1] || w.n != kRows ||
        w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("W8 exact-T split-K requires [2048,16384]");
    }
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kRows};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, x.ne[1]);
}

struct DeviceTraits {
    int sms        = 0;
    int major      = 0;
    int smem_optin = 0; ///< dynamic shared memory a block may opt into
};

// The current device's SM count and compute capability, read once per device.
DeviceTraits device_traits() {
    static std::mutex mutex;
    static std::map<int, DeviceTraits> traits;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto found = traits.find(device); found != traits.end()) { return found->second; }
    DeviceTraits t;
    CUDA_CHECK(cudaDeviceGetAttribute(&t.sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&t.major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&t.smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return traits[device] = t;
}

// Whether the consistent route runs the pipelined kernel (w8_rowsplit_gemm_pipelined.cuh): on
// sm_90, where it was measured, unless SUROGATE_SERVE_W8_PIPELINED is "0"; "1" forces it on any
// device. Its outputs are the medium-T kernel's bits either way.
bool pipelined_route() {
    static const int forced = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_W8_PIPELINED");
        return raw == nullptr || *raw == '\0' ? -1 : (raw[0] == '0' ? 0 : 1);
    }();
    return forced >= 0 ? forced == 1 : device_traits().major == 9;
}

// Whether the consistent route may give a pipelined CTA two sets of warps over one staging of
// activations (SUROGATE_SERVE_W8_ROW_GROUPS=1 turns it off).
bool row_groups_allowed() {
    static const bool allowed = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_W8_ROW_GROUPS");
        return raw == nullptr || *raw == '\0' || std::atoi(raw) >= 2;
    }();
    return allowed;
}

template <int Columns, int Stages, int RowTiles, int RowGroups = 1>
void launch_pipelined(const __nv_bfloat16* x, const Weight& w, W8ContiguousOutput output, int count,
                      cudaStream_t stream) {
    using Layout = W8PipelinedLayout<Columns, Stages, RowTiles, RowGroups>;
    auto* kernel = w8_rowsplit_pipelined_kernel<Columns, Stages, RowTiles, RowGroups>;
    // The attribute is per device; setting it again is cheap and keeps a second device right.
    static std::mutex mutex;
    static std::map<int, bool> set;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!set[device]) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(Layout::kBytes)));
            set[device] = true;
        }
    }
    const dim3 grid(static_cast<unsigned>(w.n / Layout::kRowsPerCta),
                    static_cast<unsigned>((count + Columns - 1) / Columns));
    kernel<<<grid, Layout::kThreads, Layout::kBytes, stream>>>(
        x, static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        output, count, w.k);
}

} // namespace

void launch_w8_consistent(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.k % 256 != 0 || w.n % 16 != 0 || w.padded_shape[1] != w.k) {
        throw std::invalid_argument("W8 consistent GEMM requires aligned, unpadded rows");
    }
    // Two 16-row tiles a CTA when that still leaves two CTAs an SM. Every CTA stages the
    // activations of its columns for the whole K, so 16-row CTAs over a vocab head read
    // them from L2 eight times over the weight's own bytes; 32-row CTAs halve that. The
    // per-output K order is the kernel's either way (bit-identical outputs). On an H100,
    // Qwen3-8B's W8 head (151936 x 4096): 376 -> 331 us at one token, 659 -> 520 at 64;
    // 12288 x 4096: 43 -> 33 at one token. Narrower weights lose parallelism instead
    // (6144 x 4096: 17 -> 20 us), so they keep one tile.
    const DeviceTraits traits = device_traits();
    const bool row_pairs = w.n % 32 == 0 && w.n / 32 >= 2 * traits.sms;
    const bool pipelined = pipelined_route();
    const bool wgmma     = w8_wgmma_available();
    for_each_token_slice(x.ne[1], 64, [&](int begin, int count) {
        const Tensor input = x.slice(1, begin, count);
        Tensor result = out.slice(1, begin, count);
        const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(result.data), w.n};
        // Hopper from 33 columns: the same arithmetic on wgmma (w8_rowsplit_wgmma_sm90.h), the
        // tensor rate a wide round needs. It declines shapes it does not tile, which keep the
        // kernels below.
        if (wgmma && count >= w8_wgmma_min_columns() &&
            w8_wgmma_consistent({static_cast<const __nv_bfloat16*>(input.data),
                                 static_cast<const std::uint8_t*>(w.qdata),
                                 static_cast<const std::uint8_t*>(w.scales),
                                 static_cast<__nv_bfloat16*>(result.data), w.n, w.k, count},
                                stream)) {
            return;
        }
        if (pipelined) {
            const auto* activations = static_cast<const __nv_bfloat16*>(input.data);
            // From the variants' sweep on an H100 (sinfer_w8_pipelined_test bench; Qwen3-8B's W8
            // head 151936 x 4096 and a 12288 x 4096 projection). Up to 8 columns the weights
            // dominate: four groups in flight, the head 330 -> 229 us. From 9 to 32 columns a
            // CTA stages more activations than codes, so where the rows give each SM four or
            // more CTAs of 64 rows, two warp sets share one staging: the head at 16 columns
            // 311 -> 274 us, at 32 380 -> 333. Past 32 columns the medium-T kernel stays (the
            // head 526 us against 523 grouped, the projection 45 against 54 pipelined).
            const bool grouped = row_pairs && row_groups_allowed() && w.n % 64 == 0 &&
                                 w.n / 64 >= 4 * traits.sms &&
                                 W8PipelinedLayout<32, 2, 2, 2>::kBytes <=
                                     static_cast<std::size_t>(traits.smem_optin);
            if (count <= 8) {
                if (row_pairs) { launch_pipelined<8, 4, 2>(activations, w, output, count, stream); }
                else { launch_pipelined<8, 4, 1>(activations, w, output, count, stream); }
                return;
            }
            if (count <= 16) {
                if (grouped) { launch_pipelined<16, 2, 2, 2>(activations, w, output, count, stream); }
                else if (row_pairs) { launch_pipelined<16, 4, 2>(activations, w, output, count, stream); }
                else { launch_pipelined<16, 4, 1>(activations, w, output, count, stream); }
                return;
            }
            if (count <= 32 && (grouped || row_pairs)) {
                if (grouped) { launch_pipelined<32, 2, 2, 2>(activations, w, output, count, stream); }
                else { launch_pipelined<32, 3, 2>(activations, w, output, count, stream); }
                return;
            }
        }
        const auto launch = [&]<int Columns>() {
            const dim3 tiles(1, (count + Columns - 1) / Columns);
            const auto* activations = static_cast<const __nv_bfloat16*>(input.data);
            const auto* codes       = static_cast<const std::uint8_t*>(w.qdata);
            const auto* scales      = static_cast<const std::uint8_t*>(w.scales);
            if (row_pairs) {
                w8_rowsplit_medium_t_splitk_kernel<0, Columns, 4, 1, 2, W8ContiguousOutput, false, 2>
                    <<<dim3(w.n / 32, tiles.y), 128, 0, stream>>>(activations, codes, scales, output,
                                                                 count, w.k);
            } else {
                w8_rowsplit_medium_t_splitk_kernel<0, Columns, 4, 1, 2>
                    <<<dim3(w.n / 16, tiles.y), 128, 0, stream>>>(activations, codes, scales, output,
                                                                 count, w.k);
            }
        };
        if (count <= 8) { launch.template operator()<8>(); }
        else if (count <= 16) { launch.template operator()<16>(); }
        else if (count <= 32) { launch.template operator()<32>(); }
        else { launch.template operator()<64>(); }
    });
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_exact_t_splitk(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 exact-T split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactCols](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_exact_t_composite(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] < 33 || x.ne[1] > 127) {
        throw std::invalid_argument("W8 exact-T composite requires T=33..127");
    }

    std::int32_t offset = 0;
    while (x.ne[1] - offset >= 32) {
        const Tensor x_slice = x.slice(1, offset, 32);
        Tensor out_slice     = out.slice(1, offset, 32);
        launch_w8_exact_t_splitk(x_slice, w, out_slice, stream);
        offset += 32;
    }
    const std::int32_t tail = x.ne[1] - offset;
    if (tail == 1) {
        const Tensor x_slice = x.slice(1, offset, 1);
        Tensor out_slice     = out.slice(1, offset, 1);
        w8_rowsplit_decode_r16_launch(x_slice, w, out_slice, stream);
    } else if (tail >= 2) {
        const Tensor x_slice = x.slice(1, offset, tail);
        Tensor out_slice     = out.slice(1, offset, tail);
        launch_w8_exact_t_splitk(x_slice, w, out_slice, stream);
    }
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    if (x.ne[1] > TileCols) {
        throw std::invalid_argument("W8 medium-T split-K route does not cover this T");
    }
    launch_medium<TileCols, KSplits, NGroups, MinBlocks>(x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_dflash_medium(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_problem(x, w, out);
    const int t = x.ne[1];
    if (t < 49 || t > 128) {
        throw std::invalid_argument("W8 DFlash medium route requires T=49..128");
    }

    if (t <= 64) {
        launch_medium<64, 8, 4, 1>(x, w, out, stream);
    } else if (t == 65) {
        launch_medium<80, 8, 2, 1>(x, w, out, stream);
    } else if (t <= 72) {
        launch_medium<72, 8, 3, 1>(x, w, out, stream);
    } else if (t <= 80) {
        launch_medium<80, 8, 2, 1>(x, w, out, stream);
    } else if (t <= 96) {
        launch_medium<96, 4, 6, 1>(x, w, out, stream);
    } else if (t <= 104) {
        launch_medium<104, 4, 1, 1>(x, w, out, stream);
    } else if (t <= 112) {
        launch_medium<112, 4, 7, 1>(x, w, out, stream);
    } else if (t <= 120) {
        launch_medium<120, 4, 5, 1>(x, w, out, stream);
    } else if (t <= 125) {
        launch_medium<128, 4, 4, 1>(x, w, out, stream);
    } else {
        launch_medium<128, 4, 8, 1>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_w8_medium_splitk_c144(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    launch_medium_route<144, 2, 9, 2>(x, w, out, stream);
}

} // namespace sinfer::ops::detail
