#include "api/ops/lora_base.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/ggml/ggml_moe_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <stdexcept>

namespace sinfer::ops {
namespace {
__device__ float value_at(Weight w, int row, int col) {
    using namespace detail;
    if (w.segment_count) {
        for (int i = 0; i < w.segment_count; ++i) {
            const auto segment = w.segments[i];
            if (row >= segment.row_begin && row < segment.row_begin + segment.rows) {
                row -= segment.row_begin;
                w.qdata = segment.qdata;
                w.qtype = segment.qtype;
                break;
            }
        }
    }
    if (w.input_group_map) { col = w.input_group_map[col / 32] * 32 + col % 32; }
    const auto index   = static_cast<std::int64_t>(row) * w.k + col;
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    if (w.qtype == QType::BF16_CTRL) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(w.qdata)[index]);
    }
    if (w.qtype == QType::FP32_CTRL) { return static_cast<const float*>(w.qdata)[index]; }
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S || w.qtype == QType::FP8_E4M3FN_BLK128_F32S ||
        w.qtype == QType::FP8_E4M3FN_ROW_F32S) {
        const float v = decode_nvfp4_e4m3(codes[index]);
        if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
            return v * __bfloat162float(static_cast<const __nv_bfloat16*>(w.scales)[row]);
        }
        const int si = w.qtype == QType::FP8_E4M3FN_ROW_F32S
                           ? row
                           : (row / 128) * ((w.k + 127) / 128) + col / 128;
        return v * static_cast<const float*>(w.scales)[si];
    }
    if (w.qtype == QType::NVFP4) {
        const int group = col / 16, inner = row & 127;
        const auto si =
            (static_cast<std::int64_t>(row / 128) * ((w.k / 16 + 3) / 4) + group / 4) * 512 +
            (inner & 31) * 16 + (inner / 32) * 4 + group % 4;
        const auto packed = decode_nvfp4_e2m1x2(codes[index / 2]);
        return (col % 2 ? packed.y : packed.x) * decode_nvfp4_e4m3(scales[si]) *
               (w.weight_scale_divisor > 0 ? 1.0F / w.weight_scale_divisor : 1.0F);
    }
    if (w.layout == QuantLayout::RowSplit) {
        const auto group =
            static_cast<std::int64_t>(row) * (w.padded_shape[1] / w.group) + col / w.group;
        const auto sbits = static_cast<const std::uint16_t*>(w.scales)[group];
        if (w.qtype == QType::W8G32_F16S) {
            return static_cast<float>(
                       reinterpret_cast<const std::int8_t*>(codes)[group * 32 + col % 32]) *
                   __half2float(__ushort_as_half(sbits));
        }
        float v[8];
        const int lane    = (col % 64) / 8;
        const auto packed = *reinterpret_cast<const std::uint32_t*>(codes + group * 32 + lane * 4);
        if (w.qtype == QType::Q4G64_F16S) {
            Q4SimtDecodeAtom::decode_eight(packed, sbits, v);
        } else if (w.qtype == QType::Q5G64_F16S) {
            Q5SimtDecodeAtom::decode_eight(packed, high[group * 8 + lane], sbits, v);
        } else {
            Q6SimtDecodeAtom::decode_eight(
                packed, reinterpret_cast<const std::uint16_t*>(high)[group * 8 + lane], sbits, v);
        }
        return v[col % 8];
    }
    float v[8];
    switch (w.qtype) {
#define READ_GGML(NAME)                                                                            \
    case QType::NAME: {                                                                            \
        constexpr int size = detail::ggml::block_values(detail::ggml::GgmlType::NAME);             \
        detail::ggml::decode_eight<detail::ggml::GgmlType::NAME>(codes, index / size,              \
                                                                 (index % size) / 8, v);           \
        return v[col % 8];                                                                         \
    }
        SINFER_GGML_FOR_EACH_TYPE(READ_GGML)
#undef READ_GGML
    default:
        return nanf("");
    }
}

__global__ void norms_kernel(LoraBaseView view, const __nv_bfloat16* a, const __nv_bfloat16* b,
                             int rank, int in, float scale, float* norms) {
    const int local = blockIdx.x;
    const int row   = view.output_offset + local;
    float sum       = 0;
    for (int k = threadIdx.x; k < in; k += blockDim.x) {
        float delta = 0;
        for (int r = 0; r < rank; ++r) {
            delta =
                fmaf(__bfloat162float(a[r * in + k]), __bfloat162float(b[row * rank + r]), delta);
        }
        float base = view.transpose ? value_at(view.weight, k, view.row_offset + local)
                                    : value_at(view.weight, view.row_offset + local, k);
        if (view.multiplier) { base *= *view.multiplier; }
        const float value = base + scale * delta;
        sum               = fmaf(value, value, sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = 128; width; width /= 2) {
        if (threadIdx.x < width) { partial[threadIdx.x] += partial[threadIdx.x + width]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { norms[row] = sqrtf(partial[0]); }
}
} // namespace

std::vector<float> lora_weight_norms(const std::vector<LoraBaseView>& base,
                                     const std::vector<std::uint16_t>& a,
                                     const std::vector<std::uint16_t>& b, int rank, int in, int out,
                                     float scale) {
    if (base.empty() || rank <= 0 || in <= 0 || out <= 0 || !std::isfinite(scale) ||
        a.size() != static_cast<std::size_t>(rank) * in ||
        b.size() != static_cast<std::size_t>(out) * rank) {
        throw std::invalid_argument("invalid DoRA direction geometry");
    }
    std::vector<bool> covered(out, false);
    std::size_t segment_bytes = 0;
    for (const auto& part : base) {
        const auto& w = part.weight;
        if (!w.qdata || part.rows <= 0 || part.row_offset < 0 || part.output_offset < 0 ||
            part.output_offset > out - part.rows ||
            (part.transpose ? (in != w.n || part.row_offset > w.k - part.rows)
                            : (in != w.k || part.row_offset > w.n - part.rows))) {
            throw std::invalid_argument("DoRA base mapping disagrees with the module");
        }
        for (int i = 0; i < part.rows; ++i) {
            if (covered[part.output_offset + i]) {
                throw std::invalid_argument("DoRA base rows overlap");
            }
            covered[part.output_offset + i] = true;
        }
        if (w.segment_count) { segment_bytes += w.segment_count * sizeof(WeightSegment) + 256; }
    }
    for (bool present : covered) {
        if (!present) { throw std::invalid_argument("DoRA base mapping has missing rows"); }
    }
    DeviceArena arena((a.size() + b.size()) * 2 + out * 4 + segment_bytes + 1024);
    auto da = arena.alloc_bytes(a.size() * 2), db = arena.alloc_bytes(b.size() * 2),
         dn = arena.alloc_bytes(out * 4);
    CUDA_CHECK(cudaMemcpy(da.data, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db.data, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dn.data, 0, out * 4));
    for (auto part : base) {
        if (part.weight.segment_count) {
            const auto bytes    = part.weight.segment_count * sizeof(WeightSegment);
            const auto segments = arena.alloc_bytes(bytes);
            CUDA_CHECK(cudaMemcpy(segments.data, part.weight.segments, bytes, cudaMemcpyDefault));
            part.weight.segments = static_cast<const WeightSegment*>(segments.data);
        }
        norms_kernel<<<part.rows, 256>>>(part, static_cast<const __nv_bfloat16*>(da.data),
                                         static_cast<const __nv_bfloat16*>(db.data), rank, in,
                                         scale, static_cast<float*>(dn.data));
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<float> norms(out);
    CUDA_CHECK(cudaMemcpy(norms.data(), dn.data, out * 4, cudaMemcpyDeviceToHost));
    for (float value : norms) {
        if (!(value > 0) || !std::isfinite(value)) {
            throw std::invalid_argument("DoRA direction has a zero or nonfinite norm");
        }
    }
    return norms;
}
} // namespace sinfer::ops
