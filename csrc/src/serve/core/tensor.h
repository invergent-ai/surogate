#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace sinfer {

struct Tensor {
    void* data         = nullptr;
    DType dtype        = DType::BF16;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

enum class QType : std::uint16_t {
    Q4G64_F16S           = 0,
    Q5G64_F16S           = 1,
    Q6G64_F16S           = 2,
    W8G32_F16S           = 3,
    BF16_CTRL            = 4,
    FP32_CTRL            = 5,
    I32_CTRL             = 6,
    NVFP4                = 7,
    FP8_E4M3FN_ROW_BF16S = 8,
    // GGML K-quants in their own superblock layout (QuantLayout::GgmlBlocks), bytes as the
    // GGUF stores them: 256 values per block, affine sub-scales, k a multiple of 256.
    Q2_K                 = 9,
    Q3_K                 = 10,
    Q4_K                 = 11,
    Q5_K                 = 12,
    Q6_K                 = 13,
    /// Not a K-quant: 32 values with one binary16 scale and no sub-scales. The same numbers
    /// W8G32_F16S holds, arranged as the GGUF arranges them -- interleaved per block rather than
    /// split into planes -- so a K_M quant's attention, GDN and shared-expert projections are
    /// served from the file instead of repacked.
    Q8_0                 = 14,
    /// Also not K-quants: 32 values with a binary16 scale and an additive binary16
    /// minimum. A quantiser writes these where the reduction axis is not a multiple of
    /// 256, so no superblock fits a row.
    Q4_1                 = 15,
    Q5_1                 = 16,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // The weight's bytes are Marlin tiles, not the format its QType names.
    // Residency-replacing repack (design/serve-engine-multiarch.md item 3)
    // rewrites a weight in place and stamps this, which is what makes the
    // replacement safe: a route that cannot serve Marlin tiles no longer
    // matches the weight, so it falls through to its "unsupported weight
    // format" throw at plan time instead of reading the tiles as e4m3 and
    // emitting noise (PATCHES.md #59). The tag lives here, on the weight,
    // rather than in a target, so any architecture inherits it by declaring
    // a compute profile.
    MarlinTiles         = 4,
    // qdata is an [n][k/256] array of GGML superblocks; qhigh and scales are null.
    GgmlBlocks          = 5,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4G64_F16S;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace sinfer
