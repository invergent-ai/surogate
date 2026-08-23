#include "ninfer/ops/prepare_masked_block.h"

#include "ops/launcher/prepare_masked_block.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_i32_vector(const Tensor& tensor, std::int32_t size, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != size || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 vector");
    }
}

void require_i32_matrix(const Tensor& tensor, std::int32_t width, std::int32_t batch,
                        const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] != width || tensor.ne[1] != batch ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument("prepare_masked_block: " + std::string(name) +
                                    " must be a contiguous I32 matrix");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

} // namespace

void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          cudaStream_t stream) {
    const std::int32_t block_size = ids.ne[0];
    const std::int32_t batch_size = ids.ne[1];
    if (block_size < 1 || block_size > 16) {
        throw std::invalid_argument("prepare_masked_block: W must be 1..16");
    }
    if (batch_size < 1 || batch_size > 8) {
        throw std::invalid_argument("prepare_masked_block: B must be 1..8");
    }
    if (mask_id < 0) {
        throw std::invalid_argument("prepare_masked_block: mask_id must be nonnegative");
    }
    require_i32_vector(anchors, batch_size, "anchors");
    require_i32_vector(lengths, batch_size, "lengths");
    require_i32_vector(valid_columns, batch_size, "valid_columns");
    require_i32_matrix(ids, block_size, batch_size, "ids");
    require_i32_matrix(positions, block_size, batch_size, "positions");
    if (overlaps(ids, positions) || overlaps(anchors, ids) || overlaps(anchors, positions) ||
        overlaps(lengths, ids) || overlaps(lengths, positions) || overlaps(valid_columns, ids) ||
        overlaps(valid_columns, positions)) {
        throw std::invalid_argument("prepare_masked_block: inputs and outputs must not overlap");
    }
    detail::prepare_masked_block_launch(anchors, lengths, valid_columns, mask_id, ids, positions,
                                        block_size, stream);
}

} // namespace ninfer::ops
