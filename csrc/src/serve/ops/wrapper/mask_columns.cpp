// sinfer::ops - mask_columns wrapper: public api validation and launcher dispatch.
#include "api/ops/mask_columns.h"

#include "ops/launcher/mask_columns.h"

#include <stdexcept>

namespace sinfer::ops {

void mask_columns_zero(Tensor& matrix, const Tensor& valid_columns, cudaStream_t stream) {
    if (matrix.dtype != DType::FP32 && matrix.dtype != DType::BF16) {
        throw std::invalid_argument("mask_columns_zero: matrix must be FP32 or BF16");
    }
    if (matrix.ne[0] <= 0 || matrix.ne[1] <= 0 || matrix.ne[2] != 1 || matrix.ne[3] != 1) {
        throw std::invalid_argument("mask_columns_zero: matrix must have shape [C,T]");
    }
    if (!matrix.is_contiguous() || matrix.data == nullptr) {
        throw std::invalid_argument("mask_columns_zero: matrix must be contiguous and non-null");
    }
    if (valid_columns.dtype != DType::I32 || valid_columns.ne[0] != 1 ||
        valid_columns.ne[1] != 1 || valid_columns.ne[2] != 1 || valid_columns.ne[3] != 1 ||
        valid_columns.data == nullptr) {
        throw std::invalid_argument("mask_columns_zero: valid_columns must be a device I32 scalar");
    }
    detail::mask_columns_zero_launch(matrix, valid_columns, stream);
}

} // namespace sinfer::ops
