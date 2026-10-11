#include "api/ops/dflash2.h"

#include "ops/launcher/dflash2.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

void require(bool condition, const char* op, const char* what) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + what); }
}

bool matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns) {
    return tensor.data != nullptr && tensor.dtype == dtype && tensor.ne[0] == rows &&
           tensor.ne[1] == columns && tensor.ne[2] == 1 && tensor.ne[3] == 1 &&
           tensor.is_contiguous();
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

bool aligned16(const void* pointer) { return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0; }

} // namespace

void dflash_conv(const Tensor& input, const Tensor& dynamic, const Tensor& base,
                 std::int32_t stage, std::int32_t width, const Tensor* residual, Tensor& out,
                 cudaStream_t stream) {
    constexpr const char* op = "dflash_conv";
    const std::int32_t hidden  = input.ne[0];
    const std::int32_t columns = input.ne[1];
    require(hidden > 0 && columns > 0 && matrix(input, DType::BF16, hidden, columns), op,
            "input must be a contiguous BF16 [H, columns] matrix");
    require(width > 0 && columns % width == 0, op, "columns must be whole blocks of `width` rows");
    require(stage == 0 || stage == 1, op, "stage is 0 (before the sublayer) or 1 (after it)");
    require(dynamic.ne[0] > 0 && dynamic.ne[0] % 4 == 0 &&
                matrix(dynamic, DType::BF16, dynamic.ne[0], columns), op,
            "dynamic must be a contiguous BF16 [4*G, columns] matrix");
    const std::int32_t groups = dynamic.ne[0] / 4;
    require(hidden % groups == 0 && (hidden / groups) % 8 == 0, op,
            "each channel group must be a whole multiple of 8 channels");
    require(base.data != nullptr && base.dtype == DType::BF16 && base.ne[0] == hidden &&
                base.ne[1] == 2 && base.ne[2] == 2 && base.ne[3] == 1 && base.is_contiguous(), op,
            "base must be a contiguous BF16 [H, 2 taps, 2 stages] tensor");
    require(matrix(out, DType::BF16, hidden, columns), op, "out must match input");
    require(residual == nullptr || matrix(*residual, DType::BF16, hidden, columns), op,
            "residual must match input");
    require(!overlaps(input, out) && !overlaps(dynamic, out) && !overlaps(base, out), op,
            "out must not overlap its convolution's operands");
    require(residual == nullptr || residual->data == out.data || !overlaps(*residual, out), op,
            "residual is either out itself or apart from it");
    require(aligned16(input.data) && aligned16(base.data) && aligned16(out.data) &&
                (residual == nullptr || aligned16(residual->data)), op,
            "operands must be 16-byte aligned");
    detail::dflash_conv_launch(input, dynamic, base, stage, width, residual, out, groups, stream);
}

void dflash_select_top_k(const Tensor& logits, std::int32_t domain, Tensor& ids, Tensor& values,
                         cudaStream_t stream) {
    constexpr const char* op = "dflash_select_top_k";
    const std::int32_t k       = ids.ne[0];
    const std::int32_t columns = ids.ne[1];
    require(k >= 1 && k <= 32, op, "K must be 1..32");
    require(columns > 0 && matrix(logits, DType::BF16, logits.ne[0], columns), op,
            "logits must be a contiguous BF16 [R, columns] matrix");
    require(domain >= 1 && domain <= logits.ne[0], op, "domain must be 1..R");
    require(matrix(ids, DType::I32, k, columns), op, "ids must be a contiguous I32 [K, columns] matrix");
    require(matrix(values, DType::FP32, k, columns), op,
            "values must be a contiguous F32 [K, columns] matrix");
    require(!overlaps(ids, values) && !overlaps(logits, ids) && !overlaps(logits, values), op,
            "outputs must not overlap");
    require(aligned16(logits.data), op, "logits must be 16-byte aligned");
    detail::dflash_select_top_k_launch(logits, domain, ids, values, stream);
}

void dflash_select_walk(const Tensor& anchors, const Tensor& ids, const Tensor& values,
                        const Tensor& hidden, const Tensor& predecessor, const Tensor& successor,
                        std::int32_t positions, Tensor& drafts, cudaStream_t stream) {
    constexpr const char* op = "dflash_select_walk";
    const std::int32_t k       = ids.ne[0];
    const std::int32_t columns = ids.ne[1];
    const std::int32_t dims    = hidden.ne[0];
    require(k >= 1 && k <= 32, op, "K must be 1..32");
    require(positions >= 1 && columns > 0 && columns % positions == 0, op,
            "columns must be whole lanes of `positions` drafts");
    require(matrix(ids, DType::I32, k, columns) && matrix(values, DType::FP32, k, columns), op,
            "ids and values must be contiguous [K, columns] I32 and F32 matrices");
    require(dims >= 32 && dims <= 1024 && dims % 32 == 0 && matrix(hidden, DType::BF16, dims, columns),
            op, "hidden must be a contiguous BF16 [D, columns] matrix, D a multiple of 32 up to 1024");
    require(predecessor.ne[1] > 0 && matrix(predecessor, DType::BF16, dims, predecessor.ne[1]) &&
                matrix(successor, DType::BF16, dims, predecessor.ne[1]), op,
            "the codebooks must be contiguous BF16 [D, V] matrices");
    require(anchors.data != nullptr && anchors.dtype == DType::I32 &&
                anchors.ne[0] == columns / positions && anchors.ne[1] == 1 && anchors.is_contiguous(),
            op, "anchors must be a contiguous I32 [B] vector");
    require(drafts.data != nullptr && drafts.dtype == DType::I32 && drafts.ne[0] == columns &&
                drafts.ne[1] == 1 && drafts.is_contiguous(), op,
            "drafts must be a contiguous I32 [P*B] vector");
    detail::dflash_select_walk_launch(anchors, ids, values, hidden, predecessor, successor,
                                      positions, drafts, stream);
}

} // namespace sinfer::ops
