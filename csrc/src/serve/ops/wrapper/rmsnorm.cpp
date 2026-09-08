// sinfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "api/ops/rmsnorm.h"
#include "api/ops/gated_rmsnorm.h"

#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                  const Tensor* z, Tensor& out, cudaStream_t stream,
                  GatedRmsGate gate = GatedRmsGate::Silu, bool accumulate = false) {
    const bool fp32_input = x.dtype == DType::FP32 && z == nullptr && !accumulate;
    if ((!fp32_input && x.dtype != DType::BF16) || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    (void)numel_allow_zero(weight, "weight");
    require_same_shape(x, out, "out");
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "z");
        require_same_shape(x, *z, "z");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    if (fp32_input) {
        detail::rmsnorm_fp32_launch(x, weight, eps, unit_offset, out, stream);
        return;
    }
    if (accumulate) {
        if (x.data == out.data) {
            throw std::invalid_argument("rmsnorm_add: x must not alias out");
        }
        detail::rmsnorm_add_launch(x, weight, eps, unit_offset, out, stream);
        return;
    }
    detail::rmsnorm_launch(x, weight, eps, unit_offset, z, gate, out, stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, out, stream);
}

void rmsnorm_unweighted(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    // The same validation as the weighted form minus everything about the weight, because
    // there is none: no dtype to check, no width to match, no pointer to require.
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("rmsnorm: x/out must be BF16");
    }
    if (!(eps > 0.0F) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }
    require_same_shape(x, out, "out");
    const std::int64_t total = numel_allow_zero(x, "x");
    if (total == 0) { return; }
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("rmsnorm: ne[0] must be positive"); }
    if (total / d > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("rmsnorm: x/out must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("rmsnorm: x/out data must be non-null");
    }
    if (x.data == out.data) { throw std::invalid_argument("rmsnorm: x must not alias out"); }
    detail::rmsnorm_unweighted_launch(x, eps, out, stream);
}

void rmsnorm_add(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
                 cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, out, stream, GatedRmsGate::Silu,
                 /*accumulate=*/true);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, false, &z, out, stream);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                   GatedRmsGate gate, Tensor& out, cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, false, &z, out, stream, gate);
}

} // namespace sinfer::ops
