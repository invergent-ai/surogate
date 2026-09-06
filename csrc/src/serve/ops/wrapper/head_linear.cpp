// sinfer::ops -- head_linear wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "api/ops/head_linear.h"

#include "ops/launcher/head_linear.h"

#include <stdexcept>
#include <string>

namespace sinfer::ops {

void head_linear(const Tensor& x, const Weight& w, std::int32_t heads, float out_scale,
                 Tensor& out, cudaStream_t stream) {
    if (heads <= 0) { throw std::invalid_argument("head_linear: heads must be positive"); }
    if (w.layout != QuantLayout::RowSplit || w.qtype != QType::W8G32_F16S || w.qdata == nullptr ||
        w.scales == nullptr || w.ndim != 2) {
        throw std::invalid_argument("head_linear: w must be a W8G32_F16S row-split weight");
    }
    if (w.n % heads != 0) {
        throw std::invalid_argument("head_linear: the weight's " + std::to_string(w.n) +
                                    " rows are not " + std::to_string(heads) + " heads' worth");
    }
    const std::int32_t n = w.n / heads;
    const std::int32_t k = w.k;
    if (k <= 0 || k % 32 != 0 || w.padded_shape[1] < k) {
        throw std::invalid_argument("head_linear: k must be a positive multiple of 32");
    }
    // The activation tile is staged in shared memory, eight columns of k at a time.
    if (static_cast<std::size_t>(k) * 8 * 2 > 48 * 1024) {
        throw std::invalid_argument("head_linear: k is wider than the staged tile admits");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("head_linear: x/out must be BF16");
    }
    if (x.ne[0] != static_cast<std::int64_t>(heads) * k || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument("head_linear: x must be contiguous [heads*k, T]");
    }
    if (out.ne[0] != w.n || out.ne[1] != x.ne[1] || out.ne[2] != 1 || out.ne[3] != 1 ||
        !out.is_contiguous() || out.data == nullptr) {
        throw std::invalid_argument("head_linear: out must be contiguous [heads*n, T]");
    }
    if (x.ne[1] <= 0) { throw std::invalid_argument("head_linear: T must be positive"); }
    detail::head_linear_launch(x, w, heads, n, k, out_scale, out, stream);
}

} // namespace sinfer::ops
