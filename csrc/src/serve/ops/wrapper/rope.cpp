#include "api/ops/rope.h"

#include "ops/launcher/rope.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

constexpr std::int32_t kTextHeadDim = 256;
/// Gemma 4's global attention head, which is twice the family's usual width.
constexpr std::int32_t kWideTextHeadDim = 512;
// The vision tower's head dim: 72 on the Qwen3.6 family and Flash-Next
// (27x1152), 64 on every Qwen3.5 tower (12x768, 24x1024). Both are served.
constexpr std::int32_t kVisionDim   = 72;
constexpr std::int32_t kVisionDim64 = 64;

std::int64_t numel_allow_zero(const Tensor& tensor, const char* label) {
    bool zero      = false;
    std::int64_t n = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) {
            zero = true;
            continue;
        }
        if (n > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        n *= tensor.ne[dim];
    }
    return zero ? 0 : n;
}

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope: ") + label + " must have positive heads");
    }
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

void require_common(const Tensor& positions, int rotary_dim, float theta) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive and even");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope: positions data must be non-null");
    }
}

void require_model_mode(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        const bool supported = (head_dim == kVisionDim && rotary_dim == kVisionDim) ||
                               (head_dim == kVisionDim64 && rotary_dim == kVisionDim64);
        if (!supported) {
            throw std::invalid_argument(
                "rope: 2-D Vision mode requires head_dim=rotary_dim of 72 or 64");
        }
        return;
    }
    // 1-D narrow heads: the DFlash draft head (D128/R128), the QSA indexer's
    // queries (D128/R64) and a 64-wide head rotating in full (D64/R64). All three
    // are served by the generic kernel, which reads its head dim and rotary dim
    // from the call rather than baking them, so the constraint is only that the
    // rotation fits inside the head.
    if (axes == 1 && (head_dim == 128 || head_dim == 64) && rotary_dim <= head_dim) { return; }
    // A 512-wide head rotating one-dimensionally: Gemma 4's global layers. Same generic
    // kernel, which reads its head dim and rotary dim from the call, and the same one
    // constraint -- the rotation fits inside the head.
    if (axes == 1 && head_dim == kWideTextHeadDim && rotary_dim <= head_dim) { return; }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument(
            "rope: Text mode requires D256 or D512, or one-dimensional D128 or D64, with a "
            "rotary dim no wider than the head (head_dim " +
            std::to_string(head_dim) + ", rotary_dim " + std::to_string(rotary_dim) + ", axes " +
            std::to_string(axes) + ")");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

/// How many of the rotation's pairs carry a real angle. The rest are an identity, which is
/// what a proportional rope's trailing zero frequencies are.
void require_active_pairs(int active_pairs, int rotary_dim) {
    if (active_pairs <= 0 || active_pairs > rotary_dim / 2) {
        throw std::invalid_argument("rope: active pairs must be in (0, rotary_dim/2], got " +
                                    std::to_string(active_pairs) + " of " +
                                    std::to_string(rotary_dim / 2));
    }
}

} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream) {
    rope(positions, rotary_dim, rotary_dim / 2, theta, q, k, stream);
}

void rope(const Tensor& positions, int rotary_dim, int active_pairs, float theta, Tensor& q,
          Tensor& k, cudaStream_t stream) {
    require_common(positions, rotary_dim, theta);
    require_active_pairs(active_pairs, rotary_dim);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "q");
    (void)numel_allow_zero(k, "k");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_launch(positions, rotary_dim, active_pairs, theta, q, k, stream);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream) {
    rope(positions, rotary_dim, rotary_dim / 2, theta, x, stream);
}

void rope(const Tensor& positions, int rotary_dim, int active_pairs, float theta, Tensor& x,
          cudaStream_t stream) {
    require_common(positions, rotary_dim, theta);
    require_active_pairs(active_pairs, rotary_dim);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t x_numel  = numel_allow_zero(x, "tensor");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = x.ne[0];
    const std::int32_t heads    = x.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_single_launch(positions, rotary_dim, active_pairs, theta, x, stream);
}

} // namespace sinfer::ops
