#include "api/ops/qk_norm_rope.h"

#include "ops/launcher/qk_norm_rope.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace {

bool enabled() {
    static const bool on = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_QK_NORM_ROPE");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return on;
}

void require_heads(const Tensor* x, const Tensor* out, const Tensor* gain, std::int32_t head_dim,
                   std::int32_t tokens, const char* label) {
    if (x == nullptr || out == nullptr || gain == nullptr) {
        throw std::invalid_argument(std::string("qk_norm_rope: missing ") + label + " operand");
    }
    if (x->dtype != DType::BF16 || out->dtype != DType::BF16 || gain->dtype != DType::BF16) {
        throw std::invalid_argument(std::string("qk_norm_rope: ") + label + " must be BF16");
    }
    if (x->ne[0] != head_dim || x->ne[2] != tokens || x->ne[3] != 1 || out->ne[0] != x->ne[0] ||
        out->ne[1] != x->ne[1] || out->ne[2] != x->ne[2] || out->ne[3] != 1 || x->ne[1] <= 0) {
        throw std::invalid_argument(std::string("qk_norm_rope: invalid ") + label + " shape");
    }
    if (gain->ne[0] != head_dim || gain->ne[1] != 1 || gain->ne[2] != 1 || gain->ne[3] != 1) {
        throw std::invalid_argument(std::string("qk_norm_rope: ") + label +
                                    " gain must be 1-D with the head dim");
    }
    if (!x->is_contiguous() || !out->is_contiguous() || !gain->is_contiguous()) {
        throw std::invalid_argument(std::string("qk_norm_rope: ") + label + " must be contiguous");
    }
    if (tokens > 0 && (x->data == nullptr || out->data == nullptr || gain->data == nullptr)) {
        throw std::invalid_argument(std::string("qk_norm_rope: null ") + label + " storage");
    }
}

} // namespace

bool qk_norm_rope(const QkNormRope& args, cudaStream_t stream) {
    if (!enabled()) { return false; }
    if (args.q == nullptr || args.positions == nullptr) {
        throw std::invalid_argument("qk_norm_rope: missing queries or positions");
    }
    const std::int32_t head_dim = args.q->ne[0];
    const std::int32_t tokens   = args.q->ne[2];
    require_heads(args.q, args.q_out, args.q_norm, head_dim, tokens, "query");
    if (args.k != nullptr) {
        require_heads(args.k, args.k_out, args.k_norm, head_dim, tokens, "key");
        if (args.k->ne[1] != args.k_heads) {
            throw std::invalid_argument("qk_norm_rope: key heads disagree with k_heads");
        }
    }
    if (args.k_heads <= 0) { throw std::invalid_argument("qk_norm_rope: k_heads must be positive"); }
    if (!(args.eps > 0.0F) || !std::isfinite(args.eps)) {
        throw std::invalid_argument("qk_norm_rope: eps must be positive and finite");
    }
    const Tensor& positions = *args.positions;
    if (positions.dtype != DType::I32 || positions.ne[0] != tokens || positions.ne[2] != 1 ||
        positions.ne[3] != 1 || positions.ne[1] < 1 || positions.ne[1] > 3 ||
        !positions.is_contiguous() || (tokens > 0 && positions.data == nullptr)) {
        throw std::invalid_argument("qk_norm_rope: positions must be contiguous I32 [T], [T,2] or [T,3]");
    }
    if (!(args.theta > 0.0F) || !std::isfinite(args.theta) || args.rotary_dim <= 0 ||
        (args.rotary_dim & 1) != 0) {
        throw std::invalid_argument("qk_norm_rope: theta must be positive and rotary_dim even");
    }
    if (args.table != nullptr) {
        const Tensor& table = *args.table;
        if (table.dtype != DType::FP32 || table.ne[0] != 2 || table.ne[1] != args.rotary_dim / 2 ||
            table.ne[2] < 1 || table.ne[3] != 1 || !table.is_contiguous() || table.data == nullptr) {
            throw std::invalid_argument("qk_norm_rope: table must be a contiguous FP32 [2, rotary_dim/2, positions] rope_table");
        }
        if (args.sections[0] != 0 || positions.ne[1] != 1) {
            throw std::invalid_argument("qk_norm_rope: a table rotation takes [T] positions and no sections");
        }
    }
    QkNormRope launch = args;
    if (args.sections[0] != 0) {
        const int pairs = args.rotary_dim / 2;
        if (args.sections[1] <= 0 || args.sections[2] <= 0 ||
            args.sections[0] + args.sections[1] + args.sections[2] != pairs ||
            args.sections[1] > (pairs + 1) / 3 || args.sections[2] > pairs / 3 ||
            positions.ne[1] != 3) {
            throw std::invalid_argument("qk_norm_rope: invalid temporal/height/width sections");
        }
        // rope_interleaved rotates every pair.
        launch.active_pairs = pairs;
    } else if (args.active_pairs <= 0 || args.active_pairs > args.rotary_dim / 2) {
        throw std::invalid_argument("qk_norm_rope: active pairs must be in (0, rotary_dim/2]");
    }
    return detail::qk_norm_rope_launch(launch, stream);
}

} // namespace sinfer::ops
