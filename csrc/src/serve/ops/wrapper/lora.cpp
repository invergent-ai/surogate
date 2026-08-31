// sinfer::ops — lora wrapper: validates the adapter against the projection it is
// being added to, then runs the two BF16 GEMMs through the cuBLASLt route.
// See docs/op-development.md §2.
#include "api/ops/lora.h"

#include "api/ops/residual_add.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <stdexcept>

namespace sinfer::ops {

std::size_t lora_workspace_elements(std::int32_t rank, std::int32_t n,
                                    std::int32_t tokens) noexcept {
    if (rank <= 0 || n <= 0 || tokens <= 0) { return 0; }
    // `A @ x` [rank, T] and then `B @ that` [n, T]. The second buffer exists because
    // the delta is added to the projection's output rather than accumulated into it:
    // the base result must stay intact until the add, and the plain cuBLASLt route
    // this rides on writes rather than accumulates.
    return (static_cast<std::size_t>(rank) + static_cast<std::size_t>(n)) *
           static_cast<std::size_t>(tokens);
}

void lora_prepare(std::int32_t n, std::int32_t k, std::int32_t rank, std::int32_t tokens) {
    if (rank <= 0 || tokens <= 0 || n <= 0 || k <= 0) { return; }
    detail::bf16_cublaslt_prewarm();
    detail::bf16_cublaslt_prepare(rank, k, tokens); // A @ x   -> [rank, T]
    detail::bf16_cublaslt_prepare(n, rank, tokens); // B @ low -> [n, T]
}

void lora_delta(const Tensor& x, const LoraWeights& lora, Tensor& out, Tensor& scratch,
                cudaStream_t stream) {
    if (lora.rank <= 0) { return; }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || scratch.dtype != DType::BF16) {
        throw std::invalid_argument("lora_delta: x/out/scratch must be BF16");
    }
    if (!x.is_contiguous() || !out.is_contiguous() || !scratch.is_contiguous()) {
        throw std::invalid_argument("lora_delta: x/out/scratch must be contiguous");
    }
    if (x.data == nullptr || out.data == nullptr || scratch.data == nullptr) {
        throw std::invalid_argument("lora_delta: x/out/scratch data must be non-null");
    }
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t n      = out.ne[0];
    if (tokens <= 0 || out.ne[1] != tokens) {
        throw std::invalid_argument("lora_delta: x and out must share a token count");
    }
    // The adapter must belong to this projection. An A whose k does not match the
    // activation, or a B whose n does not match the output, is a mis-bound adapter,
    // and applying it anyway would read downstream as a quality regression rather
    // than as the wiring error it is.
    if (lora.a.n != lora.rank || lora.a.k != k) {
        throw std::invalid_argument("lora_delta: A must be [rank, k] of this projection");
    }
    if (lora.b.n != n || lora.b.k != lora.rank) {
        throw std::invalid_argument("lora_delta: B must be [n, rank] of this projection");
    }
    if (lora.a.qtype != QType::BF16_CTRL || lora.b.qtype != QType::BF16_CTRL) {
        throw std::invalid_argument("lora_delta: A and B must be BF16_CTRL");
    }
    if (scratch.numel() <
        static_cast<std::int64_t>(lora_workspace_elements(lora.rank, n, tokens))) {
        throw std::invalid_argument("lora_delta: scratch is smaller than (rank + n) x tokens");
    }

    auto* base = static_cast<std::byte*>(scratch.data);
    Tensor low(base, DType::BF16, {lora.rank, tokens});
    Tensor delta(base + static_cast<std::size_t>(lora.rank) * tokens * sizeof(std::uint16_t),
                 DType::BF16, {n, tokens});

    // PEFT's alpha/r is folded into A at load, so nothing scales here.
    detail::bf16_cublaslt_gemm(lora.a, x, low, stream);
    detail::bf16_cublaslt_gemm(lora.b, low, delta, stream);
    residual_add(delta, out, stream);
}

} // namespace sinfer::ops
