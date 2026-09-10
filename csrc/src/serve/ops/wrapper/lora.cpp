// sinfer::ops — lora wrapper: validates the adapter against the projection it is
// being added to, then runs the two BF16 GEMMs through the cuBLASLt route.
// See docs/op-development.md §2.
#include "api/ops/lora.h"

#include "api/ops/residual_add.h"
#include "ops/launcher/lora_batched.h"
#include "ops/launcher/lora_fused.h"
#include "ops/kernel/lora_fused_limits.h"
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

std::size_t lora_batched_workspace_elements(std::int32_t rank, std::int32_t tokens) noexcept {
    if (rank <= 0 || tokens <= 0) { return 0; }
    return static_cast<std::size_t>(rank) * static_cast<std::size_t>(tokens);
}

void lora_delta_batched(const Tensor& x, const LoraBank& bank, const Tensor& ids,
                        const std::int32_t* uniform_slot, Tensor& out, Tensor& scratch,
                        cudaStream_t stream) {
    if (bank.rank <= 0 || bank.a == nullptr || bank.b == nullptr) { return; }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || scratch.dtype != DType::BF16) {
        throw std::invalid_argument("lora_delta_batched: x/out/scratch must be BF16");
    }
    // Either a per-token vector or one slot for the whole round; a prefill chunk
    // is the latter, since all of its columns belong to one request.
    const bool per_token = ids.data != nullptr;
    if (per_token && ids.dtype != DType::I32) {
        throw std::invalid_argument("lora_delta_batched: ids must be a device I32 tensor");
    }
    if (!per_token && uniform_slot == nullptr) { return; }
    if (!x.is_contiguous() || !out.is_contiguous() || !scratch.is_contiguous()) {
        throw std::invalid_argument("lora_delta_batched: x/out/scratch must be contiguous");
    }
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || out.ne[1] != tokens || (per_token && ids.numel() < tokens)) {
        throw std::invalid_argument("lora_delta_batched: ids must cover the round's tokens");
    }
    if (x.ne[0] != bank.k || out.ne[0] != bank.n) {
        throw std::invalid_argument("lora_delta_batched: bank does not match this projection");
    }
    if (scratch.numel() <
        static_cast<std::int64_t>(lora_batched_workspace_elements(bank.rank, tokens))) {
        throw std::invalid_argument("lora_delta_batched: scratch is smaller than rank x tokens");
    }

    Tensor low(scratch.data, DType::BF16, {bank.rank, tokens});
    const LoraBank* banks[]{&bank};
    Tensor* outputs[]{&out};
    detail::lora_split_delta_launch(x, banks, outputs, 1, ids, uniform_slot, low, stream);
}

void lora_delta_fused(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                      std::int32_t pair_count, const Tensor& ids,
                      const std::int32_t* uniform_slot, Tensor& scratch, cudaStream_t stream) {
    if (pair_count <= 0 || pair_count > kLoraFusedPairLimit) {
        throw std::invalid_argument("lora_delta_fused: 1..3 pairs");
    }
    const bool per_token = ids.data != nullptr;
    if (per_token && ids.dtype != DType::I32) {
        throw std::invalid_argument("lora_delta_fused: ids must be a device I32 tensor");
    }
    if (!per_token && uniform_slot == nullptr) { return; }
    if (x.dtype != DType::BF16 || !x.is_contiguous()) {
        throw std::invalid_argument("lora_delta_fused: x must be contiguous BF16");
    }
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0 || (per_token && ids.numel() < tokens)) {
        throw std::invalid_argument("lora_delta_fused: ids must cover the round's tokens");
    }
    for (std::int32_t p = 0; p < pair_count; ++p) {
        const LoraBank* bank = banks[p];
        Tensor* out          = outs[p];
        if (bank == nullptr || bank->rank <= 0 || bank->a == nullptr || bank->b == nullptr) {
            throw std::invalid_argument("lora_delta_fused: every pair needs a live bank");
        }
        if (out->dtype != DType::BF16 || out->nb[0] != 2 || out->nb[1] < out->ne[0] * 2 || out->ne[1] != tokens) {
            throw std::invalid_argument("lora_delta_fused: outputs must be contiguous BF16 [n, T]");
        }
        // One x for every pair is the point of the fusion, so one k too.
        if (bank->k != x.ne[0] || out->ne[0] != bank->n) {
            throw std::invalid_argument("lora_delta_fused: bank does not match this projection");
        }
    }
    // One launch when the recomputation is cheap, two when it is not. Every
    // stage-2 block of the one-launch kernel redoes stage 1, so its extra A
    // traffic is tokens x blocks x total_rank x k -- negligible for a narrow
    // site at decode, gigabytes for a wide site under a prefill chunk. The
    // threshold is where the measured curves crossed on the 0.8b/4b geometries.
    std::int64_t total_rows = 0, total_rank = 0;
    for (std::int32_t p = 0; p < pair_count; ++p) {
        total_rows += banks[p]->n;
        total_rank += banks[p]->rank;
    }
    const std::int64_t blocks = (total_rows + 127) / 128;
    const std::int64_t redundant_bytes =
        static_cast<std::int64_t>(tokens) * blocks * total_rank * x.ne[0] * 2;
    constexpr std::int64_t kRedundancyBudget = 2LL << 20;
    bool fused_rank = true;
    for (int p = 0; p < pair_count; ++p) { fused_rank &= banks[p]->rank <= kLoraFusedRankLimit; }
    if (fused_rank && redundant_bytes <= kRedundancyBudget) {
        detail::lora_fused_delta_launch(x, banks, outs, pair_count, ids, uniform_slot, stream);
        return;
    }
    if (scratch.dtype != DType::BF16 || !scratch.is_contiguous() ||
        scratch.numel() < total_rank * tokens) {
        throw std::invalid_argument(
            "lora_delta_fused: this site needs scratch of at least total_rank x tokens");
    }
    detail::lora_split_delta_launch(x, banks, outs, pair_count, ids, uniform_slot, scratch,
                                    stream);
}

} // namespace sinfer::ops
