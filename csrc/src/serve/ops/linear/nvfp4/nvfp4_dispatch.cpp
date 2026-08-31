#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

enum class Nvfp4LinearRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4LinearRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                               LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || !is_nvfp4_linear_problem(output_rows, input_rows)) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    }
    // Shapes outside the registered geometries have no in-house A16 kernel either (the gemv and
    // small-T paths are templated on Geometry), so they take W4A4 at every width - quantised
    // activations throughout, which is what the NVFP4 exports assume anyway (#82).
    // One token on a hidden-2560 shape: the decode GEMV, which is instantiated for that family
    // and nothing wider (see Nvfp4GemvOnlyProblem). Every other width of a generic shape stays
    // on cuBLASLt, so the served wide rounds do not move.
    if (tokens == 1 && is_nvfp4_gemv_only_problem(output_rows, input_rows)) {
        return Nvfp4LinearRoute::A16;
    }
    if (is_nvfp4_generic_problem(output_rows, input_rows)) { return Nvfp4LinearRoute::W4A4; }
    if (policy == LinearPolicy::A16Only) { return Nvfp4LinearRoute::A16; }

    switch (resolve_nvfp4_problem(output_rows, input_rows)) {
    case Nvfp4Problem::AttnInput:
        return tokens >= 4 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::GdnInput:
        return Nvfp4LinearRoute::W4A4;
    case Nvfp4Problem::MlpGateUp:
        return tokens >= 5 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::Residual17408:
        return tokens >= 8 ? Nvfp4LinearRoute::W4A4 : Nvfp4LinearRoute::A16;
    }
    throw std::logic_error("unreachable NVFP4 linear problem");
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            launch_nvfp4_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_nvfp4_small_t(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                  LinearPolicy policy, std::int32_t min_tokens,
                                                  std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (!is_nvfp4_linear_problem(weight.n, weight.k) || x.ne[1] <= 0) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }

    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_w4a4(x, weight, out, scratch, stream);
}

} // namespace sinfer::ops::detail
