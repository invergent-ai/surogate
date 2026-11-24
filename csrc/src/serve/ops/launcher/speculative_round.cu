// Implements: include/sinfer/ops/speculative_round.h
// Match: validated speculative state and BF16 or FP32 verification logits.
// Algorithm assumptions: the shared sampling layout selects either one block
// or a two-launch partial/group pipeline without host reads of device config.
#include "ops/launcher/speculative_round.h"

#include "ops/common/math.h"
#include "ops/kernel/speculative_round.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace sinfer::ops::detail {

void speculative_prepare_verify_inputs_launch(const Tensor& anchors, const Tensor& drafts,
                                              const Tensor& base_positions,
                                              const Tensor& current_extents, Tensor& verify_ids,
                                              Tensor& positions, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(base_positions.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), static_cast<std::int32_t*>(positions.data), k);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_prepare_verify_ids_launch(const Tensor& anchors, const Tensor& drafts,
                                           const Tensor& current_extents, Tensor& verify_ids,
                                           cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    const dim3 grid(static_cast<unsigned int>(div_up(k + 1, kBlock)),
                    static_cast<unsigned int>(batch));
    speculative_prepare_verify_inputs_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(drafts.data), nullptr,
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(verify_ids.data), nullptr, k);
    CUDA_CHECK(cudaGetLastError());
}

bool& speculative_exact_split() {
    // SUROGATE_SPEC_EXACT_SPLIT=0: untruncated speculative rows on the one-CTA wide kernel (read once; tests edit it).
    static bool on = [] {
        const char* raw = std::getenv("SUROGATE_SPEC_EXACT_SPLIT");
        return raw == nullptr || raw[0] == '\0' || std::atoi(raw) != 0;
    }();
    return on;
}

namespace {

// One verification round for logits of element type LogitT (BF16 or FP32). The routes,
// their launch order and the workspace are the same for both; only the logit loads differ.
template <typename LogitT>
void speculative_accept_greedy_drafts_launch_typed(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& current_extents, Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
    Tensor& licensed_counts, Tensor& accepted, std::int32_t token_domain,
    const SamplingConfig* configs, DeviceSpan workspace, cudaStream_t stream,
    std::int32_t draft_count, bool split_routes) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t cols              = draft_count + 1;
    const std::int32_t batch             = drafts.ne[1];
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, cols);
    const SamplingWorkspace scratch = layout.bind(workspace);
    const auto* sorted = sampling_sort_launch(logits, token_domain, configs,
        static_cast<const int32_t*>(drafts.data), static_cast<const int32_t*>(current_extents.data),
        cols, batch, scratch, layout.bytes, stream);
    // Untruncated stochastic rows of a target that opts in (split_routes): the split routes (exact split / coupled)
    // unless switched off; the wide kernel then skips them. Every other target launches exactly M7's kernels. Always
    // the same launches for a shape, so a captured graph keeps one node set. The chunk partials sit after the batch's
    // sampling layouts (speculative_split_partial_row_bytes per row).
    const std::size_t partial_row_bytes = speculative_split_partial_row_bytes(token_domain, cols);
    const bool exact_split = split_routes && partial_row_bytes != 0 && speculative_exact_split();
    auto* const partials   = exact_split ? reinterpret_cast<unsigned long long*>(
                                             static_cast<unsigned char*>(workspace.data) + layout.bytes * batch)
                                         : nullptr;
    speculative_accept_wide_kernel<LogitT><<<batch, kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data), static_cast<const int32_t*>(drafts.data),
        static_cast<const int32_t*>(current_extents.data), static_cast<int32_t*>(lengths.data),
        static_cast<int32_t*>(anchors.data), static_cast<int32_t*>(licensed_tokens.data),
        static_cast<int32_t*>(licensed_counts.data), static_cast<int32_t*>(accepted.data), configs,
        token_domain, physical_rows, draft_count, sorted, layout.bytes, exact_split ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());
    if (exact_split) {
        const std::int32_t chunks = div_up(token_domain, kSamplerFullVocabChunkItems);
        const std::int64_t units  = static_cast<std::int64_t>(chunks) * cols * batch;
        speculative_exact_max_kernel<LogitT>
            <<<sampling_resident_grid(speculative_exact_max_kernel<LogitT>, units), kSamplerBlock, 0, stream>>>(
                static_cast<const LogitT*>(logits.data), static_cast<const std::int32_t*>(drafts.data),
                static_cast<const std::int32_t*>(current_extents.data), configs, token_domain, physical_rows,
                draft_count, batch, partials, partial_row_bytes, chunks);
        CUDA_CHECK(cudaGetLastError());
        speculative_exact_partial_kernel<LogitT>
            <<<sampling_resident_grid(speculative_exact_partial_kernel<LogitT>, units), kSamplerBlock, 0, stream>>>(
                static_cast<const LogitT*>(logits.data), static_cast<const std::int32_t*>(drafts.data),
                static_cast<const std::int32_t*>(current_extents.data),
                static_cast<const std::int32_t*>(lengths.data), configs, token_domain, physical_rows, draft_count,
                batch, partials, partial_row_bytes, chunks);
        CUDA_CHECK(cudaGetLastError());
        // Coupled rows (cfg.speculative_coupled): the ordinary sampler's draw per column, greedy accept.
        speculative_coupled_partial_kernel<LogitT>
            <<<sampling_resident_grid(speculative_coupled_partial_kernel<LogitT>, units), kSamplerBlock, 0, stream>>>(
                static_cast<const LogitT*>(logits.data), static_cast<const std::int32_t*>(drafts.data),
                static_cast<const std::int32_t*>(current_extents.data),
                static_cast<const std::int32_t*>(lengths.data), configs, token_domain, physical_rows, draft_count,
                batch, partials, partial_row_bytes, chunks, -std::numeric_limits<float>::infinity());
        CUDA_CHECK(cudaGetLastError());
        speculative_coupled_finalize_kernel<LogitT><<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(drafts.data), static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data), static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, draft_count,
            partials, partial_row_bytes, chunks);
        CUDA_CHECK(cudaGetLastError());
        speculative_exact_finalize_kernel<LogitT><<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const LogitT*>(logits.data), static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data), static_cast<std::int32_t*>(lengths.data),
            static_cast<std::int32_t*>(anchors.data), static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data), configs,
            token_domain, physical_rows, draft_count, partials, partial_row_bytes, chunks);
        CUDA_CHECK(cudaGetLastError());
    }
    if (!layout.multiblock) {
        speculative_accept_greedy_drafts_kernel<LogitT><<<batch, kSamplerBlock, 0, stream>>>(
            static_cast<const std::int32_t*>(target_tokens.data),
            static_cast<const LogitT*>(logits.data),
            static_cast<const std::int32_t*>(drafts.data),
            static_cast<const std::int32_t*>(current_extents.data),
            static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
            static_cast<std::int32_t*>(licensed_tokens.data),
            static_cast<std::int32_t*>(licensed_counts.data),
            static_cast<std::int32_t*>(accepted.data), configs, token_domain, physical_rows,
            draft_count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const unsigned int partial_grid = sampling_resident_grid(
        speculative_sampling_partial_topk_kernel<LogitT>,
        static_cast<std::int64_t>(partial_blocks) * cols * batch);
    speculative_sampling_partial_topk_kernel<LogitT><<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data), configs, token_domain,
        physical_rows, cols, draft_count, batch, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
    const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                  static_cast<unsigned int>(cols),
                                  static_cast<unsigned int>(batch));
    speculative_sampling_group_finalize_kernel<<<batched_group_grid, kSamplerGroupBlock, 0,
                                                 stream>>>(
        static_cast<const std::int32_t*>(target_tokens.data),
        static_cast<const std::int32_t*>(drafts.data),
        static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(lengths.data), static_cast<std::int32_t*>(anchors.data),
        static_cast<std::int32_t*>(licensed_tokens.data),
        static_cast<std::int32_t*>(licensed_counts.data), static_cast<std::int32_t*>(accepted.data),
        configs, token_domain, cols, partial_blocks, groups, scratch, layout.bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void speculative_accept_greedy_drafts_launch(const Tensor& target_tokens, const Tensor& logits,
                                             const Tensor& drafts, const Tensor& current_extents,
                                             Tensor& lengths, Tensor& anchors,
                                             Tensor& licensed_tokens, Tensor& licensed_counts,
                                             Tensor& accepted, std::int32_t token_domain,
                                             const SamplingConfig* configs, DeviceSpan workspace,
                                             cudaStream_t stream, std::int32_t draft_count, bool split_routes) {
    if (logits.dtype == DType::FP32) {
        speculative_accept_greedy_drafts_launch_typed<float>(
            target_tokens, logits, drafts, current_extents, lengths, anchors, licensed_tokens,
            licensed_counts, accepted, token_domain, configs, workspace, stream, draft_count, split_routes);
    } else {
        speculative_accept_greedy_drafts_launch_typed<__nv_bfloat16>(
            target_tokens, logits, drafts, current_extents, lengths, anchors, licensed_tokens,
            licensed_counts, accepted, token_domain, configs, workspace, stream, draft_count, split_routes);
    }
}

void speculative_truncate_at_phase_switch_launch(const Tensor& drafts, Tensor& current_extents,
                                                 const SamplingConfig* configs, cudaStream_t stream) {
    constexpr int kBlock = 32;
    const int k          = drafts.ne[0];
    const int batch      = drafts.ne[1];
    speculative_truncate_at_phase_switch_kernel<<<static_cast<unsigned int>(div_up(batch, kBlock)), kBlock, 0,
                                                  stream>>>(
        static_cast<const std::int32_t*>(drafts.data), static_cast<std::int32_t*>(current_extents.data), configs,
        k, batch);
    CUDA_CHECK(cudaGetLastError());
}

void speculative_select_accepted_hidden_launch(const Tensor& hidden, const Tensor& selectors,
                                               Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int rows       = hidden.ne[0];
    const int batch      = hidden.ne[2];
    const dim3 grid(static_cast<unsigned int>(std::max(1, div_up(rows, kBlock))),
                    static_cast<unsigned int>(batch));
    speculative_select_accepted_hidden_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const std::int32_t*>(selectors.data), static_cast<__nv_bfloat16*>(out.data),
        rows, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void proposal_remap_token_ids_launch(Tensor& proposal_tokens, const std::int32_t* id_map,
                                     std::int32_t n, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const int count      = proposal_tokens.ne[0];
    const int grid       = std::max(1, div_up(count, kBlock));
    proposal_remap_token_ids_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<std::int32_t*>(proposal_tokens.data), count, id_map, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops::detail
