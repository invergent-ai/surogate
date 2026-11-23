// Implements: include/sinfer/ops/sampling.h
// Match: validated contiguous BF16-or-FP32 logits, I32 tensors and a shared-layout workspace.
// Algorithm assumptions: launcher and kernels use sampler_multiblock_ok() from
// the same layout authority, so exactly one finite route owns each shape.
#include "ops/launcher/sampling.h"

#include "ops/common/math.h"
#include "ops/kernel/sampling.cuh"
#include "core/device.h"

#include <cstdlib>
#include <limits>

namespace sinfer::ops::detail {
namespace {

template <typename LogitT>
void sampling_update_greedy_targets_launch_typed(const Tensor& logits, Tensor& targets,
                                                 std::int32_t token_domain,
                                                 const SamplingConfig* configs,
                                                 cudaStream_t stream) {
    sampling_update_greedy_targets_kernel<LogitT><<<dim3(logits.ne[1], logits.ne[2]), kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data), static_cast<std::int32_t*>(targets.data),
        configs, token_domain, logits.ne[0], logits.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sampling_update_greedy_targets_launch(const Tensor& logits, Tensor& targets,
                                           std::int32_t token_domain, const SamplingConfig* configs,
                                           cudaStream_t stream) {
    if (logits.dtype == DType::FP32) {
        sampling_update_greedy_targets_launch_typed<float>(logits, targets, token_domain, configs,
                                                           stream);
    } else {
        sampling_update_greedy_targets_launch_typed<__nv_bfloat16>(logits, targets, token_domain,
                                                                   configs, stream);
    }
}

bool& sampling_full_vocab_split() {
    // SUROGATE_SAMPLING_FULL_VOCAB_SPLIT=0: the one-CTA-per-row full-vocabulary draw (read once; tests edit it).
    static bool on = [] {
        const char* raw = std::getenv("SUROGATE_SAMPLING_FULL_VOCAB_SPLIT");
        return raw == nullptr || raw[0] == '\0' || std::atoi(raw) != 0;
    }();
    return on;
}

std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain, std::int32_t columns) {
    return make_sampling_workspace_layout(token_domain, columns).bytes;
}

namespace {

// One sampling call for logits of element type LogitT (BF16 or FP32). The routes, their
// launch order and the workspace are the same for both; only the logit loads differ.
template <typename LogitT>
void sample_batch_launch_typed(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                               const SamplingConfig* configs, const Tensor& logical_positions,
                               std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows     = logits.ne[0];
    const std::int32_t batch             = logits.ne[1];
    const auto* positions                = static_cast<const std::int32_t*>(logical_positions.data);
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(token_domain, batch);
    const SamplingWorkspace scratch = layout.bind(workspace);
    const auto* sorted = sampling_sort_launch(logits, token_domain, configs, nullptr, nullptr,
        1, batch, scratch, static_cast<size_t>(token_domain) * sizeof(unsigned long long), stream);
    sampling_wide_kernel<LogitT><<<batch, kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data), sorted, static_cast<int32_t*>(out.data), configs,
        positions, purpose, token_domain, physical_rows);
    CUDA_CHECK(cudaGetLastError());
    // Rows that asked for no truncation are drawn from the whole vocabulary. Which
    // rows those are is a property of the device-resident configs, so the choice is
    // made per column inside the kernels rather than here: this launch handles the
    // untruncated columns and returns immediately for the rest, and the candidate
    // routes below do the reverse. Always launched, so a captured graph records the
    // same node set whatever a later round happens to ask for.
    if (scratch.full_vocab_partials != nullptr && sampling_full_vocab_split()) {
        // The same draw, a row cut into chunks of one CTA each (sampling_full_vocab_partial_kernel).
        const std::int32_t chunks = div_up(token_domain, kSamplerFullVocabChunkItems);
        sampling_full_vocab_partial_kernel<LogitT>
            <<<dim3(static_cast<unsigned int>(chunks), static_cast<unsigned int>(batch)), kSamplerBlock, 0, stream>>>(
                static_cast<const LogitT*>(logits.data), configs, positions, purpose, token_domain, physical_rows,
                scratch.full_vocab_partials, chunks, -std::numeric_limits<float>::infinity());
        CUDA_CHECK(cudaGetLastError());
        sampling_full_vocab_finalize_kernel<LogitT><<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const LogitT*>(logits.data), static_cast<std::int32_t*>(out.data), configs, positions,
            purpose, token_domain, physical_rows, scratch.full_vocab_partials, chunks);
        CUDA_CHECK(cudaGetLastError());
    } else {
        sampling_full_vocab_kernel<LogitT><<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const LogitT*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    }
    if (!layout.multiblock) {
        sample_row_kernel<LogitT><<<static_cast<unsigned int>(batch), kSamplerBlock, 0, stream>>>(
            static_cast<const LogitT*>(logits.data), static_cast<std::int32_t*>(out.data),
            configs, positions, purpose, token_domain, physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    const unsigned int partial_grid = sampling_resident_grid(
        sampling_partial_topk_kernel<LogitT>, static_cast<std::int64_t>(partial_blocks) * batch);
    sampling_partial_topk_kernel<LogitT><<<partial_grid, kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data), configs, token_domain, physical_rows,
        batch, scratch);
    CUDA_CHECK(cudaGetLastError());
    const std::int32_t greedy_chunks = div_up(token_domain, kSamplerGreedyChunkItems);
    sampling_greedy_kernel<LogitT><<<dim3(static_cast<unsigned int>(greedy_chunks),
                                          static_cast<unsigned int>(batch)), kSamplerBlock, 0, stream>>>(
        static_cast<const LogitT*>(logits.data), static_cast<std::int32_t*>(out.data),
        configs, token_domain, physical_rows, greedy_chunks, scratch);
    CUDA_CHECK(cudaGetLastError());
    const dim3 group_grid(static_cast<unsigned int>(groups), static_cast<unsigned int>(batch));
    sampling_group_finalize_sample_kernel<<<group_grid, kSamplerGroupBlock, 0, stream>>>(
        static_cast<std::int32_t*>(out.data), configs, positions, purpose, token_domain,
        partial_blocks, groups, scratch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream) {
    if (logits.dtype == DType::FP32) {
        sample_batch_launch_typed<float>(logits, out, token_domain, configs, logical_positions,
                                         purpose, workspace, stream);
    } else {
        sample_batch_launch_typed<__nv_bfloat16>(logits, out, token_domain, configs,
                                                 logical_positions, purpose, workspace, stream);
    }
}

} // namespace sinfer::ops::detail
