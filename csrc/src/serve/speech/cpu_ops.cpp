// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
#include "cpu_ops.h"
#include <ATen/Parallel.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <algorithm>

namespace sinfer::speech::cpu {
at::Tensor PackedLinear::run(const at::Tensor& input, const at::Tensor& weight,
                             const std::optional<at::Tensor>& bias) {
    static bool available =
        at::hasMKL() && at::hasMKLDNN() &&
        c10::Dispatcher::singleton().findSchema({"mkl::_mkl_linear", ""}).has_value() &&
        c10::Dispatcher::singleton()
            .findSchema({"mkl::_mkl_reorder_linear_weight", ""})
            .has_value();
    const int64_t rows = weight.size(1) ? input.numel() / weight.size(1) : 0;
    const int threads  = at::get_num_threads();
    // Small-work partitioning can select different MKL reductions. Keep the
    // original kernel below the validated frame-per-thread cutoff.
    if (!available || !input.device().is_cpu() || input.scalar_type() != at::kFloat ||
        !input.is_contiguous() || rows < std::max<int64_t>(12, 2 * threads))
        return at::linear(input, weight, bias);
    static auto reorder =
        c10::Dispatcher::singleton().findSchemaOrThrow("mkl::_mkl_reorder_linear_weight", "");
    static auto compute = c10::Dispatcher::singleton().findSchemaOrThrow("mkl::_mkl_linear", "");
    if (!packed_.defined() || rows != rows_ || threads != threads_ || !original_.is_same(weight)) {
        // Packing every new recording length costs more than it saves. Keep the
        // hot shape through occasional misses (for example a stream finalizer),
        // and only replace it after another shape recurs.
        if (rows != pending_rows_ || threads != pending_threads_ ||
            !pending_original_.is_same(weight)) {
            pending_rows_     = rows;
            pending_threads_  = threads;
            pending_original_ = weight;
            observations_     = 0;
        }
        if (++observations_ < 3) return at::linear(input, weight, bias);
        std::vector<c10::IValue> stack{weight, rows};
        reorder.callBoxed(&stack);
        packed_   = std::move(stack.back()).toTensor();
        original_ = weight;
        rows_     = rows;
        threads_  = threads;
    }
    std::vector<c10::IValue> stack{input, packed_, weight,
                                   bias ? c10::IValue(*bias) : c10::IValue(), rows};
    compute.callBoxed(&stack);
    return std::move(stack.back()).toTensor();
}

at::Tensor relative_shift(const at::Tensor& scores, int64_t keys) {
    TORCH_CHECK(scores.dim() == 4 && keys > 0 && scores.size(2) > 0 && scores.size(2) <= keys &&
                    scores.size(3) == 2 * keys - 1 && scores.is_contiguous(),
                "Relative attention expects contiguous [batch, heads, queries, 2*keys-1] scores");
    const int64_t queries = scores.size(2);
    // The retained columns never touch the zero pad in the reference shift.
    // Row q starts at q*(2*keys-2) + queries-1 in the original matrix.
    return scores.as_strided(
        {scores.size(0), scores.size(1), queries, keys},
        {scores.stride(0), scores.stride(1), scores.stride(2) - scores.stride(3), scores.stride(3)},
        scores.storage_offset() + (queries - 1) * scores.stride(3));
}
} // namespace sinfer::speech::cpu
