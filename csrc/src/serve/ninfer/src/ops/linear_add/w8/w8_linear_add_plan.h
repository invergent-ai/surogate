#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class W8LinearAddScheduleId {
    DecodeR16,
    SplitKMmaExactT,
    MediumSplitK,
    SimtR8C4,
    MmaR32C64,
    MmaR32C80,
    MmaR32C96,
    MmaR32C128,
    MmaR48C64,
    MmaR48C96,
    MmaR48C112,
    MmaR48C128,
    MmaR64C96,
    MmaR64C112,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
};

struct W8LinearAddPlan {
    W8LinearAddScheduleId schedule;
};

struct W8LinearAddProblem {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
    std::int32_t cols;
};

const char* w8_linear_add_schedule_name(W8LinearAddScheduleId schedule) noexcept;
bool w8_linear_add_schedule_uses_mma(W8LinearAddScheduleId schedule) noexcept;
bool w8_linear_add_admits(const W8LinearAddProblem& problem) noexcept;
W8LinearAddPlan w8_linear_add_resolve_plan(const W8LinearAddProblem& problem);

void w8_linear_add_execute_plan(const W8LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
