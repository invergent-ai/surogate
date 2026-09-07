/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cutlass_fused_moe_kernels.cuh"
#include "moe_kernels.h"

// Trimmed for the serving engine: one instantiation — FP4 weights, bf16 activations quantised by
// the runner, bf16 output (the `NeedQuant` variant FlashInfer selects for bf16 inputs). The
// original file instantiates every dtype combination FlashInfer exposes.
namespace tensorrt_llm::kernels::cutlass_kernels {
template class CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;
} // namespace tensorrt_llm::kernels::cutlass_kernels
