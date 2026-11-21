#pragma once

// Inkling-2-Mini attention LoRA (lane e1, S10, vLLM parity): the six qkvr slices and wo_ud under the checkpoint's
// module names (vLLM's inkling2 PACKED_MODULES_MAPPING "qkvr" = wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv,
// wv_dv; plus wo_ud), one bank table per physical layer keyed on the qkvr / wo weight (every pass of a core layer
// shares it). Applied after the projections by ops::inkling_lora_delta (one arithmetic at every width).

#include "targets/inkling2/impl/config.h"

#include <cstdint>

namespace sinfer::targets::inkling2::detail {

struct AttnLoraSlice {
    const char* name;
    std::int32_t begin; ///< first qkvr row
    std::int32_t rows;
};

inline constexpr AttnLoraSlice kQkvrLoraSlices[6] = {
    {"attn.wq_du", 0, 3072},           {"attn.wr_du", 3072, 384}, {"attn.gate_attn", 3456, 3072},
    {"attn.attn_sink_proj", 6528, 24}, {"attn.wk_dv", 6552, 1024}, {"attn.wv_dv", 7576, 1024},
};
static_assert(kQkvrLoraSlices[5].begin + kQkvrLoraSlices[5].rows == TextConfig::qkvr_rows);
static_assert(kQkvrLoraSlices[4].begin == TextConfig::qkvr_k_begin && kQkvrLoraSlices[5].begin == TextConfig::qkvr_v_begin);
/// A fused adapter over all of qkvr (A shared, B [8600, r] in the slices' order) -- the split names' twin.
inline constexpr const char* kQkvrLoraFusedName = "attn.qkvr";
inline constexpr const char* kWoLoraName        = "attn.wo_ud";
inline constexpr std::int32_t kQkvrLoraPort     = 900; ///< + slice index, under the qkvr weight
inline constexpr std::int32_t kWoLoraPort       = 906; ///< under the wo_ud weight

} // namespace sinfer::targets::inkling2::detail
