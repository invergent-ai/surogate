#pragma once

#include "api/family/text_geometry.h"
#include "api/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sinfer::family {

/// --rope-scaling: lays YaRN (Hugging Face rope_type "yarn") over the checkpoint's rotation and
/// extends its context to original * factor, the way vLLM's derived max_model_len does. Only the
/// text rotation the rope kernels hold YaRN for (head dim 256, rotary dim 64: the Qwen3.5 / 3.6
/// family and Qwen3.8-Flash-Next) is served with it; any other is refused rather than run
/// unscaled past its native context. Applying twice is applying once.
inline void apply_rope_scaling(TextGeometry& g, const EngineOptions& options) {
    const EngineOptions::RopeScaling& r = options.rope_scaling;
    if (r.factor == 0.0F) { return; }
    if (!std::isfinite(r.factor) || r.factor <= 1.0F) {
        throw std::invalid_argument("--rope-scaling: factor must be finite and greater than 1");
    }
    if (g.head_dim != 256 || g.rotary_dim != 64 || g.rope_theta <= 0.0F) {
        throw std::invalid_argument(
            "--rope-scaling: YaRN is served for the Qwen3.5/3.6 and Flash-Next text rotation "
            "(head dim 256, rotary dim 64), not this model's");
    }
    if (g.rope_yarn_factor != 0.0F) { return; }
    const std::int64_t original = r.original_context != 0 ? r.original_context : g.max_context;
    const double extended       = std::floor(static_cast<double>(original) * r.factor);
    if (original <= 0 || extended > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("--rope-scaling: original context times factor must fit int32");
    }
    g.rope_yarn_factor           = r.factor;
    g.rope_yarn_original         = static_cast<std::int32_t>(original);
    g.rope_yarn_beta_fast        = r.beta_fast;
    g.rope_yarn_beta_slow        = r.beta_slow;
    g.rope_yarn_attention_factor = r.attention_factor;
    g.max_context = std::max(g.max_context, static_cast<std::int32_t>(extended));
}

} // namespace sinfer::family
