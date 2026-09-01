"""Emits a serve target's `impl/config.h` from a TargetSpec.

Correctness here is checked by regeneration, not by review: the committed
targets are the fixtures, and the emitter must reproduce them byte for byte
(see `check_roundtrip.py`). That is what makes it safe to delete the
hand-written copies later — a generator that cannot reproduce what it replaces
has not earned the right to replace it.
"""

from __future__ import annotations

from target_spec import TargetSpec


def _float_literal(value: float) -> str:
    """C++ float literal in the form the existing targets use (1.0e-6F)."""
    mantissa, exponent = f"{value:e}".split("e")
    mantissa = mantissa.rstrip("0")
    if mantissa.endswith("."):
        mantissa += "0"
    return f"{mantissa}e{int(exponent)}F"


def emit_config_h(spec: TargetSpec) -> str:
    spec.validate()
    a = spec.attention
    l = spec.linear_attention
    if l is None:
        return _emit_dense_config_h(spec)
    return f"""#pragma once

#include <api/targets/qwen3_6/frontend.h>
#include <api/targets/qwen3_6/hybrid_topology.h>
#include <api/targets/qwen3_6/vision.h>

#include <cstdint>

namespace sinfer::targets::{spec.name}::detail {{

struct TextConfig {{
    static constexpr int hidden       = {spec.hidden};
    static constexpr int layers       = {spec.layers};
    static constexpr int intermediate = {spec.intermediate};

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = {spec.vocab};
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = {l.conv_kernel};
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = {l.key_heads};
    static constexpr int gdn_key_head_dim     = {l.key_head_dim};
    static constexpr int gdn_value_heads      = {l.value_heads};
    static constexpr int gdn_value_head_dim   = {l.value_head_dim};

    static constexpr int query_heads = {a.query_heads};
    static constexpr int kv_heads    = {a.kv_heads};
    static constexpr int head_dim    = {a.head_dim};
    static constexpr int rotary_dim  = {a.rotary_dim};

    static constexpr int full_attention_interval = qwen3_6::kHybridAttentionInterval;
    static constexpr float rms_epsilon           = {_float_literal(spec.rms_epsilon)};
    static constexpr float rope_theta            = {_float_literal(spec.rope_theta)};

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = 2 * query_size;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {{
        return qwen3_6::is_full_attention_layer(layer);
    }}

    [[nodiscard]] static constexpr int full_attention_layers() {{
        return qwen3_6::full_attention_layers(layers);
    }}

    [[nodiscard]] static constexpr int gdn_layers() {{ return qwen3_6::gdn_layers(layers); }}

    [[nodiscard]] static constexpr int full_attention_index(int layer) {{
        return qwen3_6::full_attention_index(layer);
    }}

    [[nodiscard]] static constexpr int gdn_index(int layer) {{ return qwen3_6::gdn_index(layer); }}
}};

static_assert(TextConfig::full_attention_layers() == {spec.full_attention_layers});
static_assert(TextConfig::gdn_layers() == {spec.gdn_layers});

struct VisionConfig : qwen3_6::VisionBackboneConfig {{
    static constexpr int output_hidden = TextConfig::hidden;
}};

struct DFlashConfig {{
    static constexpr bool supported     = false;
    static constexpr int local_layers   = 0;
    static constexpr int local_capacity = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = 0;
    static constexpr int intermediate   = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
}};

inline constexpr float kAttentionScale                   = {spec.attention_scale!r}F;
inline constexpr float kGdnScale                         = {spec.gdn_scale!r}F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = {spec.mtp_draft_tokens};
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = {spec.native_context};

}} // namespace sinfer::targets::{spec.name}::detail
"""


def _emit_dense_config_h(spec: TargetSpec) -> str:
    """A pure-attention target: every layer is full attention.

    The family runtime is written over a layer schedule, not over the presence of
    a linear mixer, so a dense model is expressible as the degenerate hybrid
    where `is_full_attention` is true everywhere and `gdn_layers()` is zero. The
    GDN geometry is still emitted, as zeros, because the shared runtime reads
    those constants unconditionally when it sizes its (then empty) linear state;
    leaving them out would not compile.

    Two constants differ in kind rather than value from the hybrid emitter, and
    both are load-bearing:

      * `query_projection_rows` is `query_size`, not `2 * query_size`. The
        hybrid family fuses an output gate beside the query rows; a plain
        attention stack has no gate, and a target that claimed one would read
        the projection with the wrong stride.
      * `full_attention_interval` is 1, so the schedule helpers resolve to the
        identity rather than to the family's every-fourth-layer rule.
    """
    a = spec.attention
    return f"""#pragma once

#include <api/targets/qwen3_6/frontend.h>
#include <api/targets/qwen3_6/hybrid_topology.h>
#include <api/targets/qwen3_6/vision.h>

#include <cstdint>

namespace sinfer::targets::{spec.name}::detail {{

struct TextConfig {{
    static constexpr int hidden       = {spec.hidden};
    static constexpr int layers       = {spec.layers};
    static constexpr int intermediate = {spec.intermediate};

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = {spec.vocab};
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    // No linear mixer. These stay declared because the shared runtime reads them
    // when it sizes the linear-attention state, which is empty here.
    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    static constexpr int query_heads = {a.query_heads};
    static constexpr int kv_heads    = {a.kv_heads};
    static constexpr int head_dim    = {a.head_dim};
    static constexpr int rotary_dim  = {a.rotary_dim};

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = {_float_literal(spec.rms_epsilon)};
    static constexpr float rope_theta            = {_float_literal(spec.rope_theta)};

    static constexpr int key_dim               = 0;
    static constexpr int value_dim             = 0;
    static constexpr int convolution_dim       = 0;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    // Ungated: the projection carries query rows only, unlike the hybrid family.
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    [[nodiscard]] static constexpr bool is_full_attention(int) {{ return true; }}

    [[nodiscard]] static constexpr int full_attention_layers() {{ return layers; }}

    [[nodiscard]] static constexpr int gdn_layers() {{ return 0; }}

    [[nodiscard]] static constexpr int full_attention_index(int layer) {{ return layer; }}

    [[nodiscard]] static constexpr int gdn_index(int) {{ return -1; }}
}};

static_assert(TextConfig::full_attention_layers() == {spec.layers});
static_assert(TextConfig::gdn_layers() == 0);

struct VisionConfig : qwen3_6::VisionBackboneConfig {{
    static constexpr int output_hidden = TextConfig::hidden;
}};

struct DFlashConfig {{
    static constexpr bool supported     = false;
    static constexpr int local_layers   = 0;
    static constexpr int local_capacity = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = 0;
    static constexpr int intermediate   = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
}};

inline constexpr float kAttentionScale                   = {spec.attention_scale!r}F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = {spec.native_context};

}} // namespace sinfer::targets::{spec.name}::detail
"""
