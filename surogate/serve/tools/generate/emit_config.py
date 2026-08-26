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
        raise NotImplementedError(
            "config emission currently covers hybrid (attention + gated delta net) "
            "targets; a pure-attention family needs its own layout section"
        )
    return f"""#pragma once

#include <api/targets/qwen3_6/frontend.h>
#include <api/targets/qwen3_6/hybrid_topology.h>
#include <api/targets/qwen3_6/vision.h>

#include <cstdint>

namespace ninfer::targets::{spec.name}::detail {{

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
"""
