"""Geometry fields shared by serving artifact writers and C++ readers.

The generated C++ field lists are checked against this module by the contract tests.
These are field names and numeric constraints, never checkpoint dimensions or defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

TEXT_INT_FIELDS = (
    "hidden",
    "residual",
    "residual_fp32",
    "layers",
    "intermediate",
    "output_rows",
    "token_domain",
    "query_heads",
    "kv_heads",
    "head_dim",
    "rotary_dim",
    "mrope_temporal", "mrope_height", "mrope_width",
    "sliding_rotary_dim",
    "gdn_conv_kernel",
    "gdn_key_heads",
    "gdn_key_head_dim",
    "gdn_value_heads",
    "gdn_value_head_dim",
    "mtp_layers",
    "draft_vocab",
    "sliding_window",
    "q_lora_rank",
    "kv_lora_rank",
    "kda_gate_rank",
    "hc_streams",
    "dense_intermediate",
    "global_head_dim",
    "global_kv_heads",
    "global_rotary_angles",
    "per_layer_input_dim",
    "per_layer_vocab",
    "shared_kv_intermediate",
    "experts",
    "experts_per_token",
    "shared_intermediate",
    "qk_head_dim",
    "v_head_dim",
    "max_context",
    "kv_shared_layers",
    "attention_k_eq_v",
    "leading_dense_layers",
    "hc_sinkhorn_iterations",
    "hc_low_rank",
    "indexer_heads",
    "indexer_head_dim",
    "indexer_top_k",
    "indexer_block",
    "ple_layer",
    "ple_ngram",
    "ple_heads_per_ngram",
    "ple_head_dim",
    "ple_conv_kernel",
    "ple_table_rows",
    "ple_eos_token",
    "ple_image_token",
)

TEXT_FLOAT_FIELDS = (
    "rms_epsilon",
    "rope_theta",
    "embedding_scale",
    "logit_softcap",
    "sliding_rope_theta",
    "attention_scale",
    "gdn_scale",
    "routed_scale",
    "swiglu_limit",
    "hc_epsilon",
    "kda_gate_bound",
)

VISION_INT_FIELDS = (
    "gemma_version", "gemma_pad_token", "encoder_free", "clipped_linears", "standardize", "attention_mode", "max_image_tokens",
    "deepstack_layers",
    "siglip2",
    "projector_hidden",
    "projector_norm",
    "layers",
    "hidden",
    "intermediate",
    "heads",
    "patch_dim",
    "merge",
    "position_embeddings",
    "rotary_dim",
    "output_hidden",
)

VISION_FLOAT_FIELDS = (
    "rope_theta",
    "norm_epsilon",
)

DFLASH_INT_FIELDS = (
    "hidden", "layers", "local_layers", "intermediate", "query_heads", "kv_heads", "head_dim",
    "local_capacity", "mask_token", "block_size", "max_context", "feature_layers", "feature_rows",
)
DFLASH_FLOAT_FIELDS = ("rms_epsilon", "rope_theta", "attention_scale")

REQUIRED_TEXT_FIELDS = (
    "hidden", "residual", "layers", "intermediate", "output_rows", "token_domain",
    "query_heads", "kv_heads", "head_dim", "rotary_dim", "rms_epsilon", "rope_theta",
    "max_context", "attention_scale",
)


def validate_resolved_geometry(values: Mapping[str, object]) -> dict[str, int | float]:
    """Validate complete text metadata. Missing dimensions never select a compiled size."""
    result = validate_geometry(values)
    for name in REQUIRED_TEXT_FIELDS:
        if name not in result:
            raise ValueError(f"missing geometry.{name}; rebuild the serving cache")
        if name not in ("rotary_dim", "rope_theta") and result[name] <= 0:
            raise ValueError(f"geometry.{name} must be positive")
    if result["token_domain"] > result["output_rows"]:
        raise ValueError("geometry.token_domain exceeds output_rows")
    if result["query_heads"] % result["kv_heads"]:
        raise ValueError("geometry.query_heads must be divisible by kv_heads")
    if result["rotary_dim"] > result["head_dim"] or result["rotary_dim"] % 2:
        raise ValueError("geometry.rotary_dim must be even and no greater than head_dim")
    if result["rotary_dim"] and result["rope_theta"] <= 0:
        raise ValueError("geometry.rope_theta must be positive when rotary_dim is nonzero")
    sections = [result.get(name, 0) for name in ("mrope_temporal", "mrope_height", "mrope_width")]
    pairs = result["rotary_dim"] // 2
    if any(sections) and (min(sections) <= 0 or sum(sections) != pairs or
                          sections[1] > (pairs + 1) // 3 or sections[2] > pairs // 3):
        raise ValueError("invalid interleaved MRoPE sections")
    if result.get("residual_fp32", 0) not in (0, 1):
        raise ValueError("geometry.residual_fp32 must be 0 or 1")
    sliding_dim = result.get("sliding_rotary_dim", 0)
    if sliding_dim > result["head_dim"] or sliding_dim % 2:
        raise ValueError("geometry.sliding_rotary_dim must be even and no greater than head_dim")
    if sliding_dim and (not result.get("sliding_window") or not result.get("sliding_rope_theta")):
        raise ValueError("sliding_rotary_dim requires sliding_window and sliding_rope_theta")
    if result["layers"] > 256:
        raise ValueError("serving supports at most 256 layers")
    for heads in ("query_heads", "kv_heads"):
        if result[heads] * result["head_dim"] > 2147483647:
            raise ValueError(f"geometry.{heads} * head_dim exceeds int32")
    return result

def validate_geometry(values: Mapping[str, object], *, vision: bool = False, dflash: bool = False) -> dict[str, int | float]:
    """Validate a geometry map without inventing values for absent fields."""
    ints = DFLASH_INT_FIELDS if dflash else VISION_INT_FIELDS if vision else TEXT_INT_FIELDS
    floats = DFLASH_FLOAT_FIELDS if dflash else VISION_FLOAT_FIELDS if vision else TEXT_FLOAT_FIELDS
    scope = "dflash_geometry" if dflash else "vision_geometry" if vision else "geometry"
    result: dict[str, int | float] = {}
    for name, value in values.items():
        if name not in ints and name not in floats:
            raise ValueError(f"unknown {scope} field {name!r}; rebuild the serving cache")
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                (isinstance(value, float) and not math.isfinite(value))):
            raise ValueError(f"{scope}.{name} must be a finite number")
        if name in ints:
            if value != int(value) or not 0 <= value <= 2147483647:
                raise ValueError(f"{scope}.{name} must be a nonnegative int32")
            result[name] = int(value)
        else:
            if not 0 <= value <= 3.4028234663852886e38:
                raise ValueError(f"{scope}.{name} must be a nonnegative float32")
            result[name] = float(value)
    return result


def cpp_fields(*, vision: bool = False, dflash: bool = False) -> str:
    """Generate an X-macro list consumed by the standalone C++ geometry headers."""
    ints = DFLASH_INT_FIELDS if dflash else VISION_INT_FIELDS if vision else TEXT_INT_FIELDS
    floats = DFLASH_FLOAT_FIELDS if dflash else VISION_FLOAT_FIELDS if vision else TEXT_FLOAT_FIELDS
    return (
        "// Generated from surogate/serve/artifact/geometry.py. Do not edit.\n"
        + "".join(f"SINFER_GEOMETRY_INT({name})\n" for name in ints)
        + "".join(f"SINFER_GEOMETRY_FLOAT({name})\n" for name in floats)
    )


def cpp_required_fields() -> str:
    return (
        "// Generated from surogate/serve/artifact/geometry.py. Do not edit.\n"
        + "".join(f"SINFER_GEOMETRY_REQUIRED({name})\n" for name in REQUIRED_TEXT_FIELDS)
    )


def validate_dflash_geometry(values, targets, text):
    g = validate_geometry(values, dflash=True)
    for name in (*DFLASH_INT_FIELDS, *DFLASH_FLOAT_FIELDS):
        if name not in g or (name != "mask_token" and g[name] <= 0):
            raise ValueError(f"dflash_geometry.{name} is required and must be positive")
    if not 2 <= g["layers"] <= 256 or g["local_layers"] != g["layers"] - 1:
        raise ValueError("DFlash requires local layers followed by one full attention layer")
    if g["hidden"] != text.get("hidden") or g["mask_token"] >= text.get("output_rows", 0):
        raise ValueError("DFlash hidden width or mask token is incompatible with the target")
    if g["query_heads"] % g["kv_heads"] or g["head_dim"] % 2 or not 2 <= g["block_size"] <= 16:
        raise ValueError("invalid DFlash heads or block size")
    if not isinstance(targets, list) or len(targets) != g["feature_layers"] or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < text.get("layers", 0) for i in targets
    ) or len(set(targets)) != len(targets):
        raise ValueError("dflash_target_layers must list distinct target layers matching feature_layers")
    if g["feature_rows"] != len(targets) * g["hidden"]:
        raise ValueError("DFlash feature_rows disagrees with target layers and hidden width")
    if max((g["query_heads"] + 2 * g["kv_heads"]) * g["head_dim"], 2 * g["intermediate"]) > 2147483647:
        raise ValueError("DFlash projection dimensions exceed int32")
    return g
