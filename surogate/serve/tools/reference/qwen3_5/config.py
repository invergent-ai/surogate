"""Resolved reference-model configuration; defaults below are numerical test fixtures only.

Artifact loading requires complete metadata and an explicit layer schedule. It never uses
fixture dimensions to fill an absent checkpoint field.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isqrt


@dataclass(frozen=True)
class ModelConfig:
    """Dimensions for one decoder; artifact constructors supply every checkpoint field."""

    hidden: int = 1024
    layers: int = 24
    intermediate: int = 3584
    vocab: int = 248320
    token_domain: int = 248077
    q_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    full_interval: int = 4
    layer_types: tuple[str, ...] = ()
    attention_scale_value: float | None = None
    gdn_scale_value: float | None = None
    mtp_layers: int = 1
    draft_vocab: int = 131072
    gdn_k_heads: int = 16
    gdn_v_heads: int = 16
    gdn_k_dim: int = 128
    gdn_v_dim: int = 128
    conv_width: int = 4
    rms_eps: float = 1.0e-6
    rope_theta: float = 1.0e7
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    prefill_chunk: int = 1024
    max_position_embeddings: int = 262144

    @property
    def q_size(self) -> int:
        """Rows of the attention query projection, and of its output projection's input."""

        return self.q_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def key_dim(self) -> int:
        """Rows of one GDN query or key projection -- unrelated to `kv_size`."""

        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def value_dim(self) -> int:
        """Rows of the GDN value projection, and of its output projection's input."""

        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def attention_input_rows(self) -> int:
        """Rows of the fused query|key|gate|value parent, in text layers and in MTP."""

        return 2 * self.q_size + 2 * self.kv_size

    @property
    def mtp_input_rows(self) -> int:
        """Columns of the MTP input projection: the embedding and the hidden state."""

        return 2 * self.hidden

    @property
    def full_layers(self) -> int:
        return sum(self.is_full(i) for i in range(self.layers))

    @property
    def gdn_layers(self) -> int:
        return self.layers - self.full_layers

    def is_full(self, layer: int) -> bool:
        return (
            self.layer_types[layer] == "full_attention" if self.layer_types else (layer + 1) % self.full_interval == 0
        )

    def full_index(self, layer: int) -> int:
        return sum(self.is_full(i) for i in range(layer + 1)) - 1

    def gdn_index(self, layer: int) -> int:
        return sum(not self.is_full(i) for i in range(layer + 1)) - 1

    @property
    def attention_scale(self) -> float:
        return self.attention_scale_value if self.attention_scale_value is not None else self.head_dim**-0.5

    @property
    def gdn_scale(self) -> float:
        return self.gdn_scale_value if self.gdn_scale_value is not None else self.gdn_k_dim**-0.5


@dataclass(frozen=True)
class VisionConfig:
    """The vision tower's dimensions.  Only some checkpoints ship one.

    `out_hidden` is the width the merger projects into -- the text model's hidden
    state -- so it belongs to the tower's output contract rather than to the tower,
    and :func:`vision_config_for` validates it against the text geometry.
    """

    depth: int = 27
    hidden: int = 1152
    intermediate: int = 4304
    out_hidden: int = 1024
    heads: int = 16
    in_channels: int = 3
    patch: int = 16
    temporal_patch: int = 2
    spatial_merge: int = 2
    position_embeddings: int = 2304
    rope_theta: float = 10000.0
    norm_eps: float = 1.0e-6
    patch_dim_value: int | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return (
            self.patch_dim_value
            if self.patch_dim_value is not None
            else self.in_channels * self.temporal_patch * self.patch * self.patch
        )

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge * self.spatial_merge

    @property
    def merger_hidden(self) -> int:
        return self.hidden * self.merge_unit

    @property
    def position_side(self) -> int:
        return isqrt(self.position_embeddings)


#: The engine names these dimensions in an artifact's `geometry` member.  The reference
#: spells two of them differently, so the declaration is translated rather than copied.
_TEXT_GEOMETRY_KEYS: dict[str, str] = {
    "hidden": "hidden",
    "layers": "layers",
    "intermediate": "intermediate",
    "output_rows": "vocab",
    "token_domain": "token_domain",
    "query_heads": "q_heads",
    "kv_heads": "kv_heads",
    "head_dim": "head_dim",
    "rotary_dim": "rotary_dim",
    "gdn_conv_kernel": "conv_width",
    "gdn_key_heads": "gdn_k_heads",
    "gdn_key_head_dim": "gdn_k_dim",
    "gdn_value_heads": "gdn_v_heads",
    "gdn_value_head_dim": "gdn_v_dim",
    "max_context": "max_position_embeddings",
    "mtp_layers": "mtp_layers",
    "draft_vocab": "draft_vocab",
}

_TEXT_GEOMETRY_FLOAT_KEYS: dict[str, str] = {
    "attention_scale": "attention_scale_value",
    "gdn_scale": "gdn_scale_value",
    "rms_epsilon": "rms_eps",
    "rope_theta": "rope_theta",
}

_VISION_GEOMETRY_KEYS: dict[str, str] = {
    "patch_dim": "patch_dim_value",
    "layers": "depth",
    "hidden": "hidden",
    "intermediate": "intermediate",
    "heads": "heads",
    "merge": "spatial_merge",
    "position_embeddings": "position_embeddings",
    "output_hidden": "out_hidden",
}

_VISION_GEOMETRY_FLOAT_KEYS: dict[str, str] = {
    "rope_theta": "rope_theta",
    "norm_epsilon": "norm_eps",
}


def _required_fields(declared, int_keys, float_keys, *, vision=False):
    from surogate.serve.artifact.geometry import validate_geometry

    values = validate_geometry(declared or {}, vision=vision)
    missing = sorted((int_keys.keys() | float_keys.keys()) - values.keys())
    if missing:
        raise ValueError(f"missing checkpoint geometry fields: {', '.join(missing)}; rebuild the artifact")
    return {
        **{field: int(values[key]) for key, field in int_keys.items()},
        **{field: float(values[key]) for key, field in float_keys.items()},
    }


def model_config_from_declared(declared, base=None, *, layer_types) -> ModelConfig:
    from surogate.serve.artifact.geometry import validate_resolved_geometry

    values = validate_resolved_geometry(declared or {})
    schedule = tuple(layer_types)
    if len(schedule) != values["layers"] or any(k not in ("full_attention", "linear_attention") for k in schedule):
        raise ValueError("hybrid reference requires the complete checkpoint layer_types schedule")
    changes = _required_fields(values, _TEXT_GEOMETRY_KEYS, _TEXT_GEOMETRY_FLOAT_KEYS)
    if values["rotary_dim"] != 64 or values.get("mtp_layers") not in (0, 1):
        raise ValueError("hybrid reference requires rotary_dim 64 and zero or one MTP layer")
    return replace(base if base is not None else ModelConfig(), **changes, layer_types=schedule)


def vision_config_from_declared(declared, base=None) -> VisionConfig:
    changes = _required_fields(declared, _VISION_GEOMETRY_KEYS, _VISION_GEOMETRY_FLOAT_KEYS, vision=True)
    return replace(base if base is not None else VisionConfig(), **changes)


def vision_config_for(cfg: ModelConfig, base: VisionConfig) -> VisionConfig:
    if base.out_hidden != cfg.hidden:
        raise ValueError("vision output width disagrees with text checkpoint")
    return base


def model_config_from_config(config: dict, *, token_domain: int) -> ModelConfig:
    from surogate.serve.convert.common import qwen3_5

    g = qwen3_5.geometry_from_config(config, token_domain=token_domain)
    return model_config_from_declared(qwen3_5.geometry_block(g), layer_types=g.layer_types)


#: Explicit numerical fixtures. Loaded artifacts always use required checkpoint metadata.
CFG = ModelConfig()
VISION_CFG = VisionConfig()
ATTN_SCALE = CFG.attention_scale
GDN_SCALE = CFG.gdn_scale


__all__ = [
    "ATTN_SCALE",
    "CFG",
    "GDN_SCALE",
    "VISION_CFG",
    "ModelConfig",
    "VisionConfig",
    "model_config_from_config",
    "model_config_from_declared",
    "vision_config_for",
    "vision_config_from_declared",
]
