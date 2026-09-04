"""The dimensions of one hybrid GDN/attention decoder, as a value.

One reference program serves every size of this architecture, so the numbers
below are data rather than constants: the defaults describe the smallest
checkpoint, and a loaded artifact replaces them with its own.  This mirrors
`family::TextGeometry` and `family::VisionGeometry` on the engine side, down to
the member names the artifact's `geometry` and `vision_geometry` declarations use,
so a checkpoint that moves a dimension moves it identically on both sides.

Only primary dimensions are fields.  Everything derived from them -- `key_dim`,
`conv_dim`, the MTP row counts -- is a property, so a checkpoint that declares
`gdn_k_heads` cannot leave a stale `key_dim` behind.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isqrt


@dataclass(frozen=True)
class ModelConfig:
    """The text decoder's dimensions.  Defaults describe the 0.8B checkpoint."""

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
        return self.layers // self.full_interval

    @property
    def gdn_layers(self) -> int:
        return self.layers - self.full_layers

    def is_full(self, layer: int) -> bool:
        return (layer + 1) % self.full_interval == 0

    def full_index(self, layer: int) -> int:
        return (layer + 1) // self.full_interval - 1

    def gdn_index(self, layer: int) -> int:
        return layer - (layer + 1) // self.full_interval

    @property
    def attention_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def gdn_scale(self) -> float:
        return self.gdn_k_dim ** -0.5


@dataclass(frozen=True)
class VisionConfig:
    """The vision tower's dimensions.  Only some checkpoints ship one.

    `out_hidden` is the width the merger projects into -- the text model's hidden
    state -- so it belongs to the tower's output contract rather than to the tower,
    and :func:`vision_config_for` sets it from the text geometry.
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

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch * self.patch * self.patch

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
}

_TEXT_GEOMETRY_FLOAT_KEYS: dict[str, str] = {
    "rms_epsilon": "rms_eps",
    "rope_theta": "rope_theta",
}

_VISION_GEOMETRY_KEYS: dict[str, str] = {
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


def _overridden(
    base, declared, int_keys: dict[str, str], float_keys: dict[str, str]
):
    """`base` with every dimension the artifact declares laid over it.

    An absent key keeps the default, so an artifact written before the member
    existed loads exactly as it did; an unknown key is ignored, so an artifact may
    name a dimension a future reference reads and this one does not.
    """

    if not declared:
        return base
    changes: dict[str, object] = {}
    for key, field in int_keys.items():
        if key in declared:
            changes[field] = int(declared[key])
    for key, field in float_keys.items():
        if key in declared:
            changes[field] = float(declared[key])
    return replace(base, **changes) if changes else base


def model_config_from_declared(
    declared: dict[str, float] | None, base: ModelConfig | None = None
) -> ModelConfig:
    """The text geometry an artifact declares, over `base`."""

    return _overridden(
        base if base is not None else ModelConfig(),
        declared,
        _TEXT_GEOMETRY_KEYS,
        _TEXT_GEOMETRY_FLOAT_KEYS,
    )


def vision_config_from_declared(
    declared: dict[str, float] | None, base: VisionConfig | None = None
) -> VisionConfig:
    """The vision geometry an artifact declares, over `base`."""

    return _overridden(
        base if base is not None else VisionConfig(),
        declared,
        _VISION_GEOMETRY_KEYS,
        _VISION_GEOMETRY_FLOAT_KEYS,
    )


def vision_config_for(cfg: ModelConfig, base: VisionConfig | None = None) -> VisionConfig:
    """A tower whose merger projects into `cfg`'s residual stream."""

    return replace(base if base is not None else VisionConfig(), out_hidden=cfg.hidden)


def model_config_from_config(config: dict) -> ModelConfig:
    """The dimensions a checkpoint's `config.json` declares.

    `config` is the whole file; a multimodal checkpoint nests the decoder's numbers
    under `text_config`.
    """

    text = config.get("text_config", config)
    base = ModelConfig()
    return ModelConfig(
        hidden=int(text["hidden_size"]),
        layers=int(text["num_hidden_layers"]),
        intermediate=int(text["intermediate_size"]),
        vocab=int(text.get("vocab_size", base.vocab)),
        token_domain=int(text.get("token_domain", base.token_domain)),
        q_heads=int(text["num_attention_heads"]),
        kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        rotary_dim=int(text.get("partial_rotary_dim", base.rotary_dim)),
        full_interval=int(text.get("full_attention_interval", base.full_interval)),
        gdn_k_heads=int(text["linear_num_key_heads"]),
        gdn_v_heads=int(text["linear_num_value_heads"]),
        gdn_k_dim=int(text["linear_key_head_dim"]),
        gdn_v_dim=int(text["linear_value_head_dim"]),
        conv_width=int(text["linear_conv_kernel_dim"]),
        rms_eps=float(text.get("rms_norm_eps", base.rms_eps)),
        rope_theta=float(text.get("rope_theta", base.rope_theta)),
        max_position_embeddings=int(
            text.get("max_position_embeddings", base.max_position_embeddings)
        ),
    )


#: The family default, for a caller that has no checkpoint in hand.  A loaded artifact
#: carries its own geometry on the binding; nothing on the model path reads this.
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
