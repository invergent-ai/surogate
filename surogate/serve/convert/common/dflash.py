"""Resolve a DFlash checkpoint independently of the target checkpoint."""

from dataclasses import dataclass, replace
from collections.abc import Mapping
import math

from .checkpoint import positive_int


@dataclass(frozen=True)
class Geometry:
    hidden: int
    layers: int
    intermediate: int
    query_heads: int
    kv_heads: int
    head_dim: int
    local_capacity: int
    mask_token: int
    block_size: int
    max_context: int
    rms_epsilon: float
    rope_theta: float
    layer_types: tuple[str, ...]
    target_feature_layers: tuple[int, ...]

    @property
    def query_size(self):
        return self.query_heads * self.head_dim

    @property
    def kv_size(self):
        return self.kv_heads * self.head_dim

    @property
    def feature_rows(self):
        return len(self.target_feature_layers) * self.hidden

    @property
    def local_layers(self):
        return self.layers - 1

    def declaration(self, target):
        return replace(target.declared, section_config={
            "dflash_layers": self.layers, "dflash_head_dim": self.head_dim,
            "dflash_qkv_rows": self.query_size + 2 * self.kv_size,
            "dflash_attn_cols": self.query_size, "dflash_kv_rows": self.kv_size,
            "dflash_gate_up_rows": 2 * self.intermediate, "dflash_ffn": self.intermediate,
            "dflash_feature_rows": self.feature_rows,
        })


def geometry_from_config(config: Mapping, target) -> Geometry:
    for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "head_dim",
                 "num_attention_heads", "num_key_value_heads", "sliding_window",
                 "max_position_embeddings", "vocab_size", "num_target_layers"):
        positive_int(config, name)
    if config["hidden_size"] != target.hidden or config["vocab_size"] != target.vocab or config["num_target_layers"] != target.layers:
        raise ValueError("DFlash hidden_size, vocab_size and num_target_layers must match the target checkpoint")
    if config["num_attention_heads"] % config["num_key_value_heads"]:
        raise ValueError("DFlash query heads must be divisible by KV heads")
    if config["head_dim"] % 2:
        raise ValueError("DFlash head_dim must be even")
    if config.get("hidden_act", "silu") != "silu" or config.get("attention_bias", False):
        raise ValueError("DFlash requires SiLU and attention without projection biases")
    layers = config["num_hidden_layers"]
    schedule = config.get("layer_types")
    # The backend's cache organization supports a local stack ending in one full layer.
    if layers < 2 or layers > 256 or schedule != ["sliding_attention"] * (layers - 1) + ["full_attention"]:
        raise ValueError("DFlash requires local attention layers followed by one full attention layer")
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError("DFlash rope_parameters and dflash_config are required")
    if rope.get("rope_type", "default") != "default":
        raise ValueError("DFlash supports default RoPE only")
    for name, value in (("rms_norm_eps", config.get("rms_norm_eps")), ("rope_theta", rope.get("rope_theta"))):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"DFlash {name} must be finite and positive")
    block = positive_int(draft, "block_size")
    if block < 2 or block > 16:
        raise ValueError("DFlash currently supports block_size in 2..16")
    mask = draft.get("mask_token_id")
    if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < target.vocab:
        raise ValueError("DFlash mask_token_id must index the target embedding")
    targets = draft.get("target_layer_ids")
    if not isinstance(targets, list) or not targets or any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < target.layers for i in targets
    ) or len(set(targets)) != len(targets):
        raise ValueError("DFlash target_layer_ids must be distinct valid target layer indices")
    g = Geometry(target.hidden, layers, config["intermediate_size"], config["num_attention_heads"],
                 config["num_key_value_heads"], config["head_dim"], config["sliding_window"], mask,
                 block, config["max_position_embeddings"], config["rms_norm_eps"], rope["rope_theta"],
                 tuple(schedule), tuple(targets))
    if max(g.query_size + 2 * g.kv_size, g.feature_rows, 2 * g.intermediate) > 2147483647:
        raise ValueError("DFlash projection dimensions exceed int32")
    return g


def geometry_block(g: Geometry):
    return dict(hidden=g.hidden, layers=g.layers, local_layers=g.local_layers,
                intermediate=g.intermediate, query_heads=g.query_heads, kv_heads=g.kv_heads,
                head_dim=g.head_dim, local_capacity=g.local_capacity, mask_token=g.mask_token,
                block_size=g.block_size, max_context=g.max_context, rms_epsilon=g.rms_epsilon,
                rope_theta=g.rope_theta, attention_scale=g.head_dim ** -.5,
                feature_layers=len(g.target_feature_layers), feature_rows=g.feature_rows)
