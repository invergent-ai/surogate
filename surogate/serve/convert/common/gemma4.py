"""Resolve Gemma 4 checkpoint configuration before deriving serving objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import math
import struct
from typing import Any

from surogate.serve.artifact.geometry import validate_resolved_geometry

from . import declaration
from .checkpoint import positive_int
from .inventory import (
    BF16, DIRECT_FORMATS, LogicalAliasSpec, ResourceSpec, TensorSpec, W8, tensor_spec,
)

ARCHITECTURES = (
    "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)
CAPABILITIES = ("text",)
RESOURCE_SPECS = tuple(ResourceSpec("frontend/" + name) for name in (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
))
ALIAS_SPECS = (LogicalAliasSpec("text/output_head", ("text/token_embedding",)),)
ALIASED_OBJECT_NAMES = frozenset(spec.role_pattern for spec in ALIAS_SPECS)


def text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def architecture_of(config: Mapping[str, Any], target: str) -> str:
    text = text_config(config)
    for key in ("enable_moe_block", "attention_k_eq_v", "use_double_wide_mlp", "tie_word_embeddings"):
        if key in text and not isinstance(text[key], bool):
            raise ValueError(f"config.{key} must be boolean")
    actual = ("gemma4_moe" if text.get("enable_moe_block") else
              "gemma4_e" if (text.get("hidden_size_per_layer_input") or
                             text.get("num_kv_shared_layers")) else "gemma4")
    if actual != target:
        raise ValueError(f"checkpoint configuration belongs to {actual}, not {target}")
    for name in config.get("architectures") or ():
        if name in ARCHITECTURES:
            return name
    raise ValueError(f"{target} requires one of {ARCHITECTURES} in config.architectures")


@dataclass(frozen=True, slots=True)
class Geometry:
    declared: declaration.Declaration = field(repr=False, compare=False)
    hidden: int
    layers: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    global_kv_heads: int
    global_head_dim: int
    windowed: tuple[bool, ...]
    sliding_window: int
    rope_theta: float
    sliding_rope_theta: float
    partial_rotary_factor: float
    k_eq_v: bool
    final_logit_softcapping: float
    rms_epsilon: float
    max_context: int
    # Zero denotes an absent optional feature, never a checkpoint size default.
    per_layer_input_dim: int = 0
    per_layer_vocab: int = 0
    kv_shared_layers: int = 0
    shared_kv_intermediate: int = 0
    experts: int = 0
    experts_per_token: int = 0
    expert_intermediate: int = 0

    @property
    def query_size(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def global_query_size(self) -> int:
        return self.query_heads * self.global_head_dim

    @property
    def global_kv_size(self) -> int:
        return self.global_kv_heads * self.global_head_dim

    @property
    def embedding_scale(self) -> float:
        # The model multiplies by sqrt(hidden) rounded to BF16.
        bits = struct.unpack("<I", struct.pack("<f", math.sqrt(self.hidden)))[0]
        bits += 0x7FFF + ((bits >> 16) & 1)
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]

    @property
    def partial_rotary_angles(self) -> int:
        return int(self.partial_rotary_factor * self.global_head_dim // 2)

    def is_windowed(self, layer: int) -> bool:
        return self.windowed[layer]

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(i for i, windowed in enumerate(self.windowed) if not windowed)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple("sliding_attention" if w else "full_attention" for w in self.windowed)

    @property
    def first_shared_layer(self) -> int:
        return self.layers - self.kv_shared_layers

    @property
    def per_layer_input_total(self) -> int:
        return self.layers * self.per_layer_input_dim


def geometry_from_config(config: Mapping[str, Any], target: str) -> Geometry:
    architecture = architecture_of(config, target)
    source = deepcopy(dict(config))
    text = text_config(source)
    for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
                 "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
                 "max_position_embeddings", "sliding_window"):
        positive_int(text, name)
    # Global KV heads inherit the local count in the architecture's configuration.
    if text.get("num_global_key_value_heads") is None:
        text["num_global_key_value_heads"] = text["num_key_value_heads"]
    positive_int(text, "num_global_key_value_heads")
    epsilon = text.get("rms_norm_eps")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 < epsilon < math.inf:
        raise ValueError("config.rms_norm_eps must be a finite positive number")
    for name in ("num_key_value_heads", "num_global_key_value_heads"):
        if text["num_attention_heads"] % text[name]:
            raise ValueError(f"config.num_attention_heads must be divisible by {name}")
    for name in ("head_dim", "global_head_dim"):
        if text[name] % 2:
            raise ValueError(f"config.{name} must be even")
    if target == "gemma4_e":
        for name in ("hidden_size_per_layer_input", "vocab_size_per_layer_input"):
            positive_int(text, name)
    else:
        if text.get("hidden_size_per_layer_input") or text.get("num_kv_shared_layers"):
            raise ValueError(f"{target} does not support per-layer inputs or shared KV layers")
        # Prevent training constructor defaults from adding an absent PLI stack.
        text["hidden_size_per_layer_input"] = 0
        text["vocab_size_per_layer_input"] = 0
    shared = text.get("num_kv_shared_layers", 0)
    if isinstance(shared, bool) or not isinstance(shared, int) or not 0 <= shared < text["num_hidden_layers"]:
        raise ValueError("config.num_kv_shared_layers must be in [0, num_hidden_layers)")
    if target == "gemma4_moe":
        for name in ("num_experts", "top_k_experts", "moe_intermediate_size"):
            positive_int(text, name)
        if text["top_k_experts"] > text["num_experts"]:
            raise ValueError("config.top_k_experts exceeds num_experts")
    else:
        for name in ("num_experts", "top_k_experts", "moe_intermediate_size"):
            if text.get(name):
                raise ValueError(f"config.{name} requires enable_moe_block")
            text[name] = 0
    text.setdefault("num_kv_shared_layers", 0)
    text.setdefault("use_double_wide_mlp", False)
    text.setdefault("enable_moe_block", False)
    declared = declaration.declare(architecture, source)
    resolved = declared.config
    names = set(declared.model._serve_windowed_blocks_)
    windowed = tuple(block in names for block in declaration.block_types(resolved, declared.model))
    if shared:
        owners = set(windowed[:-shared])
        if not set(windowed[-shared:]) <= owners:
            raise ValueError("every shared KV layer requires an earlier owner of the same attention type")
    if resolved["sliding_rope_type"] != "default" or resolved["full_rope_type"] != "proportional":
        raise ValueError("Gemma 4 serving requires default local RoPE and proportional global RoPE")
    factor = resolved["full_partial_rotary_factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 <= factor <= 1:
        raise ValueError("config full_attention.partial_rotary_factor must be in [0, 1]")
    geometry = Geometry(
        declared=declared, hidden=resolved["d_model"], layers=resolved["n_layers"],
        intermediate=resolved["d_ff"], vocab=resolved["vocab_size"],
        query_heads=resolved["num_query_heads"], kv_heads=resolved["num_kv_heads"],
        head_dim=resolved["head_size"], global_kv_heads=resolved["global_num_kv_heads"],
        global_head_dim=resolved["global_head_dim"], windowed=windowed,
        sliding_window=resolved["sliding_window"], rope_theta=resolved["full_rope_theta"],
        sliding_rope_theta=resolved["sliding_rope_theta"], partial_rotary_factor=factor,
        k_eq_v=bool(resolved["k_eq_v"]),
        final_logit_softcapping=resolved.get("final_logit_softcapping") or 0.0,
        rms_epsilon=resolved["eps"], max_context=resolved["max_seq"],
        per_layer_input_dim=resolved["d_per_layer_input"],
        per_layer_vocab=resolved["vocab_size_per_layer_input"],
        kv_shared_layers=shared,
        shared_kv_intermediate=resolved["d_ff"] * (2 if resolved["use_double_wide_mlp"] else 1),
        experts=resolved["num_experts"], experts_per_token=resolved["top_k_experts"],
        expert_intermediate=resolved["moe_d_ff"],
    )
    if geometry.partial_rotary_angles <= 0:
        raise ValueError("Gemma 4 serving requires at least one active global rotary pair")
    geometry_block(geometry, token_domain=geometry.vocab)
    return geometry


def geometry_block(g: Geometry, *, token_domain: int) -> dict[str, int | float]:
    values = {
        "hidden": g.hidden, "residual": g.hidden, "layers": g.layers,
        "intermediate": g.expert_intermediate if g.experts else g.intermediate,
        "output_rows": g.vocab, "token_domain": token_domain, "query_heads": g.query_heads,
        "kv_heads": g.kv_heads, "head_dim": g.head_dim, "rotary_dim": g.head_dim,
        "global_kv_heads": g.global_kv_heads, "global_head_dim": g.global_head_dim,
        "global_rotary_angles": g.partial_rotary_angles,
        "sliding_window": g.sliding_window, "rope_theta": g.rope_theta,
        "sliding_rope_theta": g.sliding_rope_theta, "logit_softcap": g.final_logit_softcapping,
        "embedding_scale": g.embedding_scale, "rms_epsilon": g.rms_epsilon,
        "max_context": g.max_context, "attention_scale": 1.0,
        "attention_k_eq_v": int(g.k_eq_v),
    }
    if g.per_layer_input_dim:
        values.update(per_layer_input_dim=g.per_layer_input_dim, per_layer_vocab=g.per_layer_vocab,
                      kv_shared_layers=g.kv_shared_layers, shared_kv_intermediate=g.shared_kv_intermediate)
    if g.experts:
        values.update(experts=g.experts, experts_per_token=g.experts_per_token,
                      dense_intermediate=g.intermediate)
    return validate_resolved_geometry(values)


def declared_objects(geometry: Geometry) -> list[declaration.DeclaredObject]:
    return list(geometry.declared.objects(capabilities=set(CAPABILITIES)))


def stored_objects(geometry: Geometry, *, tied_output_head: bool) -> list[declaration.DeclaredObject]:
    return [obj for obj in declared_objects(geometry)
            if not tied_output_head or obj.name not in ALIASED_OBJECT_NAMES]


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    result = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        result.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(result)
