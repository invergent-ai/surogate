"""Resolve checkpoint values before deriving serving objects or recipes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from .declaration import Declaration, declare


def tokenizer_ids(root) -> tuple[int, ...]:
    """Read actual token IDs, including added tokens and Unigram vocabularies."""
    import json
    from pathlib import Path

    tokenizer = json.loads((Path(root) / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    ids = list(vocab.values()) if isinstance(vocab, dict) else list(range(len(vocab)))
    ids.extend(token["id"] for token in tokenizer.get("added_tokens", ()))
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("tokenizer must declare nonnegative integer token IDs")
    positive_int({"token_domain": max(ids) + 1}, "token_domain")
    return tuple(sorted(set(ids)))


def tokenizer_domain(root) -> int:
    """Return the output-head domain needed to address this tokenizer's IDs."""
    return tokenizer_ids(root)[-1] + 1


def positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2147483647:
        raise ValueError(f"config.{name} must be a positive int32")
    return value


def resolve_dense(architecture: str, config: Mapping[str, Any]) -> Declaration:
    """Resolve a dense checkpoint without letting DSL size defaults fill missing keys."""
    source = deepcopy(dict(config))
    if architecture in ("Gemma3ForCausalLM", "Gemma3TextModel"):
        # Exporters use both spellings; normalize before DSL resolution so the
        # object walk and the serialized schedule read the same value.
        source["_sliding_window_pattern"] = (
            source.get("sliding_window_pattern") or source.get("_sliding_window_pattern")
        )
    for name in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
                 "num_attention_heads", "num_key_value_heads"):
        positive_int(source, name)
    if source.get("head_dim") is None:
        # Llama defines its head width by exact division. Qwen declares it independently.
        if architecture != "LlamaForCausalLM":
            raise ValueError("config.head_dim is required for this architecture")
        hidden, heads = source["hidden_size"], source["num_attention_heads"]
        if hidden % heads:
            raise ValueError("config.hidden_size must be divisible by num_attention_heads")
        source["head_dim"] = hidden // heads
    positive_int(source, "head_dim")
    if source["num_attention_heads"] % source["num_key_value_heads"]:
        raise ValueError("config.num_attention_heads must be divisible by num_key_value_heads")
    return declare(architecture, source)


def dense_geometry(geometry: Any, *, token_domain: int) -> dict[str, int | float]:
    """Serialize the resolved dense text stack, including its execution parameters."""
    from surogate.serve.artifact.geometry import validate_resolved_geometry

    config = geometry.declared.hf_config
    return validate_resolved_geometry({
        "hidden": geometry.hidden, "residual": geometry.hidden,
        "layers": geometry.layers, "intermediate": geometry.intermediate,
        "output_rows": geometry.vocab, "token_domain": token_domain,
        "query_heads": geometry.query_heads, "kv_heads": geometry.kv_heads,
        "head_dim": geometry.head_dim, "rotary_dim": geometry.head_dim,
        "rms_epsilon": config["rms_norm_eps"], "rope_theta": config["rope_theta"],
        "max_context": positive_int(config, "max_position_embeddings"),
        "attention_scale": 1.0 / math.sqrt(geometry.head_dim),
    })
