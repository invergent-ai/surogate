"""Weight-mapping test: Qwen3.5/3.6 MoE expert layout.

The Qwen3.5/3.6 MoE family (Qwen3-Next-style hybrid, ``model_type=qwen3_5_moe``)
ships expert weights *pre-stacked and pre-fused* — one batched tensor per layer:

    model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [E, 2*M, C]
    model.language_model.layers.{L}.mlp.experts.down_proj      # [E, C,   M]

with NO per-expert ``experts.{e}.gate_proj.weight`` tensors. The DSL block
mapping must therefore reference those batched keys directly (a passthrough),
not the per-expert names — otherwise ``import_weights`` throws
``Entry not found: ...experts.0.down_proj.weight``.

This is a fast, GPU-free test: it only resolves the static block mapping and
checks the referenced source keys against the checkpoint's safetensors index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surogate.dsl.hf import StackExpertsMapping
from surogate.dsl.models.qwen3_5_moe import Qwen3_5MoEConditionalModel

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"


def _checkpoint_index() -> dict:
    """Locate the cached Qwen3.6-35B-A3B safetensors index, or skip."""
    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    model_cache = cache_root / f"models--{MODEL_ID.replace('/', '--')}"
    snaps = model_cache / "snapshots"
    if snaps.exists():
        for snap in sorted(snaps.iterdir(), reverse=True):
            idx = snap / "model.safetensors.index.json"
            if idx.exists():
                return json.loads(idx.read_text())["weight_map"]
    pytest.skip(f"{MODEL_ID} not cached")


def _source_keys(mapping_value, *, layer: int, num_experts: int) -> list[str]:
    """Enumerate the concrete HF checkpoint keys a mapping value reads."""
    if isinstance(mapping_value, StackExpertsMapping):
        pat = mapping_value.pattern.replace("{layer}", str(layer))
        keys = [pat.replace("{expert}", str(e)) for e in range(num_experts)]
        if mapping_value.fuse_gate_up:
            keys += [k.replace("gate_proj", "up_proj") for k in keys]
        return keys
    if isinstance(mapping_value, str):
        return [mapping_value.replace("{layer}", str(layer))]
    pytest.fail(f"unhandled mapping type for expert weights: {mapping_value!r}")


def test_expert_mapping_keys_exist_in_checkpoint():
    weight_map = _checkpoint_index()
    mappings = Qwen3_5MoEConditionalModel._hf_block_mappings_

    missing = []
    for param in ("experts_gate_up", "experts_down"):
        for src in _source_keys(mappings[param], layer=0, num_experts=256):
            if src not in weight_map:
                missing.append(f"{param}: {src}")

    assert not missing, (
        "expert mapping references keys absent from checkpoint:\n"
        + "\n".join(missing[:5])
        + (f"\n... (+{len(missing) - 5} more)" if len(missing) > 5 else "")
    )
