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

import inspect

import surogate.dsl.models as dsl_models
from surogate.dsl.hf import GateFirstExpertsMapping, StackExpertsMapping, mapping_to_dict
from surogate.dsl.models.qwen3_5_moe import Qwen3_5MoECausalModel, Qwen3_5MoEConditionalModel
from surogate.dsl.py_compiler import _serialize_hf_spec

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
    if isinstance(mapping_value, GateFirstExpertsMapping):
        return [mapping_value.source.replace("{layer}", str(layer))]
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


def _all_mappings(cls) -> dict:
    mappings = dict(getattr(cls, "_hf_block_mappings_", {}) or {})
    hf = getattr(cls, "_hf_mapping_", None)
    if hf is not None:
        mappings.update(getattr(hf, "mappings", {}) or {})
    return mappings


@pytest.mark.parametrize("model", [Qwen3_5MoECausalModel, Qwen3_5MoEConditionalModel])
def test_qwen3_5_moe_routed_experts_are_declared_gate_first(model):
    """HF reads experts.gate_up_proj as `gate, up = chunk(2)`; the runtime's SwiGLU reads
    [up | gate]. Read as stored, every routed expert computed gate * silu(up) (#270)."""
    mapping = model._hf_block_mappings_["experts_gate_up"]
    assert isinstance(mapping, GateFirstExpertsMapping)
    assert mapping.source.endswith("mlp.experts.gate_up_proj")
    # down_proj has no halves: it maps straight through.
    assert isinstance(model._hf_block_mappings_["experts_down"], str)
    # What the C++ loader receives: a direct mapping that carries the flag.
    assert _serialize_hf_spec(mapping) == {"type": "direct", "source": mapping.source, "gate_first": True}
    assert mapping_to_dict(mapping) == {"kind": "direct", "path": mapping.source, "gate_first": True}


def test_only_qwen3_5_moe_declares_gate_first_experts():
    """Every other model's expert layout is unchanged: the swap is opt-in per model."""
    declared = set()
    for name in dsl_models.__all__:
        cls = getattr(dsl_models, name)
        if not inspect.isclass(cls):
            continue
        for key, value in _all_mappings(cls).items():
            if isinstance(value, GateFirstExpertsMapping):
                declared.add((name, key))
    assert declared == {
        ("Qwen3_5MoECausalModel", "experts_gate_up"),
        ("Qwen3_5MoEConditionalModel", "experts_gate_up"),
    }


@pytest.mark.parametrize("block", ["Qwen3_5MoEAttentionBlock", "Qwen3_5MoELinearBlock"])
def test_qwen3_5_moe_renormalises_the_top_k_routing_weights(block):
    """HF's Qwen3_5MoeTopKRouter divides the top-k softmax weights by their sum, always (the config
    carries no norm_topk_prob). Not renormalising weighted every routed expert by the top-k's share
    of the softmax: on Qwen3.6-35B-A3B the first layer's output was at cos 0.98 against HF."""
    from surogate.dsl.blocks import qwen3_5_moe as blocks

    cls = getattr(blocks, block)
    assert cls.schema.routing.norm_topk_prob is True
    dims = dict(d_model=64, d_ff=32, num_experts=8, num_experts_per_tok=2, shared_expert_intermediate=32)
    if block == "Qwen3_5MoEAttentionBlock":
        dims.update(num_query_heads=2, num_kv_heads=1, head_size=32, max_seq=128)
    else:
        dims.update(linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
                    linear_num_value_heads=4)
    assert cls(**dims).moe.norm_topk_prob is True
