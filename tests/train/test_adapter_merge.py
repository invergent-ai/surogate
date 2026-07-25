import pytest
import torch

from surogate.utils.adapter_merge import _build_lora_lookup


def _adapter_pair(module: str):
    return {
        f"base_model.model.{module}.lora_A.weight": torch.ones(2, 3),
        f"base_model.model.{module}.lora_B.weight": torch.ones(4, 2),
    }


def test_lora_lookup_prefers_qwen_language_model_over_mtp():
    adapter = _adapter_pair("model.layers.0.mlp.down_proj")
    base_keys = [
        "mtp.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]

    lookup = _build_lora_lookup(adapter, base_keys)

    assert list(lookup) == ["model.language_model.layers.0.mlp.down_proj.weight"]


def test_lora_lookup_keeps_unambiguous_prefix_remap():
    adapter = _adapter_pair("model.layers.3.self_attn.q_proj")
    base_keys = ["language_model.model.layers.3.self_attn.q_proj.weight"]

    lookup = _build_lora_lookup(adapter, base_keys)

    assert list(lookup) == ["language_model.model.layers.3.self_attn.q_proj.weight"]


def test_lora_lookup_rejects_ambiguous_suffix_matches():
    adapter = _adapter_pair("other.layers.0.mlp.down_proj")
    base_keys = [
        "model.layers.0.mlp.down_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ]

    with pytest.raises(ValueError, match="ambiguous LoRA target"):
        _build_lora_lookup(adapter, base_keys)
