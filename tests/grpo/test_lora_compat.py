import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from surogate.utils.lora_compat import ensure_surogate_lora_compat, ensure_vllm_lora_compat


def _write_base(path: Path, architecture: str) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": [architecture]}),
        encoding="utf-8",
    )


def _write_adapter(path: Path) -> None:
    path.mkdir()
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 3),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
        },
        str(path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )


def test_qwen35_conditional_adapter_keys_are_remapped(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _write_base(base, "Qwen3_5ForConditionalGeneration")
    _write_adapter(adapter)

    assert ensure_vllm_lora_compat(adapter, base)
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    assert set(tensors) == {
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight",
    }
    assert not ensure_vllm_lora_compat(adapter, base)

    assert ensure_surogate_lora_compat(adapter, base)
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    assert set(tensors) == {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }
    assert not ensure_surogate_lora_compat(adapter, base)


def test_text_only_architecture_is_not_remapped(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _write_base(base, "Qwen3_5ForCausalLM")
    _write_adapter(adapter)

    assert not ensure_vllm_lora_compat(adapter, base)
    assert not ensure_surogate_lora_compat(adapter, base)
    tensors = load_file(str(adapter / "adapter_model.safetensors"))
    assert all("language_model" not in key for key in tensors)
