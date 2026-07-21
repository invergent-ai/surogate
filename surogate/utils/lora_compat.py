"""Compatibility fixes for exported PEFT LoRA adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

_QWEN35_MM_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
_EXPORTED_PREFIX = "base_model.model.model."
_VLLM_PREFIX = "base_model.model.model.language_model."


def _base_architectures(base_model_dir: str | Path | None) -> set[str]:
    if base_model_dir is None:
        return set()
    config_path = Path(base_model_dir) / "config.json"
    if not config_path.is_file():
        return set()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(name) for name in config.get("architectures", [])}


def ensure_vllm_lora_compat(
    adapter_dir: str | Path,
    base_model_dir: str | Path | None,
) -> bool:
    """Normalize Qwen3.5 multimodal adapter keys for vLLM dynamic loading.

    Surogate trains the text model directly and exports ``model.layers`` keys.
    The Hugging Face Qwen3.5 conditional-generation wrapper nests those layers
    under ``model.language_model``. vLLM uses that wrapper path when attaching
    LoRA weights, so the missing component otherwise makes the adapter a no-op.

    Returns ``True`` only when the safetensors file was rewritten.
    """
    if not (_base_architectures(base_model_dir) & _QWEN35_MM_ARCHITECTURES):
        return False

    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_path.is_file():
        return False

    with safe_open(adapter_path, framework="pt") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()

    needs_remap = any(key.startswith(_EXPORTED_PREFIX) and not key.startswith(_VLLM_PREFIX) for key in keys)
    if not needs_remap:
        return False

    tensors = load_file(str(adapter_path), device="cpu")
    remapped = {}
    for key, tensor in tensors.items():
        if key.startswith(_EXPORTED_PREFIX) and not key.startswith(_VLLM_PREFIX):
            key = _VLLM_PREFIX + key[len(_EXPORTED_PREFIX) :]
        if key in remapped:
            raise ValueError(f"LoRA key collision while normalizing {adapter_path}: {key}")
        remapped[key] = tensor

    temporary_path = adapter_path.with_suffix(".safetensors.tmp")
    save_file(remapped, str(temporary_path), metadata=metadata)
    os.replace(temporary_path, adapter_path)
    return True


def ensure_surogate_lora_compat(
    adapter_dir: str | Path,
    base_model_dir: str | Path | None,
) -> bool:
    """Restore Qwen3.5 multimodal adapter keys before native checkpoint resume.

    Native Surogate checkpoints address the text tower as ``model.layers``.
    Older checkpoints may have been normalized in place for vLLM, which uses
    ``model.language_model.layers``. Convert those keys back before calling the
    native checkpoint loader so resume cannot silently skip adapter weights.

    Returns ``True`` only when the safetensors file was rewritten.
    """
    if not (_base_architectures(base_model_dir) & _QWEN35_MM_ARCHITECTURES):
        return False

    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_path.is_file():
        return False

    with safe_open(adapter_path, framework="pt") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()

    if not any(key.startswith(_VLLM_PREFIX) for key in keys):
        return False

    tensors = load_file(str(adapter_path), device="cpu")
    remapped = {}
    for key, tensor in tensors.items():
        if key.startswith(_VLLM_PREFIX):
            key = _EXPORTED_PREFIX + key[len(_VLLM_PREFIX) :]
        if key in remapped:
            raise ValueError(f"LoRA key collision while restoring {adapter_path}: {key}")
        remapped[key] = tensor

    temporary_path = adapter_path.with_suffix(".safetensors.tmp")
    save_file(remapped, str(temporary_path), metadata=metadata)
    os.replace(temporary_path, adapter_path)
    return True
