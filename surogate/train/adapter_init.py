"""Validated LoRA parent initialization shared by SFT trainer paths."""

from __future__ import annotations

import json
from pathlib import Path


def configure_initial_adapter(config, trainer: object, *, fresh_run: bool) -> Path | None:
    """Configure merged initialization or return a trainable parent to import."""
    adapter_path = getattr(config, "adapter_path", None)
    if not adapter_path:
        return None
    path = Path(adapter_path).expanduser().resolve()
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    missing = [item.name for item in (config_path, weights_path) if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"initial adapter is incomplete at {path}: missing {', '.join(missing)}"
        )

    mode = getattr(config, "adapter_init_mode", "merge")
    if mode == "merge":
        set_adapter_path = getattr(trainer, "set_adapter_path", None)
        if not callable(set_adapter_path):
            raise RuntimeError("SFT trainer does not support merged adapter initialization")
        set_adapter_path(str(path))
        return None
    if mode != "trainable":
        raise ValueError("adapter_init_mode must be 'merge' or 'trainable'")
    if not fresh_run:
        return None

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(adapter_config.get("r", -1)) != int(config.lora_rank):
        raise ValueError("trainable parent adapter rank does not match trainer LoRA rank")
    if float(adapter_config.get("lora_alpha", -1.0)) != float(config.lora_alpha):
        raise ValueError("trainable parent adapter alpha does not match trainer LoRA alpha")
    if set(adapter_config.get("target_modules", ())) != set(config.lora_target_modules):
        raise ValueError("trainable parent adapter targets do not match trainer LoRA targets")
    return weights_path


def import_initial_trainable_adapter(trainer: object, weights_path: Path | None) -> None:
    if weights_path is None:
        return
    import_adapter = getattr(trainer, "import_adapter", None)
    if not callable(import_adapter):
        raise RuntimeError("SFT trainer does not support trainable adapter initialization")
    import_adapter(str(weights_path))
