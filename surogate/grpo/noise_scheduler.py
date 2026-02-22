"""QeRL Adaptive Quantization Noise (AQN) for GRPO inference weights.

Adds Gaussian noise to RMSNorm weights in the exported model/adapter before
the inference engine picks them up.  The noise standard deviation follows a
geometric decay schedule from sigma_start to sigma_end over num_stages
intervals, matching the QeRL paper (arXiv:2510.11696).

Usage:
    Called automatically by SurogateWeightBroadcast when noise_scheduler is
    enabled in the GRPO config.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from surogate.grpo.config import NoiseSchedulerConfig
from surogate.utils.logger import get_logger

logger = get_logger()

# Regex matching RMSNorm weight tensor names in HuggingFace format.
_NORM_WEIGHT_RE = re.compile(
    r"(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm"
    r"|post_feedforward_layernorm|final_layernorm|model\.norm"
    r"|backbone\.norm_f)"
    r"\.weight$"
)

# PEFT modules_to_save naming — short module names for adapter_config.json.
_PEFT_MODULES_TO_SAVE = ["input_layernorm", "post_attention_layernorm"]


def compute_sigma(step: int, total_steps: int, config: NoiseSchedulerConfig) -> float:
    """Compute the noise sigma for the current step.

    Geometric decay: sigma_i = sigma_start * (sigma_end/sigma_start)^(i/(N-2))
    where i is the stage index (0-based) and N = num_stages.
    The first interval has sigma=0 (no noise).
    """
    if not config.enabled or total_steps <= 0:
        return 0.0

    num_stages = int(config.num_stages)
    sigma_start = float(config.sigma_start)
    sigma_end = float(config.sigma_end)

    # Build geometric decay schedule (num_stages - 1 values)
    if num_stages <= 2:
        sigma_trend = [sigma_start]
    else:
        exponents = np.arange(num_stages - 1) / (num_stages - 2)
        sigma_trend = (sigma_start * (sigma_end / sigma_start) ** exponents).tolist()

    # Determine which interval the current step falls in
    step = min(step, total_steps)
    num_intervals = len(sigma_trend) + 1  # +1 for the no-noise first interval
    steps_per_interval = total_steps / num_intervals
    interval_id = int(step // steps_per_interval)

    # First interval: no noise
    if interval_id == 0:
        return 0.0

    sigma_id = min(interval_id - 1, len(sigma_trend) - 1)
    return sigma_trend[sigma_id]


def _add_noise_to_tensor(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise N(0, sigma^2) to a tensor, preserving dtype."""
    orig_dtype = t.dtype
    w = t.float()
    noise = torch.normal(mean=0.0, std=sigma, size=w.shape)
    return (w + noise).to(orig_dtype)


def inject_noise_into_safetensors(
    safetensors_path: Path,
    sigma: float,
) -> int:
    """Add Gaussian noise N(0, sigma^2) to RMSNorm weights in a safetensors file.

    Returns the number of tensors modified.
    """
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(safetensors_path), device="cpu")
    modified = 0

    for name in list(tensors.keys()):
        # Match PEFT-prefixed names too (base_model.model.xxx.weight)
        clean_name = name.replace("base_model.model.", "")
        if _NORM_WEIGHT_RE.search(clean_name):
            tensors[name] = _add_noise_to_tensor(tensors[name], sigma)
            modified += 1

    if modified > 0:
        save_file(tensors, str(safetensors_path))

    return modified


def inject_noise_adapter(
    adapter_dir: Path,
    base_model_dir: Path,
    sigma: float,
) -> int:
    """Inject noise into a LoRA adapter export, adding base model norms if needed.

    If the adapter doesn't already contain RMSNorm weights (i.e. norms are
    not in modules_to_save), reads them from the base model, adds noise, and
    appends them to the adapter safetensors with PEFT-compatible naming.

    Returns the number of tensors modified.
    """
    from safetensors.torch import load_file, save_file

    adapter_st = adapter_dir / "adapter_model.safetensors"
    if not adapter_st.exists():
        logger.warning(f"Adapter safetensors not found: {adapter_st}")
        return 0

    adapter_tensors = load_file(str(adapter_st), device="cpu")

    # Check if adapter already has norm weights
    has_norms = any(
        _NORM_WEIGHT_RE.search(k.replace("base_model.model.", ""))
        for k in adapter_tensors
    )

    if has_norms:
        # Norms already in adapter (modules_to_save) — just add noise
        return inject_noise_into_safetensors(adapter_st, sigma)

    # Norms not in adapter — read from base model and add noisy copies
    base_norms = _load_base_model_norms(base_model_dir)
    if not base_norms:
        logger.warning("No RMSNorm weights found in base model")
        return 0

    modified = 0
    modules_to_save_set = set()

    for hf_name, weight in base_norms.items():
        noisy = _add_noise_to_tensor(weight, sigma)

        # PEFT modules_to_save key format:
        # base_model.model.{path}.modules_to_save.default.weight
        # e.g. model.layers.0.input_layernorm.weight ->
        #      base_model.model.model.layers.0.input_layernorm.modules_to_save.default.weight
        parts = hf_name.rsplit(".weight", 1)
        peft_key = f"base_model.model.{parts[0]}.modules_to_save.default.weight"
        adapter_tensors[peft_key] = noisy
        modified += 1

        # Track module short names for adapter_config
        for suffix in _PEFT_MODULES_TO_SAVE:
            if suffix in hf_name:
                modules_to_save_set.add(suffix)

    if modified > 0:
        save_file(adapter_tensors, str(adapter_st))

        # Update adapter_config.json with modules_to_save
        config_path = adapter_dir / "adapter_config.json"
        if config_path.exists():
            with open(config_path) as f:
                adapter_config = json.load(f)
            adapter_config["modules_to_save"] = sorted(modules_to_save_set)
            with open(config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)

    return modified


def inject_noise_model(model_dir: Path, sigma: float) -> int:
    """Inject noise into a full model export's RMSNorm weights.

    Returns the total number of tensors modified.
    """
    total = 0
    for st_file in sorted(model_dir.glob("*.safetensors")):
        total += inject_noise_into_safetensors(st_file, sigma)
    return total


def _load_base_model_norms(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load RMSNorm weight tensors from a HuggingFace model directory."""
    from safetensors.torch import load_file

    norms: dict[str, torch.Tensor] = {}

    # Handle sharded models (model.safetensors.index.json)
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        # Collect norm weights and their shard files
        shard_keys: dict[str, list[str]] = {}  # shard_file -> [key, ...]
        for key, shard in index.get("weight_map", {}).items():
            if _NORM_WEIGHT_RE.search(key):
                shard_keys.setdefault(shard, []).append(key)
        for shard, keys in shard_keys.items():
            shard_path = model_dir / shard
            if shard_path.exists():
                all_tensors = load_file(str(shard_path), device="cpu")
                for key in keys:
                    if key in all_tensors:
                        norms[key] = all_tensors[key]
    else:
        # Single file model
        st_file = model_dir / "model.safetensors"
        if st_file.exists():
            all_tensors = load_file(str(st_file), device="cpu")
            for key, val in all_tensors.items():
                if _NORM_WEIGHT_RE.search(key):
                    norms[key] = val

    return norms
