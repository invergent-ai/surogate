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

# Sigma file written alongside LoRA adapter for vLLM worker to read.
NOISE_SIGMA_FILENAME = "qerl_sigma.json"


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


def write_noise_sigma(adapter_dir: Path, sigma: float) -> None:
    """Write the noise sigma to a JSON file alongside the LoRA adapter.

    The vLLM worker reads this file and applies noise in-place to the base
    model's RMSNorm weights on GPU, matching the QeRL reference implementation.
    """
    sigma_file = adapter_dir / NOISE_SIGMA_FILENAME
    with open(sigma_file, "w") as f:
        json.dump({"sigma": sigma}, f)


def inject_noise_model(model_dir: Path, sigma: float) -> int:
    """Inject noise into a full model export's RMSNorm weights.

    Returns the total number of tensors modified.
    """
    total = 0
    for st_file in sorted(model_dir.glob("*.safetensors")):
        total += inject_noise_into_safetensors(st_file, sigma)
    return total


