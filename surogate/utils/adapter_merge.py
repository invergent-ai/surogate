#!/usr/bin/env python3
"""
Merge LoRA adapter into base model by operating directly on safetensors files.

Copies the original base model files, then applies LoRA deltas in-place,
preserving the exact original key structure for compatibility with vLLM
and other serving frameworks.
"""

import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from surogate.utils.logger import get_logger

logger = get_logger()


def load_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    logger.info(f"Loading adapter weights from {adapter_file}...")
    weights = load_file(adapter_file)
    logger.info(f"Loaded {len(weights)} adapter tensors")
    return weights


def _strip_adapter_key(key: str) -> str:
    """Strip PEFT prefix and .weight suffix from an adapter key.

    Input:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    Output: model.layers.0.self_attn.q_proj.lora_A
    """
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    return key


def _build_lora_lookup(
    adapter_weights: dict[str, torch.Tensor],
    safetensor_keys: list[str],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build a mapping from base weight safetensor key -> (lora_A, lora_B).

    The adapter keys (PEFT format) are mapped to match the actual safetensor
    keys from the base model, auto-detecting any prefix differences
    (e.g. model.layers.* vs model.language_model.layers.*).
    """
    # Step 1: strip PEFT prefix, group into (lora_A, lora_B) pairs keyed by module
    stripped = {}
    for key, tensor in adapter_weights.items():
        stripped[_strip_adapter_key(key)] = tensor

    # Collect lora pairs: module_name -> (lora_A, lora_B)
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in stripped:
        if key.endswith(".lora_A"):
            module = key[: -len(".lora_A")]
            lora_b_key = module + ".lora_B"
            if lora_b_key in stripped:
                pairs[module] = (stripped[key], stripped[lora_b_key])

    if not pairs:
        return {}

    # Step 2: detect prefix remapping by probing one pair against safetensor keys
    sample_module = next(iter(pairs))
    expected_st_key = sample_module + ".weight"

    safetensor_key_set = set(safetensor_keys)
    prefix_remap = ("", "")  # (from_prefix, to_prefix)

    if expected_st_key not in safetensor_key_set:
        # Find the matching safetensor key by suffix
        suffix = expected_st_key.split(".", 1)[1] if "." in expected_st_key else expected_st_key
        for st_key in safetensor_keys:
            if st_key.endswith(suffix):
                actual_prefix = st_key[: len(st_key) - len(suffix)]
                expected_prefix = expected_st_key[: len(expected_st_key) - len(suffix)]
                if actual_prefix != expected_prefix:
                    prefix_remap = (expected_prefix, actual_prefix)
                break

    # Step 3: build final lookup keyed by safetensor key (with .weight suffix)
    from_pfx, to_pfx = prefix_remap
    lookup: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module, (lora_a, lora_b) in pairs.items():
        st_key = module + ".weight"
        if from_pfx and st_key.startswith(from_pfx):
            st_key = to_pfx + st_key[len(from_pfx) :]
        lookup[st_key] = (lora_a, lora_b)

    return lookup


def _build_router_lookup(
    adapter_weights: dict[str, torch.Tensor],
    safetensor_keys: list[str],
    prefix_remap: tuple[str, str],
) -> dict[str, torch.Tensor]:
    """Build lookup for trained MoE router weights (direct replacement, not LoRA)."""
    from_pfx, to_pfx = prefix_remap
    lookup: dict[str, torch.Tensor] = {}

    for key, tensor in adapter_weights.items():
        stripped = _strip_adapter_key(key)
        if ".mlp.gate" not in stripped or ".lora_" in stripped:
            continue
        st_key = stripped + ".weight"
        if from_pfx and st_key.startswith(from_pfx):
            st_key = to_pfx + st_key[len(from_pfx) :]
        lookup[st_key] = tensor

    return lookup


def merge_lora_into_linear(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    lora_rank: int,
    scaling: float | None = None,
) -> torch.Tensor:
    """Merge LoRA weights into base linear layer: W' = W + (B @ A) * scaling."""
    if scaling is None:
        scaling = lora_alpha / lora_rank

    # lora_A: [rank, in_features], lora_B: [out_features, rank]
    orig_dtype = base_weight.dtype
    # Compute in float32 for accuracy
    delta = (lora_B.float() @ lora_A.float()) * scaling
    merged = base_weight.float() + delta
    return merged.to(orig_dtype)


def merge_adapter(
    base_model_path: str, adapter_path: str, output_path: str, max_shard_size: str = "5GB", cpu_offload: bool = True
) -> None:
    """
    Merge LoRA adapter into base model by operating directly on safetensors files.

    Copies the base model's safetensors to output_path, then applies LoRA deltas
    in-place. This preserves the exact original key structure, ensuring
    compatibility with vLLM and other serving frameworks.

    Args:
        base_model_path: Path to the base model directory
        adapter_path: Path to the adapter directory
        output_path: Output directory for merged model
        max_shard_size: Unused (kept for API compat)
        cpu_offload: Unused (always operates on CPU)
    """
    # Load adapter config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    lora_alpha = adapter_config["lora_alpha"]
    lora_rank = adapter_config["r"]

    # Load adapter weights
    adapter_weights = load_adapter_weights(adapter_path)

    # Find all safetensor files in the base model
    st_files = sorted(glob.glob(os.path.join(base_model_path, "*.safetensors")))
    st_files = [f for f in st_files if "adapter" not in os.path.basename(f)]
    if not st_files:
        raise FileNotFoundError(f"No safetensors files found in {base_model_path}")

    # Collect all base model keys across shards
    all_base_keys: list[str] = []
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            all_base_keys.extend(f.keys())

    # Build LoRA lookup keyed by safetensor weight key
    lora_lookup = _build_lora_lookup(adapter_weights, all_base_keys)
    if not lora_lookup:
        logger.warning("No LoRA pairs found! Check adapter key format vs base model keys.")
        return

    logger.info(f"Found {len(lora_lookup)} LoRA pairs to merge")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Copy all non-safetensor files from base model (config, tokenizer, etc.)
    for item in os.listdir(base_model_path):
        src = os.path.join(base_model_path, item)
        dst = os.path.join(output_path, item)
        if os.path.isfile(src) and not item.endswith(".safetensors"):
            shutil.copy2(src, dst)

    # Process each safetensor shard: copy then merge LoRA in-place
    merged_count = 0
    router_count = 0

    for st_file in st_files:
        shard_name = os.path.basename(st_file)
        output_shard = os.path.join(output_path, shard_name)

        # Load all tensors from this shard
        with safe_open(st_file, framework="pt", device="cpu") as f:
            shard_keys = list(f.keys())
            shard_tensors = {key: f.get_tensor(key) for key in shard_keys}

        # Apply LoRA deltas
        shard_modified = False
        for key in shard_keys:
            if key in lora_lookup:
                lora_a, lora_b = lora_lookup[key]
                base_weight = shard_tensors[key]
                shard_tensors[key] = merge_lora_into_linear(base_weight, lora_a, lora_b, lora_alpha, lora_rank)
                merged_count += 1
                shard_modified = True

        if shard_modified:
            # Save modified shard
            save_file(shard_tensors, output_shard)
        else:
            # Just copy the file as-is (faster than re-serializing)
            shutil.copy2(st_file, output_shard)

    logger.info(f"Merged {merged_count}/{len(lora_lookup)} LoRA weights")

    if merged_count != len(lora_lookup):
        missing = set(lora_lookup.keys()) - {
            k for st_file in st_files for k in safe_open(st_file, framework="pt", device="cpu").keys()
        }
        for key in missing:
            logger.warning(f"LoRA target not found in base model: {key}")

    # Move adapter files into a subdirectory so vLLM doesn't misdetect as PEFT adapter
    adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
    if any(os.path.exists(os.path.join(output_path, f)) for f in adapter_files):
        adapter_subdir = os.path.join(output_path, "adapter")
        os.makedirs(adapter_subdir, exist_ok=True)
        for adapter_file in adapter_files:
            src = os.path.join(output_path, adapter_file)
            if os.path.exists(src):
                shutil.move(src, os.path.join(adapter_subdir, adapter_file))
        logger.info(f"Moved adapter files to {adapter_subdir}/")

    logger.info(f"Merged model saved to {output_path}")
