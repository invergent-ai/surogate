#!/usr/bin/env python3
"""
Merge LoRA adapter into base model with memory-efficient CPU offloading.

This module handles both regular and MoE LoRA adapters.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from surogate.utils.logger import get_logger

logger = get_logger()


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter weights from safetensors."""
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    logger.info(f"Loading adapter weights from {adapter_file}...")
    weights = load_file(adapter_file)
    logger.info(f"Loaded {len(weights)} adapter tensors")
    return weights


def normalize_adapter_keys(adapter_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize adapter weight keys by stripping PEFT prefix and .weight suffix.

    Input:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    Output: model.layers.0.self_attn.q_proj.lora_A
    """
    normalized = {}
    for key, tensor in adapter_weights.items():
        clean_key = key
        if key.startswith("base_model.model."):
            clean_key = key[len("base_model.model."):]
        if clean_key.endswith(".weight"):
            clean_key = clean_key[:-len(".weight")]
        normalized[clean_key] = tensor
    return normalized


def merge_lora_into_linear(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    lora_alpha: float,
    lora_rank: int,
    scaling: Optional[float] = None
) -> torch.Tensor:
    """Merge LoRA weights into base linear layer: W' = W + (B @ A) * scaling."""
    if scaling is None:
        scaling = lora_alpha / lora_rank

    # Compute LoRA delta: B @ A
    # lora_A: [rank, in_features]
    # lora_B: [out_features, rank]
    # delta: [out_features, in_features]
    delta = (lora_B @ lora_A) * scaling

    # Add to base weight
    merged = base_weight + delta
    return merged


def merge_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    max_shard_size: str = "5GB",
    cpu_offload: bool = True
) -> None:
    """
    Merge LoRA adapter into base model with CPU offloading for memory efficiency.

    Args:
        base_model_path: Path to the base model directory
        adapter_path: Path to the adapter directory
        output_path: Output directory for merged model
        max_shard_size: Maximum shard size for saving (default: 5GB)
        cpu_offload: Use CPU offloading for memory efficiency (default: True)
    """
    # Load adapter config
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    lora_alpha = adapter_config["lora_alpha"]
    lora_rank = adapter_config["r"]
    target_modules = adapter_config["target_modules"]

    # Load base model config
    base_config = AutoConfig.from_pretrained(base_model_path)

    # Load and normalize adapter weights
    adapter_weights = load_adapter_weights(adapter_path)
    normalized_weights = normalize_adapter_keys(adapter_weights)

    # Load base model (CPU offload if needed)
    device_map = "cpu" if cpu_offload else "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    logger.info("Merging LoRA weights into base model...")
    # Build LoRA pair mapping: {base_key: (lora_A_key, lora_B_key)}
    lora_pairs = {}
    for key in normalized_weights:
        if key.endswith(".lora_A"):
            base_key = key.replace(".lora_A", "")
            lora_a_key = key
            lora_b_key = key.replace(".lora_A", ".lora_B")
            if lora_b_key in normalized_weights:
                lora_pairs[base_key] = (lora_a_key, lora_b_key)

    # Merge each LoRA pair into base model
    merged_count = 0
    base_state_dict = base_model.state_dict()

    for base_key, (lora_a_key, lora_b_key) in lora_pairs.items():
        # Base model has .weight suffix, but our keys don't
        base_key_with_weight = base_key + ".weight"
        if base_key_with_weight not in base_state_dict:
            logger.warning(f"Base weight not found: {base_key_with_weight}")
            continue

        base_weight = base_state_dict[base_key_with_weight]
        lora_A = normalized_weights[lora_a_key].to(base_weight.device)
        lora_B = normalized_weights[lora_b_key].to(base_weight.device)

        merged_weight = merge_lora_into_linear(
            base_weight, lora_A, lora_B,
            lora_alpha, lora_rank
        )

        base_state_dict[base_key_with_weight] = merged_weight
        merged_count += 1

        if merged_count % 100 == 0:
            logger.info(f"  Merged {merged_count}/{len(lora_pairs)} weights...")

    # Handle trained router weights (not LoRA, just direct replacement)
    # Router keys are like: model.layers.X.mlp.gate (without lora_A/B suffix)
    router_count = 0
    for key, tensor in normalized_weights.items():
        if ".mlp.gate" in key and ".lora_" not in key:
            # This is a trained router weight, replace directly
            base_key_with_weight = key + ".weight"
            if base_key_with_weight in base_state_dict:
                base_state_dict[base_key_with_weight] = tensor.to(
                    base_state_dict[base_key_with_weight].device
                )
                router_count += 1
            else:
                logger.warning(f"Router weight not found in base model: {base_key_with_weight}")

    if router_count > 0:
        logger.info(f"Replaced {router_count} trained router weights")

    # Load merged weights back into model
    base_model.load_state_dict(base_state_dict)

    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    base_model.save_pretrained(
        output_path,
        max_shard_size=max_shard_size,
        safe_serialization=True
    )

    # Copy tokenizer files
    logger.info("Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt"
    ]
    for filename in tokenizer_files:
        src = os.path.join(base_model_path, filename)
        if os.path.exists(src):
            import shutil
            shutil.copy(src, os.path.join(output_path, filename))

