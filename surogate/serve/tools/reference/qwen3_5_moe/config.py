"""Checkpoint configuration for the hybrid MoE reference program."""

from dataclasses import asdict, dataclass

from ..qwen3_5.config import (
    ModelConfig as DenseConfig,
    VisionConfig,
    vision_config_from_declared,
    model_config_from_declared as dense_config_from_declared,
)


@dataclass(frozen=True)
class ModelConfig(DenseConfig):
    # Defaults are retained solely for the standalone numerical fixtures below.
    hidden: int = 2048
    layers: int = 40
    intermediate: int = 512
    q_heads: int = 16
    gdn_v_heads: int = 32
    experts: int = 256
    experts_per_token: int = 8
    expert_intermediate: int = 512
    shared_intermediate: int = 512
    routed_scale: float = 1.0
    conv_state_bytes: int = 2


def model_config_from_declared(declared, *, layer_types) -> ModelConfig:
    dense = dense_config_from_declared(declared, layer_types=layer_types)
    required = ("experts", "experts_per_token", "shared_intermediate", "routed_scale")
    if any(key not in declared or declared[key] <= 0 for key in required):
        raise ValueError("missing or invalid MoE checkpoint geometry")
    if declared["experts_per_token"] > declared["experts"]:
        raise ValueError("experts_per_token exceeds checkpoint expert count")
    return ModelConfig(
        **asdict(dense),
        experts=int(declared["experts"]),
        experts_per_token=int(declared["experts_per_token"]),
        expert_intermediate=int(declared["intermediate"]),
        shared_intermediate=int(declared["shared_intermediate"]),
        routed_scale=float(declared["routed_scale"]),
    )


# Explicit numerical fixtures, never used to resolve loaded artifact configuration.
CFG = ModelConfig()
VISION_CFG = VisionConfig(out_hidden=CFG.hidden)
ATTN_SCALE = CFG.attention_scale
GDN_SCALE = CFG.gdn_scale

__all__ = [
    "ATTN_SCALE",
    "CFG",
    "GDN_SCALE",
    "ModelConfig",
    "VISION_CFG",
    "VisionConfig",
    "model_config_from_declared",
    "vision_config_from_declared",
]
