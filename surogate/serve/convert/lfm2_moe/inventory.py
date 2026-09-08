"""LFM2-MoE artifact objects derived from the checkpoint and training DSL."""

from dataclasses import dataclass
from typing import Any, Mapping

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.checkpoint import positive_int
from surogate.serve.convert.lfm2.inventory import (
    Geometry as TextGeometry, RESOURCE_SPECS, declared_objects, tensor_specs, execution_config,
)

ARCHITECTURE = "Lfm2MoeForCausalLM"
TARGET_KEY = MODEL_ID = "lfm2_moe"
WEIGHTS_ID = "groupwise-int"
CAPABILITIES = ("text",)


@dataclass(frozen=True, slots=True)
class Geometry(TextGeometry):
    moe_intermediate: int
    experts: int
    experts_per_token: int
    dense_layers: int
    routed_scale: float


def geometry_from_config(config: Mapping[str, Any]) -> Geometry:
    source = execution_config(config)
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "conv_L_cache", "intermediate_size",
                "moe_intermediate_size", "num_experts", "num_experts_per_tok"):
        positive_int(source, key)
    if source["hidden_size"] % source["num_attention_heads"]:
        raise ValueError("LFM2-MoE hidden_size must be divisible by num_attention_heads")
    if source["num_attention_heads"] % source["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if source["num_experts_per_tok"] > source["num_experts"]:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    dense = source.get("num_dense_layers")
    if isinstance(dense, bool) or not isinstance(dense, int) or not 0 <= dense < source["num_hidden_layers"]:
        raise ValueError("num_dense_layers must be an integer in [0, num_hidden_layers)")
    if source.get("conv_bias", False):
        raise ValueError("LFM2 serving does not support convolution bias")
    if not source.get("norm_topk_prob", True):
        raise ValueError("LFM2-MoE serving requires norm_topk_prob=true")
    if not source.get("use_expert_bias", True):
        raise ValueError("LFM2-MoE serving requires use_expert_bias=true")
    if float(source.get("routed_scaling_factor", 1.0)) != 1.0:
        raise ValueError("LFM2-MoE serving currently requires routed_scaling_factor=1")
    if "layer_types" not in source and "full_attn_idxs" not in source:
        raise ValueError("LFM2-MoE config must declare layer_types or full_attn_idxs")
    source["rms_norm_eps"] = source["norm_eps"]
    declared = declaration.declare(ARCHITECTURE, source)
    resolved = declared.config
    schedule = declaration.block_types(resolved, declared.model)
    return Geometry(
        hidden=int(resolved["d_model"]), layers=int(resolved["n_layers"]),
        query_heads=int(resolved["num_query_heads"]), kv_heads=int(resolved["num_kv_heads"]),
        head_dim=int(resolved["head_size"]), intermediate=int(resolved["d_ff"]),
        vocab=int(resolved["vocab_size"]), conv_kernel=int(resolved["conv_kernel"]),
        attention_layers=tuple(i for i, kind in enumerate(schedule) if kind.startswith("attention")),
        declared=declared, moe_intermediate=int(resolved["moe_d_ff"]),
        experts=int(resolved["num_experts"]), experts_per_token=int(resolved["num_experts_per_tok"]),
        dense_layers=dense, routed_scale=float(source.get("routed_scaling_factor", 1.0)),
    )
