"""Serving objects derived from a GLM checkpoint's GGUF configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Sequence

from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    DIRECT_FORMATS,
    RESOURCE_SPECS,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    W8,
    tensor_spec,
)

ARCHITECTURE = "Glm5NextForConditionalGeneration"
GGUF_ARCHITECTURE = "glm5next"
TARGET_KEY = "glm5_next"
MODEL_ID = "glm5-next"
WEIGHTS_ID = "w8-mhc-v1"
CAPABILITIES = ("text",)

#: The Hub repository the frontend (tokenizer, chat template, generation config) comes from.
#: The weights are not fetched from it -- they are the GGUF's -- but a GGUF carries a token
#: table and not a `tokenizer.json`, and the engine's frontend reads the latter.


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions a GLM-5.3 checkpoint states about itself."""

    hidden: int
    layers: int
    query_heads: int
    dense_intermediate: int
    expert_intermediate: int
    shared_intermediate: int
    vocab: int
    experts: int
    experts_per_token: int
    routed_scale: float
    swiglu_limit: float
    hc_streams: int
    hc_sinkhorn_iterations: int
    hc_epsilon: float
    kda_heads: int
    kda_head_dim: int
    kda_conv_kernel: int
    kda_gate_rank: int
    kda_lower_bound: float
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    rms_epsilon: float
    index_topk: int
    attention_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    nextn_layers: int
    max_context: int
    index_pool: int
    declared: declaration.Declaration = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from surogate.serve.convert.common.checkpoint import positive_int

        required = ("hidden", "layers", "query_heads", "dense_intermediate", "expert_intermediate",
                    "vocab", "experts", "experts_per_token", "hc_streams", "hc_sinkhorn_iterations",
                    "kda_heads", "kda_head_dim", "kda_conv_kernel", "kda_gate_rank", "q_lora_rank",
                    "kv_lora_rank", "qk_head_dim", "v_head_dim", "max_context", "index_topk", "index_pool")
        for name in required:
            positive_int({name: getattr(self, name)}, name)
        if self.layers > 256 or self.nextn_layers not in (0, 1):
            raise ValueError("GLM serving supports at most 256 text layers and one NextN layer")
        if self.experts_per_token > self.experts:
            raise ValueError("expert_used_count exceeds expert_count")
        if self.shared_intermediate < 0 or self.shared_intermediate % self.expert_intermediate:
            raise ValueError("shared expert width must be a nonnegative multiple of expert width")
        for name in ("rms_epsilon", "hc_epsilon", "routed_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")
        if not math.isfinite(self.kda_lower_bound) or self.kda_lower_bound >= 0:
            raise ValueError("kda.gate_lower_bound must be finite and negative")
        for schedule in (self.attention_layers, self.dense_layers):
            if len(set(schedule)) != len(schedule) or any(i < 0 or i >= self.layers for i in schedule):
                raise ValueError("invalid GLM layer schedule")
        if self.dense_layers != tuple(range(len(self.dense_layers))):
            raise ValueError("GLM GGUF serving requires leading dense layers")
        object.__setattr__(self, "declared", declaration.declare(ARCHITECTURE, config_from_geometry(self)))

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple("full_attention" if self.is_attention(i) else "linear_attention"
                     for i in range(self.layers))

    @property
    def serving_context(self) -> int:
        # Without the sparse indexer, all complete pools and the unfinished tail fit here.
        return min(self.max_context, self.index_topk + self.index_pool - 1)

    @property
    def residual(self) -> int:
        """Hyper-connections carry `hc_streams` copies of the hidden state."""
        return self.hc_streams * self.hidden

    @property
    def hc_mix_rows(self) -> int:
        """`pre`, `post` and the stream x stream combine, from one projection."""
        return (2 + self.hc_streams) * self.hc_streams

    @property
    def kda_dim(self) -> int:
        return self.kda_heads * self.kda_head_dim

    @property
    def query_dim(self) -> int:
        return self.query_heads * self.qk_head_dim

    @property
    def value_dim(self) -> int:
        return self.query_heads * self.v_head_dim

    def is_attention(self, layer: int) -> bool:
        return layer in self.attention_layers

    def is_dense(self, layer: int) -> bool:
        return layer in self.dense_layers


def geometry_from_gguf(kv: Callable[[str, Any], Any]) -> Geometry:
    """The checkpoint's dimensions, from its own key-values.

    `attention.head_count_kv` is per layer and is one exactly where the layer attends, which is
    how the file states its schedule; `leading_dense_block_count` states where the mixture
    starts. Reading either from a pattern instead would be assuming a regularity the file is
    perfectly capable of contradicting.
    """
    def need(key: str) -> Any:
        value = kv(f"{GGUF_ARCHITECTURE}.{key}", None)
        if value is None:
            raise ValueError(
                f"this GGUF does not state `{GGUF_ARCHITECTURE}.{key}`; a GLM-5.3 artifact "
                f"cannot be shaped without it"
            )
        return value

    def integer(value):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2147483647:
            raise ValueError("GGUF dimensions must be nonnegative int32 values")
        return value

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("GGUF execution settings must be finite numbers")
        return value

    blocks = integer(need("block_count"))
    nextn = integer(kv(f"{GGUF_ARCHITECTURE}.nextn_predict_layers", 0))
    layers = blocks - nextn
    kv_heads = list(need("attention.head_count_kv"))
    if len(kv_heads) != blocks:
        raise ValueError(
            f"`attention.head_count_kv` has {len(kv_heads)} entries for {blocks} blocks; it is "
            f"this file's attention schedule and has to name every one"
        )
    leading_dense = integer(need("leading_dense_block_count"))
    if not 0 <= leading_dense <= layers:
        raise ValueError("leading_dense_block_count must be within the text layer count")
    if any(integer(value) not in (0, 1) for value in kv_heads):
        raise ValueError("GLM absorbed attention requires head_count_kv entries of zero or one")
    clamps = list(need("swiglu_clamp_exp"))
    shared_clamps = list(need("swiglu_clamp_shexp"))
    if len(clamps) != blocks or len(shared_clamps) != blocks or len(set(clamps + shared_clamps)) != 1:
        raise ValueError("GLM serving currently requires one SwiGLU clamp shared by all layers and experts")
    if integer(need("rope.dimension_count")) != 0:
        raise ValueError(
            "this checkpoint rotates its latent attention; every released GLM-5.3 is the NoPE "
            "variant and the served attention applies no rotary at all"
        )
    return Geometry(
        hidden=integer(need("embedding_length")),
        layers=layers,
        query_heads=integer(need("attention.head_count")),
        dense_intermediate=integer(need("feed_forward_length")),
        expert_intermediate=integer(need("expert_feed_forward_length")),
        shared_intermediate=integer(kv(f"{GGUF_ARCHITECTURE}.expert_shared_feed_forward_length", 0))
        * integer(kv(f"{GGUF_ARCHITECTURE}.expert_shared_count", 0)),
        vocab=integer(need("vocab_size")),
        experts=integer(need("expert_count")),
        experts_per_token=integer(need("expert_used_count")),
        routed_scale=number(kv(f"{GGUF_ARCHITECTURE}.expert_weights_scale", 1.0)),
        swiglu_limit=number(clamps[0]),
        hc_streams=integer(need("hyper_connection.count")),
        hc_sinkhorn_iterations=integer(need("hyper_connection.sinkhorn_iterations")),
        hc_epsilon=number(need("hyper_connection.epsilon")),
        kda_heads=integer(need("attention.head_count")),
        kda_head_dim=integer(need("kda.head_dim")),
        kda_conv_kernel=integer(need("ssm.conv_kernel")),
        kda_gate_rank=integer(need("kda.head_dim")),
        kda_lower_bound=number(need("kda.gate_lower_bound")),
        q_lora_rank=integer(need("attention.q_lora_rank")),
        kv_lora_rank=integer(need("attention.kv_lora_rank")),
        qk_head_dim=integer(need("attention.key_length_mla")),
        v_head_dim=integer(need("attention.value_length_mla")),
        rms_epsilon=number(need("attention.layer_norm_rms_epsilon")),
        index_topk=integer(need("attention.indexer.top_k")),
        attention_layers=tuple(i for i in range(layers) if int(kv_heads[i]) > 0),
        dense_layers=tuple(range(min(leading_dense, layers))),
        nextn_layers=nextn,
        max_context=integer(need("context_length")),
        index_pool=integer(need("attention.indexer.kpool")),
    )


def config_from_geometry(geometry: Geometry) -> dict[str, Any]:
    """The geometry in the spelling the training declaration reads.

    The GGUF's key-values and training's config are two names for the same numbers; this is the
    one place that translates, so the declaration stays the single statement of what an artifact
    holds and nothing downstream needs to know the file was a GGUF.
    """
    # Nested under `text_config`, which is where the declaration's bindings read every text
    # dimension from (`d_model="text_config.hidden_size"`); a flat spelling binds nothing and
    # the declaration falls back to its compiled defaults, which happen to be the released
    # checkpoint's -- silently right for that one file and wrong for any other.
    text = {
        "hidden_size": geometry.hidden,
        "num_hidden_layers": geometry.layers,
        "num_attention_heads": geometry.query_heads,
        "num_key_value_heads": geometry.query_heads,
        "intermediate_size": geometry.dense_intermediate,
        "moe_intermediate_size": geometry.expert_intermediate,
        "n_routed_experts": geometry.experts,
        "num_experts_per_tok": geometry.experts_per_token,
        "n_shared_experts": (
            geometry.shared_intermediate // geometry.expert_intermediate
            if geometry.expert_intermediate
            else 0
        ),
        "vocab_size": geometry.vocab,
        "kv_lora_rank": geometry.kv_lora_rank,
        "q_lora_rank": geometry.q_lora_rank,
        "qk_rope_head_dim": 0,
        "qk_nope_head_dim": geometry.qk_head_dim,
        "v_head_dim": geometry.v_head_dim,
        "routed_scaling_factor": geometry.routed_scale,
        "swiglu_limit": geometry.swiglu_limit,
        "norm_topk_prob": True,
        "rms_norm_eps": geometry.rms_epsilon,
        "hc_mult": geometry.hc_streams,
        "hc_eps": geometry.hc_epsilon,
        "hc_sinkhorn_iters": geometry.hc_sinkhorn_iterations,
        "linear_head_dim": geometry.kda_head_dim,
        "linear_num_heads": geometry.kda_heads,
        "linear_conv_kernel_dim": geometry.kda_conv_kernel,
        "linear_lower_bound": geometry.kda_lower_bound,
        "index_topk": geometry.index_topk,
        "index_kpool": geometry.index_pool,
        "index_kpool_always_select_tail": True,
        "max_position_embeddings": geometry.max_context,
        "nextn_predict_layers": geometry.nextn_layers,
        "first_k_dense_replace": len(geometry.dense_layers),
        "layer_types": [
            "deepseek_sparse_attention" if geometry.is_attention(i) else "linear_attention"
            for i in range(geometry.layers)
        ],
        "mlp_layer_types": [
            "dense" if geometry.is_dense(i) else "sparse" for i in range(geometry.layers)
        ],
    }
    return {
        "architectures": [ARCHITECTURE],
        "model_type": "glm5_next",
        "text_config": text,
        "tie_word_embeddings": False,
    }


def declared_objects(geometry: Geometry) -> list[declaration.DeclaredObject]:
    return list(geometry.declared.objects(capabilities=set(CAPABILITIES)))


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    `quantised` is a storage class, not a width, and this target's choice is group-wise int8 --
    for the objects that do not stay in the file's own K-quant format, which is most of the
    bytes. The norms, the router bias, the convolution taps, the per-head decay and the three
    hyper-connection scalars stay in the precision the declaration names: a router that picks
    a subset of its experts and a decay that is exponentiated are both places where a coarse width
    costs far more than the bytes it saves.
    """
    out: list[TensorSpec] = []
    for obj in objects:
        numeric = W8 if obj.format == "quantised" else obj.format.upper()
        if numeric not in DIRECT_FORMATS and numeric != W8:
            numeric = BF16
        out.append(tensor_spec(obj.name, tuple(obj.shape), numeric))
    return tuple(out)


def build_tensor_specs(geometry: Geometry) -> tuple[TensorSpec, ...]:
    return tensor_specs(declared_objects(geometry))


def object_specs(geometry: Geometry) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(geometry)


__all__ = [
    "ARCHITECTURE",
    "BF16",
    "CAPABILITIES",
    "GGUF_ARCHITECTURE",
    "Geometry",
    "MODEL_ID",
    "RESOURCE_SPECS",
    "ResourceSpec",
    "TARGET_KEY",
    "TensorSpec",
    "W8",
    "WEIGHTS_ID",
    "build_tensor_specs",
    "config_from_geometry",
    "declared_objects",
    "geometry_from_gguf",
    "object_specs",
    "tensor_specs",
]
