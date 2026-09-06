"""What a GLM-5.3-Flash serving artifact holds, and where its dimensions come from.

The GGUF *is* the source here. gguf-py has no `glm5next` name map, so there is no bridge to
route through and no synthesised `config.json` to read; the checkpoint states its own geometry
in its key-values and this module reads them, then hands the declaration a config in the
spelling training uses. Everything about the object list -- which objects a block kind has,
their shapes, which are quantised -- stays the declaration's answer rather than a second copy
of it here.

Four block kinds, because the two axes are independent: a layer runs either Kimi Delta
Attention or multi-head latent attention, and its feed-forward is either dense or the mixture.
The released 45-layer checkpoint uses three of them -- three leading dense KDA layers, then KDA
and MLA layers over the mixture.

The NextN draft head (`blk.45.nextn.*`, one MLA layer over the mixture plus three norms and a
fold) is carried under `mtp/` when the file has it: it is the checkpoint's own speculative
draft head, and `--spec mtp` runs it. One thing the released file carries that this artifact
does not:

- **the sparse indexer** (`blk.N.indexer.*`). It selects 512 pools of four tokens each, so for
  any context at or below `index_topk` every visible token is selected and full attention is
  exactly what the indexer would have asked for. Past that bound they differ, which is why the
  artifact records the bound and the engine refuses beyond it rather than quietly attending to
  more than the model was trained to.
"""

from __future__ import annotations

from dataclasses import dataclass
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
FRONTEND_REPO = "zai-org/GLM-5.3-Flash"


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

    blocks = int(need("block_count"))
    nextn = int(kv(f"{GGUF_ARCHITECTURE}.nextn_predict_layers", 0) or 0)
    layers = blocks - nextn
    kv_heads = list(need("attention.head_count_kv"))
    if len(kv_heads) != blocks:
        raise ValueError(
            f"`attention.head_count_kv` has {len(kv_heads)} entries for {blocks} blocks; it is "
            f"this file's attention schedule and has to name every one"
        )
    leading_dense = int(need("leading_dense_block_count"))
    if int(need("rope.dimension_count")) != 0:
        raise ValueError(
            "this checkpoint rotates its latent attention; every released GLM-5.3 is the NoPE "
            "variant and the served attention applies no rotary at all"
        )
    return Geometry(
        hidden=int(need("embedding_length")),
        layers=layers,
        query_heads=int(need("attention.head_count")),
        dense_intermediate=int(need("feed_forward_length")),
        expert_intermediate=int(need("expert_feed_forward_length")),
        shared_intermediate=int(kv(f"{GGUF_ARCHITECTURE}.expert_shared_feed_forward_length", 0) or 0)
        * int(kv(f"{GGUF_ARCHITECTURE}.expert_shared_count", 0) or 0),
        vocab=int(need("vocab_size")),
        experts=int(need("expert_count")),
        experts_per_token=int(need("expert_used_count")),
        routed_scale=float(kv(f"{GGUF_ARCHITECTURE}.expert_weights_scale", 1.0) or 1.0),
        swiglu_limit=float(list(need("swiglu_clamp_exp"))[0]),
        hc_streams=int(need("hyper_connection.count")),
        hc_sinkhorn_iterations=int(need("hyper_connection.sinkhorn_iterations")),
        hc_epsilon=float(need("hyper_connection.epsilon")),
        kda_heads=int(need("attention.head_count")),
        kda_head_dim=int(need("kda.head_dim")),
        kda_conv_kernel=int(need("ssm.conv_kernel")),
        kda_gate_rank=int(need("kda.head_dim")),
        kda_lower_bound=float(need("kda.gate_lower_bound")),
        q_lora_rank=int(need("attention.q_lora_rank")),
        kv_lora_rank=int(need("attention.kv_lora_rank")),
        qk_head_dim=int(need("attention.key_length_mla")),
        v_head_dim=int(need("attention.value_length_mla")),
        rms_epsilon=float(need("attention.layer_norm_rms_epsilon")),
        index_topk=int(need("attention.indexer.top_k")),
        attention_layers=tuple(i for i in range(layers) if int(kv_heads[i]) > 0),
        dense_layers=tuple(range(min(leading_dense, layers))),
        nextn_layers=nextn,
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
    declared = declaration.declare(ARCHITECTURE, config_from_geometry(geometry))
    return list(declared.objects(capabilities=set(CAPABILITIES)))


def tensor_specs(objects: Sequence[declaration.DeclaredObject]) -> tuple[TensorSpec, ...]:
    """Declared objects as artifact specs.

    `quantised` is a storage class, not a width, and this target's choice is group-wise int8 --
    for the objects that do not stay in the file's own K-quant format, which is most of the
    bytes. The norms, the router bias, the convolution taps, the per-head decay and the three
    hyper-connection scalars stay in the precision the declaration names: a router that picks
    eight of 288 experts and a decay that is exponentiated are both places where a coarse width
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
    "FRONTEND_REPO",
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
