"""Persistent-object contract for the EmbeddingGemma target.

The object list is not written down here: it is derived from the DSL declaration by
``generate/emit_inventory.py``. So the contract this module owns is the *other* three
things a caller needs to know which objects the artifact holds and what shape they are
-- the dimensions the declaration compiles against, what the GGUF says they are, and
what format and layout the artifact stores the result in. Where each object's numbers
come from lives in :mod:`recipe`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from surogate.serve.artifact.container import TensorSpec
from surogate.serve.convert.common import declaration
from surogate.serve.convert.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    W8,
)

if TYPE_CHECKING:  # a converter's GGUF reader, needed only for the annotation
    from surogate.serve.convert.common.gguf_source import GgufSource


MODEL_ID = "gemma_embedding"
TARGET_KEY = "gemma_embedding"
#: Every quantised object comes from Q8_0, which is W8G32_F16S bit for bit.
WEIGHTS_ID = "w8"
#: What the target's C++ binder must consume for the artifact to load.
CAPABILITIES = frozenset({"text", "embedding"})

#: An embedding request carries no chat template and generates no tokens, so the
#: frontend is a tokenizer and nothing else.
#:
#: The SentencePiece model rather than `tokenizer.json`: upstream SentencePiece
#: reproduces the HF ids exactly on it, and it is 4.7 MB where the JSON is 33 MB.
#: `tokenizer_config.json` still travels for the special-token ids and the
#: sequence limit.
FRONTEND_RESOURCES = ("frontend/tokenizer.model", "frontend/tokenizer_config.json")

#: The checkpoint's own account of itself, HF-spelled. Wider than `Geometry` -- it also
#: carries rope bases, the window and the epsilon, which shape the runtime and not the
#: object list -- and spelled differently on purpose: the same name for a different shape
#: is the trap.
SourceConfig = dict[str, Any]

#: The declaration an EmbeddingGemma checkpoint is compiled against. It is Gemma 3's:
#: this is a Gemma 3 decoder with the causal head replaced by a pooled projection.
DECLARATION = "Gemma3TextModel"
_MODEL_TYPE = "gemma3_text"


@dataclass(frozen=True, slots=True)
class Geometry:
    """The dimensions and declaration resolved for this checkpoint."""

    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    declared: declaration.Declaration = field(repr=False, compare=False)

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple("sliding_attention" if kind == "sliding" else "full_attention"
                     for kind in self.declared.block_types)


def config_from_gguf(source: "GgufSource") -> SourceConfig:
    """Everything the GGUF says about the checkpoint, HF-spelled.

    Read from the GGUF where the GGUF has it, declared where it does not.
    """

    def kv(key: str) -> Any:
        return source.fields[f"gemma-embedding.{key}"].contents()

    architecture = source.fields["general.architecture"].contents()
    if architecture != "gemma-embedding":
        raise ValueError(f"expected a gemma-embedding GGUF, got {architecture!r}")

    vocab, hidden = source.tensor("token_embd.weight").shape
    head_dim = int(kv("attention.key_length"))
    period_field = source.fields.get("gemma-embedding.attention.sliding_window_pattern")
    # GGUF defines six as this architecture's default when the optional period is absent.
    # Its attention scale is defined from the declared head width (llama.cpp's model reader).
    config = {
        "architectures": [DECLARATION], "model_type": _MODEL_TYPE,
        "num_hidden_layers": int(kv("block_count")), "hidden_size": int(hidden),
        "intermediate_size": int(kv("feed_forward_length")), "vocab_size": int(vocab),
        "num_attention_heads": int(kv("attention.head_count")),
        "num_key_value_heads": int(kv("attention.head_count_kv")), "head_dim": head_dim,
        "_sliding_window_pattern": int(period_field.contents()) if period_field is not None else 6,
        "use_bidirectional_attention": True, "query_pre_attn_scalar": head_dim,
        "max_position_embeddings": int(kv("context_length")),
        "rms_norm_eps": float(kv("attention.layer_norm_rms_epsilon")),
        "sliding_window": int(kv("attention.sliding_window")),
        "rope_theta": float(kv("rope.freq_base")),
        "rope_local_base_freq": float(kv("rope.freq_base_swa")),
    }
    if int(kv("pooling_type")) != 1:  # LLAMA_POOLING_TYPE_MEAN
        raise ValueError(f"expected mean pooling, got pooling_type {kv('pooling_type')}")
    return config


# --------------------------------------------------------------------------------------------
# The objects
# --------------------------------------------------------------------------------------------


def declared_objects(geometry: Geometry) -> list[dict[str, Any]]:
    """Every object the artifact stores, from the declaration."""

    return [obj.as_dict() for obj in geometry.declared.objects(capabilities=set(CAPABILITIES))]


def tensor_specs(objects: Sequence[dict[str, Any]]) -> list[TensorSpec]:
    """Declared objects as artifact specs.

    The declaration says ``quantised`` and leaves the width to the target; every
    quantised object here comes from Q8_0, so W8 is the whole mapping.
    """
    out = []
    for obj in objects:
        fmt = W8 if obj["format"] == "quantised" else BF16
        layout = ROW_SPLIT_LAYOUT if fmt == W8 else CONTIGUOUS_LAYOUT
        out.append(TensorSpec(name=obj["name"], shape=obj["shape"], format=fmt, layout=layout))
    return out
