"""Persistent-object contract for the EmbeddingGemma target.

The object list is not written down here: it is derived from the DSL declaration by
``generate/emit_inventory.py``. So the contract this module owns is the *other* three
things a caller needs to know which objects the artifact holds and what shape they are
-- the dimensions the declaration compiles against, what the GGUF says they are, and
what format and layout the artifact stores the result in. Where each object's numbers
come from lives in :mod:`recipe`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from surogate.serve.artifact.container import TensorSpec
from surogate.serve.convert.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    ROW_SPLIT_LAYOUT,
    W8,
)

if TYPE_CHECKING:  # a converter's GGUF reader, needed only for the annotation
    from surogate.serve.convert.qwen4exp.convert import GgufSource


MODEL_ID = "embeddinggemma-300m"
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
    """The dimensions the artifact's object list depends on.

    Read off the checkpoint at convert time by `recipe.geometry_from_config`, so a second
    published size of this encoder needs no edit here. Every field is an input to the
    declaration and nothing else is: the object list is a function of exactly these seven
    numbers and the capability set, which is why `declared_objects` can be asked for the
    registered size with no checkpoint in hand.
    """

    layers: int
    hidden: int
    intermediate: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int


#: `csrc/src/serve/encoder/gemma_embedding.h::GemmaEmbeddingConfig`, which this must agree
#: with: that binder is compiled for one geometry and reads nothing from a config at serve
#: time. It carries no kv_heads because EmbeddingGemma is multi-query.
EMBEDDINGGEMMA_300M = Geometry(
    layers=24,
    hidden=768,
    intermediate=1152,
    vocab=262144,
    query_heads=3,
    kv_heads=1,
    head_dim=256,
)

GEOMETRY = EMBEDDINGGEMMA_300M


# --------------------------------------------------------------------------------------------
# The config the declaration is compiled against
# --------------------------------------------------------------------------------------------

#: What the GGUF does not carry, and what llama.cpp supplies from the
#: architecture identity instead (``models/gemma-embedding.cpp``). Every one is
#: silent when wrong, which is the case for keeping them in a declaration rather
#: than reading them from whatever file happens to be at hand.
DECLARED_NOT_IN_GGUF = {
    # swa_period = 6 there, as a default for an optional key this file omits.
    "_sliding_window_pattern": 6,
    # hparams.causal_attn = false, hardcoded for the architecture.
    "use_bidirectional_attention": True,
    # llama.cpp uses 1/sqrt(n_embd_head_k), which coincides with the real scalar
    # here because head_dim is also 256. It does not for Gemma3-27B.
    "query_pre_attn_scalar": 256,
}


def declaration_config(geometry: Geometry) -> SourceConfig:
    """The half of the config the object list is a function of.

    Split out because it is the half a caller can produce without a checkpoint: the
    registered geometry goes in, and the declaration answers with the same object list
    it would give for a real 300M file.
    """

    return {
        "architectures": [DECLARATION],
        "model_type": _MODEL_TYPE,
        "vocab_size": geometry.vocab,
        "hidden_size": geometry.hidden,
        "num_hidden_layers": geometry.layers,
        "num_attention_heads": geometry.query_heads,
        "num_key_value_heads": geometry.kv_heads,
        "intermediate_size": geometry.intermediate,
        "head_dim": geometry.head_dim,
        **DECLARED_NOT_IN_GGUF,
    }


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
    geometry = Geometry(
        layers=int(kv("block_count")),
        hidden=int(hidden),
        intermediate=int(kv("feed_forward_length")),
        vocab=int(vocab),
        query_heads=int(kv("attention.head_count")),
        kv_heads=int(kv("attention.head_count_kv")),
        head_dim=int(kv("attention.key_length")),
    )
    config = {
        **declaration_config(geometry),
        # Runtime, not shape. No object list depends on these; the binder is compiled
        # with its own copy of them, and the artifact carries no config for it to read.
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


def declared_objects(geometry: Geometry = GEOMETRY) -> list[dict[str, Any]]:
    """Every object the artifact stores, from the declaration."""

    generate = Path(__file__).resolve().parents[2] / "tools" / "generate"
    if str(generate) not in sys.path:
        sys.path.insert(0, str(generate))
    import emit_inventory  # noqa: PLC0415

    return emit_inventory.inventory_for(
        DECLARATION, declaration_config(geometry), capabilities=set(CAPABILITIES)
    )


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


__all__ = [
    "BF16",
    "CAPABILITIES",
    "CONTIGUOUS_LAYOUT",
    "DECLARATION",
    "DECLARED_NOT_IN_GGUF",
    "EMBEDDINGGEMMA_300M",
    "FRONTEND_RESOURCES",
    "GEOMETRY",
    "Geometry",
    "MODEL_ID",
    "ROW_SPLIT_LAYOUT",
    "SourceConfig",
    "TARGET_KEY",
    "TensorSpec",
    "W8",
    "WEIGHTS_ID",
    "config_from_gguf",
    "declaration_config",
    "declared_objects",
    "tensor_specs",
]
