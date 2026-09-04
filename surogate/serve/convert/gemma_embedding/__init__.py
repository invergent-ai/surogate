"""EmbeddingGemma artifact conversion target.

An encoder: one forward, no KV cache, no sampler. So the contract has no output head,
no draft head and no MTP -- there is nothing to predict a next token with -- and the
declaration ends at a pooled embedding instead. What the artifact does hold is in
:mod:`inventory`, where each object comes from is in :mod:`recipe`, and
:mod:`convert` is the driver that puts the two together.
"""

from .inventory import CAPABILITIES, MODEL_ID, TARGET_KEY, WEIGHTS_ID
from .recipe import LAYER_SOURCES, MODEL_SOURCES, Source, gguf_names, source_for

__all__ = [
    "CAPABILITIES",
    "LAYER_SOURCES",
    "MODEL_ID",
    "MODEL_SOURCES",
    "Source",
    "TARGET_KEY",
    "WEIGHTS_ID",
    "gguf_names",
    "source_for",
]
