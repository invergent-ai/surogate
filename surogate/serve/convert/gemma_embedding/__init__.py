"""EmbeddingGemma artifact conversion target.

The inventory is derived from the DSL declaration rather than restated here;
this package supplies only the source mapping and the conversion driver.
"""

from .sources import LAYER_SOURCES, MODEL_SOURCES, Source, gguf_names, source_for

MODEL_ID = "embeddinggemma-300m"
TARGET_KEY = "gemma_embedding"
#: Every quantised object comes from Q8_0, which is W8G32_F16S bit for bit.
WEIGHTS_ID = "w8"
#: What the target's C++ binder must consume for the artifact to load.
CAPABILITIES = frozenset({"text", "embedding"})

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
