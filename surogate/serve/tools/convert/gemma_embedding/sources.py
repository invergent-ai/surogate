"""Where each artifact object comes from in the ``gemma-embedding`` GGUF.

The inventory is not restated here -- it is derived from the DSL declaration by
``generate/emit_inventory.py``. This module answers only the other half: given an
object name, which GGUF tensors build it and what has to happen to them.

Three kinds of work, and only the first is free:

* **Repack.** ``Q8_0`` and the artifact's ``W8G32_F16S`` are the same format
  (int8 codes, one binary16 scale per 32-group), so an object built by row
  algebra over Q8_0 sources moves across bit-exactly -- no dequantize, no
  requantize, no GPU. Everything under ``attention/`` and ``mlp/`` qualifies,
  including the Q/K/V fuse, because concatenating on the row axis leaves each
  row's groups intact.

* **Unfold.** Gemma norms are zero-centred: HF stores ``w``, the GGUF stores the
  folded ``1 + w``, and the runtime re-applies the offset itself
  (``rmsnorm(..., unit_offset=true)``). So every norm loses a one on the way in.
  Verified against the HF checkpoint: ``cos(gguf - 1, hf) == 1.000000``.

* **Compose.** The two sentence-transformers Dense modules both declare an
  Identity activation, so ``768 -> 3072 -> 768`` is a single linear map and the
  artifact stores the product. This is the one object that must dequantize: a
  matrix product mixes ``k``, so it is not row algebra.

The GGUF stores ``ne = [k, n]`` with ``k`` fastest, which gguf-py hands back as
``[n, k]`` -- the logical orientation, and the artifact's. No transpose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    """How one artifact object is built from GGUF tensors.

    ``op`` is the identity the declaration named (``ServeObject.transform``) or
    the implicit one for a plain copy. ``tensors`` are GGUF names, in row order
    for a concatenation and in application order for a composition.
    """

    tensors: tuple[str, ...]
    op: str = "copy"

    @property
    def repackable(self) -> bool:
        """Whether this object can move from Q8_0 without dequantizing.

        Row algebra only: a copy or a row-axis concatenation. ``unfold`` touches
        values and ``compose_linear`` mixes columns, so neither qualifies.
        """
        return self.op in ("copy", "concat_rows")


#: Per-layer objects. Keys are the object names the declaration emits, minus the
#: ``text/layers/{layer}/`` prefix; values name GGUF tensors minus ``blk.{i}.``.
LAYER_SOURCES: dict[str, Source] = {
    # The four sandwich norms. GGUF names two of them differently from HF:
    # `ffn_norm` is the *pre*-feedforward norm, and `post_ffw_norm` the post one.
    "input_norm": Source(("attn_norm.weight",), "unfold"),
    "post_attention_norm": Source(("post_attention_norm.weight",), "unfold"),
    "pre_feedforward_norm": Source(("ffn_norm.weight",), "unfold"),
    "post_feedforward_norm": Source(("post_ffw_norm.weight",), "unfold"),
    # Q, K and V stack on the row axis: 3*256 + 256 + 256 = 1280 rows of k=768.
    # Gemma 3 has no attention output gate, so nothing is interleaved -- unlike
    # the Qwen families, whose fuse splits an interleaved query/gate projection.
    "attention/query_key_value": Source(
        ("attn_q.weight", "attn_k.weight", "attn_v.weight"), "concat_rows"
    ),
    "attention/query_norm": Source(("attn_q_norm.weight",), "unfold"),
    "attention/key_norm": Source(("attn_k_norm.weight",), "unfold"),
    "attention/output": Source(("attn_output.weight",)),
    "mlp/gate": Source(("ffn_gate.weight",)),
    "mlp/up": Source(("ffn_up.weight",)),
    "mlp/down": Source(("ffn_down.weight",)),
}

#: Objects outside the stack.
MODEL_SOURCES: dict[str, Source] = {
    "text/token_embedding": Source(("token_embd.weight",)),
    "text/final_norm": Source(("output_norm.weight",), "unfold"),
    # dense_3 @ dense_2: [768,3072] @ [3072,768] -> [768,768], applied in that
    # order because the checkpoint runs 2_Dense first.
    "text/embedding_head": Source(("dense_3.weight", "dense_2.weight"), "compose_linear"),
}


def source_for(object_name: str) -> Source:
    """The GGUF source for one artifact object name."""

    if object_name.startswith("text/layers/"):
        rest = object_name.split("/", 3)[3]
        return LAYER_SOURCES[rest]
    return MODEL_SOURCES[object_name]


def gguf_names(object_name: str) -> tuple[str, ...]:
    """Fully-qualified GGUF tensor names backing one artifact object."""

    source = source_for(object_name)
    if not object_name.startswith("text/layers/"):
        return source.tensors
    layer = object_name.split("/")[2]
    return tuple(f"blk.{layer}.{name}" for name in source.tensors)
