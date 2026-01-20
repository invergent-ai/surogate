"""
HuggingFace Mapping Utilities for Python DSL

Provides helper functions for defining HuggingFace weight mappings,
including fuse, split, transform operations.

Example:
    @model
    @hf_mapping.indexed("blocks", layer="layer",
        qkv_weight=fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0
        ),
        mlp_down_weight=split(
            "model.layers.{layer}.mlp.down_proj.weight",
            ranges=[(0, 2048), (2048, 4096)],
            dim=0
        ),
    )
    class MyModel:
        ...
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class FuseMapping:
    """Specification to fuse multiple HF tensors into one.

    Example:
        fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0
        )
    """
    sources: tuple[str, ...]
    dim: int = 0

    def __repr__(self) -> str:
        sources_str = ", ".join(f'"{s}"' for s in self.sources)
        return f"fuse({sources_str}, dim={self.dim})"


@dataclass(frozen=True)
class SplitMapping:
    """Specification to split an HF tensor into parts.

    Example:
        split(
            "model.layers.{layer}.mlp.gate_up_proj.weight",
            ranges=[(0, 2048), (2048, 4096)],
            dim=0
        )
    """
    source: str
    ranges: tuple[tuple[int, int], ...]
    dim: int = 0

    def __repr__(self) -> str:
        ranges_str = ", ".join(f"[{r[0]}, {r[1]}]" for r in self.ranges)
        return f'split("{self.source}", ranges=[{ranges_str}], dim={self.dim})'


@dataclass(frozen=True)
class TransformMapping:
    """Specification to transform an HF tensor.

    Example:
        transform("model.embed_tokens.weight", fn="transpose")
    """
    source: str
    fn: str

    def __repr__(self) -> str:
        return f'transform("{self.source}", fn="{self.fn}")'


@dataclass(frozen=True)
class TiedToMapping:
    """Specification to tie a weight to another parameter.

    Example:
        tied_to("embedding")
    """
    target: str

    def __repr__(self) -> str:
        return f'tied_to("{self.target}")'


def fuse(*sources: str, dim: int = 0) -> FuseMapping:
    """Create a fuse mapping to combine multiple HF tensors.

    Concatenates the specified HF checkpoint tensors along the given dimension.

    Args:
        *sources: HF checkpoint paths to fuse
        dim: Dimension to concatenate along (default: 0)

    Example:
        # Fuse separate Q, K, V projections into combined QKV
        qkv_weight=fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0
        )
    """
    return FuseMapping(sources=tuple(sources), dim=dim)


def split(
    source: str,
    *,
    ranges: list[tuple[int, int]] | None = None,
    parts: int | None = None,
    dim: int = 0,
) -> SplitMapping:
    """Create a split mapping to extract part of an HF tensor.

    Either specify explicit ranges or number of equal parts.

    Args:
        source: HF checkpoint path
        ranges: List of (start, end) ranges to extract
        parts: Number of equal parts to split into
        dim: Dimension to split along (default: 0)

    Example:
        # Split fused gate_up into separate tensors
        gate_weight=split(
            "model.layers.{layer}.mlp.gate_up_proj.weight",
            ranges=[(0, 2048)],
            dim=0
        )
    """
    if ranges is None and parts is None:
        raise ValueError("Must specify either 'ranges' or 'parts'")

    if ranges is not None:
        return SplitMapping(source=source, ranges=tuple(ranges), dim=dim)

    # parts specified - ranges will be computed at load time
    # Store as special marker
    return SplitMapping(source=source, ranges=((-1, parts),), dim=dim)


def transform(source: str, *, fn: str) -> TransformMapping:
    """Create a transform mapping to modify an HF tensor.

    Args:
        source: HF checkpoint path
        fn: Transform function name ("transpose", "permute_qkv", etc.)

    Example:
        # Transpose embedding for tied lm_head
        lm_head=transform("model.embed_tokens.weight", fn="transpose")
    """
    return TransformMapping(source=source, fn=fn)


def tied_to(target: str) -> TiedToMapping:
    """Create a tied mapping to share weights with another parameter.

    Args:
        target: Internal parameter name to tie to

    Example:
        # Tie lm_head to embedding
        lm_head=tied_to("embedding")
    """
    return TiedToMapping(target=target)


# Type alias for any HF mapping spec
HFMappingValue = str | FuseMapping | SplitMapping | TransformMapping | TiedToMapping


def is_hf_mapping_spec(value: Any) -> bool:
    """Check if a value is an HF mapping specification."""
    return isinstance(value, (str, FuseMapping, SplitMapping, TransformMapping, TiedToMapping))


def mapping_to_dict(mapping: HFMappingValue) -> dict[str, Any]:
    """Convert an HF mapping spec to a dictionary representation."""
    if isinstance(mapping, str):
        return {"kind": "direct", "path": mapping}
    elif isinstance(mapping, FuseMapping):
        return {"kind": "fuse", "sources": list(mapping.sources), "dim": mapping.dim}
    elif isinstance(mapping, SplitMapping):
        return {"kind": "split", "source": mapping.source, "ranges": list(mapping.ranges), "dim": mapping.dim}
    elif isinstance(mapping, TransformMapping):
        return {"kind": "transform", "source": mapping.source, "fn": mapping.fn}
    elif isinstance(mapping, TiedToMapping):
        return {"kind": "tied_to", "target": mapping.target}
    else:
        raise TypeError(f"Unknown mapping type: {type(mapping)}")
