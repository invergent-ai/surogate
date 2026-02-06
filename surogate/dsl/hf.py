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


@dataclass(frozen=True)
class StackExpertsMapping:
    """Specification to stack per-expert HF tensors into a batched format.

    Used for MoE models where HuggingFace stores expert weights individually:
        model.layers.0.mlp.experts.0.down_proj.weight
        model.layers.0.mlp.experts.1.down_proj.weight
        ...

    This mapping stacks them into a single tensor of shape [num_experts, ...].

    Example:
        stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
            num_experts=64
        )

        # Or auto-detect num_experts from config:
        stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        )
    """
    pattern: str  # Pattern with {expert} placeholder
    num_experts: int = 0  # 0 = auto-detect from model config
    fuse_gate_up: bool = False  # If True, fuse gate_proj and up_proj into gate_up

    def __repr__(self) -> str:
        if self.num_experts > 0:
            return f'stack_experts("{self.pattern}", num_experts={self.num_experts})'
        return f'stack_experts("{self.pattern}")'


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


def stack_experts(
    pattern: str,
    *,
    num_experts: int = 0,
    fuse_gate_up: bool = False,
) -> StackExpertsMapping:
    """Create a mapping to stack per-expert HF tensors into batched format.

    For MoE models where HuggingFace stores expert weights individually,
    this loads and stacks them into a single tensor of shape [num_experts, ...].

    Args:
        pattern: HF checkpoint path pattern with {expert} placeholder
        num_experts: Number of experts to stack (0 = auto-detect from model config)
        fuse_gate_up: If True, pattern should be for gate_proj and this will
                      also load up_proj and fuse them into gate_up format

    Example:
        # Stack individual expert down projections into batched tensor
        experts_down=stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
            num_experts=64
        )

        # Auto-detect num_experts from config
        experts_down=stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"
        )

        # Stack and fuse gate+up projections into batched gate_up tensor
        experts_gate_up=stack_experts(
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            fuse_gate_up=True
        )
    """
    return StackExpertsMapping(pattern=pattern, num_experts=num_experts, fuse_gate_up=fuse_gate_up)


# Type alias for any HF mapping spec
HFMappingValue = str | FuseMapping | SplitMapping | TransformMapping | TiedToMapping | StackExpertsMapping


def is_hf_mapping_spec(value: Any) -> bool:
    """Check if a value is an HF mapping specification."""
    return isinstance(value, (str, FuseMapping, SplitMapping, TransformMapping, TiedToMapping, StackExpertsMapping))


# =============================================================================
# Module mapping composition utilities
# =============================================================================


def _expand_hf_prefix(mapping: Any, hf_prefix: str) -> Any:
    """Replace {prefix} placeholder in an HF mapping with the actual prefix."""
    if isinstance(mapping, str):
        return mapping.replace("{prefix}", hf_prefix)
    elif isinstance(mapping, FuseMapping):
        return FuseMapping(
            sources=tuple(s.replace("{prefix}", hf_prefix) for s in mapping.sources),
            dim=mapping.dim,
        )
    elif isinstance(mapping, SplitMapping):
        return SplitMapping(
            source=mapping.source.replace("{prefix}", hf_prefix),
            ranges=mapping.ranges,
            dim=mapping.dim,
        )
    elif isinstance(mapping, TransformMapping):
        return TransformMapping(
            source=mapping.source.replace("{prefix}", hf_prefix),
            fn=mapping.fn,
        )
    elif isinstance(mapping, StackExpertsMapping):
        return StackExpertsMapping(
            pattern=mapping.pattern.replace("{prefix}", hf_prefix),
            num_experts=mapping.num_experts,
            fuse_gate_up=mapping.fuse_gate_up,
        )
    else:
        return mapping


def expand_module_mapping(
    defaults: dict[str, Any],
    *,
    hf_prefix: str,
    param_prefix: str = "",
) -> dict[str, Any]:
    """Expand module-level HF mapping defaults with concrete prefixes.

    Takes a module's ``_hf_mapping_defaults_`` dict (which uses ``{prefix}``
    placeholders) and returns a new dict with ``{prefix}`` replaced by
    *hf_prefix*, and param names optionally prefixed with *param_prefix*.

    Args:
        defaults: Module's ``_hf_mapping_defaults_`` dict.
        hf_prefix: HF path prefix to replace ``{prefix}`` with.
        param_prefix: Optional prefix for param names (e.g., ``"mlp_"``).

    Returns:
        Dict suitable for inclusion in ``_hf_block_mappings_``.

    Example::

        >>> expand_module_mapping(
        ...     SwiGLUMLP._hf_mapping_defaults_,
        ...     hf_prefix="model.layers.{layer}.mlp",
        ...     param_prefix="mlp_",
        ... )
        {
            "mlp_up_weight": fuse("model.layers.{layer}.mlp.up_proj.weight", ...),
            "mlp_down_weight": "model.layers.{layer}.mlp.down_proj.weight",
        }
    """
    result: dict[str, Any] = {}
    for param_name, mapping in defaults.items():
        full_param_name = param_prefix + param_name
        result[full_param_name] = _expand_hf_prefix(mapping, hf_prefix)
    return result


def build_norm_mappings(
    layer_prefix: str = "model.layers.{layer}",
    *,
    ln1_suffix: str = "input_layernorm",
    ln2_suffix: str = "post_attention_layernorm",
) -> dict[str, Any]:
    """Build HF mappings for pre-attention and pre-MLP layer norms.

    Args:
        layer_prefix: HF prefix for the layer (e.g., ``"model.layers.{layer}"``).
        ln1_suffix: Suffix for the pre-attention norm.
        ln2_suffix: Suffix for the pre-MLP norm.
    """
    from .modules.rmsnorm import RMSNorm

    mappings: dict[str, Any] = {}
    mappings.update(expand_module_mapping(
        RMSNorm._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{ln1_suffix}",
        param_prefix="ln1_",
    ))
    mappings.update(expand_module_mapping(
        RMSNorm._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{ln2_suffix}",
        param_prefix="ln2_",
    ))
    return mappings


def build_attn_mappings(
    layer_prefix: str = "model.layers.{layer}",
    *,
    attn_module: type | None = None,
    attn_suffix: str = "self_attn",
) -> dict[str, Any]:
    """Build HF mappings for attention params from module defaults.

    Args:
        layer_prefix: HF prefix for the layer.
        attn_module: Attention module class (default: ``GQAAttention``).
                     Must have ``_hf_mapping_defaults_``.
        attn_suffix: Suffix for the attention submodule.
    """
    if attn_module is None:
        from .modules.attention import GQAAttention
        attn_module = GQAAttention

    return expand_module_mapping(
        attn_module._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{attn_suffix}",
    )


def build_mlp_mappings(
    layer_prefix: str = "model.layers.{layer}",
    *,
    mlp_module: type | None = None,
    mlp_suffix: str = "mlp",
    param_prefix: str = "mlp_",
) -> dict[str, Any]:
    """Build HF mappings for MLP params from module defaults.

    Args:
        layer_prefix: HF prefix for the layer.
        mlp_module: MLP module class (default: ``SwiGLUMLP``).
                    Must have ``_hf_mapping_defaults_``.
        mlp_suffix: Suffix for the MLP submodule.
        param_prefix: Prefix for param names (default: ``"mlp_"``).
    """
    if mlp_module is None:
        from .modules.mlp import SwiGLUMLP
        mlp_module = SwiGLUMLP

    return expand_module_mapping(
        mlp_module._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{mlp_suffix}",
        param_prefix=param_prefix,
    )


def build_mamba_mappings(
    layer_prefix: str = "backbone.layers.{layer}",
    *,
    mamba_module: type | None = None,
    mamba_suffix: str = "mixer",
) -> dict[str, Any]:
    """Build HF mappings for Mamba2 params from module defaults.

    Args:
        layer_prefix: HF prefix for the layer.
        mamba_module: Mamba module class (default: ``Mamba2Mixer``).
                      Must have ``_hf_mapping_defaults_``.
        mamba_suffix: Suffix for the Mamba submodule.
    """
    if mamba_module is None:
        from .modules.mamba import Mamba2Mixer
        mamba_module = Mamba2Mixer

    return expand_module_mapping(
        mamba_module._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{mamba_suffix}",
    )


def build_simple_mlp_mappings(
    layer_prefix: str = "backbone.layers.{layer}",
    *,
    mlp_module: type | None = None,
    mlp_suffix: str = "mixer",
    param_prefix: str = "",
) -> dict[str, Any]:
    """Build HF mappings for SimpleMLP params from module defaults.

    Args:
        layer_prefix: HF prefix for the layer.
        mlp_module: MLP module class (default: ``SimpleMLP``).
                    Must have ``_hf_mapping_defaults_``.
        mlp_suffix: Suffix for the MLP submodule.
        param_prefix: Prefix for param names.
    """
    if mlp_module is None:
        from .modules.mamba import SimpleMLP
        mlp_module = SimpleMLP

    return expand_module_mapping(
        mlp_module._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{mlp_suffix}",
        param_prefix=param_prefix,
    )


def build_moe_mappings(
    layer_prefix: str = "model.layers.{layer}",
    *,
    moe_module: type | None = None,
    moe_suffix: str = "mlp",
    include_shared: bool = False,
    shared_module: type | None = None,
) -> dict[str, Any]:
    """Build HF mappings for MoE params from module defaults.

    Composes router + expert mappings from ``MoEExpertsGated`` (default)
    or another MoE module. Optionally includes shared expert mappings.

    Args:
        layer_prefix: HF prefix for the layer.
        moe_module: MoE module class (default: ``MoEExpertsGated``).
                    Must have ``_hf_mapping_defaults_``.
        moe_suffix: Suffix for the MoE submodule.
        include_shared: If True, also include shared expert mappings.
        shared_module: Shared expert module class (default: ``MoESharedExpert``).
    """
    if moe_module is None:
        from .modules.moe import MoEExpertsGated
        moe_module = MoEExpertsGated

    mappings = expand_module_mapping(
        moe_module._hf_mapping_defaults_,
        hf_prefix=f"{layer_prefix}.{moe_suffix}",
    )

    if include_shared:
        if shared_module is None:
            from .modules.moe import MoESharedExpert
            shared_module = MoESharedExpert
        mappings.update(expand_module_mapping(
            shared_module._hf_mapping_defaults_,
            hf_prefix=f"{layer_prefix}.{moe_suffix}",
        ))

    return mappings


def build_dense_block_mappings(
    layer_prefix: str = "model.layers.{layer}",
    *,
    attn_module: type | None = None,
    mlp_module: type | None = None,
) -> dict[str, Any]:
    """Build complete ``_hf_block_mappings_`` for a dense transformer block.

    Composes norm, attention, and MLP mappings from module-level defaults.

    Args:
        layer_prefix: HF prefix for the layer (e.g., ``"model.layers.{layer}"``).
        attn_module: Attention module class (default: ``GQAAttention``).
        mlp_module: MLP module class (default: ``SwiGLUMLP``).

    Returns:
        Complete ``_hf_block_mappings_`` dict ready for use in a model class.

    Example::

        @model
        class LlamaModel:
            _hf_block_mappings_ = build_dense_block_mappings()

        @model
        class Qwen3Model:
            _hf_block_mappings_ = build_dense_block_mappings(
                attn_module=Qwen3Attention,
            )
    """
    return {
        **build_norm_mappings(layer_prefix),
        **build_attn_mappings(layer_prefix, attn_module=attn_module),
        **build_mlp_mappings(layer_prefix, mlp_module=mlp_module),
    }


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
    elif isinstance(mapping, StackExpertsMapping):
        return {
            "kind": "stack_experts",
            "pattern": mapping.pattern,
            "num_experts": mapping.num_experts,
            "fuse_gate_up": mapping.fuse_gate_up,
        }
    else:
        raise TypeError(f"Unknown mapping type: {type(mapping)}")
