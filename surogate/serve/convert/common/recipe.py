"""Leaf source-recipe primitives shared by Qwen3.6 artifact targets.

The expression language and mechanical materialization are common.  Target
packages remain responsible for their complete recipe sequence and checkpoint
geometry; only checkpoint-invariant Vision recipes are built here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from pathlib import Path
from typing import Mapping, Sequence

import torch

from surogate.serve.convert.common.safetensors import ShardReader

from .inventory import FP32, TensorSpec


SOURCE_DTYPE = "BF16"


@dataclass(frozen=True, slots=True)
class SourceTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = SOURCE_DTYPE


@dataclass(frozen=True, slots=True)
class Slice:
    source: Expression
    axis: int
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class Reshape:
    source: Expression
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Transpose:
    source: Expression
    axes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Concat:
    sources: tuple[Expression, ...]
    axis: int


@dataclass(frozen=True, slots=True)
class Cast:
    source: Expression
    dtype: str


@dataclass(frozen=True, slots=True)
class DraftHeadTokenIds:
    ranking_path: str
    tokenizer_resource: str
    vocab_rows: int
    tokenizer_id_count: int
    rows: int


@dataclass(frozen=True, slots=True)
class AnyOf:
    """The same object, expressed however the checkpoint at hand stores it.

    A routed MoE is the case that forced this: the HF checkpoint fuses gate and up into one
    `gate_up_proj`, and a GGUF of the same model keeps them as two stacked tensors. Both
    describe identical rows in identical order, so rather than fork the recipes per input
    format the object carries both spellings and the resolver takes whichever one the source
    actually provides. Options are tried in order; the first whose sources are all present wins.
    """

    options: tuple["Expression", ...]


@dataclass(frozen=True, slots=True)
class GatherRows:
    source: SourceTensor
    token_ids_object: str
    rows: int


Expression = (
    SourceTensor
    | Slice
    | Reshape
    | Transpose
    | Concat
    | Cast
    | DraftHeadTokenIds
    | GatherRows
    | AnyOf
)


@dataclass(frozen=True, slots=True)
class TensorRecipe:
    object_name: str
    expression: Expression


@dataclass(frozen=True, slots=True)
class SourcePreflight:
    recipe_count: int
    source_tensor_count: int
    source_shard_count: int
    source_dtype_counts: dict[str, int]


def source(name: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(name=name, shape=shape)


def attention_qproj_part(
    source_name: str,
    gate: bool,
    *,
    num_heads: int,
    hidden_size: int,
    head_dim: int,
) -> Expression:
    """Extract query or output-gate rows from head-interleaved Q projection."""

    source_tensor = source(source_name, (num_heads * 2 * head_dim, hidden_size))
    per_head = Reshape(source_tensor, (num_heads, 2 * head_dim, hidden_size))
    begin = head_dim if gate else 0
    return Reshape(
        Slice(per_head, 1, begin, begin + head_dim),
        (num_heads * head_dim, hidden_size),
    )


def expression_shape(expression: Expression) -> tuple[int, ...]:
    if isinstance(expression, SourceTensor):
        return expression.shape

    if isinstance(expression, Slice):
        shape = list(expression_shape(expression.source))
        if not 0 <= expression.axis < len(shape):
            raise ValueError(f"slice axis {expression.axis} is outside rank {len(shape)}")
        if not 0 <= expression.begin < expression.end <= shape[expression.axis]:
            raise ValueError(
                f"invalid slice [{expression.begin},{expression.end}) for axis size "
                f"{shape[expression.axis]}"
            )
        shape[expression.axis] = expression.end - expression.begin
        return tuple(shape)

    if isinstance(expression, Reshape):
        source_shape = expression_shape(expression.source)
        if prod(source_shape) != prod(expression.shape):
            raise ValueError(f"cannot reshape {source_shape} to {expression.shape}")
        return expression.shape

    if isinstance(expression, Transpose):
        source_shape = expression_shape(expression.source)
        if tuple(sorted(expression.axes)) != tuple(range(len(source_shape))):
            raise ValueError(f"invalid transpose axes {expression.axes} for {source_shape}")
        return tuple(source_shape[axis] for axis in expression.axes)

    if isinstance(expression, Concat):
        if not expression.sources:
            raise ValueError("concat requires at least one source")
        shapes = [expression_shape(part) for part in expression.sources]
        rank = len(shapes[0])
        if not 0 <= expression.axis < rank:
            raise ValueError(f"concat axis {expression.axis} is outside rank {rank}")
        output = list(shapes[0])
        output[expression.axis] = 0
        for shape in shapes:
            if len(shape) != rank:
                raise ValueError("concat sources have different ranks")
            for axis, (got, expected) in enumerate(zip(shape, shapes[0])):
                if axis != expression.axis and got != expected:
                    raise ValueError("concat sources have incompatible shapes")
            output[expression.axis] += shape[expression.axis]
        return tuple(output)

    if isinstance(expression, Cast):
        return expression_shape(expression.source)

    if isinstance(expression, DraftHeadTokenIds):
        return (expression.rows,)

    if isinstance(expression, GatherRows):
        return (expression.rows, *expression.source.shape[1:])
    if isinstance(expression, AnyOf):
        shapes = {expression_shape(option) for option in expression.options}
        if len(shapes) != 1:
            raise ValueError(f"AnyOf options disagree on shape: {sorted(shapes)}")
        return shapes.pop()

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def expression_sources(expression: Expression) -> tuple[SourceTensor, ...]:
    if isinstance(expression, SourceTensor):
        return (expression,)
    if isinstance(expression, (Slice, Reshape, Transpose, Cast)):
        return expression_sources(expression.source)
    if isinstance(expression, Concat):
        return tuple(
            item
            for part in expression.sources
            for item in expression_sources(part)
        )
    if isinstance(expression, GatherRows):
        return (expression.source,)
    if isinstance(expression, DraftHeadTokenIds):
        return ()
    if isinstance(expression, AnyOf):
        return tuple(item for option in expression.options for item in expression_sources(option))
    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def choose_option(expression: "AnyOf", available) -> "Expression":
    """The first option whose every source `available(name)` accepts."""
    for option in expression.options:
        if all(available(item.name) for item in expression_sources(option)):
            return option
    wanted = " | ".join(
        ", ".join(item.name for item in expression_sources(option))
        for option in expression.options
    )
    raise KeyError(f"no AnyOf option is satisfiable; wanted one of: {wanted}")


def resolve_options(expression: "Expression", available) -> "Expression":
    """`expression` with every AnyOf replaced by the option this checkpoint can satisfy."""
    if isinstance(expression, AnyOf):
        return resolve_options(choose_option(expression, available), available)
    if isinstance(expression, (Slice, Reshape, Transpose, Cast)):
        return replace(expression, source=resolve_options(expression.source, available))
    if isinstance(expression, Concat):
        return replace(
            expression,
            sources=tuple(resolve_options(part, available) for part in expression.sources),
        )
    return expression



def build_vision_recipes(
    text_width: int,
    *,
    layers: int = 27,
    hidden: int = 1152,
    intermediate: int = 4304,
    qkv_rows: int = 3456,
    patch_channels: int = 3,
    patch_temporal: int = 2,
    patch_size: int = 16,
    position_embeddings: int = 2304,
    merger_hidden: int = 4608,
) -> tuple[TensorRecipe, ...]:
    """Build Vision recipes for a target's own tower.

    The defaults are the Qwen3.6 tower the 27B and 35B carry. They are defaults
    only: every target ships a vision tower and the towers differ, so the geometry
    must be a parameter here for the same reason it is in `build_vision_specs` —
    and the two must agree, which `validate_recipe_coverage` enforces.
    """

    patch_rows = patch_channels * patch_temporal * patch_size * patch_size
    source_prefix = "model.visual."
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "vision/patch_embedding",
            Reshape(
                source(source_prefix + "patch_embed.proj.weight",
                       (hidden, patch_channels, patch_temporal, patch_size, patch_size)),
                (hidden, patch_rows),
            ),
        ),
        TensorRecipe(
            "vision/patch_embedding_bias",
            source(source_prefix + "patch_embed.proj.bias", (hidden,)),
        ),
        TensorRecipe(
            "vision/position_embedding",
            source(source_prefix + "pos_embed.weight", (position_embeddings, hidden)),
        ),
    ]

    for layer in range(layers):
        source_layer = source_prefix + f"blocks.{layer}."
        object_layer = f"vision/layers/{layer}/"
        for object_suffix, source_suffix, shape in (
            ("attention/qkv", "attn.qkv.weight", (qkv_rows, hidden)),
            ("attention/qkv_bias", "attn.qkv.bias", (qkv_rows,)),
            ("attention/output", "attn.proj.weight", (hidden, hidden)),
            ("attention/output_bias", "attn.proj.bias", (hidden,)),
            ("mlp/fc1", "mlp.linear_fc1.weight", (intermediate, hidden)),
            ("mlp/fc1_bias", "mlp.linear_fc1.bias", (intermediate,)),
            ("mlp/fc2", "mlp.linear_fc2.weight", (hidden, intermediate)),
            ("mlp/fc2_bias", "mlp.linear_fc2.bias", (hidden,)),
            ("norm1/weight", "norm1.weight", (hidden,)),
            ("norm1/bias", "norm1.bias", (hidden,)),
            ("norm2/weight", "norm2.weight", (hidden,)),
            ("norm2/bias", "norm2.bias", (hidden,)),
        ):
            recipes.append(
                TensorRecipe(
                    object_layer + object_suffix,
                    source(source_layer + source_suffix, shape),
                )
            )

    for object_suffix, source_suffix, shape in (
        ("fc1", "linear_fc1.weight", (merger_hidden, merger_hidden)),
        ("fc1_bias", "linear_fc1.bias", (merger_hidden,)),
        ("fc2", "linear_fc2.weight", (text_width, merger_hidden)),
        ("fc2_bias", "linear_fc2.bias", (text_width,)),
        ("norm/weight", "norm.weight", (hidden,)),
        ("norm/bias", "norm.bias", (hidden,)),
    ):
        recipes.append(
            TensorRecipe(
                "vision/merger/" + object_suffix,
                source(source_prefix + "merger." + source_suffix, shape),
            )
        )
    return tuple(recipes)


def validate_recipe_coverage(
    recipes: Sequence[TensorRecipe],
    tensor_specs: Sequence[TensorSpec],
) -> None:
    inventory_names = tuple(spec.name for spec in tensor_specs)
    recipe_names = tuple(recipe.object_name for recipe in recipes)
    if recipe_names != inventory_names:
        raise ValueError("recipe order or coverage does not match the tensor inventory")
    if len({recipe.object_name for recipe in recipes}) != len(recipes):
        raise ValueError("more than one recipe targets the same artifact object")
    inventory_by_name = {spec.name: spec for spec in tensor_specs}
    for recipe in recipes:
        expected = inventory_by_name[recipe.object_name].shape
        actual = expression_shape(recipe.expression)
        if actual != expected:
            raise ValueError(
                f"{recipe.object_name}: recipe shape {actual} != inventory {expected}"
            )


def source_requirements(
    recipes: Sequence[TensorRecipe],
) -> dict[str, SourceTensor]:
    requirements: dict[str, SourceTensor] = {}
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            previous = requirements.setdefault(requirement.name, requirement)
            if previous != requirement:
                raise ValueError(f"inconsistent source declaration for {requirement.name}")
    return requirements


def preflight_sources(
    model_dir: str | Path,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    with ShardReader.for_directory(model_dir) as reader:
        return preflight_source_reader(reader, recipes)



def _squeeze_units(shape) -> tuple[int, ...]:
    return tuple(extent for extent in shape if extent != 1)


def _same_up_to_unit_axes(stored, required) -> bool:
    """Whether two shapes describe the same rows, ignoring unit axes.

    A vector is a bare `[n]` in one export and a `[1, n]` Linear weight in another; the values
    and their order are identical either way. This is a storage convention, so it is tolerated
    for every checkpoint rather than listed per model, and the reader reshapes on read.
    """
    return _squeeze_units(stored) == _squeeze_units(required)


def preflight_source_reader(
    reader: ShardReader,
    recipes: Sequence[TensorRecipe],
) -> SourcePreflight:
    # An AnyOf object lists every spelling it accepts; only the one this checkpoint actually
    # has may be demanded, so options are resolved against the reader before requirements are
    # collected.
    resolved = tuple(
        replace(item, expression=resolve_options(item.expression, reader.has)) for item in recipes
    )
    requirements = source_requirements(resolved)
    metadata = reader.metadata(requirements)

    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, requirement in requirements.items():
        actual = metadata[name]
        if not _same_up_to_unit_axes(actual.shape, requirement.shape):
            raise ValueError(f"{name}: source shape {actual.shape} != required {requirement.shape}")
        if actual.dtype != requirement.dtype and not (
            requirement.dtype == "BF16" and actual.dtype == "F32"
        ):
            # surogate vendor patch (PATCHES.md #16): official Qwen3.5 releases
            # store some 1D control tensors (A_log, dt_bias) in F32 where the
            # GGUF-bridged dialect carries BF16; the encode path narrows to the
            # registered format either way, so F32 sources widen-in cleanly.
            raise ValueError(f"{name}: source dtype {actual.dtype} != required {requirement.dtype}")
        dtype_counts[actual.dtype] = dtype_counts.get(actual.dtype, 0) + 1
        shards.add(actual.shard)

    return SourcePreflight(
        recipe_count=len(resolved),
        source_tensor_count=len(requirements),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def materialize_expression(
    expression: Expression,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if isinstance(expression, AnyOf):
        return materialize_expression(
            choose_option(expression, reader.has), reader, derived_tensors
        )
    if isinstance(expression, SourceTensor):
        tensor = reader.get(expression.name)
        if tuple(tensor.shape) != tuple(expression.shape) and _same_up_to_unit_axes(
            tuple(tensor.shape), tuple(expression.shape)
        ):
            tensor = tensor.reshape(tuple(expression.shape))
        if tuple(tensor.shape) != expression.shape:
            raise ValueError(
                f"{expression.name}: source shape {tuple(tensor.shape)} != {expression.shape}"
            )
        return tensor

    if isinstance(expression, Slice):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.narrow(expression.axis, expression.begin, expression.end - expression.begin)

    if isinstance(expression, Reshape):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.reshape(expression.shape)

    if isinstance(expression, Transpose):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        return tensor.permute(expression.axes).contiguous()

    if isinstance(expression, Concat):
        tensors = [
            materialize_expression(part, reader, derived_tensors)
            for part in expression.sources
        ]
        return torch.cat(tensors, dim=expression.axis)

    if isinstance(expression, Cast):
        tensor = materialize_expression(expression.source, reader, derived_tensors)
        if expression.dtype != FP32:
            raise ValueError(f"unsupported direct cast target {expression.dtype}")
        return tensor.to(torch.float32)

    if isinstance(expression, DraftHeadTokenIds):
        if derived_tensors is None or "text/draft_head_token_ids" not in derived_tensors:
            raise ValueError("draft-head token IDs have not been derived")
        return derived_tensors["text/draft_head_token_ids"].to(torch.int32)

    if isinstance(expression, GatherRows):
        if derived_tensors is None or expression.token_ids_object not in derived_tensors:
            raise ValueError("draft-head token IDs are required to gather head rows")
        source_tensor = materialize_expression(expression.source, reader, derived_tensors)
        token_ids = derived_tensors[expression.token_ids_object].to(torch.int64)
        return source_tensor.index_select(0, token_ids)

    raise TypeError(f"unknown recipe expression {type(expression)!r}")


def materialize_recipe(
    recipe: TensorRecipe,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    tensor = materialize_expression(recipe.expression, reader, derived_tensors)
    expected = expression_shape(recipe.expression)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{recipe.object_name}: materialized shape {tuple(tensor.shape)} != {expected}"
        )
    return tensor


__all__ = [
    "Cast",
    "Concat",
    "DraftHeadTokenIds",
    "Expression",
    "GatherRows",
    "Reshape",
    "SOURCE_DTYPE",
    "ShardReader",
    "Slice",
    "SourcePreflight",
    "SourceTensor",
    "TensorRecipe",
    "Transpose",
    "attention_qproj_part",
    "build_vision_recipes",
    "expression_shape",
    "expression_sources",
    "materialize_expression",
    "materialize_recipe",
    "preflight_source_reader",
    "preflight_sources",
    "source",
    "source_requirements",
    "validate_recipe_coverage",
]
