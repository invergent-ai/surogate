"""Leaf storage contracts shared by explicit Qwen3.6 artifact targets.

This module owns only format/layout vocabulary, passive inventory records, the
common frontend resources, and the checkpoint-invariant Vision inventory.
Each target still owns its complete ordered object inventory.
"""

from __future__ import annotations

from dataclasses import dataclass


CONTIGUOUS_LAYOUT = "contiguous-le-v1"
ROW_SPLIT_LAYOUT = "row-split-k128-v1"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
ROW_SCALE_LAYOUT = "row-scale-v1"
GGML_BLOCKS_LAYOUT = "ggml-blocks-v1"
RESOURCE_ENCODING = "raw-bytes-v1"

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
Q4 = "Q4G64_F16S"
Q5 = "Q5G64_F16S"
Q6 = "Q6G64_F16S"
W8 = "W8G32_F16S"
NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
Q2_K = "Q2_K"
Q3_K = "Q3_K"
Q4_K = "Q4_K"
Q5_K = "Q5_K"
Q6_K = "Q6_K"
Q8_0 = "Q8_0"
GGML_BLOCK_FORMAT_NAMES = (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0)

DIRECT_FORMATS = frozenset((BF16, FP32, I32))
# The engine's nine formats and four layouts (csrc/src/serve/artifact/reader.h). Every
# target used to re-declare the NVFP4 pair beside its own inventory; they are named
# once here so a converter that reads formats off a checkpoint has one vocabulary.
FORMAT_NAMES = (BF16, FP32, I32, Q4, Q5, Q6, W8, NVFP4, FP8, *GGML_BLOCK_FORMAT_NAMES)
LAYOUT_NAMES = (CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT, BLOCK_SCALE_LAYOUT, ROW_SCALE_LAYOUT, GGML_BLOCKS_LAYOUT)

VISION_LAYERS = tuple(range(27))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str
    layout: str
    #: Stretches of a file the artifact serves this object from instead of storing it; see
    #: `surogate.serve.tools.artifact.container.TensorSpec.runs`.
    runs: tuple[tuple[int, int, int], ...] = ()
    transform: str = ""
    #: Source group for each destination group, when the transform also carries a column
    #: permutation. Empty when the columns are in order.
    group_map: tuple[int, ...] = ()

    @property
    def kind(self) -> str:
        return "tensor"


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    name: str
    encoding: str = RESOURCE_ENCODING

    @property
    def kind(self) -> str:
        return "resource"


@dataclass(frozen=True, slots=True)
class LogicalRowViewSpec:
    name_pattern: str
    parent_pattern: str
    row_begin: int
    row_end: int
    shape: tuple[int, int]
    layers: tuple[int, ...] | None

    @property
    def row_count(self) -> int:
        return self.row_end - self.row_begin


@dataclass(frozen=True, slots=True)
class LogicalAliasSpec:
    role_pattern: str
    object_patterns: tuple[str, ...]
    layers: tuple[int, ...] | None = None
    axis_order: tuple[int, ...] | None = None


StoredObjectSpec = TensorSpec | ResourceSpec


def tensor_spec(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
) -> TensorSpec:
    """Build a tensor spec with the canonical layout for its numeric format."""

    if numeric_format in DIRECT_FORMATS:
        layout = CONTIGUOUS_LAYOUT
    elif numeric_format in GGML_BLOCK_FORMAT_NAMES:
        layout = GGML_BLOCKS_LAYOUT
    else:
        layout = ROW_SPLIT_LAYOUT
    return TensorSpec(name=name, shape=shape, format=numeric_format, layout=layout)


RESOURCE_SPECS = tuple(
    ResourceSpec(name)
    for name in (
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    )
)


def build_vision_specs(
    text_width: int,
    *,
    layers: int = 27,
    hidden: int = 1152,
    intermediate: int = 4304,
    qkv_rows: int = 3456,
    patch_rows: int = 1536,
    position_embeddings: int = 2304,
    merger_hidden: int = 4608,
) -> tuple[TensorSpec, ...]:
    """Build a vision inventory for a target's own tower.

    The defaults are the Qwen3.6 tower (27 layers of 1152) that the 27B and 35B
    carry. They are only defaults: the Qwen3.5 targets have towers of their own —
    the 0.8B is 12 layers of 768, the 2B and 4B are 24 of 1024 — and serving
    carries the tower on every target, so the geometry has to be a parameter
    rather than a constant. The DSL declaration is where each target's numbers
    come from; `tests/test_serve_contract.py` checks that these agree with it.
    """

    specs: list[TensorSpec] = [
        tensor_spec("vision/patch_embedding", (hidden, patch_rows), Q6),
        tensor_spec("vision/patch_embedding_bias", (hidden,), BF16),
        tensor_spec("vision/position_embedding", (position_embeddings, hidden), BF16),
    ]

    for layer in range(layers):
        prefix = f"vision/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "attention/qkv", (qkv_rows, hidden), Q4),
                tensor_spec(prefix + "attention/qkv_bias", (qkv_rows,), BF16),
                tensor_spec(prefix + "attention/output", (hidden, hidden), Q5),
                tensor_spec(prefix + "attention/output_bias", (hidden,), BF16),
                tensor_spec(prefix + "mlp/fc1", (intermediate, hidden), Q4),
                tensor_spec(prefix + "mlp/fc1_bias", (intermediate,), BF16),
                tensor_spec(prefix + "mlp/fc2", (hidden, intermediate), Q5),
                tensor_spec(prefix + "mlp/fc2_bias", (hidden,), BF16),
                tensor_spec(prefix + "norm1/weight", (hidden,), BF16),
                tensor_spec(prefix + "norm1/bias", (hidden,), BF16),
                tensor_spec(prefix + "norm2/weight", (hidden,), BF16),
                tensor_spec(prefix + "norm2/bias", (hidden,), BF16),
            )
        )

    specs.extend(
        (
            tensor_spec("vision/merger/fc1", (merger_hidden, merger_hidden), W8),
            tensor_spec("vision/merger/fc1_bias", (merger_hidden,), BF16),
            tensor_spec("vision/merger/fc2", (text_width, merger_hidden), W8),
            tensor_spec("vision/merger/fc2_bias", (text_width,), BF16),
            tensor_spec("vision/merger/norm/weight", (hidden,), BF16),
            tensor_spec("vision/merger/norm/bias", (hidden,), BF16),
        )
    )
    return tuple(specs)



def tied_duplicate_objects(recipes_by_name, specs) -> tuple[str, ...]:
    """Object names whose recipe is byte-for-byte another object's, in spec order.

    A tied language-model head is the whole of this today: the checkpoint has no
    `lm_head.weight` and the recipe reads `embed_tokens.weight`, so the artifact would carry
    the vocabulary table twice -- on a 2B Q4_K_M that is 417 MB of 2,050, and every one of
    those bytes is read again on each decode step. The loader binds the survivor once and
    points both plans at it; an artifact that predates this still carries both and still loads.
    """
    seen: dict = {}
    duplicates: list[str] = []
    for spec in specs:
        name = getattr(spec, "name", None)
        recipe = recipes_by_name.get(name)
        if recipe is None:
            continue
        key = (repr(recipe.expression), tuple(spec.shape), spec.format, spec.layout)
        if key in seen:
            duplicates.append(name)
        else:
            seen[key] = name
    return tuple(duplicates)

__all__ = [
    "tied_duplicate_objects",
    "BF16",
    "CONTIGUOUS_LAYOUT",
    "DIRECT_FORMATS",
    "FORMAT_NAMES",
    "FP32",
    "I32",
    "LAYOUT_NAMES",
    "LogicalAliasSpec",
    "LogicalRowViewSpec",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_ENCODING",
    "RESOURCE_SPECS",
    "ROW_SPLIT_LAYOUT",
    "ResourceSpec",
    "StoredObjectSpec",
    "TensorSpec",
    "VISION_LAYERS",
    "W8",
    "build_vision_specs",
    "tensor_spec",
]
