"""Single-source recipe for the Qwen3.5-4B NVFP4 artifact.

``AxionML/Qwen3.5-4B-NVFP4`` is a ModelOpt export: every linear weight ships as
packed E2M1 codes with per-16 E4M3 block scales, a global weight divisor and a
calibrated activation divisor, while the norms, the convolution and the tied
embedding stay BF16 in the same file. So one checkpoint supplies the whole
artifact - the NVFP4 blocks pass through untouched, and only the embedding is
re-encoded (FP8 row-scaled, as the byte-wide head this target expects).

ModelOpt spells its fields ``weight`` / ``weight_scale`` / ``weight_scale_2``
where compressed-tensors exports use ``weight_packed`` / ``weight_scale`` /
``weight_global_scale``; the materializers below read the ModelOpt names.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from surogate.serve.convert.common.safetensors import ShardReader

from .. import inventory as family_inventory


#: The one checkpoint published under this export; the tables below are its size.
GEOMETRY = family_inventory.Geometry(
    layers=32, hidden=2560, intermediate=9216, query_heads=16, kv_heads=4,
    gdn_value_heads=32,
)
EXPORT = family_inventory.export_inventory(
    family_inventory.NVFP4_UNIFORM, GEOMETRY)



QUANTIZED_REPOSITORY = "AxionML/Qwen3.5-4B-NVFP4"
QUANTIZED_REVISION = "main"
BASE_REPOSITORY = QUANTIZED_REPOSITORY
BASE_REVISION = QUANTIZED_REVISION

HIDDEN = 2560
LAYERS = 32
ATTENTION_HEADS = 16
HEAD_DIM = 256

WEIGHT_FIELD = "weight"
SCALE_FIELD = "weight_scale"
GLOBAL_SCALE_FIELD = "weight_scale_2"
INPUT_SCALE_FIELD = "input_scale"

EMBEDDING_SOURCE = "model.embed_tokens.weight"


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


@dataclass(frozen=True, slots=True)
class InputDivisorRecipe:
    object_name: str
    sources: tuple[MatrixSource, ...]
    weight_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectRecipe:
    object_name: str
    source_name: str
    shape: tuple[int, ...]
    dequantize: bool = False
    # The convolution ships channel-major; the artifact stores it tap-major, so the
    # source is folded to source_shape and transposed rather than reshaped in place.
    source_shape: tuple[int, ...] | None = None
    transpose: bool = False


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    """Query rows interleave query and gate per head; split them apart."""

    begin = HEAD_DIM if gate else 0
    stride = 2 * HEAD_DIM
    return MatrixPart(
        source,
        tuple(
            RowRange(head * stride + begin, head * stride + begin + HEAD_DIM)
            for head in range(ATTENTION_HEADS)
        ),
    )


def _build_matrix_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
]:
    weights: list[Nvfp4WeightRecipe] = []
    divisors: list[InputDivisorRecipe] = []

    for layer in range(LAYERS):
        source_prefix = f"model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"

        if layer in EXPORT.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 8192, HIDDEN)
            key = _source(source_prefix + "self_attn.k_proj", 1024, HIDDEN)
            value = _source(source_prefix + "self_attn.v_proj", 1024, HIDDEN)
            output = _source(source_prefix + "self_attn.o_proj", HIDDEN, 4096)
            fused_sources = (query, key, value)
            weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (10240, HIDDEN),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                        fused_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/input_projection/input_scale_divisor",
                        fused_sources,
                        (object_prefix + "attention/query_key_gate_value",),
                    ),
                    InputDivisorRecipe(
                        object_prefix + "attention/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "attention/output",),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 8192, HIDDEN
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 4096, HIDDEN)
            output = _source(source_prefix + "linear_attn.out_proj", HIDDEN, 4096)
            fused_sources = (query_key_value, z)
            weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        (12288, HIDDEN),
                        (_all(query_key_value), _all(z)),
                        fused_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix + "gdn/input_projection/input_scale_divisor",
                        fused_sources,
                        (object_prefix + "gdn/query_key_value_z",),
                    ),
                    InputDivisorRecipe(
                        object_prefix + "gdn/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "gdn/output",),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 9216, HIDDEN)
        up = _source(source_prefix + "mlp.up_proj", 9216, HIDDEN)
        down = _source(source_prefix + "mlp.down_proj", HIDDEN, 9216)
        weights.extend(
            (
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (18432, HIDDEN),
                    (_all(gate), _all(up)),
                    (gate, up),
                ),
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/down", down.shape, (_all(down),), (down,)
                ),
            )
        )
        divisors.extend(
            (
                InputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    (gate, up),
                    (object_prefix + "mlp/gate_up",),
                ),
                InputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "mlp/down",),
                ),
            )
        )

    return tuple(weights), tuple(divisors)


def _build_direct_recipes() -> tuple[DirectRecipe, ...]:
    items: list[DirectRecipe] = []
    for layer in range(LAYERS):
        source_prefix = f"model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        items.append(
            DirectRecipe(
                object_prefix + "input_norm",
                source_prefix + "input_layernorm.weight",
                (HIDDEN,),
            )
        )
        if layer in EXPORT.FULL_ATTENTION_LAYERS:
            items.extend(
                (
                    DirectRecipe(
                        object_prefix + "attention/query_norm",
                        source_prefix + "self_attn.q_norm.weight",
                        (HEAD_DIM,),
                    ),
                    DirectRecipe(
                        object_prefix + "attention/key_norm",
                        source_prefix + "self_attn.k_norm.weight",
                        (HEAD_DIM,),
                    ),
                )
            )
        else:
            items.extend(
                (
                    DirectRecipe(
                        object_prefix + "gdn/a_log",
                        source_prefix + "linear_attn.A_log",
                        (32,),
                    ),
                    DirectRecipe(
                        object_prefix + "gdn/dt_bias",
                        source_prefix + "linear_attn.dt_bias",
                        (32,),
                    ),
                    DirectRecipe(
                        object_prefix + "gdn/convolution",
                        source_prefix + "linear_attn.conv1d.weight",
                        (4, 8192),
                        source_shape=(8192, 4),
                        transpose=True,
                    ),
                    # in_proj_a and in_proj_b ship as NVFP4 blocks; the artifact wants
                    # them dense, so they are decoded back to BF16 on the way in.
                    DirectRecipe(
                        object_prefix + "gdn/a_projection",
                        source_prefix + "linear_attn.in_proj_a",
                        (32, HIDDEN),
                        dequantize=True,
                    ),
                    DirectRecipe(
                        object_prefix + "gdn/b_projection",
                        source_prefix + "linear_attn.in_proj_b",
                        (32, HIDDEN),
                        dequantize=True,
                    ),
                    DirectRecipe(
                        object_prefix + "gdn/norm",
                        source_prefix + "linear_attn.norm.weight",
                        (128,),
                    ),
                )
            )
        items.append(
            DirectRecipe(
                object_prefix + "post_attention_norm",
                source_prefix + "post_attention_layernorm.weight",
                (HIDDEN,),
            )
        )
    items.append(
        DirectRecipe(
            "text/final_norm", "model.norm.weight", (HIDDEN,)
        )
    )
    return tuple(items)


NVFP4_WEIGHT_RECIPES, INPUT_DIVISOR_RECIPES = _build_matrix_recipes()
NVFP4_WEIGHTS_BY_NAME = {item.object_name: item for item in NVFP4_WEIGHT_RECIPES}
INPUT_DIVISORS_BY_NAME = {item.object_name: item for item in INPUT_DIVISOR_RECIPES}
DIRECT_RECIPES = _build_direct_recipes()
DIRECT_BY_NAME = {item.object_name: item for item in DIRECT_RECIPES}
NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source for recipe in NVFP4_WEIGHT_RECIPES for part in recipe.parts
    )
)
EXPECTED_FIELDS = (WEIGHT_FIELD, SCALE_FIELD, GLOBAL_SCALE_FIELD, INPUT_SCALE_FIELD)


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    if len(part.rows) == 1 and part.rows[0].begin == 0 and part.rows[0].end == tensor.shape[0]:
        return tensor
    return torch.cat([tensor[item.begin : item.end] for item in part.rows], dim=0)


def _reciprocal(value: float, label: str) -> float:
    """ModelOpt stores multipliers; the runtime fields are divisors.

    A ModelOpt export writes ``weight_scale_2 = amax / (448 * 6)`` and the matching
    ``input_scale``, i.e. the number a code is multiplied by. The engine's
    ``weight_scale_divisor`` / ``input_scale_divisor`` are the compressed-tensors
    convention instead - block_scale = divisor * max_abs / 6, undone by
    alpha = 1 / (input_divisor * weight_divisor) - so each value has to be inverted
    on the way in.
    """

    if not value > 0.0:
        raise ValueError(f"{label}: scale must be positive, got {value}")
    return 1.0 / value


def _same_scalar(reader: ShardReader, sources: tuple[MatrixSource, ...], field: str) -> float:
    values = []
    for source in sources:
        value = reader.get(source.field(field))
        if value.numel() != 1:
            raise ValueError(f"{source.name}.{field}: expected one scalar")
        values.append(float(value.reshape(()).to(torch.float32)))
    if any(abs(value - values[0]) > 0.0 for value in values):
        raise ValueError(
            f"fused sources disagree on {field}: " + ", ".join(s.name for s in sources)
        )
    return values[0]


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe, reader: ShardReader
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for part in recipe.parts:
        words = cache.get(part.source.name)
        if words is None:
            n, k = part.source.shape
            packed = reader.get(part.source.field(WEIGHT_FIELD))
            scales = reader.get(part.source.field(SCALE_FIELD))
            if (
                packed.dtype != torch.uint8
                or tuple(packed.shape) != (n, k // 2)
                or scales.dtype != torch.float8_e4m3fn
                or tuple(scales.shape) != (n, k // 16)
            ):
                raise ValueError(f"{part.source.name}: NVFP4 source signature mismatch")
            words = (packed, scales.view(torch.uint8))
            cache[part.source.name] = words
        packed_parts.append(_select_rows(words[0], part))
        scale_parts.append(_select_rows(words[1], part))
    packed = packed_parts[0].contiguous() if len(packed_parts) == 1 else torch.cat(packed_parts, dim=0)
    scales = scale_parts[0].contiguous() if len(scale_parts) == 1 else torch.cat(scale_parts, dim=0)
    if tuple(packed.shape) != (recipe.shape[0], recipe.shape[1] // 2) or tuple(scales.shape) != (
        recipe.shape[0],
        recipe.shape[1] // 16,
    ):
        raise ValueError(f"{recipe.object_name}: NVFP4 shape mismatch after fusion")
    divisor = _reciprocal(
        _same_scalar(reader, recipe.divisor_sources, GLOBAL_SCALE_FIELD),
        recipe.object_name + "." + GLOBAL_SCALE_FIELD,
    )
    return packed, scales, struct.pack("<f", divisor)


def materialize_input_divisor(recipe: InputDivisorRecipe, reader: ShardReader) -> bytes:
    return struct.pack(
        "<f",
        _reciprocal(
            _same_scalar(reader, recipe.sources, INPUT_SCALE_FIELD),
            recipe.object_name,
        ),
    )


def validate_recipe() -> None:
    """Every NVFP4 object in the inventory must have exactly one recipe."""

    expected = {
        spec.name
        for spec in EXPORT.TEXT_CORE_TENSOR_SPECS
        if spec.format == EXPORT.NVFP4
    }
    produced = set(NVFP4_WEIGHTS_BY_NAME)
    if expected != produced:
        missing = sorted(expected - produced)[:4]
        extra = sorted(produced - expected)[:4]
        raise ValueError(
            f"NVFP4 recipe coverage mismatch; missing={missing} extra={extra}"
        )
    for recipe in NVFP4_WEIGHT_RECIPES:
        rows = sum(part.output_rows for part in recipe.parts)
        if rows != recipe.shape[0]:
            raise ValueError(f"{recipe.object_name}: fused rows {rows} != {recipe.shape[0]}")
        if any(part.source.shape[1] != recipe.shape[1] for part in recipe.parts):
            raise ValueError(f"{recipe.object_name}: incompatible source K")
