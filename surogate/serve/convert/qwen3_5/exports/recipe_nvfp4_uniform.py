"""Matrix row programs for hybrid projections at resolved checkpoint dimensions.

These source ranges also serve FP8 and mixed storage. Encodings are resolved by the
checkpoint reader; no object list is constructed until a Geometry is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from surogate.serve.convert.common.safetensors import ShardReader

from .. import inventory as family_inventory


WEIGHT_FIELD = "weight"
SCALE_FIELD = "weight_scale"
GLOBAL_SCALE_FIELD = "weight_scale_2"
INPUT_SCALE_FIELD = "input_scale"

#: Resolve the decoder prefix from checkpoint tensor names.
SOURCE_ROOTS = ("model.language_model.", "model.")


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


def _q_part(source: MatrixSource, gate: bool, geometry: family_inventory.Geometry) -> MatrixPart:
    """Query rows interleave query and gate per head; split them apart."""

    head_dim = geometry.head_dim
    begin = head_dim if gate else 0
    stride = 2 * head_dim
    return MatrixPart(
        source,
        tuple(
            RowRange(head * stride + begin, head * stride + begin + head_dim)
            for head in range(geometry.query_heads)
        ),
    )


@dataclass(frozen=True, slots=True)
class UniformRecipes:
    """Every recipe of one checkpoint: its NVFP4 matrices, their activation divisors, and
    the objects copied dense, all at the checkpoint's own dimensions."""

    geometry: family_inventory.Geometry
    source_root: str
    nvfp4_weights: tuple[Nvfp4WeightRecipe, ...]
    input_divisors: tuple[InputDivisorRecipe, ...]
    direct: tuple[DirectRecipe, ...]

    @property
    def embedding_source(self) -> str:
        return self.source_root + "embed_tokens.weight"

    @property
    def nvfp4_weights_by_name(self) -> dict[str, Nvfp4WeightRecipe]:
        return {item.object_name: item for item in self.nvfp4_weights}

    @property
    def input_divisors_by_name(self) -> dict[str, InputDivisorRecipe]:
        return {item.object_name: item for item in self.input_divisors}

    @property
    def direct_by_name(self) -> dict[str, DirectRecipe]:
        return {item.object_name: item for item in self.direct}

    @property
    def nvfp4_sources(self) -> tuple[MatrixSource, ...]:
        return tuple(dict.fromkeys(part.source for r in self.nvfp4_weights for part in r.parts))


def source_root_of(names) -> str:
    """The decoder's tensor-name root, from the names the checkpoint holds."""
    names = list(names)
    for root in SOURCE_ROOTS:
        if any(name.startswith(root + "layers.") for name in names):
            return root
    raise ValueError("no decoder layers under any known root: " + ", ".join(SOURCE_ROOTS))


def build(geometry: family_inventory.Geometry, source_root: str) -> UniformRecipes:
    """The recipes at these dimensions. Row counts follow the geometry: the query projection
    interleaves query and gate per head (2 x query_size rows), the GDN input projection is
    [q | k | v] over the convolution width with z beside it, and every other matrix is the
    shape its object declares."""
    g = geometry
    hidden = g.hidden
    weights: list[Nvfp4WeightRecipe] = []
    divisors: list[InputDivisorRecipe] = []
    direct: list[DirectRecipe] = []
    attention_layers = set(g.full_attention_layers)

    for layer in range(g.layers):
        source_prefix = f"{source_root}layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        direct.append(DirectRecipe(object_prefix + "input_norm",
                                   source_prefix + "input_layernorm.weight", (hidden,)))
        if layer in attention_layers:
            query = _source(source_prefix + "self_attn.q_proj", 2 * g.query_size, hidden)
            key = _source(source_prefix + "self_attn.k_proj", g.kv_size, hidden)
            value = _source(source_prefix + "self_attn.v_proj", g.kv_size, hidden)
            output = _source(source_prefix + "self_attn.o_proj", hidden, g.query_size)
            fused_sources = (query, key, value)
            weights.extend((
                Nvfp4WeightRecipe(
                    object_prefix + "attention/query_key_gate_value",
                    (2 * g.query_size + 2 * g.kv_size, hidden),
                    (_q_part(query, False, g), _all(key), _q_part(query, True, g), _all(value)),
                    fused_sources,
                ),
                Nvfp4WeightRecipe(object_prefix + "attention/output", output.shape,
                                  (_all(output),), (output,)),
            ))
            divisors.extend((
                InputDivisorRecipe(object_prefix + "attention/input_projection/input_scale_divisor",
                                   fused_sources, (object_prefix + "attention/query_key_gate_value",)),
                InputDivisorRecipe(object_prefix + "attention/output_projection/input_scale_divisor",
                                   (output,), (object_prefix + "attention/output",)),
            ))
            direct.extend((
                DirectRecipe(object_prefix + "attention/query_norm",
                             source_prefix + "self_attn.q_norm.weight", (g.head_dim,)),
                DirectRecipe(object_prefix + "attention/key_norm",
                             source_prefix + "self_attn.k_norm.weight", (g.head_dim,)),
            ))
        else:
            query_key_value = _source(source_prefix + "linear_attn.in_proj_qkv", g.convolution_dim, hidden)
            z = _source(source_prefix + "linear_attn.in_proj_z", g.value_dim, hidden)
            output = _source(source_prefix + "linear_attn.out_proj", hidden, g.value_dim)
            fused_sources = (query_key_value, z)
            weights.extend((
                Nvfp4WeightRecipe(object_prefix + "gdn/query_key_value_z",
                                  (g.convolution_dim + g.value_dim, hidden),
                                  (_all(query_key_value), _all(z)), fused_sources),
                Nvfp4WeightRecipe(object_prefix + "gdn/output", output.shape, (_all(output),), (output,)),
            ))
            divisors.extend((
                InputDivisorRecipe(object_prefix + "gdn/input_projection/input_scale_divisor",
                                   fused_sources, (object_prefix + "gdn/query_key_value_z",)),
                InputDivisorRecipe(object_prefix + "gdn/output_projection/input_scale_divisor",
                                   (output,), (object_prefix + "gdn/output",)),
            ))
            direct.extend((
                DirectRecipe(object_prefix + "gdn/a_log", source_prefix + "linear_attn.A_log",
                             (g.gdn_value_heads,)),
                DirectRecipe(object_prefix + "gdn/dt_bias", source_prefix + "linear_attn.dt_bias",
                             (g.gdn_value_heads,)),
                DirectRecipe(object_prefix + "gdn/convolution", source_prefix + "linear_attn.conv1d.weight",
                             (g.gdn_conv_kernel, g.convolution_dim),
                             source_shape=(g.convolution_dim, g.gdn_conv_kernel), transpose=True),
                # in_proj_a and in_proj_b ship as NVFP4 blocks; the artifact wants them
                # dense, so they are decoded back to BF16 on the way in.
                DirectRecipe(object_prefix + "gdn/a_projection", source_prefix + "linear_attn.in_proj_a",
                             (g.gdn_value_heads, hidden), dequantize=True),
                DirectRecipe(object_prefix + "gdn/b_projection", source_prefix + "linear_attn.in_proj_b",
                             (g.gdn_value_heads, hidden), dequantize=True),
                DirectRecipe(object_prefix + "gdn/norm", source_prefix + "linear_attn.norm.weight",
                             (g.gdn_value_head_dim,)),
            ))
        gate = _source(source_prefix + "mlp.gate_proj", g.intermediate, hidden)
        up = _source(source_prefix + "mlp.up_proj", g.intermediate, hidden)
        down = _source(source_prefix + "mlp.down_proj", hidden, g.intermediate)
        weights.extend((
            Nvfp4WeightRecipe(object_prefix + "mlp/gate_up", (2 * g.intermediate, hidden),
                              (_all(gate), _all(up)), (gate, up)),
            Nvfp4WeightRecipe(object_prefix + "mlp/down", down.shape, (_all(down),), (down,)),
        ))
        divisors.extend((
            InputDivisorRecipe(object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                               (gate, up), (object_prefix + "mlp/gate_up",)),
            InputDivisorRecipe(object_prefix + "mlp/down_projection/input_scale_divisor",
                               (down,), (object_prefix + "mlp/down",)),
        ))
        direct.append(DirectRecipe(object_prefix + "post_attention_norm",
                                   source_prefix + "post_attention_layernorm.weight", (hidden,)))
    direct.append(DirectRecipe("text/final_norm", source_root + "norm.weight", (hidden,)))
    return UniformRecipes(g, source_root, tuple(weights), tuple(divisors), tuple(direct))


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


def validate(recipes: UniformRecipes, export) -> None:
    """Every NVFP4 object in the inventory must have exactly one recipe."""
    expected = {
        spec.name for spec in export.TEXT_CORE_TENSOR_SPECS if spec.format == export.NVFP4
    }
    produced = set(recipes.nvfp4_weights_by_name)
    if expected != produced:
        missing = sorted(expected - produced)[:4]
        extra = sorted(produced - expected)[:4]
        raise ValueError(
            f"NVFP4 recipe coverage mismatch; missing={missing} extra={extra}"
        )
    for recipe in recipes.nvfp4_weights:
        rows = sum(part.output_rows for part in recipe.parts)
        if rows != recipe.shape[0]:
            raise ValueError(f"{recipe.object_name}: fused rows {rows} != {recipe.shape[0]}")
        if any(part.source.shape[1] != recipe.shape[1] for part in recipe.parts):
            raise ValueError(f"{recipe.object_name}: incompatible source K")
