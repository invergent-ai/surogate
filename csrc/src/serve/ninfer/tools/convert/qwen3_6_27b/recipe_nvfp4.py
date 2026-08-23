"""Closed dual-source recipe for the additive Qwen3.6-27B NVFP4 artifact."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader

from . import inventory_nvfp4 as inventory


BASE_REPOSITORY = "Qwen/Qwen3.6-27B"
BASE_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
NVFP4_REPOSITORY = (
    "rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm"
)
NVFP4_REVISION = "9b5389d4a1e207daab2d47732efea57d7e946dcf"


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class Nvfp4Source:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class Nvfp4Part:
    source: Nvfp4Source
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[Nvfp4Part, ...]
    divisor_sources: tuple[Nvfp4Source, ...]


@dataclass(frozen=True, slots=True)
class InputDivisorRecipe:
    object_name: str
    sources: tuple[Nvfp4Source, ...]
    weight_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Bf16Comparison:
    source: Nvfp4Source


def _source(name: str, n: int, k: int) -> Nvfp4Source:
    return Nvfp4Source(name, (n, k))


def _all(source: Nvfp4Source) -> Nvfp4Part:
    return Nvfp4Part(source, (RowRange(0, source.shape[0]),))


def _part(source: Nvfp4Source, begin: int, end: int) -> Nvfp4Part:
    return Nvfp4Part(source, (RowRange(begin, end),))


def _q_part(source: Nvfp4Source, gate: bool) -> Nvfp4Part:
    begin = 256 if gate else 0
    return Nvfp4Part(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _build_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
    tuple[tuple[Nvfp4Source, ...], ...],
]:
    weights: list[Nvfp4WeightRecipe] = []
    inputs: list[InputDivisorRecipe] = []
    weight_groups: list[tuple[Nvfp4Source, ...]] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"

        if layer in inventory.FULL_ATTENTION_LAYERS:
            q = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            k = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            v = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            o = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            if layer in inventory.NVFP4_ATTENTION_INPUT_LAYERS:
                group = (q, k, v)
                weight_groups.append(group)
                weights.append(
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (_q_part(q, False), _all(k), _q_part(q, True), _all(v)),
                        group,
                    )
                )
                inputs.append(
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/input_projection/input_scale_divisor",
                        group,
                        (
                            object_prefix + "attention/query_key_gate_value",
                        ),
                    )
                )
            if layer in inventory.NVFP4_ATTENTION_OUTPUT_LAYERS:
                weights.append(
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/output",
                        o.shape,
                        (_all(o),),
                        (o,),
                    )
                )
                inputs.append(
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/output_projection/input_scale_divisor",
                        (o,),
                        (object_prefix + "attention/output",),
                    )
                )
        else:
            qkv = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            out = _source(source_prefix + "linear_attn.out_proj", 5120, 6144)
            group = (qkv, z)
            weight_groups.append(group)
            weights.append(
                Nvfp4WeightRecipe(
                    object_prefix + "gdn/query_key_value_z",
                    (16384, 5120),
                    (_all(qkv), _all(z)),
                    group,
                )
            )
            inputs.append(
                InputDivisorRecipe(
                    object_prefix + "gdn/input_projection/input_scale_divisor",
                    group,
                    (
                        object_prefix + "gdn/query_key_value_z",
                    ),
                )
            )
            if layer in inventory.NVFP4_GDN_OUTPUT_LAYERS:
                weights.append(
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/output",
                        out.shape,
                        (_all(out),),
                        (out,),
                    )
                )
                inputs.append(
                    InputDivisorRecipe(
                        object_prefix + "gdn/output_projection/input_scale_divisor",
                        (out,),
                        (object_prefix + "gdn/output",),
                    )
                )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_group = (gate, up)
        weight_groups.append(gate_up_group)
        weights.extend(
            (
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (34816, 5120),
                    (_all(gate), _all(up)),
                    gate_up_group,
                ),
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/down",
                    down.shape,
                    (_all(down),),
                    (down,),
                ),
            )
        )
        inputs.extend(
            (
                InputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    gate_up_group,
                    (object_prefix + "mlp/gate_up",),
                ),
                InputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "mlp/down",),
                ),
            )
        )

    return tuple(weights), tuple(inputs), tuple(weight_groups)


def _build_bf16_comparisons() -> tuple[Bf16Comparison, ...]:
    result: list[Bf16Comparison] = []
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}."
        if layer in inventory.FULL_ATTENTION_LAYERS:
            if layer in inventory.EARLY_ATTENTION_INPUT_LAYERS:
                result.extend(
                    Bf16Comparison(source)
                    for source in (
                        _source(prefix + "self_attn.q_proj", 12288, 5120),
                        _source(prefix + "self_attn.k_proj", 1024, 5120),
                        _source(prefix + "self_attn.v_proj", 1024, 5120),
                    )
                )
            if layer in inventory.BF16_ATTENTION_OUTPUT_LAYERS:
                result.append(
                    Bf16Comparison(
                        _source(prefix + "self_attn.o_proj", 5120, 6144)
                    )
                )
        else:
            result.extend(
                Bf16Comparison(source)
                for source in (
                    _source(prefix + "linear_attn.in_proj_a", 48, 5120),
                    _source(prefix + "linear_attn.in_proj_b", 48, 5120),
                )
            )
            if layer in inventory.BF16_GDN_OUTPUT_LAYERS:
                result.append(
                    Bf16Comparison(
                        _source(prefix + "linear_attn.out_proj", 5120, 6144)
                    )
                )
    return tuple(result)


NVFP4_WEIGHT_RECIPES, INPUT_DIVISOR_RECIPES, WEIGHT_DIVISOR_GROUPS = (
    _build_recipes()
)
NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in NVFP4_WEIGHT_RECIPES
}
INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in INPUT_DIVISOR_RECIPES
}
BF16_COMPARISONS = _build_bf16_comparisons()

NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in NVFP4_WEIGHT_RECIPES
        for part in recipe.parts
    )
)


def validate_recipe() -> None:
    if (
        len(NVFP4_WEIGHT_RECIPES),
        len(INPUT_DIVISOR_RECIPES),
        len(WEIGHT_DIVISOR_GROUPS),
        len(NVFP4_SOURCES),
        len(BF16_COMPARISONS),
    ) != (247, 247, 122, 379, 117):
        raise ValueError("NVFP4 source recipe is incomplete")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    ):
        raise ValueError("NVFP4 weight recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("input-divisor recipe order does not match inventory")
    for recipe in NVFP4_WEIGHT_RECIPES:
        rows = sum(part.output_rows for part in recipe.parts)
        if (rows, recipe.parts[0].source.shape[1]) != recipe.shape:
            raise ValueError(f"{recipe.object_name}: invalid fused row geometry")
        if any(part.source.shape[1] != recipe.shape[1] for part in recipe.parts):
            raise ValueError(f"{recipe.object_name}: incompatible source K")
    bound_weights = tuple(
        name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names
    )
    if (
        len(bound_weights) != 247
        or len(set(bound_weights)) != 247
        or set(bound_weights) != set(NVFP4_WEIGHTS_BY_NAME)
    ):
        raise ValueError("input-divisor sites do not cover NVFP4 parents exactly once")


def _expected_source_names() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in NVFP4_SOURCES:
        n, k = source.shape
        fields = {
            source.field("weight_packed"): ((n, k // 2), "U8"),
            source.field("weight_scale"): ((n, k // 16), "F8_E4M3"),
            source.field("weight_global_scale"): ((1,), "F32"),
            source.field("input_global_scale"): ((1,), "F32"),
        }
        result.update(fields)
    for comparison in BF16_COMPARISONS:
        result[comparison.source.field("weight")] = (
            comparison.source.shape,
            "BF16",
        )
    return result


SOURCE_REQUIREMENTS = _expected_source_names()


def preflight_metadata(reader: ShardReader) -> dict[str, int]:
    expected_names = set(SOURCE_REQUIREMENTS)
    missing = expected_names.difference(reader.names)
    if missing:
        raise ValueError(f"NVFP4 source is missing {sorted(missing)[0]}")

    quant_prefixes = {source.name for source in NVFP4_SOURCES}
    bf16_prefixes = {item.source.name for item in BF16_COMPARISONS}
    forbidden: list[str] = []
    for prefix in quant_prefixes:
        if reader.has(prefix + ".weight"):
            forbidden.append(prefix + ".weight")
    for prefix in bf16_prefixes:
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            name = f"{prefix}.{suffix}"
            if reader.has(name):
                forbidden.append(name)
    if forbidden:
        raise ValueError(f"NVFP4 source allocation is not closed: {forbidden[0]}")

    metadata = reader.metadata(SOURCE_REQUIREMENTS)
    dtype_counts: dict[str, int] = {}
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        actual = metadata[name]
        if actual.shape != shape or actual.dtype != dtype:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return dtype_counts


def _word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: divisor must be FP32[1]")
    raw = tensor.detach().contiguous().cpu().view(torch.int32)
    word = int(raw.item()) & 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return word


def _same_divisor(
    reader: ShardReader,
    sources: Iterable[Nvfp4Source],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _word(reader.get(source.field(suffix)), source.field(suffix))
        for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(
            f"{items[0].name}: fused {suffix} words do not match"
        )
    return words[0]


def validate_nvfp4_words(reader: ShardReader) -> None:
    for source in NVFP4_SOURCES:
        scales = reader.get(source.field("weight_scale")).view(torch.uint8)
        invalid = ((scales & 0x80) != 0) | (scales == 0x7F)
        if bool(invalid.any()):
            raise ValueError(
                f"{source.field('weight_scale')}: invalid E4M3FN scale word"
            )
        _word(
            reader.get(source.field("weight_global_scale")),
            source.field("weight_global_scale"),
        )
        _word(
            reader.get(source.field("input_global_scale")),
            source.field("input_global_scale"),
        )

    for group in WEIGHT_DIVISOR_GROUPS:
        _same_divisor(reader, group, "weight_global_scale")
    for recipe in INPUT_DIVISOR_RECIPES:
        _same_divisor(reader, recipe.sources, "input_global_scale")


def compare_bf16_sources(
    base_reader: ShardReader,
    nvfp4_reader: ShardReader,
) -> None:
    for comparison in BF16_COMPARISONS:
        name = comparison.source.field("weight")
        base = base_reader.get(name)
        selected = nvfp4_reader.get(name)
        if (
            base.dtype != torch.bfloat16
            or selected.dtype != torch.bfloat16
            or tuple(base.shape) != comparison.source.shape
            or tuple(selected.shape) != comparison.source.shape
            or not torch.equal(base.view(torch.int16), selected.view(torch.int16))
        ):
            raise ValueError(f"{name}: BF16 source words differ")


def _select_rows(tensor: torch.Tensor, part: Nvfp4Part) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows)
        for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe,
    reader: ShardReader,
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for part in recipe.parts:
        packed_parts.append(
            _select_rows(reader.get(part.source.field("weight_packed")), part)
        )
        scale_parts.append(
            _select_rows(
                reader.get(part.source.field("weight_scale")).view(torch.uint8),
                part,
            )
        )
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    word = _same_divisor(
        reader, recipe.divisor_sources, "weight_global_scale"
    )
    return packed, scales, struct.pack("<I", word)


def materialize_input_divisor(
    recipe: InputDivisorRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    word = _same_divisor(reader, recipe.sources, "input_global_scale")
    return torch.frombuffer(
        bytearray(struct.pack("<I", word)), dtype=torch.float32
    ).reshape(())


validate_recipe()


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "BF16_COMPARISONS",
    "INPUT_DIVISOR_RECIPES",
    "INPUT_DIVISORS_BY_NAME",
    "NVFP4_REPOSITORY",
    "NVFP4_REVISION",
    "NVFP4_SOURCES",
    "NVFP4_WEIGHT_RECIPES",
    "NVFP4_WEIGHTS_BY_NAME",
    "SOURCE_REQUIREMENTS",
    "WEIGHT_DIVISOR_GROUPS",
    "compare_bf16_sources",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "preflight_metadata",
    "validate_nvfp4_words",
    "validate_recipe",
]
