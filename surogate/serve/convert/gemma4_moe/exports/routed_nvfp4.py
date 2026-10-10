"""Gemma 4 mixture routed experts in NVFP4, copied from a ModelOpt or compressed-tensors export.

What an NVFP4 export of the 26B-A4B quantises is its **routed experts** -- NVIDIA's
`nvidia/Gemma-4-26B-A4B-NVFP4` leaves the attention, the dense feed-forward beside the experts,
the router, the head and the vision tower in BF16 (its `ignore` list names every one of them on
every layer) -- and those are 91 % of a layer's bytes and every one of its grouped FLOPs. So this
module owns exactly the two routed objects of each layer, and everything else keeps the recipe
the BF16 checkpoint would have used.

The experts are stored per expert and per projection, unfused, each with its own two scales:

    ModelOpt            ...experts.{e}.{gate,up,down}_proj.weight          (uint8 [n, k/2])
                                                      .weight_scale    (E4M3 [n, k/16])
                                                      .weight_scale_2  (FP32, multiplies)
                                                      .input_scale     (FP32, multiplies)
    compressed-tensors  ...experts.{e}.{proj}.weight_packed / weight_scale /
                                    weight_global_scale / input_global_scale  (both divide)

The artifact keeps the words verbatim and states the scales once, in the engine's direction:

    moe/routed_gate_up            NVFP4 [E * 2 * I, H], an expert's rows **[up; gate]**
    moe/routed_gate_up_scale      FP32 [2E]: 1 / weight divisor, up then gate
    moe/routed_gate_up_act_scale  FP32 [E]:  the activation divisor (6 * 448 / calibrated amax)
    moe/routed_gate_up_alpha      FP32 [E]:  1 / (act divisor * weight divisor)
    moe/routed_down (+ the same three, one entry per expert)

-- the contract `qwen3_5_moe`'s routed profile set, so the engine's routed-NVFP4 path (the
vendored TensorRT-LLM runner at width, our decode kernels at one token) serves both. [up; gate]
is the runner's order (`linear` from the first half, `gate` from the second).

The runner carries one weight divisor per expert for fc1, so an expert's gate and up must share
it. ModelOpt calibrates them separately and a few experts come out with two (layer 0 of
NVIDIA's export has two such experts). Those halves are **re-quantised** from the values their
words represent to the pair's common divisor -- the smaller one, which is the larger amax, so
neither half clips -- rather than refused. It is a second rounding of those experts only, and
the conversion report counts them. Their input divisors must agree (one input feeds both), and
differing ones take the smaller for the same reason.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from surogate.serve.artifact.layouts import encode_direct, encode_nvfp4
from surogate.serve.convert.common import nvfp4
from surogate.serve.convert.common.inventory import BLOCK_SCALE_LAYOUT, FP32, NVFP4, TensorSpec, tensor_spec
from surogate.serve.convert.common.safetensors import ShardReader

WEIGHTS_ID = "routed-nvfp4"

GATE_UP_SUFFIX = "/moe/routed_gate_up"
DOWN_SUFFIX = "/moe/routed_down"
SCALE_SUFFIX = "_scale"
ACT_SCALE_SUFFIX = "_act_scale"
ALPHA_SUFFIX = "_alpha"
_SECONDARY = (SCALE_SUFFIX, ACT_SCALE_SUFFIX, ALPHA_SUFFIX)

#: How each exporter spells an NVFP4 Linear's four tensors, and whether its two global scales
#: multiply (ModelOpt) or divide (compressed-tensors, and the engine).
@dataclass(frozen=True)
class Convention:
    name: str
    codes: str
    block_scales: str
    weight_global: str
    input_global: str
    multiplies: bool


MODELOPT = Convention("modelopt", "weight", "weight_scale", "weight_scale_2", "input_scale", True)
COMPRESSED_TENSORS = Convention("compressed-tensors", "weight_packed", "weight_scale",
                                "weight_global_scale", "input_global_scale", False)


def convention_of(config: Mapping[str, object]) -> Convention | None:
    """The export convention `config.json` declares, or None for an unquantised checkpoint.

    The declaration is the authority on the *direction* of the scales -- the two exporters name
    the same number with reciprocal meanings -- so a quantised checkpoint whose declaration this
    cannot read is refused rather than guessed at.
    """
    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        return None
    method = str(quant.get("quant_method", ""))
    if method == "modelopt":
        algo = str(quant.get("quant_algo", ""))
        if algo != "NVFP4":
            raise ValueError(f"ModelOpt export is quant_algo {algo!r}; only NVFP4 is served for Gemma 4")
        _require_group_16(quant)
        return MODELOPT
    if method == "compressed-tensors":
        formats = set()
        for group in (quant.get("config_groups") or {}).values():
            if isinstance(group, Mapping) and "format" in group:
                formats.add(group["format"])
        if "format" in quant:
            formats.add(quant["format"])
        if formats != {"nvfp4-pack-quantized"}:
            raise ValueError(f"compressed-tensors export has formats {sorted(formats)}; "
                             "only nvfp4-pack-quantized is served for Gemma 4")
        _require_group_16(quant)
        return COMPRESSED_TENSORS
    raise ValueError(f"quantization_config.quant_method {method!r} is not served for Gemma 4")


def _require_group_16(quant: Mapping[str, object]) -> None:
    groups = quant.get("config_groups")
    if isinstance(groups, Mapping):
        for name, group in groups.items():
            for side in ("weights", "input_activations"):
                spec = group.get(side) if isinstance(group, Mapping) else None
                if isinstance(spec, Mapping) and spec.get("group_size") not in (None, 16):
                    raise ValueError(f"quantization group {name} {side}: NVFP4 needs groups of 16")
    if quant.get("group_size") not in (None, 16):
        raise ValueError("NVFP4 needs groups of 16")


def is_routed_object(name: str) -> bool:
    """The objects this module owns rather than the base recipe."""
    if not name.startswith("text/layers/"):
        return False
    return any(name.endswith(suffix) or any(name.endswith(suffix + s) for s in _SECONDARY)
               for suffix in (GATE_UP_SUFFIX, DOWN_SUFFIX))


def layer_of(name: str) -> int:
    return int(name.split("/")[2])


def tensor_specs(specs: Sequence[TensorSpec], experts: int) -> tuple[TensorSpec, ...]:
    """The base specs with each layer's two routed objects in NVFP4 and their scale arrays."""
    out: list[TensorSpec] = []
    for spec in specs:
        gate_up = spec.name.startswith("text/layers/") and spec.name.endswith(GATE_UP_SUFFIX)
        down = spec.name.startswith("text/layers/") and spec.name.endswith(DOWN_SUFFIX)
        if not (gate_up or down):
            out.append(spec)
            continue
        out.append(TensorSpec(name=spec.name, shape=tuple(spec.shape), format=NVFP4,
                              layout=BLOCK_SCALE_LAYOUT))
        out.append(tensor_spec(spec.name + SCALE_SUFFIX, ((2 if gate_up else 1) * experts,), FP32))
        out.append(tensor_spec(spec.name + ACT_SCALE_SUFFIX, (experts,), FP32))
        out.append(tensor_spec(spec.name + ALPHA_SUFFIX, (experts,), FP32))
    return tuple(out)


@dataclass
class Geometry:
    layers: int
    experts: int
    hidden: int
    intermediate: int


def geometry_of(g) -> Geometry:
    """The four numbers this module needs, from the target's resolved geometry."""
    geometry = Geometry(g.layers, g.experts, g.hidden, g.expert_intermediate)
    if geometry.hidden % 64 or geometry.intermediate % 64:
        raise ValueError("routed NVFP4 needs hidden and expert widths that are multiples of 64")
    if (2 * geometry.intermediate) % 128 or geometry.hidden % 128:
        raise ValueError("routed NVFP4 needs every expert to be a whole number of 128-row tiles")
    return geometry


def _module(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.experts.{expert}.{projection}"


def source_names(g: Geometry, convention: Convention, layer: int) -> Iterator[str]:
    for expert in range(g.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = _module(layer, expert, projection) + "."
            for suffix in (convention.codes, convention.block_scales, convention.weight_global,
                           convention.input_global):
                yield base + suffix


def _scalar(reader: ShardReader, name: str) -> float:
    value = reader.get(name)
    if value.numel() != 1:
        raise ValueError(f"{name}: expected one scale, found {value.numel()}")
    result = float(value.to(torch.float32).reshape(()))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} is {result}, expected finite positive")
    return result


def preflight_source(reader: ShardReader, g: Geometry, convention: Convention) -> None:
    """Every expert's four tensors at the signature the geometry implies, before any is read."""
    for layer in range(g.layers):
        metadata = reader.metadata(tuple(source_names(g, convention, layer)))
        for expert in range(g.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                n, k = ((g.hidden, g.intermediate) if projection == "down_proj"
                        else (g.intermediate, g.hidden))
                base = _module(layer, expert, projection) + "."
                for suffix, shape, dtype in ((convention.codes, (n, k // 2), "U8"),
                                             (convention.block_scales, (n, k // 16), "F8_E4M3")):
                    found = metadata[base + suffix]
                    if tuple(found.shape) != shape or found.dtype != dtype:
                        raise ValueError(f"{base + suffix}: stored as {found.dtype} {tuple(found.shape)}, "
                                         f"expected {dtype} {shape}")
                for suffix in (convention.weight_global, convention.input_global):
                    found = metadata[base + suffix]
                    if math.prod(found.shape) != 1 or found.dtype not in ("BF16", "F16", "F32"):
                        raise ValueError(f"{base + suffix}: expected one floating-point scale")


@dataclass
class Projection:
    """One expert projection in the engine's direction: both globals are divisors."""
    codes: torch.Tensor
    scales: torch.Tensor  # uint8 E4M3FN words, natural [n, k/16]
    weight_divisor: float
    input_divisor: float


def read_projection(reader: ShardReader, convention: Convention, layer: int, expert: int,
                    projection: str) -> Projection:
    base = _module(layer, expert, projection) + "."
    codes = reader.get(base + convention.codes)
    scales = reader.get(base + convention.block_scales)
    if codes.dtype != torch.uint8:
        raise TypeError(f"{base}{convention.codes} is {codes.dtype}, expected uint8")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{base}{convention.block_scales} is {scales.dtype}, expected float8_e4m3fn")
    weight = _scalar(reader, base + convention.weight_global)
    activation = _scalar(reader, base + convention.input_global)
    if convention.multiplies:
        weight, activation = 1.0 / weight, 1.0 / activation
    return Projection(codes, scales.view(torch.uint8), weight, activation)


def requantize(projection: Projection, divisor: float) -> Projection:
    """The same values at another weight divisor: decode the words, quantise them again."""
    values = nvfp4.dequantize(projection.codes, projection.scales, projection.weight_divisor)
    codes, scales = nvfp4.quantize(values, divisor)
    return Projection(codes, scales, divisor, projection.input_divisor)


@dataclass
class LayerStats:
    requantised_experts: list[int] = field(default_factory=list)
    input_divisor_mismatches: list[int] = field(default_factory=list)


def build_gate_up(reader: ShardReader, g: Geometry, convention: Convention, layer: int,
                  stats: LayerStats) -> tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = g.experts * 2 * g.intermediate
    codes = torch.empty((rows, g.hidden // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.hidden // 16), dtype=torch.uint8)
    second = torch.empty(2 * g.experts, dtype=torch.float32)
    act = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        up = read_projection(reader, convention, layer, expert, "up_proj")
        gate = read_projection(reader, convention, layer, expert, "gate_proj")
        if up.weight_divisor != gate.weight_divisor:
            common = min(up.weight_divisor, gate.weight_divisor)
            up = up if up.weight_divisor == common else requantize(up, common)
            gate = gate if gate.weight_divisor == common else requantize(gate, common)
            stats.requantised_experts.append(expert)
        input_divisor = up.input_divisor
        if up.input_divisor != gate.input_divisor:
            input_divisor = min(up.input_divisor, gate.input_divisor)
            stats.input_divisor_mismatches.append(expert)
        for half, part in enumerate((up, gate)):
            if tuple(part.codes.shape) != (g.intermediate, g.hidden // 2):
                raise ValueError(f"layer {layer} expert {expert}: gate/up codes are "
                                 f"{tuple(part.codes.shape)}")
            begin = (expert * 2 + half) * g.intermediate
            codes[begin:begin + g.intermediate] = part.codes
            scales[begin:begin + g.intermediate] = part.scales
            second[expert * 2 + half] = 1.0 / part.weight_divisor
        act[expert] = input_divisor
        alpha[expert] = 1.0 / (input_divisor * up.weight_divisor)
    payload = encode_nvfp4(codes, scales, struct.pack("<f", 1.0), (rows, g.hidden))
    return payload, second, act, alpha


def build_down(reader: ShardReader, g: Geometry, convention: Convention,
               layer: int) -> tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = g.experts * g.hidden
    codes = torch.empty((rows, g.intermediate // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.intermediate // 16), dtype=torch.uint8)
    second = torch.empty(g.experts, dtype=torch.float32)
    act = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        down = read_projection(reader, convention, layer, expert, "down_proj")
        if tuple(down.codes.shape) != (g.hidden, g.intermediate // 2):
            raise ValueError(f"layer {layer} expert {expert}: down codes are {tuple(down.codes.shape)}")
        begin = expert * g.hidden
        codes[begin:begin + g.hidden] = down.codes
        scales[begin:begin + g.hidden] = down.scales
        second[expert] = 1.0 / down.weight_divisor
        act[expert] = down.input_divisor
        alpha[expert] = 1.0 / (down.input_divisor * down.weight_divisor)
    payload = encode_nvfp4(codes, scales, struct.pack("<f", 1.0), (rows, g.intermediate))
    return payload, second, act, alpha


# ---- dense matrices -------------------------------------------------------------------------
#
# An export that quantises more than the experts -- the attention projections, the dense
# feed-forward beside the experts -- states it the same way, one module at a time. Those become
# ordinary NVFP4 linears in the artifact: the words verbatim, the weight divisor in the payload's
# tail word and the activation divisor in `<name>/input_scale_divisor`, which is what
# `artifact::bind_linear` reads for any target. The dense gate and up are the one exception: at
# 2,112 rows each they are not whole 128-row scale tiles, so they are stored as the single
# `mlp/gate_up` matrix `[gate; up]` the engine binds in their place (one divisor for the pair,
# re-quantising a half whose divisor differs, as for the experts).

INPUT_DIVISOR_SUFFIX = "/input_scale_divisor"
DENSE_GATE_UP = "mlp/gate_up"


@dataclass(frozen=True)
class DenseSource:
    """One NVFP4 object built from one or two quantised modules of the checkpoint."""
    name: str
    modules: tuple[str, ...]
    rows: int
    columns: int


def _plain_source(recipe) -> str | None:
    """The checkpoint tensor a recipe copies verbatim, if that is all it does."""
    from surogate.serve.convert.common.recipe import SourceTensor

    expression = recipe.expression
    if isinstance(expression, SourceTensor) and expression.name.endswith(".weight"):
        return expression.name
    return None


#: The per-layer projections a dense Gemma target serves from NVFP4 codes. Anything else an
#: export quantises -- the E-series' per-layer input gate and projections, a few hundred
#: kilobytes a layer -- is read through `DequantizingReader` into the text format instead.
DENSE_SUFFIXES = ("/attention/query", "/attention/key", "/attention/value", "/attention/output",
                  "/mlp/gate", "/mlp/up", "/mlp/down")


def dense_plan(specs: Sequence[TensorSpec], recipes: Mapping[str, object], reader: ShardReader,
               convention: Convention, *, suffixes: Sequence[str] | None = None,
               ) -> tuple[tuple[TensorSpec, ...], dict[str, DenseSource]]:
    """The text specs with every quantised dense module in NVFP4, and what builds each.

    A module is quantised when the checkpoint holds its global weight scale beside it; the
    declaration's `ignore` list is a claim, the scale is the fact (`quant_scope`). `suffixes`
    limits the objects that keep their codes; None keeps every plain copy's.
    """
    out: list[TensorSpec] = []
    sources: dict[str, DenseSource] = {}
    by_name = {spec.name: spec for spec in specs}

    def quantised(name: str) -> str | None:
        recipe = recipes.get(name)
        source = _plain_source(recipe) if recipe is not None else None
        if source is None or not name.startswith("text/layers/") or is_routed_object(name):
            return None
        if suffixes is not None and not name.endswith(tuple(suffixes)):
            return None
        module = source[: -len(".weight")]
        return module if reader.has(module + "." + convention.weight_global) else None

    for spec in specs:
        module = quantised(spec.name)
        if module is None:
            out.append(spec)
            continue
        prefix, _, leaf = spec.name.rpartition("/")
        if spec.name.endswith("/mlp/gate") or spec.name.endswith("/mlp/up"):
            layer = spec.name[: -len("/mlp/gate") if spec.name.endswith("/mlp/gate") else -len("/mlp/up")]
            gate, up = layer + "/mlp/gate", layer + "/mlp/up"
            gate_module, up_module = quantised(gate), quantised(up)
            if gate_module is None or up_module is None:
                raise ValueError(f"{layer}: the export quantises one of the dense gate/up pair and not the "
                                 "other; the engine serves them as one NVFP4 matrix or two plain ones")
            fused = layer + "/" + DENSE_GATE_UP
            if fused not in sources:
                rows = by_name[gate].shape[0] + by_name[up].shape[0]
                sources[fused] = DenseSource(fused, (gate_module, up_module), rows, spec.shape[1])
                out.append(TensorSpec(fused, (rows, spec.shape[1]), NVFP4, BLOCK_SCALE_LAYOUT))
                out.append(tensor_spec(fused + INPUT_DIVISOR_SUFFIX, (), FP32))
            continue
        if spec.shape[0] % 128 or spec.shape[1] % 64:
            raise ValueError(f"{spec.name}: {spec.shape} is not an NVFP4 shape the engine serves "
                             "(rows a multiple of 128, columns of 64)")
        sources[spec.name] = DenseSource(spec.name, (module,), spec.shape[0], spec.shape[1])
        out.append(TensorSpec(spec.name, tuple(spec.shape), NVFP4, BLOCK_SCALE_LAYOUT))
        out.append(tensor_spec(spec.name + INPUT_DIVISOR_SUFFIX, (), FP32))
    return tuple(out), sources


def owns_dense(name: str, sources: Mapping[str, DenseSource]) -> bool:
    return name in sources or (name.endswith(INPUT_DIVISOR_SUFFIX)
                               and name[: -len(INPUT_DIVISOR_SUFFIX)] in sources)


def _dense_module(reader: ShardReader, convention: Convention, module: str) -> Projection:
    codes = reader.get(module + "." + convention.codes)
    scales = reader.get(module + "." + convention.block_scales)
    if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{module}: expected uint8 codes and E4M3FN block scales")
    weight = _scalar(reader, module + "." + convention.weight_global)
    activation = _scalar(reader, module + "." + convention.input_global)
    if convention.multiplies:
        weight, activation = 1.0 / weight, 1.0 / activation
    return Projection(codes, scales.view(torch.uint8), weight, activation)


def dense_payload(name: str, sources: Mapping[str, DenseSource], reader: ShardReader,
                  convention: Convention) -> bytes:
    """The payload of a dense NVFP4 object or of its activation divisor."""
    divisor_object = name.endswith(INPUT_DIVISOR_SUFFIX)
    source = sources[name[: -len(INPUT_DIVISOR_SUFFIX)] if divisor_object else name]
    parts = [_dense_module(reader, convention, module) for module in source.modules]
    input_divisor = min(part.input_divisor for part in parts)
    if divisor_object:
        return struct.pack("<f", input_divisor)
    weight_divisor = min(part.weight_divisor for part in parts)
    parts = [part if part.weight_divisor == weight_divisor else requantize(part, weight_divisor)
             for part in parts]
    codes = torch.cat([part.codes for part in parts], dim=0)
    scales = torch.cat([part.scales for part in parts], dim=0)
    if tuple(codes.shape) != (source.rows, source.columns // 2):
        raise ValueError(f"{name}: modules stack to {tuple(codes.shape)}, expected "
                         f"{(source.rows, source.columns // 2)}")
    return encode_nvfp4(codes, scales, struct.pack("<f", weight_divisor), (source.rows, source.columns))


class DequantizingReader:
    """The checkpoint as the BF16 recipes read it: an NVFP4 module's `.weight` is the values its
    words represent rather than its packed codes.

    For the recipes that read a quantised matrix without copying it -- the E-series' stacked
    per-layer model projection, which the artifact cuts one slice per layer, and any module
    `dense_plan` leaves to the text format. A ModelOpt export stores the codes under the very
    name such a recipe asks for, so without this the recipe would read uint8 words as weights;
    a compressed-tensors export has no `.weight` at all. Every other name passes through, and the
    dense payloads read the plain reader, not this one.
    """

    def __init__(self, reader: ShardReader, convention: Convention) -> None:
        self._reader = reader
        self._convention = convention

    def _module(self, name: str) -> str | None:
        if not name.endswith(".weight"):
            return None
        module = name[: -len(".weight")]
        return module if self._reader.has(module + "." + self._convention.weight_global) else None

    def has(self, name: str) -> bool:
        return self._module(name) is not None or self._reader.has(name)

    def get(self, name: str) -> torch.Tensor:
        module = self._module(name)
        if module is None:
            return self._reader.get(name)
        part = _dense_module(self._reader, self._convention, module)
        return nvfp4.dequantize(part.codes, part.scales, part.weight_divisor).to(torch.bfloat16)

    def metadata(self, names):
        from surogate.serve.convert.common.safetensors import TensorMetadata

        names = list(names)
        modules = {name: self._module(name) for name in names}
        result = self._reader.metadata([name for name in names if modules[name] is None])
        quantised = {name: module for name, module in modules.items() if module is not None}
        codes = self._reader.metadata([module + "." + self._convention.codes
                                       for module in quantised.values()])
        for name, module in quantised.items():
            found = codes[module + "." + self._convention.codes]
            result[name] = TensorMetadata(name=name, shard=found.shard,
                                          shape=(found.shape[0], 2 * found.shape[1]), dtype="BF16")
        return result

    def __getattr__(self, attribute):
        return getattr(self._reader, attribute)


class LayerCache:
    """Builds a layer's eight objects in one pass and hands them out in plan order."""

    def __init__(self, reader: ShardReader, geometry: Geometry, convention: Convention) -> None:
        self._reader = reader
        self._geometry = geometry
        self._convention = convention
        self._layer: int | None = None
        self._objects: dict[str, object] = {}
        self.stats: dict[int, LayerStats] = {}

    def payload_for(self, name: str) -> bytes:
        layer = layer_of(name)
        if layer != self._layer:
            stats = self.stats.setdefault(layer, LayerStats())
            gate_up = build_gate_up(self._reader, self._geometry, self._convention, layer, stats)
            down = build_down(self._reader, self._geometry, self._convention, layer)
            prefix = f"text/layers/{layer}"
            self._objects = {}
            for suffix, built in ((GATE_UP_SUFFIX, gate_up), (DOWN_SUFFIX, down)):
                payload, second, act, alpha = built
                self._objects[prefix + suffix] = payload
                self._objects[prefix + suffix + SCALE_SUFFIX] = second
                self._objects[prefix + suffix + ACT_SCALE_SUFFIX] = act
                self._objects[prefix + suffix + ALPHA_SUFFIX] = alpha
            self._layer = layer
        value = self._objects.pop(name)
        return value if isinstance(value, bytes) else encode_direct(value, FP32)

    def summary(self) -> dict[str, object]:
        return {
            "requantised_gate_up_experts": {
                str(layer): stats.requantised_experts
                for layer, stats in sorted(self.stats.items()) if stats.requantised_experts},
            "input_divisor_mismatches": {
                str(layer): stats.input_divisor_mismatches
                for layer, stats in sorted(self.stats.items()) if stats.input_divisor_mismatches},
        }


__all__ = [
    "COMPRESSED_TENSORS",
    "DENSE_GATE_UP",
    "DENSE_SUFFIXES",
    "DequantizingReader",
    "DenseSource",
    "INPUT_DIVISOR_SUFFIX",
    "dense_payload",
    "dense_plan",
    "owns_dense",
    "Convention",
    "Geometry",
    "LayerCache",
    "MODELOPT",
    "WEIGHTS_ID",
    "convention_of",
    "geometry_of",
    "is_routed_object",
    "preflight_source",
    "read_projection",
    "requantize",
    "tensor_specs",
]
