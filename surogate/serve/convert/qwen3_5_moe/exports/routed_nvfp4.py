"""Preserve routed NVFP4 words and per-expert calibration from the checkpoint."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from surogate.serve.artifact.layouts import encode_nvfp4
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common.inventory import TensorSpec, tensor_spec

from .. import inventory


NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
WEIGHTS_ID = "routed-nvfp4"

GATE_UP_SUFFIX = "/moe/routed_gate_up"
DOWN_SUFFIX = "/moe/routed_down"
SCALE_SUFFIX = "_scale"
ACT_SCALE_SUFFIX = "_act_scale"
ALPHA_SUFFIX = "_alpha"

# The block scale plane is one e4m3 byte per 16 values; the code plane two values per byte.
_BLOCK = 16


def _nvfp4_spec(name: str, shape: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(
        name=name, shape=shape, format=NVFP4, layout=BLOCK_SCALE_LAYOUT
    )


def tensor_specs(g: inventory.Geometry, *, dflash=None) -> tuple[TensorSpec, ...]:
    """The target's tensor specs with the text routed experts in NVFP4.

    Every other object — including the MTP block's own W8 experts, which this
    profile does not touch — keeps the format ``inventory`` gives it.
    """

    specs: list[TensorSpec] = []
    for spec in inventory.build_tensor_specs(g, dflash=dflash):
        if spec.name.startswith("text/layers/") and spec.name.endswith(GATE_UP_SUFFIX):
            specs.append(_nvfp4_spec(spec.name, spec.shape))
            specs.append(
                tensor_spec(spec.name + SCALE_SUFFIX, (2 * g.experts,), inventory.FP32)
            )
            specs.append(
                tensor_spec(spec.name + ACT_SCALE_SUFFIX, (g.experts,), inventory.FP32)
            )
            specs.append(
                tensor_spec(spec.name + ALPHA_SUFFIX, (g.experts,), inventory.FP32)
            )
        elif spec.name.startswith("text/layers/") and spec.name.endswith(DOWN_SUFFIX):
            specs.append(_nvfp4_spec(spec.name, spec.shape))
            specs.append(
                tensor_spec(spec.name + SCALE_SUFFIX, (g.experts,), inventory.FP32)
            )
            specs.append(
                tensor_spec(spec.name + ACT_SCALE_SUFFIX, (g.experts,), inventory.FP32)
            )
            specs.append(
                tensor_spec(spec.name + ALPHA_SUFFIX, (g.experts,), inventory.FP32)
            )
        else:
            specs.append(spec)
    return tuple(specs)


def is_routed_object(name: str) -> bool:
    """True for the objects this module owns rather than the base recipe."""

    if not name.startswith("text/layers/"):
        return False
    for suffix in (GATE_UP_SUFFIX, DOWN_SUFFIX):
        if name.endswith(suffix):
            return True
        for second in (SCALE_SUFFIX, ACT_SCALE_SUFFIX, ALPHA_SUFFIX):
            if name.endswith(suffix + second):
                return True
    return False


def layer_of(name: str) -> int:
    return int(name.split("/")[2])


def validate_config(config: Mapping[str, object], geometry: inventory.Geometry) -> dict[str, object]:
    """Refuse anything but a compressed-tensors NVFP4 checkpoint.

    ``quantization_config`` is the authority, not the tensor names: a ModelOpt
    export names its global scale ``weight_scale_2`` and multiplies where
    compressed-tensors divides, so a checkpoint that merely looks similar would
    mis-scale every expert.
    """

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping):
        raise ValueError("routed NVFP4 source has no quantization_config")
    method = quant.get("quant_method")
    if method != "compressed-tensors":
        raise ValueError(
            f"routed NVFP4 source is quant_method '{method}', expected "
            "'compressed-tensors' (a ModelOpt export inverts the global-scale "
            "convention and is not supported)"
        )
    groups = quant.get("config_groups")
    formats = set()
    if isinstance(groups, Mapping):
        for group in groups.values():
            if isinstance(group, Mapping) and "format" in group:
                formats.add(group["format"])
    if "format" in quant:
        formats.add(quant["format"])
    if formats != {"nvfp4-pack-quantized"}:
        raise ValueError(
            f"routed NVFP4 source has formats {sorted(formats)}, expected "
            "['nvfp4-pack-quantized']"
        )
    text = config.get("text_config", config)
    layers = text.get("num_hidden_layers")
    experts = text.get("num_experts")
    resolved = inventory.geometry_from_config(config, token_domain=geometry.token_domain)
    for name in ("layers", "hidden", "intermediate", "experts", "experts_per_token", "shared_intermediate",
                 "vocab", "layer_types", "head_dim", "query_heads", "kv_heads"):
        if getattr(resolved, name) != getattr(geometry, name):
            raise ValueError(f"routed NVFP4 source disagrees with the target checkpoint: {name}")
    if geometry.hidden % 16 or geometry.intermediate % 16:
        raise ValueError("routed NVFP4 requires hidden and expert widths divisible by 16")
    return {
        "quant_method": method,
        "format": sorted(formats)[0],
        "num_hidden_layers": layers,
        "num_experts": experts,
    }


def _positive(tensor: torch.Tensor, what: str) -> float:
    value = float(tensor.reshape(()).to(torch.float32))
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"{what} is {value}, expected finite positive")
    return value


def _projection(reader: ShardReader, layer: int, expert: int, projection: str):
    """One projection of one expert: codes, block scales, and the two global scales.

    ``weight_global_scale`` and ``input_global_scale`` both *divide* in the
    checkpoint's convention; the caller decides which reciprocal each object needs.
    """

    base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
    codes = reader.get(base + "weight_packed")
    scales = reader.get(base + "weight_scale")
    global_scale = reader.get(base + "weight_global_scale")
    input_global_scale = reader.get(base + "input_global_scale")
    if codes.dtype != torch.uint8:
        raise TypeError(f"{base}weight_packed is {codes.dtype}, expected uint8")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{base}weight_scale is {scales.dtype}, expected float8_e4m3fn")
    value = _positive(global_scale, base + "weight_global_scale")
    activation = _positive(input_global_scale, base + "input_global_scale")
    return codes, scales.view(torch.uint8), value, activation


def _require_shape(tensor: torch.Tensor, shape: tuple[int, int], what: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{what} has shape {tuple(tensor.shape)}, expected {shape}")


def source_names(layer: int, g: inventory.Geometry) -> Iterator[str]:
    """Every source tensor one layer's routed experts read, for preflight."""

    for expert in range(g.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
            yield base + "weight_packed"
            yield base + "weight_scale"
            yield base + "weight_global_scale"
            yield base + "input_global_scale"


def preflight_source(reader: ShardReader, g: inventory.Geometry) -> None:
    """Validate every expert signature and scalar calibration before opening the artifact."""
    for layer in range(g.layers):
        metadata = reader.metadata(tuple(source_names(layer, g)))
        for expert in range(g.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                n, k = (g.hidden, g.intermediate) if projection == "down_proj" else (g.intermediate, g.hidden)
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
                for suffix, shape, dtype in (("weight_packed", (n, k // 2), "U8"),
                                              ("weight_scale", (n, k // 16), "F8_E4M3")):
                    actual = metadata[base + suffix]
                    if actual.shape != shape or actual.dtype != dtype:
                        raise ValueError(f"{base + suffix}: stored signature disagrees with checkpoint dimensions")
                for suffix in ("weight_global_scale", "input_global_scale"):
                    actual = metadata[base + suffix]
                    if math.prod(actual.shape) != 1 or actual.dtype not in ("BF16", "F16", "F32"):
                        raise ValueError(f"{base + suffix}: expected one floating-point scale")
                    _positive(reader.get(base + suffix), base + suffix)


def build_gate_up(
    reader: ShardReader, layer: int, g: inventory.Geometry
) -> tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One layer's stacked up/gate experts: the payload and its three scale arrays.

    Rows run ``(expert * 2 + half) * intermediate + row`` with **up first, gate second** —
    the order TensorRT-LLM's fused MoE runner reads (``linear`` from the first half,
    ``gate`` from the second; vLLM performs the same swap in
    ``reorder_w1w3_to_w3w1`` before handing it that kernel). This is the opposite of
    the groupwise-int profile's ``[gate; up]``, so an artifact of this profile is
    served only through that runner.

    The runner carries one activation scale and one epilogue alpha per expert, so
    gate and up must agree on ``input_global_scale`` and ``weight_global_scale``;
    a source where they do not is refused rather than half-applied.
    """

    rows = g.experts * 2 * g.intermediate
    codes = torch.empty((rows, g.hidden // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.hidden // _BLOCK), dtype=torch.uint8)
    second = torch.empty(2 * g.experts, dtype=torch.float32)
    act_scale = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        halves: list[tuple[float, float]] = []
        for half, projection in enumerate(("up_proj", "gate_proj")):
            plane, scale_plane, global_scale, activation = _projection(
                reader, layer, expert, projection
            )
            _require_shape(plane, (g.intermediate, g.hidden // 2), f"L{layer} e{expert} {projection} codes")
            _require_shape(
                scale_plane, (g.intermediate, g.hidden // _BLOCK), f"L{layer} e{expert} {projection} scales"
            )
            begin = (expert * 2 + half) * g.intermediate
            codes[begin : begin + g.intermediate] = plane
            scales[begin : begin + g.intermediate] = scale_plane
            second[expert * 2 + half] = 1.0 / global_scale
            halves.append((global_scale, activation))
        (up_weight, up_act), (gate_weight, gate_act) = halves
        if up_weight != gate_weight or up_act != gate_act:
            raise ValueError(
                f"L{layer} e{expert}: gate and up disagree on their global scales "
                f"(weight {gate_weight} vs {up_weight}, input {gate_act} vs {up_act}); "
                "the fused runner carries one of each per expert"
            )
        act_scale[expert] = up_act
        alpha[expert] = 1.0 / (up_act * up_weight)
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0, dtype=torch.float32), (g.experts * 2 * g.intermediate, g.hidden))
    return payload, second, act_scale, alpha


def build_down(
    reader: ShardReader, layer: int, g: inventory.Geometry
) -> tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One layer's stacked down experts: the payload and its three scale arrays."""

    rows = g.experts * g.hidden
    codes = torch.empty((rows, g.intermediate // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.intermediate // _BLOCK), dtype=torch.uint8)
    second = torch.empty(g.experts, dtype=torch.float32)
    act_scale = torch.empty(g.experts, dtype=torch.float32)
    alpha = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        plane, scale_plane, global_scale, activation = _projection(
            reader, layer, expert, "down_proj"
        )
        _require_shape(plane, (g.hidden, g.intermediate // 2), f"L{layer} e{expert} down codes")
        _require_shape(
            scale_plane, (g.hidden, g.intermediate // _BLOCK), f"L{layer} e{expert} down scales"
        )
        begin = expert * g.hidden
        codes[begin : begin + g.hidden] = plane
        scales[begin : begin + g.hidden] = scale_plane
        second[expert] = 1.0 / global_scale
        act_scale[expert] = activation
        alpha[expert] = 1.0 / (activation * global_scale)
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0, dtype=torch.float32), (g.experts * g.hidden, g.intermediate))
    return payload, second, act_scale, alpha


class LayerCache:
    """Builds a layer's four objects once and hands them out in inventory order.

    The two weights and their two scale arrays come from the same pass over the
    checkpoint, but the writer asks for them one object at a time.
    """

    def __init__(self, reader: ShardReader, geometry: inventory.Geometry) -> None:
        self._reader = reader
        self._geometry = geometry
        self._layer: int | None = None
        self._objects: dict[str, object] = {}

    def payload_for(self, name: str, encode_direct) -> bytes:
        layer = layer_of(name)
        if layer != self._layer:
            gate_up, gate_up_scale, gate_up_act, gate_up_alpha = build_gate_up(
                self._reader, layer, self._geometry
            )
            down, down_scale, down_act, down_alpha = build_down(self._reader, layer, self._geometry)
            prefix = f"text/layers/{layer}"
            self._objects = {
                prefix + GATE_UP_SUFFIX: gate_up,
                prefix + GATE_UP_SUFFIX + SCALE_SUFFIX: gate_up_scale,
                prefix + GATE_UP_SUFFIX + ACT_SCALE_SUFFIX: gate_up_act,
                prefix + GATE_UP_SUFFIX + ALPHA_SUFFIX: gate_up_alpha,
                prefix + DOWN_SUFFIX: down,
                prefix + DOWN_SUFFIX + SCALE_SUFFIX: down_scale,
                prefix + DOWN_SUFFIX + ACT_SCALE_SUFFIX: down_act,
                prefix + DOWN_SUFFIX + ALPHA_SUFFIX: down_alpha,
            }
            self._layer = layer
        value = self._objects.pop(name)
        if isinstance(value, bytes):
            return value
        return encode_direct(value)
