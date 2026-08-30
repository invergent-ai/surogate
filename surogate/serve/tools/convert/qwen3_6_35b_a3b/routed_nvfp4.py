"""Routed experts in NVFP4 for the Qwen3.6-35B-A3B target.

The 35B is the last model whose routed experts come from a groupwise-int export
(Q4 gate/up, Q5 or Q6 down), and the board's own reading is that on this hardware
the weight format is worth more than any scheduling lever.  This module builds the
routed half of a second artifact profile — ``routed-nvfp4`` — from an NVFP4
checkpoint, leaving every other object exactly as ``convert`` already writes it.

Two things make the routed case different from the dense NVFP4 export the 27B
uses:

* **The experts are stacked.**  The checkpoint stores one tensor per expert per
  projection; the artifact stores one matrix per layer with the experts as row
  blocks.  Codes concatenate verbatim (they are row-major nibble pairs either
  way), and the block scales are swizzled once over the stacked matrix — an
  expert is 1,024 rows of gate/up and 2,048 of down, both multiples of the
  layout's 128-row tile, so no expert straddles a tile.

* **The second level does not stack.**  NVFP4 pairs an e4m3 scale per 16 values
  with a global scale, and the checkpoint stores that global scale per expert and
  per projection.  Measured over one layer's 256 experts it takes 96-118 distinct
  values spanning 3.3-7.0x, so it cannot be folded into the single per-tensor
  divisor the NVFP4 payload carries, and re-deriving the e4m3 block scales around
  a shared global scale would re-round every block on a four-bit format.  It is
  therefore written as a separate FP32 object per layer and applied by the MoE
  kernels once per expert dot (``ops::SparseMoeWeights::routed_gate_up_scale``).

The checkpoint's global scale *divides* (compressed-tensors computes it as
``6 * 448 / amax``); the engine's arrays *multiply*, so this module writes the
reciprocal.  Only compressed-tensors ``nvfp4-pack-quantized`` is accepted — a
ModelOpt export spells the same numbers differently and inverts that convention,
so it is refused by name rather than mis-scaled silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from surogate.serve.tools.artifact.layouts import encode_nvfp4
from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.qwen3_6.common.inventory import TensorSpec, tensor_spec

from . import inventory


NVFP4 = "NVFP4"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-v1"
WEIGHTS_ID = "routed-nvfp4"

EXPERTS = 256
HIDDEN = 2048
MOE_INTERMEDIATE = 512

GATE_UP_SHAPE = (EXPERTS * 2 * MOE_INTERMEDIATE, HIDDEN)
DOWN_SHAPE = (EXPERTS * HIDDEN, MOE_INTERMEDIATE)

GATE_UP_SUFFIX = "/moe/routed_gate_up"
DOWN_SUFFIX = "/moe/routed_down"
SCALE_SUFFIX = "_scale"

# The block scale plane is one e4m3 byte per 16 values; the code plane two values per byte.
_BLOCK = 16


def _nvfp4_spec(name: str, shape: tuple[int, ...]) -> TensorSpec:
    return TensorSpec(
        name=name, shape=shape, format=NVFP4, layout=BLOCK_SCALE_LAYOUT
    )


def tensor_specs() -> tuple[TensorSpec, ...]:
    """The target's tensor specs with the text routed experts in NVFP4.

    Every other object — including the MTP block's own W8 experts, which this
    profile does not touch — keeps the format ``inventory`` gives it.
    """

    specs: list[TensorSpec] = []
    for spec in inventory.TENSOR_SPECS:
        if spec.name.startswith("text/layers/") and spec.name.endswith(GATE_UP_SUFFIX):
            specs.append(_nvfp4_spec(spec.name, spec.shape))
            specs.append(
                tensor_spec(spec.name + SCALE_SUFFIX, (2 * EXPERTS,), inventory.FP32)
            )
        elif spec.name.startswith("text/layers/") and spec.name.endswith(DOWN_SUFFIX):
            specs.append(_nvfp4_spec(spec.name, spec.shape))
            specs.append(
                tensor_spec(spec.name + SCALE_SUFFIX, (EXPERTS,), inventory.FP32)
            )
        else:
            specs.append(spec)
    return tuple(specs)


def is_routed_object(name: str) -> bool:
    """True for the objects this module owns rather than the base recipe."""

    if not name.startswith("text/layers/"):
        return False
    for suffix in (GATE_UP_SUFFIX, DOWN_SUFFIX):
        if name.endswith(suffix) or name.endswith(suffix + SCALE_SUFFIX):
            return True
    return False


def layer_of(name: str) -> int:
    return int(name.split("/")[2])


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
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
    if layers != len(inventory.TEXT_LAYERS) or experts != EXPERTS:
        raise ValueError(
            f"routed NVFP4 source has {layers} layers and {experts} experts, "
            f"expected {len(inventory.TEXT_LAYERS)} and {EXPERTS}"
        )
    return {
        "quant_method": method,
        "format": sorted(formats)[0],
        "num_hidden_layers": layers,
        "num_experts": experts,
    }


def _projection(reader: ShardReader, layer: int, expert: int, projection: str):
    base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
    codes = reader.get(base + "weight_packed")
    scales = reader.get(base + "weight_scale")
    global_scale = reader.get(base + "weight_global_scale")
    if codes.dtype != torch.uint8:
        raise TypeError(f"{base}weight_packed is {codes.dtype}, expected uint8")
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{base}weight_scale is {scales.dtype}, expected float8_e4m3fn")
    value = float(global_scale.reshape(()).to(torch.float32))
    if not (value > 0.0) or value != value:
        raise ValueError(f"{base}weight_global_scale is {value}, expected finite positive")
    return codes, scales.view(torch.uint8), value


def _require_shape(tensor: torch.Tensor, shape: tuple[int, int], what: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{what} has shape {tuple(tensor.shape)}, expected {shape}")


def source_names(layer: int) -> Iterator[str]:
    """Every source tensor one layer's routed experts read, for preflight."""

    for expert in range(EXPERTS):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
            yield base + "weight_packed"
            yield base + "weight_scale"
            yield base + "weight_global_scale"


def build_gate_up(reader: ShardReader, layer: int) -> tuple[bytes, torch.Tensor]:
    """One layer's stacked gate/up experts: the NVFP4 payload and its second level.

    Rows run ``expert * 1024 + half * 512 + row``, matching the contiguous reshape
    the groupwise recipe performs on the fused ``experts.gate_up_proj``, so the two
    profiles address an expert identically.
    """

    rows = EXPERTS * 2 * MOE_INTERMEDIATE
    codes = torch.empty((rows, HIDDEN // 2), dtype=torch.uint8)
    scales = torch.empty((rows, HIDDEN // _BLOCK), dtype=torch.uint8)
    second = torch.empty(2 * EXPERTS, dtype=torch.float32)
    for expert in range(EXPERTS):
        for half, projection in enumerate(("gate_proj", "up_proj")):
            plane, scale_plane, global_scale = _projection(reader, layer, expert, projection)
            _require_shape(plane, (MOE_INTERMEDIATE, HIDDEN // 2), f"L{layer} e{expert} {projection} codes")
            _require_shape(
                scale_plane, (MOE_INTERMEDIATE, HIDDEN // _BLOCK), f"L{layer} e{expert} {projection} scales"
            )
            begin = (expert * 2 + half) * MOE_INTERMEDIATE
            codes[begin : begin + MOE_INTERMEDIATE] = plane
            scales[begin : begin + MOE_INTERMEDIATE] = scale_plane
            second[expert * 2 + half] = 1.0 / global_scale
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0, dtype=torch.float32), GATE_UP_SHAPE)
    return payload, second


def build_down(reader: ShardReader, layer: int) -> tuple[bytes, torch.Tensor]:
    """One layer's stacked down experts: the NVFP4 payload and its second level."""

    rows = EXPERTS * HIDDEN
    codes = torch.empty((rows, MOE_INTERMEDIATE // 2), dtype=torch.uint8)
    scales = torch.empty((rows, MOE_INTERMEDIATE // _BLOCK), dtype=torch.uint8)
    second = torch.empty(EXPERTS, dtype=torch.float32)
    for expert in range(EXPERTS):
        plane, scale_plane, global_scale = _projection(reader, layer, expert, "down_proj")
        _require_shape(plane, (HIDDEN, MOE_INTERMEDIATE // 2), f"L{layer} e{expert} down codes")
        _require_shape(
            scale_plane, (HIDDEN, MOE_INTERMEDIATE // _BLOCK), f"L{layer} e{expert} down scales"
        )
        begin = expert * HIDDEN
        codes[begin : begin + HIDDEN] = plane
        scales[begin : begin + HIDDEN] = scale_plane
        second[expert] = 1.0 / global_scale
    payload = encode_nvfp4(codes, scales, torch.tensor(1.0, dtype=torch.float32), DOWN_SHAPE)
    return payload, second


class LayerCache:
    """Builds a layer's four objects once and hands them out in inventory order.

    The two weights and their two scale arrays come from the same pass over the
    checkpoint, but the writer asks for them one object at a time.
    """

    def __init__(self, reader: ShardReader) -> None:
        self._reader = reader
        self._layer: int | None = None
        self._objects: dict[str, object] = {}

    def payload_for(self, name: str, encode_direct) -> bytes:
        layer = layer_of(name)
        if layer != self._layer:
            gate_up, gate_up_scale = build_gate_up(self._reader, layer)
            down, down_scale = build_down(self._reader, layer)
            prefix = f"text/layers/{layer}"
            self._objects = {
                prefix + GATE_UP_SUFFIX: gate_up,
                prefix + GATE_UP_SUFFIX + SCALE_SUFFIX: gate_up_scale,
                prefix + DOWN_SUFFIX: down,
                prefix + DOWN_SUFFIX + SCALE_SUFFIX: down_scale,
            }
            self._layer = layer
        value = self._objects.pop(name)
        if isinstance(value, bytes):
            return value
        return encode_direct(value)
