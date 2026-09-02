"""One artifact object per HF Linear, in the format the export declares.

A compressed-tensors export decides per module what it quantized; nothing in
a converter may decide otherwise. This module applies that to the 35B-A3B's
text core: the inventory still says which objects exist and the recipes still
say where their rows come from, but the *format* of every Linear-derived
object comes from ``quant_schemes.resolve_checkpoint`` -- NVFP4 words copied
verbatim where the file packs the module, BF16 rows copied verbatim where it
does not -- and a fused parent whose constituents are separate Linears in the
file becomes one object per constituent, because each carries its own global
scale and there is no honest way to fuse two of those.

The recipes speak in logical tensors (``self_attn.q_proj.weight`` at
``[n, k]``); which stored tensors realise one -- ``weight`` or
``weight_packed`` + ``weight_scale`` + ``weight_global_scale`` -- is a
per-source lookup here, so the recipes did not change.

Where this stops for now: MTP, vision, the draft head and the DFlash family
keep their existing objects. The engine validates those even when their
features are off, so they move to this rule together with their bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np
import torch

from surogate.core.model.quant_schemes import (
    PACKED_WEIGHT,
    ResolvedQuantization,
    read_tensor_shapes,
    resolve_checkpoint,
)
from surogate.serve.tools.artifact.layouts import encode_direct, encode_nvfp4
from surogate.serve.tools.convert.common.inventory import (
    BF16,
    FP32,
    BLOCK_SCALE_LAYOUT,
    CONTIGUOUS_LAYOUT,
    NVFP4,
    TensorSpec,
)
from surogate.serve.tools.convert.common.recipe import TensorRecipe
from surogate.serve.tools.convert.common.row_algebra import evaluate_rows
from surogate.serve.tools.convert.common.safetensors import ShardReader

from . import inventory

WEIGHTS_ID = "compressed-tensors"

#: The fused parents the engine consumed as one weight, and the names their
#: constituents take when the file stores them as separate Linears. Order is
#: the parent's row order, which is the recipe's Concat order.
SEGMENT_NAMES: Mapping[str, tuple[str, ...]] = {
    "attention/query_key_gate_value": (
        "attention/query",
        "attention/key",
        "attention/gate",
        "attention/value",
    ),
    "gdn/query_key_value_z": ("gdn/query_key_value", "gdn/z"),
    "moe/shared_gate_up": ("moe/shared_gate", "moe/shared_up"),
}

INPUT_DIVISOR_SUFFIX = "/input_scale_divisor"


@dataclass(frozen=True)
class ObjectSource:
    """One artifact object as a run of rows from one logical source."""

    name: str
    source: str  # logical HF tensor, e.g. "...self_attn.q_proj.weight"
    rows: np.ndarray  # row ids into the source, in object row order
    k: int
    quantized: bool  # NVFP4 (weight_packed) if True, BF16 (weight) if not

    @property
    def n(self) -> int:
        return int(self.rows.size)


@dataclass(frozen=True)
class SourcePlan:
    specs: tuple[TensorSpec, ...]
    objects: Mapping[str, ObjectSource]
    resolved: ResolvedQuantization
    #: The inventory's original object names this plan took over, so a preflight
    #: knows which recipes it no longer needs to find sources for.
    covered: frozenset[str]


def _module_of(logical: str) -> str:
    assert logical.endswith(".weight"), logical
    return logical[: -len(".weight")]


class CompressedTensorsSource:
    """The text core of one compressed-tensors checkpoint, one object per Linear."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        resolved = resolve_checkpoint(self.model_dir)
        if resolved is None:
            raise ValueError(f"{self.model_dir}: not a compressed-tensors checkpoint")
        self.resolved = resolved
        stored = read_tensor_shapes(self.model_dir)
        # logical shapes under the recipe's names; a packed source is [n, k/2] on disk
        self.logical: dict[str, tuple[int, ...]] = {}
        for name, shape in stored.items():
            if name.endswith("." + PACKED_WEIGHT):
                self.logical[name[: -len(PACKED_WEIGHT)] + "weight"] = (shape[0], shape[1] * 2)
            elif name.endswith(".weight"):
                self.logical.setdefault(name, tuple(shape))

    def is_quantized(self, logical: str) -> bool:
        item = self.resolved.modules.get(_module_of(logical))
        return item is not None and item.scheme is not None and item.scheme.weights is not None

    # -- planning --------------------------------------------------------------

    def plan(
        self,
        specs: Sequence[TensorSpec],
        recipes_by_name: Mapping[str, TensorRecipe],
    ) -> SourcePlan:
        """Rewrite the text-core specs: formats from the export, parents split.

        Only objects whose recipe is a pure row program over 2-D sources are
        this module's business; a fused parent listed in SEGMENT_NAMES becomes
        one object per segment, any other Linear-derived object keeps its name
        and takes the file's format. Everything else passes through untouched.
        """

        out: list[TensorSpec] = []
        objects: dict[str, ObjectSource] = {}
        covered: set[str] = set()
        for spec in specs:
            recipe = recipes_by_name.get(spec.name)
            suffix = _text_core_suffix(spec.name)
            if recipe is None or suffix is None:
                out.append(spec)
                continue
            program = evaluate_rows(recipe.expression, self.logical, None)
            if program is None:
                out.append(spec)
                continue
            segments = program.segments()
            if suffix in SEGMENT_NAMES:
                names = SEGMENT_NAMES[suffix]
                if len(segments) != len(names):
                    raise ValueError(
                        f"{spec.name}: recipe has {len(segments)} segments, "
                        f"SEGMENT_NAMES lists {len(names)}"
                    )
                prefix = spec.name[: -len(suffix)]
                for name, (source, rows) in zip(names, segments):
                    self._emit(out, objects, prefix + name, source, rows, program.k)
                covered.add(spec.name)
            elif len(segments) == 1 and program.rows.size == spec.shape[0]:
                (source, rows), = segments
                self._emit(out, objects, spec.name, source, rows, program.k)
                covered.add(spec.name)
            else:
                out.append(spec)  # a multi-source BF16 parent the engine reads fused
        return SourcePlan(tuple(out), objects, self.resolved, frozenset(covered))

    def _emit(
        self,
        out: list[TensorSpec],
        objects: dict[str, ObjectSource],
        name: str,
        source: str,
        rows: np.ndarray,
        k: int,
    ) -> None:
        quantized = self.is_quantized(source)
        item = ObjectSource(name, source, rows, k, quantized)
        objects[name] = item
        if quantized:
            out.append(TensorSpec(name, (item.n, k), NVFP4, BLOCK_SCALE_LAYOUT))
            out.append(TensorSpec(name + INPUT_DIVISOR_SUFFIX, (), FP32, CONTIGUOUS_LAYOUT))
        else:
            out.append(TensorSpec(name, (item.n, k), BF16, CONTIGUOUS_LAYOUT))

    # -- payloads ----------------------------------------------------------------

    def payload_for(self, name: str, objects: Mapping[str, ObjectSource], reader: ShardReader) -> bytes:
        if name.endswith(INPUT_DIVISOR_SUFFIX):
            item = objects[name[: -len(INPUT_DIVISOR_SUFFIX)]]
            word = reader.get(_module_of(item.source) + ".input_global_scale")
            return _fp32_word(word, item.source + " input_global_scale")
        item = objects[name]
        module = _module_of(item.source)
        rows = torch.from_numpy(np.ascontiguousarray(item.rows))
        if not item.quantized:
            weight = reader.get(item.source)
            if weight.dtype != torch.bfloat16:
                raise TypeError(f"{item.source} is {weight.dtype}, expected bfloat16")
            return encode_direct(weight.index_select(0, rows).contiguous(), BF16)
        codes = reader.get(module + "." + PACKED_WEIGHT)
        scales = reader.get(module + ".weight_scale")
        if codes.dtype != torch.uint8:
            raise TypeError(f"{module}.{PACKED_WEIGHT} is {codes.dtype}, expected uint8")
        if scales.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{module}.weight_scale is {scales.dtype}, expected float8_e4m3fn")
        # Rows are self-contained in NVFP4 (codes and scales both per row), so a row
        # gather is exact; the global scale is one word per Linear, copied as stored.
        # compressed-tensors' global scale *divides*, which is the engine's convention.
        divisor = _fp32_word(reader.get(module + ".weight_global_scale"), module + " weight_global_scale")
        return encode_nvfp4(
            codes.index_select(0, rows).contiguous(),
            scales.view(torch.uint8).index_select(0, rows).contiguous(),
            divisor,
            (item.n, item.k),
        )


def _fp32_word(tensor: torch.Tensor, what: str) -> bytes:
    value = float(tensor.reshape(()).to(torch.float32))
    if not (value > 0.0) or value != value:
        raise ValueError(f"{what} is {value}, expected finite positive")
    return struct.pack("<f", value)


def _text_core_suffix(name: str) -> str | None:
    """The per-layer or top-level text-core suffix, or None for other families."""

    for layer in inventory.TEXT_LAYERS:
        prefix = f"text/layers/{layer}/"
        if name.startswith(prefix):
            return name[len(prefix):]
    if name in ("text/token_embedding", "text/output_head"):
        return name
    return None


__all__ = [
    "INPUT_DIVISOR_SUFFIX",
    "SEGMENT_NAMES",
    "WEIGHTS_ID",
    "CompressedTensorsSource",
    "ObjectSource",
    "SourcePlan",
]
