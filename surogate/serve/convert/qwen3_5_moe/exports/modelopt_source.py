"""A ModelOpt mixed-precision export, one artifact object per HF Linear, in the format its
module's entry in ``quantized_layers`` declares.

ModelOpt states what it did per module: ``quantization_config.quantized_layers`` maps every
quantised module to its ``quant_algo``, and that map -- not the tensor names -- is the
authority. A ``MIXED_PRECISION`` export of this family (Ornith-1.5-122B-NVFP4, modelopt 0.45)
mixes two of them:

- ``W4A16_NVFP4``, weight-only NVFP4: ``weight`` holds the E2M1 codes two to a byte
  (U8 ``[n, k/2]``, the even column in the low nibble), ``weight_scale`` one E4M3 block scale
  per 16 values, ``weight_scale_2`` the FP32 second level. ModelOpt's second level
  **multiplies** -- ``value = e2m1 * block * weight_scale_2`` -- where compressed-tensors'
  ``weight_global_scale`` divides. There is no activation scale: the activations stay BF16.
  The routed experts, the shared expert and the output head.
- ``FP8``, static per-tensor: ``weight`` E4M3 codes, ``weight_scale`` one FP32 multiplier,
  ``input_scale`` the activation scale the export calibrated. Attention and the linear
  attention's in/out projections.

What the artifact holds, as this module writes it:

- A weight-only NVFP4 module is an NVFP4_A16 object: its codes and block scales exactly as
  stored, and the payload's trailing word the second level as a multiplier -- ModelOpt's
  ``weight_scale_2``, bit for bit -- for a dense module; the identity for a routed bank, whose
  second level is per expert and projection in ``routed_gate_up_scale`` / ``routed_down_scale``,
  as stored. There is no ``/input_scale_divisor`` and no routed activation scale or alpha: the
  format says the activations stay BF16. Nothing is requantised and no activation scale is
  invented.
- An FP8 module keeps its E4M3 codes as FP8_ROW_F32, its per-tensor multiplier repeated for
  every row -- the same values, exactly. ``input_scale`` is not carried: the engine's row-scaled
  FP8 kernels read BF16 activations, or quantise them per token per 128, never with one static
  scale for the tensor.
- A module the export left in BF16 is BF16 in the text core, and the rest (the MTP block, which
  the export ignores; the draft head, derived from the output head; the vision tower) takes the
  base inventory's format, exactly as for a BF16 checkpoint. The draft head's base recipe reads
  the output head, so it reads it through ``dequantizing_reader``: the values the stored words
  stand for.

The routed experts are stored per expert (``mlp.experts.{e}.{gate,up,down}_proj``); a layer's
bank is stacked **gate first, up second** per expert (the groupwise-int order, and the order
the weight-only kernels fold), and gate and up must share their second level, since one GEMM
over the stacked rows carries one per expert.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch

from surogate.core.model.quant_schemes import read_tensor_headers
from surogate.serve.artifact.layouts import encode_direct, encode_fp8_row_f32, encode_nvfp4_a16
from surogate.serve.convert.common.inventory import (
    BF16,
    BLOCK_SCALE_LAYOUT,
    CONTIGUOUS_LAYOUT,
    FP32,
    FP8_ROW_F32,
    NVFP4_A16,
    ROW_SCALE_F32_LAYOUT,
    TensorSpec,
    tensor_spec,
)
from surogate.serve.convert.common import nvfp4 as nvfp4_words
from surogate.serve.convert.common.recipe import TensorRecipe, expression_sources
from surogate.serve.convert.common.row_algebra import evaluate_rows
from surogate.serve.convert.common.safetensors import ShardReader

from .. import inventory
from .compressed_tensors_source import (
    NESTED_TEXT_PREFIX,
    SEGMENT_NAMES,
    _fp8_words,
    _text_core_suffix,
)


WEIGHTS_ID = "modelopt"

#: The per-module algorithms this reader serves, by ModelOpt's names.
W4A16_NVFP4 = "W4A16_NVFP4"
FP8 = "FP8"
SERVED_ALGOS = frozenset({W4A16_NVFP4, FP8})

GATE_UP_SUFFIX = "/moe/routed_gate_up"
DOWN_SUFFIX = "/moe/routed_down"
#: The routed banks' second level, one multiplier per row block: [experts][2] (gate, up) and
#: [experts] -- the contract `ops::SparseMoeWeights::routed_*_scale` reads.
ROUTED_SCALE_SUFFIX = "_scale"

_BLOCK = nvfp4_words.BLOCK

#: The fused parents stored as one object per constituent: attention's q|k|gate|v and the GDN
#: input's qkv|z are separate FP8 Linears, each with its own multiplier. The shared expert's
#: gate|up stays one object -- its two NVFP4 halves share their second level, so one GEMM over
#: the stacked rows is exact.
SPLIT_PARENTS = frozenset({"attention/query_key_gate_value", "gdn/query_key_value_z"})


def is_modelopt_export(config: Mapping[str, object]) -> bool:
    """Whether ``config`` declares a ModelOpt export; refuses one this reader cannot serve."""

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping) or quant.get("quant_method") != "modelopt":
        return False
    algo = quant.get("quant_algo")
    if algo != "MIXED_PRECISION":
        raise ValueError(
            f"ModelOpt export is quant_algo {algo!r}; this family serves MIXED_PRECISION exports "
            f"whose modules are {sorted(SERVED_ALGOS)}"
        )
    return True


def _fold(name: str) -> str:
    """The flat spelling the recipes use (``model.language_model.x`` -> ``model.x``)."""
    if name.startswith(NESTED_TEXT_PREFIX):
        return "model." + name[len(NESTED_TEXT_PREFIX):]
    return name


def _module_of(logical: str) -> str:
    assert logical.endswith(".weight"), logical
    return logical[: -len(".weight")]


def _positive(tensor: torch.Tensor, what: str) -> float:
    value = float(tensor.reshape(()).to(torch.float32))
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"{what} is {value}, expected finite positive")
    return value


def _fp32_bits(tensor: torch.Tensor, what: str) -> bytes:
    """One stored FP32 scalar's four bytes, as stored (a finite FP32 value round-trips a
    Python float exactly)."""
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise TypeError(f"{what} is {tensor.dtype} {tuple(tensor.shape)}, expected one FP32 value")
    return struct.pack("<f", _positive(tensor, what))


#: The trailing word of a routed bank, whose second level is per expert beside it.
_IDENTITY = struct.pack("<f", 1.0)


@dataclass(frozen=True)
class ObjectSource:
    """One artifact object as runs of rows from logical sources, all of one algorithm.

    ``algo`` is W4A16_NVFP4, FP8, or None for BF16 rows. A fused parent kept whole (the shared
    expert's gate|up) has more than one segment; its constituents must share their second
    level, which `payload_for` checks against the stored words.
    """

    name: str
    segments: tuple[tuple[str, np.ndarray], ...]
    k: int
    algo: str | None

    @property
    def n(self) -> int:
        return sum(int(rows.size) for _, rows in self.segments)


@dataclass(frozen=True)
class SourcePlan:
    specs: tuple[TensorSpec, ...]
    #: Text-core objects read from the stored words.
    objects: Mapping[str, ObjectSource]
    #: The inventory's object names this plan took over, so the base preflight skips them.
    covered: frozenset[str]
    #: Objects outside the text core whose base recipes read a quantised module (the draft
    #: head gathers rows of the output head): materialised through ``dequantizing_reader``.
    dequantized: frozenset[str]
    #: Per-layer routed objects, built by `RoutedCache`.
    routed: frozenset[str]
    #: The MTP block's routed banks: BF16 in the export but stored one tensor per expert,
    #: which the base recipes (stacked or fused expert tensors) do not read. `stacked_experts`
    #: assembles the rows they describe; the base format's encoder takes it from there.
    stacked: frozenset[str]
    counts: Mapping[str, int]


class ModelOptSource:
    """The text core of one ModelOpt MIXED_PRECISION checkpoint, one object per Linear."""

    def __init__(self, model_dir: str | Path, config: Mapping[str, object],
                 geometry: inventory.Geometry) -> None:
        self.model_dir = Path(model_dir)
        self.geometry = geometry
        if not is_modelopt_export(config):
            raise ValueError(f"{self.model_dir}: not a ModelOpt export")
        quant = config["quantization_config"]
        layers = quant.get("quantized_layers")
        if not isinstance(layers, Mapping) or not layers:
            raise ValueError("ModelOpt quantization_config has no quantized_layers")
        self.algo: dict[str, str] = {}
        for module, entry in layers.items():
            algo = entry.get("quant_algo") if isinstance(entry, Mapping) else None
            if algo not in SERVED_ALGOS:
                raise ValueError(f"{module}: quant_algo {algo!r} is not one of {sorted(SERVED_ALGOS)}")
            if algo == W4A16_NVFP4 and entry.get("group_size") != _BLOCK:
                raise ValueError(f"{module}: W4A16_NVFP4 needs group_size {_BLOCK}, "
                                 f"the export states {entry.get('group_size')}")
            self.algo[_fold(module)] = algo
        self.kv_cache_scheme = quant.get("kv_cache_scheme")
        self.headers = {
            _fold(name): (tuple(shape), dtype)
            for name, (shape, dtype) in read_tensor_headers(self.model_dir).items()
        }
        self.check_against_checkpoint()
        # Logical [n, k] of every Linear weight under the recipes' names; a packed NVFP4 weight
        # is [n, k/2] U8 on disk.
        self.logical: dict[str, tuple[int, ...]] = {}
        for name, (shape, dtype) in self.headers.items():
            if not name.endswith(".weight"):
                continue
            if self.module_algo(name) == W4A16_NVFP4:
                self.logical[name] = (shape[0], shape[1] * 2)
            else:
                self.logical[name] = shape

    # -- the declaration against the tensors ----------------------------------------------

    def module_algo(self, logical: str) -> str | None:
        """The algorithm that applies to a module (``...weight`` or the module name).

        The routed experts are one entry, ``mlp.experts``, over per-expert modules
        ``mlp.experts.{e}.{gate,up,down}_proj``.
        """
        module = _module_of(logical) if logical.endswith(".weight") else logical
        if module in self.algo:
            return self.algo[module]
        parts = module.rsplit(".", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[0] in self.algo:
            return self.algo[parts[0]]
        return None

    def check_against_checkpoint(self) -> None:
        """Every module the declaration quantises stores the words its algorithm needs, and
        every quantised tensor in the file is claimed by the declaration.

        The declaration is a claim, the tensors are the fact; a disagreement either way means
        the file is not what its config says, and nothing downstream should guess.
        """

        problems: list[str] = []
        for name, (shape, dtype) in self.headers.items():
            if not name.endswith(".weight"):
                continue
            algo = self.module_algo(name)
            module = _module_of(name)
            if dtype == "U8" and algo != W4A16_NVFP4:
                problems.append(f"{name} is packed U8 but no W4A16_NVFP4 entry claims {module}")
            elif dtype == "F8_E4M3" and algo != FP8:
                problems.append(f"{name} is E4M3 but no FP8 entry claims {module}")
            elif algo == W4A16_NVFP4:
                scale = self.headers.get(module + ".weight_scale")
                second = self.headers.get(module + ".weight_scale_2")
                if dtype != "U8" or len(shape) != 2:
                    problems.append(f"{name}: W4A16_NVFP4 codes must be U8 [n, k/2], found {dtype} {shape}")
                elif scale is None or scale[1] != "F8_E4M3" or scale[0] != (shape[0], shape[1] * 2 // _BLOCK):
                    problems.append(f"{module}.weight_scale must be F8_E4M3 [n, k/16], found {scale}")
                elif second is None or second[1] != "F32" or math.prod(second[0]) != 1:
                    problems.append(f"{module}.weight_scale_2 must be one FP32 value, found {second}")
                if module + ".input_scale" in self.headers:
                    problems.append(f"{module} stores an input_scale; W4A16 is weight-only")
            elif algo == FP8:
                scale = self.headers.get(module + ".weight_scale")
                if dtype != "F8_E4M3" or len(shape) != 2:
                    problems.append(f"{name}: FP8 codes must be E4M3 [n, k], found {dtype} {shape}")
                elif scale is None or scale[1] != "F32" or math.prod(scale[0]) not in (1, shape[0]):
                    problems.append(f"{module}.weight_scale must be FP32 per tensor or per row, found {scale}")
        claimed = {m for m in self.algo}
        stored_modules = {_module_of(n) for n in self.headers if n.endswith(".weight")}
        for module in sorted(claimed):
            if module in stored_modules:
                continue
            if any(m.startswith(module + ".") for m in stored_modules):
                continue  # the routed experts' entry names their parent module
            problems.append(f"{module}: quantized_layers claims it but the file stores no weight for it")
        if problems:
            raise ValueError("quantization_config disagrees with the checkpoint's tensors:\n  "
                             + "\n  ".join(problems[:12])
                             + (f"\n  ... and {len(problems) - 12} more" if len(problems) > 12 else ""))

    def summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for algo in self.algo.values():
            counts[algo] = counts.get(algo, 0) + 1
        return {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                "modules": counts, "kv_cache_scheme": self.kv_cache_scheme}

    # -- planning ------------------------------------------------------------------------

    def plan(self, specs: Sequence[TensorSpec], recipes_by_name: Mapping[str, TensorRecipe]) -> SourcePlan:
        """Rewrite the text-core specs: formats from the export, attention and GDN parents split.

        The routed banks become NVFP4 with their per-expert second level; every other
        Linear-derived text-core object takes its module's format (one object per segment for
        the parents in SEGMENT_NAMES, the shared expert's gate|up kept whole); anything else
        passes through to the base recipe.
        """

        out: list[TensorSpec] = []
        objects: dict[str, ObjectSource] = {}
        covered: set[str] = set()
        dequantized: set[str] = set()
        routed: set[str] = set()
        stacked: set[str] = set()
        g = self.geometry
        for spec in specs:
            recipe = recipes_by_name.get(spec.name)
            suffix = _text_core_suffix(spec.name)
            if recipe is None:
                out.append(spec)
                continue
            if spec.name.startswith("mtp/") and (spec.name.endswith(GATE_UP_SUFFIX)
                                                 or spec.name.endswith(DOWN_SUFFIX)):
                if self.module_algo("mtp.layers.0.mlp.experts") is not None:
                    raise ValueError("the MTP block's experts are quantised in this export; this "
                                     "reader serves them as the BF16 the export's `ignore` leaves")
                stacked.add(spec.name)
                covered.add(spec.name)
                out.append(spec)
                continue
            if suffix is None:
                if any(self.module_algo(source.name) for source in expression_sources(recipe.expression)
                       if source.name.endswith(".weight")):
                    dequantized.add(spec.name)
                    covered.add(spec.name)
                out.append(spec)
                continue
            if spec.name.endswith(GATE_UP_SUFFIX) or spec.name.endswith(DOWN_SUFFIX):
                layer = int(spec.name.split("/")[2])
                if self.module_algo(f"model.layers.{layer}.mlp.experts") != W4A16_NVFP4:
                    raise ValueError(f"layer {layer}: the routed experts are not W4A16_NVFP4 in this export")
                gate_up = spec.name.endswith(GATE_UP_SUFFIX)
                out.append(TensorSpec(spec.name, spec.shape, NVFP4_A16, BLOCK_SCALE_LAYOUT))
                out.append(tensor_spec(spec.name + ROUTED_SCALE_SUFFIX,
                                       ((2 if gate_up else 1) * g.experts,), FP32))
                routed.update((spec.name, spec.name + ROUTED_SCALE_SUFFIX))
                covered.add(spec.name)
                continue
            program = evaluate_rows(recipe.expression, self.logical, None)
            if program is None:
                out.append(spec)
                continue
            segments = program.segments()
            if suffix in SPLIT_PARENTS:
                names = SEGMENT_NAMES[suffix]
                if len(segments) != len(names):
                    raise ValueError(f"{spec.name}: recipe has {len(segments)} segments, "
                                     f"SEGMENT_NAMES lists {len(names)}")
                prefix = spec.name[: -len(suffix)]
                for name, segment in zip(names, segments):
                    self._emit(out, objects, prefix + name, (segment,), program.k)
                covered.add(spec.name)
                continue
            algos = {self.module_algo(source) for source, _ in segments}
            if len(algos) != 1:
                raise ValueError(f"{spec.name}: its sources mix formats {sorted(map(str, algos))}")
            if program.rows.size != spec.shape[0]:
                out.append(spec)
                continue
            if len(segments) > 1 and algos == {None}:
                out.append(spec)  # a multi-source BF16 parent the engine reads fused
                continue
            self._emit(out, objects, spec.name, tuple(segments), program.k)
            covered.add(spec.name)
        counts: dict[str, int] = {}
        for item in objects.values():
            key = item.algo or BF16
            counts[key] = counts.get(key, 0) + 1
        counts["routed"] = len([n for n in routed if not n.endswith(ROUTED_SCALE_SUFFIX)])
        counts["mtp_stacked"] = len(stacked)
        return SourcePlan(tuple(out), objects, frozenset(covered), frozenset(dequantized),
                          frozenset(routed), frozenset(stacked), counts)

    def _emit(self, out: list[TensorSpec], objects: dict[str, ObjectSource], name: str,
              segments: tuple[tuple[str, np.ndarray], ...], k: int) -> None:
        algo = self.module_algo(segments[0][0])
        item = ObjectSource(name, segments, k, algo)
        objects[name] = item
        if algo == W4A16_NVFP4:
            out.append(TensorSpec(name, (item.n, k), NVFP4_A16, BLOCK_SCALE_LAYOUT))
        elif algo == FP8:
            if len(segments) != 1:
                raise ValueError(f"{name}: an FP8 object is read from one Linear")
            out.append(TensorSpec(name, (item.n, k), FP8_ROW_F32, ROW_SCALE_F32_LAYOUT))
        else:
            out.append(TensorSpec(name, (item.n, k), BF16, CONTIGUOUS_LAYOUT))

    # -- payloads ------------------------------------------------------------------------

    def _second_level(self, reader: ShardReader, module: str) -> torch.Tensor:
        return reader.get(module + ".weight_scale_2")

    def payload_for(self, name: str, objects: Mapping[str, ObjectSource], reader: ShardReader) -> bytes:
        item = objects[name]
        if item.algo == W4A16_NVFP4:
            # The second level as stored; a fused parent's constituents must agree on it.
            words = {_fp32_bits(self._second_level(reader, _module_of(source)), source)
                     for source, _ in item.segments}
            if len(words) != 1:
                raise ValueError(f"{item.name}: its constituents' weight_scale_2 differ; one "
                                 "object carries one second level")
            codes, scales = [], []
            for source, rows in item.segments:
                module = _module_of(source)
                index = torch.from_numpy(np.ascontiguousarray(rows))
                packed = reader.get(module + ".weight")
                block = reader.get(module + ".weight_scale")
                if packed.dtype != torch.uint8 or block.dtype != torch.float8_e4m3fn:
                    raise TypeError(f"{module}: expected U8 codes and E4M3 block scales")
                # Rows are self-contained in NVFP4 (codes and block scales both per row), so a
                # row gather is exact.
                codes.append(packed.index_select(0, index))
                scales.append(block.view(torch.uint8).index_select(0, index))
            return encode_nvfp4_a16(torch.cat(codes).contiguous(), torch.cat(scales).contiguous(),
                                    words.pop(), (item.n, item.k))
        if item.algo == FP8:
            # Rows are self-contained (codes and one multiplier each), so a row gather is exact;
            # the per-tensor multiplier is repeated per row, widened exactly to FP32.
            (source, rows), = item.segments
            index = torch.from_numpy(np.ascontiguousarray(rows))
            codes, multipliers = _fp8_words(reader, source)
            return encode_fp8_row_f32(codes.index_select(0, index).contiguous(),
                                      multipliers.index_select(0, index).to(torch.float32).contiguous(),
                                      (item.n, item.k))
        parts = []
        for source, rows in item.segments:
            weight = reader.get(source)
            if weight.dtype != torch.bfloat16:
                raise TypeError(f"{source} is {weight.dtype}, expected bfloat16")
            parts.append(weight.index_select(0, torch.from_numpy(np.ascontiguousarray(rows))))
        return encode_direct(torch.cat(parts).contiguous(), BF16)

    def dequantizing_reader(self, reader: ShardReader) -> "DequantizingReader":
        """``reader``, answering a quantised module's ``weight`` with the values it stands for."""
        return DequantizingReader(self, reader)


def dequantize_nvfp4(reader: ShardReader, module: str) -> torch.Tensor:
    """The FP32 values a W4A16_NVFP4 module stands for: ``e2m1 * block * weight_scale_2``."""
    second = _positive(reader.get(module + ".weight_scale_2"), module + ".weight_scale_2")
    # The shared helper divides; ModelOpt's second level multiplies, and a division by its
    # reciprocal would round, so the multiplier is applied here.
    return nvfp4_words.dequantize(reader.get(module + ".weight"), reader.get(module + ".weight_scale"),
                                  1.0) * second


class DequantizingReader:
    """A shard reader for the base recipes: a quantised module's ``weight`` reads as its values.

    The draft head gathers rows of the output head, which the export stores as NVFP4; it gets
    the FP32 values those words stand for, so whatever format it encodes into starts from the
    export's own numbers. Every other tensor passes through untouched.
    """

    def __init__(self, source: ModelOptSource, reader: ShardReader) -> None:
        self._source = source
        self._reader = reader

    def has(self, name: str) -> bool:
        return self._reader.has(name)

    def get(self, name: str) -> torch.Tensor:
        if name.endswith(".weight"):
            algo = self._source.module_algo(name)
            if algo == W4A16_NVFP4:
                return dequantize_nvfp4(self._reader, _module_of(name))
            if algo == FP8:
                codes, multipliers = _fp8_words(self._reader, name)
                return codes.view(torch.float8_e4m3fn).to(torch.float32) * multipliers.to(
                    torch.float32).unsqueeze(1)
        return self._reader.get(name)

    def metadata(self, names):
        return self._reader.metadata(names)


# -- the routed banks --------------------------------------------------------------------


def _expert_projection(reader: ShardReader, layer: int, expert: int, projection: str,
                       rows: int, columns: int):
    """One projection of one expert: codes, block scales, and its second level."""
    base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
    codes = reader.get(base + "weight")
    scales = reader.get(base + "weight_scale")
    second = reader.get(base + "weight_scale_2")
    if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{base}: expected U8 codes and E4M3 block scales")
    if tuple(codes.shape) != (rows, columns // 2) or tuple(scales.shape) != (rows, columns // _BLOCK):
        raise ValueError(f"{base}: stored {tuple(codes.shape)} / {tuple(scales.shape)}, expected "
                         f"({rows}, {columns // 2}) / ({rows}, {columns // _BLOCK})")
    _positive(second, base + "weight_scale_2")
    if second.dtype != torch.float32:
        raise TypeError(f"{base}weight_scale_2 is {second.dtype}, expected float32")
    return codes, scales.view(torch.uint8), second.reshape(())


def build_gate_up(reader: ShardReader, layer: int, g: inventory.Geometry) -> tuple[bytes, torch.Tensor]:
    """One layer's stacked gate/up experts and their second levels [experts][2] (gate, up).

    Rows run ``(expert * 2 + half) * intermediate + row`` with gate first. The weight-only
    kernels carry one second level per expert for the stacked rows, so gate and up must agree
    on it; a source where they do not is refused rather than half-applied.
    """
    rows = g.experts * 2 * g.intermediate
    codes = torch.empty((rows, g.hidden // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.hidden // _BLOCK), dtype=torch.uint8)
    second = torch.empty(2 * g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        for half, projection in enumerate(("gate_proj", "up_proj")):
            plane, block, level = _expert_projection(reader, layer, expert, projection,
                                                     g.intermediate, g.hidden)
            begin = (expert * 2 + half) * g.intermediate
            codes[begin: begin + g.intermediate] = plane
            scales[begin: begin + g.intermediate] = block
            second[expert * 2 + half] = level
        if second[expert * 2].view(torch.int32) != second[expert * 2 + 1].view(torch.int32):
            raise ValueError(f"layer {layer} expert {expert}: gate and up weight_scale_2 differ "
                             f"({float(second[expert * 2])} vs {float(second[expert * 2 + 1])}); "
                             "the stacked GEMM carries one per expert")
    payload = encode_nvfp4_a16(codes, scales, _IDENTITY, (rows, g.hidden))
    return payload, second


def build_down(reader: ShardReader, layer: int, g: inventory.Geometry) -> tuple[bytes, torch.Tensor]:
    """One layer's stacked down experts and their second levels [experts]."""
    rows = g.experts * g.hidden
    codes = torch.empty((rows, g.intermediate // 2), dtype=torch.uint8)
    scales = torch.empty((rows, g.intermediate // _BLOCK), dtype=torch.uint8)
    second = torch.empty(g.experts, dtype=torch.float32)
    for expert in range(g.experts):
        plane, block, level = _expert_projection(reader, layer, expert, "down_proj",
                                                 g.hidden, g.intermediate)
        codes[expert * g.hidden: (expert + 1) * g.hidden] = plane
        scales[expert * g.hidden: (expert + 1) * g.hidden] = block
        second[expert] = level
    payload = encode_nvfp4_a16(codes, scales, _IDENTITY, (rows, g.intermediate))
    return payload, second


def stacked_experts(reader: ShardReader, name: str, g: inventory.Geometry) -> torch.Tensor:
    """The MTP block's routed bank ``name`` as the BF16 rows the base recipe describes.

    The export stores ``mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight`` one tensor
    per expert; the object's rows are expert-major, gate then up within an expert for gate/up
    -- the rows a stacked or fused expert tensor would have given.
    """
    base = "mtp.layers.0.mlp.experts."
    gate_up = name.endswith(GATE_UP_SUFFIX)
    projections = ("gate_proj", "up_proj") if gate_up else ("down_proj",)
    rows, columns = (g.intermediate, g.hidden) if gate_up else (g.hidden, g.intermediate)
    out = torch.empty((g.experts * len(projections) * rows, columns), dtype=torch.bfloat16)
    for expert in range(g.experts):
        for half, projection in enumerate(projections):
            weight = reader.get(f"{base}{expert}.{projection}.weight")
            if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (rows, columns):
                raise ValueError(f"{base}{expert}.{projection}.weight is {weight.dtype} "
                                 f"{tuple(weight.shape)}, expected bfloat16 ({rows}, {columns})")
            begin = (expert * len(projections) + half) * rows
            out[begin: begin + rows] = weight
    return out


def routed_source_names(layer: int, g: inventory.Geometry) -> Iterator[str]:
    for expert in range(g.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
            yield from (base + leaf for leaf in ("weight", "weight_scale", "weight_scale_2"))


def preflight_routed(reader: ShardReader, g: inventory.Geometry) -> None:
    """Every expert's three tensors exist with the signature the geometry implies."""
    for layer in range(g.layers):
        metadata = reader.metadata(tuple(routed_source_names(layer, g)))
        for expert in range(g.experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                n, k = (g.hidden, g.intermediate) if projection == "down_proj" else (g.intermediate, g.hidden)
                base = f"model.layers.{layer}.mlp.experts.{expert}.{projection}."
                for leaf, shape, dtype in (("weight", (n, k // 2), "U8"),
                                           ("weight_scale", (n, k // _BLOCK), "F8_E4M3")):
                    actual = metadata[base + leaf]
                    if actual.shape != shape or actual.dtype != dtype:
                        raise ValueError(f"{base + leaf}: stored {actual.dtype} {actual.shape}, "
                                         f"expected {dtype} {shape}")
                actual = metadata[base + "weight_scale_2"]
                if math.prod(actual.shape) != 1 or actual.dtype != "F32":
                    raise ValueError(f"{base}weight_scale_2: expected one FP32 value")


class RoutedCache:
    """Builds a layer's four routed objects once and hands them out in inventory order."""

    def __init__(self, reader: ShardReader, geometry: inventory.Geometry) -> None:
        self._reader = reader
        self._geometry = geometry
        self._layer: int | None = None
        self._objects: dict[str, object] = {}

    def payload_for(self, name: str) -> bytes:
        layer = int(name.split("/")[2])
        if layer != self._layer:
            gate_up, gate_up_scale = build_gate_up(self._reader, layer, self._geometry)
            down, down_scale = build_down(self._reader, layer, self._geometry)
            prefix = f"text/layers/{layer}"
            self._objects = {
                prefix + GATE_UP_SUFFIX: gate_up,
                prefix + GATE_UP_SUFFIX + ROUTED_SCALE_SUFFIX: gate_up_scale,
                prefix + DOWN_SUFFIX: down,
                prefix + DOWN_SUFFIX + ROUTED_SCALE_SUFFIX: down_scale,
            }
            self._layer = layer
        value = self._objects.pop(name)
        if isinstance(value, bytes):
            return value
        return encode_direct(value.contiguous(), FP32)


__all__ = [
    "FP8",
    "ROUTED_SCALE_SUFFIX",
    "SERVED_ALGOS",
    "W4A16_NVFP4",
    "WEIGHTS_ID",
    "DequantizingReader",
    "ModelOptSource",
    "ObjectSource",
    "RoutedCache",
    "SourcePlan",
    "dequantize_nvfp4",
    "is_modelopt_export",
    "preflight_routed",
    "stacked_experts",
]
