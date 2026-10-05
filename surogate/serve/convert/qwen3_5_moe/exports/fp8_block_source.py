"""Qwen's block-FP8 export of this architecture (``quant_method: fp8``, 128 x 128 blocks).

Every quantised Linear is stored as E4M3 codes ``X.weight`` [n, k] with one multiplier per
128 x 128 block in ``X.weight_scale_inv`` [n/128, k/128] (DeepSeek's name: the value
multiplies). Qwen keeps those multipliers in BF16; they widen to the artifact's FP32 exactly.

Where the engine has a block-FP8 kernel the stored codes become the artifact's codes, with no
dequantise-requantise round trip:

- the routed experts, which the export stores one Linear per expert
  (``mlp.experts.{e}.gate_proj``) and the artifact stacks into the inventory's two parents,
  ``[experts * 2 * intermediate, hidden]`` with each expert's rows ``[gate; up]`` and
  ``[experts * hidden, intermediate]``. Hopper's grouped GEMM serves them
  (ops/sparse_moe/fp8_sm90);
- the text layers' dense projections (attention and GDN, input and output), whose recipes
  are pure row programs over whole 128-row blocks, so a fused parent is its constituents'
  blocks in recipe order.

The shared expert's kernels read W8 only, and the MTP block's dense projections stay the
base converter's W8, so those are dequantised through the stored words and encoded as the
base converter encodes them. Everything the export keeps in BF16 (router, embedding, output
head, norms, GDN gates, vision) takes the base recipes unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from surogate.core.model.quant_schemes import read_tensor_shapes
from surogate.serve.artifact.layouts import encode_fp8_block_scaled
from surogate.serve.convert.common.conversion import encode_tensor_payload
from surogate.serve.convert.common.inventory import BLOCK128_LAYOUT, FP8_BLOCK, TensorSpec
from surogate.serve.convert.common.recipe import TensorRecipe, expression_sources
from surogate.serve.convert.common.row_algebra import RowShapeMismatch, evaluate_rows
from surogate.serve.convert.common.safetensors import ShardReader


WEIGHTS_ID = "fp8-block"

BLOCK = 128
SCALE_SUFFIX = "_scale_inv"  # appended to a logical ``.weight`` name

#: Per-layer objects that keep the stored codes when every row block they read is FP8.
NATIVE_SUFFIXES = (
    "attention/query_key_gate_value",
    "attention/output",
    "gdn/query_key_value_z",
    "gdn/output",
)
ROUTED_SUFFIXES = ("moe/routed_gate_up", "moe/routed_down")

#: The VL-style nesting official Qwen3.5/3.6 exports use for the text tower; the recipes speak
#: the flat dialect, as ShardReader does (PATCHES.md #16).
NESTED_TEXT_PREFIX = "model.language_model."


def is_fp8_block_export(config: Mapping[str, object]) -> bool:
    """Whether ``config`` declares a block-FP8 export; refuses FP8 layouts this reader lacks."""

    quant = config.get("quantization_config")
    if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
        return False
    block = list(quant.get("weight_block_size") or ())
    fmt = quant.get("fmt", "e4m3")
    if block != [BLOCK, BLOCK] or fmt != "e4m3":
        raise ValueError(
            f"FP8 export with weight_block_size {block or None} and fmt {fmt!r}: only E4M3 codes "
            f"with {BLOCK} x {BLOCK} block scales are supported"
        )
    return True


def _fold(name: str) -> str:
    if name.startswith(NESTED_TEXT_PREFIX):
        return "model." + name[len(NESTED_TEXT_PREFIX):]
    return name


def _role(name: str) -> str | None:
    """``moe/routed_down`` for ``text/layers/3/moe/routed_down`` or ``mtp/layer/moe/...``."""

    parts = name.split("/", 3)
    if len(parts) == 4 and parts[:2] == ["text", "layers"] and parts[2].isdigit():
        return parts[3]
    if name.startswith("mtp/layer/"):
        return name[len("mtp/layer/"):]
    return None


def _whole_blocks(rows: np.ndarray) -> bool:
    """Rows that are a sequence of whole, aligned 128-row blocks of their source."""

    if rows.size % BLOCK:
        return False
    blocks = rows.reshape(-1, BLOCK)
    return bool((blocks[:, 0] % BLOCK == 0).all() and (blocks == blocks[:, :1] + np.arange(BLOCK)).all())


@dataclass(frozen=True)
class ObjectSource:
    """How one artifact object is produced from the export.

    ``codes``: the stored codes and block scales of ``segments``, in order. ``requantize``: the
    segments' values, encoded in ``spec.format`` as the base converter would. ``routed``: the
    per-expert Linears under ``experts`` stacked into the inventory's parent.
    """

    spec: TensorSpec
    kind: str
    segments: tuple[tuple[str, np.ndarray], ...] = ()
    experts: str = ""
    expert_count: int = 0


@dataclass(frozen=True)
class SourcePlan:
    specs: tuple[TensorSpec, ...]
    objects: Mapping[str, ObjectSource]
    #: The inventory's object names this plan writes, so the base preflight skips their sources.
    covered: frozenset[str]
    counts: Mapping[str, int] = field(default_factory=dict)


class Fp8BlockSource:
    """The tensors of one block-FP8 export, under the recipes' flat names."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        stored = read_tensor_shapes(self.model_dir)
        ambiguous = any(n.startswith(NESTED_TEXT_PREFIX) and _fold(n) in stored for n in stored)
        self.shapes: dict[str, tuple[int, ...]] = {
            (name if ambiguous else _fold(name)): tuple(shape) for name, shape in stored.items()
        }

    def is_quantized(self, logical: str) -> bool:
        return logical + SCALE_SUFFIX in self.shapes

    # -- planning --------------------------------------------------------------

    def plan(
        self,
        specs: Sequence[TensorSpec],
        recipes_by_name: Mapping[str, TensorRecipe],
        experts: int,
    ) -> SourcePlan:
        out: list[TensorSpec] = []
        objects: dict[str, ObjectSource] = {}
        counts = {"codes": 0, "routed": 0, "requantize": 0}
        for spec in specs:
            recipe = recipes_by_name.get(spec.name)
            role = _role(spec.name)
            if recipe is None or not hasattr(spec, "format"):
                out.append(spec)
                continue
            if role in ROUTED_SUFFIXES:
                item = self._routed(spec, recipe, experts)
            else:
                item = self._rows(spec, recipe, role)
            if item is None:
                out.append(spec)  # BF16 sources: the base recipes read them
                continue
            objects[spec.name] = item
            counts[item.kind] += 1
            out.append(item.spec)
        return SourcePlan(tuple(out), objects, frozenset(objects), counts)

    def _routed(self, spec: TensorSpec, recipe: TensorRecipe, experts: int) -> ObjectSource:
        # The recipe names the stacked-expert tensors a BF16 export has
        # (`mlp.experts.gate_up_proj`); their module is where this export keeps one Linear per
        # expert.
        stacked = {source.name for source in expression_sources(recipe.expression)}
        modules = {name.rsplit(".", 1)[0] for name in stacked}
        if len(modules) != 1:
            raise ValueError(f"{spec.name}: expected one experts module, found {sorted(modules)}")
        module = modules.pop()
        projections = ("down_proj",) if spec.name.endswith("routed_down") else ("gate_proj", "up_proj")
        n, k = spec.shape
        rows = n // (experts * len(projections))
        for expert in range(experts):
            for projection in projections:
                logical = f"{module}.{expert}.{projection}.weight"
                if self.shapes.get(logical) != (rows, k):
                    raise ValueError(
                        f"{spec.name}: {logical} is {self.shapes.get(logical)}, expected {(rows, k)}"
                    )
                if not self.is_quantized(logical):
                    raise ValueError(f"{spec.name}: {logical} has no block scales")
        if rows % BLOCK or k % BLOCK:
            raise ValueError(f"{spec.name}: expert matrices [{rows}, {k}] are not whole 128 x 128 blocks")
        return ObjectSource(TensorSpec(spec.name, spec.shape, FP8_BLOCK, BLOCK128_LAYOUT), "routed",
                            experts=module, expert_count=experts)

    def _rows(self, spec: TensorSpec, recipe: TensorRecipe, role: str | None) -> ObjectSource | None:
        try:
            program = evaluate_rows(recipe.expression, self.shapes, None)
        except RowShapeMismatch:
            # A BF16 source stored with unit axes the recipe omits; the base path reshapes those.
            if any(self.is_quantized(s.name) for s in expression_sources(recipe.expression)):
                raise
            return None
        if program is None:
            return None
        segments = tuple(program.segments())
        quantized = [self.is_quantized(source) for source, _ in segments]
        if not any(quantized):
            return None
        native = (
            spec.name.startswith("text/layers/")
            and role in NATIVE_SUFFIXES
            and all(quantized)
            and program.k % BLOCK == 0
            and all(_whole_blocks(rows) for _, rows in segments)
        )
        if native:
            return ObjectSource(TensorSpec(spec.name, spec.shape, FP8_BLOCK, BLOCK128_LAYOUT),
                                "codes", segments)
        return ObjectSource(spec, "requantize", segments)

    # -- payloads ----------------------------------------------------------------

    def payload_for(
        self,
        name: str,
        objects: Mapping[str, ObjectSource],
        reader: ShardReader,
        device: str | torch.device = "cpu",
    ) -> bytes:
        item = objects[name]
        if item.kind == "routed":
            projections = (
                ("down_proj",) if name.endswith("routed_down") else ("gate_proj", "up_proj")
            )
            codes, scales = [], []
            for expert in range(item.expert_count):
                for projection in projections:
                    logical = f"{item.experts}.{expert}.{projection}.weight"
                    codes.append(_codes(reader, logical))
                    scales.append(_scales(reader, logical, codes[-1].shape))
            return encode_fp8_block_scaled(torch.cat(codes), torch.cat(scales), item.spec.shape)
        if item.kind == "codes":
            codes, scales = [], []
            for source, rows in item.segments:
                stored = _codes(reader, source)
                block_scales = _scales(reader, source, stored.shape)
                index = torch.from_numpy(np.ascontiguousarray(rows))
                codes.append(stored.index_select(0, index))
                scales.append(block_scales.index_select(0, index[::BLOCK] // BLOCK))
            return encode_fp8_block_scaled(torch.cat(codes), torch.cat(scales), item.spec.shape)
        parts = []
        for source, rows in item.segments:
            index = torch.from_numpy(np.ascontiguousarray(rows))
            if self.is_quantized(source):
                parts.append(_dequantize(reader, source).index_select(0, index))
            else:
                parts.append(reader.get(source).index_select(0, index).to(torch.float32))
        return encode_tensor_payload(torch.cat(parts).contiguous(), item.spec, device)


def _codes(reader: ShardReader, logical: str) -> torch.Tensor:
    codes = reader.get(logical)
    if codes.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{logical} is {codes.dtype}, expected float8_e4m3fn")
    return codes.view(torch.uint8)


def _scales(reader: ShardReader, logical: str, shape: Sequence[int]) -> torch.Tensor:
    """One multiplier per block, FP32; a ragged last block row or column has its own."""

    scales = reader.get(logical + SCALE_SUFFIX)
    n, k = shape
    expected = (-(-n // BLOCK), -(-k // BLOCK))
    if tuple(scales.shape) != expected or scales.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"{logical}{SCALE_SUFFIX}: {tuple(scales.shape)} {scales.dtype}, expected {expected} "
            "BF16 or FP32"
        )
    return scales.to(torch.float32)


def _dequantize(reader: ShardReader, logical: str) -> torch.Tensor:
    """The values a block-FP8 Linear represents, FP32: code * multiplier of its block."""

    codes = reader.get(logical)
    if codes.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{logical} is {codes.dtype}, expected float8_e4m3fn")
    n, k = codes.shape
    scales = _scales(reader, logical, codes.shape)
    expanded = scales.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)[:, :k]
    return codes.to(torch.float32) * expanded


__all__ = [
    "WEIGHTS_ID",
    "Fp8BlockSource",
    "ObjectSource",
    "SourcePlan",
    "is_fp8_block_export",
]
