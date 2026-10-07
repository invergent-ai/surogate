"""One artifact object per HF Linear, in the format the export declares.

A compressed-tensors export decides per module what it quantized; nothing in
a converter may decide otherwise. This module applies that to the resolved checkpoint’s
text core: the inventory still says which objects exist and the recipes still
say where their rows come from, but the *format* of every Linear-derived
object comes from ``quant_schemes.resolve_checkpoint`` -- NVFP4 words copied
verbatim where the file packs the module, FP8 codes and their per-row
multipliers copied verbatim where an eight-bit float scheme stores them, BF16
rows copied verbatim where it quantized nothing -- and a fused parent whose
constituents are separate Linears in the file becomes one object per
constituent, because each carries its own global scale and there is no honest
way to fuse two of those.

An FP8 module is served with BF16 activations: the export's dynamic per-token
activation quantization is not applied, so its products are the exact
products of the stored weights.

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
    packs_weights,
    read_tensor_headers,
    read_tensor_shapes,
    resolve_checkpoint,
)
from surogate.serve.artifact.layouts import (
    encode_direct,
    encode_fp8_row_f32,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from surogate.serve.convert.common.conversion import encode_tensor_payload
from surogate.serve.convert.common.inventory import (
    BF16,
    FP8,
    FP8_ROW_F32,
    FP32,
    BLOCK_SCALE_LAYOUT,
    CONTIGUOUS_LAYOUT,
    NVFP4,
    ROW_SCALE_F32_LAYOUT,
    ROW_SCALE_LAYOUT,
    ROW_SPLIT_LAYOUT,
    W8,
    TensorSpec,
)
from surogate.serve.convert.common.recipe import TensorRecipe, expression_sources
from surogate.serve.convert.common.row_algebra import evaluate_rows
from surogate.serve.convert.common.safetensors import ShardReader


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

#: The shared expert's objects. The MoE kernels compute the shared expert inside their
#: bodies and admit W8 only, so until they take NVFP4 the converter requantises these to
#: the W8 the base converter would have produced -- the one place the export's format is
#: not kept. ``auto`` (the default, what ``surogate serve`` uses) does so whenever the
#: inventory asks for W8 there; ``w8`` is the same, spelled explicitly; ``as-stored``
#: refuses instead, for the day the kernels take NVFP4. The effective choice is recorded
#: in the plan and the report.
SHARED_EXPERT_SUFFIXES = ("moe/shared_gate_up", "moe/shared_down")
SHARED_EXPERT_CHOICES = ("auto", "as-stored", "w8")

#: The VL-style nesting official Qwen3.5/3.6 exports use for the text tower. The recipes
#: address sources in the flat dialect (``model.layers...``), as ShardReader does
#: (PATCHES.md #16); this planner folds the same way so that a nested export plans
#: exactly like a flat one.
NESTED_TEXT_PREFIX = "model.language_model."


@dataclass(frozen=True)
class ObjectSource:
    """One artifact object as a run of rows from one logical source.

    ``extra`` carries further (source, rows) segments for an object that keeps a
    fused parent while being requantised from separately stored halves.
    """

    name: str
    source: str  # logical HF tensor, e.g. "...self_attn.q_proj.weight"
    rows: np.ndarray  # row ids into the source, in object row order
    k: int
    quantized: bool  # NVFP4 (weight_packed) if True, FP8 or BF16 (weight) if not
    requantize_to: str | None = None  # a stored format to re-encode into, e.g. W8
    extra: tuple[tuple[str, np.ndarray], ...] = ()
    #: The artifact format of FP8 codes kept as stored (FP8 or FP8_ROW_F32), else None.
    fp8: str | None = None

    @property
    def n(self) -> int:
        return int(self.rows.size) + sum(int(rows.size) for _, rows in self.extra)

    def segments(self) -> tuple[tuple[str, np.ndarray], ...]:
        return ((self.source, self.rows),) + self.extra


@dataclass(frozen=True)
class SourcePlan:
    specs: tuple[TensorSpec, ...]
    objects: Mapping[str, ObjectSource]
    resolved: ResolvedQuantization
    #: The inventory's original object names this plan took over, so a preflight
    #: knows which recipes it no longer needs to find sources for.
    covered: frozenset[str]
    #: What became of the shared expert: "w8" (requantised) or "as-stored".
    shared_expert: str = "as-stored"
    #: Objects outside the text core whose base recipes read an FP8 module (the draft head
    #: gathers rows of the output head): the converter materialises them through
    #: ``dequantizing_reader``, and the base preflight, which wants BF16 sources, skips them.
    dequantized: frozenset[str] = frozenset()


def _module_of(logical: str) -> str:
    assert logical.endswith(".weight"), logical
    return logical[: -len(".weight")]


def _fold(name: str) -> str:
    """The flat spelling of one stored name (``model.language_model.x`` -> ``model.x``)."""
    if name.startswith(NESTED_TEXT_PREFIX):
        return "model." + name[len(NESTED_TEXT_PREFIX):]
    return name


def _nested_and_flat_both_present(stored: Mapping[str, object]) -> bool:
    """An export that spells one tensor both ways is ambiguous; keep names as stored."""
    return any(
        name.startswith(NESTED_TEXT_PREFIX) and _fold(name) in stored for name in stored
    )


class CompressedTensorsSource:
    """The text core of one compressed-tensors checkpoint, one object per Linear."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        resolved = resolve_checkpoint(self.model_dir)
        if resolved is None:
            raise ValueError(f"{self.model_dir}: not a compressed-tensors checkpoint")
        self.resolved = resolved
        stored = read_tensor_shapes(self.model_dir)
        self._dtypes: dict[str, str] | None = None  # read on the first FP8 question
        # logical shapes under the recipe's names; a packed source is [n, k/2] on disk.
        # Names are folded to the flat dialect the recipes speak; `self.stored_module`
        # remembers the module name the file (and the resolved quantization) uses.
        self.logical: dict[str, tuple[int, ...]] = {}
        self.stored_module: dict[str, str] = {}
        ambiguous = _nested_and_flat_both_present(stored)
        for name, shape in stored.items():
            if name.endswith("." + PACKED_WEIGHT):
                logical = name[: -len(PACKED_WEIGHT)] + "weight"
                folded = logical if ambiguous else _fold(logical)
                self.logical[folded] = (shape[0], shape[1] * 2)
                self.stored_module[_module_of(folded)] = _module_of(logical)
            elif name.endswith(".weight"):
                folded = name if ambiguous else _fold(name)
                if folded not in self.logical:
                    self.logical[folded] = tuple(shape)
                    self.stored_module[_module_of(folded)] = _module_of(name)

    def module_as_stored(self, logical: str) -> str:
        """The file's module name for a recipe-spelled logical tensor."""
        module = _module_of(logical)
        return self.stored_module.get(module, module)

    def is_quantized(self, logical: str) -> bool:
        """Whether the export packs this module's weight (NVFP4 words in ``weight_packed``)."""
        item = self.resolved.modules.get(self.module_as_stored(logical))
        return item is not None and packs_weights(item.scheme)

    def fp8_format(self, logical: str) -> str | None:
        """The artifact format that keeps this module's FP8 codes, or None if it has none.

        An eight-bit float scheme stores E4M3 codes in ``weight`` and one multiplier per
        row (``channel``) or one for the matrix (``tensor``) in ``weight_scale``. BF16
        multipliers keep their words in FP8 (E4M3 rows, BF16 scales); any other dtype
        widens exactly into FP8_ROW_F32. Anything else eight-bit is refused here, at plan
        time, rather than read as BF16.
        """

        if not logical.endswith(".weight"):
            return None  # a bias, a fused expert stack, a control vector: never quantized here
        module = self.module_as_stored(logical)
        item = self.resolved.modules.get(module)
        if item is None or item.scheme is None or item.scheme.weights is None:
            return None
        if packs_weights(item.scheme):
            return None
        weights = item.scheme.weights
        strategy = str(getattr(weights.strategy, "value", weights.strategy))
        if (str(getattr(weights.type, "value", weights.type)) != "float" or weights.num_bits != 8
                or not weights.symmetric or weights.dynamic or strategy not in ("channel", "tensor")):
            raise ValueError(
                f"{module}: eight-bit weights must be symmetric static FP8 per channel or per "
                f"tensor; the export declares {weights.type} {weights.num_bits}-bit {strategy}"
            )
        codes = self._dtype(module + ".weight")
        if codes != "F8_E4M3":
            raise ValueError(f"{module}.weight is {codes}, expected F8_E4M3 under an FP8 scheme")
        return FP8 if self._dtype(module + ".weight_scale") == "BF16" else FP8_ROW_F32

    def _dtype(self, stored: str) -> str:
        if self._dtypes is None:
            self._dtypes = {name: dtype for name, (_, dtype) in read_tensor_headers(self.model_dir).items()}
        if stored not in self._dtypes:
            raise ValueError(f"{self.model_dir}: no tensor {stored}")
        return self._dtypes[stored]

    def dequantizing_reader(self, reader: ShardReader) -> "DequantizingReader":
        """``reader``, answering an FP8 module's ``weight`` with the values it stands for."""
        return DequantizingReader(self, reader)

    # -- planning --------------------------------------------------------------

    def plan(
        self,
        specs: Sequence[TensorSpec],
        recipes_by_name: Mapping[str, TensorRecipe],
        *,
        shared_expert: str = "auto",
    ) -> SourcePlan:
        """Rewrite the text-core specs: formats from the export, parents split.

        Only objects whose recipe is a pure row program over 2-D sources are
        this module's business; a fused parent listed in SEGMENT_NAMES becomes
        one object per segment, any other Linear-derived object keeps its name
        and takes the file's format. Everything else passes through untouched.
        The shared expert's inventory objects keep their names and are
        requantised to W8 from the stored halves (see SHARED_EXPERT_SUFFIXES)
        under ``shared_expert="auto"`` (the default) or ``"w8"``;
        ``"as-stored"`` refuses while the MoE kernels admit W8 only.
        """

        if shared_expert not in SHARED_EXPERT_CHOICES:
            raise ValueError(f"shared_expert must be one of {SHARED_EXPERT_CHOICES}")
        requantise_shared = shared_expert in ("auto", "w8")
        out: list[TensorSpec] = []
        objects: dict[str, ObjectSource] = {}
        covered: set[str] = set()
        effective = "as-stored"
        dequantized: set[str] = set()
        for spec in specs:
            recipe = recipes_by_name.get(spec.name)
            suffix = _text_core_suffix(spec.name)
            if recipe is not None and suffix is None:
                if any(self.fp8_format(source.name) for source in expression_sources(recipe.expression)):
                    dequantized.add(spec.name)
                    covered.add(spec.name)
                out.append(spec)
                continue
            if recipe is None:
                out.append(spec)
                continue
            if suffix in SHARED_EXPERT_SUFFIXES and not requantise_shared:
                raise ValueError(
                    f"{spec.name}: serving currently requires W8 shared experts; "
                    "--shared-expert as-stored cannot be served until the MoE kernels take NVFP4"
                )
            program = evaluate_rows(recipe.expression, self.logical, None)
            if program is None:
                if suffix in SHARED_EXPERT_SUFFIXES:
                    raise ValueError(
                        f"{spec.name}: the shared expert's sources were not found in the "
                        "compressed-tensors export under either name dialect"
                    )
                out.append(spec)
                continue
            segments = program.segments()
            if requantise_shared and suffix in SHARED_EXPERT_SUFFIXES:
                if spec.format != W8:
                    raise ValueError(f"{spec.name}: expected a W8 inventory object, got {spec.format}")
                effective = "w8"
                (source, rows), *rest = segments
                objects[spec.name] = ObjectSource(
                    spec.name, source, rows, program.k, self.is_quantized(source),
                    requantize_to=W8, extra=tuple((s, r) for s, r in rest),
                )
                covered.add(spec.name)
                out.append(spec)
                continue
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
        return SourcePlan(tuple(out), objects, self.resolved, frozenset(covered), effective,
                          frozenset(dequantized))

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
        fp8 = self.fp8_format(source)
        item = ObjectSource(name, source, rows, k, quantized, fp8=fp8)
        objects[name] = item
        if quantized:
            out.append(TensorSpec(name, (item.n, k), NVFP4, BLOCK_SCALE_LAYOUT))
            out.append(TensorSpec(name + INPUT_DIVISOR_SUFFIX, (), FP32, CONTIGUOUS_LAYOUT))
        elif fp8 is not None:
            layout = ROW_SCALE_LAYOUT if fp8 == FP8 else ROW_SCALE_F32_LAYOUT
            out.append(TensorSpec(name, (item.n, k), fp8, layout))
        else:
            out.append(TensorSpec(name, (item.n, k), BF16, CONTIGUOUS_LAYOUT))

    # -- payloads ----------------------------------------------------------------

    def payload_for(
        self,
        name: str,
        objects: Mapping[str, ObjectSource],
        reader: ShardReader,
        device: str | torch.device = "cpu",
    ) -> bytes:
        if name.endswith(INPUT_DIVISOR_SUFFIX):
            item = objects[name[: -len(INPUT_DIVISOR_SUFFIX)]]
            word = reader.get(self.module_as_stored(item.source) + ".input_global_scale")
            return _fp32_word(word, item.source + " input_global_scale")
        item = objects[name]
        if item.requantize_to is not None:
            # Every segment brought to BF16 -- the NVFP4 ones dequantised through the same
            # words the engine would read -- then encoded as the base converter encodes.
            parts = []
            for source, rows in item.segments():
                index = torch.from_numpy(np.ascontiguousarray(rows))
                if self.is_quantized(source):
                    parts.append(_dequantize_nvfp4(reader, self.module_as_stored(source), index))
                elif self.fp8_format(source) is not None:
                    parts.append(_dequantize_fp8(reader, source).index_select(0, index))
                else:
                    parts.append(reader.get(source).index_select(0, index).to(torch.bfloat16))
            fused = torch.cat(parts, dim=0).contiguous()
            layout = ROW_SPLIT_LAYOUT if item.requantize_to == W8 else CONTIGUOUS_LAYOUT
            spec = TensorSpec(name, (item.n, item.k), item.requantize_to, layout)
            return encode_tensor_payload(fused, spec, device)
        module = self.module_as_stored(item.source)
        rows = torch.from_numpy(np.ascontiguousarray(item.rows))
        if item.fp8 is not None:
            # Rows are self-contained (codes and one multiplier each), so a row gather is
            # exact, and the multipliers keep their words: BF16 as stored, or widened to FP32.
            codes, scales = _fp8_words(reader, item.source)
            codes = codes.index_select(0, rows).contiguous()
            scales = scales.index_select(0, rows).contiguous()
            if item.fp8 == FP8:
                return encode_fp8_row_scaled(codes, scales, (item.n, item.k))
            return encode_fp8_row_f32(codes, scales.to(torch.float32), (item.n, item.k))
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


#: E2M1 nibble -> value: sign in bit 3, exponent bits 2-1, mantissa bit 0.
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequantize_nvfp4(reader: ShardReader, module: str, rows: torch.Tensor) -> torch.Tensor:
    """The values an NVFP4 module represents, as BF16 rows.

    compressed-tensors: value = code * fp8_block_scale / weight_global_scale (the global
    scale divides). Low nibble is the even column, as the packed layout stores it.
    """

    codes = reader.get(module + ".weight_packed").index_select(0, rows)
    scales = reader.get(module + ".weight_scale").index_select(0, rows)
    global_scale = float(reader.get(module + ".weight_global_scale").reshape(()).to(torch.float32))
    if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{module}: unexpected packed dtypes {codes.dtype}, {scales.dtype}")
    n, half = codes.shape
    values = torch.empty((n, half * 2), dtype=torch.float32)
    values[:, 0::2] = _E2M1[(codes & 0x0F).long()]
    values[:, 1::2] = _E2M1[(codes >> 4).long()]
    block_scale = scales.to(torch.float32).repeat_interleave(16, dim=1)
    return (values * block_scale / global_scale).to(torch.bfloat16)


def _fp8_words(reader: ShardReader, logical: str) -> tuple[torch.Tensor, torch.Tensor]:
    """An FP8 Linear's E4M3 code words [n, k] and one multiplier per row [n], as stored.

    A per-tensor multiplier is repeated for every row, which describes the same values.
    """

    codes = reader.get(logical)
    if codes.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{logical} is {codes.dtype}, expected float8_e4m3fn")
    scales = reader.get(logical + "_scale").reshape(-1)
    n = codes.shape[0]
    if scales.numel() == 1:
        scales = scales.expand(n)
    if scales.numel() != n:
        raise ValueError(f"{logical}_scale holds {scales.numel()} values, expected 1 or {n}")
    if not bool(torch.isfinite(scales.to(torch.float32)).all()) or bool((scales.to(torch.float32) < 0).any()):
        raise ValueError(f"{logical}_scale: multipliers must be finite and nonnegative")
    return codes.view(torch.uint8), scales


def _dequantize_fp8(reader: ShardReader, logical: str) -> torch.Tensor:
    """The values an FP8 Linear stands for, FP32: code * its row's multiplier.

    An E4M3 code (4 significant bits) times a BF16 or FP32 multiplier is exact in FP32.
    """

    codes, scales = _fp8_words(reader, logical)
    return codes.view(torch.float8_e4m3fn).to(torch.float32) * scales.to(torch.float32).unsqueeze(1)


class DequantizingReader:
    """A shard reader for the base recipes: an FP8 module's ``weight`` reads as its values.

    The base recipes read BF16 sources; the few that reach an FP8 module (the draft head,
    which gathers rows of the output head) get the FP32 values the codes stand for, so
    whatever format they encode into starts from the export's own numbers. Every other
    tensor passes through untouched.
    """

    def __init__(self, source: CompressedTensorsSource, reader: ShardReader) -> None:
        self._source = source
        self._reader = reader

    def has(self, name: str) -> bool:
        return self._reader.has(name)

    def get(self, name: str) -> torch.Tensor:
        if name.endswith(".weight") and self._source.fp8_format(name) is not None:
            return _dequantize_fp8(self._reader, name)
        return self._reader.get(name)

    def metadata(self, names):
        return self._reader.metadata(names)


def _fp32_word(tensor: torch.Tensor, what: str) -> bytes:
    value = float(tensor.reshape(()).to(torch.float32))
    if not (value > 0.0) or value != value:
        raise ValueError(f"{what} is {value}, expected finite positive")
    return struct.pack("<f", value)


def _text_core_suffix(name: str) -> str | None:
    """The per-layer or top-level text-core suffix, or None for other families."""

    parts = name.split("/", 3)
    if len(parts) == 4 and parts[:2] == ["text", "layers"] and parts[2].isdigit():
        return parts[3]
    if name in ("text/token_embedding", "text/output_head"):
        return name
    return None


__all__ = [
    "INPUT_DIVISOR_SUFFIX",
    "NESTED_TEXT_PREFIX",
    "SHARED_EXPERT_CHOICES",
    "SHARED_EXPERT_SUFFIXES",
    "SEGMENT_NAMES",
    "WEIGHTS_ID",
    "CompressedTensorsSource",
    "DequantizingReader",
    "ObjectSource",
    "SourcePlan",
]
