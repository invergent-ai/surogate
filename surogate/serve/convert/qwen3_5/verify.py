"""Structural and representative-source verification for one `.sinfer` artifact.

One converter serves every size of the family, so every check here is made against the
geometry of the source checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open
import torch

from surogate.serve.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from surogate.serve.artifact.layouts import (
    align_up,
    decode_direct,
    decode_row_split_codes,
    encoded_size,
    gather_row_planes,
    row_split_geometry,
)
from surogate.serve.artifact.numeric import QuantFormat, get_format
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion, qwen3_5 as checkpoint

from . import draft_head, inventory, recipe


TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"

DIRECT_PROBE_OBJECTS = (
    "text/layers/0/input_norm",
    "text/layers/0/gdn/a_log",
    "text/layers/0/gdn/convolution",
)


def quant_probe_objects(g: inventory.Geometry) -> tuple[str, ...]:
    names = []
    if g.full_attention_layers:
        names.append(f"text/layers/{g.full_attention_layers[0]}/attention/query_key_gate_value")
    if g.gdn_layers:
        names.append(f"text/layers/{g.gdn_layers[0]}/gdn/query_key_value_z")
    return tuple(names) + ("vision/patch_embedding", "mtp/layer/attention/query_key_gate_value", "text/draft_head")


_FP16_MIN_SUBNORMAL = 2.0**-24


class VerificationError(ValueError):
    """The artifact does not match the target contract at its declared size."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int
    row_view_templates: int
    row_view_bindings: int
    alias_templates: int
    alias_bindings: int


@dataclass(frozen=True, slots=True)
class PayloadSummary:
    direct_probes: int
    quant_probes: int
    quant_rows: int
    quant_groups: int
    draft_rows: int
    resources: int
    processor_class: str
    generation_config_class: str


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    structure: StructureSummary
    payload: PayloadSummary


def _contract_error(message: str) -> None:
    raise VerificationError(message)


def _object_index(objects: Sequence[ResourceObject | TensorObject]):
    return {obj.name: obj for obj in objects}


def validate_logical_bindings(
    objects: Sequence[ResourceObject | TensorObject],
    geometry: inventory.Geometry,
) -> tuple[int, int]:
    """Validate every fixed row view and alias against bound physical objects."""

    index = _object_index(objects)
    row_bindings = 0
    for view in inventory.build_logical_row_views(geometry):
        layers: tuple[int | None, ...]
        layers = (None,) if view.layers is None else view.layers
        for layer in layers:
            parent_name = (
                view.parent_pattern
                if layer is None
                else view.parent_pattern.format(l=layer)
            )
            parent = index.get(parent_name)
            if not isinstance(parent, TensorObject):
                _contract_error(f"logical view parent is missing: {parent_name}")
            if len(parent.shape) != 2:
                _contract_error(f"logical view parent is not a matrix: {parent_name}")
            if view.row_end > parent.shape[0]:
                _contract_error(f"logical view exceeds parent rows: {view.name_pattern}")
            if view.shape != (view.row_end - view.row_begin, parent.shape[1]):
                _contract_error(f"logical view shape is inconsistent: {view.name_pattern}")
            row_bindings += 1

    alias_bindings = 0
    convolution_shape = (geometry.convolution_dim, geometry.gdn_conv_kernel)
    for alias in inventory.build_alias_specs(geometry):
        layers = (None,) if alias.layers is None else alias.layers
        for layer in layers:
            names = tuple(
                pattern if layer is None else pattern.format(l=layer)
                for pattern in alias.object_patterns
            )
            bound = [index.get(name) for name in names]
            if geometry.tied_embeddings:
                bound = [index.get("text/token_embedding") if name == "text/output_head" and obj is None else obj
                         for name, obj in zip(names, bound)]
            if any(obj is None for obj in bound):
                _contract_error(f"logical alias has a missing object: {alias.role_pattern}")
            if alias.axis_order is not None:
                if len(bound) != 1 or not isinstance(bound[0], TensorObject):
                    _contract_error(f"axis alias does not bind one tensor: {alias.role_pattern}")
                source_shape = bound[0].shape
                if tuple(sorted(alias.axis_order)) != tuple(range(len(source_shape))):
                    _contract_error(f"axis alias is invalid: {alias.role_pattern}")
                target_shape = tuple(source_shape[axis] for axis in alias.axis_order)
                if target_shape != convolution_shape:
                    _contract_error(f"GDN convolution alias has shape {target_shape}")
            alias_bindings += 1

    return row_bindings, alias_bindings


def validate_structure(
    artifact: Artifact,
    geometry: inventory.Geometry,
    *,
    mtp: bool = True,
    vision: bool = True,
) -> StructureSummary:
    """Validate the complete directory without reading tensor payload values."""

    _, object_specs = inventory.active_specs(mtp=mtp, vision=vision, geometry=geometry)
    present = _object_index(artifact.objects)
    required_resources = {"frontend/tokenizer.json", "frontend/tokenizer_config.json", "frontend/generation_config.json"}
    object_specs = tuple(s for s in object_specs
                         if not (isinstance(s, inventory.ResourceSpec) and s.name not in required_resources and s.name not in present)
                         and not (s.name == "text/output_head" and geometry.tied_embeddings and s.name not in present))
    expected_identity = ArtifactIdentity(
        inventory.model_id_for(geometry), inventory.WEIGHTS_ID
    , architecture="qwen3_5")
    if (artifact.identity.architecture, artifact.identity.weights_id) != (
        expected_identity.architecture, expected_identity.weights_id
    ):
        _contract_error(
            f"artifact identity is {artifact.identity!r}, expected "
            f"{expected_identity!r}"
        )
    if len(artifact.objects) != len(object_specs):
        _contract_error(
            f"artifact has {len(artifact.objects)} objects, expected "
            f"{len(object_specs)}"
        )

    cursor = 0
    tensor_count = 0
    resource_count = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, object_specs)
    ):
        if actual.name != expected.name:
            _contract_error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _contract_error(
                f"{actual.name}: offset {actual.offset}, expected {expected_offset}"
            )

        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _contract_error(f"{actual.name}: expected a tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _contract_error(
                    f"{actual.name}: signature {signature} does not match {registered}"
                )
            required_bytes = encoded_size(actual.layout, actual.format, actual.shape)
            if actual.bytes != required_bytes:
                _contract_error(
                    f"{actual.name}: stores {actual.bytes} bytes, expected {required_bytes}"
                )
            tensor_count += 1
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _contract_error(f"{actual.name}: expected a resource descriptor")
            if actual.encoding != expected.encoding:
                _contract_error(
                    f"{actual.name}: encoding {actual.encoding!r}, expected {expected.encoding!r}"
                )
            resource_count += 1

        cursor = actual.offset + actual.bytes

    expected_formats = Counter(spec.format for spec in object_specs
                               if isinstance(spec, inventory.TensorSpec))
    if formats != expected_formats:
        _contract_error(f"numeric-format counts are {dict(formats)}")
    expected_layouts = Counter(spec.layout for spec in object_specs
                               if isinstance(spec, inventory.TensorSpec))
    if layouts != expected_layouts:
        _contract_error(f"layout counts are {dict(layouts)}")

    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _contract_error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")

    row_bindings, alias_bindings = validate_logical_bindings(artifact.objects, geometry)
    return StructureSummary(
        objects=len(artifact.objects),
        tensors=tensor_count,
        resources=resource_count,
        payload_bytes=payload_bytes,
        row_view_templates=len(inventory.build_logical_row_views(geometry)),
        row_view_bindings=row_bindings,
        alias_templates=len(inventory.build_alias_specs(geometry)),
        alias_bindings=alias_bindings,
    )


def _three_indices(count: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, count // 2, count - 1)))


def _logical_words(tensor: torch.Tensor, format_name: str) -> torch.Tensor:
    contiguous = tensor.detach().contiguous().cpu()
    if format_name == inventory.BF16:
        return contiguous.view(torch.int16)
    if format_name in (inventory.FP32, inventory.I32):
        return contiguous.view(torch.int32)
    raise TypeError(f"{format_name} is not a direct format")


def _verify_direct_probe(
    artifact: Artifact,
    source_reader: ShardReader,
    object_name: str,
    recipes: Mapping[str, recipe.TensorRecipe],
) -> None:
    obj = artifact.find(object_name)
    if not isinstance(obj, TensorObject):
        _contract_error(f"{object_name} is not a tensor")
    expected = recipe.materialize_recipe(recipes[object_name], source_reader)
    stored = decode_direct(artifact.payload(obj), obj.format, obj.shape)
    expected_words = _logical_words(expected, obj.format).reshape(-1)
    stored_words = _logical_words(stored, obj.format).reshape(-1)
    indices = torch.tensor(_three_indices(stored_words.numel()), dtype=torch.long)
    if not torch.equal(
        stored_words.index_select(0, indices),
        expected_words.index_select(0, indices),
    ):
        _contract_error(f"{object_name}: representative direct words differ")


class _SourceSlices:
    """Read only selected first-axis rows from recipe source tensors."""

    def __init__(self, reader: ShardReader) -> None:
        self.reader = reader
        self.model_dir = reader.model_dir
        self.weight_map = reader.weight_map

    def rows(self, source: recipe.SourceTensor, rows: Sequence[int]) -> torch.Tensor:
        resolved = self.reader._resolve(source.name)
        shard = self.weight_map[resolved]
        with safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        ) as handle:
            tensor_slice = handle.get_slice(self.reader._stored_name(resolved))
            pieces = [tensor_slice[row : row + 1] for row in rows]
        return torch.cat(pieces, dim=0)


def _qproj_rows(
    expression: recipe.Reshape,
    rows: Sequence[int],
    sources: _SourceSlices,
) -> torch.Tensor | None:
    selected = expression.source
    if not isinstance(selected, recipe.Slice) or selected.axis != 1:
        return None
    per_head = selected.source
    if not isinstance(per_head, recipe.Reshape):
        return None
    source = per_head.source
    if not isinstance(source, recipe.SourceTensor) or len(source.shape) != 2:
        return None
    if len(per_head.shape) != 3 or per_head.shape[-1] != source.shape[-1]:
        return None
    part_rows = selected.end - selected.begin
    if expression.shape != (per_head.shape[0] * part_rows, source.shape[-1]):
        return None
    source_rows = [
        (row // part_rows) * per_head.shape[1]
        + selected.begin
        + (row % part_rows)
        for row in rows
    ]
    return sources.rows(source, source_rows)


def _materialize_rows(
    expression: recipe.Expression,
    rows: Sequence[int],
    sources: _SourceSlices,
    draft_ids: torch.Tensor,
) -> torch.Tensor:
    """Materialize selected rows from a rank-two recipe result."""

    shape = recipe.expression_shape(expression)
    if len(shape) != 2:
        raise TypeError(f"row probes require a matrix expression, got {shape}")

    from surogate.serve.convert.common.recipe import resolve_options
    resolved = resolve_options(expression, sources.reader.has)
    if resolved != expression:
        return _materialize_rows(resolved, rows, sources, draft_ids)
    if isinstance(expression, recipe.SourceTensor):
        return sources.rows(expression, rows)

    if isinstance(expression, recipe.Slice):
        if expression.axis != 0:
            raise TypeError("only leading-axis slices can be probed directly")
        return _materialize_rows(
            expression.source,
            [expression.begin + row for row in rows],
            sources,
            draft_ids,
        )

    if isinstance(expression, recipe.Reshape):
        qproj = _qproj_rows(expression, rows, sources)
        if qproj is not None:
            return qproj
        if isinstance(expression.source, recipe.SourceTensor):
            source = expression.source
            if source.shape[0] == expression.shape[0]:
                selected = sources.rows(source, rows)
                return selected.reshape(len(rows), expression.shape[1])
        raise TypeError(f"unsupported row-wise reshape probe: {expression}")

    if isinstance(expression, recipe.Concat):
        if expression.axis != 0:
            raise TypeError("row probes require leading-axis concatenation")
        boundaries: list[tuple[int, int, recipe.Expression]] = []
        begin = 0
        for part in expression.sources:
            part_rows = recipe.expression_shape(part)[0]
            boundaries.append((begin, begin + part_rows, part))
            begin += part_rows
        selected_rows = []
        for row in rows:
            for part_begin, part_end, part in boundaries:
                if part_begin <= row < part_end:
                    selected_rows.append(
                        _materialize_rows(
                            part,
                            [row - part_begin],
                            sources,
                            draft_ids,
                        )
                    )
                    break
            else:
                raise IndexError(f"row {row} is outside concatenated shape {shape}")
        return torch.cat(selected_rows, dim=0)

    if isinstance(expression, recipe.Cast):
        return _materialize_rows(
            expression.source, rows, sources, draft_ids
        ).to(torch.float32)

    if isinstance(expression, recipe.GatherRows):
        source_rows = [int(draft_ids[row]) for row in rows]
        return sources.rows(expression.source, source_rows)

    raise TypeError(f"unsupported matrix row probe: {type(expression).__name__}")


def _profile_quantize_rows(
    source_rows: torch.Tensor,
    format_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independent host oracle to selected logical rows."""

    format_spec = get_format(format_name)
    if not isinstance(format_spec, QuantFormat):
        raise TypeError(f"{format_name} is not quantized")
    geometry = row_split_geometry(format_spec, source_rows.shape)
    values = source_rows.detach().cpu().float().numpy()
    if not np.isfinite(values).all():
        _contract_error("quantized source probe contains NaN or infinity")
    if geometry.k_pad != geometry.k:
        padded = np.zeros(
            (geometry.n, geometry.k_pad),
            dtype=np.float32,
        )
        padded[:, : geometry.k] = values
        values = padded
    grouped = values.reshape(
        geometry.n,
        geometry.groups_per_row,
        format_spec.group_size,
    )
    max_abs = np.max(np.abs(grouped), axis=-1)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_scale = (
            max_abs.astype(np.float64) / float(format_spec.qmax)
        ).astype(np.float32)
        scales = raw_scale.astype(np.float16)
    underflow = (scales == 0) & (max_abs > 0)
    if underflow.any():
        scales = scales.copy()
        scales[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((max_abs > 0) & (~np.isfinite(scales) | (scales <= 0))):
        _contract_error("quantized source probe has an invalid binary16 scale")
    reciprocal = np.zeros(scales.shape, dtype=np.float32)
    positive = scales > 0
    reciprocal[positive] = (
        1.0 / scales[positive].astype(np.float64)
    ).astype(np.float32)
    # Products of two binary32 values are exact in binary64. Casting explicitly
    # supplies the specified binary32 rounding before integral ties-to-even.
    normalized = (
        grouped.astype(np.float64) * reciprocal.astype(np.float64)[..., None]
    ).astype(np.float32)
    codes = np.clip(
        np.rint(normalized),
        format_spec.qmin,
        format_spec.qmax,
    ).astype(np.int8)
    return torch.from_numpy(scales).to(device), torch.from_numpy(codes).to(device)


def verify_quantized_rows(
    payload: bytes | bytearray | memoryview,
    format_name: str,
    shape: tuple[int, int],
    row_indices: Sequence[int],
    source_rows: torch.Tensor,
    device: str | torch.device = "cpu",
) -> int:
    """Compare representative stored groups with profile results for source rows."""

    geometry = row_split_geometry(format_name, shape)
    gathered = gather_row_planes(payload, geometry, row_indices)
    target = torch.device(device)
    stored_scales, stored_codes = decode_row_split_codes(
        gathered,
        format_name,
        (len(row_indices), shape[1]),
        device=target,
    )
    expected_scales, expected_codes = _profile_quantize_rows(
        source_rows,
        format_name,
        target,
    )
    group_indices = torch.tensor(
        _three_indices(geometry.groups_per_row),
        dtype=torch.long,
        device=target,
    )
    if not torch.equal(
        stored_scales.index_select(1, group_indices).view(torch.int16),
        expected_scales.index_select(1, group_indices).view(torch.int16),
    ):
        _contract_error("representative stored quantization scales differ")
    if not torch.equal(
        stored_codes.index_select(1, group_indices),
        expected_codes.index_select(1, group_indices),
    ):
        _contract_error("representative stored quantization codes differ")
    return len(row_indices) * len(group_indices)


def validate_draft_token_ids(token_ids: torch.Tensor, geometry: inventory.Geometry) -> None:
    if token_ids.dtype != torch.int32 or tuple(token_ids.shape) != (geometry.draft_vocab,):
        _contract_error("draft token IDs disagree with the resolved shortlist shape")
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= geometry.token_domain:
        _contract_error("draft token IDs are outside the tokenizer domain")
    if torch.unique(token_ids).numel() != geometry.draft_vocab:
        _contract_error("draft token IDs are not unique")


def _load_and_validate_draft_ids(
    artifact: Artifact,
    model_dir: Path,
    geometry: inventory.Geometry,
) -> torch.Tensor:
    obj = artifact.find(draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT)
    if not isinstance(obj, TensorObject):
        _contract_error("draft token ID object is not a tensor")
    token_ids = decode_direct(artifact.payload(obj), obj.format, obj.shape)
    validate_draft_token_ids(token_ids, geometry)

    recipes = {r.object_name: r for r in recipe.build_recipes(geometry)}
    draft_expression = recipes[draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT].expression
    if not isinstance(draft_expression, recipe.DraftHeadTokenIds):
        _contract_error("draft ID recipe is not the registered derivation")
    context = draft_head.compute_shortlist(
        TOOLS_ROOT / draft_expression.ranking_path,
        model_dir,
        geometry=geometry,
    )
    expected = draft_head.materialize_draft_head_token_ids(context)
    if not torch.equal(token_ids, expected):
        _contract_error("stored draft token IDs differ from the registered shortlist")
    return token_ids


def _verify_resources_and_frontend(
    artifact: Artifact,
    model_dir: Path,
) -> tuple[str, str]:
    payloads: dict[str, bytes] = {}
    for resource in conversion.load_resources(model_dir, inventory.RESOURCE_SPECS):
        obj = artifact.find(resource.name)
        if not isinstance(obj, ResourceObject):
            _contract_error(f"{resource.name} is not a resource")
        payload = bytes(artifact.payload(obj))
        filename = resource.name.removeprefix("frontend/")
        source = resource.data
        if payload != source:
            _contract_error(f"frontend resource differs from source: {resource.name}")
        payloads[filename] = payload

    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

    with tempfile.TemporaryDirectory(prefix="sinfer-frontend-") as temporary:
        directory = Path(temporary)
        for filename, payload in payloads.items():
            (directory / filename).write_bytes(payload)
        factory = AutoProcessor if "preprocessor_config.json" in payloads else AutoTokenizer
        processor = factory.from_pretrained(directory, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(
            directory,
            local_files_only=True,
        )
        if factory is AutoProcessor and getattr(processor, "tokenizer", None) is None:
            _contract_error("AutoProcessor did not construct its tokenizer")
        return type(processor).__name__, type(generation_config).__name__


def _probe_rows(object_name: str, rows: int,
                geometry: inventory.Geometry) -> tuple[int, ...]:
    """The rows a probe reads: the ends and middle, except across a fused GDN projection.

    Its halves are the one place a mistake moves whole row blocks around rather than changing
    values, so it is probed at the seam the two meet at instead.
    """
    seam = {
        "gdn/query_key_value_z": 2 * geometry.key_dim,
        "gdn/value_z": geometry.value_dim,
    }.get(object_name.split("/", 3)[-1])
    if seam is None:
        return _three_indices(rows)
    return (0, seam - 1, seam, rows - 1)


def verify_payloads(
    artifact: Artifact,
    model_dir: str | Path,
    device: str | torch.device,
    geometry: inventory.Geometry,
) -> PayloadSummary:
    """Verify representative values without requantizing complete matrices."""

    source_dir = Path(model_dir)
    recipes = {item.object_name: item for item in recipe.build_recipes(geometry)}
    draft_ids = _load_and_validate_draft_ids(artifact, source_dir, geometry)
    target = torch.device(device)

    # A text-only or no-MTP export carries no object for some probes; the artifact states
    # which capabilities it has and the probe list follows it rather than assuming.
    present = {obj.name for obj in artifact.objects}
    quant_probes = tuple(
        name for name in quant_probe_objects(geometry) if name in present
    )
    quant_groups = 0
    quant_rows = 0
    direct_probes = ["text/layers/0/input_norm"]
    if geometry.gdn_layers:
        direct_probes.extend(f"text/layers/{geometry.gdn_layers[0]}/gdn/{role}" for role in ("a_log", "convolution"))
    with ShardReader.for_directory(source_dir) as source_reader:
        from surogate.serve.convert.common.recipe import preflight_source_reader
        preflight_source_reader(source_reader, tuple(recipes.values()))
        for object_name in direct_probes:
            _verify_direct_probe(artifact, source_reader, object_name, recipes)

        source_slices = _SourceSlices(source_reader)
        for object_name in quant_probes:
            obj = artifact.find(object_name)
            if not isinstance(obj, TensorObject) or len(obj.shape) != 2:
                _contract_error(f"quantized probe is not a matrix: {object_name}")
            rows = _probe_rows(object_name, obj.shape[0], geometry)
            source_rows = _materialize_rows(
                recipes[object_name].expression,
                rows,
                source_slices,
                draft_ids,
            )
            quant_groups += verify_quantized_rows(
                artifact.payload(obj),
                obj.format,
                obj.shape,
                rows,
                source_rows,
                target,
            )
            quant_rows += len(rows)

    processor_class, generation_config_class = _verify_resources_and_frontend(
        artifact,
        source_dir,
    )
    return PayloadSummary(
        direct_probes=len(direct_probes),
        quant_probes=len(quant_probes),
        quant_rows=quant_rows,
        quant_groups=quant_groups,
        draft_rows=draft_ids.numel(),
        resources=sum(isinstance(obj, ResourceObject) for obj in artifact.objects),
        processor_class=processor_class,
        generation_config_class=generation_config_class,
    )


def verify_artifact(
    artifact: Artifact,
    model_dir: str | Path,
    device: str | torch.device = "cpu",
    geometry: inventory.Geometry | None = None,
) -> VerificationSummary:
    # The source checkpoint states the size, and the artifact must be the artifact of that
    # checkpoint, so its own config is what the contract is read at.
    if geometry is None:
        from .exports.quantized import Sources, _geometry
        with ShardReader.for_directory(model_dir) as source:
            geometry = _geometry(conversion.load_json(Path(model_dir) / "config.json"), model_dir, Sources(source),
                                 mtp=bool(artifact.geometry["mtp_layers"]), vision=bool(artifact.vision_geometry))
    if artifact.geometry != checkpoint.geometry_block(geometry) or tuple(artifact.layer_types) != geometry.layer_types:
        _contract_error("artifact geometry does not match the source checkpoint")
    if any(getattr(obj, "runs", ()) for obj in artifact.objects):
        _contract_error("native GGUF artifacts require verification against their original GGUF source")
    structure = validate_structure(artifact, geometry, mtp=bool(geometry.mtp_layers),
                                   vision=bool(inventory.vision_tower(geometry)))
    payload = verify_payloads(artifact, model_dir, device, geometry)
    return VerificationSummary(structure=structure, payload=payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an interleaved gated-delta SInfer artifact against its "
                    "source checkpoint"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with Artifact.open(arguments.artifact) as artifact:
        summary = verify_artifact(artifact, arguments.model, arguments.device)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
