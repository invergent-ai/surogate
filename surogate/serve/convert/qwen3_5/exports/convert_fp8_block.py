"""Single-source conversion of a Hugging Face fine-grained FP8 export of this family.

Every linear ships as E4M3 `weight` [n, k] with an FP32 `weight_scale_inv` [n/128, k/128];
the norms, the convolution, the GDN control projections and the tied embedding stay BF16 in
the same file. The blocks pass through untouched -- the artifact stores the same codes and the
same scale grid -- and only the embedding is re-encoded byte-wide, as this family's head is.
The object graph is the NVFP4 uniform export's: fused parents whose components are read from
the checkpoint's separate tensors and stacked, which stacks their scale rows too, because a
component's rows are whole 128-blocks (a head is 256 rows).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.layouts import encode_fp8_block_scaled, encode_fp8_row_f32
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion

from .. import draft_head
from . import recipe_nvfp4_uniform as recipe
from .. import inventory as family_inventory
from .. import recipe as family_recipe
from .convert_nvfp4_uniform import geometry_block

RECIPE_ID = "qwen3_5-fp8-block-hf-v1"
OUTPUT_BASENAME = "qwen3_5_fp8_block.sinfer"
BLOCK = 128


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    object_plan: family_conversion.ObjectPlan
    resources: tuple
    draft: "draft_head.DraftHeadContext"
    geometry: family_inventory.Geometry
    export: family_inventory.ExportInventory
    recipes: recipe.UniformRecipes
    base_recipes: Mapping[str, object]


def _reader_factory(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.is_file() and not (model_dir / "model.safetensors.index.json").is_file():
        return lambda: ShardReader.from_file(single)
    return lambda: ShardReader(model_dir)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    source = Path(model_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"FP8 source directory not found: {source}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    geometry = family_inventory.geometry_from_config(config)
    from ..convert import profile_for_checkpoint
    profile = profile_for_checkpoint(config)
    if profile not in (family_inventory.FP8_BLOCK, family_inventory.FP8_CHANNEL):
        raise ValueError(f"not an FP8 export: {profile}")
    export = family_inventory.export_inventory(profile, geometry)
    with _reader_factory(source)() as reader:
        root = recipe.source_root_of(reader.names)
    recipes = recipe.build(geometry, root)
    fp8_format = family_inventory.FP8_BLOCK_FORMAT if profile == family_inventory.FP8_BLOCK else family_inventory.FP8_ROW_F32_FORMAT
    expected = {spec.name for spec in export.TEXT_CORE_TENSOR_SPECS if spec.format == fp8_format}
    produced = set(recipes.nvfp4_weights_by_name)
    if expected != produced:
        raise ValueError(f"FP8 block recipe coverage mismatch; missing={sorted(expected - produced)[:4]} "
                         f"extra={sorted(produced - expected)[:4]}")
    resources = family_conversion.load_resources(source, export.RESOURCE_SPECS)
    plan = family_conversion.build_object_plan(
        export.OBJECT_SPECS, {item.name: item.data for item in resources}
    )
    from ..convert import _tools_root
    draft = draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, source)
    base_recipes = {item.object_name: item for item in family_recipe.build_recipes(geometry)}
    return ConversionPreflight(source, plan, resources, draft, geometry, export, recipes, base_recipes)


def materialize_fp8_block_weight(entry: recipe.Nvfp4WeightRecipe, reader: ShardReader,
                                 per_row: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused object's codes [rows, k] and its scales -- a [rows/128, k/128] grid from
    `weight_scale_inv`, or one per row from a compressed-tensors `weight_scale` [rows, 1]
    (widened to FP32 exactly) -- stacked from its components in the order the parts name."""
    codes_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unit = 1 if per_row else BLOCK
    for part in entry.parts:
        words = cache.get(part.source.name)
        if words is None:
            n, k = part.source.shape
            packed = reader.get(part.source.name + ".weight")
            if packed.dtype != torch.float8_e4m3fn or tuple(packed.shape) != (n, k):
                raise ValueError(f"{part.source.name}: expected E4M3 weight {(n, k)}, got {packed.dtype} {tuple(packed.shape)}")
            if per_row:
                scales = reader.get(part.source.name + ".weight_scale").reshape(-1).to(torch.float32)
                if scales.numel() != n:
                    raise ValueError(f"{part.source.name}: expected one weight_scale per row, got {scales.numel()} for {n}")
                scales = scales.reshape(n, 1)
            else:
                scales = reader.get(part.source.name + ".weight_scale_inv")
                if scales.dtype != torch.float32 or tuple(scales.shape) != (n // BLOCK, k // BLOCK):
                    raise ValueError(f"{part.source.name}: expected FP32 weight_scale_inv {(n // BLOCK, k // BLOCK)}")
            words = (packed.view(torch.uint8), scales)
            cache[part.source.name] = words
        for rows in part.rows:
            if rows.begin % unit or rows.end % unit:
                raise ValueError(f"{entry.object_name}: row range {rows.begin}:{rows.end} is not whole scale rows")
            codes_parts.append(words[0][rows.begin:rows.end])
            scale_parts.append(words[1][rows.begin // unit:rows.end // unit])
    codes = torch.cat(codes_parts, dim=0) if len(codes_parts) > 1 else codes_parts[0].contiguous()
    scales = torch.cat(scale_parts, dim=0) if len(scale_parts) > 1 else scale_parts[0].contiguous()
    expect = (entry.shape[0], 1) if per_row else (entry.shape[0] // BLOCK, entry.shape[1] // BLOCK)
    if tuple(codes.shape) != tuple(entry.shape) or tuple(scales.shape) != expect:
        raise ValueError(f"{entry.object_name}: shape mismatch after fusion")
    return codes, scales


def _dense_control_projection(stem: str, reader: ShardReader, per_row: bool) -> torch.Tensor:
    """`in_proj_a` / `in_proj_b` as BF16: read dense when the export left them so, dequantised
    from their E4M3 codes and scale (per row, or per 128x128 block) when it did not."""
    weight = reader.get(stem + ".weight")
    if weight.dtype != torch.float8_e4m3fn:
        return weight
    codes = weight.float()
    if per_row:
        scale = reader.get(stem + ".weight_scale").float().reshape(-1, 1)
        return (codes * scale).to(torch.bfloat16)
    scale = reader.get(stem + ".weight_scale_inv").float()
    n, k = codes.shape
    grid = scale.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)[:, :k]
    return (codes * grid).to(torch.bfloat16)


def convert(model_dir: str | Path, out_path: str | Path, *, device: str | torch.device = "cuda") -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir)
    export, recipes = preflight.export, preflight.recipes
    per_row = export.profile == family_inventory.FP8_CHANNEL
    fp8_by_name = recipes.nvfp4_weights_by_name
    direct_by_name = recipes.direct_by_name
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipes.nvfp4_sources)} FP8 source matrices, hidden={preflight.geometry.hidden} "
        f"layers={preflight.geometry.layers} root={recipes.source_root!r}, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(export.OBJECT_SPECS)
    with _reader_factory(preflight.model_dir)() as reader:
        with ArtifactWriter(
            output, ArtifactIdentity(export.MODEL_ID, export.WEIGHTS_ID),
            preflight.object_plan.specs, geometry=geometry_block(model_dir),
        ) as writer:
            for index, spec in enumerate(export.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, export.ResourceSpec):
                    if spec.name not in resources:
                        continue # an optional resource the checkpoint does not carry; not in the plan either
                    payload = resources[spec.name]
                elif spec.name in ("text/token_embedding", "text/output_head"):
                    tensor = reader.get(recipes.embedding_source)
                    payload = family_conversion.encode_tensor_payload(tensor.reshape(spec.shape), spec, resolved_device)
                    del tensor
                elif spec.name in fp8_by_name:
                    codes, scales = materialize_fp8_block_weight(fp8_by_name[spec.name], reader, per_row)
                    payload = (encode_fp8_row_f32(codes, scales, spec.shape) if per_row
                               else encode_fp8_block_scaled(codes, scales, spec.shape))
                elif spec.name in direct_by_name:
                    entry = direct_by_name[spec.name]
                    # The GDN control projections: dense in some exports, FP8 with the same scale
                    # kind as the big matrices in others; the artifact wants them dense either way.
                    tensor = (_dense_control_projection(entry.source_name, reader, per_row)
                              if entry.dequantize else reader.get(entry.source_name))
                    dtype = {export.FP32: torch.float32, export.BF16: torch.bfloat16, export.I32: torch.int32}.get(spec.format)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    if entry.source_shape is not None:
                        tensor = tensor.reshape(entry.source_shape)
                    if entry.transpose:
                        tensor = tensor.transpose(0, 1).contiguous()
                    payload = family_conversion.encode_tensor_payload(tensor.reshape(spec.shape), spec, resolved_device)
                    del tensor
                elif spec.name in (draft_head.DRAFT_HEAD_OBJECT, draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT):
                    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_head.materialize_draft_head_token_ids(preflight.draft)}
                    tensor = family_recipe.materialize_recipe(preflight.base_recipes[spec.name], reader, derived)
                    payload = family_conversion.encode_tensor_payload(tensor.reshape(spec.shape), spec, resolved_device)
                    del tensor
                else:
                    raise KeyError(f"no recipe covers artifact object {spec.name!r}")
                writer.write(spec.name, payload)
                del payload
                if index % 25 == 0 or index == total:
                    print(f"[{index}/{total}] {spec.name}", flush=True)
    print(f"conversion finished in {time.perf_counter() - started:.1f} s -> {output}", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FP8 export directory")
    parser.add_argument("--out", required=True, help="artifact output path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
