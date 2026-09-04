"""Single-source conversion of a ModelOpt NVFP4 export of this family into an artifact.

Everything comes from one ModelOpt checkpoint: the NVFP4 blocks pass through
untouched with their calibrated activation divisors, the norms and the
convolution are copied dense, and the tied embedding is re-encoded FP8
row-scaled for both the embedding table and the output head. The dimensions
and the tensor-name root are the checkpoint's own, so the 4B, 2B and 0.8B
releases convert through the same program.
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
from surogate.serve.artifact.layouts import encode_direct, encode_nvfp4, swizzle_nvfp4_scales
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion

from .. import draft_head
from . import recipe_nvfp4_uniform as recipe
from .. import inventory as family_inventory
from .. import recipe as family_recipe

RECIPE_ID = "qwen3_5-nvfp4-modelopt-v1"
OUTPUT_BASENAME = "qwen3_5_nvfp4.sinfer"


def geometry_block(model_dir) -> dict[str, float]:
    """The dimensions the engine binds against, read from the checkpoint that is being
    converted rather than restated here. The family's one target reads them from the
    artifact instead of compiling them."""
    import json as _json
    from pathlib import Path as _Path

    config = _json.loads((_Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", config)
    rope = text.get("rope_parameters") or {}
    return {
        "hidden": int(text["hidden_size"]),
        "layers": int(text["num_hidden_layers"]),
        "intermediate": int(text["intermediate_size"]),
        "output_rows": int(text["vocab_size"]),
        "query_heads": int(text["num_attention_heads"]),
        "kv_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "gdn_key_heads": int(text["linear_num_key_heads"]),
        "gdn_key_head_dim": int(text["linear_key_head_dim"]),
        "gdn_value_heads": int(text["linear_num_value_heads"]),
        "gdn_value_head_dim": int(text["linear_value_head_dim"]),
        "gdn_conv_kernel": int(text["linear_conv_kernel_dim"]),
        "mtp_layers": int(text["mtp_num_hidden_layers"]),
        "rms_epsilon": float(text["rms_norm_eps"]),
        "rope_theta": float(rope.get("rope_theta", text.get("rope_theta", 1.0e7))),
    }

@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    object_plan: family_conversion.ObjectPlan
    resources: tuple
    draft: "draft_head.DraftHeadContext"
    geometry: family_inventory.Geometry
    export: family_inventory.ExportInventory
    recipes: recipe.UniformRecipes
    #: The BF16 checkpoint's recipes at this size, for the objects this export does not
    #: quantise (the draft head).
    base_recipes: Mapping[str, object]


# E2M1 code -> value, the 4-bit float NVFP4 stores (sign in the high bit).
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _decode_nvfp4_dense(
    source_name: str, shape: tuple[int, ...], reader: ShardReader
) -> torch.Tensor:
    """in_proj_a/in_proj_b ship quantised; the artifact wants them dense."""

    packed = reader.get(f"{source_name}.{recipe.WEIGHT_FIELD}")
    scales = reader.get(f"{source_name}.{recipe.SCALE_FIELD}")
    divisor = float(
        reader.get(f"{source_name}.{recipe.GLOBAL_SCALE_FIELD}").reshape(()).to(torch.float32)
    )
    rows, half = packed.shape
    low = (packed & 0x0F).to(torch.long)
    high = (packed >> 4).to(torch.long)
    values = torch.empty(rows, half * 2, dtype=torch.float32)
    table = _E2M1_VALUES.to(packed.device)
    values[:, 0::2] = table[low]
    values[:, 1::2] = table[high]
    block_scales = scales.to(torch.float32).repeat_interleave(16, dim=1)
    dense = values * block_scales * divisor
    if tuple(dense.shape) != tuple(shape):
        raise ValueError(f"{source_name}: dequantised shape {tuple(dense.shape)} != {tuple(shape)}")
    return dense.to(torch.bfloat16)


def _reader_factory(model_dir: Path):
    single = model_dir / "model.safetensors"
    if single.is_file() and not (model_dir / "model.safetensors.index.json").is_file():
        return lambda: ShardReader.from_file(single)
    return lambda: ShardReader(model_dir)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    source = Path(model_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"NVFP4 source directory not found: {source}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    geometry = family_inventory.geometry_from_config(config)
    export = family_inventory.export_inventory(family_inventory.NVFP4_UNIFORM, geometry)
    with _reader_factory(source)() as reader:
        root = recipe.source_root_of(reader.names)
    recipes = recipe.build(geometry, root)
    recipe.validate(recipes, export)
    resources = family_conversion.load_resources(source, export.RESOURCE_SPECS)
    plan = family_conversion.build_object_plan(
        export.OBJECT_SPECS, {item.name: item.data for item in resources}
    )
    from ..convert import _tools_root  # the ranking the draft head's shortlist is cut from
    draft = draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, source)
    base_recipes = {item.object_name: item for item in family_recipe.build_recipes(geometry)}
    return ConversionPreflight(source, plan, resources, draft, geometry, export, recipes,
                               base_recipes)


def _encode_nvfp4_object(spec, entry: recipe.Nvfp4WeightRecipe, reader: ShardReader) -> bytes:
    packed, scales, divisor = recipe.materialize_nvfp4_weight(entry, reader)
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir)
    export, recipes = preflight.export, preflight.recipes
    nvfp4_by_name = recipes.nvfp4_weights_by_name
    divisors_by_name = recipes.input_divisors_by_name
    direct_by_name = recipes.direct_by_name
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipes.nvfp4_sources)} NVFP4 source matrices, "
        f"hidden={preflight.geometry.hidden} layers={preflight.geometry.layers} "
        f"root={recipes.source_root!r}, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(export.OBJECT_SPECS)
    with _reader_factory(preflight.model_dir)() as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(export.MODEL_ID, export.WEIGHTS_ID),
            preflight.object_plan.specs,
            geometry=geometry_block(model_dir),
        ) as writer:
            for index, spec in enumerate(export.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, export.ResourceSpec):
                    if spec.name not in resources:
                        continue # an optional resource the checkpoint does not carry; not in the plan either
                    payload = resources[spec.name]
                elif spec.name in ("text/token_embedding", "text/output_head"):
                    # The embedding is tied, so the head reads the same source.
                    tensor = reader.get(recipes.embedding_source)
                    payload = family_conversion.encode_tensor_payload(
                        tensor.reshape(spec.shape), spec, resolved_device
                    )
                    del tensor
                elif spec.name in nvfp4_by_name:
                    payload = _encode_nvfp4_object(spec, nvfp4_by_name[spec.name], reader)
                elif spec.name in divisors_by_name:
                    payload = recipe.materialize_input_divisor(divisors_by_name[spec.name], reader)
                elif spec.name in direct_by_name:
                    entry = direct_by_name[spec.name]
                    tensor = (
                        _decode_nvfp4_dense(entry.source_name, entry.shape, reader)
                        if entry.dequantize
                        else reader.get(entry.source_name)
                    )
                    # The export stores some scalars in bf16 where the artifact wants fp32.
                    dtype = {
                        export.FP32: torch.float32,
                        export.BF16: torch.bfloat16,
                        export.I32: torch.int32,
                    }.get(spec.format)
                    if dtype is not None and tensor.dtype != dtype:
                        tensor = tensor.to(dtype)
                    if entry.source_shape is not None:
                        tensor = tensor.reshape(entry.source_shape)
                    if entry.transpose:
                        tensor = tensor.transpose(0, 1).contiguous()
                    payload = family_conversion.encode_tensor_payload(
                        tensor.reshape(spec.shape), spec, resolved_device
                    )
                    del tensor
                elif spec.name in (
                    draft_head.DRAFT_HEAD_OBJECT,
                    draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT,
                ):
                    derived = {
                        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
                            draft_head.materialize_draft_head_token_ids(preflight.draft)
                        )
                    }
                    tensor = family_recipe.materialize_recipe(
                        preflight.base_recipes[spec.name], reader, derived
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor.reshape(spec.shape), spec, resolved_device
                    )
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
    parser.add_argument("--model", required=True, help="NVFP4 export directory")
    parser.add_argument("--out", required=True, help="artifact output path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
