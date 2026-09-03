"""Single-source conversion of the Qwen3.5-4B NVFP4 export into an artifact.

Everything comes from one ModelOpt checkpoint: the NVFP4 blocks pass through
untouched with their calibrated activation divisors, the norms and the
convolution are copied dense, and the tied embedding is re-encoded FP8
row-scaled for both the embedding table and the output head.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.layouts import encode_direct, encode_nvfp4, swizzle_nvfp4_scales
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion

from . import draft_head
from . import inventory_nvfp4 as inventory
from . import recipe as base_recipe
from . import recipe_nvfp4 as recipe


RECIPE_ID = "qwen3_5_4b-nvfp4-modelopt-v1"
OUTPUT_BASENAME = "qwen3_5_4b_nvfp4.sinfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    object_plan: family_conversion.ObjectPlan
    resources: tuple
    draft: "draft_head.DraftHeadContext" 


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


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    source = Path(model_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"NVFP4 source directory not found: {source}")
    recipe.validate_recipe()
    resources = family_conversion.load_resources(source, inventory.RESOURCE_SPECS)
    plan = family_conversion.build_object_plan(
        inventory.OBJECT_SPECS, {item.name: item.data for item in resources}
    )
    ranking = Path(__file__).resolve().parents[2] / "tools" / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, source)
    return ConversionPreflight(source, plan, resources, draft)


def _encode_nvfp4_object(spec, reader: ShardReader) -> bytes:
    entry = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
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
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source matrices, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(inventory.OBJECT_SPECS)
    single = preflight.model_dir / "model.safetensors"
    reader_factory = (
        (lambda: ShardReader.from_file(single))
        if single.is_file() and not (preflight.model_dir / "model.safetensors.index.json").is_file()
        else (lambda: ShardReader(preflight.model_dir))
    )
    with reader_factory() as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in ("text/token_embedding", "text/output_head"):
                    # The embedding is tied, so the head reads the same source.
                    tensor = reader.get(recipe.EMBEDDING_SOURCE)
                    payload = family_conversion.encode_tensor_payload(
                        tensor.reshape(spec.shape), spec, resolved_device
                    )
                    del tensor
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_nvfp4_object(spec, reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    payload = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name], reader
                    )
                elif spec.name in recipe.DIRECT_BY_NAME:
                    entry = recipe.DIRECT_BY_NAME[spec.name]
                    tensor = (
                        _decode_nvfp4_dense(entry.source_name, entry.shape, reader)
                        if entry.dequantize
                        else reader.get(entry.source_name)
                    )
                    # The export stores some scalars in bf16 where the artifact wants fp32.
                    dtype = {
                        inventory.FP32: torch.float32,
                        inventory.BF16: torch.bfloat16,
                        inventory.I32: torch.int32,
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
                    tensor = base_recipe.materialize_recipe(
                        base_recipe.RECIPES_BY_NAME[spec.name], reader, derived
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
