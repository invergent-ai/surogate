"""Convert `sakamakismile/Qwen3.8-27B-MTP-NVFP4` into an all-NVFP4 `.ninfer` artifact.

One checkpoint supplies everything: every language linear as NVFP4 blocks, and
the bf16 embedding, lm_head, MTP block and vision tower alongside them. The
`nvfp4` artifact (convert_nvfp4) keeps the attention and GDN projections FP8
because its source exports them that way; this one does not, which halves those
weights and moves their GEMMs onto the NVFP4 path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from surogate.serve.tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from surogate.serve.tools.artifact.layouts import encode_direct, encode_nvfp4
from surogate.serve.tools.convert.common.quantize import pick_device
from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.qwen3_6.common import conversion as family_conversion
from surogate.serve.tools.convert.qwen3_6_27b import draft_head

from . import fp8_embedding
from . import inventory_nvfp4_all as inventory
from . import recipe_nvfp4 as mixed_recipe
from . import recipe_nvfp4_all as recipe

RECIPE_ID = "qwen3_8_27b-nvfp4-all-v1"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4_all.ninfer"

_CONTROL_SUFFIX = "gdn/a_b_projection"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    object_plan: family_conversion.ObjectPlan
    resources: tuple
    draft: "draft_head.DraftHeadContext"


def _repo_root() -> Path:
    # As in convert_nvfp4: DEFAULT_RANKING is relative to surogate/serve/tools.
    return Path(__file__).resolve().parents[2]


def _resources_from_artifact(path: Path) -> tuple:
    """Take the frontend resources from an existing artifact for this model.

    This export is the MTP+vision variant and its tokenizer_config.json does not satisfy the
    target's Qwen3.6 prefix-semantics check, while its weights are the same model. The
    resources are identity, not weights, so they come from an artifact already known to load.
    """

    artifact = Artifact.open(path)
    available = {obj.name: obj for obj in artifact.objects}
    payloads = []
    for spec in inventory.RESOURCE_SPECS:
        obj = available.get(spec.name)
        if obj is None:
            raise KeyError(f"{path}: no resource {spec.name!r} to borrow")
        payloads.append(
            family_conversion.ResourcePayload(spec.name, bytes(artifact.payload(obj)))
        )
    return tuple(payloads)


def preflight_conversion(
    model_dir: str | Path, resources_from: str | Path | None = None
) -> ConversionPreflight:
    source = Path(model_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"NVFP4 source directory not found: {source}")
    inventory.validate_inventory()
    recipe.validate_recipe()
    resources = (
        _resources_from_artifact(Path(resources_from))
        if resources_from is not None
        else family_conversion.load_resources(source, inventory.RESOURCE_SPECS)
    )
    plan = family_conversion.build_object_plan(
        inventory.OBJECT_SPECS, {item.name: item.data for item in resources}
    )
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, source)
    return ConversionPreflight(source, plan, resources, draft)


def _layer_of(object_name: str) -> int:
    return int(object_name.split("text/layers/")[1].split("/")[0])


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    resources_from: str | Path | None = None,
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir, resources_from)
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.NVFP4_WEIGHT_RECIPES)} NVFP4 matrices, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    total = len(inventory.OBJECT_SPECS)
    derived = {
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
            draft_head.materialize_draft_head_token_ids(preflight.draft)
        )
    }
    with ShardReader(preflight.model_dir) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                payload: bytes | Iterable[bytes]
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    try:
                        packed, scales, divisor = recipe.materialize_nvfp4_weight(
                            recipe.NVFP4_WEIGHTS_BY_NAME[spec.name], reader
                        )
                    except ValueError as error:
                        if "query_key_value_z" in spec.name:
                            raise ValueError(
                                f"{spec.name}: {recipe.FUSED_GDN_DIVISOR_BLOCKER}"
                            ) from error
                        raise
                    payload = encode_nvfp4(packed, scales, divisor, spec.shape)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name], reader
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in ("text/token_embedding", "text/output_head"):
                    # Both endpoints ship bf16 in this export and the target reads a byte-wide
                    # head, so both are encoded FP8 row-scaled here (the mixed recipe takes its
                    # head straight from an already-FP8 source instead).
                    source = (
                        mixed_recipe.OFFICIAL_EMBEDDING_SOURCE.name
                        if spec.name == "text/token_embedding"
                        else "lm_head.weight"
                    )
                    payload = fp8_embedding.iter_reader_payload(reader, source, spec.shape)
                elif spec.name.endswith(_CONTROL_SUFFIX):
                    # in_proj_a/in_proj_b arrive quantised in this export.
                    tensor = recipe.materialize_control_projection(
                        _layer_of(spec.name), reader
                    )
                    payload = encode_direct(tensor.reshape(spec.shape), spec.format)
                elif spec.name in mixed_recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = mixed_recipe.materialize_quantized_direct(spec.name, reader)
                    payload = encode_direct(tensor.reshape(spec.shape), spec.format)
                elif spec.name in mixed_recipe.OFFICIAL_RECIPES_BY_NAME:
                    tensor = mixed_recipe.materialize_official(spec.name, reader, derived)
                    payload = family_conversion.encode_tensor_payload(
                        tensor.reshape(spec.shape), spec, resolved_device
                    )
                else:
                    raise KeyError(f"no recipe covers artifact object {spec.name!r}")
                writer.write(spec.name, payload)
                del payload
                if index % 50 == 0 or index == total:
                    print(f"[{index}/{total}] {spec.name}", flush=True)
    print(
        f"conversion finished in {time.perf_counter() - started:.1f} s -> {output}",
        flush=True,
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NVFP4 export directory")
    parser.add_argument("--out", required=True, help="artifact output path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resources-from",
        default=None,
        help="take the frontend resources from this existing .ninfer instead of the checkpoint",
    )
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, resources_from=args.resources_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
