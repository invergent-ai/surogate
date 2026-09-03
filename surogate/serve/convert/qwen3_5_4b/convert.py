"""Convert the registered Qwen3.5-4B checkpoint into one complete artifact.

Canonical invocation::

    python -m surogate.serve.convert.qwen3_5_4b.convert \
      --model /path/to/Qwen3.5-4B/base-hf-bf16 \
      --out out/qwen3_5_4b.sinfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from surogate.serve.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from surogate.serve.convert.common.gguf_repack import (
    GgufRepackSource,
    RepackError,
    half_names,
)
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common import official_resources

from . import draft_head, inventory, recipe


RECIPE_ID = "qwen3_5_4b-v2"

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 32,
    "full_attention_interval": 4,
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": True,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 5120,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}


ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _tools_root() -> Path:
    """`serve/tools/`, which holds the fixtures a conversion reads (the draft-head ranking)."""
    return Path(__file__).resolve().parents[2] / "tools"


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def _check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    family_conversion.check_members(scope, actual, expected)


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact registered checkpoint dimensions and summarize them."""

    _check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("config.json must contain text_config")
    _check_members("text_config", text, _TEXT_CONFIG)
    expected_layer_types = tuple(
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in range(32)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match the registered 32-layer schedule"
        )
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    _check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": 24 - len(inventory.FULL_ATTENTION_LAYERS),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": None,  # text-only target
        "mtp_num_hidden_layers": text["mtp_num_hidden_layers"],
        "vision_token_ids": {
            name: config[name]
            for name in (
                "vision_start_token_id",
                "vision_end_token_id",
                "image_token_id",
                "video_token_id",
            )
        },
    }


def preflight_inventory() -> None:
    """Establish the one complete target inventory and recipe pairing."""

    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != (6, 355, 2, 12, 0, 369, 375):
        raise ValueError("registered inventory is incomplete")
    if (
        len(inventory.TENSOR_SPECS_NO_MTP),
        len(inventory.OBJECT_SPECS_NO_MTP),
    ) != (357, 363):
        raise ValueError("registered no-MTP inventory is incomplete")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    # This target has no pinned official-resource profile yet (the family pins
    # are the 27B files). Frontend correctness is covered by the tokenizer
    # equivalence tests in tests/serve/; load unpinned.
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def build_object_plan(
    resources: Mapping[str, bytes], *, mtp: bool = True, vision: bool = True,
    native: Mapping[str, str] | None = None, object_specs=None
) -> ObjectPlan:
    """Compute every payload-relative object offset for the selected variant. `native` names
    the objects served as K-quants verbatim from a GGUF, with their format rewritten."""
    preflight_inventory()
    if object_specs is None:
        _, object_specs = inventory.active_specs(mtp=mtp, vision=vision)
    if native:
        object_specs = GgufRepackSource.native_specs(object_specs, native)
    return family_conversion.build_object_plan(object_specs, resources)


def active_recipes(*, mtp: bool, vision: bool = True) -> dict[str, recipe.TensorRecipe]:
    """Recipes for the requested artifact variant (PATCHES.md #15)."""
    dropped = tuple(
        prefix for prefix, keep in (("mtp/", mtp), ("vision/", vision)) if not keep
    )
    if not dropped:
        return dict(recipe.RECIPES_BY_NAME)
    return {
        name: tensor_recipe
        for name, tensor_recipe in recipe.RECIPES_BY_NAME.items()
        if not name.startswith(dropped)
    }


def plan_repack(
    repack: GgufRepackSource | None,
    recipes_by_name: dict[str, recipe.TensorRecipe],
    tensor_specs,
    native: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Objects the repack source covers; verifies the map is not over-broad.

    surogate vendor patch (PATCHES.md #14): every recipe left on the
    materialize path must find its sources in the bridged checkpoint, so a
    mapped source consumed by an un-planned recipe is a hard error.
    """

    if repack is None:
        return ()
    planned = repack.plan(recipes_by_name, tensor_specs)
    covered = set(planned) | set(native or ())
    stray = {
        source.name
        for name, tensor_recipe in recipes_by_name.items()
        if name not in covered
        for source in recipe.expression_sources(tensor_recipe.expression)
        if source.name in repack.sources
    }
    if stray:
        raise RepackError(
            "repack map names sources still needed by materialized recipes: "
            + ", ".join(sorted(stray))
        )
    return planned


def preflight_conversion(
    model_dir: str | Path,
    repack: GgufRepackSource | None = None,
    planned: tuple[str, ...] = (),
    *,
    mtp: bool = True,
    vision: bool = True,
    native: Mapping[str, str] | None = None,
    object_specs=None,
) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    recipes = active_recipes(mtp=mtp, vision=vision)
    if planned or not mtp or not vision:
        # surogate vendor patches (PATCHES.md #14/#15): repacked objects read
        # the GGUF directly, and the no-MTP variant has no mtp recipes; only
        # the remaining recipes need bridged sources.
        remaining = tuple(
            tensor_recipe
            for name, tensor_recipe in recipes.items()
            if name not in planned
        )
        source = recipe.preflight_sources(model, remaining)
    else:
        source = recipe.preflight_sources(model)
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map, mtp=mtp, vision=vision, native=native,
                                    object_specs=object_specs)
    ranking = _tools_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    derived = None
    if spec.name in (
        draft_head.DRAFT_HEAD_OBJECT,
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT,
    ):
        derived = {
            draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
                draft_head.materialize_draft_head_token_ids(draft)
            )
        }
    tensor = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME[spec.name],
        reader,
        derived,
    )
    if spec.format == "BF16" and tensor.dtype == torch.float32:
        # surogate vendor patch (PATCHES.md #16): official Qwen3.5 releases
        # store some control tensors (A_log, dt_bias) in F32; the registered
        # artifact format is BF16, so narrow explicitly here.
        tensor = tensor.to(torch.bfloat16)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized tensor according to its registered signature."""

    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the external descriptive conversion report."""

    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_tools_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    gguf_repack: str | Path | None = None,
    mtp: bool = True,
    vision: bool = True,
) -> Path:
    """Run the complete registered conversion and return the report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    repack = GgufRepackSource(gguf_repack) if gguf_repack else None
    recipes = active_recipes(mtp=mtp, vision=vision)
    active_tensor_specs, active_object_specs = inventory.active_specs(mtp=mtp, vision=vision)
    # A tied head duplicates the vocabulary table; drop the copy and let the loader point
    # both plans at the survivor.
    tied = set(inventory.tied_duplicate_objects(recipes, active_tensor_specs))
    if tied:
        active_tensor_specs = tuple(s for s in active_tensor_specs if s.name not in tied)
        active_object_specs = tuple(
            s for s in active_object_specs if getattr(s, "name", None) not in tied
        )
        recipes = {n: r for n, r in recipes.items() if n not in tied}
        print(f"tied objects dropped: {', '.join(sorted(tied))}", flush=True)
    native = repack.plan_native(recipes, active_tensor_specs) if repack is not None else {}
    # A fused parent whose halves carry different K-quant types is stored as two objects; the
    # loader binds the pair and the split ops project each half straight into its destination.
    halves = repack.plan_native_halves(recipes, active_tensor_specs) if repack is not None else {}
    planned = plan_repack(repack, recipes, active_tensor_specs, set(native) | set(halves))
    repacked_names = frozenset(planned)
    if halves:
        active_tensor_specs = GgufRepackSource.native_half_specs(active_tensor_specs, halves)
        active_object_specs = GgufRepackSource.native_half_specs(active_object_specs, halves)
        print(f"native K-quant halves: {len(halves)} fused parents stored as typed pairs",
              flush=True)
    if native:
        active_tensor_specs = GgufRepackSource.native_specs(active_tensor_specs, native)
        active_object_specs = GgufRepackSource.native_specs(active_object_specs, native)
        print(f"native K-quants: {len(native)} objects served as the GGUF stores them", flush=True)
    preflight = preflight_conversion(
        model, repack, planned + tuple(native) + tuple(sorted(tied)) + tuple(halves), mtp=mtp, vision=vision, native=native, object_specs=active_object_specs
    )

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    # object name -> (fused parent recipe, the rows of that parent it holds)
    half_lookup: dict[str, tuple[str, slice]] = {}
    for parent, runs in halves.items():
        names = half_names(parent)
        first = 0
        for name, (_, rows) in zip(names, runs):
            half_lookup[name] = (parent, slice(first, first + rows))
            first += rows

    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(active_object_specs, start=1):
                repacked = False
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif repack is not None and spec.name in half_lookup:
                    parent, row_slice = half_lookup[spec.name]
                    payload = repack.payload_for_native(
                        spec, recipes[parent], None, row_slice=row_slice
                    )
                    repacked = True
                elif repack is not None and spec.name in native:
                    # A K-quant served as the GGUF stores it: rows gathered verbatim.
                    token_ids = None
                    if spec.name == draft_head.DRAFT_HEAD_OBJECT:
                        token_ids = draft_head.materialize_draft_head_token_ids(
                            preflight.draft
                        )
                    payload = repack.payload_for_native(spec, recipes[spec.name], token_ids)
                    repacked = True
                elif repack is not None and spec.name in repacked_names:
                    # surogate vendor patch (PATCHES.md #14): bit-exact Q8_0
                    # plane repack; no dequantization or requantization.
                    token_ids = None
                    if spec.name == draft_head.DRAFT_HEAD_OBJECT:
                        token_ids = draft_head.materialize_draft_head_token_ids(
                            preflight.draft
                        )
                    payload = repack.payload_for(spec, recipes[spec.name], token_ids)
                    repacked = True
                else:
                    tensor = materialize_tensor(spec, reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(active_object_specs)}] {spec.name}"
                    + (" (repacked)" if repacked else ""),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _tools_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
        "gguf_repack": str(gguf_repack) if gguf_repack else None,
        "repacked_objects": len(repacked_names),
        "mtp": mtp,
    }
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path, default=None)
    parser.add_argument("--no-vision", action="store_true",
                        help="convert without the vision tower (a text-only export)")
    parser.add_argument("--no-mtp", action="store_true",
                        help="source checkpoint has no MTP (nextn) block; "
                             "emit the artifact variant without mtp/* objects")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack,
            mtp=not args.no_mtp, vision=not args.no_vision)


if __name__ == "__main__":
    main()
