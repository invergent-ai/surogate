"""Convert one interleaved gated-delta checkpoint into one complete artifact.

Canonical invocation::

    python -m surogate.serve.convert.qwen3_5.convert \
      --model /path/to/Qwen3.5-2B/base-hf-bf16 \
      --out out/qwen3_5.sinfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import os
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
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common import qwen3_5 as checkpoint

from . import draft_head, inventory, recipe


RECIPE_ID = "qwen3_5-config-v3"
ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


def recipe_id_for(geometry: "inventory.Geometry") -> str:
    return RECIPE_ID


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config: dict[str, object]
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



def geometry_block(geometry: inventory.Geometry, *, mtp: bool | None = None) -> dict:
    return checkpoint.geometry_block(geometry, mtp=mtp)


def validate_config(config: Mapping[str, object], *, vision: bool = True) -> dict:
    g = inventory.geometry_from_config(config)
    return _config_summary(g, vision=vision)


def _config_summary(g: inventory.Geometry, *, vision: bool = True) -> dict:
    return {"architecture": g.declared.hf_config["architectures"][0],
            "text": dict(g.text_config), "layer_types": list(g.layer_types),
            "vision": g.declared.hf_config.get("vision_config") if vision else None}


def preflight_inventory(geometry: inventory.Geometry) -> None:
    inventory.export_inventory(inventory.GROUPWISE_INT, geometry).validate_inventory()
    recipe.validate_recipe_coverage(geometry)


def load_resources(model_dir: str | Path,
                   geometry: inventory.Geometry | None = None) -> tuple[ResourcePayload, ...]:
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def vision_geometry_block(config: Mapping[str, object]) -> dict | None:
    return checkpoint.vision_geometry_block(config)


def carries_vision(object_specs) -> bool:
    """Whether an object plan holds the tower, so the artifact declares its geometry only
    when there is one to bind."""
    return any(spec.name.startswith("vision/") for spec in object_specs)


def build_object_plan(
    resources: Mapping[str, bytes], *, mtp: bool = True, vision: bool = True,
    native: Mapping[str, str] | None = None, object_specs=None, geometry=None
) -> ObjectPlan:
    """Compute every payload-relative object offset for the selected variant. `native` names
    the objects served as K-quants verbatim from a GGUF, with their format rewritten."""
    preflight_inventory(geometry)
    if object_specs is None:
        _, object_specs = inventory.active_specs(mtp=mtp, vision=vision,
                                                 geometry=geometry)
    if native:
        object_specs = GgufRepackSource.native_specs(object_specs, native)
    return family_conversion.build_object_plan(object_specs, resources)


def active_recipes(*, mtp: bool, vision: bool = True,
                   geometry=None) -> dict[str, recipe.TensorRecipe]:
    """Recipes for the requested artifact variant, at this checkpoint's size."""
    by_name = {r.object_name: r for r in recipe.build_recipes(geometry)}
    dropped = tuple(
        prefix for prefix, keep in (("mtp/", mtp), ("vision/", vision)) if not keep
    )
    if not dropped:
        return by_name
    return {
        name: tensor_recipe
        for name, tensor_recipe in by_name.items()
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
    geometry: inventory.Geometry | None = None,
) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config = _load_config(model)
    if geometry is None:
        geometry = inventory.geometry_from_checkpoint(model, config, extra_names=repack.sources if repack else ())
    config_summary = _config_summary(geometry, vision=vision)
    preflight_inventory(geometry)
    recipes = active_recipes(mtp=mtp, vision=vision, geometry=geometry)
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
        source = recipe.preflight_sources(model, tuple(recipes.values()))
    resources = load_resources(model, geometry)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map, mtp=mtp, vision=vision, native=native,
                                    object_specs=object_specs, geometry=geometry)
    ranking = _tools_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model, geometry=geometry)
    return ConversionPreflight(
        model_dir=model,
        config=config,
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
    recipes: Mapping[str, recipe.TensorRecipe],
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
        recipes[spec.name],
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
    ranking_path: str | Path | None,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
    geometry: "inventory.Geometry | None" = None,
) -> dict[str, object]:
    """Build the external descriptive conversion report."""

    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(
            inventory.model_id_for(geometry), inventory.WEIGHTS_ID
        , architecture="qwen3_5"),
        target_key=inventory.TARGET_KEY,
        recipe_id=recipe_id_for(geometry),
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
    vision_storage: str = inventory.VISION_BF16,
) -> Path:
    """Run the complete registered conversion and return the report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    repack = GgufRepackSource(gguf_repack) if gguf_repack else None
    geometry = inventory.geometry_from_checkpoint(model, extra_names=repack.sources if repack else ())
    source_names = set(family_conversion.checkpoint_tensor_names(model)) | set(repack.sources if repack else ())
    mtp = mtp and bool(geometry.mtp_layers) and any(name.startswith("mtp.") for name in source_names)
    vision = vision and bool(geometry.declared.hf_config.get("vision_config")) and any(
        "visual." in name for name in source_names)
    geometry = replace(geometry, mtp_layers=geometry.mtp_layers if mtp else 0)
    recipes = active_recipes(mtp=mtp, vision=vision, geometry=geometry)
    active_tensor_specs, active_object_specs = inventory.active_specs(
        mtp=mtp, vision=vision, geometry=geometry, vision_storage=vision_storage)
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
    # A weight whose kernel reads row-split planes holds the same numbers the GGUF does, in a
    # different arrangement: read it from the file too and let the loader rearrange it.
    in_place: dict = {}
    if repack is not None and os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
        in_place = repack.plan_repack_in_place(
            recipes, active_tensor_specs, getattr(recipe, "NATIVE_EXCLUDE_SUFFIXES", ())
        )
        if in_place:
            moved = sum(sum(r[2] for r in entry[0]) for entry in in_place.values())
            print(f"rearranged at load: {len(in_place)} objects read from the GGUF "
                  f"({moved / 1e9:.2f} GB not copied)", flush=True)
            repacked_names = frozenset(n for n in repacked_names if n not in in_place)
    if halves:
        active_tensor_specs = GgufRepackSource.native_half_specs(active_tensor_specs, halves)
        active_object_specs = GgufRepackSource.native_half_specs(active_object_specs, halves)
        print(f"native K-quant halves: {len(halves)} fused parents stored as typed pairs",
              flush=True)
    external = ()
    native_runs: dict = {}
    if native and repack is not None and os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
        # By default those objects are not copied at all: the artifact names the GGUF and the
        # stretches of it each object reads. SUROGATE_GGUF_COPY=1 writes the bytes in.
        draft_ids = draft_head.materialize_draft_head_token_ids(
            draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, model, geometry=geometry)
        )
        native_runs = {
            spec.name: repack.runs_for_native(
                spec, recipes[spec.name],
                draft_ids if spec.name == draft_head.DRAFT_HEAD_OBJECT else None,
            )
            for spec in GgufRepackSource.native_specs(active_tensor_specs, native)
            if spec.name in native
        }
        external = ((str(Path(repack.gguf_path).resolve()),
                     Path(repack.gguf_path).stat().st_size),)
        not_copied = sum(sum(r[2] for r in runs) for runs in native_runs.values())
        print(f"native K-quants: {len(native_runs)} objects read from the GGUF in place "
              f"({not_copied / 1e9:.2f} GB not copied)", flush=True)
    if in_place:
        active_tensor_specs = GgufRepackSource.in_place_specs(active_tensor_specs, in_place)
        active_object_specs = GgufRepackSource.in_place_specs(active_object_specs, in_place)
    if native:
        active_tensor_specs = GgufRepackSource.native_specs(active_tensor_specs, native,
                                                            native_runs)
        active_object_specs = GgufRepackSource.native_specs(active_object_specs, native,
                                                            native_runs)
        if not native_runs:
            print(f"native K-quants: {len(native)} objects served as the GGUF stores them",
                  flush=True)
    preflight = preflight_conversion(
        model, repack, planned + tuple(native) + tuple(sorted(tied)) + tuple(halves) + tuple(in_place),
        mtp=mtp, vision=vision, native=native, object_specs=active_object_specs, geometry=geometry
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

    with ShardReader.for_directory(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.model_id_for(geometry), inventory.WEIGHTS_ID, architecture="qwen3_5"),
            preflight.object_plan.specs,
            geometry=geometry_block(geometry),
            layer_types=geometry.layer_types,
            vision_geometry=(vision_geometry_block(preflight.config)
                             if carries_vision(preflight.object_plan.specs) else None),
            external=external,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(active_object_specs, start=1):
                repacked = False
                if isinstance(spec, inventory.ResourceSpec):
                    # A base model carries no chat template, and the plan drops the object
                    # rather than storing an empty one.
                    if spec.name not in resources:
                        continue
                    payload = resources[spec.name]
                elif repack is not None and spec.name in half_lookup:
                    parent, row_slice = half_lookup[spec.name]
                    payload = repack.payload_for_native(
                        spec, recipes[parent], None, row_slice=row_slice
                    )
                    repacked = True
                elif repack is not None and (spec.name in native_runs or spec.name in in_place):
                    # Read from the GGUF where it lies; the artifact carries no bytes for it.
                    continue
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
                    tensor = materialize_tensor(spec, reader, preflight.draft, recipes)
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
    ranking = preflight.draft.ranking
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
        "gguf_repack": str(gguf_repack) if gguf_repack else None,
        "repacked_objects": len(repacked_names),
        "mtp": mtp,
    }
    report = build_conversion_report(
        geometry=geometry,
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


#: The exports that read two checkpoints: the published BF16 weights supply the objects the
#: quantised release does not carry, so both directories have to be named.
DUAL_SOURCE_PROFILES = (inventory.NVFP4_MIXED_BF16, inventory.NVFP4_MLP_ONLY)


def _export_writer(profile: str):
    from importlib import import_module
    modules = {
        inventory.FP8_BLOCK: "convert_fp8_block", inventory.FP8_CHANNEL: "convert_fp8_block",
        inventory.NVFP4_UNIFORM: "convert_nvfp4_uniform",
        inventory.NVFP4_MIXED_BF16: "convert_nvfp4_mixed_bf16",
        inventory.NVFP4_MLP_ONLY: "convert_nvfp4_mlp_only",
        inventory.NVFP4_ALL: "convert_nvfp4_all",
    }
    return import_module(f"{__package__}.exports.{modules[profile]}")


def profile_for_checkpoint(config: Mapping[str, object]) -> str:
    """Which export this checkpoint is, from what it says about itself.

    Unquantized checkpoints use the groupwise policy; quantized checkpoints dispatch by
    their declared encoding. Tensor metadata resolves the per-layer exceptions.
    """
    quantization = config.get("quantization_config") or {}
    text = json.dumps(quantization) if isinstance(quantization, Mapping) else ""
    method = str(quantization.get("quant_method", "")) if isinstance(quantization, Mapping) else ""
    if method == "fp8" and isinstance(quantization, Mapping) and quantization.get("weight_block_size"):
        # Hugging Face fine-grained FP8: a [128, 128] block scale grid per weight.
        if list(quantization["weight_block_size"]) != [128, 128]:
            raise ValueError(f"fp8 weight_block_size {quantization['weight_block_size']} is not [128, 128]")
        return inventory.FP8_BLOCK
    if method == "compressed-tensors" and isinstance(quantization, Mapping):
        groups = quantization.get("config_groups") or {}
        weights = next(iter(groups.values()), {}).get("weights") or {} if groups else {}
        if (str(weights.get("type", "")).lower() == "float" and int(weights.get("num_bits", 0) or 0) == 8
                and str(weights.get("strategy", "")) == "channel"):
            return inventory.FP8_CHANNEL
    if "nvfp4" not in text.lower() and not any(
        (group.get("weights") or {}).get("num_bits") == 4
        for group in (quantization.get("config_groups") or {}).values()
    ):
        return inventory.GROUPWISE_INT
    if method.lower() in ("modelopt", "nvfp4", "modelopt_fp4"):
        return inventory.NVFP4_UNIFORM
    groups = quantization.get("config_groups", {})
    kinds = {(g.get("weights") or {}).get("num_bits") for g in groups.values()}
    if 4 in kinds and 8 in kinds:
        return inventory.NVFP4_MLP_ONLY
    if 4 in kinds:
        return inventory.NVFP4_ALL
    raise ValueError("NVFP4 quantization_config does not declare a supported storage scheme")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path, default=None)
    parser.add_argument("--profile", choices=inventory.PROFILES, default=None,
                        help="which export this checkpoint is; read from its "
                             "quantization_config when not given")
    parser.add_argument("--quantized-model", type=Path, default=None,
                        help="the quantised release, for the exports that publish the "
                             "unquantised objects separately")
    parser.add_argument("--vision-storage", choices=inventory.VISION_STORAGE,
                        default=inventory.VISION_BF16,
                        help="How to store the vision tower. `bf16` is the weights the "
                             "checkpoint ships, and the default. `quantized` is about a third "
                             "of the size and measurably further from the source tower.")
    parser.add_argument("--no-vision", action="store_true",
                        help="convert without the vision tower (a text-only export)")
    parser.add_argument("--no-mtp", action="store_true",
                        help="source checkpoint has no MTP (nextn) block; "
                             "emit the artifact variant without mtp/* objects")
    args = parser.parse_args(argv)
    _model = Path(args.model)
    _config = _load_config(_model)
    # What the checkpoint says about its own quantisation. Here rather than in
    # `preflight_conversion` because every export profile funnels through this dispatch and
    # only some of them take that path.
    _scope = family_conversion.honour_declared_scope(
        _config, inventory.geometry_from_config(_config), _model, what=family_conversion.checkpoint_label(_model))
    if _scope:
        print(_scope, flush=True)
    profile = args.profile or profile_for_checkpoint(_config)
    if profile == inventory.GROUPWISE_INT:
        convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack,
                mtp=not args.no_mtp, vision=not args.no_vision,
                vision_storage=args.vision_storage)
        return
    writer = _export_writer(profile)
    if profile in DUAL_SOURCE_PROFILES:
        if args.quantized_model is None:
            writer.convert(args.model, args.out, device=args.device, mtp=not args.no_mtp, vision=not args.no_vision)
        else:
            writer.convert(args.model, args.quantized_model, args.out, device=args.device, mtp=not args.no_mtp, vision=not args.no_vision)
        return
    writer.convert(args.model, args.out, device=args.device, mtp=not args.no_mtp, vision=not args.no_vision)


if __name__ == "__main__":
    main()
