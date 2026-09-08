"""Convert a hybrid MoE checkpoint using its resolved configuration and stored weights."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from surogate.core.model import quant_schemes
from surogate.serve.convert.common.gguf_repack import (
    GgufRepackSource,
    RepackError,
)
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common import dflash as dflash_checkpoint
from surogate.serve.convert.common import recipe as family_recipe

from . import draft_head, inventory, recipe
from .exports import compressed_tensors_source, routed_nvfp4


RECIPE_ID = "qwen3_5_moe-config-v3"
ENCODER_PROFILE = "MAXABS_F16_RECIP_RNE_V1"

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    geometry: inventory.Geometry
    dflash_geometry: dflash_checkpoint.Geometry | None
    recipes: tuple[recipe.TensorRecipe, ...]
    model_dir: Path
    dflash_model_dir: Path | None
    base_config_summary: dict[str, object]
    dflash_config_summary: dict[str, object] | None
    base_source: recipe.SourcePreflight
    dflash_source: recipe.SourcePreflight | None
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan
    routed_nvfp4_dir: Path | None = None
    routed_nvfp4_summary: dict[str, object] | None = None
    #: Set when the source is a compressed-tensors export: the config decided every
    #: text-core object's format, and these read the stored words for them.
    compressed_source: compressed_tensors_source.CompressedTensorsSource | None = None
    compressed_plan: compressed_tensors_source.SourcePlan | None = None


def _tools_root() -> Path:
    """`serve/tools/`, which holds the fixtures a conversion reads (the draft-head ranking)."""
    return Path(__file__).resolve().parents[2] / "tools"


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    return inventory.geometry_block(inventory.geometry_from_config(config))


def validate_dflash_config(config: Mapping[str, object], geometry: inventory.Geometry):
    return asdict(dflash_checkpoint.geometry_from_config(config, geometry))


def preflight_inventory(geometry: inventory.Geometry, *, dflash=None) -> None:
    recipe.validate_recipe_coverage(geometry, dflash=dflash)


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def tensor_specs(geometry: inventory.Geometry, routed_nvfp4_source: bool = False, *, dflash=None):
    return (routed_nvfp4.tensor_specs(geometry, dflash=dflash) if routed_nvfp4_source
            else inventory.build_tensor_specs(geometry, dflash=dflash))


def object_specs(geometry: inventory.Geometry, routed_nvfp4_source: bool = False, *, dflash=None):
    return inventory.RESOURCE_SPECS + tensor_specs(geometry, routed_nvfp4_source, dflash=dflash)


def build_object_plan(resources, geometry: inventory.Geometry, *, routed_nvfp4_source=False, dflash=None):
    return family_conversion.build_object_plan(
        object_specs(geometry, routed_nvfp4_source, dflash=dflash), resources)


def preflight_conversion(
    model_dir: str | Path,
    dflash_model_dir: str | Path | None,
    routed_nvfp4_dir: str | Path | None = None,
    *, shared_expert: str = "as-stored", covered: tuple[str, ...] = (),
    object_specs=None, geometry: inventory.Geometry | None = None, dflash_geometry=None,
) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    geometry = geometry or inventory.geometry_from_checkpoint(model, config)
    dflash_model = Path(dflash_model_dir) if dflash_model_dir is not None else None
    if dflash_model is not None and dflash_geometry is None:
        dflash_geometry = dflash_checkpoint.geometry_from_config(
            family_conversion.load_json(dflash_model / "config.json"), geometry)
    recipes = recipe.build_recipes(geometry, dflash=dflash_geometry)
    base = {r.object_name: r for r in recipes if not r.object_name.startswith("dflash/")}
    scope = family_conversion.honour_declared_scope(
        config, geometry, model, what=family_conversion.checkpoint_label(model))
    if scope:
        print(scope, flush=True)
    compressed_source = (compressed_tensors_source.CompressedTensorsSource(model)
                         if quant_schemes.quantization_config_of(config) is not None else None)
    if compressed_source is not None and routed_nvfp4_dir is None:
        routed_nvfp4_dir = model
    routed = Path(routed_nvfp4_dir) if routed_nvfp4_dir is not None else None
    routed_summary = (routed_nvfp4.validate_config(
        family_conversion.load_json(routed / "config.json"), geometry) if routed is not None else None)
    preflight_inventory(geometry, dflash=dflash_geometry)
    compressed_plan = (compressed_source.plan(
        routed_nvfp4.tensor_specs(geometry, dflash=dflash_geometry), base, shared_expert=shared_expert)
        if compressed_source is not None else None)
    excluded = set(covered)
    if compressed_plan is not None:
        excluded.update(compressed_plan.covered)
    if routed is not None:
        excluded.update(name for name in base if routed_nvfp4.is_routed_object(name))
    base_source = family_recipe.preflight_sources(model, tuple(
        r for name, r in base.items() if name not in excluded))
    if routed is not None:
        # Validate the separate expert checkpoint before opening the output artifact.
        with ShardReader.for_directory(routed) as reader:
            routed_nvfp4.preflight_source(reader, geometry)
    dflash_source = recipe.preflight_dflash_sources(dflash_model, recipes) if dflash_model else None
    resources = load_resources(model)
    resource_map = {r.name: r.data for r in resources}
    specs = (compressed_plan.specs if compressed_plan is not None else tuple(object_specs)
             if object_specs is not None else tensor_specs(geometry, routed is not None, dflash=dflash_geometry))
    object_plan = family_conversion.build_object_plan(inventory.RESOURCE_SPECS + specs, resource_map)
    draft = draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, model, geometry=geometry)
    return ConversionPreflight(
        geometry=geometry, dflash_geometry=dflash_geometry, recipes=recipes,
        model_dir=model, dflash_model_dir=dflash_model,
        base_config_summary=inventory.geometry_block(geometry),
        dflash_config_summary=asdict(dflash_geometry) if dflash_geometry else None,
        base_source=base_source, dflash_source=dflash_source, resources=resources,
        draft=draft, object_plan=object_plan, routed_nvfp4_dir=routed,
        routed_nvfp4_summary=routed_summary, compressed_source=compressed_source,
        compressed_plan=compressed_plan)


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
    recipes_by_name: Mapping[str, recipe.TensorRecipe],
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
    return recipe.materialize_recipe(
        recipes_by_name[spec.name],
        reader,
        derived,
    )


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    dflash_model_dir: str | Path | None,
    out_path: str | Path,
    arguments: Mapping[str, object],
    base_config_summary: Mapping[str, object],
    dflash_config_summary: Mapping[str, object] | None,
    base_source_preflight: recipe.SourcePreflight,
    dflash_source_preflight: recipe.SourcePreflight | None,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path | None,
    weights_id: str = inventory.WEIGHTS_ID,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    combined_source = recipe.SourcePreflight(
        recipe_count=(
            base_source_preflight.recipe_count
            + (dflash_source_preflight.recipe_count if dflash_source_preflight else 0)
        ),
        source_tensor_count=(
            base_source_preflight.source_tensor_count
            + (dflash_source_preflight.source_tensor_count if dflash_source_preflight else 0)
        ),
        source_shard_count=(
            base_source_preflight.source_shard_count
            + (dflash_source_preflight.source_shard_count if dflash_source_preflight else 0)
        ),
        source_dtype_counts={
            "BF16": (
                base_source_preflight.source_dtype_counts.get("BF16", 0)
                + (dflash_source_preflight.source_dtype_counts.get("BF16", 0) if dflash_source_preflight else 0)
            )
        },
    )
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, weights_id, architecture="qwen3_5_moe"),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_tools_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary={
            "base": dict(base_config_summary),
            "dflash": dict(dflash_config_summary) if dflash_config_summary else None,
        },
        source_preflight=combined_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )
    report["source"]["base_model_path"] = report["source"].pop("model_path")
    report["source"]["dflash_model_path"] = (
        str(Path(dflash_model_dir).resolve()) if dflash_model_dir is not None else None
    )
    report["source_preflight"] = {
        "base": {
            "recipes": base_source_preflight.recipe_count,
            "tensors": base_source_preflight.source_tensor_count,
            "shards": base_source_preflight.source_shard_count,
            "dtypes": dict(base_source_preflight.source_dtype_counts),
        },
        "dflash": None if dflash_source_preflight is None else {
            "recipes": dflash_source_preflight.recipe_count,
            "tensors": dflash_source_preflight.source_tensor_count,
            "files": dflash_source_preflight.source_shard_count,
            "dtypes": dict(dflash_source_preflight.source_dtype_counts),
        },
        "combined": {
            "recipes": combined_source.recipe_count,
            "tensors": combined_source.source_tensor_count,
            "files": combined_source.source_shard_count,
            "dtypes": dict(combined_source.source_dtype_counts),
        },
    }
    report["draft_head"] = {"rows": base_config_summary["draft_vocab"],
                            "tokenizer_vocab_size": base_config_summary["token_domain"]}
    report["quantization"] = {"encoder_profile": ENCODER_PROFILE}
    return report


def convert(
    model_dir: str | Path,
    dflash_model_dir: str | Path | None,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    routed_nvfp4_dir: str | Path | None = None,
    shared_expert: str = "as-stored",
    gguf_repack: str | Path | None = None,
    mtp: bool = True,
    vision: bool = True,
) -> Path:
    """Run the complete target conversion and return its report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    dflash_model = Path(dflash_model_dir) if dflash_model_dir is not None else None
    # A GGUF source serves what the file already holds: the routed experts are K-quant
    # superblocks and the Q8_0 tensors repack into W8 bit-exactly, so only the remainder takes
    # the dequantise path.
    repack = GgufRepackSource(gguf_repack) if gguf_repack else None
    geometry = inventory.geometry_from_checkpoint(model, extra_names=repack.sources if repack else ())
    if not mtp:
        geometry = replace(geometry, mtp_layers=0)
    if not vision and "vision_config" in geometry.declared.hf_config:
        source_config = dict(geometry.declared.hf_config)
        source_config.pop("vision_config")
        geometry = inventory.geometry_from_checkpoint(model, source_config, extra_names=repack.sources if repack else ())
        if not mtp:
            geometry = replace(geometry, mtp_layers=0)
    dflash_geometry = (dflash_checkpoint.geometry_from_config(
        family_conversion.load_json(dflash_model / "config.json"), geometry) if dflash_model else None)
    recipes = recipe.build_recipes(geometry, dflash=dflash_geometry)
    base_recipes = {r.object_name: r for r in recipes if not r.object_name.startswith("dflash/")}
    dflash_recipes = {r.object_name: r for r in recipes if r.object_name.startswith("dflash/")}
    gguf_specs = tensor_specs(geometry, routed_nvfp4_dir is not None, dflash=dflash_geometry)
    native: dict[str, str] = {}
    repacked: tuple[str, ...] = ()
    dropped_prefixes = tuple(
        prefix for prefix, keep in (("mtp/", mtp), ("vision/", vision)) if not keep
    )
    if dropped_prefixes:
        gguf_specs = tuple(
            spec for spec in gguf_specs
            if not getattr(spec, "name", "").startswith(dropped_prefixes)
        )
        print(f"this export carries no {', '.join(p.rstrip('/') for p in dropped_prefixes)} — "
              "omitting those objects", flush=True)
    native_runs: dict = {}
    in_place: dict = {}
    external: tuple = ()
    if repack is not None:
        # The fused projections and the shared expert run kernels that read the row-split
        # W8 planes; until those learn the GGUF's interleaved Q8_0 block, those objects are
        # repacked rather than served from the file. Everything else -- the K-quant experts,
        # the output head, the embedding table, the attention output -- is read in place.
        native = repack.plan_native(
            base_recipes,
            gguf_specs,
            exclude_suffixes=recipe.NATIVE_EXCLUDE_SUFFIXES,
        )
        repacked = repack.plan(base_recipes, gguf_specs)
        if native:
            # By default those objects are not copied at all: the artifact names the GGUF and
            # the stretches of it each object reads. SUROGATE_GGUF_COPY=1 writes the bytes in.
            if os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
                # The draft head gathers its rows by shortlist rather than in order, so it needs
                # the same ids the write path uses. The shortlist is a pure function of the
                # ranking file and the checkpoint, so computing it here matches what preflight
                # computes later.
                draft_ids = draft_head.materialize_draft_head_token_ids(
                    draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, model, geometry=geometry)
                )
                native_runs = {
                    spec.name: repack.runs_for_native(
                        spec,
                        base_recipes[spec.name],
                        draft_ids if spec.name == draft_head.DRAFT_HEAD_OBJECT else None,
                    )
                    for spec in GgufRepackSource.native_specs(gguf_specs, native)
                    if spec.name in native
                }
                external = ((str(Path(repack.gguf_path).resolve()),
                             Path(repack.gguf_path).stat().st_size),)
                copied = sum(sum(r[2] for r in runs) for runs in native_runs.values())
                print(f"native K-quants: {len(native_runs)} objects read from the GGUF in place "
                      f"({copied / 1e9:.1f} GB not copied)", flush=True)
            gguf_specs = GgufRepackSource.native_specs(gguf_specs, native, native_runs)
            if not native_runs:
                print(f"native K-quants: {len(native)} objects served as the GGUF stores them",
                      flush=True)
        if repacked:
            print(f"bit-exact repack: {len(repacked)} objects", flush=True)
        # The objects whose kernels want the row-split planes: their rows are all Q8_0, which is
        # the same numbers in a different arrangement, so they are read from the file too and the
        # loader rearranges them.
        if os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
            in_place = repack.plan_repack_in_place(
                base_recipes, gguf_specs, recipe.NATIVE_EXCLUDE_SUFFIXES
            )
            if in_place:
                if not external:
                    external = ((str(Path(repack.gguf_path).resolve()),
                                 Path(repack.gguf_path).stat().st_size),)
                gguf_specs = GgufRepackSource.in_place_specs(gguf_specs, in_place)
                moved = sum(sum(r[2] for r in entry[0]) for entry in in_place.values())
                print(f"rearranged at load: {len(in_place)} objects read from the GGUF "
                      f"({moved / 1e9:.1f} GB not copied)", flush=True)
                repacked = tuple(n for n in repacked if n not in in_place)
    preflight = preflight_conversion(
        model, dflash_model, routed_nvfp4_dir, shared_expert=shared_expert,
        geometry=geometry, dflash_geometry=dflash_geometry,
        covered=tuple(native) + tuple(repacked) + tuple(in_place) + tuple(
            n for n in base_recipes if n.startswith(dropped_prefixes)
        ) if dropped_prefixes or native or repacked else (),
        object_specs=gguf_specs if (native or repacked or dropped_prefixes) else None,
    )

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"base={preflight.base_source.source_tensor_count} source tensors, "
        f"dflash={preflight.dflash_source.source_tensor_count if preflight.dflash_source else 0} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ArtifactWriter(
        output,
        ArtifactIdentity(
            inventory.MODEL_ID,
            compressed_tensors_source.WEIGHTS_ID
            if preflight.compressed_plan is not None
            else routed_nvfp4.WEIGHTS_ID
            if preflight.routed_nvfp4_dir is not None
            else inventory.WEIGHTS_ID,
         architecture="qwen3_5_moe"),
        preflight.object_plan.specs,
        external=external,
        geometry=inventory.geometry_block(geometry),
        layer_types=geometry.layer_types,
        dflash_geometry=dflash_checkpoint.geometry_block(dflash_geometry) if dflash_geometry else None,
        dflash_target_layers=dflash_geometry.target_feature_layers if dflash_geometry else None,
        vision_geometry=inventory.vision_geometry_block(geometry.declared.hf_config),
    ) as writer:
        index = 0

        def write_payload(spec, payload: bytes) -> None:
            nonlocal index
            writer.write(spec.name, payload)
            index += 1
            print(
                f"[{index}/{len(preflight.object_plan.specs)}] {spec.name}",
                flush=True,
            )

        for spec in preflight.object_plan.specs:
            if spec.name in resources:
                write_payload(spec, resources[spec.name])

        all_specs = tuple(s for s in preflight.object_plan.specs if hasattr(s, "format"))
        base_specs = tuple(s for s in all_specs if not s.name.startswith("dflash/"))
        routed_reader = ShardReader.for_directory(preflight.routed_nvfp4_dir) if preflight.routed_nvfp4_dir else None
        routed_cache = routed_nvfp4.LayerCache(routed_reader, geometry) if routed_reader else None
        try:
            with ShardReader.for_directory(model) as reader:
                for spec in base_specs:
                    if repack is not None and (
                        spec.name in native or spec.name in repacked or spec.name in in_place
                    ):
                        # The draft head gathers its rows by shortlist rather than in order.
                        token_ids = (
                            draft_head.materialize_draft_head_token_ids(preflight.draft)
                            if spec.name == draft_head.DRAFT_HEAD_OBJECT
                            else None
                        )
                        source_recipe = base_recipes[spec.name]
                        if spec.name in native_runs or spec.name in in_place:
                            continue  # read from the GGUF in place; nothing to write here
                        payload = (
                            repack.payload_for_native(spec, source_recipe, token_ids)
                            if spec.name in native
                            else repack.payload_for(spec, source_recipe, token_ids)
                        )
                        write_payload(spec, payload)
                        del payload
                        continue
                    if preflight.compressed_plan is not None and (
                        spec.name in preflight.compressed_plan.objects
                        or spec.name.endswith(compressed_tensors_source.INPUT_DIVISOR_SUFFIX)
                    ):
                        # Stored words copied as they are: NVFP4 codes, scales and global
                        # scale where the export packed the module, BF16 rows where it did
                        # not. No dequantise-requantise round trip in either case.
                        payload = preflight.compressed_source.payload_for(
                            spec.name, preflight.compressed_plan.objects, reader, resolved_device
                        )
                        write_payload(spec, payload)
                        del payload
                        continue
                    if routed_cache is not None and routed_nvfp4.is_routed_object(spec.name):
                        # The routed experts come from the NVFP4 checkpoint as stored words:
                        # no dequantise-requantise round trip, so the artifact holds exactly
                        # the codes vLLM serves from the same checkpoint.
                        payload = routed_cache.payload_for(
                            spec.name,
                            lambda tensor, spec=spec: encode_tensor_payload(
                                tensor, spec, resolved_device
                            ),
                        )
                        write_payload(spec, payload)
                        del payload
                        continue
                    tensor = materialize_tensor(
                        spec,
                        reader,
                        preflight.draft,
                        base_recipes,
                    )
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                    write_payload(spec, payload)
                    del payload
        finally:
            if routed_reader is not None:
                routed_reader.close()

        if dflash_model is not None:
            with ShardReader.for_directory(dflash_model) as reader:
                for spec in (s for s in all_specs if s.name.startswith("dflash/")):
                    tensor = materialize_tensor(
                        spec,
                        reader,
                        preflight.draft,
                        dflash_recipes,
                    )
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                    write_payload(spec, payload)
                    del payload

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = preflight.draft.ranking
    arguments = {
        "model": str(model_dir),
        "dflash_model": str(dflash_model_dir) if dflash_model_dir is not None else None,
        "routed_nvfp4": str(routed_nvfp4_dir) if routed_nvfp4_dir is not None else None,
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        model_dir=model,
        dflash_model_dir=dflash_model,
        out_path=output,
        arguments=arguments,
        base_config_summary=preflight.base_config_summary,
        dflash_config_summary=preflight.dflash_config_summary,
        base_source_preflight=preflight.base_source,
        dflash_source_preflight=preflight.dflash_source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
        weights_id=compressed_tensors_source.WEIGHTS_ID if preflight.compressed_plan is not None
                   else routed_nvfp4.WEIGHTS_ID if preflight.routed_nvfp4_dir else inventory.WEIGHTS_ID,
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
    parser.add_argument("--dflash-model", type=Path, default=None,
                        help="DFlash drafter checkpoint; omit to build an artifact without the dflash/* family")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--routed-nvfp4",
        type=Path,
        default=None,
        help=(
            "NVFP4 checkpoint (compressed-tensors nvfp4-pack-quantized) whose routed "
            "expert tensors replace the groupwise-int ones; writes the routed-nvfp4 "
            "weights profile"
        ),
    )
    parser.add_argument("--no-vision", action="store_true",
                        help="convert without the vision tower (a text-only export)")
    parser.add_argument("--no-mtp", action="store_true",
                        help="convert without the MTP (nextn) block, which community GGUFs strip")
    parser.add_argument("--gguf-repack", type=Path, default=None,
                        help="repack map from a GGUF bridge; serves what the file already holds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shared-expert",
        choices=compressed_tensors_source.SHARED_EXPERT_CHOICES,
        default="as-stored",
        help=(
            "compressed-tensors sources only: keep the shared expert in the format the export "
            "stored (default), or requantise it to W8 for the MoE kernels, which admit W8 only "
            "until they take NVFP4 -- the one place the export's format is not kept"
        ),
    )
    args = parser.parse_args(argv)
    convert(
        args.model,
        args.dflash_model,
        args.out,
        device=args.device,
        routed_nvfp4_dir=args.routed_nvfp4,
        shared_expert=args.shared_expert,
        gguf_repack=args.gguf_repack,
        mtp=not args.no_mtp,
        vision=not args.no_vision,
    )


if __name__ == "__main__":
    main()
