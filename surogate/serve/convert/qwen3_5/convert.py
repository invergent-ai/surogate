"""Convert one interleaved gated-delta checkpoint into one complete artifact.

Canonical invocation::

    python -m surogate.serve.convert.qwen3_5.convert \
      --model /path/to/Qwen3.5-2B/base-hf-bf16 \
      --out out/qwen3_5.sinfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from surogate.serve.convert.common import official_resources

from . import draft_head, inventory, recipe


#: The registered size's recipe. Each generation is a separately published lineage with its
#: own id, and every artifact records the one that made it, so the id follows the checkpoint.
RECIPE_ID = "qwen3_5-v2"
RECIPE_IDS = {"qwen3.6-27b": "qwen3_6_27b-v2", "qwen3.8-27b": "qwen3_8_27b-v1"}


def recipe_id_for(geometry: "inventory.Geometry") -> str:
    return RECIPE_IDS.get(inventory.model_id_for(geometry), RECIPE_ID)


#: The published frontend of each generation, pinned by content. Qwen3.8 ships a different
#: tokenizer and chat template from Qwen3.6, so the profile is per model id; the smaller
#: sizes have no pinned profile yet and their frontend correctness is covered by the
#: tokenizer equivalence tests in tests/serve/.
_QWEN3_8_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
    ),
    "frontend/tokenizer_config.json": (
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
    ),
    "frontend/chat_template.jinja": (
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    ),
    "frontend/generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}


_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
#: The generations this converter serves. They share every graph and differ only in how their
#: exports quantise, which the artifact records rather than the code branching on.
_MODEL_TYPES = ("qwen3_5", "qwen3_6", "qwen3_8")
#: Members that must hold for any size of this family, checked by value. The dimensions are
#: *not* here: one converter serves every size, so they are read from the checkpoint and the
#: artifact states them. What remains is what makes a checkpoint this architecture.
_TEXT_CONFIG = {
    "full_attention_interval": 4,
    "vocab_size": 248320,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
}

#: The dimensions, checked only for presence and self-consistency.
_TEXT_DIMENSIONS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
}
#: The tower members that hold at every size. The four that do not are checked against the
#: tower this size's object list is built from, so a config and an inventory cannot disagree.
_VISION_CONFIG = {
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}


def _vision_dimensions(geometry: "inventory.Geometry") -> dict[str, int]:
    tower = inventory.vision_tower(geometry)
    return {
        "depth": tower["layers"],
        "hidden_size": tower["hidden"],
        "intermediate_size": tower["intermediate"],
        "out_hidden_size": geometry.hidden,
    }


ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


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



def geometry_block(config: Mapping[str, object]) -> dict[str, float]:
    """The artifact's `geometry` member: the dimensions the engine reads at load.

    This target is one compiled size, and these are the numbers that make it that size.
    Stating them in the artifact is what lets one engine target serve every size of the
    family, and it is the checkpoint's own config that states them here.
    """
    text = config["text_config"]
    rope = text["rope_parameters"]
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
        "rope_theta": float(rope["rope_theta"]),
    }

def validate_config(config: Mapping[str, object], *,
                    vision: bool = True) -> dict[str, object]:
    """Validate the checkpoint against the family contract and summarize it.

    `vision` says whether the artifact will carry the tower; a text-only export is converted
    from a config that still names one, and validating a tower nothing is read from would
    refuse a checkpoint the artifact never touches.
    """

    _check_members("config", config, _ROOT_CONFIG)
    if str(config.get("model_type")) not in _MODEL_TYPES:
        raise ValueError(
            f"config.model_type is {config.get('model_type')!r}, expected one of "
            + ", ".join(_MODEL_TYPES)
        )
    # Only the 27B publishes it, and only ever as False; absent means the same thing.
    if config.get("language_model_only", False):
        raise ValueError("config.language_model_only must be false")
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise ValueError("config.json must contain text_config")
    _check_members("text_config", text, _TEXT_CONFIG)
    missing = [name for name in _TEXT_DIMENSIONS if name not in text]
    if missing:
        raise ValueError("text_config is missing dimensions: " + ", ".join(missing))
    # The schedule is the family's -- three linear layers then every fourth attending -- at
    # whatever depth this checkpoint has.
    geometry_for_schedule = inventory.geometry_from_config(config)
    full = set(geometry_for_schedule.full_attention_layers)
    expected_layer_types = tuple(
        "full_attention" if layer in full else "linear_attention"
        for layer in range(geometry_for_schedule.layers)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match this family's schedule at "
            f"{geometry_for_schedule.layers} layers"
        )
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    _check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    # A text-only export of this family declares no vision tower, and the artifact then omits
    # `vision/*`. The engine probes for the tower rather than assuming it.
    vision_config = config.get("vision_config")
    if vision_config is not None and not isinstance(vision_config, Mapping):
        raise ValueError("config.json vision_config must be an object when present")
    vision_members = {**_VISION_CONFIG, **_vision_dimensions(geometry_for_schedule)}
    if vision and vision_config is not None:
        _check_members("vision_config", vision_config, vision_members)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in (*_TEXT_CONFIG, *_TEXT_DIMENSIONS)},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(full),
            "linear_attention": len(layer_types) - len(full),
            "full_attention_layers": sorted(full),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": ({name: vision_config[name] for name in vision_members}
                   if vision_config is not None else None),
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


#: (resources, text core, draft head, MTP, vision, tensors, objects) per registered size,
#: and what the no-MTP variant of each holds. This restates what the inventory builds, on
#: purpose: it is the check that blocks a conversion run, so a size whose object list grows
#: without this table growing with it fails here rather than after hours of writing.
_SECTION_COUNTS = {
    "qwen3.5-0.8b": ((6, 267, 2, 12, 297, 578, 584), (566, 572)),
    "qwen3.5-2b": ((6, 267, 2, 12, 297, 578, 584), (566, 572)),
    "qwen3.5-4b": ((6, 355, 2, 12, 297, 666, 672), (654, 660)),
    "qwen3.6-27b": ((6, 771, 2, 12, 333, 1118, 1124), (1106, 1112)),
    "qwen3.8-27b": ((6, 771, 2, 12, 333, 1118, 1124), (1106, 1112)),
}


def preflight_inventory(geometry: "inventory.Geometry | None" = None) -> None:
    """Establish the complete target inventory and recipe pairing at this size."""

    geometry = geometry or inventory.GEOMETRY
    model_id = inventory.model_id_for(geometry)
    registered = _SECTION_COUNTS.get(model_id)
    if registered is None:
        raise ValueError(f"no registered inventory for {model_id}")
    complete, without_mtp = registered
    tensors, objects = inventory.active_specs(mtp=True, vision=True, geometry=geometry)
    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.build_text_core_specs(geometry)),
        len(inventory.build_draft_head_specs(geometry)),
        len(inventory.build_mtp_specs(geometry)),
        len(inventory.build_vision_specs(geometry)),
        len(tensors),
        len(objects),
    ) != complete:
        raise ValueError(f"registered inventory is incomplete for {model_id}")
    no_mtp_tensors, no_mtp_objects = inventory.active_specs(
        mtp=False, vision=True, geometry=geometry
    )
    if (len(no_mtp_tensors), len(no_mtp_objects)) != without_mtp:
        raise ValueError(f"registered no-MTP inventory is incomplete for {model_id}")
    if geometry == inventory.GEOMETRY:
        recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path,
                   geometry: "inventory.Geometry | None" = None) -> tuple[ResourcePayload, ...]:
    if geometry is None:
        geometry = inventory.geometry_from_config(_load_config(Path(model_dir)))
    model_id = inventory.model_id_for(geometry)
    if model_id == inventory.QWEN3_8_MODEL_ID:
        return _load_pinned_resources(model_dir, _QWEN3_8_RESOURCE_SHA256)
    if inventory.is_27b(geometry):
        # The 3.6 profile lives in the shared module because the MoE target pins it too, and
        # it downgrades to a warning for a GGUF-derived frontend the way that path needs.
        return official_resources.load_official_resources(
            model_dir, inventory.RESOURCE_SPECS
        )
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def _load_pinned_resources(model_dir: str | Path,
                           expected: Mapping[str, str]) -> tuple[ResourcePayload, ...]:
    """Load exactly the pinned frontend of one generation, refusing any substitution."""
    spec_names = tuple(spec.name for spec in inventory.RESOURCE_SPECS)
    if spec_names != tuple(expected):
        raise ValueError(
            "converter resource inventory does not match the pinned profile: "
            f"expected {tuple(expected)!r}, got {spec_names!r}"
        )
    resources = family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)
    for resource in resources:
        actual = hashlib.sha256(resource.data).hexdigest()
        if actual != expected[resource.name]:
            filename = resource.name.removeprefix("frontend/")
            raise ValueError(
                f"official resource hash mismatch for {filename}: "
                f"expected {expected[resource.name]}, got {actual}"
            )
    return resources


def build_object_plan(
    resources: Mapping[str, bytes], *, mtp: bool = True, vision: bool = True,
    native: Mapping[str, str] | None = None, object_specs=None, geometry=None
) -> ObjectPlan:
    """Compute every payload-relative object offset for the selected variant. `native` names
    the objects served as K-quants verbatim from a GGUF, with their format rewritten."""
    preflight_inventory(geometry)
    if object_specs is None:
        _, object_specs = inventory.active_specs(mtp=mtp, vision=vision,
                                                 geometry=geometry or inventory.GEOMETRY)
    if native:
        object_specs = GgufRepackSource.native_specs(object_specs, native)
    return family_conversion.build_object_plan(object_specs, resources)


def active_recipes(*, mtp: bool, vision: bool = True,
                   geometry=None) -> dict[str, recipe.TensorRecipe]:
    """Recipes for the requested artifact variant, at this checkpoint's size."""
    by_name = (
        dict(recipe.RECIPES_BY_NAME) if geometry is None or geometry == inventory.GEOMETRY
        else {r.object_name: r for r in recipe.build_recipes(geometry)}
    )
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
) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config = _load_config(model)
    config_summary = validate_config(config, vision=vision)
    geometry = inventory.geometry_from_config(config)
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
    draft = draft_head.compute_shortlist(ranking, model)
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
    recipes: Mapping[str, recipe.TensorRecipe] | None = None,
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
        (recipes or recipe.RECIPES_BY_NAME)[spec.name],
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
    geometry: "inventory.Geometry | None" = None,
) -> dict[str, object]:
    """Build the external descriptive conversion report."""

    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(
            inventory.model_id_for(geometry or inventory.GEOMETRY), inventory.WEIGHTS_ID
        ),
        target_key=inventory.TARGET_KEY,
        recipe_id=recipe_id_for(geometry or inventory.GEOMETRY),
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
    geometry = inventory.geometry_from_config(_load_config(model))
    recipes = active_recipes(mtp=mtp, vision=vision, geometry=geometry)
    active_tensor_specs, active_object_specs = inventory.active_specs(
        mtp=mtp, vision=vision, geometry=geometry)
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
            draft_head.compute_shortlist(_tools_root() / draft_head.DEFAULT_RANKING, model)
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
        model, repack, planned + tuple(native) + tuple(sorted(tied)) + tuple(halves) + tuple(in_place), mtp=mtp, vision=vision, native=native, object_specs=active_object_specs
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
            ArtifactIdentity(inventory.model_id_for(geometry), inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
            geometry=geometry_block(preflight.config),
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
    """The module that writes one export's artifact. Imported on use: each pulls in its own
    source-format machinery, and a group-wise conversion needs none of it."""
    from .exports import convert_fp8_block, convert_nvfp4_all, convert_nvfp4_mixed_bf16
    from .exports import convert_nvfp4_mlp_only, convert_nvfp4_uniform

    return {
        inventory.FP8_BLOCK: convert_fp8_block,
        inventory.FP8_CHANNEL: convert_fp8_block,
        inventory.NVFP4_UNIFORM: convert_nvfp4_uniform,
        inventory.NVFP4_MIXED_BF16: convert_nvfp4_mixed_bf16,
        inventory.NVFP4_MLP_ONLY: convert_nvfp4_mlp_only,
        inventory.NVFP4_ALL: convert_nvfp4_all,
    }[profile]


def profile_for_checkpoint(config: Mapping[str, object]) -> str:
    """Which export this checkpoint is, from what it says about itself.

    A BF16 or GGUF-bridged checkpoint declares no quantisation and converts group-wise.
    Beyond that the published releases are one per (size, generation), so the geometry and
    the model type name the export; `--profile` overrides when a size grows a second one.
    """
    quantization = config.get("quantization_config") or {}
    text = json.dumps(quantization)[:2000] if isinstance(quantization, Mapping) else ""
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
    if "NVFP4" not in text and "nvfp4" not in method:
        return inventory.GROUPWISE_INT
    geometry = inventory.geometry_from_config(config)
    if not inventory.is_27b(geometry):
        return inventory.NVFP4_UNIFORM
    if inventory.model_id_for(geometry) == inventory.QWEN3_8_MODEL_ID:
        return inventory.NVFP4_MLP_ONLY
    return inventory.NVFP4_MIXED_BF16


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
    # An export table that names exception layers is a measurement of a published file. Check
    # it against the file in hand: a stale table builds an artifact whose formats do not match
    # its own weights, and nothing downstream would say so.
    _export = inventory.export_for(profile, inventory.geometry_from_config(_config))
    if _export.exceptions:
        from surogate.serve.convert.common import quant_scope as _qs
        _observed = _qs.observed_scope(family_conversion.checkpoint_tensor_names(_model))
        _differ = inventory.exception_disagreement(_export, _observed)
        if _differ:
            _detail = "; ".join(
                f"{role}: the table says {table} and the checkpoint has {found}"
                for role, (table, found) in sorted(_differ.items())
            )
            raise SystemExit(
                f"the {profile} export table does not describe this checkpoint -- {_detail}. "
                f"The tables were measured from published files; this one differs, so the "
                f"artifact would claim formats its own weights do not have."
            )
    if profile == inventory.GROUPWISE_INT:
        convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack,
                mtp=not args.no_mtp, vision=not args.no_vision)
        return
    writer = _export_writer(profile)
    if profile in DUAL_SOURCE_PROFILES:
        if args.quantized_model is None:
            parser.error(f"--profile {profile} reads two checkpoints; pass "
                         "--quantized-model as well as --model")
        writer.convert(args.model, args.quantized_model, args.out, device=args.device)
        return
    writer.convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
