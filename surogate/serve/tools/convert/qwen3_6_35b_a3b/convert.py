"""Convert Qwen3.6-35B-A3B BF16 weights into its exact SInfer artifact.

Canonical invocation::

    python -m tools.convert.qwen3_6_35b_a3b.convert \
      --model /home/densemax2/work/models/hf/qwen/Qwen3.6-35B-A3B/base-hf-bf16 \
      --dflash-model /home/densemax2/work/models/hf/qwen/Qwen3.6-35B-A3B/dflash-bf16 \
      --out out/qwen3_6_35b_a3b.sinfer

The target deliberately reuses the measured 27B ranking because both checkpoints
have the same semantic token-id vocabulary.  Draft rows are always gathered from
the 35B output head.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from surogate.serve.tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from surogate.core.model import quant_schemes
from surogate.serve.tools.convert.common.gguf_repack import (
    GgufRepackSource,
    RepackError,
)
from surogate.serve.tools.convert.common.quantize import pick_device
from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.common import conversion as family_conversion
from surogate.serve.tools.convert.common import official_resources
from surogate.serve.tools.convert.common import recipe as family_recipe

from . import compressed_tensors_source, draft_head, inventory, recipe, routed_nvfp4


RECIPE_ID = "qwen3_6_35b_a3b-v2"
ENCODER_PROFILE = "MAXABS_F16_RECIP_RNE_V1"
GGUF_EVIDENCE_PATH = Path(
    "/home/densemax2/work/models/hf/qwen/Qwen3.6-35B-A3B/"
    "gguf-ud-q4_k_m/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    "num_hidden_layers": 40,
    "full_attention_interval": 4,
    "hidden_size": 2048,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "attn_output_gate": True,
    "hidden_act": "silu",
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "max_position_embeddings": 262144,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
    "mrope_interleaved": True,
    "partial_rotary_factor": 0.25,
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 2048,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [],
}

_DFLASH_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "max_position_embeddings": 262144,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 6,
    "num_key_value_heads": 8,
    "num_target_layers": 40,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4096,
    "tie_word_embeddings": False,
    "use_sliding_window": True,
    "vocab_size": 248320,
}
_DFLASH_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "rope_type": "default",
}
_DFLASH_DRAFT_CONFIG = {
    "block_size": 16,
    "mask_token_id": 248077,
    "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
}

EXPECTED_TENSOR_BYTES = 22_770_245_536
EXPECTED_DEVICE_ARENA_BYTES = 22_770_260_992
EXPECTED_RESIDENT_TENSOR_BYTES = 22_360_191_904
EXPECTED_RESIDENT_DEVICE_ARENA_BYTES = 22_360_207_360
EXPECTED_COMPONENT_BYTES = {
    "main_text": 21_038_461_952,
    "draft_head": 143_130_624,
    "mtp": 897_934_336,
    "vision": 280_664_992,
    "dflash": 410_053_632,
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every checkpoint fact that fixes storage or execution shape."""

    family_conversion.check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)

    expected_layer_types = tuple(
        "full_attention"
        if layer in inventory.FULL_ATTENTION_LAYERS
        else "linear_attention"
        for layer in range(40)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            "text_config.layer_types does not match the target 40-layer schedule"
        )

    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(inventory.FULL_ATTENTION_LAYERS),
            "linear_attention": len(inventory.GDN_LAYERS),
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
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


def validate_dflash_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate every DFlash fact that fixes storage or future execution shape."""

    family_conversion.check_members("dflash config", config, _DFLASH_CONFIG)
    rope = config.get("rope_parameters")
    draft = config.get("dflash_config")
    if not isinstance(rope, Mapping) or not isinstance(draft, Mapping):
        raise ValueError(
            "DFlash config.json must contain rope_parameters and dflash_config"
        )
    family_conversion.check_members(
        "dflash config.rope_parameters",
        rope,
        _DFLASH_ROPE_CONFIG,
    )
    family_conversion.check_members(
        "dflash config.dflash_config",
        draft,
        _DFLASH_DRAFT_CONFIG,
    )
    return {
        name: config[name] for name in _DFLASH_CONFIG
    } | {
        "rope_parameters": {
            name: rope[name] for name in _DFLASH_ROPE_CONFIG
        },
        "dflash_config": {
            name: draft[name] for name in _DFLASH_DRAFT_CONFIG
        },
    }


def preflight_inventory() -> None:
    """Prove the target-private inventory before any payload is written."""

    counts = (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.DFLASH_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    )
    if counts != (6, 533, 2, 15, 333, 51, 934, 940):
        raise ValueError(f"target inventory is incomplete: {counts}")
    if {k: v for k, v in inventory.FORMAT_COUNTS.items() if v} != {
        inventory.BF16: 487,
        inventory.FP32: 60,
        inventory.I32: 1,
        inventory.Q4: 95,
        inventory.Q5: 91,
        inventory.Q6: 5,
        inventory.W8: 195,
    }:
        raise ValueError(f"target format counts drifted: {inventory.FORMAT_COUNTS}")
    if {k: v for k, v in inventory.LAYOUT_COUNTS.items() if v} != {
        inventory.CONTIGUOUS_LAYOUT: 548,
        inventory.ROW_SPLIT_LAYOUT: 386,
    }:
        raise ValueError(f"target layout counts drifted: {inventory.LAYOUT_COUNTS}")
    if family_conversion.tensor_payload_bytes(inventory.TENSOR_SPECS) != EXPECTED_TENSOR_BYTES:
        raise ValueError("target tensor payload byte total drifted")
    if (
        family_conversion.device_arena_bytes(inventory.TENSOR_SPECS)
        != EXPECTED_DEVICE_ARENA_BYTES
    ):
        raise ValueError("target device-arena byte total drifted")
    component_bytes = {
        "main_text": family_conversion.tensor_payload_bytes(
            inventory.TEXT_CORE_TENSOR_SPECS
        ),
        "draft_head": family_conversion.tensor_payload_bytes(
            inventory.DRAFT_HEAD_TENSOR_SPECS
        ),
        "mtp": family_conversion.tensor_payload_bytes(inventory.MTP_TENSOR_SPECS),
        "vision": family_conversion.tensor_payload_bytes(
            inventory.VISION_TENSOR_SPECS
        ),
        "dflash": family_conversion.tensor_payload_bytes(
            inventory.DFLASH_TENSOR_SPECS
        ),
    }
    if component_bytes != EXPECTED_COMPONENT_BYTES:
        raise ValueError(f"target component byte totals drifted: {component_bytes}")
    resident_specs = inventory.TENSOR_SPECS[: -len(inventory.DFLASH_TENSOR_SPECS)]
    if (
        family_conversion.tensor_payload_bytes(resident_specs)
        != EXPECTED_RESIDENT_TENSOR_BYTES
        or family_conversion.device_arena_bytes(resident_specs)
        != EXPECTED_RESIDENT_DEVICE_ARENA_BYTES
    ):
        raise ValueError("default resident Text/MTP/Vision byte totals drifted")
    recipe.validate_recipe_coverage()


def load_resources(
    model_dir: str | Path, *, accept_source: bool = False
) -> tuple[ResourcePayload, ...]:
    return official_resources.load_official_resources(
        model_dir, inventory.RESOURCE_SPECS, accept_source=accept_source
    )


def tensor_specs(routed_nvfp4_source: bool) -> tuple[inventory.TensorSpec, ...]:
    """The tensor half of the inventory for the requested weights profile."""
    return routed_nvfp4.tensor_specs() if routed_nvfp4_source else inventory.TENSOR_SPECS


def object_specs(
    include_dflash: bool, routed_nvfp4_source: bool = False
) -> tuple[inventory.StoredObjectSpec, ...]:
    """The artifact's objects, with the dflash/* family only when a drafter
    checkpoint was supplied. The engine probes for that family and refuses
    --spec dflash against an artifact that lacks it."""
    specs = inventory.RESOURCE_SPECS + tensor_specs(routed_nvfp4_source)
    if include_dflash:
        return specs
    return tuple(spec for spec in specs if spec not in inventory.DFLASH_TENSOR_SPECS)


def build_object_plan(
    resources: Mapping[str, bytes],
    include_dflash: bool = True,
    routed_nvfp4_source: bool = False,
) -> ObjectPlan:
    return family_conversion.build_object_plan(
        object_specs(include_dflash, routed_nvfp4_source), resources
    )


def preflight_conversion(
    model_dir: str | Path,
    dflash_model_dir: str | Path | None,
    routed_nvfp4_dir: str | Path | None = None,
    *,
    shared_expert: str = "as-stored",
    covered: tuple[str, ...] = (),
    object_specs=None,
) -> ConversionPreflight:
    """Complete config, source, shortlist, and offset work before writing.

    ``dflash_model_dir`` is optional: the DFlash drafter is a separate
    checkpoint, and without it the artifact simply omits the dflash/* family
    (speculation stays available through MTP and the draft head)."""

    model = Path(model_dir)
    dflash_model = Path(dflash_model_dir) if dflash_model_dir is not None else None
    # A compressed-tensors export supplies every role itself: its config says which
    # modules are quantized, its packed routed experts come from the same shards, and
    # the modules it left alone are read as the BF16 they are stored in.
    compressed_source = (
        compressed_tensors_source.CompressedTensorsSource(model)
        if quant_schemes.quantization_config_of(family_conversion.load_json(model / "config.json"))
        is not None
        else None
    )
    if compressed_source is not None and routed_nvfp4_dir is None:
        routed_nvfp4_dir = model
    routed_nvfp4_model = Path(routed_nvfp4_dir) if routed_nvfp4_dir is not None else None
    routed_nvfp4_summary = (
        routed_nvfp4.validate_config(
            family_conversion.load_json(routed_nvfp4_model / "config.json")
        )
        if routed_nvfp4_model is not None
        else None
    )
    base_config_summary = validate_config(
        family_conversion.load_json(model / "config.json")
    )
    dflash_config_summary = (
        validate_dflash_config(
            family_conversion.load_json(dflash_model / "config.json")
        )
        if dflash_model is not None
        else None
    )
    preflight_inventory()
    compressed_plan = (
        compressed_source.plan(
            routed_nvfp4.tensor_specs(), recipe.BASE_RECIPES_BY_NAME, shared_expert=shared_expert
        )
        if compressed_source is not None
        else None
    )
    if compressed_plan is None and covered:
        # A GGUF source: the objects it serves from the file need no bridged tensor, so the
        # rest are checked leniently the way the compressed-tensors path is.
        remaining = tuple(
            item for name, item in recipe.BASE_RECIPES_BY_NAME.items() if name not in covered
        )
        base_source = family_recipe.preflight_sources(model, remaining)
    elif compressed_plan is None:
        base_source = recipe.preflight_base_sources(model)
    else:
        # The exact-inventory preflight cannot apply here: the export stores packed
        # tensors the recipes never name and lacks the plain ones they do. The recipes
        # the plan did not take over are checked leniently, as the GGUF path does.
        remaining = tuple(
            item
            for name, item in recipe.BASE_RECIPES_BY_NAME.items()
            if name not in compressed_plan.covered and not routed_nvfp4.is_routed_object(name)
        )
        base_source = family_recipe.preflight_sources(model, remaining)
    dflash_source = (
        recipe.preflight_dflash_sources(dflash_model)
        if dflash_model is not None
        else None
    )

    resources = load_resources(model, accept_source=compressed_source is not None)
    resource_map = {resource.name: resource.data for resource in resources}
    if compressed_plan is None and object_specs is not None:
        specs = inventory.RESOURCE_SPECS + tuple(object_specs)
        if dflash_model is None:
            specs = tuple(spec for spec in specs if spec not in inventory.DFLASH_TENSOR_SPECS)
        object_plan = family_conversion.build_object_plan(specs, resource_map)
    elif compressed_plan is None:
        object_plan = build_object_plan(
            resource_map,
            include_dflash=dflash_model is not None,
            routed_nvfp4_source=routed_nvfp4_model is not None,
        )
    else:
        specs = inventory.RESOURCE_SPECS + compressed_plan.specs
        if dflash_model is None:
            specs = tuple(spec for spec in specs if spec not in inventory.DFLASH_TENSOR_SPECS)
        object_plan = family_conversion.build_object_plan(specs, resource_map)

    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        dflash_model_dir=dflash_model,
        base_config_summary=base_config_summary,
        dflash_config_summary=dflash_config_summary,
        base_source=base_source,
        dflash_source=dflash_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
        routed_nvfp4_dir=routed_nvfp4_model,
        routed_nvfp4_summary=routed_nvfp4_summary,
        compressed_source=compressed_source,
        compressed_plan=compressed_plan,
    )


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
    ranking_path: str | Path,
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
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
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
    report["draft_head"] = {
        "rows": draft_head.DRAFT_HEAD_N,
        "tokenizer_vocab_size": draft_head.TOKENIZER_VOCAB_SIZE,
        "ranking_source_target": draft_head.RANKING_SOURCE_TARGET,
        "shared_semantic_vocabulary": True,
    }
    report["source"]["gguf_evidence_path"] = str(GGUF_EVIDENCE_PATH)
    report["quantization"] = {
        "encoder_profile": ENCODER_PROFILE,
        "component_tensor_bytes": {
            **EXPECTED_COMPONENT_BYTES,
            "total": EXPECTED_TENSOR_BYTES,
            "all_tensor_device_arena": EXPECTED_DEVICE_ARENA_BYTES,
            "default_resident": EXPECTED_RESIDENT_TENSOR_BYTES,
            "default_resident_device_arena": (
                EXPECTED_RESIDENT_DEVICE_ARENA_BYTES
            ),
        },
    }
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
    gguf_specs = tensor_specs(routed_nvfp4_dir is not None)
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
            recipe.BASE_RECIPES_BY_NAME,
            gguf_specs,
            exclude_suffixes=recipe.NATIVE_EXCLUDE_SUFFIXES,
        )
        repacked = repack.plan(recipe.BASE_RECIPES_BY_NAME, gguf_specs)
        if native:
            # By default those objects are not copied at all: the artifact names the GGUF and
            # the stretches of it each object reads. SUROGATE_GGUF_COPY=1 writes the bytes in.
            if os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
                # The draft head gathers its rows by shortlist rather than in order, so it needs
                # the same ids the write path uses. The shortlist is a pure function of the
                # ranking file and the checkpoint, so computing it here matches what preflight
                # computes later.
                draft_ids = draft_head.materialize_draft_head_token_ids(
                    draft_head.compute_shortlist(_repo_root() / draft_head.DEFAULT_RANKING, model)
                )
                native_runs = {
                    spec.name: repack.runs_for_native(
                        spec,
                        recipe.BASE_RECIPES_BY_NAME[spec.name],
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
                recipe.BASE_RECIPES_BY_NAME, gguf_specs, recipe.NATIVE_EXCLUDE_SUFFIXES
            )
            if in_place:
                gguf_specs = GgufRepackSource.in_place_specs(gguf_specs, in_place)
                moved = sum(sum(r[2] for r in entry[0]) for entry in in_place.values())
                print(f"rearranged at load: {len(in_place)} objects read from the GGUF "
                      f"({moved / 1e9:.1f} GB not copied)", flush=True)
                repacked = tuple(n for n in repacked if n not in in_place)
    preflight = preflight_conversion(
        model, dflash_model, routed_nvfp4_dir, shared_expert=shared_expert,
        covered=tuple(native) + tuple(repacked) + tuple(in_place) + tuple(
            n for n in recipe.BASE_RECIPES_BY_NAME if n.startswith(dropped_prefixes)
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
        ),
        preflight.object_plan.specs,
        external=external,
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

        for spec in inventory.RESOURCE_SPECS:
            write_payload(spec, resources[spec.name])

        all_specs = (
            gguf_specs
            if (native or repacked or dropped_prefixes)
            else preflight.compressed_plan.specs
            if preflight.compressed_plan is not None
            else tensor_specs(preflight.routed_nvfp4_dir is not None)
        )
        # A GGUF plan rewrites formats and may drop whole families, so the DFlash tail cannot be
        # sliced off by length any more.
        dflash_names = {spec.name for spec in inventory.DFLASH_TENSOR_SPECS}
        base_specs = tuple(spec for spec in all_specs if spec.name not in dflash_names)
        routed_reader = (
            ShardReader.from_index(
                preflight.routed_nvfp4_dir / "model.safetensors.index.json"
            )
            if preflight.routed_nvfp4_dir is not None
            else None
        )
        routed_cache = routed_nvfp4.LayerCache(routed_reader) if routed_reader else None
        try:
            with ShardReader.from_index(
                model / "model.safetensors.index.json"
            ) as reader:
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
                        source_recipe = recipe.BASE_RECIPES_BY_NAME[spec.name]
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
                        recipe.BASE_RECIPES_BY_NAME,
                    )
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                    write_payload(spec, payload)
                    del payload
        finally:
            if routed_reader is not None:
                routed_reader.close()

        if dflash_model is not None:
            with ShardReader.from_file(
                dflash_model / "model.safetensors"
            ) as reader:
                for spec in all_specs[-len(inventory.DFLASH_TENSOR_SPECS) :]:
                    tensor = materialize_tensor(
                        spec,
                        reader,
                        preflight.draft,
                        recipe.DFLASH_RECIPES_BY_NAME,
                    )
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                    write_payload(spec, payload)
                    del payload

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
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
