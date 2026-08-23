"""Build the additive Qwen3.6-27B NVFP4 artifact from two fixed source roles.

Canonical invocation::

    python3 -m tools.convert.qwen3_6_27b.convert_nvfp4 \
      --model /path/to/Qwen3.6-27B/base-hf-bf16 \
      --nvfp4-model /path/to/Qwen3.6-27B/vllm-nvfp4-bf16 \
      --out out/qwen3_6_27b_nvfp4.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.artifact.layouts import (
    decode_nvfp4_words,
    encode_direct,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import convert as base_convert
from . import draft_head
from . import inventory_nvfp4 as inventory
from . import recipe as base_recipe
from . import recipe_nvfp4 as recipe


RECIPE_ID = "qwen3_6_27b_nvfp4-v1"
OUTPUT_BASENAME = "qwen3_6_27b_nvfp4.ninfer"


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    base_dir: Path
    nvfp4_dir: Path
    config_summary: dict[str, object]
    base_source: base_recipe.SourcePreflight
    nvfp4_dtype_counts: dict[str, int]
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard, str)
        or not shard
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path}: invalid weight_map entry")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(
            f"{model_dir}: safetensors shard set does not match the index"
        )
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def _validate_nvfp4_config(config: Mapping[str, object]) -> dict[str, object]:
    summary = base_convert.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("NVFP4 config is missing quantization_config")
    family_conversion.check_members(
        "quantization_config",
        quantization,
        {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
        },
    )
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or tuple(groups) != ("group_0",):
        raise ValueError("NVFP4 config must contain exactly config_groups.group_0")
    group = groups["group_0"]
    if not isinstance(group, Mapping):
        raise ValueError("quantization_config.config_groups.group_0 must be an object")
    family_conversion.check_members(
        "quantization_config.config_groups.group_0",
        group,
        {"format": "nvfp4-pack-quantized"},
    )
    weights = group.get("weights")
    activations = group.get("input_activations")
    if not isinstance(weights, Mapping) or not isinstance(activations, Mapping):
        raise ValueError("NVFP4 config is missing weight or activation settings")
    common = {
        "num_bits": 4,
        "type": "float",
        "strategy": "tensor_group",
        "group_size": 16,
        "symmetric": True,
        "scale_dtype": "torch.float8_e4m3fn",
    }
    family_conversion.check_members(
        "quantization_config.config_groups.group_0.weights",
        weights,
        {**common, "dynamic": False},
    )
    family_conversion.check_members(
        "quantization_config.config_groups.group_0.input_activations",
        activations,
        {**common, "dynamic": "local"},
    )
    return summary


def _validate_source_manifest(model_dir: Path) -> None:
    value = family_conversion.load_json(model_dir / "mixed_native_manifest.json")
    family_conversion.check_members(
        "mixed_native_manifest",
        value,
        {
            "format_histogram": {
                "head_passthrough/BF16": 3,
                "linear/BF16": 117,
                "linear/NVFP4": 379,
                "layer_passthrough/BF16": 352,
                "mtp_linear/NVFP4": 7,
                "mtp_passthrough/BF16": 8,
            },
            "n_assignment_entries": 614,
            "ignore": ["lm_head"],
            "prune": None,
            "passthrough_dtype_policy": (
                "source_dtype_preserved; corrected from silent norm fp32 upcast"
            ),
            "corrected_norm_passthrough_tensors": 216,
        },
    )


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    model_dir: str | Path,
    nvfp4_model_dir: str | Path,
) -> ConversionPreflight:
    base = Path(model_dir)
    nvfp4 = Path(nvfp4_model_dir)
    _validate_index(base)
    _validate_index(nvfp4)
    base_summary = base_convert.validate_config(
        family_conversion.load_json(base / "config.json")
    )
    nvfp4_summary = _validate_nvfp4_config(
        family_conversion.load_json(nvfp4 / "config.json")
    )
    if base_summary != nvfp4_summary:
        raise ValueError("base and NVFP4 source model configs do not match")
    _validate_source_manifest(nvfp4)
    preflight_inventory()

    base_source = base_recipe.preflight_sources(base)
    with ShardReader(nvfp4) as nvfp4_reader:
        nvfp4_dtype_counts = recipe.preflight_metadata(nvfp4_reader)
        recipe.validate_nvfp4_words(nvfp4_reader)
    with ShardReader(base) as base_reader, ShardReader(nvfp4) as nvfp4_reader:
        recipe.compare_bf16_sources(base_reader, nvfp4_reader)

    resources = base_convert.load_resources(base)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, base)
    return ConversionPreflight(
        base_dir=base,
        nvfp4_dir=nvfp4,
        config_summary=base_summary,
        base_source=base_source,
        nvfp4_dtype_counts=nvfp4_dtype_counts,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_nvfp4_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(selected, reader)
    payload = encode_nvfp4(packed, scales, divisor, spec.shape)
    decoded_packed, decoded_scales, decoded_divisor = decode_nvfp4_words(
        payload, spec.shape
    )
    if (
        not torch.equal(decoded_packed, packed)
        or not torch.equal(decoded_scales, scales)
        or bytes(decoded_divisor.reshape(1).view(torch.uint8).numpy()) != divisor
    ):
        raise RuntimeError(f"{spec.name}: NVFP4 layout word verification failed")
    return payload


def _is_fused_text_attention_parent(name: str) -> bool:
    return name.startswith("text/layers/") and name.endswith(
        "/attention/query_key_gate_value"
    )


def _materialize_base_tensor(
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
            draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT:
            draft_head.materialize_draft_head_token_ids(draft)
        }
    if _is_fused_text_attention_parent(spec.name):
        prefix = spec.name.removesuffix("query_key_gate_value")
        tensor = torch.cat(
            (
                base_recipe.materialize_recipe(
                    base_recipe.RECIPES_BY_NAME[prefix + "query_key"],
                    reader,
                ),
                base_recipe.materialize_recipe(
                    base_recipe.RECIPES_BY_NAME[prefix + "gate_value"],
                    reader,
                ),
            ),
            dim=0,
        )
    else:
        tensor = base_recipe.materialize_recipe(
            base_recipe.RECIPES_BY_NAME[spec.name],
            reader,
            derived,
        )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.base_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.base_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "base": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.base_dir.resolve()),
        },
        "nvfp4": {
            "repository": recipe.NVFP4_REPOSITORY,
            "revision": recipe.NVFP4_REVISION,
            "model_path": str(preflight.nvfp4_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "base": {
            "recipes": preflight.base_source.recipe_count,
            "tensors": preflight.base_source.source_tensor_count,
            "shards": preflight.base_source.source_shard_count,
            "dtypes": dict(preflight.base_source.source_dtype_counts),
        },
        "nvfp4": {
            "linear_nvfp4": len(recipe.NVFP4_SOURCES),
            "linear_bf16": len(recipe.BF16_COMPARISONS),
            "dtypes": dict(preflight.nvfp4_dtype_counts),
        },
    }
    return report


def convert(
    model_dir: str | Path,
    nvfp4_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed dual-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"NVFP4 converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir, nvfp4_model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source linears, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ShardReader(preflight.base_dir) as base_reader, ShardReader(
        preflight.nvfp4_dir
    ) as nvfp4_reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.format == inventory.NVFP4:
                    payload = _encode_nvfp4_weight(spec, nvfp4_reader)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name],
                        nvfp4_reader,
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                else:
                    tensor = _materialize_base_tensor(
                        spec, base_reader, preflight.draft
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "nvfp4_model": str(nvfp4_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
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
    parser.add_argument("--nvfp4-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.nvfp4_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
