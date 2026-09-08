"""Build an LFM2 serving artifact from safetensors or a bridged GGUF checkpoint.

Compatible GGUF linear weights are repacked into W8 without changing their values.
Other quantizations are dequantized by the bridge and encoded into the same W8
profile. Norms, embeddings and convolution taps use BF16.

What is worth saying about the shape is said in `inventory.py`, which derives it
from the declaration rather than restating it. This module is the driver: read the
config, check the checkpoint has what the recipes name, then write each object in
plan order.
"""

from __future__ import annotations

from surogate.serve.convert.common.checkpoint import tokenizer_domain

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.gguf_repack import GgufRepackSource, RepackError
from surogate.serve.convert.common.recipe import expression_sources, materialize_recipe

from . import inventory, recipe

RECIPE_ID = "lfm2-v1"

#: What the config must state for the artifact to be shaped at all. Everything else
#: -- the FFN adjustment, the layer schedule -- is derived through the declaration,
#: which is where those rules already live.
_REQUIRED_CONFIG = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "conv_L_cache",
)


def validate_config(config: Mapping[str, object]) -> inventory.Geometry:
    absent = [name for name in _REQUIRED_CONFIG if config.get(name) is None]
    if absent:
        raise ValueError(
            "this checkpoint's config.json does not state "
            + ", ".join(absent)
            + "; an LFM2 artifact cannot be shaped without them"
        )
    return inventory.geometry_from_config(config)


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    gguf_repack: str | Path | None = None,
) -> Path:
    """Run the conversion and return the path of its report."""
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)

    config = family_conversion.load_json(model / "config.json")
    geometry = validate_config(config)
    family_conversion.honour_declared_scope(
        config, geometry, model, what=family_conversion.checkpoint_label(model)
    )

    objects = inventory.declared_objects(geometry)
    tensor_specs = inventory.tensor_specs(objects)
    recipes = {r.object_name: r for r in recipe.build_recipes(geometry)}
    repack = GgufRepackSource(gguf_repack) if gguf_repack is not None else None
    planned = repack.plan(recipes, tensor_specs) if repack is not None else ()
    covered = set(planned)
    remaining = tuple(r for name, r in recipes.items() if name not in covered)
    if repack is not None:
        stray = {s.name for r in remaining for s in expression_sources(r.expression)
                 if s.name in repack.sources}
        if stray:
            raise RepackError("repack map names sources still needed by materialized recipes: "
                              + ", ".join(sorted(stray)))

    resources = family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
    resource_payloads = {item.name: item.data for item in resources}
    plan = family_conversion.build_object_plan(
        tuple(tensor_specs) + tuple(inventory.RESOURCE_SPECS), resource_payloads
    )

    with recipe.open_reader(model) as reader:
        source = recipe.preflight_sources(reader, remaining)
        print(
            f"preflight complete: {len(plan.objects)} objects, "
            f"{source.source_tensor_count} source tensors, device={resolved_device}",
            flush=True,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID, architecture="lfm2"),
            plan.specs,
            geometry=_geometry_block(geometry, token_domain=tokenizer_domain(model)),
            layer_types=geometry.layer_types,
        ) as writer:
            total = len(plan.specs)
            for index, spec in enumerate(plan.specs, start=1):
                if spec.name in resource_payloads:
                    payload = resource_payloads[spec.name]
                elif repack is not None and spec.name in planned:
                    payload = repack.payload_for(spec, recipes[spec.name], None)
                else:
                    tensor = materialize_recipe(recipes[spec.name], reader)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID, architecture="lfm2")
    report = family_conversion.build_conversion_report(
        identity=identity,
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=Path(__file__).resolve().parents[4],
        ranking_path=model,
        model_dir=model,
        out_path=output,
        arguments={"model": str(model_dir), "out": str(out_path), "device": requested_device,
                   "gguf_repack": str(gguf_repack) if gguf_repack is not None else None},
        config_summary={
            "architecture": inventory.ARCHITECTURE,
            "hidden": geometry.hidden,
            "layers": geometry.layers,
            "intermediate": geometry.intermediate,
            "conv_kernel": geometry.conv_kernel,
            "attention_layers": list(geometry.attention_layers),
        },
        source_preflight=source,
        objects=plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def _geometry_block(geometry: inventory.Geometry, *, token_domain: int) -> dict[str, float]:
    """Serialize dimensions and execution settings from the resolved checkpoint."""
    from surogate.serve.convert.common.checkpoint import dense_geometry

    metadata = dense_geometry(geometry, token_domain=token_domain)
    metadata.update(gdn_conv_kernel=geometry.conv_kernel)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path,
                        help="GGUF source map written by the ingestion bridge")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack)


if __name__ == "__main__":
    main()
