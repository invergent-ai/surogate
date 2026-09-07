"""Build an LFM2 serving artifact from a HuggingFace checkpoint.

Safetensors in, artifact out. There is no GGUF repack path here yet: LFM2 GGUFs
exist, but reading one in place is a separate piece of work from serving the
architecture at all, and the two are better landed apart.

What is worth saying about the shape is said in `inventory.py`, which derives it
from the declaration rather than restating it. This module is the driver: read the
config, check the checkpoint has what the recipes name, then write each object in
plan order.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import materialize_recipe

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
    if gguf_repack is not None:
        raise NotImplementedError(
            "the LFM2 target converts from safetensors; reading a GGUF in place is "
            "not implemented for this architecture yet"
        )

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

    objects = inventory.declared_objects(config)
    tensor_specs = inventory.tensor_specs(objects)
    recipes = {r.object_name: r for r in recipe.build_recipes(config)}

    resources = family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
    resource_payloads = {item.name: item.data for item in resources}
    plan = family_conversion.build_object_plan(
        tuple(tensor_specs) + tuple(inventory.RESOURCE_SPECS), resource_payloads
    )

    with recipe.open_reader(model) as reader:
        source = recipe.preflight_sources(reader, tuple(recipes.values()))
        print(
            f"preflight complete: {len(plan.objects)} objects, "
            f"{source.source_tensor_count} source tensors, device={resolved_device}",
            flush=True,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            plan.specs,
            geometry=_geometry_block(geometry),
        ) as writer:
            total = len(plan.specs)
            for index, spec in enumerate(plan.specs, start=1):
                if spec.name in resource_payloads:
                    payload = resource_payloads[spec.name]
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
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    report = family_conversion.build_conversion_report(
        identity=identity,
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=Path(__file__).resolve().parents[4],
        ranking_path=model,
        model_dir=model,
        out_path=output,
        arguments={"model": str(model_dir), "out": str(out_path), "device": requested_device},
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


def _geometry_block(geometry: inventory.Geometry) -> dict[str, float]:
    """The dimensions the artifact states about itself, which the engine's binder
    validates its compiled constants against."""
    return {
        "hidden": float(geometry.hidden),
        "layers": float(geometry.layers),
        "query_heads": float(geometry.query_heads),
        "kv_heads": float(geometry.kv_heads),
        "head_dim": float(geometry.head_dim),
        "intermediate": float(geometry.intermediate),
        "vocab": float(geometry.vocab),
        "conv_kernel": float(geometry.conv_kernel),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
