"""Convert Spark-X2.5 safetensors into a serving artifact."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, fields
from pathlib import Path

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.artifact.geometry import validate_resolved_geometry
from surogate.serve.convert.common import conversion
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import materialize_recipe, validate_recipe_coverage
from surogate.serve.convert.llama.convert import load_resources as load_text_resources

from . import inventory, recipe

RECIPE_ID = "spark2_5-v1"


def load_resources(model_dir):
    resources = {item.name: item.data for item in load_text_resources(model_dir)}
    template = resources.get("frontend/chat_template.jinja")
    if template is not None:
        # HF gives the standalone template precedence. Spark publishes a formatted
        # standalone version and a different serialization in tokenizer_config.json.
        config = json.loads(resources["frontend/tokenizer_config.json"])
        config["chat_template"] = template.decode("utf-8")
        resources["frontend/tokenizer_config.json"] = json.dumps(config, ensure_ascii=False).encode("utf-8")
    return tuple(conversion.ResourcePayload(name, data) for name, data in resources.items())


def geometry_block(g: inventory.Geometry, *, token_domain: int):
    return validate_resolved_geometry(
        {
            "hidden": g.hidden,
            "residual": g.hidden,
            "residual_fp32": 1,
            "layers": g.layers,
            "intermediate": g.intermediate,
            "output_rows": g.vocab,
            "token_domain": token_domain,
            "query_heads": g.query_heads,
            "kv_heads": g.kv_heads,
            "head_dim": g.head_dim,
            "rotary_dim": g.rotary_dim,
            "sliding_rotary_dim": g.sliding_rotary_dim,
            "rms_epsilon": g.rms_epsilon,
            "rope_theta": g.rope_theta,
            "sliding_rope_theta": g.sliding_rope_theta,
            "sliding_window": g.sliding_window,
            "max_context": g.max_context,
            "attention_scale": g.head_dim**-0.5,
        }
    )


def validate_config(config):
    g = recipe.geometry_from_config(config)
    return g, {f.name: getattr(g, f.name) for f in fields(g) if f.name != "declared"}


def convert(model_dir, out_path, *, device="cuda") -> Path:
    started = time.perf_counter()
    model, output = Path(model_dir), Path(out_path)
    config = conversion.load_json(model / "config.json")
    g, summary = validate_config(config)
    if config.get("quantization_config"):
        raise ValueError("Spark conversion requires unquantized safetensors")
    geometry = geometry_block(g, token_domain=tokenizer_domain(model))
    recipes = recipe.build_recipes(g)
    specs = inventory.build_tensor_specs(g)
    validate_recipe_coverage(recipes, specs)
    source = recipe.preflight_sources(model, recipes)
    resources = {item.name: item.data for item in load_resources(model)}
    plan = conversion.build_object_plan(inventory.build_object_specs(g), resources)
    by_name = {r.object_name: r for r in recipes}
    resolved_device = pick_device(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"preflight complete: {len(plan.objects)} objects, {source.source_tensor_count} source tensors", flush=True)
    with (
        recipe.open_reader(model) as reader,
        ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID, architecture=inventory.TARGET_KEY),
            plan.specs,
            geometry=geometry,
            layer_types=list(g.layer_types),
        ) as writer,
    ):
        for i, spec in enumerate(inventory.build_object_specs(g), 1):
            if isinstance(spec, inventory.ResourceSpec):
                if spec.name not in resources:
                    continue
                payload = resources[spec.name]
            else:
                tensor = materialize_recipe(by_name[spec.name], reader)
                payload = conversion.encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
            writer.write(spec.name, payload)
            del payload
            print(f"[{i}/{len(plan.objects)}] {spec.name}", flush=True)
    report = {
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {"model_path": str(model.resolve())},
        "config_summary": summary,
        "arguments": {"model": str(model_dir), "out": str(out_path), "device": str(device)},
        "source_preflight": asdict(source),
        "converter": {
            "environment": conversion.environment(resolved_device),
            "revision": conversion.converter_revision(Path(__file__).resolve().parents[4]),
        },
        "objects": conversion.object_statistics(plan.objects),
        "artifact": {"path": str(output), "bytes": output.stat().st_size},
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
