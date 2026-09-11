"""Convert a Qwen3-VL dense or MoE checkpoint for text, image, and video serving."""

import argparse
import json
import time
from pathlib import Path

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.official_resources import chat_template_bytes, tokenizer_config_with_template
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import materialize_recipe, validate_recipe_coverage

from . import inventory, recipe

validate_config = inventory.geometry_from_config


def convert(model_dir, out_path, *, device="cuda", vision_storage=inventory.VISION_BF16):
    started = time.perf_counter()
    model, output = Path(model_dir), Path(out_path)
    device = pick_device(device)
    config = conversion.load_json(model / "config.json")
    geometry = validate_config(config)
    conversion.honour_declared_scope(config, geometry, model, what=conversion.checkpoint_label(model))
    tensors = inventory.build_tensor_specs(geometry, vision_storage=vision_storage)
    with recipe.open_reader(model) as reader:
        recipes = recipe.build_recipes(geometry, reader=reader)
    validate_recipe_coverage(recipes, tensors)
    source = recipe.preflight_sources(model, recipes)
    resources = {r.name: r.data for r in conversion.load_resources(model, inventory.RESOURCE_SPECS)}
    template = chat_template_bytes(model)
    if template is not None:
        resources["frontend/chat_template.jinja"] = template
        resources["frontend/tokenizer_config.json"] = tokenizer_config_with_template(
            resources["frontend/tokenizer_config.json"], template)
    if "frontend/preprocessor_config.json" not in resources:
        raise ValueError("Qwen3-VL requires preprocessor_config.json for its image processor")
    resources = {s.name: resources[s.name] for s in inventory.RESOURCE_SPECS if s.name in resources}
    plan = conversion.build_object_plan(
        inventory.build_object_specs(geometry, vision_storage=vision_storage), resources)
    recipes = {r.object_name: r for r in recipes}
    identity = ArtifactIdentity(geometry.architecture, inventory.WEIGHTS_ID, architecture=geometry.architecture)
    print(f"preflight complete: {len(plan.objects)} objects, {source.source_tensor_count} source tensors", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with recipe.open_reader(model) as reader, ArtifactWriter(
        output, identity, plan.specs,
        geometry=inventory.geometry_block(geometry, token_domain=tokenizer_domain(model)),
        layer_types=["full_attention"] * geometry.layers, vision_geometry=geometry.vision,
    ) as writer:
        for i, spec in enumerate(plan.specs, 1):
            if spec.name in resources:
                payload = resources[spec.name]
            else:
                tensor = materialize_recipe(recipes[spec.name], reader)
                payload = conversion.encode_tensor_payload(tensor, spec, device)
                del tensor
            writer.write(spec.name, payload)
            del payload
            print(f"[{i}/{len(plan.specs)}] {spec.name}", flush=True)
    report = conversion.build_conversion_report(
        identity=identity, target_key=inventory.TARGET_KEY, recipe_id="qwen3_vl-v1",
        repo_root=Path(__file__).resolve().parents[4], ranking_path=model, model_dir=model,
        out_path=output, arguments={"model": str(model), "out": str(output), "device": str(device)},
        config_summary=config, source_preflight=source, objects=plan.objects,
        elapsed_seconds=time.perf_counter() - started, final_bytes=output.stat().st_size, device=device,
    )
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"complete: {output}; report={report_path}", flush=True)
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vision-storage", choices=inventory.VISION_STORAGE,
                        default=inventory.VISION_BF16,
                        help="How to store the vision tower. `bf16` is the weights the "
                             "checkpoint ships, and the default. `quantized` is smaller and "
                             "measurably further from the source tower.")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device,
            vision_storage=args.vision_storage)


if __name__ == "__main__":
    main()
