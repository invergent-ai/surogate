"""Prepare Gemma 3/4 text and vision weights for native image/video serving."""

import argparse
import json
from pathlib import Path

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.inventory import RESOURCE_SPECS
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import materialize_recipe, preflight_source_reader, validate_recipe_coverage
from surogate.serve.convert.common.safetensors import ShardReader

from . import inventory


def resources_for(model, g):
    resources = {r.name: r.data for r in conversion.load_resources(model, RESOURCE_SPECS)}
    config, v = g.config, g.vision
    image = json.loads(resources["frontend/preprocessor_config.json"])
    token_config = json.loads(resources["frontend/tokenizer_config.json"])

    def marker(name, token_id):
        value = token_config.get(name)
        if isinstance(value, dict):
            value = value.get("content")
        if isinstance(value, str):
            return value
        tokenizer = json.loads(resources["frontend/tokenizer.json"])
        for added in tokenizer.get("added_tokens", []):
            if added["id"] == token_id:
                return added["content"]
        vocab = tokenizer["model"]["vocab"]
        if isinstance(vocab, list):
            return vocab[token_id][0]
        return next(key for key, value in vocab.items() if value == token_id)

    image_id = config.get("image_token_id", config.get("image_token_index"))
    video_id = config.get("video_token_id", image_id)
    if v["gemma_version"] == 4:
        tok = json.loads(resources["frontend/tokenizer.json"])
        video_id = next((a["id"] for a in tok.get("added_tokens", []) if a["content"] == "<|video|>"), video_id)
    image.update(
        gemma_version=v["gemma_version"],
        encoder_free=v["encoder_free"],
        image_processor_type="Gemma3ImageProcessor" if v["gemma_version"] == 3 else "Gemma4ImageProcessor",
        image_token_id=image_id,
        video_token_id=video_id,
        image_token=marker("image_token", image_id),
        video_token=marker("video_token", video_id),
        boi_token=marker("boi_token", config.get("boi_token_id", config.get("boi_token_index", 255999))),
        eoi_token=marker("eoi_token", config.get("eoi_token_id", config.get("eoi_token_index", 256000))),
        patch_size=int((v["patch_dim"] // 3) ** 0.5),
        spatial_merge_size=v["merge"],
        image_seq_length=config.get("mm_tokens_per_image", 280),
        position_embeddings=v["position_embeddings"],
        do_pan_and_scan=bool(image.get("do_pan_and_scan")),
        pan_and_scan_min_crop_size=image.get("pan_and_scan_min_crop_size") or 256,
        pan_and_scan_max_num_crops=image.get("pan_and_scan_max_num_crops") or 4,
        pan_and_scan_min_ratio_to_activate=image.get("pan_and_scan_min_ratio_to_activate") or 1.2,
    )
    resources["frontend/preprocessor_config.json"] = json.dumps(image).encode()
    return resources


def convert(model_dir, out_path, *, device="cpu"):
    model, output = Path(model_dir), Path(out_path)
    config = json.loads((model / "config.json").read_text())
    g = inventory.geometry_from_config(config)
    text_specs, text_recipes = inventory.text_specs_and_recipes(g)
    vision_specs, vision_recipes = inventory.vision_recipes(g)
    specs, recipes = (*text_specs, *vision_specs), (*text_recipes, *vision_recipes)
    validate_recipe_coverage(recipes, specs)
    resources = resources_for(model, g)
    plan = conversion.build_object_plan((*specs, *RESOURCE_SPECS), resources)
    recipe_map = {r.object_name: r for r in recipes}
    geometry = inventory.geometry_block(g, token_domain=tokenizer_domain(model))
    with ShardReader.for_directory(model) as reader:
        preflight_source_reader(reader, recipes)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(g.target, "groupwise-int", architecture=g.target),
            plan.specs,
            geometry=geometry,
            vision_geometry=g.vision,
            layer_types=g.text.layer_types,
        ) as writer:
            for spec in plan.specs:
                payload = resources.get(spec.name)
                if payload is None:
                    tensor = materialize_recipe(recipe_map[spec.name], reader)
                    if spec.format == "FP32":
                        tensor = tensor.to(torch.float32)
                    payload = conversion.encode_tensor_payload(tensor, spec, pick_device(device))
                writer.write(spec.name, payload)
                print(spec.name, flush=True)
    Path(str(output) + ".conversion.json").write_text(
        json.dumps(
            {"architecture": g.target, "model": str(model), "vision": g.vision, "objects": len(plan.specs)}, indent=2
        )
        + "\n"
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
