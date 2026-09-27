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

#: What a checkpoint's non-routed text matrices are stored as. The engine binds whichever format
#: the artifact states, so this is a storage choice, not a model variant: `w8` is the groupwise
#: int8 every Gemma artifact has used, `bf16` keeps the checkpoint's own words.
#:
#: The default follows the checkpoint. A BF16 checkpoint keeps `w8`, the artifact it always
#: produced. A quantised export keeps what it stored: the modules its declaration left alone stay
#: BF16, as the exporter shipped them -- which on Gemma 4 is also the fast route, because the
#: 2,112-wide dense feed-forward is not a shape the W8 tensor-core kernels take and runs on their
#: SIMT fallback (docs/inference/serving-models.md, "Gemma 4 mixture in NVFP4").
TEXT_FORMATS = ("w8", "bf16")


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


def _text_storage(specs, text_format):
    """The text specs with every dense per-layer matrix in `text_format`. The embedding
    table (and the head tied to it) keeps W8 either way: it is the largest single object and
    the decode step reads one of its rows per token."""
    if text_format not in TEXT_FORMATS:
        raise ValueError(f"text format must be one of {TEXT_FORMATS}, not {text_format!r}")
    if text_format == "w8":
        return tuple(specs)
    from dataclasses import replace

    from surogate.serve.convert.common.inventory import BF16, CONTIGUOUS_LAYOUT, W8

    # The routed experts are not dense matrices: the MoE kernels take them W8 or quantised, never
    # BF16, and they keep whatever the checkpoint and the profile gave them.
    return tuple(
        replace(spec, format=BF16, layout=CONTIGUOUS_LAYOUT)
        if spec.format == W8 and spec.name.startswith("text/layers/") and "/moe/routed_" not in spec.name
        else spec
        for spec in specs
    )


def _dense_replaces(name, dense_sources):
    """Whether an NVFP4 dense object stands in for this base object: itself, or -- for the dense
    gate and up -- the fused `mlp/gate_up` of its layer."""
    if name in dense_sources:
        return True
    for half in ("/mlp/gate", "/mlp/up"):
        if name.endswith(half) and name[: -len(half)] + "/mlp/gate_up" in dense_sources:
            return True
    return False


def convert(model_dir, out_path, *, device="cpu", text_format=None, dflash_model=None):
    model, output = Path(model_dir), Path(out_path)
    config = json.loads((model / "config.json").read_text())
    g = inventory.geometry_from_config(config)
    text_specs, text_recipes = inventory.text_specs_and_recipes(g)
    # A separate DFlash drafter trained against this target (z-lab's Gemma 4 drafters): its
    # objects come from its own checkpoint, and its geometry is written beside the target's.
    dflash_geometry = None
    dflash_specs, dflash_recipes = (), ()
    if dflash_model is not None:
        if g.target != "gemma4_moe":
            raise ValueError(f"{g.target}: DFlash is served for the Gemma 4 mixture only")
        from surogate.serve.convert.common import dflash as dflash_checkpoint

        dflash_model = Path(dflash_model)
        dflash_geometry = dflash_checkpoint.geometry_from_config(
            json.loads((dflash_model / "config.json").read_text()), g.text)
        dflash_specs, dflash_recipes = dflash_checkpoint.conversion_plan(dflash_geometry, g.text)
    # An NVFP4 export of the mixture quantises its routed experts, which the checkpoint stores
    # per expert rather than as the stacked tensors the base recipes read: those objects come
    # from `routed_nvfp4` instead, and every other text object keeps its recipe.
    convention = None
    if g.target == "gemma4_moe":
        from surogate.serve.convert.gemma4_moe.exports import routed_nvfp4

        convention = routed_nvfp4.convention_of(config)
    elif config.get("quantization_config"):
        raise ValueError(f"{g.target}: quantised Gemma checkpoints are served for the mixture only")
    weights_id = "groupwise-int"
    routed_geometry = None
    if convention is not None:
        routed_geometry = routed_nvfp4.geometry_of(g.text)
        text_specs = routed_nvfp4.tensor_specs(text_specs, routed_geometry.experts)
        text_recipes = tuple(r for r in text_recipes if not routed_nvfp4.is_routed_object(r.object_name))
        weights_id = routed_nvfp4.WEIGHTS_ID
    dense_sources = {}
    if convention is not None:
        # Anything else the export quantised -- attention, the dense feed-forward -- becomes an
        # NVFP4 linear; what it left alone keeps the base recipe.
        with ShardReader.for_directory(model) as probe:
            text_specs, dense_sources = routed_nvfp4.dense_plan(
                text_specs, {r.object_name: r for r in text_recipes}, probe, convention)
        text_recipes = tuple(r for r in text_recipes if not _dense_replaces(r.object_name, dense_sources))
    if text_format is None:
        text_format = "bf16" if convention is not None else "w8"
    text_specs = _text_storage(text_specs, text_format)
    vision_specs, vision_recipes = inventory.vision_recipes(g)
    specs = (*text_specs, *dflash_specs, *vision_specs)
    recipes = (*text_recipes, *dflash_recipes, *vision_recipes)
    if convention is not None:
        def owned(name):
            return routed_nvfp4.is_routed_object(name) or routed_nvfp4.owns_dense(name, dense_sources)
    else:
        def owned(name):
            return False
    validate_recipe_coverage(recipes, tuple(spec for spec in specs if not owned(spec.name)))
    resources = resources_for(model, g)
    plan = conversion.build_object_plan((*specs, *RESOURCE_SPECS), resources)
    recipe_map = {r.object_name: r for r in recipes}
    geometry = inventory.geometry_block(g, token_domain=tokenizer_domain(model))
    routed_summary = None
    from contextlib import ExitStack

    with ExitStack() as stack:
        reader = stack.enter_context(ShardReader.for_directory(model))
        dflash_reader = (stack.enter_context(ShardReader.for_directory(dflash_model))
                         if dflash_model is not None else None)
        preflight_source_reader(reader, tuple(r for r in recipes if not r.object_name.startswith("dflash/")))
        if dflash_reader is not None:
            preflight_source_reader(dflash_reader, dflash_recipes)
        routed = None
        if convention is not None:
            routed_nvfp4.preflight_source(reader, routed_geometry, convention)
            routed = routed_nvfp4.LayerCache(reader, routed_geometry, convention)
            print(f"routed experts: NVFP4 from the {convention.name} export", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(g.target, weights_id, architecture=g.target),
            plan.specs,
            geometry=geometry,
            vision_geometry=g.vision,
            layer_types=g.text.layer_types,
            dflash_geometry=(dflash_checkpoint.geometry_block(dflash_geometry)
                             if dflash_geometry is not None else None),
            dflash_target_layers=(dflash_geometry.target_feature_layers
                                  if dflash_geometry is not None else None),
        ) as writer:
            for spec in plan.specs:
                payload = resources.get(spec.name)
                if payload is None and convention is not None and routed_nvfp4.owns_dense(spec.name, dense_sources):
                    payload = routed_nvfp4.dense_payload(spec.name, dense_sources, reader, convention)
                elif payload is None and owned(spec.name):
                    payload = routed.payload_for(spec.name)
                elif payload is None:
                    source = dflash_reader if spec.name.startswith("dflash/") else reader
                    tensor = materialize_recipe(recipe_map[spec.name], source)
                    if spec.format == "FP32":
                        tensor = tensor.to(torch.float32)
                    payload = conversion.encode_tensor_payload(tensor, spec, pick_device(device))
                writer.write(spec.name, payload)
                print(spec.name, flush=True)
        if routed is not None:
            routed_summary = {"convention": convention.name, **routed.summary(),
                              "dense_nvfp4_objects": len(dense_sources)}
    report = {"architecture": g.target, "model": str(model), "vision": g.vision, "objects": len(plan.specs),
              "weights_id": weights_id, "text_format": text_format}
    if routed_summary is not None:
        report["routed_nvfp4"] = routed_summary
    if dflash_geometry is not None:
        report["dflash"] = {"model": str(dflash_model),
                            "geometry": dflash_checkpoint.geometry_block(dflash_geometry),
                            "target_layers": list(dflash_geometry.target_feature_layers)}
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-format", choices=TEXT_FORMATS, default=None,
                        help="storage of the non-quantised text matrices (attention, dense feed-forward); "
                             "default w8 for a BF16 checkpoint, bf16 (as stored) for an NVFP4 export")
    parser.add_argument("--dflash-model", type=Path, default=None,
                        help="a DFlash drafter checkpoint trained for this target, stored beside it")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, text_format=args.text_format,
            dflash_model=args.dflash_model)


if __name__ == "__main__":
    main()
