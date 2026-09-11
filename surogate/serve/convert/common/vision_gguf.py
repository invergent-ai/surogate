"""Shared projector pairing and artifact writing for multimodal GGUF checkpoints."""

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from surogate.serve.artifact.container import ArtifactWriter
from surogate.serve.convert.common import conversion
from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.gguf_repack import GgufRepackSource
from surogate.serve.convert.common.gguf_source import GgufRecipeReader, candidate_sources
from surogate.serve.convert.common.inventory import BF16, W8
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import expression_sources, materialize_recipe, validate_recipe_coverage
from surogate.serve.gguf.frontend import extract_generation_config, write_frontend
from surogate.serve.gguf.lean import LeanGguf


def find_projector(text_path, explicit, validate):
    text_path = Path(text_path)
    candidates = [Path(explicit).expanduser().resolve()] if explicit else sorted(text_path.parent.glob("*mmproj*.gguf"))
    compatible = []
    with LeanGguf(text_path) as text:
        for path in candidates:
            try:
                with LeanGguf(path) as vision:
                    validate(text, vision)
                compatible.append(path)
            except (ValueError, OSError, KeyError, TypeError) as error:
                if explicit:
                    raise ValueError(f"incompatible --mmproj {path}: {error}") from error
    if len(compatible) != 1:
        raise ValueError(
            "vision GGUF requires a matching projector; pass --mmproj PATH "
            f"({len(compatible)} compatible projectors found beside the text GGUF)"
        )
    return compatible[0].resolve()


def frontend_resources(text, specs):
    with tempfile.TemporaryDirectory(prefix="surogate-vl-frontend-") as temp:
        frontend = Path(temp)
        write_frontend(text, text.kv("general.architecture"), frontend)
        (frontend / "generation_config.json").write_text(json.dumps(extract_generation_config(text)))
        resources = {r.name: r.data for r in conversion.load_resources(frontend, specs)}
        return resources, tokenizer_domain(frontend)


def write_artifact(
    source,
    output,
    *,
    identity,
    geometry,
    vision_geometry,
    layer_types,
    specs,
    recipes,
    resources,
    device="cpu",
    native_text=True,
    tensor_transform=None,
):
    validate_recipe_coverage(recipes, specs)
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            actual = source.tensor(requirement.name)
            if actual.shape != requirement.shape:
                raise ValueError(f"{requirement.name}: shape {actual.shape} != {requirement.shape}")
    recipe_map = {r.object_name: r for r in recipes}
    repack = GgufRepackSource.from_sources(source.shards, candidate_sources(source))
    # Norms and learned position tables are not linears. Unquantized vision
    # matrices retain BF16; supported quantized matrices retain their source blocks.
    candidates = tuple(
        replace(s, format=W8)
        if s.name.startswith("vision/") and len(s.shape) == 2 and s.name != "vision/position_embedding"
        else s
        for s in specs
    )
    if not native_text:
        candidates = tuple(s for s in candidates if s.name.startswith("vision/"))
    native = repack.plan_native(recipe_map, candidates)
    native_specs = GgufRepackSource.native_specs(specs, native)
    runs = {s.name: repack.runs_for_native(s, recipe_map[s.name], None) for s in native_specs if s.name in native}
    # LFM's established text profile uses exact Q8 -> W8 repacks; wider or
    # incompatible source formats use its ordinary conversion arithmetic.
    repacked = repack.plan(recipe_map, specs) if not native_text else {}
    specs = GgufRepackSource.native_specs(specs, native, runs)
    from surogate.serve.convert.common.inventory import ResourceSpec

    plan = conversion.build_object_plan((*specs, *(ResourceSpec(name) for name in resources)), resources)
    reader = GgufRecipeReader(source)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(
        output,
        identity,
        plan.specs,
        external=tuple((str(p.resolve()), p.stat().st_size) for p in source.shards),
        geometry=geometry,
        layer_types=layer_types,
        vision_geometry=vision_geometry,
    ) as writer:
        for spec in plan.specs:
            if getattr(spec, "runs", ()):
                continue
            payload = resources.get(spec.name)
            if payload is None:
                if spec.name in repacked:
                    payload = repack.payload_for(spec, recipe_map[spec.name], None)
                else:
                    tensor = materialize_recipe(recipe_map[spec.name], reader)
                    if tensor_transform is not None:
                        tensor = tensor_transform(spec, tensor)
                    if spec.format == BF16:
                        tensor = tensor.bfloat16()
                    payload = conversion.encode_tensor_payload(tensor, spec, pick_device(device))
            writer.write(spec.name, payload)
    Path(str(output) + ".conversion.json").write_text(
        json.dumps(
            {
                "architecture": identity.architecture,
                "sources": [str(p) for p in source.shards],
                "native_objects": len(native),
                "objects": len(plan.specs),
            },
            indent=2,
        )
        + "\n"
    )
    return output
