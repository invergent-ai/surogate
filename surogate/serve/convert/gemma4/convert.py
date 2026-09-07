"""Build a dense Gemma 4 serving artifact from a HuggingFace checkpoint.

Safetensors in, artifact out -- or a GGUF read where it lies. `--gguf-repack` names the
map the bridge wrote beside the checkpoint it synthesised, and the objects that map covers
are served from the GGUF's own bytes instead of being dequantised and copied.

What is worth saying about the shape is said in `inventory.py`, which derives it from
the declaration rather than restating it. This module is the driver: read the config,
check the checkpoint has what the recipes name, then write each object in plan order.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from surogate.serve.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common import official_resources
from surogate.serve.convert.common.conversion import ResourcePayload
from surogate.serve.convert.common.gguf_repack import (
    GgufRepackSource,
    RepackError,
    half_names,
)
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.recipe import expression_sources, materialize_recipe

from . import inventory, recipe

RECIPE_ID = "gemma4-v1"

#: What the config must state for the artifact to be shaped at all. Everything the
#: declaration derives -- the layer schedule, the second attention geometry's
#: defaults -- is left to it.
_REQUIRED_CONFIG = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "head_dim",
)


def _plan_repack(repack, recipes_by_name, tensor_specs, native) -> tuple[str, ...]:
    """Objects a bit-exact plane repack covers, and a check that the map is not over-broad.

    Every recipe left on the materialize path must find its sources in the bridged checkpoint,
    so a mapped source an un-planned recipe still consumes is a hard error rather than a tensor
    read from two places.
    """
    if repack is None:
        return ()
    planned = repack.plan(recipes_by_name, tensor_specs)
    covered = set(planned) | set(native or ())
    stray = {
        source.name
        for name, tensor_recipe in recipes_by_name.items()
        if name not in covered
        for source in expression_sources(tensor_recipe.expression)
        if source.name in repack.sources
    }
    if stray:
        raise RepackError(
            "repack map names sources still needed by materialized recipes: "
            + ", ".join(sorted(stray))
        )
    return planned


def validate_config(config: Mapping[str, object]) -> inventory.Geometry:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    absent = [name for name in _REQUIRED_CONFIG if text.get(name) is None]
    if absent:
        raise ValueError(
            "this checkpoint's config.json does not state "
            + ", ".join(absent)
            + "; a Gemma 4 artifact cannot be shaped without them"
        )
    geometry = inventory.geometry_from_config(config)
    # The two head geometries are what this target exists to carry. A checkpoint that
    # states one and means both would otherwise convert into an artifact whose global
    # layers are sized like its windowed ones, which loads and is wrong.
    if geometry.global_head_dim <= 0 or geometry.global_kv_heads <= 0:
        raise ValueError(
            "this checkpoint states no global attention geometry "
            f"(global_head_dim={geometry.global_head_dim}, "
            f"num_global_key_value_heads={geometry.global_kv_heads})"
        )
    return geometry


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
    repack = GgufRepackSource(gguf_repack) if gguf_repack else None

    config = family_conversion.load_json(model / "config.json")
    geometry = validate_config(config)
    family_conversion.honour_declared_scope(
        config, geometry, model, what=family_conversion.checkpoint_label(model)
    )

    # What the artifact writes, which on a tied checkpoint is one object fewer than the
    # model has: the head is the embedding table and the binder fills both roles from it.
    tied = recipe.tied_output_head(config)
    objects = inventory.stored_objects(config, tied_output_head=tied)
    tensor_specs = inventory.tensor_specs(objects)
    recipes = {r.object_name: r for r in recipe.build_recipes(config)}

    # What the GGUF can serve as it stores it. `native` is a whole object read verbatim,
    # `halves` a fused parent whose two halves carry different K-quant types, `planned` the
    # objects a bit-exact plane repack covers. Everything else is materialised from the
    # bridged checkpoint, exactly as a safetensors conversion materialises everything.
    native = repack.plan_native(recipes, tensor_specs) if repack is not None else {}
    halves = repack.plan_native_halves(recipes, tensor_specs) if repack is not None else {}
    planned = _plan_repack(repack, recipes, tensor_specs, set(native) | set(halves))
    repacked_names = frozenset(planned)
    if halves:
        tensor_specs = GgufRepackSource.native_half_specs(tensor_specs, halves)
        print(f"native K-quant halves: {len(halves)} fused parents stored as typed pairs",
              flush=True)
    external: tuple = ()
    native_runs: dict = {}
    if native and repack is not None and os.environ.get("SUROGATE_GGUF_COPY", "0") == "0":
        native_runs = {
            spec.name: repack.runs_for_native(spec, recipes[spec.name], None)
            for spec in GgufRepackSource.native_specs(tensor_specs, native)
            if spec.name in native
        }
        external = ((str(Path(repack.gguf_path).resolve()),
                     Path(repack.gguf_path).stat().st_size),)
        not_copied = sum(sum(r[2] for r in runs) for runs in native_runs.values())
        print(f"native K-quants: {len(native_runs)} objects read from the GGUF in place "
              f"({not_copied / 1e9:.2f} GB not copied)", flush=True)
    if native:
        tensor_specs = GgufRepackSource.native_specs(tensor_specs, native, native_runs)

    resources = load_resources(model)
    resource_payloads = {item.name: item.data for item in resources}
    carried = tuple(spec for spec in inventory.RESOURCE_SPECS if spec.name in resource_payloads)
    plan = family_conversion.build_object_plan(tuple(tensor_specs) + carried, resource_payloads)

    # Repacked objects read the GGUF directly; only what remains needs a bridged source.
    remaining = tuple(r for name, r in recipes.items()
                      if name not in repacked_names and name not in native and name not in halves)
    # object name -> (fused parent recipe, the rows of that parent it holds)
    half_lookup: dict[str, tuple[str, slice]] = {}
    for parent, runs in halves.items():
        first = 0
        for name, (_, rows) in zip(half_names(parent), runs):
            half_lookup[name] = (parent, slice(first, first + rows))
            first += rows

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
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            plan.specs,
            geometry=_geometry_block(geometry),
            external=external,
        ) as writer:
            total = len(plan.specs)
            for index, spec in enumerate(plan.specs, start=1):
                note = ""
                if spec.name in resource_payloads:
                    payload = resource_payloads[spec.name]
                elif repack is not None and spec.name in half_lookup:
                    parent, row_slice = half_lookup[spec.name]
                    payload = repack.payload_for_native(spec, recipes[parent], None,
                                                        row_slice=row_slice)
                    note = " (repacked)"
                elif repack is not None and spec.name in native_runs:
                    # Read from the GGUF where it lies: the index names the runs and the
                    # artifact carries no bytes for it, so there is nothing to write.
                    print(f"[{index}/{total}] {spec.name} (in place)", flush=True)
                    continue
                elif repack is not None and spec.name in native:
                    payload = repack.payload_for_native(spec, recipes[spec.name], None)
                    note = " (repacked)"
                elif repack is not None and spec.name in repacked_names:
                    payload = repack.payload_for(spec, recipes[spec.name], None)
                    note = " (repacked)"
                else:
                    tensor = materialize_recipe(recipes[spec.name], reader)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}{note}", flush=True)

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
            "architecture": inventory.architecture_of(config),
            "hidden": geometry.hidden,
            "layers": geometry.layers,
            "intermediate": geometry.intermediate,
            "head_dim": geometry.head_dim,
            "global_head_dim": geometry.global_head_dim,
            "global_kv_heads": geometry.global_kv_heads,
            "global_layers": list(geometry.global_layers),
            "k_eq_v": geometry.k_eq_v,
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


def load_resources(model_dir: Path) -> tuple[ResourcePayload, ...]:
    """The four frontend files, from wherever this release states them.

    Gemma 4 states its chat template in a different place at each size, and one size does
    not state one at all:

    * `google/gemma-4-31B-it` ships `chat_template.jinja` as a file, and the engine also
      requires `tokenizer_config.json` to carry the same bytes -- it compares the two -- so
      the template is inserted into the config where the release left it out.
    * `google/gemma-4-12B` is a **base** model. It publishes no template in either place, so
      the artifact carries no `chat_template.jinja` at all, and the engine reads that absence
      as "this is a base model": the chat endpoints refuse it by name and `/v1/completions`
      serves it. That is a property of this checkpoint, not a gap in the conversion.
    """
    root = Path(model_dir)
    template = official_resources.chat_template_bytes(root)
    payloads: list[ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if filename == "chat_template.jinja":
            if template is None:
                continue
            data = template
        elif filename == "tokenizer_config.json":
            data = (path.read_bytes() if template is None
                    else official_resources.tokenizer_config_with_template(
                        path.read_bytes(), template))
        elif path.exists():
            data = path.read_bytes()
        elif filename == "generation_config.json":
            data = family_conversion._synthesize_generation_config(root)  # noqa: SLF001
        else:
            raise FileNotFoundError(f"checkpoint is missing {filename}")
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        payloads.append(ResourcePayload(spec.name, data))
    return tuple(payloads)


def _geometry_block(geometry: inventory.Geometry) -> dict[str, float]:
    """The dimensions the artifact states about itself, which the engine's binder
    validates its compiled constants against.

    The global members are why this target's artifact says more than the other dense
    ones: `head_dim` and `kv_heads` describe the windowed layers only, and a binder
    that read them for every layer would size half the cache wrong.
    """
    return {
        "hidden": float(geometry.hidden),
        "layers": float(geometry.layers),
        "intermediate": float(geometry.intermediate),
        # `output_rows` is what the head is padded to and `token_domain` what the tokenizer
        # can address. Gemma 4 pads neither, so both are the vocabulary.
        "output_rows": float(geometry.vocab),
        "token_domain": float(geometry.vocab),
        "query_heads": float(geometry.query_heads),
        "kv_heads": float(geometry.kv_heads),
        "head_dim": float(geometry.head_dim),
        "global_kv_heads": float(geometry.global_kv_heads),
        "global_head_dim": float(geometry.global_head_dim),
        "global_rotary_angles": float(geometry.partial_rotary_angles),
        "sliding_window": float(geometry.sliding_window),
        "rope_theta": float(geometry.rope_theta),
        "sliding_rope_theta": float(geometry.sliding_rope_theta),
        # Named as `family::TextGeometry` names them: the binder lays this object over the
        # target's compiled constants by member name, so a key that does not match is a
        # dimension the engine silently keeps its own value for.
        "logit_softcap": float(geometry.final_logit_softcapping),
        "embedding_scale": float(geometry.embedding_scale),
        "rms_epsilon": float(geometry.rms_epsilon),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path, default=None,
                        help="serve this GGUF's K-quants in place instead of copying them")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack)


if __name__ == "__main__":
    main()
