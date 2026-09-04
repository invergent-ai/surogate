"""Convert a Hugging Face `Qwen3ForCausalLM` checkpoint into one `.sinfer` artifact.

Canonical invocation::

    python -m surogate.serve.convert.qwen3.convert \
      --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev> \
      --out /tmp/qwen3-0.6b.sinfer

Two things differ from the hybrid converters in this package and are the reason
this target has a recipe of its own rather than a parameter on theirs:

* Qwen3's attention has no output gate, so the fused projection is
  ``q_proj | k_proj | v_proj`` concatenated on the row axis.  The hybrid
  recipes split an interleaved ``q_proj`` into a query half and a gate half and
  fuse four blocks; there is nothing to split here.
* The Hugging Face release is a single unsharded `model.safetensors` with no
  index, and it publishes no `chat_template.jinja`.  Both are handled below —
  the reader opens the file directly, and the template is lifted out of
  `tokenizer_config.json`, which is the same string the engine cross-checks the
  resource against.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from surogate.serve.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from surogate.serve.convert.common.quantize import pick_device
from surogate.serve.convert.common.gguf_repack import (
    GgufRepackSource,
    RepackError,
    half_names,
)
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common.recipe import (
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    materialize_recipe,
)
from surogate.serve.convert.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory, recipe

RECIPE_ID = "qwen3-v1"

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan

#: Members of `config.json` that must hold for the registered target. Geometry is
#: read out of the file rather than asserted against a second copy of itself;
#: these are the members that are not geometry, plus the architecture identity.
_REQUIRED_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_act": "silu",
    "attention_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "sliding_window": None,
    "use_sliding_window": False,
}


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


def validate_config(config: Mapping[str, object]) -> tuple[inventory.Geometry, dict]:
    """Validate the checkpoint and summarize it for the conversion report."""

    family_conversion.check_members("config", config, _REQUIRED_CONFIG)
    geometry = recipe.geometry_from_config(config)
    # Any size of the family converts. The artifact states its own dimensions in the
    # `geometry` member and the engine binds against those, so what has to hold is that
    # the checkpoint is self-consistent -- not that it is one particular checkpoint.
    if geometry.query_heads % geometry.kv_heads != 0:
        raise ValueError(
            f"query heads ({geometry.query_heads}) must be a multiple of key/value heads "
            f"({geometry.kv_heads})"
        )
    if geometry.hidden <= 0 or geometry.layers <= 0 or geometry.intermediate <= 0:
        raise ValueError(f"checkpoint geometry has a non-positive dimension: {geometry}")
    summary = {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {
            name: config[name]
            for name in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "vocab_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "rms_norm_eps",
                "rope_theta",
                "max_position_embeddings",
                "tie_word_embeddings",
                "attention_bias",
            )
        },
        "attention": {
            "query_size": geometry.query_size,
            "kv_size": geometry.kv_size,
            "fused_rows": geometry.attention_fused_rows,
            "output_gate": False,
            "qk_norm": True,
        },
        "vision": None,  # Qwen3ForCausalLM is text-only
        "mtp_num_hidden_layers": 0,
    }
    return geometry, summary


# ---------------------------------------------------------------------------
# frontend resources
# ---------------------------------------------------------------------------


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    """The four text frontend files, in inventory order.

    `chat_template.jinja` is not a file in the Qwen3 release — the template lives
    only in `tokenizer_config.json`. The engine both binds the resource and
    checks it byte-for-byte against `tokenizer_config.json.chat_template`, so it
    is written out verbatim, with no added trailing newline.
    """

    root = Path(model_dir)
    payloads: list[ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if path.exists():
            data = path.read_bytes()
        elif filename == "chat_template.jinja":
            data = _chat_template_from_tokenizer_config(root)
        elif filename == "generation_config.json":
            data = family_conversion._synthesize_generation_config(root)  # noqa: SLF001
        else:
            raise FileNotFoundError(f"checkpoint is missing {filename}")
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        payloads.append(ResourcePayload(spec.name, data))
    return tuple(payloads)


def token_domain(root: Path) -> int:
    """How many token ids the tokenizer defines: the rows of the head that are real tokens.

    `config.json.vocab_size` is the padded row count of the embedding and the head; the ids a
    prompt can contain, and the ids sampling may return, stop earlier. The tokenizer is the
    authority on where -- its vocabulary plus the added tokens -- and the engine restricts
    sampling to exactly this many rows.
    """
    tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    ids = list(tokenizer["model"]["vocab"].values())
    ids.extend(int(token["id"]) for token in tokenizer.get("added_tokens", ()))
    return max(ids) + 1


def geometry_block(preflight: "ConversionPreflight", root: Path) -> dict[str, float]:
    """The artifact's `geometry` member: the numbers the engine's one `qwen3` target reads at
    load instead of compiling, keyed as the family's `TextGeometry` names them."""
    geometry = preflight.geometry
    text = preflight.config_summary["text"]
    return {
        "hidden": geometry.hidden,
        "layers": geometry.layers,
        "intermediate": geometry.intermediate,
        "output_rows": geometry.vocab,
        "token_domain": token_domain(root),
        "query_heads": geometry.query_heads,
        "kv_heads": geometry.kv_heads,
        "head_dim": geometry.head_dim,
        "rotary_dim": geometry.head_dim,
        "rms_epsilon": float(text["rms_norm_eps"]),
        "rope_theta": float(text["rope_theta"]),
    }


def _chat_template_from_tokenizer_config(root: Path) -> bytes:
    config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError(
            "checkpoint publishes neither chat_template.jinja nor "
            "tokenizer_config.json.chat_template; the engine needs one"
        )
    return template.encode("utf-8")


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    geometry: inventory.Geometry
    config_summary: dict
    recipes: tuple[TensorRecipe, ...]
    source: SourcePreflight
    resources: tuple[ResourcePayload, ...]
    object_plan: ObjectPlan

    @property
    def recipes_by_name(self) -> dict[str, TensorRecipe]:
        return {tensor_recipe.object_name: tensor_recipe for tensor_recipe in self.recipes}


def preflight_inventory() -> None:
    """The registered inventory, which describes the size this target compiles.

    A conversion of a differently sized Qwen3 does not go through here: it builds the
    checkpoint's own specs and recipes and checks those against each other. This stays
    as the check that the module's own constants have not drifted apart.
    """

    expected_tensors = 2 + inventory.LAYERS * 8 + 1
    if len(inventory.TENSOR_SPECS) != expected_tensors:
        raise ValueError(
            f"registered inventory holds {len(inventory.TENSOR_SPECS)} tensors, "
            f"expected {expected_tensors}"
        )
    if len(inventory.RESOURCE_SPECS) != 4:
        raise ValueError("registered inventory does not hold the four text resources")
    if len(inventory.OBJECT_SPECS) != expected_tensors + 4:
        raise ValueError("registered object inventory is incomplete")
    recipe.validate_recipe_coverage()


def build_object_plan(
    resources: Mapping[str, bytes], geometry: inventory.Geometry = inventory.GEOMETRY,
    *, native=None, object_specs=None
) -> ObjectPlan:
    """`native` names the objects a GGUF serves as it stores them, with their format
    rewritten to the stored one."""
    if object_specs is None:
        object_specs = inventory.build_object_specs(geometry)
    if native:
        object_specs = GgufRepackSource.native_specs(object_specs, native)
    return family_conversion.build_object_plan(object_specs, resources)



def plan_repack(repack, recipes_by_name, tensor_specs, native=None) -> tuple[str, ...]:
    """Objects the repack source covers; verifies the map is not over-broad.

    Every recipe left on the materialize path must find its sources in the bridged
    checkpoint, so a mapped source consumed by an un-planned recipe is a hard error.
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


def preflight_conversion(
    model_dir: str | Path,
    repack=None,
    planned: tuple[str, ...] = (),
    *,
    native=None,
    object_specs=None,
) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    geometry, summary = validate_config(config)
    preflight_inventory()
    tied = bool(config.get("tie_word_embeddings", False))
    # Which tensor the output head reads is a property of the checkpoint, not of
    # the target, so the recipe is rebuilt for the checkpoint in hand rather than
    # the module-level one being used blind.
    recipes = recipe.build_recipes(geometry, tied_output_head=tied)
    _validate_recipe_coverage(recipes, inventory.build_tensor_specs(geometry))
    # Repacked objects read the GGUF directly; only what remains needs a bridged source.
    remaining = tuple(r for r in recipes if r.object_name not in planned) if planned else recipes
    source_preflight = recipe.preflight_sources(model, remaining)
    resources = load_resources(model)
    plan = build_object_plan({item.name: item.data for item in resources}, geometry,
                             native=native, object_specs=object_specs)
    return ConversionPreflight(
        model_dir=model,
        geometry=geometry,
        config_summary=summary,
        recipes=recipes,
        source=source_preflight,
        resources=resources,
        object_plan=plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    recipes: Mapping[str, TensorRecipe],
) -> torch.Tensor:
    tensor = materialize_recipe(recipes[spec.name], reader)
    if spec.format == inventory.BF16 and tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {"model_path": str(Path(model_dir).resolve())},
        "arguments": dict(arguments),
        "config_summary": dict(config_summary),
        "source_preflight": {
            "recipes": source_preflight.recipe_count,
            "tensors": source_preflight.source_tensor_count,
            "shards": source_preflight.source_shard_count,
            "dtypes": dict(source_preflight.source_dtype_counts),
        },
        "converter": {
            "revision": family_conversion.converter_revision(repo_root),
            "environment": family_conversion.environment(device),
        },
        "objects": family_conversion.object_statistics(objects),
        "elapsed_seconds": elapsed_seconds,
        "artifact": {"path": str(Path(out_path)), "bytes": final_bytes},
    }


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    gguf_repack: str | Path | None = None,
) -> Path:
    """Run the complete conversion and return the conversion-report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    repack = GgufRepackSource(gguf_repack) if gguf_repack else None

    # What the GGUF can serve as it stores it. `native` is a whole object read verbatim,
    # `halves` a fused parent whose two halves carry different K-quant types, `planned` the
    # objects a bit-exact plane repack covers. Everything else is materialised.
    config = family_conversion.load_json(model / "config.json")
    geometry, _ = validate_config(config)
    recipes = {
        r.object_name: r
        for r in recipe.build_recipes(
            geometry, tied_output_head=bool(config.get("tie_word_embeddings", False)))
    }
    tensor_specs = inventory.build_tensor_specs(geometry)
    object_specs = inventory.build_object_specs(geometry)
    native = repack.plan_native(recipes, tensor_specs) if repack is not None else {}
    halves = repack.plan_native_halves(recipes, tensor_specs) if repack is not None else {}
    planned = plan_repack(repack, recipes, tensor_specs, set(native) | set(halves))
    repacked_names = frozenset(planned)
    if halves:
        tensor_specs = GgufRepackSource.native_half_specs(tensor_specs, halves)
        object_specs = GgufRepackSource.native_half_specs(object_specs, halves)
        print(f"native K-quant halves: {len(halves)} fused parents stored as typed pairs",
              flush=True)
    external = ()
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
        object_specs = GgufRepackSource.native_specs(object_specs, native, native_runs)
        if not native_runs:
            print(f"native K-quants: {len(native)} objects served as the GGUF stores them",
                  flush=True)

    preflight = preflight_conversion(
        model, repack, planned + tuple(native) + tuple(halves),
        native=native, object_specs=object_specs,
    )
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    # object name -> (fused parent recipe, the rows of that parent it holds)
    half_lookup: dict[str, tuple[str, slice]] = {}
    for parent, runs in halves.items():
        first = 0
        for name, (_, rows) in zip(half_names(parent), runs):
            half_lookup[name] = (parent, slice(first, first + rows))
            first += rows
    resources = {item.name: item.data for item in preflight.resources}
    recipes = preflight.recipes_by_name
    with recipe.open_reader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
            geometry=geometry_block(preflight, model),
            external=external,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            checkpoint_specs = object_specs
            total = len(checkpoint_specs)
            for index, spec in enumerate(checkpoint_specs, start=1):
                repacked = False
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif repack is not None and spec.name in half_lookup:
                    parent, row_slice = half_lookup[spec.name]
                    payload = repack.payload_for_native(spec, recipes[parent], None,
                                                        row_slice=row_slice)
                    repacked = True
                elif repack is not None and spec.name in native_runs:
                    # Read from the GGUF where it lies: the index names the runs and the
                    # artifact carries no bytes for it, so there is nothing to write.
                    print(f"[{index}/{total}] {spec.name} (in place)", flush=True)
                    continue
                elif repack is not None and spec.name in native:
                    # A K-quant served as the GGUF stores it: rows gathered verbatim.
                    payload = repack.payload_for_native(spec, recipes[spec.name], None)
                    repacked = True
                elif repack is not None and spec.name in repacked_names:
                    # Bit-exact plane repack; no dequantization or requantization.
                    payload = repack.payload_for(spec, recipes[spec.name], None)
                    repacked = True
                else:
                    tensor = materialize_tensor(spec, reader, recipes)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}" + (" (repacked)" if repacked else ""),
                      flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
        },
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path, default=None,
                        help="repack map: objects this GGUF serves as it stores them")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack)


if __name__ == "__main__":
    main()
