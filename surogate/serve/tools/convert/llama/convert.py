"""Convert a Hugging Face `LlamaForCausalLM` checkpoint into one `.sinfer` artifact.

Canonical invocation::

    python -m surogate.serve.tools.convert.llama.convert \
      --model ~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/<rev> \
      --out /tmp/tinyllama-1.1b.sinfer

This is the Qwen3 recipe with the two differences the architecture actually
has, and nothing else:

* **Llama has no query or key norm.**  Qwen3 stores a `head_dim`-wide RMS scale
  for each; there is no `self_attn.q_norm.weight` or `k_norm.weight` in a Llama
  checkpoint to read, and no object here to fill.
* **The output head is untied.**  `TinyLlama/TinyLlama-1.1B-Chat-v1.0` declares
  `tie_word_embeddings: false` and ships `lm_head.weight` as its own tensor, so
  `text/output_head` comes from that tensor.  The tied path is still built,
  because `tie_word_embeddings` is a property of the checkpoint in hand — a tied
  Llama export has no `lm_head.weight` at all and must read the embedding.

Two further facts about the release shape the code below and are handled the
same way Qwen3's converter handles them: it is a single unsharded
`model.safetensors` with no index, and it publishes no `chat_template.jinja`,
only the template embedded in `tokenizer_config.json`.

`config.json` here is a `transformers` 4.35 export, which predates both the
`head_dim` and the `mlp_bias` keys.  Neither absence is a gap: `head_dim` is
`hidden_size // num_attention_heads` by construction, and an export from before
`mlp_bias` existed is an export with no MLP bias.  Both are read that way rather
than the checkpoint being refused for keys its exporter could not have written.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from surogate.serve.tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from surogate.serve.tools.convert.common.quantize import pick_device
from surogate.serve.tools.convert.common.safetensors import ShardReader
from surogate.serve.tools.convert.common import conversion as family_conversion
from surogate.serve.tools.convert.common.recipe import (
    Concat,
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    materialize_recipe,
    preflight_source_reader,
    source,
)
from surogate.serve.tools.convert.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory

RECIPE_ID = "llama-v1"

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan

#: Members of `config.json` that must hold for the registered target. Geometry is
#: read out of the file rather than asserted against a second copy of itself;
#: these are the members that are not geometry, plus the architecture identity.
_REQUIRED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_act": "silu",
    "attention_bias": False,
    "rms_norm_eps": 1e-5,
    "rope_scaling": None,
}

#: Members that only some exporter versions write, with the value an absent key
#: asserts. `mlp_bias` was added to the Llama config after this checkpoint was
#: exported and defaults to false; `pretraining_tp` above 1 makes Hugging Face
#: split the projections into shards that this fused recipe does not reproduce,
#: so a checkpoint that declares it is refused rather than silently mis-fused.
_OPTIONAL_CONFIG = {
    "mlp_bias": False,
    "pretraining_tp": 1,
}


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`.

    `head_dim` is derived: a Llama config in this dialect has no such key, and
    the architecture fixes the width at `hidden_size // num_attention_heads`.
    """

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    return inventory.Geometry(
        layers=int(config["num_hidden_layers"]),
        hidden=hidden,
        intermediate=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        query_heads=heads,
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
    )


def check_optional_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require the named members only where the exporter wrote them."""

    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual[name]!r}"
        for name, value in expected.items()
        if name in actual and actual[name] != value
    ]
    if mismatches:
        raise ValueError("checkpoint config mismatch:\n  " + "\n  ".join(mismatches))


def validate_config(config: Mapping[str, object]) -> tuple[inventory.Geometry, dict]:
    """Validate the checkpoint and summarize it for the conversion report."""

    family_conversion.check_members("config", config, _REQUIRED_CONFIG)
    check_optional_members("config", config, _OPTIONAL_CONFIG)
    geometry = geometry_from_config(config)
    if geometry != inventory.GEOMETRY:
        raise ValueError(
            "checkpoint geometry is not the registered llama target:\n"
            f"  checkpoint {geometry}\n"
            f"  target     {inventory.GEOMETRY}\n"
            "The registered target describes one size; a differently sized "
            "LlamaForCausalLM needs its own target header before its artifact "
            "can be bound."
        )
    text = {
        name: config[name]
        for name in (
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
            "attention_bias",
        )
        if name in config
    }
    # Recorded as derived, because the file does not carry it.
    text["head_dim"] = geometry.head_dim
    text["mlp_bias"] = bool(config.get("mlp_bias", False))
    summary = {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": text,
        "attention": {
            "query_size": geometry.query_size,
            "kv_size": geometry.kv_size,
            "fused_rows": geometry.attention_fused_rows,
            "output_gate": False,
            "qk_norm": False,
        },
        "vision": None,  # LlamaForCausalLM is text-only
        "mtp_num_hidden_layers": 0,
    }
    return geometry, summary


# ---------------------------------------------------------------------------
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry = inventory.GEOMETRY,
    *,
    tied_output_head: bool = False,
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from in the checkpoint, in object order."""

    hidden = geometry.hidden
    query, kv = geometry.query_size, geometry.kv_size
    embedding = source("model.embed_tokens.weight", (geometry.vocab, hidden))

    recipes: list[TensorRecipe] = [TensorRecipe("text/token_embedding", embedding)]

    for layer in range(geometry.layers):
        src = f"model.layers.{layer}."
        obj = f"text/layers/{layer}/"
        recipes.extend(
            (
                TensorRecipe(
                    obj + "input_norm",
                    source(src + "input_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "attention/query_key_value",
                    # Ungated attention: q | k | v, with no gate block between
                    # k and v. The row order is the one the fused decode kernel
                    # reads and the one `LOGICAL_ROW_VIEW_SPECS` publishes.
                    Concat(
                        (
                            source(src + "self_attn.q_proj.weight", (query, hidden)),
                            source(src + "self_attn.k_proj.weight", (kv, hidden)),
                            source(src + "self_attn.v_proj.weight", (kv, hidden)),
                        ),
                        0,
                    ),
                ),
                # No query_norm / key_norm recipes: Llama has no such tensors.
                TensorRecipe(
                    obj + "attention/output",
                    source(src + "self_attn.o_proj.weight", (hidden, query)),
                ),
                TensorRecipe(
                    obj + "post_attention_norm",
                    source(src + "post_attention_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "mlp/gate_up",
                    Concat(
                        (
                            source(src + "mlp.gate_proj.weight",
                                   (geometry.intermediate, hidden)),
                            source(src + "mlp.up_proj.weight",
                                   (geometry.intermediate, hidden)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    obj + "mlp/down",
                    source(src + "mlp.down_proj.weight", (hidden, geometry.intermediate)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe("text/final_norm", source("model.norm.weight", (hidden,))),
            TensorRecipe(
                "text/output_head",
                # tie_word_embeddings=False for TinyLlama: the head is its own
                # tensor and is nothing like the embedding, so reading the
                # embedding here would produce an artifact that loads and is
                # quietly wrong. A tied Llama export ships no `lm_head.weight`
                # at all, hence the other branch.
                embedding
                if tied_output_head
                else source("lm_head.weight", (geometry.vocab, hidden)),
            ),
        )
    )
    return tuple(recipes)


RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements(recipes: Sequence[TensorRecipe] = RECIPE_SPECS) -> dict:
    requirements: dict = {}
    for recipe in recipes:
        for requirement in expression_sources(recipe.expression):
            requirements.setdefault(requirement.name, requirement)
    return requirements


validate_recipe_coverage()


# ---------------------------------------------------------------------------
# checkpoint access
# ---------------------------------------------------------------------------


def open_reader(model_dir: str | Path) -> ShardReader:
    """Open a sharded or single-file safetensors checkpoint.

    TinyLlama is one 2.1 GB `model.safetensors` with no index at all, and a
    reader that only knew how to follow an index could not open it; larger
    Llama releases are sharded, so both doors stay open.
    """

    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.exists():
        return ShardReader(root)
    single = root / "model.safetensors"
    if single.exists():
        return ShardReader.from_file(single)
    raise FileNotFoundError(
        f"{root} holds neither model.safetensors.index.json nor model.safetensors"
    )


def preflight_sources(
    model_dir: str | Path,
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
) -> SourcePreflight:
    with open_reader(model_dir) as reader:
        return preflight_source_reader(reader, recipes)


# ---------------------------------------------------------------------------
# frontend resources
# ---------------------------------------------------------------------------


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    """The four text frontend files, in inventory order.

    TinyLlama ships a `tokenizer.json` — the fast-tokenizer serialization of its
    SentencePiece vocabulary — beside the `tokenizer.model` the Llama tokenizer
    was originally distributed as. The engine binds the former; the latter holds
    the same vocabulary in a format nothing here reads, so it is not carried.

    `chat_template.jinja` is not a file in the release — the template lives only
    in `tokenizer_config.json`. The engine both binds the resource and checks it
    byte-for-byte against `tokenizer_config.json.chat_template`, so it is written
    out verbatim, with no added trailing newline.
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
        elif filename == "tokenizer.json":
            raise FileNotFoundError(
                "checkpoint is missing tokenizer.json; a Llama release that ships "
                "only the SentencePiece tokenizer.model must be converted to the "
                "fast-tokenizer serialization before it can be bound"
            )
        else:
            raise FileNotFoundError(f"checkpoint is missing {filename}")
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        payloads.append(ResourcePayload(spec.name, data))
    return tuple(payloads)


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
        return {recipe.object_name: recipe for recipe in self.recipes}


def preflight_inventory() -> None:
    """The inventory the recipe and the writer agree to produce."""

    expected_tensors = 1 + inventory.LAYERS * inventory.LAYER_OBJECT_COUNT + 2
    if len(inventory.TENSOR_SPECS) != expected_tensors:
        raise ValueError(
            f"registered inventory holds {len(inventory.TENSOR_SPECS)} tensors, "
            f"expected {expected_tensors}"
        )
    if len(inventory.RESOURCE_SPECS) != 4:
        raise ValueError("registered inventory does not hold the four text resources")
    if len(inventory.OBJECT_SPECS) != expected_tensors + 4:
        raise ValueError("registered object inventory is incomplete")
    validate_recipe_coverage()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    geometry, summary = validate_config(config)
    preflight_inventory()
    tied = bool(config.get("tie_word_embeddings", False))
    # Which tensor the output head reads is a property of the checkpoint, not of
    # the target, so the recipe is rebuilt for the checkpoint in hand rather than
    # the module-level one being used blind.
    recipes = build_recipes(geometry, tied_output_head=tied)
    _validate_recipe_coverage(recipes, inventory.TENSOR_SPECS)
    source_preflight = preflight_sources(model, recipes)
    resources = load_resources(model)
    plan = build_object_plan({item.name: item.data for item in resources})
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
    repo_root = Path(__file__).resolve().parents[5]
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
) -> Path:
    """Run the complete conversion and return the conversion-report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)

    preflight = preflight_conversion(model)
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {item.name: item.data for item in preflight.resources}
    recipes = preflight.recipes_by_name
    with open_reader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            total = len(inventory.OBJECT_SPECS)
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, recipes)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

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
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
