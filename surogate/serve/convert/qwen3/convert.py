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
from surogate.serve.convert.common.safetensors import ShardReader
from surogate.serve.convert.common import conversion as family_conversion
from surogate.serve.convert.common.recipe import (
    Concat,
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    materialize_recipe,
    preflight_source_reader,
    source,
)
from surogate.serve.convert.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory

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


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`."""

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


def validate_config(config: Mapping[str, object]) -> tuple[inventory.Geometry, dict]:
    """Validate the checkpoint and summarize it for the conversion report."""

    family_conversion.check_members("config", config, _REQUIRED_CONFIG)
    geometry = geometry_from_config(config)
    if geometry != inventory.GEOMETRY:
        raise ValueError(
            "checkpoint geometry is not the registered qwen3 target:\n"
            f"  checkpoint {geometry}\n"
            f"  target     {inventory.GEOMETRY}\n"
            "csrc/src/serve/targets/qwen3/impl/config.h describes one size; a "
            "differently sized Qwen3 needs its own target header before its "
            "artifact can be bound."
        )
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
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry = inventory.GEOMETRY,
    *,
    tied_output_head: bool = True,
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
                TensorRecipe(
                    obj + "attention/query_norm",
                    source(src + "self_attn.q_norm.weight", (geometry.head_dim,)),
                ),
                TensorRecipe(
                    obj + "attention/key_norm",
                    source(src + "self_attn.k_norm.weight", (geometry.head_dim,)),
                ),
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
                # tie_word_embeddings=True: Qwen3-0.6B still ships an
                # `lm_head.weight`, bit-identical to the embedding, but not every
                # tied export does, so the tied path reads the embedding.
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

    The hybrid targets in this package are all multi-shard, so their converters
    open by index. Qwen3-0.6B is one 1.4 GB `model.safetensors` with no index at
    all, and a converter that only knew how to follow an index could not read it.
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
        return {recipe.object_name: recipe for recipe in self.recipes}


def preflight_inventory() -> None:
    """The inventory the recipe and the writer agree to produce."""

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
            geometry=geometry_block(preflight, model),
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
