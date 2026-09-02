"""Convert a Hugging Face `Gemma3ForCausalLM` checkpoint into one `.sinfer` artifact.

Canonical invocation::

    python -m surogate.serve.tools.convert.gemma3.convert \
      --model ~/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots/<rev> \
      --out /tmp/gemma3-270m.sinfer

This is the Llama recipe with the differences Gemma 3 actually has.  Four are
structural and live in `inventory.py`: four sandwich norms per layer, Q/K/V and
gate/up kept separate rather than fused, per-head query and key norms, and a
head that is tied to the embedding and therefore *not stored* — `text/output_head`
is a role `inventory.ALIAS_SPECS` puts on `text/token_embedding`, and
`targets/gemma3/impl/load/bindings.cpp` fills both from the one table.  Three
more are properties of the numbers rather than of the object list, and each is
silent when wrong:

* **The norms are zero-centred and pass through untouched.**  `Gemma3RMSNorm`
  is `x_normed * (1 + w)`, so a Gemma checkpoint stores `w`, not the scale.  The
  DSL declares every norm object with `transform="unfold_unit_offset"` and states
  the invariant the transform names: the artifact holds the *unfolded* `w`
  because the runtime re-applies the one itself.  A safetensors checkpoint
  already holds `w`, so this converter copies.  The subtraction that name
  suggests belongs to the GGUF path — `convert/gemma_embedding/` reads the folded
  `1 + w` and takes a one off every norm on the way in.  Getting this backwards
  in either direction is quiet: the artifact would hold `1 + w`, the runtime
  would make it `2 + w`, and nothing would raise.

* **The embedding is stored unscaled.**  Gemma multiplies the looked-up row by
  `sqrt(hidden)` before the first block; the engine carries that factor as
  `config.h::embedding_scale` (25.29822 for hidden 640), so folding it in here
  would apply it twice.

* **The local/global schedule and the attention scale are compile-time constants
  in the engine.**  `config.h` bakes in `sliding_window 512`, the schedule itself
  as `kWindowedAttention` (one bool per layer), the two rope bases, and
  `kAttentionScale = 0.0625` — which is `query_pre_attn_scalar ** -0.5` for a
  scalar of 256, *not* `1/sqrt(head_dim)` in general (Gemma3-27B has scalar 168
  against head_dim 128).  So the converter resolves the checkpoint's own schedule
  — from `layer_types` or from a `sliding_window_pattern` period, whichever the
  export wrote — and refuses one that disagrees, rather than letting an artifact
  load and be served with the wrong masks and rope bases.

Two facts about the release shape the code below.  It is a single unsharded
`model.safetensors` with no index, like the TinyLlama and Qwen3 releases.  And
unlike them it *does* publish `chat_template.jinja` as a file of its own — the
fallback that lifts the template out of `tokenizer_config.json` is kept anyway,
because the engine cross-checks the two and a Gemma release that shipped only
the config would otherwise be refused for a file it does not need.

`config.json` here declares `model_type: "gemma3_text"`, which the DSL registers
against `Gemma3TextModel` — the *bare backbone* EmbeddingGemma publishes, with
root-level tensors and no head.  The architecture, not the model type, is what
separates the two, so `architectures[0]` is what this converter matches on.
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
from surogate.serve.tools.convert.qwen3_6.common import conversion as family_conversion
from surogate.serve.tools.convert.qwen3_6.common.recipe import (
    SourcePreflight,
    TensorRecipe,
    expression_sources,
    materialize_recipe,
    preflight_source_reader,
    source,
)
from surogate.serve.tools.convert.qwen3_6.common.recipe import (
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory

RECIPE_ID = "gemma3-v1"

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan

#: Members of `config.json` that must hold for the registered target, and that
#: every exporter writes. Geometry is read out of the file rather than asserted
#: against a second copy of itself; these are the members that are not geometry,
#: plus the architecture identity.
#:
#: `hidden_activation`, not `hidden_act`: Gemma 3 spells it the long way, and the
#: value is the tanh approximation of GELU rather than SiLU.
#:
#: This table is checked with `check_members`, whose test is
#: `actual.get(name) != value` — an absent key reads as `None` and so is a
#: mismatch for every entry whose expected value is not `None`. That is what
#: "required" means here, and it is why the table below has to stay disjoint from
#: this one: a key named in both is required, and the optional table never gets a
#: say.
_REQUIRED_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
}

#: Members only some exporter versions write, with the value an absent key
#: asserts: present-and-wrong is refused, absent is tolerated
#: (`check_optional_members`). Nothing here may also appear above.
#:
#: `use_bidirectional_attention` separates a generative Gemma 3 from the encoder
#: backbone that shares the declaration; the engine is causal, and the key
#: post-dates the first Gemma 3 exports, which is exactly the case this table
#: exists for. The two softcapping members are absent from older configs and
#: inactive when absent; the engine implements neither, so a checkpoint that set
#: one would be served without it. `attention_dropout` is inference-inert but
#: names a checkpoint trained with a dropout the engine cannot reproduce.
_OPTIONAL_CONFIG = {
    "use_bidirectional_attention": False,
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "attention_dropout": 0.0,
}

#: What the engine holds as compile-time constants in
#: `csrc/src/serve/targets/gemma3/impl/config.h`. A checkpoint that disagrees
#: with any of them would load and be served wrong, so it is refused here.
#:
#: The local/global schedule is *not* here. It used to be, as
#: `"_sliding_window_pattern": 6`, and that entry refused every newer
#: `transformers` export: those state the schedule as a `layer_types` list and
#: write no period at all, so `check_members` saw an absent key and called it a
#: mismatch. `check_layer_schedule` below subsumes it — it accepts either
#: spelling, resolves the per-layer schedule the way the DSL does, and compares
#: the result against the array the header states.
_ENGINE_CONSTANTS = {
    "sliding_window": inventory.SLIDING_WINDOW,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    # kAttentionScale = 0.0625 = 256 ** -0.5. Not the same thing as
    # 1/sqrt(head_dim) in general, even though it coincides here.
    "query_pre_attn_scalar": 256,
    "max_position_embeddings": 32768,
}


# ---------------------------------------------------------------------------
# checkpoint geometry
# ---------------------------------------------------------------------------


def geometry_from_config(config: Mapping[str, object]) -> inventory.Geometry:
    """Read the artifact-shaping dimensions straight off `config.json`.

    `head_dim` is read, never derived. Gemma 3 decouples the head width from the
    hidden size: 270M is 4 heads of 256 against a hidden of 640, so the
    `hidden // heads` shortcut the Llama converter can take would give 160 here.
    """

    head_dim = config.get("head_dim")
    if not head_dim:
        raise ValueError(
            "config.json declares no head_dim; Gemma 3 does not derive it from "
            "hidden_size // num_attention_heads and the value cannot be guessed"
        )
    return inventory.Geometry(
        layers=int(config["num_hidden_layers"]),
        hidden=int(config["hidden_size"]),
        intermediate=int(config["intermediate_size"]),
        vocab=int(config["vocab_size"]),
        query_heads=int(config["num_attention_heads"]),
        kv_heads=int(config["num_key_value_heads"]),
        head_dim=int(head_dim),
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


def resolve_layer_schedule(
    config: Mapping[str, object],
    geometry: inventory.Geometry,
) -> list[str]:
    """The checkpoint's own local/global schedule, one entry per layer.

    Gemma 3 states this two ways and a given export writes only one of them. The
    original configs carry a *period* — `sliding_window_pattern`, written
    `_sliding_window_pattern` by the exporter of the day — and layer `i` is
    global when `(i + 1) % period == 0`, so the last layer is. Newer
    `transformers` exports resolve that themselves and ship a `layer_types` list
    instead, with no period anywhere in the file. Both are valid and both have to
    land on the same schedule.

    The rule is not restated here: `surogate/dsl/models/gemma3.py` already owns
    it, and a second copy is a second thing to get wrong. Entries come back as
    the DSL names them — `"sliding"` or `"full"`.
    """

    # Imported inside the function: the DSL package pulls in the training stack,
    # and a converter that fails to import because of it would be worse than the
    # duplication this avoids.
    from surogate.dsl.models.gemma3 import (  # noqa: PLC2701 - one rule, one owner
        _parse_gemma3_layer_types,
    )

    layer_types = config.get("layer_types")
    # Both spellings are live. `Gemma3TextConfig.__post_init__` reads
    # `sliding_window_pattern` as a back-compat kwarg and keeps it as
    # `_sliding_window_pattern`, so Hub configs written before that move carry the
    # plain name and ones written since carry the underscored one —
    # gemma-3-270m-it carries `_sliding_window_pattern: 6`. `or`, not a `.get`
    # default: a config that states the plain key as `null` would otherwise
    # shadow the underscored one that holds the value.
    period = config.get("sliding_window_pattern") or config.get("_sliding_window_pattern")
    if not layer_types and not period:
        raise ValueError(
            "config.json states no attention schedule: neither a layer_types "
            "list nor a sliding_window_pattern / _sliding_window_pattern period. "
            "Gemma 3 alternates windowed against global attention and the engine "
            "bakes the resolved schedule in, so it cannot be guessed"
        )
    return _parse_gemma3_layer_types(
        list(layer_types) if layer_types else None,
        geometry.layers,
        int(period) if period else 0,
    )


def check_layer_schedule(config: Mapping[str, object], geometry: inventory.Geometry) -> None:
    """The schedule the engine bakes in, against the one the checkpoint states.

    The target header states the schedule as data — `config.h::kWindowedAttention`,
    one bool per layer, read through `is_windowed_attention(layer)` — and
    `inventory.WINDOWED_ATTENTION` is this side's copy of it. A checkpoint that
    disagreed would be served with the wrong mask and the wrong rope base on
    every layer where they differ, and nothing downstream would notice: a
    windowed layer and a global one store identical objects.
    """

    resolved = resolve_layer_schedule(config, geometry)
    expected = ["sliding" if windowed else "full" for windowed in inventory.WINDOWED_ATTENTION]
    disagreeing = [
        layer for layer, (got, want) in enumerate(zip(resolved, expected)) if got != want
    ]
    if disagreeing:
        detail = ", ".join(
            f"{layer}: checkpoint {resolved[layer]}, target {expected[layer]}"
            for layer in disagreeing[:8]
        )
        raise ValueError(
            "checkpoint attention schedule disagrees with the one "
            "csrc/src/serve/targets/gemma3/impl/config.h states as "
            f"kWindowedAttention; {len(disagreeing)} of {geometry.layers} layers "
            f"disagree ({detail}"
            f"{', ...' if len(disagreeing) > 8 else ''})"
        )


def validate_config(config: Mapping[str, object]) -> tuple[inventory.Geometry, dict]:
    """Validate the checkpoint and summarize it for the conversion report."""

    family_conversion.check_members("config", config, _REQUIRED_CONFIG)
    check_optional_members("config", config, _OPTIONAL_CONFIG)
    family_conversion.check_members("config", config, _ENGINE_CONSTANTS)
    geometry = geometry_from_config(config)
    if geometry != inventory.GEOMETRY:
        raise ValueError(
            "checkpoint geometry is not the registered gemma3 target:\n"
            f"  checkpoint {geometry}\n"
            f"  target     {inventory.GEOMETRY}\n"
            "csrc/src/serve/targets/gemma3/impl/config.h describes one size; a "
            "differently sized Gemma3ForCausalLM needs its own target header "
            "before its artifact can be bound."
        )
    check_layer_schedule(config, geometry)
    text = {
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
            "rope_local_base_freq",
            "sliding_window",
            # Whichever spelling this export used; newer ones write neither and
            # state `layer_types` instead. The report records what was read
            # rather than a normalisation of it.
            "sliding_window_pattern",
            "_sliding_window_pattern",
            "layer_types",
            "query_pre_attn_scalar",
            "max_position_embeddings",
            "attention_bias",
        )
        if name in config
    }
    # Recorded as derived: the checkpoint omits the key and the architecture ties.
    text["tie_word_embeddings"] = tied_output_head(config)
    summary = {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": text,
        "attention": {
            "query_size": geometry.query_size,
            "kv_size": geometry.kv_size,
            "fused_rows": None,  # Q, K and V are stored as three objects
            "output_gate": False,
            "qk_norm": True,
            "global_layers": list(inventory.GLOBAL_ATTENTION_LAYERS),
            # The resolved schedule, so the report says which layers were served
            # windowed whichever way the checkpoint spelled it.
            "layer_schedule": resolve_layer_schedule(config, geometry),
        },
        "norms": {
            "per_layer": 4,
            "unit_offset": True,
            "stored": "w (unfolded); the runtime applies 1 + w",
        },
        # sqrt(hidden), applied by the engine and not folded into the embedding.
        "embedding_scale": float(geometry.hidden) ** 0.5,
        "vision": None,  # Gemma3ForCausalLM is text-only
        "mtp_num_hidden_layers": 0,
    }
    return geometry, summary


def tied_output_head(config: Mapping[str, object]) -> bool:
    """Whether the head reads the embedding table.

    Defaults to true, where the Llama and Qwen3 converters default to false.
    That is not a style difference: `Gemma3TextConfig` ties by default, this
    checkpoint writes no `tie_word_embeddings` key at all, and it ships no
    `lm_head.weight` — so reading the key with a false default would send the
    recipe looking for a tensor that does not exist.
    """

    return bool(config.get("tie_word_embeddings", True))


# ---------------------------------------------------------------------------
# source recipe
# ---------------------------------------------------------------------------


def build_recipes(
    geometry: inventory.Geometry = inventory.GEOMETRY,
    *,
    tied_output_head: bool = True,
) -> tuple[TensorRecipe, ...]:
    """Where every artifact object comes from in the checkpoint, in object order.

    Every entry is a plain read. Gemma 3 fuses nothing, so there is no `Concat`
    anywhere in this recipe — and the norms are copied rather than adjusted,
    because a safetensors checkpoint already stores the unfolded `w` the runtime
    wants (see the module docstring).
    """

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
                    obj + "post_attention_norm",
                    source(src + "post_attention_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "pre_feedforward_norm",
                    source(src + "pre_feedforward_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "post_feedforward_norm",
                    source(src + "post_feedforward_layernorm.weight", (hidden,)),
                ),
                TensorRecipe(
                    obj + "attention/query",
                    source(src + "self_attn.q_proj.weight", (query, hidden)),
                ),
                TensorRecipe(
                    obj + "attention/key",
                    source(src + "self_attn.k_proj.weight", (kv, hidden)),
                ),
                TensorRecipe(
                    obj + "attention/value",
                    source(src + "self_attn.v_proj.weight", (kv, hidden)),
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
                    obj + "mlp/gate",
                    source(src + "mlp.gate_proj.weight", (geometry.intermediate, hidden)),
                ),
                TensorRecipe(
                    obj + "mlp/up",
                    source(src + "mlp.up_proj.weight", (geometry.intermediate, hidden)),
                ),
                TensorRecipe(
                    obj + "mlp/down",
                    source(src + "mlp.down_proj.weight", (hidden, geometry.intermediate)),
                ),
            )
        )

    recipes.append(TensorRecipe("text/final_norm", source("model.norm.weight", (hidden,))))
    if not tied_output_head:
        # `tie_word_embeddings` is a property of the checkpoint in hand, not of
        # the architecture, so an untied Gemma 3 stores a head of its own and
        # reads it from `lm_head.weight`. Every published `Gemma3ForCausalLM`
        # ties, and gemma-3-270m-it ships no `lm_head.weight` at all, so this is
        # the branch that does not run today.
        recipes.append(
            TensorRecipe(
                "text/output_head", source("lm_head.weight", (geometry.vocab, hidden))
            )
        )
    # The tied case has no `text/output_head` recipe because it has no such
    # object: `inventory.ALIAS_SPECS` makes the head a role served by
    # `text/token_embedding`, and the binder fills both from the one table.
    # Giving it the embedding's expression instead would quantise 167.8M elements
    # a second time and write ~170 MB of byte-identical duplicate.
    return tuple(recipes)


RECIPE_SPECS = build_recipes()
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage(
    recipes: Sequence[TensorRecipe] = RECIPE_SPECS,
    *,
    tied_output_head: bool = True,
) -> None:
    stored, _ = inventory.active_specs(tied_output_head=tied_output_head)
    _validate_recipe_coverage(recipes, stored)


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

    Gemma 3 270M is one 536 MB `model.safetensors` with no index at all, and a
    reader that only knew how to follow an index could not open it; the larger
    Gemma 3 releases are sharded, so both doors stay open.
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

    Gemma 3 ships `tokenizer.json` — the fast-tokenizer serialization of its
    SentencePiece vocabulary — beside the `tokenizer.model` the tokenizer was
    originally distributed as. The engine binds the former even now that it hands
    SentencePiece checkpoints to the project tokenizer, because the scheme, the
    vocabulary and the added tokens are all read out of the JSON; the `.model`
    holds the same vocabulary in a format nothing here reads, so it is not
    carried and an artifact carrying it would be refused.

    **The chat template travels the other way round here**, and it is the one
    place this converter rewrites a checkpoint file rather than copying it.
    TinyLlama and Qwen state the template only inside `tokenizer_config.json`,
    and their converters synthesize `chat_template.jinja` from it.  Gemma 3 is
    the newer `transformers` convention: the template is a file of its own and
    `tokenizer_config.json` carries no `chat_template` key at all.  The engine
    requires both and requires them equal —
    `family/impl/frontend/frontend.cpp` throws
    "tokenizer_config.json.chat_template must contain the loaded chat template"
    on an artifact that omits the key — so a byte-for-byte copy of this
    checkpoint's `tokenizer_config.json` would be refused at load.  The template
    is therefore carried into the config, by a minimal textual insertion that
    leaves every other byte of the file untouched.  Both directions of the
    synthesis are kept: a Gemma release that states the template only in the
    config still converts.
    """

    root = Path(model_dir)
    template = _chat_template(root)
    payloads: list[ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if filename == "chat_template.jinja":
            data = template
        elif filename == "tokenizer_config.json":
            data = _tokenizer_config_with_template(path.read_bytes(), template)
        elif path.exists():
            data = path.read_bytes()
        elif filename == "generation_config.json":
            data = family_conversion._synthesize_generation_config(root)  # noqa: SLF001
        elif filename == "tokenizer.json":
            raise FileNotFoundError(
                "checkpoint is missing tokenizer.json; a Gemma release that ships "
                "only the SentencePiece tokenizer.model must be converted to the "
                "fast-tokenizer serialization before it can be bound"
            )
        else:
            raise FileNotFoundError(f"checkpoint is missing {filename}")
        if not data:
            raise ValueError(f"frontend resource {filename} is empty")
        payloads.append(ResourcePayload(spec.name, data))
    return tuple(payloads)


def _chat_template(root: Path) -> bytes:
    """The template the artifact serves, from whichever place the release states it.

    Written out verbatim, with no added trailing newline: the engine compares it
    byte for byte against the copy in `tokenizer_config.json`.
    """

    path = root / "chat_template.jinja"
    if path.exists():
        return path.read_bytes()
    config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    template = config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError(
            "checkpoint publishes neither chat_template.jinja nor "
            "tokenizer_config.json.chat_template; the engine needs one"
        )
    return template.encode("utf-8")


def _tokenizer_config_with_template(raw: bytes, template: bytes) -> bytes:
    """`tokenizer_config.json`, guaranteed to state the template the artifact serves.

    Returned unchanged when the config already states it — which is the Qwen and
    TinyLlama case, and where an existing key that *disagrees* with
    `chat_template.jinja` is a checkpoint contradicting itself and is refused
    rather than silently normalized. Where the key is absent, it is inserted
    immediately after the opening brace so that every other byte of the file
    survives; re-serializing 1.1 MB of `added_tokens_decoder` to add one member
    would rewrite the whole file to no purpose.
    """

    config = json.loads(raw.decode("utf-8"))
    existing = config.get("chat_template")
    if isinstance(existing, str):
        if existing.encode("utf-8") != template:
            raise ValueError(
                "tokenizer_config.json.chat_template disagrees with "
                "chat_template.jinja; the engine compares the two and would "
                "refuse the artifact"
            )
        return raw
    if existing is not None:
        raise ValueError(
            "tokenizer_config.json.chat_template is not a string; the engine "
            f"requires one, got {type(existing).__name__}"
        )
    member = f'"chat_template": {json.dumps(template.decode("utf-8"), ensure_ascii=False)}'
    if not config:
        return ("{" + member + "}").encode("utf-8")
    text = raw.decode("utf-8")
    brace = text.index("{")
    return (text[: brace + 1] + member + "," + text[brace + 1 :]).encode("utf-8")


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
    #: What this checkpoint stores, which is one object short of the declaration
    #: when the head is tied. The writer walks this, not the module-level list.
    object_specs: tuple[inventory.StoredObjectSpec, ...]
    tied_output_head: bool

    @property
    def recipes_by_name(self) -> dict[str, TensorRecipe]:
        return {recipe.object_name: recipe for recipe in self.recipes}


def preflight_inventory(*, tied_output_head: bool = True) -> None:
    """The inventory the recipe and the writer agree to produce.

    The embedding, thirteen objects a layer, the final norm — and the output head
    only where the checkpoint unties it. A tied checkpoint stores the head as a
    role on `text/token_embedding` (`inventory.ALIAS_SPECS`), so it is one object
    short of the declaration, which is the count `TENSOR_SPECS` still carries.
    """

    declared_tensors = 1 + inventory.LAYERS * inventory.LAYER_OBJECT_COUNT + 2
    if len(inventory.TENSOR_SPECS) != declared_tensors:
        raise ValueError(
            f"registered inventory holds {len(inventory.TENSOR_SPECS)} tensors, "
            f"expected {declared_tensors}"
        )
    stored_tensors, object_specs = inventory.active_specs(tied_output_head=tied_output_head)
    expected_stored = declared_tensors - (1 if tied_output_head else 0)
    if len(stored_tensors) != expected_stored:
        raise ValueError(
            f"stored inventory holds {len(stored_tensors)} tensors, "
            f"expected {expected_stored}"
        )
    if len(inventory.RESOURCE_SPECS) != 4:
        raise ValueError("registered inventory does not hold the four text resources")
    if len(object_specs) != expected_stored + 4:
        raise ValueError("registered object inventory is incomplete")
    validate_recipe_coverage(
        build_recipes(tied_output_head=tied_output_head),
        tied_output_head=tied_output_head,
    )


def build_object_plan(
    resources: Mapping[str, bytes],
    *,
    tied_output_head: bool = True,
) -> ObjectPlan:
    preflight_inventory(tied_output_head=tied_output_head)
    _, object_specs = inventory.active_specs(tied_output_head=tied_output_head)
    return family_conversion.build_object_plan(object_specs, resources)


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config = family_conversion.load_json(model / "config.json")
    geometry, summary = validate_config(config)
    # Whether the head is its own object is a property of the checkpoint, not of
    # the target, so both the recipe and the object list are built for the
    # checkpoint in hand rather than the module-level ones being used blind.
    tied = tied_output_head(config)
    preflight_inventory(tied_output_head=tied)
    recipes = build_recipes(geometry, tied_output_head=tied)
    stored_specs, object_specs = inventory.active_specs(tied_output_head=tied)
    _validate_recipe_coverage(recipes, stored_specs)
    source_preflight = preflight_sources(model, recipes)
    resources = load_resources(model)
    plan = build_object_plan(
        {item.name: item.data for item in resources}, tied_output_head=tied
    )
    return ConversionPreflight(
        model_dir=model,
        geometry=geometry,
        config_summary=summary,
        recipes=recipes,
        source=source_preflight,
        resources=resources,
        object_plan=plan,
        object_specs=object_specs,
        tied_output_head=tied,
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
            total = len(preflight.object_specs)
            for index, spec in enumerate(preflight.object_specs, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, recipes)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 25 == 0 or index == total:
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
