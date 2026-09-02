# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Config validation and the artifact object list for the Gemma 3 converter
# (surogate/serve/tools/convert/gemma3/). Every case here is a synthetic
# `config.json` dict: none of it reads a checkpoint, opens a safetensors shard
# or touches a GPU, because what is under test is what the converter *refuses*
# and what it agrees to store, both of which are decided before a tensor is
# read.

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from surogate.serve.tools.convert.gemma3 import convert, inventory  # noqa: E402

#: `google/gemma-3-270m-it`'s own `config.json`, member for member for everything
#: this converter reads: the schedule stated both as a `layer_types` list and as
#: an `_sliding_window_pattern` period, and no `tie_word_embeddings` key at all
#: (`Gemma3TextConfig` ties by default).
GEMMA3_270M_CONFIG: dict[str, object] = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3_text",
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "final_logit_softcapping": None,
    "use_bidirectional_attention": False,
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "rope_local_base_freq": 10000.0,
    "query_pre_attn_scalar": 256,
    "max_position_embeddings": 32768,
    "sliding_window": 512,
    "_sliding_window_pattern": 6,
    "layer_types": [
        "full_attention" if layer in inventory.GLOBAL_ATTENTION_LAYERS else "sliding_attention"
        for layer in range(inventory.LAYERS)
    ],
    "num_hidden_layers": inventory.LAYERS,
    "hidden_size": inventory.HIDDEN,
    "intermediate_size": inventory.INTERMEDIATE,
    "vocab_size": inventory.VOCAB,
    "num_attention_heads": inventory.QUERY_HEADS,
    "num_key_value_heads": inventory.KV_HEADS,
    "head_dim": inventory.HEAD_DIM,
}


def config(**overrides: object) -> dict[str, object]:
    """The reference config with members replaced, or dropped by passing `None`
    to `without` — see the callers; `None` is a *value* several members hold."""

    out = copy.deepcopy(GEMMA3_270M_CONFIG)
    out.update(overrides)
    return out


def without(*names: str, **overrides: object) -> dict[str, object]:
    out = config(**overrides)
    for name in names:
        out.pop(name, None)
    return out


# ---------------------------------------------------------------------------
# optional config members
# ---------------------------------------------------------------------------


def test_reference_config_validates() -> None:
    geometry, summary = convert.validate_config(GEMMA3_270M_CONFIG)
    assert geometry == inventory.GEOMETRY
    assert summary["architecture"] == "Gemma3ForCausalLM"
    # Recorded as derived: this checkpoint writes no `tie_word_embeddings` key.
    assert summary["text"]["tie_word_embeddings"] is True


@pytest.mark.parametrize("absent", sorted(convert._OPTIONAL_CONFIG))
def test_an_export_that_omits_an_optional_member_still_converts(absent: str) -> None:
    """The whole point of the optional table.

    `use_bidirectional_attention` post-dates the first Gemma 3 exports and
    `attention_dropout` is absent from some too; both used to be in the required
    table as well, and the required check runs first with
    `actual.get(name) != value` — so an export missing either was refused before
    a tensor was read, by the very check the optional table exists to relax.
    """

    geometry, _ = convert.validate_config(without(absent))
    assert geometry == inventory.GEOMETRY


def test_the_two_config_tables_are_disjoint() -> None:
    """A key in both is required, and the optional table never gets a say."""

    assert not set(convert._REQUIRED_CONFIG) & set(convert._OPTIONAL_CONFIG)


@pytest.mark.parametrize(
    "member,wrong",
    [
        ("use_bidirectional_attention", True),
        ("attention_dropout", 0.1),
        ("attn_logit_softcapping", 30.0),
        ("final_logit_softcapping", 30.0),
    ],
)
def test_an_optional_member_that_is_present_and_wrong_is_refused(
    member: str, wrong: object
) -> None:
    """Optional means absent-is-fine, not unchecked. The engine implements no
    softcapping and is causal, so a checkpoint that asks for either would be
    served without it."""

    with pytest.raises(ValueError, match=member):
        convert.validate_config(config(**{member: wrong}))


def test_a_required_member_that_is_absent_is_still_refused() -> None:
    with pytest.raises(ValueError, match="hidden_activation"):
        convert.validate_config(without("hidden_activation"))


# ---------------------------------------------------------------------------
# the local/global attention schedule
# ---------------------------------------------------------------------------


def test_layer_types_that_disagree_with_the_target_schedule_are_refused() -> None:
    """The header states the schedule as data (`config.h::kWindowedAttention`).

    A checkpoint that windowed different layers would load and be served with the
    wrong mask and the wrong rope base on each one, in silence: a windowed layer
    and a global one store identical objects. The refusal names the layers.
    """

    layer_types = list(GEMMA3_270M_CONFIG["layer_types"])
    # Gemma counts the period from the end, so layer 5 is global and layer 6 is
    # windowed. Swap exactly those two.
    layer_types[5], layer_types[6] = layer_types[6], layer_types[5]

    with pytest.raises(ValueError) as excinfo:
        convert.validate_config(config(layer_types=layer_types))

    message = str(excinfo.value)
    assert "kWindowedAttention" in message
    assert "2 of 18 layers disagree" in message
    assert "5: checkpoint sliding, target full" in message
    assert "6: checkpoint full, target sliding" in message


def test_a_layer_types_only_export_converts() -> None:
    """The case the old `_ENGINE_CONSTANTS` entry refused.

    Newer `transformers` exports resolve the period themselves and write only
    `layer_types`; `_sliding_window_pattern` was checked with
    `actual.get(name) != value`, so its absence read as a mismatch and a valid
    checkpoint was refused by the one path the schedule check exists to handle.
    """

    geometry, summary = convert.validate_config(
        without("sliding_window_pattern", "_sliding_window_pattern")
    )
    assert geometry == inventory.GEOMETRY
    assert summary["attention"]["layer_schedule"][:6] == [
        "sliding",
        "sliding",
        "sliding",
        "sliding",
        "sliding",
        "full",
    ]


@pytest.mark.parametrize("spelling", ["sliding_window_pattern", "_sliding_window_pattern"])
def test_a_period_only_export_converts_in_either_spelling(spelling: str) -> None:
    """And the period resolves to the same schedule the list states."""

    period_only = without("layer_types", "sliding_window_pattern", "_sliding_window_pattern")
    period_only[spelling] = inventory.SLIDING_WINDOW_PERIOD
    _, summary = convert.validate_config(period_only)
    assert summary["attention"]["layer_schedule"] == [
        "sliding" if windowed else "full" for windowed in inventory.WINDOWED_ATTENTION
    ]


def test_an_explicit_null_period_does_not_shadow_the_underscore_spelling() -> None:
    """`Gemma3TextConfig` reads `sliding_window_pattern` as a back-compat kwarg
    and keeps it as `_sliding_window_pattern`, so a config can carry both. A
    `.get(plain, config.get(underscored))` default would be skipped by a present
    `null` and lose the period that is actually there."""

    both = without("layer_types")
    both["sliding_window_pattern"] = None
    _, summary = convert.validate_config(both)
    assert summary["attention"]["layer_schedule"] == [
        "sliding" if windowed else "full" for windowed in inventory.WINDOWED_ATTENTION
    ]


def test_a_period_that_disagrees_with_the_target_schedule_is_refused() -> None:
    period_only = without("layer_types", "sliding_window_pattern", "_sliding_window_pattern")
    period_only["sliding_window_pattern"] = 4
    with pytest.raises(ValueError, match="kWindowedAttention"):
        convert.validate_config(period_only)


def test_a_config_stating_no_schedule_at_all_is_refused() -> None:
    """Neither spelling means the schedule cannot be resolved, and the engine
    bakes it in — so it must not be guessed."""

    with pytest.raises(ValueError, match="no attention schedule"):
        convert.validate_config(
            without("layer_types", "sliding_window_pattern", "_sliding_window_pattern")
        )


def test_the_target_schedule_matches_the_dsl_derivation() -> None:
    """`inventory.WINDOWED_ATTENTION` mirrors the header; the DSL owns the rule.
    Restating it would be a second thing to get wrong."""

    from surogate.dsl.models.gemma3 import _parse_gemma3_layer_types

    assert _parse_gemma3_layer_types(
        None, inventory.LAYERS, inventory.SLIDING_WINDOW_PERIOD
    ) == ["sliding" if windowed else "full" for windowed in inventory.WINDOWED_ATTENTION]


# ---------------------------------------------------------------------------
# the tied output head
# ---------------------------------------------------------------------------


def test_the_tied_output_head_is_aliased_not_stored() -> None:
    """The largest object in the run, stored once.

    `text/output_head` is declared — the model has an `lm_head` — but a tied
    checkpoint stores it as a role on `text/token_embedding`, so the artifact
    carries one 262144x640 table rather than two byte-identical ones.
    """

    declared = [spec.name for spec in inventory.TENSOR_SPECS]
    stored = [spec.name for spec in inventory.STORED_TENSOR_SPECS]

    assert "text/output_head" in declared
    assert "text/output_head" not in stored
    assert set(declared) - set(stored) == {"text/output_head"}
    assert stored[-1] == "text/final_norm"
    assert inventory.ALIAS_SPECS == (
        inventory.ALIAS_SPECS[0].__class__("text/output_head", ("text/token_embedding",)),
    )
    assert inventory.ALIASED_OBJECT_NAMES == {"text/output_head"}


def test_the_recipe_does_not_quantise_the_embedding_twice() -> None:
    """The head used to carry the embedding's own source expression, so the
    single largest quantization in the run was done twice."""

    tied = convert.build_recipes()
    assert [r.object_name for r in tied] == [s.name for s in inventory.STORED_TENSOR_SPECS]
    assert "text/output_head" not in {r.object_name for r in tied}

    untied = convert.build_recipes(tied_output_head=False)
    assert [r.object_name for r in untied] == [s.name for s in inventory.TENSOR_SPECS]
    head = next(r for r in untied if r.object_name == "text/output_head")
    assert "lm_head.weight" in {s.name for s in convert.expression_sources(head.expression)}


def test_the_alias_removes_one_embedding_table_from_the_artifact() -> None:
    """What the alias is worth, in bytes the writer would actually emit."""

    resources = {spec.name: b"{}" for spec in inventory.RESOURCE_SPECS}
    tied = convert.build_object_plan(resources, tied_output_head=True)
    untied = convert.build_object_plan(resources, tied_output_head=False)

    assert len(untied.objects) - len(tied.objects) == 1
    # 262144 x 640 as W8G32_F16S: one byte an element plus one f16 scale per 32.
    elements = inventory.VOCAB * inventory.HIDDEN
    assert untied.payload_span_bytes - tied.payload_span_bytes == elements + elements // 32 * 2


def test_preflight_accepts_both_artifact_shapes() -> None:
    """`preflight_inventory` is the check that actually blocks a conversion run,
    and it counts objects — so it has to know which shape it is counting."""

    convert.preflight_inventory(tied_output_head=True)
    convert.preflight_inventory(tied_output_head=False)
    assert len(inventory.OBJECT_SPECS) == len(inventory.STORED_TENSOR_SPECS) + 4
