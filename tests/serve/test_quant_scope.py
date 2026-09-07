# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# A checkpoint's `quantization_config` states more than its format. These cover the two
# things the converter now does with the rest of it: refuse what the serving path cannot
# honour, and report where the declaration disagrees with the checkpoint's own tensors.
# CPU-only.
from __future__ import annotations

import pytest

from surogate.serve.convert.common import quant_scope as qs


def config(**quant):
    return {"quantization_config": {"quant_method": "compressed-tensors", **quant}}


# --------------------------------------------------------------------------- refusal


def test_a_scheme_the_engine_has_no_cache_for_is_refused():
    with pytest.raises(qs.QuantScopeError, match="integer KV cache"):
        qs.kv_cache_request(config(kv_cache_scheme={"type": "int", "num_bits": 8}))
    with pytest.raises(qs.QuantScopeError, match="names no"):
        qs.kv_cache_request(config(kv_cache_scheme={"type": "float", "num_bits": 4}))


def test_a_cache_request_the_default_already_satisfies_is_honoured_not_refused():
    """Two published 27B NVFP4 exports ask for an 8-bit float KV cache. On a stack whose
    linear-attention layers carry it, `auto` resolves to e4m3 and the request is already
    met; refusing there would turn a served checkpoint into an unserved one."""
    asks_fp8 = config(kv_cache_scheme={"type": "float", "num_bits": 8, "dynamic": False})
    assert qs.kv_cache_request(asks_fp8) == "fp8"
    assert qs.auto_kv_dtype(gdn_layers=48) == "fp8"
    qs.require_honourable(asks_fp8, gdn_layers=48)  # honoured, no raise

    # The same request on a pure-attention stack is a different cache from the one asked for.
    with pytest.raises(qs.QuantScopeError, match="--kv-cache-dtype fp8"):
        qs.require_honourable(asks_fp8, gdn_layers=0)


def test_a_structure_the_engine_does_not_implement_is_refused():
    for field in ("sparsity_config", "transform_config"):
        with pytest.raises(qs.QuantScopeError, match=field):
            qs.require_honourable(config(**{field: {"format": "2:4"}}), gdn_layers=0)
    # Present and empty is how every published checkpoint carries them: not a request.
    qs.require_honourable(config(sparsity_config={}, transform_config={}), gdn_layers=0)


def test_an_unquantised_checkpoint_declares_nothing_and_is_never_refused():
    assert qs.declared_scope({}).patterns == ()
    assert qs.kv_cache_request({}) is None
    qs.require_honourable({}, gdn_layers=0)


# --------------------------------------------------------------------------- cross-check


def test_a_weight_is_quantised_exactly_when_a_scale_sits_beside_it():
    observed = qs.observed_scope((
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.k_proj.weight",          # no scale: left alone
        "model.layers.1.mlp.gate_proj.weight_packed",      # codes, compressed-tensors
        "model.layers.1.mlp.gate_proj.weight_scale",
        "model.layers.0.input_layernorm.weight",
    ))
    assert "model.layers.0.self_attn.q_proj" in observed.quantised
    assert "model.layers.1.mlp.gate_proj" in observed.quantised
    assert "model.layers.0.self_attn.k_proj" in observed.plain
    assert observed.layers_of("self_attn.k_proj", quantised=False) == (0,)
    assert observed.layers_of("self_attn.q_proj", quantised=True) == (0,)


def test_the_declaration_is_a_claim_and_the_tensors_are_the_fact():
    """`nvidia/Qwen3.6-27B-NVFP4` names two modules in `ignore` and ships twenty-seven
    unquantised vision projections that it never mentions. The point is that this is
    reported rather than absorbed."""
    declared = qs.declared_scope(config(ignore=["lm_head"]))
    observed = qs.observed_scope((
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.visual.blocks.0.attn.qkv.weight",   # plain, and nothing declared it
        "lm_head.weight",                          # plain, and `ignore` said so
    ))
    differ = qs.disagreement(declared, observed)
    assert differ, "an undeclared plain projection must be reported"
    assert differ.undeclared_but_plain == ("model.visual.blocks.0.attn.qkv",)
    assert "lm_head" not in differ.undeclared_but_plain
    assert "stored plain" in differ.describe()


def test_a_module_named_by_ignore_but_quantised_is_reported_too():
    declared = qs.declared_scope(config(ignore=["re:.*mlp.*"]))
    observed = qs.observed_scope((
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.gate_proj.weight_scale",
    ))
    differ = qs.disagreement(declared, observed)
    assert differ.declared_but_quantised == ("model.layers.0.mlp.gate_proj",)


def test_ignore_patterns_match_the_way_compressed_tensors_writes_them():
    declared = qs.declared_scope(config(ignore=["re:^mtp.*", "model.visual*", "lm_head"]))
    assert declared.ignores("mtp.layers.0.self_attn.q_proj")
    assert declared.ignores("model.visual.blocks.3.attn.qkv")
    assert declared.ignores("lm_head")
    assert not declared.ignores("model.layers.0.self_attn.q_proj")


def test_norms_and_embeddings_are_not_a_disagreement():
    """No profile quantises them, so their absence from `ignore` says nothing."""
    declared = qs.declared_scope(config(ignore=["lm_head"]))
    observed = qs.observed_scope((
        "model.layers.0.input_layernorm.weight",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.bias",
    ))
    assert not qs.disagreement(declared, observed)


def test_report_is_empty_for_an_unquantised_checkpoint_and_names_the_verdict_otherwise():
    assert qs.report({}, ("model.embed_tokens.weight",)) == ""
    line = qs.report(config(ignore=["lm_head"]),
                     ("model.layers.0.self_attn.q_proj.weight",
                      "model.layers.0.self_attn.q_proj.weight_scale"))
    assert "compressed-tensors" in line and line.endswith("agrees")


def test_a_sub_stack_layer_is_not_a_text_layer():
    """`mtp.layers.0` and a vision block number their own stacks from zero. Folding those
    onto text layer 0 reported an exception that was not one, on both published 27B NVFP4
    exports, until the lookup was anchored to the text stack."""
    observed = qs.observed_scope((
        "mtp.layers.0.self_attn.q_proj.weight",              # the draft head, left alone
        "model.visual.blocks.0.attn.qkv.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight_scale",
    ))
    assert observed.layers_of("self_attn.q_proj", quantised=False) == ()
    assert observed.layers_of("self_attn.q_proj", quantised=True) == (3,)


def test_a_stale_exception_table_is_a_disagreement_not_a_wrong_artifact():
    """The export tables were measured from published files and written down. Both
    `nvidia/Qwen3.6-27B-NVFP4` and `unsloth/Qwen3.6-27B-NVFP4` quantise every attention layer,
    so neither matches `_BF16_ATTENTION_INPUT_LAYERS`; the converter refuses rather than
    building an artifact that claims formats its own weights do not have."""
    from surogate.serve.convert.qwen3_5 import inventory as inv

    export = inv.export_for(inv.NVFP4_MIXED_BF16, inv.GEOMETRY_27B)
    assert export.exceptions, "this profile is the one that carries measured exceptions"

    # A checkpoint that quantises every attention layer, as both published ones do.
    every_layer_quantised = qs.observed_scope(tuple(
        f"model.language_model.layers.{layer}.self_attn.{proj}.weight{suffix}"
        for layer in inv.GEOMETRY_27B.full_attention_layers
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        for suffix in ("", "_scale")
    ))
    differ = inv.exception_disagreement(export, every_layer_quantised)
    assert "attention_input" in differ
    table, found = differ["attention_input"]
    assert table == (3, 7, 11, 15, 19, 23) and found == ()

    # And a checkpoint that matches the table is not a disagreement.
    matching = qs.observed_scope(
        tuple(f"model.language_model.layers.{layer}.self_attn.{proj}.weight"
              for layer in (3, 7, 11, 15, 19, 23) for proj in ("q_proj", "k_proj", "v_proj"))
        + tuple(f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
                for layer in (3, 7))
        + tuple(f"model.language_model.layers.{layer}.linear_attn.out_proj.weight"
                for layer in (4,))
    )
    assert "attention_input" not in inv.exception_disagreement(export, matching)
