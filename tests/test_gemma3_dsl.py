"""The Gemma 3 declaration, checked against the checkpoint it describes.

CPU-only and, except where noted, weightless: these pin the properties that are
easy to get silently wrong and that a numerical parity run would only catch as a
vague drift.

Three of them have already been wrong somewhere in this repo:

* **Direction.** ``use_bidirectional_attention`` has to survive all the way into
  the emitted op. It did not until 2026-08-31 -- the DSL emitted ``causal``,
  ``graph_compiler.cpp`` never parsed it, and every backend hardcoded true, so
  the vision tower trained causally for a year while declaring otherwise.

* **Softmax scale.** EmbeddingGemma's ``query_pre_attn_scalar`` happens to equal
  its ``head_dim`` (both 256), so a wrong implementation that inherits the
  kernel's 1/sqrt(head_dim) default looks perfect on this checkpoint and breaks
  on Gemma3-27B, whose scalar is 168 against head_dim 128.

* **Schedule.** Gemma 3 states local-vs-global as a period, not a list.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from surogate.dsl import models as _models  # noqa: F401 - registers @hf_config
from surogate.dsl.models.gemma3 import (
    Gemma3CausalModel,
    Gemma3TextModel,
    _build_gemma3_mappings,
)
from surogate.dsl.py_compiler import compile_model_for_hf


#: The published config for google/embeddinggemma-300m. Inlined rather than read
#: from the hub: it is the contract under test, so a silent upstream edit should
#: fail here rather than quietly redefine what we assert.
EMBEDDINGGEMMA_CONFIG = {
    "architectures": ["Gemma3TextModel"],
    "model_type": "gemma3_text",
    "_sliding_window_pattern": 6,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 768,
    "intermediate_size": 1152,
    "max_position_embeddings": 2048,
    "num_attention_heads": 3,
    "num_hidden_layers": 24,
    "num_key_value_heads": 1,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "sliding_window": 512,
    "use_bidirectional_attention": True,
    "vocab_size": 262144,
}

CHECKPOINT = Path(
    "~/.cache/huggingface/hub/models--google--embeddinggemma-300m/"
    "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
).expanduser()


def build_from_config(cfg: dict) -> Gemma3TextModel:
    return Gemma3TextModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["hidden_size"],
        n_layers=cfg["num_hidden_layers"],
        num_query_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        d_ff=cfg["intermediate_size"],
        max_seq=cfg["max_position_embeddings"],
        head_size=cfg["head_dim"],
        eps=cfg["rms_norm_eps"],
        sliding_window=cfg["sliding_window"],
        sliding_window_pattern=cfg["_sliding_window_pattern"],
        sliding_rope_theta=cfg["rope_local_base_freq"],
        full_rope_theta=cfg["rope_theta"],
        query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
        use_bidirectional_attention=cfg["use_bidirectional_attention"],
    )


def attention_ops(cfg: dict, arch: str = "Gemma3TextModel") -> list[dict]:
    ir = json.loads(compile_model_for_hf(arch, cfg))
    assert ir["success"], ir.get("errors")
    ops = ir["modules"][0]["forward"]["operations"]
    return [o for o in ops if o["kernel_type"] == "flash_attention"]


# ---------------------------------------------------------------------------
# Geometry and schedule
# ---------------------------------------------------------------------------


def test_layer_schedule_comes_from_the_period():
    """Gemma 3 publishes ``_sliding_window_pattern``, not a layer_types list.

    Period 6 over 24 layers puts global attention at 5/11/17/23, which is what
    the checkpoint has. Getting the phase wrong (0/6/12/18) would still produce
    four global layers and a model that trains.
    """
    model = build_from_config(EMBEDDINGGEMMA_CONFIG)
    assert model.n_layers == 24
    assert [i for i, t in enumerate(model.block_types) if t == "full"] == [5, 11, 17, 23]
    assert (model.n_sliding_blocks, model.n_full_blocks) == (20, 4)


def test_geometry_matches_the_projection_shapes():
    """QKV fuses q[768,768] + k[256,768] + v[256,768] -> 1280 rows."""
    from surogate.dsl.blocks.gemma3 import Gemma3SlidingBlock

    block = Gemma3SlidingBlock(768, 3, 1, 256, 1152, 2048)
    assert block.QKV == 3 * 256 + 256 + 256 == 1280
    assert block.AttnDim == 3 * 256 == 768


def test_both_rope_bases_are_carried():
    """Local layers use a 100x smaller base than global ones; dsl_model.cpp
    reads these two config attributes per layer type."""
    model = build_from_config(EMBEDDINGGEMMA_CONFIG)
    assert model.sliding_rope_theta == 10000.0
    assert model.full_rope_theta == 1000000.0


# ---------------------------------------------------------------------------
# What the compiled IR actually emits
# ---------------------------------------------------------------------------


def test_every_attention_op_is_bidirectional():
    """The whole point of the checkpoint. A causal EmbeddingGemma still produces
    plausible-looking embeddings, so nothing downstream would notice."""
    ops = attention_ops(EMBEDDINGGEMMA_CONFIG)
    assert len(ops) == 24
    assert all(op["attrs"]["causal"] is False for op in ops)


def test_window_is_emitted_on_local_layers_only():
    ops = attention_ops(EMBEDDINGGEMMA_CONFIG)
    windowed = [o for o in ops if "window_size" in o["attrs"]]
    assert len(windowed) == 20
    assert all(o["attrs"]["window_size"] == 512 for o in windowed)
    assert len(ops) - len(windowed) == 4


def test_softmax_scale_is_declared_not_inherited():
    """query_pre_attn_scalar ** -0.5, passed explicitly.

    For this checkpoint that equals 1/sqrt(head_dim), so an implementation that
    simply omitted the scale would pass every numerical check here. The 27B is
    the case that separates them, and it is asserted below.
    """
    ops = attention_ops(EMBEDDINGGEMMA_CONFIG)
    assert all(op["attrs"]["softmax_scale"] == pytest.approx(0.0625) for op in ops)


def test_the_27b_scale_differs_from_the_head_dim_default():
    """Gemma3-27B: query_pre_attn_scalar 168, head_dim 128. Inheriting the
    kernel default here would be wrong by 15% and entirely silent."""
    cfg = dict(
        EMBEDDINGGEMMA_CONFIG,
        architectures=["Gemma3ForCausalLM"],
        model_type="gemma3",
        head_dim=128,
        query_pre_attn_scalar=168,
        use_bidirectional_attention=False,
    )
    ops = attention_ops(cfg, arch="Gemma3ForCausalLM")
    assert all(op["attrs"]["softmax_scale"] == pytest.approx(168.0**-0.5) for op in ops)
    assert 168.0**-0.5 != pytest.approx(128.0**-0.5)


def test_the_generative_variant_stays_causal():
    """Direction is a property of the checkpoint, not of the family."""
    cfg = dict(
        EMBEDDINGGEMMA_CONFIG,
        architectures=["Gemma3ForCausalLM"],
        model_type="gemma3",
        use_bidirectional_attention=False,
    )
    ops = attention_ops(cfg, arch="Gemma3ForCausalLM")
    assert len(ops) == 24
    assert all(op["attrs"]["causal"] is True for op in ops)


def test_the_declaration_compiles_for_both_entry_points():
    for arch, cfg in (
        ("Gemma3TextModel", EMBEDDINGGEMMA_CONFIG),
        (
            "Gemma3ForCausalLM",
            dict(EMBEDDINGGEMMA_CONFIG, architectures=["Gemma3ForCausalLM"], model_type="gemma3"),
        ),
    ):
        ir = json.loads(compile_model_for_hf(arch, cfg))
        # compile_model_for_hf reports failure in-band rather than raising, so an
        # unregistered model yields an empty IR that a careless caller would use.
        assert ir["success"], (arch, ir.get("errors"))
        assert ir["modules"], arch


# ---------------------------------------------------------------------------
# Weight mapping, against the real checkpoint
# ---------------------------------------------------------------------------


def safetensors_keys(path: Path) -> set[str]:
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return set(json.loads(handle.read(length))) - {"__metadata__"}


@pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").exists(),
    reason="google/embeddinggemma-300m not cached (gated repo)",
)
def test_mapping_covers_the_checkpoint_exactly():
    """Every declared weight resolves, and every checkpoint tensor is claimed.

    The second half is the one that catches a quietly-dropped parameter: a
    declaration missing, say, post_feedforward_layernorm still loads and still
    trains.
    """
    have = safetensors_keys(CHECKPOINT / "model.safetensors")
    mappings = _build_gemma3_mappings("layers.{layer}", "", tied_lm_head=True)

    missing, claimed = [], set()
    for slot, target in mappings.items():
        for name in list(getattr(target, "sources", ())) or [target]:
            if name.replace("{layer}", "0") not in have:
                missing.append((slot, name))
            for layer in range(EMBEDDINGGEMMA_CONFIG["num_hidden_layers"]):
                claimed.add(name.replace("{layer}", str(layer)))

    assert missing == []
    assert sorted(have - claimed) == []


@pytest.mark.skipif(
    not (CHECKPOINT / "config.json").exists(),
    reason="google/embeddinggemma-300m not cached (gated repo)",
)
def test_inlined_config_still_matches_the_published_one():
    """Guards the constant above against upstream drift."""
    published = json.loads((CHECKPOINT / "config.json").read_text())
    for key, value in EMBEDDINGGEMMA_CONFIG.items():
        if key == "architectures":
            continue
        assert published[key] == value, key


# ---------------------------------------------------------------------------
# How a serving artifact stores it
# ---------------------------------------------------------------------------


def test_every_norm_unfolds_its_unit_offset():
    """Gemma stores RMSNorm weights zero-centred and applies them as 1 + w.

    An artifact shipping the raw tensor scales by roughly zero, so every norm
    object — four per block, the two QK norms, and the final norm — must name the
    transform. This is the check that catches a norm added later without it.
    """
    from surogate.dsl.blocks.gemma3 import Gemma3FullBlock, Gemma3SlidingBlock
    from surogate.dsl.models.gemma3 import GEMMA3_MODEL_SERVE_OBJECTS

    objects = list(Gemma3SlidingBlock.schema.serve_objects) + list(GEMMA3_MODEL_SERVE_OBJECTS)
    norms = [o for o in objects if "norm" in o.name]
    assert len(norms) == 4 + 2 + 1
    for obj in norms:
        assert obj.transform == "unfold_unit_offset", obj.name
    assert Gemma3FullBlock.schema.serve_objects == Gemma3SlidingBlock.schema.serve_objects


def test_block_schemas_satisfy_the_serve_contract():
    from surogate.dsl.blocks.gemma3 import Gemma3FullBlock, Gemma3SlidingBlock

    for block in (Gemma3SlidingBlock, Gemma3FullBlock):
        assert block.schema.contract_errors() == (), block.__name__
        assert len(block.schema.serve_objects) == 11


def test_only_the_generative_variant_carries_an_output_head():
    """EmbeddingGemma's checkpoint stops at norm.weight — there is no lm_head to
    store, so an artifact holding one would describe a model that does not exist."""
    text = {o.name for o in Gemma3TextModel._serve_objects_}
    causal = {o.name for o in Gemma3CausalModel._serve_objects_}
    assert "text/output_head" in causal
    assert "text/output_head" not in text
    assert "text/embedding_head" in text
    assert "text/embedding_head" not in causal


def test_the_embedding_head_is_one_folded_matrix_behind_a_capability():
    """Both Dense modules declare Identity, so 768->3072->768 composes to [C, C].

    Capability-gated because a text-only target must not be handed an object no
    binder consumes — the engine refuses to load such an artifact outright.
    """
    (head,) = [o for o in Gemma3TextModel._serve_objects_ if o.name == "text/embedding_head"]
    assert head.shape == ("C", "C")
    assert head.transform == "compose_linear"
    assert head.capability == "embedding"
    assert head.scope == "model"
    # Everything else is text, so a text-only target exports the backbone alone.
    assert {o.capability for o in Gemma3TextModel._serve_objects_} == {"text", "embedding"}


def test_pooling_policy_is_declared():
    """The prefix is inside the mean, and Matryoshka truncates before it
    renormalises. Both are silent when wrong: still unit-ish vectors, worse
    ranking."""
    model = build_from_config(EMBEDDINGGEMMA_CONFIG)
    assert model.pooling == "mean"
    assert model.pooling_include_prompt is True
    assert model.embedding_normalize == "l2"
    assert model.embedding_dim == 768
    assert model.matryoshka_dims == (768, 512, 256, 128)
