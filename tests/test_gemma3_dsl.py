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

* **Schedule.** Gemma 3 states local-vs-global as a period *or* as a list, and
  which one depends on the export: embeddinggemma-300m publishes
  ``_sliding_window_pattern`` alone, gemma-3-270m-it publishes both, and newer
  exports publish ``layer_types`` alone. Reading a missing period as zero is how
  every layer became windowed -- global ones included, which then also rotated at
  the local base.
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
        assert len(block.schema.serve_objects) == 13


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


# ---------------------------------------------------------------------------
# What the GGUF conversion source does and does not carry
# ---------------------------------------------------------------------------

GGUF = Path("models/embeddinggemma-300M-Q8_0.gguf")

#: Present in the GGUF KV, so the converter may read them from the file.
GGUF_SUPPLIES = {
    "gemma-embedding.block_count": 24,
    "gemma-embedding.embedding_length": 768,
    "gemma-embedding.feed_forward_length": 1152,
    "gemma-embedding.attention.head_count": 3,
    "gemma-embedding.attention.head_count_kv": 1,
    "gemma-embedding.attention.key_length": 256,
    "gemma-embedding.attention.sliding_window": 512,
    "gemma-embedding.rope.freq_base": 1000000.0,
    "gemma-embedding.rope.freq_base_swa": 10000.0,
    "gemma-embedding.pooling_type": 1,  # MEAN
    "gemma-embedding.dense_2_feat_out": 3072,
    "gemma-embedding.dense_3_feat_out": 768,
}

#: Absent from the GGUF KV. llama.cpp supplies each from the architecture
#: identity (models/gemma-embedding.cpp); here they come from the declaration.
#: All three are silent when wrong, which is what makes the list worth pinning.
GGUF_OMITS = (
    "gemma-embedding.attention.query_pre_attn_scalar",
    "gemma-embedding.use_bidirectional_attention",
    "gemma-embedding.attention.sliding_window_pattern",
)


@pytest.mark.skipif(not GGUF.exists(), reason="embeddinggemma-300M-Q8_0.gguf not downloaded")
def test_gguf_carries_the_geometry_but_not_the_three_that_matter():
    from gguf import GGUFReader

    reader = GGUFReader(str(GGUF))
    fields = reader.fields
    assert fields["general.architecture"].contents() == "gemma-embedding"

    for key, expected in GGUF_SUPPLIES.items():
        assert key in fields, key
        assert fields[key].contents() == pytest.approx(expected), key

    for key in GGUF_OMITS:
        assert key not in fields, f"{key} is now in the GGUF — read it instead of declaring it"

    # And the declaration supplies exactly those three.
    model = build_from_config(EMBEDDINGGEMMA_CONFIG)
    assert model.query_pre_attn_scalar == 256
    assert model.causal is False
    assert (model.n_sliding_blocks, model.n_full_blocks) == (20, 4)


@pytest.mark.skipif(not GGUF.exists(), reason="embeddinggemma-300M-Q8_0.gguf not downloaded")
def test_gguf_ships_the_head_and_the_norms_pre_unfolded():
    """316 tensors: the 314 of the safetensors checkpoint plus dense_2/dense_3.

    The norms arrive folded as ``1 + w`` where the HF checkpoint stores ``w``.
    The runtime re-applies the offset itself (``unit_offset=true``), so the
    artifact must hold ``w`` and the converter subtracts from *this* source where
    it would pass safetensors through. Skipping the subtraction gives a gain of
    ``2 + w``, raises nothing, and inverts retrieval at plausible similarities.
    """
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(GGUF))
    tensors = {t.name: t for t in reader.tensors}
    assert len(tensors) == 316
    assert {"dense_2.weight", "dense_3.weight"} <= set(tensors)

    have = safetensors_keys(CHECKPOINT / "model.safetensors") if (
        CHECKPOINT / "model.safetensors"
    ).exists() else None
    if have is None:
        pytest.skip("safetensors checkpoint not cached; nothing to compare the offset against")

    import numpy as np
    from safetensors.torch import load_file

    hf = load_file(CHECKPOINT / "model.safetensors")
    for gguf_name, hf_name in (
        ("blk.0.attn_norm.weight", "layers.0.input_layernorm.weight"),
        ("blk.0.attn_q_norm.weight", "layers.0.self_attn.q_norm.weight"),
        ("output_norm.weight", "norm.weight"),
    ):
        g = dequantize(tensors[gguf_name].data, tensors[gguf_name].tensor_type).astype("float64")
        h = hf[hf_name].float().numpy().astype("float64")
        unfolded = g - 1.0
        cos = float(unfolded @ h / (np.linalg.norm(unfolded) * np.linalg.norm(h)))
        assert cos == pytest.approx(1.0, abs=1e-6), (gguf_name, cos)


# ---------------------------------------------------------------------------
# The converter's source mapping, against the real GGUF
# ---------------------------------------------------------------------------


def generated_inventory():
    import sys

    sys.path.insert(0, str(Path("surogate/serve/tools/generate").resolve()))
    emit_inventory = pytest.importorskip("emit_inventory")
    return emit_inventory.inventory_for(
        "Gemma3TextModel", EMBEDDINGGEMMA_CONFIG, capabilities={"text", "embedding"}
    )


def test_inventory_derives_from_the_declaration():
    """267 = 1 embedding + 24 layers x 11 + final norm + head.

    Nothing here is written down twice: the converter has no inventory module,
    it asks the declaration.
    """
    inventory = generated_inventory()
    assert len(inventory) == 1 + 24 * 13 + 1 + 1 == 315

    by_name = {obj["name"]: obj for obj in inventory}
    assert by_name["text/token_embedding"]["shape"] == (262144, 768)
    assert by_name["text/layers/0/attention/query"]["shape"] == (768, 768)
    assert by_name["text/layers/0/attention/key"]["shape"] == (256, 768)
    assert by_name["text/layers/0/attention/value"]["shape"] == (256, 768)
    assert by_name["text/final_norm"]["shape"] == (768,)
    # The two Dense modules composed into one square matrix.
    assert by_name["text/embedding_head"]["shape"] == (768, 768)
    assert by_name["text/embedding_head"]["transform"] == "compose_linear"


@pytest.mark.skipif(not GGUF.exists(), reason="embeddinggemma-300M-Q8_0.gguf not downloaded")
def test_every_object_maps_onto_gguf_tensors_of_the_right_shape():
    """The mapping is only worth having if it closes over the real file.

    Checks both directions: every artifact object resolves to tensors that exist
    and whose rows sum to the declared shape, and every tensor in the GGUF is
    consumed by some object. A source the recipe forgets is a weight silently
    left behind.
    """
    from gguf import GGUFReader

    from surogate.serve.tools.convert.gemma_embedding import gguf_names, source_for

    reader = GGUFReader(str(GGUF))
    # gguf-py reports ne as [k, n]; the logical shape is the reverse.
    shapes = {t.name: tuple(int(d) for d in reversed(t.shape)) for t in reader.tensors}

    consumed: set[str] = set()
    for obj in generated_inventory():
        names = gguf_names(obj["name"])
        source = source_for(obj["name"])
        for name in names:
            assert name in shapes, f"{obj['name']} -> missing {name}"
            consumed.add(name)

        parts = [shapes[n] for n in names]
        if source.op == "compose_linear":
            # (n, k) @ (k, m) -> (n, m); here [768,3072] @ [3072,768].
            (n, k), (k2, m) = parts
            assert k == k2, obj["name"]
            assert (n, m) == obj["shape"], obj["name"]
        else:
            assert parts[0] == obj["shape"], (obj["name"], parts[0], obj["shape"])

    assert sorted(set(shapes) - consumed) == [], "GGUF tensors no object consumes"


@pytest.mark.skipif(not GGUF.exists(), reason="embeddinggemma-300M-Q8_0.gguf not downloaded")
def test_only_the_norms_and_the_head_leave_the_quantised_path():
    """Q8_0 is W8G32_F16S, so anything built by row algebra repacks bit-exactly.

    The norms are F32 in the GGUF and must lose their folded one; the head has to
    dequantize because a matrix product mixes k. That leaves seven matrices per
    layer — Q, K, V, the attention output and the three MLP projections —
    plus the embedding table, all moving across untouched. 169 of 315 objects,
    but very nearly all of the bytes.
    """
    from surogate.serve.tools.convert.gemma_embedding import source_for

    repackable = [o["name"] for o in generated_inventory() if source_for(o["name"]).repackable]
    assert len(repackable) == 1 + 24 * 7 == 169
    assert "text/embedding_head" not in repackable
    assert not any(name.endswith("_norm") for name in repackable)


# ---------------------------------------------------------------------------
# What the serve target reads off the declaration
# ---------------------------------------------------------------------------

#: google/gemma-3-270m-it, the checkpoint `targets/gemma3` is built for. Inlined
#: for the same reason the one above is: it is the contract under test.
GEMMA3_270M_CONFIG = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3_text",
    "_sliding_window_pattern": 6,
    "head_dim": 256,
    "hidden_size": 640,
    "intermediate_size": 2048,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"]
    + ["sliding_attention"] * 5 + ["full_attention"]
    + ["sliding_attention"] * 5 + ["full_attention"],
    "max_position_embeddings": 32768,
    "num_attention_heads": 4,
    "num_hidden_layers": 18,
    "num_key_value_heads": 1,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000.0,
    "rope_theta": 1000000.0,
    "sliding_window": 512,
    "vocab_size": 262144,
}


GEMMA3_270M_CHECKPOINT = Path(
    "~/.cache/huggingface/hub/models--google--gemma-3-270m-it/snapshots"
).expanduser()


@pytest.mark.skipif(
    not sorted(GEMMA3_270M_CHECKPOINT.glob("*/config.json")),
    reason="google/gemma-3-270m-it not cached",
)
def test_the_inlined_270m_config_still_matches_the_published_one():
    """Guards the constant above, the way the EmbeddingGemma one is guarded: the
    serve target is generated from the published file, so an upstream edit must
    fail here rather than quietly redefine what the tests assert."""

    published = json.loads(
        sorted(GEMMA3_270M_CHECKPOINT.glob("*/config.json"))[0].read_text()
    )
    for key, value in GEMMA3_270M_CONFIG.items():
        assert published[key] == value, key


def serve_spec(cfg: dict, *, arch: str = "Gemma3ForCausalLM", name: str = "probe"):
    """The serve contract the generator reads out of the declaration."""

    import sys

    sys.path.insert(0, str(Path("surogate/serve/tools/generate").resolve()))
    from_dsl = pytest.importorskip("from_dsl")
    return from_dsl.from_dsl(arch, cfg, name=name)


def test_the_schedule_survives_a_config_that_publishes_only_layer_types():
    """`_sliding_window_pattern` is not guaranteed, and its absence used to read
    as period 0 -- which the header then took as "every layer windowed", global
    layers included, on the local rope base.

    The list here is deliberately not periodic: no period can express it, which
    is the point of carrying the resolved schedule rather than a rule.
    """

    cfg = dict(GEMMA3_270M_CONFIG)
    cfg.pop("_sliding_window_pattern")
    cfg["layer_types"] = ["full_attention", "full_attention"] + ["sliding_attention"] * 16

    spec = serve_spec(cfg)
    assert spec.sliding_window == 512
    assert spec.sliding_window_schedule == (False, False) + (True,) * 16
    assert len(spec.sliding_window_schedule) == spec.layers


def test_the_published_270m_schedule_is_every_sixth_layer_global():
    spec = serve_spec(GEMMA3_270M_CONFIG)
    assert [i for i, w in enumerate(spec.sliding_window_schedule) if not w] == [5, 11, 17]


def test_a_window_with_no_schedule_at_all_is_refused():
    """Neither a list nor a period. Every way of inventing one is silent, so the
    declaration raises instead of picking."""

    cfg = dict(GEMMA3_270M_CONFIG)
    cfg.pop("_sliding_window_pattern")
    cfg.pop("layer_types")
    with pytest.raises(ValueError, match="layer schedule is undetermined"):
        serve_spec(cfg)


def test_the_generator_refuses_a_window_whose_block_types_are_unnamed(monkeypatch):
    """The generator asks the declaration which of its block types are windowed;
    a model that declares a window and does not answer must be refused rather
    than defaulted, because the default that used to apply was "all of them"."""

    monkeypatch.setattr(Gemma3CausalModel, "_serve_windowed_blocks_", None)
    with pytest.raises(ValueError, match="no per-layer window schedule"):
        serve_spec(GEMMA3_270M_CONFIG)


def test_the_serve_attention_scale_is_the_declared_scalar():
    """The 270M hides this -- scalar 256 is its head dim, so both give 0.0625 --
    and the 27B is where the two part company."""

    assert serve_spec(GEMMA3_270M_CONFIG).attention_scale == pytest.approx(0.0625)

    wide = dict(GEMMA3_270M_CONFIG, head_dim=128, query_pre_attn_scalar=168)
    spec = serve_spec(wide)
    assert spec.query_pre_attn_scalar == 168
    assert spec.attention_scale == pytest.approx(0.07715167498104596)
    assert spec.attention_scale != pytest.approx(128.0**-0.5)


def test_the_embedding_scale_is_rounded_to_bf16_as_hf_rounds_it():
    """`Gemma3TextScaledWordEmbedding.forward` multiplies by
    `self.embed_scale.to(self.weight.dtype)`, so on a bf16 checkpoint the factor
    is the bf16 rounding of sqrt(hidden): 25.25, not 25.298221. The declaration,
    the training module and the serve contract must all carry the same number --
    the serving engine rounds again, so a mismatch here is invisible there and
    visible in training."""

    model = Gemma3CausalModel(
        vocab_size=GEMMA3_270M_CONFIG["vocab_size"],
        d_model=640,
        n_layers=18,
        num_query_heads=4,
        num_kv_heads=1,
        d_ff=2048,
        max_seq=32768,
        head_size=256,
    )
    assert model.embedding_scale == 25.25
    assert model.embedding.embed_scale == model.embedding_scale
    assert model.embedding_scale != pytest.approx(640.0**0.5)
    assert serve_spec(GEMMA3_270M_CONFIG).embedding_scale == 25.25
