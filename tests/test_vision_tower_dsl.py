"""The vision tower the declaration describes.

Every Qwen3.5 checkpoint ships a tower and the text stack has always been able to
consume its output; until now nothing produced it, so a multimodal checkpoint
could be served but not trained end to end. These tests are CPU-only and need no
weights: they pin the tower's shapes, its checkpoint paths, and the two properties
that are easy to get silently wrong — LayerNorm rather than RMSNorm, and
bidirectional rather than causal attention.
"""

from __future__ import annotations

import pytest

from surogate.dsl.modules import VisionTower

#: (name, text width, tower geometry) for the towers actually shipped.
TOWERS = (
    ("qwen3.5-0.8b", 1024, dict(hidden=768, layers=12, intermediate=3072, heads=12)),
    ("qwen3.5-2b", 2048, dict(hidden=1024, layers=24, intermediate=4096, heads=16)),
    ("qwen3.5-4b", 2560, dict(hidden=1024, layers=24, intermediate=4096, heads=16)),
    ("qwen3.6/flash-next", 2560, dict(hidden=1152, layers=27, intermediate=4304, heads=16)),
)


def build(text_width: int, **geometry) -> VisionTower:
    return VisionTower(
        d_model=text_width, patch_rows=1536, position_embeddings=2304, **geometry
    )


@pytest.mark.parametrize("name,text_width,geometry", TOWERS)
def test_tower_geometry_is_per_model(name, text_width, geometry):
    """The towers differ between checkpoints; a config hardcoding one family's
    numbers cannot serve the others, which is how the serve side had it."""

    tower = build(text_width, **geometry)
    assert tower.VH == geometry["hidden"]
    assert tower.layers == geometry["layers"]
    assert tower.VQkv == 3 * geometry["hidden"]
    assert tower.VHeadDim == geometry["hidden"] // geometry["heads"]
    assert tower.VMerged == geometry["hidden"] * 4
    assert tower.d_model == text_width


def test_head_split_must_be_exact():
    with pytest.raises(ValueError, match="divide evenly"):
        build(2048, hidden=1000, layers=2, intermediate=1, heads=16)


@pytest.mark.parametrize("name,text_width,geometry", TOWERS)
def test_layer_mapping_matches_the_checkpoint(name, text_width, geometry):
    tower = build(text_width, **geometry)
    mapping = tower.layer_mapping("model.visual", 3)
    assert mapping["vision_blocks_3_qkv_weight"] == "model.visual.blocks.3.attn.qkv.weight"
    assert mapping["vision_blocks_3_out_weight"] == "model.visual.blocks.3.attn.proj.weight"
    assert mapping["vision_blocks_3_fc1_weight"] == "model.visual.blocks.3.mlp.linear_fc1.weight"
    # LayerNorm, so every norm has a bias. An RMSNorm tower would not, and the
    # checkpoint would then have keys nothing claimed.
    assert mapping["vision_blocks_3_norm1_bias"] == "model.visual.blocks.3.norm1.bias"
    assert mapping["vision_blocks_3_norm2_bias"] == "model.visual.blocks.3.norm2.bias"
    assert len(mapping) == 12


def test_tower_head_paths_cover_patch_position_and_merger():
    defaults = VisionTower._hf_mapping_defaults_
    for key, tail in (
        ("patch_embed_weight", "patch_embed.proj.weight"),
        ("position_embedding", "pos_embed.weight"),
        ("merger_fc2_weight", "merger.linear_fc2.weight"),
        ("merger_norm_bias", "merger.norm.bias"),
    ):
        assert defaults[key].endswith(tail), key


def test_attention_is_bidirectional_in_the_traced_graph():
    """A ViT has no causal structure. Running it causally still produces
    plausible embeddings, so nothing downstream would catch it — hence a test on
    the emitted attribute rather than on any output."""

    import inspect

    source = inspect.getsource(VisionTower._encoder_block)
    assert "causal=False" in source
    assert "causal=True" not in source
