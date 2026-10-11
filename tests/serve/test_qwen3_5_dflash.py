"""A dense target must preserve and validate its separate DFlash checkpoint."""

import json

import numpy as np
import pytest

from surogate.serve.artifact.container import Artifact
from surogate.serve.convert.common import dflash
from surogate.serve.convert.common.draft_head import compute_shortlist
from surogate.serve.convert.qwen3_5 import convert, inventory
from tests.serve.test_qwen3_5_checkpoint_config import config_for
from tests.serve.test_qwen3_5_moe_checkpoint_config import _save, dflash2_config, draft_config


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("all_local", [False, True])
def test_dense_drafter_round_trip_and_mismatch(tmp_path, monkeypatch, vision, all_local):
    config = config_for(vision=vision)
    config["mtp_num_hidden_layers"] = 0
    target = inventory.geometry_from_config(config, token_domain=500)
    draft = draft_config(target, targets=(0, 2))
    if all_local:
        draft["layer_types"][-1] = "sliding_attention"
    geometry = dflash.geometry_from_config(draft, target)
    specs, recipes = dflash.conversion_plan(geometry, target)
    model, auxiliary = tmp_path / "target", tmp_path / "drafter"
    _save(
        model, config, tuple(convert.active_recipes(mtp=False, vision=vision, geometry=target).values()), frontend=True
    )
    _save(auxiliary, draft, recipes)
    ranking = tmp_path / "counts.i64"
    np.arange(target.vocab, dtype="<i8").tofile(ranking)
    monkeypatch.setattr(
        convert.draft_head,
        "compute_shortlist",
        lambda _path, root, *, geometry: compute_shortlist(
            ranking, root, n=geometry.draft_vocab, vocab=geometry.vocab, tokenizer_vocab_size=geometry.token_domain
        ),
    )
    output = tmp_path / "paired.sinfer"
    convert.convert(model, output, device="cpu", mtp=False, vision=vision, dflash_model_dir=auxiliary)
    with Artifact(output) as artifact:
        assert artifact.geometry == convert.geometry_block(target)
        assert artifact.dflash_geometry == dflash.geometry_block(geometry)
        assert artifact.dflash_target_layers == [0, 2]
        assert bool(artifact.vision_geometry) == vision
        assert any(obj.name.startswith("vision/") for obj in artifact.objects) == vision
        assert {obj.name for obj in artifact.objects if obj.name.startswith("dflash/")} == {s.name for s in specs}

    draft["hidden_size"] += 64
    (auxiliary / "config.json").write_text(json.dumps(draft))
    refused = tmp_path / "incompatible.sinfer"
    with pytest.raises(ValueError, match="must match the target"):
        convert.convert(model, refused, device="cpu", mtp=False, vision=vision, dflash_model_dir=auxiliary)
    assert not refused.exists()


def test_second_generation_drafter_is_validated():
    config = config_for()
    config["mtp_num_hidden_layers"] = 0
    target = inventory.geometry_from_config(config, token_domain=500)
    g = dflash.geometry_from_config(dflash2_config(target), target)
    assert (g.conv_kernel, g.conv_group, g.selector_rank, g.selector_top_k, g.conv_rows) == (2, 16, 64, 16, 32)
    unstated = dflash2_config(target)
    del unstated["dflash_config"]["selector_rank"], unstated["dflash_config"]["selector_top_k"]
    with pytest.raises(ValueError, match="DFlash2DraftModel"):
        dflash.geometry_from_config(unstated, target)
    three_taps = dflash2_config(target)
    three_taps["dflash_config"]["conv_kernel_size"] = 3
    with pytest.raises(ValueError, match="two taps"):
        dflash.geometry_from_config(three_taps, target)
    wide = dflash2_config(target)
    wide["dflash_config"]["selector_top_k"] = 33
    with pytest.raises(ValueError, match="at most 32"):
        dflash.geometry_from_config(wide, target)
