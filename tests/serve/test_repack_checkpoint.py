"""Repack planning must call checkpoint builders for every supported inventory style."""
import importlib
from pathlib import Path

import pytest

from surogate.serve import ingest
from tests.serve.test_checkpoint_inventory import config_for as dense
from tests.serve.test_qwen3_5_checkpoint_config import config_for as hybrid
from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for as mixture
from tests.serve.test_gemma4_checkpoint_config import checkpoint as gemma4
from tests.serve.test_gemma3_convert import config as gemma3


@pytest.mark.parametrize("target", ["qwen3", "llama", "gemma3", "gemma4", "gemma4_e", "gemma4_moe", "qwen3_5", "qwen3_5_moe", "lfm2"])
def test_repack_uses_resolved_inventory(tmp_path, monkeypatch, target):
    if target in ("qwen3", "llama"):
        config = dense(target)
    elif target == "gemma3":
        config = gemma3()
    elif target.startswith("gemma4"):
        config = gemma4(target)
    elif target == "qwen3_5":
        config = hybrid()
    elif target == "qwen3_5_moe":
        config = mixture()
    else:
        config = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2",
                  "hidden_size": 384, "num_hidden_layers": 3, "num_attention_heads": 6,
                  "num_key_value_heads": 2, "vocab_size": 512, "conv_L_cache": 5,
                  "block_ff_dim": 640, "block_auto_adjust_ff_dim": True, "block_multiple_of": 128,
                  "full_attn_idxs": [0, 2], "max_position_embeddings": 8192,
                  "norm_eps": 1e-5, "rope_theta": 123456., "tie_word_embeddings": False}
    inventory = importlib.import_module(f"surogate.serve.convert.{target}.inventory")
    recipe = importlib.import_module(f"surogate.serve.convert.{target}.recipe")
    resolve = getattr(inventory, "geometry_from_config", None) or recipe.geometry_from_config
    geometry = resolve(config)
    monkeypatch.setattr(ingest, "_gguf_geometry", lambda *args: geometry)
    source = tmp_path / "renamed.gguf"
    source.write_bytes(b"fixture: no candidate tensor bytes read")
    plan = ingest._repack_planner(Path(__file__).resolve().parents[2], target)
    assert plan(source, {}) == {}
