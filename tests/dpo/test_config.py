"""DPOTrainConfig parses the nested DPO loss block and inherits SFT fields."""

import yaml

from surogate.core.config.loader import load_config
from surogate.dpo.config import DPOLossConfig, DPOTrainConfig

MODEL = "ro/train/out_opmix_wordform_s50_a010_eval"


def _write(tmp_path, loss):
    p = tmp_path / "dpo.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "recipe": "bf16",
                "gpus": 1,
                "loss": loss,
                "datasets": [{"path": "x.jsonl", "type": "preference"}],
            }
        )
    )
    return str(p)


def test_parses_dpo_loss_block(tmp_path):
    cfg = load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "dpo_beta": 0.2, "length_norm": True}))
    assert isinstance(cfg.loss, DPOLossConfig)
    assert cfg.loss.type == "dpo"
    assert cfg.loss.dpo_beta == 0.2
    assert cfg.loss.length_norm is True


def test_defaults_when_loss_absent(tmp_path):
    cfg = load_config(DPOTrainConfig, _write(tmp_path, None))
    assert isinstance(cfg.loss, DPOLossConfig)
    assert cfg.loss.dpo_beta == 0.1
    assert cfg.loss.length_norm is False


def test_rejects_unknown_loss_key(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown DPO loss config"):
        load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "bogus": 1}))


def test_inherits_sft_fields(tmp_path):
    cfg = load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "dpo_beta": 0.1}))
    assert cfg.lora is True
    assert cfg.lora_rank == 16
    # DPO bypasses SFT packing and disables CUDA graphs.
    assert cfg.use_cuda_graphs is False
