"""DPOTrainConfig parses the nested DPO loss block and inherits SFT fields."""

from types import SimpleNamespace

import pytest
import yaml

from surogate.core.config.loader import load_config
from surogate.dpo.config import DPOLossConfig, DPOTrainConfig

MODEL = "local-test-model"


@pytest.fixture(autouse=True)
def _skip_model_loading(monkeypatch):
    """Config parsing tests must not download or require a model checkout."""

    def fake_post_init(config):
        config.runtime_config = SimpleNamespace(use_cuda_graphs=config.use_cuda_graphs)

    monkeypatch.setattr(DPOTrainConfig, "__post_init__", fake_post_init)


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
    cfg = load_config(
        DPOTrainConfig,
        _write(
            tmp_path,
            {
                "type": "dpo",
                "dpo_beta": 0.2,
                "length_norm": True,
                "reference_free": True,
                "target_margin": 1.0,
            },
        ),
    )
    assert isinstance(cfg.loss, DPOLossConfig)
    assert cfg.loss.type == "dpo"
    assert cfg.loss.dpo_beta == 0.2
    assert cfg.loss.length_norm is True
    assert cfg.loss.reference_free is True
    assert cfg.loss.target_margin == 1.0


def test_defaults_when_loss_absent(tmp_path):
    cfg = load_config(DPOTrainConfig, _write(tmp_path, None))
    assert isinstance(cfg.loss, DPOLossConfig)
    assert cfg.loss.dpo_beta == 0.1
    assert cfg.loss.length_norm is False


def test_rejects_unknown_loss_key(tmp_path):
    with pytest.raises(ValueError, match="unknown DPO loss config"):
        load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "bogus": 1}))


def test_target_margin_requires_reference_free(tmp_path):
    with pytest.raises(ValueError, match="target_margin requires"):
        load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "target_margin": 1.0}))


def test_dpo_beta_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="dpo_beta must be positive"):
        load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "dpo_beta": 0.0}))


def test_inherits_sft_fields(tmp_path):
    cfg = load_config(DPOTrainConfig, _write(tmp_path, {"type": "dpo", "dpo_beta": 0.1}))
    assert cfg.lora is True
    assert cfg.lora_rank == 16
    # DPO bypasses SFT packing and disables CUDA graphs.
    assert cfg.use_cuda_graphs is False


def test_dataset_type_defaults_to_preference(tmp_path):
    from surogate.core.config.dataset_config import PreferenceDatasetConfig

    p = tmp_path / "dpo.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "lora": True,
                "recipe": "bf16",
                "gpus": 1,
                "datasets": [{"path": "x.jsonl"}],
            }
        )
    )
    cfg = load_config(DPOTrainConfig, str(p))
    assert isinstance(cfg.datasets[0], PreferenceDatasetConfig)
    assert cfg.datasets[0].type == "preference"


def test_rejects_non_preference_dataset_type(tmp_path):
    p = tmp_path / "dpo.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "model": MODEL,
                "lora": True,
                "recipe": "bf16",
                "gpus": 1,
                "datasets": [{"path": "x.jsonl", "type": "conversation"}],
            }
        )
    )
    with pytest.raises(ValueError, match="must be 'type: preference'"):
        load_config(DPOTrainConfig, str(p))
