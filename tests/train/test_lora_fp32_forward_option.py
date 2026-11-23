"""`lora_fp32_forward` reaches the LoRA kernels on the fine-tuning path too, not only under GRPO."""

import os

import pytest

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault


def config(**kw):
    return SFTConfig(DictDefault({"model": "unused", "datasets": [], **kw}))


def test_fine_tuning_config_reads_it():
    assert config(lora_fp32_forward=True).lora_fp32_forward is True
    assert config().lora_fp32_forward is False


def test_it_needs_fp32_masters():
    with pytest.raises(ValueError, match="lora_dtype: fp32"):
        config(lora_fp32_forward=True, lora_dtype="bf16")


def test_the_wrapper_turns_it_on(monkeypatch):
    pytest.importorskip("surogate._surogate")
    from surogate.train.trainer import SurogateTrainerWrapper

    class Stop(Exception):
        pass

    class Config:
        lora_fp32_forward = True
        train_vision = False
        is_multimodal = False
        @property
        def model_dir(self):
            raise Stop

    monkeypatch.delenv("SUROGATE_LORA_FP32_FORWARD", raising=False)
    # __init__ reads the option before anything heavy; the first thing after it stops the construction here.
    with pytest.raises(Stop):
        SurogateTrainerWrapper(Config(), [], None)
    assert os.environ["SUROGATE_LORA_FP32_FORWARD"] == "1"
