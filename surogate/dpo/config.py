"""Offline DPO training configuration.

DPOTrainConfig extends SFTConfig with the DPO loss block. Model, LoRA, QLoRA,
precision, optimizer, and runtime fields are inherited from SFTConfig. Unlike
GRPO, DPO keeps `datasets` (it reads static preference pairs of the form
{prompt, chosen, rejected}); see surogate/dpo/data.py.
"""

from dataclasses import dataclass, fields
from typing import Literal

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class DPOLossConfig:
    """Sigmoid-DPO loss parameters."""

    type: Literal["dpo"] = "dpo"
    # KL temperature. The implicit reward is beta * (logpi_theta - logpi_ref).
    dpo_beta: float = 0.1
    # Divide each sequence's response-token logprob sum by its response length
    # (SimPO-style). Off for minimal pairs where chosen/rejected lengths match.
    length_norm: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "DPOLossConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown DPO loss config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class DPOTrainConfig(SFTConfig):
    """Configuration for offline DPO training with Surogate."""

    loss: DPOLossConfig | None = None

    def __init__(self, cfg: DictDefault):
        # The DPO per-token gradient is beta * sigmoid(-margin) * (softmax - 1{target}),
        # typically far smaller than an SFT gradient — keep fp32 master/grad so BF16
        # rounding does not swamp the preference signal (mirrors GRPO).
        if "master_dtype" not in cfg:
            cfg["master_dtype"] = "fp32"
        if "gradient_dtype" not in cfg:
            cfg["gradient_dtype"] = "fp32"

        # The DPO trainer tokenizes preference pairs itself (surogate/dpo/data.py)
        # and packs each (chosen, rejected) pair atomically, so the generic SFT
        # sample-packing path is bypassed.
        cfg["sample_packing"] = "false"

        # Pairs are heavily masked (prompt + padding tokens carry no loss); the
        # compact lm_head path skips work on those rows.
        if "lmhead_drop_ignored_rows" not in cfg:
            cfg["lmhead_drop_ignored_rows"] = True

        super().__init__(cfg)

        loss_cfg = cfg.get("loss", {})
        if isinstance(loss_cfg, DPOLossConfig):
            self.loss = loss_cfg
        elif isinstance(loss_cfg, dict) and loss_cfg:
            self.loss = DPOLossConfig.from_dict(dict(loss_cfg))
        else:
            self.loss = DPOLossConfig()

        # Initialize inherited config (model_dir, runtime_config, lora_config, ...).
        # The SFT path runs this from TokenizeDatasets.__init__(); the DPO trainer
        # bypasses that pipeline, so call it here directly (as GRPO does).
        self.__post_init__()

        # The native DPO step is not graph-captured (it mirrors the native GRPO
        # step, which also disables CUDA graphs).
        if self.use_cuda_graphs:
            self.use_cuda_graphs = False
            self.runtime_config.use_cuda_graphs = False
