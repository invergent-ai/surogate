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
    # Confine the loss to the tokens where chosen and rejected differ (common
    # prefix/suffix excluded). For minimal word-substitution pairs the gradient
    # then cannot shift style/length/language — only the substituted form.
    span_mask: bool = False
    # Optimize the chosen/rejected likelihood gap directly instead of its
    # change from the frozen start policy.
    reference_free: bool = False
    # Required beta-scaled likelihood margin for reference-free training
    # (SimPO-style gamma).
    target_margin: float = 0.0

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
        # far smaller than an SFT gradient, so keep the trainable parameters in fp32 to
        # avoid BF16 rounding swamping the signal. This is a LoRA-only run (the base is
        # frozen), so precision must come from the *adapter*, not a full-model master:
        #   - lora_dtype defaults to "fp32" -> the trainable LoRA weights are fp32.
        #   - gradient_dtype=fp32 -> the (tiny) LoRA gradients are fp32.
        # Do NOT force master_dtype=fp32: it would allocate an fp32 copy of the entire
        # frozen base (~2x the model, e.g. +9 GB for a 2B) in the persistent arena for
        # no benefit, and that waste is what tipped fp8-hybrid / multi-GPU / large-batch
        # runs into spurious arena OOMs at trainer construction.
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

        if self.loss.dpo_beta <= 0:
            raise ValueError("loss.dpo_beta must be positive")
        if self.loss.target_margin < 0:
            raise ValueError("loss.target_margin must be non-negative")
        if self.loss.target_margin and not self.loss.reference_free:
            raise ValueError("loss.target_margin requires loss.reference_free=true")

        # Initialize inherited config (model_dir, runtime_config, lora_config, ...).
        # The SFT path runs this from TokenizeDatasets.__init__(); the DPO trainer
        # bypasses that pipeline, so call it here directly (as GRPO does).
        self.__post_init__()

        # The native DPO step is not graph-captured (it mirrors the native GRPO
        # step, which also disables CUDA graphs).
        if self.use_cuda_graphs:
            self.use_cuda_graphs = False
            self.runtime_config.use_cuda_graphs = False
