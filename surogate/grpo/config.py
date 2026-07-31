"""GRPO training configuration.

GRPOTrainConfig extends SFTConfig with GRPO-specific fields (loss config,
prime-rl transport settings). All model, training, LoRA, QLoRA, precision,
and runtime fields are inherited from SFTConfig.
"""

from dataclasses import dataclass
from typing import Literal

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class GRPOLossConfig:
    """GRPO loss parameters (mirrors prime-rl's LossConfig)."""

    ipo_mask_low: float = 0.2  # The low threshold for masking tokens (probability difference)
    ipo_mask_high: float = 0.2  # The high threshold for masking tokens (probability difference)
    adv_tau: float = 1.0
    teacher_tau: float = 0.0
    kl_tau: float = 1e-3  # The tau for KL divergence
    # OPD / replay knobs used by the reference loss (surogate/grpo/loss.py);
    # off by default. These fields were referenced by loss.py and its tests
    # but never added here (pre-existing gap, fixed 2026-07-27).
    opd_tau: float = 0.0
    opd_beta: float = 1.0
    replay_tau: float = 0.0
    # Hard cap on exp(trainer - inference) in the policy-gradient seed. The
    # IPO probability-difference mask does NOT bound the ratio: a rare token
    # (both probs tiny, trainer >> inference in ratio terms) passes the mask
    # with a ratio of 1e3-1e5 and dominates the batch gradient. e^2 keeps
    # ~all honest off-policy correction (periodic-reload staleness) while
    # bounding single-token influence. Uncapped values still feed the
    # mismatch_kl metric so staleness monitoring sees the truth.
    ratio_clip: float = 7.389056  # e^2


@dataclass
class NoiseSchedulerConfig:
    """QeRL Adaptive Quantization Noise (AQN) parameters.

    Adds Gaussian noise to RMSNorm weights in the inference model before
    rollout generation.  The noise standard deviation decays geometrically
    from sigma_start to sigma_end over num_stages intervals.

    Reference: https://arxiv.org/abs/2510.11696
    """

    enabled: bool = False
    sigma_start: float = 5e-2
    sigma_end: float = 5e-4
    num_stages: int = 10


@dataclass
class GRPOTrainConfig(SFTConfig):
    """Configuration for GRPO RL training with Surogate.

    Extends SFTConfig with GRPO-specific fields. Data comes from prime-rl's
    transport layer (not from tokenized files), so the `datasets` field is
    typically left empty.
    """

    # GRPO loss
    loss: GRPOLossConfig | None = None

    # QeRL noise scheduler (Adaptive Quantization Noise)
    noise_scheduler: NoiseSchedulerConfig | None = None

    # Prime-RL integration
    transport_type: Literal["filesystem", "zmq"] = "filesystem"
    # Weight broadcast backend: "filesystem" (disk), "nccl" (GPU broadcast), "colocate" (zero-copy shared memory)
    weight_broadcast_type: Literal["filesystem", "nccl", "colocate"] = "filesystem"
    max_async_level: int = 1
    # Padding multiple for packed micro-batches.
    pad_to_multiple_of: int = 1
    # One sample per micro-batch (disables FFD packing). REQUIRED with
    # sequence_chunks > 1: chunked training attention has no packed-doc
    # isolation. Pair with pad_to_multiple_of = chunk size; all-padding
    # tail chunks are skipped natively.
    single_sample_bins: bool = False
    # Document-level attention masking for packed sequences.
    doc_masking: bool = True
    # Turn-resolved supervision diagnostics (TurnOPD arXiv:2607.05804 §4).
    # Routes the micro-step through forward_for_grpo + Python loss +
    # backward_grpo instead of the fused native step, so per-token logprobs are
    # visible to Python. Same math, same gradients (covered by
    # tests/grpo/test_native_formula.py), but slower — measurement only.
    turn_diagnostics: bool = False

    def __init__(self, cfg: DictDefault):
        # Each token's gradient is advantage * importance_ratio_clip * (softmax - 1{target}) / N_valid.
        # The advantage is often < 0.1, the clipped ratio is near 1.0 ± epsilon, and the loss mask removes prompt tokens.
        # The effective gradient per parameter ends up 10-100x smaller than SFT, so the *gradients* stay fp32.
        #
        # The master weights do not: measured GRPO runs show a BF16 master is enough.
        # Masters are allocated for every param, frozen ones included, and a master
        # dtype that differs from the param dtype also forces a separate work copy
        # (dsl_weight_manager.cpp) — so fp32 cost 6 bytes/param (fp32 master + bf16
        # work) against 2 when they alias, e.g. 12 GB vs 4 GB of arena on a 2B, most
        # of it for a frozen base a LoRA run never trains. Precision where it matters
        # comes from the adapter instead: lora_dtype defaults to "fp32", so the
        # trainable LoRA weights and their optimizer state stay full precision.
        #
        # Leaving master_dtype unset falls back to the model dtype (BF16). A full
        # fine-tune that wants the old behaviour can set master_dtype: fp32 in its yaml.
        if "gradient_dtype" not in cfg:
            cfg["gradient_dtype"] = "fp32"

        cfg["sample_packing"] = "false"
        cfg["datasets"] = []

        # GRPO packed batches are heavily masked (prompt + padding tokens are -100,
        # only the response gets loss). The compact lm_head path skips work on those
        # rows for a meaningful step-time win — measured ~13% on Qwen3-0.6B BF16
        # LoRA, ~14-21% on fp8-hybrid (the helpers route the gathered slices
        # through the FP8 cache lookup + on-the-fly quantization).
        if "lmhead_drop_ignored_rows" not in cfg:
            cfg["lmhead_drop_ignored_rows"] = True

        super().__init__(cfg)

        # Parse nested loss config
        loss_dict = cfg.get("loss", {})
        if isinstance(loss_dict, dict) and loss_dict:
            self.loss = GRPOLossConfig(**loss_dict)
        elif isinstance(loss_dict, GRPOLossConfig):
            self.loss = loss_dict
        else:
            self.loss = GRPOLossConfig()

        # Parse nested noise scheduler config
        ns_dict = cfg.get("noise_scheduler", {})
        if isinstance(ns_dict, dict) and ns_dict:
            self.noise_scheduler = NoiseSchedulerConfig(**ns_dict)
        elif isinstance(ns_dict, NoiseSchedulerConfig):
            self.noise_scheduler = ns_dict
        else:
            self.noise_scheduler = NoiseSchedulerConfig()

        self.transport_type = cfg.get("transport_type", self.transport_type)
        self.single_sample_bins = bool(cfg.get("single_sample_bins", self.single_sample_bins))
        self.weight_broadcast_type = cfg.get("weight_broadcast_type", self.weight_broadcast_type)
        self.max_async_level = cfg.get("max_async_level", self.max_async_level)
        self.pad_to_multiple_of = cfg.get("pad_to_multiple_of", self.pad_to_multiple_of)
        self.doc_masking = cfg.get("doc_masking", self.doc_masking)
        self.turn_diagnostics = cfg.get("turn_diagnostics", self.turn_diagnostics)

        # Initialize inherited config: model_dir, runtime_config, lora_config, etc.
        # In the SFT path this is called by TokenizeDatasets.__init__(), but GRPO
        # bypasses tokenization so we call it here directly.
        self.__post_init__()
        # Propagate GRPO-specific doc masking to runtime options.
        # This controls document-level attention masking in the C++ engine
        # while still allowing position_ids (RoPE resets) to be passed.
        self.runtime_config.doc_masking = self.doc_masking

        # Disable CUDA graphs for GRPO for now. The native GRPO step removes the
        # Python logprob round trip, but graph capture needs separate wiring.
        if self.use_cuda_graphs:
            self.use_cuda_graphs = False
            self.runtime_config.use_cuda_graphs = False
