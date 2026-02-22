"""GRPO training configuration.

GRPOTrainConfig extends SFTConfig with GRPO-specific fields (loss config,
prime-rl transport settings). All model, training, LoRA, QLoRA, precision,
and runtime fields are inherited from SFTConfig.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class GRPOLossConfig:
    """GRPO loss parameters (mirrors prime-rl's LossConfig)."""

    ratio_type: Literal["token", "sequence"] = "token"
    kl_tau: float = 0.0
    adv_tau: float = 1.0
    teacher_tau: float = 0.0
    token_mask_low: float = 0.125
    token_mask_high: float = 8.0
    geo_mask_low: float = 0.1
    geo_mask_high: float = 10.0
    sequence_mask_low: float = 0.0
    sequence_mask_high: float = 100.0
    sequence_clip_high: float = 10.0


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
    loss: Optional[GRPOLossConfig] = None

    # QeRL noise scheduler (Adaptive Quantization Noise)
    noise_scheduler: Optional[NoiseSchedulerConfig] = None

    # Prime-RL integration
    transport_type: Literal["filesystem", "zmq"] = "filesystem"
    # Weight broadcast backend: "filesystem" (disk), "nccl" (GPU broadcast), "colocate" (zero-copy shared memory)
    weight_broadcast_type: Literal["filesystem", "nccl", "colocate"] = "filesystem"
    max_async_level: int = 1
    # Padding multiple for packed micro-batches.
    pad_to_multiple_of: int = 1
    # Document-level attention masking for packed sequences.
    doc_masking: bool = True

    def __init__(self, cfg: DictDefault):
        # Each token's gradient is advantage * importance_ratio_clip * (softmax - 1{target}) / N_valid. 
        # The advantage is often < 0.1, the clipped ratio is near 1.0 ± epsilon, and the loss mask removes prompt tokens. 
        # The effective gradient per parameter ends up 10-100x smaller than SFT. 
        # At that scale, BF16 rounding kills the signal.
        if "master_dtype" not in cfg:
            cfg["master_dtype"] = "fp32"
        if "gradient_dtype" not in cfg:
            cfg["gradient_dtype"] = "fp32"

        cfg["sample_packing"] = "false"
        cfg["datasets"] = []
        
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
        self.weight_broadcast_type = cfg.get("weight_broadcast_type", self.weight_broadcast_type)
        self.max_async_level = cfg.get("max_async_level", self.max_async_level)
        self.pad_to_multiple_of = cfg.get("pad_to_multiple_of", self.pad_to_multiple_of)
        self.doc_masking = cfg.get("doc_masking", self.doc_masking)

        # Initialize inherited config: model_dir, runtime_config, lora_config, etc.
        # In the SFT path this is called by TokenizeDatasets.__init__(), but GRPO
        # bypasses tokenization so we call it here directly.
        self.__post_init__()
        # Propagate GRPO-specific doc masking to runtime options.
        # This controls document-level attention masking in the C++ engine
        # while still allowing position_ids (RoPE resets) to be passed.
        self.runtime_config.doc_masking = self.doc_masking

        # Disable CUDA graphs for GRPO — the compute_logprobs → step_with_custom_loss
        # pattern means save buffers aren't preallocated in the right order for capture.
        if self.use_cuda_graphs:
            self.use_cuda_graphs = False
            self.runtime_config.use_cuda_graphs = False
