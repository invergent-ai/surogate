"""Main GRPO trainer

Training loop:
    1. Wait for batch from orchestrator (packer on master, transport to all ranks)
    2. For each micro-batch (packed sequence with multiple samples):
       a. step_grpo_native() on packed batch -> forward + native GRPO dloss + backward
    3. update_with_config() -> optimizer step
    4. Broadcast updated weights to inference engine

Document-level attention masking (Flash Attention varlen) can be enabled to
prevent cross-sample attention in packed sequences.
"""

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from surogate import _surogate
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.data import GRPODataLoader
from surogate.grpo.runs import get_multi_run_manager
from surogate.grpo.weight_broadcast import SurogateWeightBroadcast
from surogate.train.lr_schedule import LRSchedule
from surogate.train.metrics_writer import MetricsWriter
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


class NativeGRPOUpdateRejected(RuntimeError):
    """A replay-anchored native update failed its pre-mutation safety gate."""


MAX_REPLAY_ANCHORED_MISMATCH_KL = 0.10


_NATIVE_FINITE_UPDATE_METRICS = (
    "policy_loss",
    "mismatch_kl",
    "masked_mismatch_kl",
    "unmasked_mismatch_kl",
    "teacher_kl",
    "opd_loss",
    "replay_loss",
)


def _finite_native_metric(metrics: dict[str, float], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise NativeGRPOUpdateRejected(f"native GRPO update rejected: {name} is missing or non-finite")
    return float(value)


def _validate_replay_anchored_native_update(
    *,
    metrics: dict[str, float],
    grad_norms: list[float],
    valid_token_count: int,
    expected_loss_scale: int,
) -> None:
    """Reject an unsafe replay-bearing update before optimizer state or weights mutate."""

    for name in _NATIVE_FINITE_UPDATE_METRICS:
        _finite_native_metric(metrics, name)

    replay_tokens = _finite_native_metric(metrics, "replay_tokens")
    replay_weight_sum = _finite_native_metric(metrics, "replay_weight_sum")
    total_tokens = _finite_native_metric(metrics, "total_tokens")
    policy_sample_count = _finite_native_metric(metrics, "policy_sample_count")
    mismatch_kl = float(metrics["mismatch_kl"])
    replay_loss = float(metrics["replay_loss"])
    if mismatch_kl > MAX_REPLAY_ANCHORED_MISMATCH_KL:
        raise NativeGRPOUpdateRejected(
            "native GRPO update rejected: "
            f"mismatch KL {mismatch_kl:.6g} exceeds {MAX_REPLAY_ANCHORED_MISMATCH_KL:.6g}"
        )
    if replay_tokens <= 0.0 or replay_weight_sum <= 0.0 or replay_loss <= 0.0:
        raise NativeGRPOUpdateRejected(
            "native GRPO update rejected: replay activity, weight sum, and loss must all be positive"
        )
    if policy_sample_count <= 0.0:
        raise NativeGRPOUpdateRejected(
            "native GRPO update rejected: behavior-likelihood policy sample count must be positive"
        )
    if total_tokens != float(expected_loss_scale):
        raise NativeGRPOUpdateRejected(
            "native GRPO update rejected: "
            f"native selected-token count {total_tokens!r} differs from exact loss scale {expected_loss_scale}"
        )

    if not grad_norms:
        raise NativeGRPOUpdateRejected("native GRPO update rejected: gradient norm preflight returned no replicas")
    for rank, norm in enumerate(grad_norms):
        if isinstance(norm, bool) or not isinstance(norm, (int, float)) or not math.isfinite(float(norm)):
            raise NativeGRPOUpdateRejected(
                f"native GRPO update rejected: replica {rank} gradient norm is missing or non-finite"
            )
        if float(norm) <= 0.0:
            raise NativeGRPOUpdateRejected(
                f"native GRPO update rejected: replica {rank} gradient norm has no usable signal"
            )

    if (
        isinstance(valid_token_count, bool)
        or not isinstance(valid_token_count, int)
        or valid_token_count <= 0
    ):
        raise NativeGRPOUpdateRejected(
            f"native GRPO update rejected: last native-row valid-token count {valid_token_count!r} is not positive"
        )


def _latest_checkpoint_step(config: GRPOTrainConfig) -> int:
    """Return the latest trainer checkpoint step, or 0 when resume is disabled/unavailable."""

    if not config.resume_from_checkpoint:
        return 0
    step = _surogate.find_latest_checkpoint(config.checkpoint_dir)
    if step >= 0:
        logger.info(f"Found GRPO trainer checkpoint at step {step}")
        return int(step)
    logger.warning(f"No GRPO trainer checkpoint found in {config.checkpoint_dir}; starting from base weights")
    return 0


def _weights_path_for_start_step(config: GRPOTrainConfig, model_weights_path: str, start_step: int) -> str:
    """Choose the weight file to import before optional checkpoint restore."""

    if start_step <= 0 or config.lora:
        return model_weights_path
    checkpoint_weights = Path(config.checkpoint_dir) / f"step_{start_step:08d}" / "model.safetensors"
    return str(checkpoint_weights) if checkpoint_weights.exists() else model_weights_path


def _set_initial_adapter(config: GRPOTrainConfig, trainer: object, start_step: int) -> str | None:
    """Seed a fresh GRPO LoRA run from the configured PEFT adapter."""
    adapter_path = getattr(config, "adapter_path", None)
    if not adapter_path or start_step > 0:
        return None
    path = Path(adapter_path).expanduser().resolve()
    required = (path / "adapter_config.json", path / "adapter_model.safetensors")
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"GRPO initial adapter is incomplete at {path}: missing {', '.join(missing)}")
    if not config.lora:
        raise ValueError("GRPO adapter_path requires lora: true")
    mode = getattr(config, "adapter_init_mode", "merge")
    if mode == "merge":
        set_adapter_path = getattr(trainer, "set_adapter_path", None)
        if not callable(set_adapter_path):
            raise RuntimeError("GRPO trainer does not support merged adapter initialization")
        set_adapter_path(str(path))
    elif mode == "trainable":
        import_adapter = getattr(trainer, "import_adapter", None)
        if not callable(import_adapter):
            raise RuntimeError("GRPO trainer does not support trainable adapter initialization")
        adapter_config = json.loads(required[0].read_text(encoding="utf-8"))
        expected_rank = int(config.lora_rank)
        expected_alpha = float(config.lora_alpha)
        expected_targets = set(config.lora_target_modules)
        if int(adapter_config.get("r", -1)) != expected_rank:
            raise ValueError("trainable parent adapter rank does not match trainer LoRA rank")
        if float(adapter_config.get("lora_alpha", -1.0)) != expected_alpha:
            raise ValueError("trainable parent adapter alpha does not match trainer LoRA alpha")
        if set(adapter_config.get("target_modules", ())) != expected_targets:
            raise ValueError("trainable parent adapter targets do not match trainer LoRA targets")
    else:
        raise ValueError("adapter_init_mode must be 'merge' or 'trainable'")
    return str(path)


def _load_initial_trainable_adapter(
    config: GRPOTrainConfig,
    trainer: object,
    adapter_path: str | None,
) -> None:
    """Load a validated parent after the frozen base weights are available."""
    if adapter_path is None or getattr(config, "adapter_init_mode", "merge") != "trainable":
        return
    import_adapter = getattr(trainer, "import_adapter", None)
    if not callable(import_adapter):
        raise RuntimeError("GRPO trainer does not support trainable adapter initialization")
    import_adapter(str(Path(adapter_path) / "adapter_model.safetensors"))


def _filtered_config_for_logging(config: GRPOTrainConfig) -> dict:
    """Flatten config into JSON-serializable scalars for metrics_writer.log_config()."""
    raw = dict(vars(config))
    raw.pop("model_info", None)
    raw.pop("model", None)
    raw.pop("tokenizer", None)
    out: dict[str, bool | int | float | str] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _find_sample_boundaries(position_ids_flat: np.ndarray) -> list[tuple[int, int]]:
    """Find sample boundaries in packed position_ids.

    Packed sequences reset position_ids at each sample boundary (e.g.
    [0,1,2,0,1,0,1,2,3]).  Returns (start, end) tuples for each sample.
    """
    boundaries = [0]
    for i in range(1, len(position_ids_flat)):
        if position_ids_flat[i] == 0 and position_ids_flat[i - 1] != 0:
            # Only treat as a new sample if the next position is 1.
            if i + 1 < len(position_ids_flat) and position_ids_flat[i + 1] == 1:
                boundaries.append(i)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(position_ids_flat)
        ranges.append((start, end))
    return ranges


def _prepare_grpo_rank_row(micro_batch: dict[str, np.ndarray | None], seq_len: int) -> dict[str, np.ndarray | None]:
    """Pad one packed GRPO micro-batch into one native data-parallel rank row."""

    orig_input_ids = micro_batch["input_ids"]
    orig_position_ids = micro_batch["position_ids"]
    orig_targets = micro_batch["targets"]
    assert isinstance(orig_input_ids, np.ndarray)
    assert isinstance(orig_position_ids, np.ndarray)
    assert isinstance(orig_targets, np.ndarray)
    actual_tokens = int(orig_input_ids.shape[1])
    if actual_tokens <= 0 or actual_tokens > seq_len:
        raise ValueError(f"invalid GRPO micro-batch length {actual_tokens} for sequence_len={seq_len}")

    position_flat = orig_position_ids.flatten()
    sample_ranges = _find_sample_boundaries(position_flat)

    input_ids = np.zeros((1, seq_len), dtype=np.int32)
    input_ids[0, :actual_tokens] = orig_input_ids[0, :actual_tokens]
    position_ids = np.zeros((1, seq_len), dtype=np.int32)
    position_ids[0, :actual_tokens] = orig_position_ids[0, :actual_tokens]
    if actual_tokens < seq_len:
        last_position = int(position_ids[0, actual_tokens - 1])
        position_ids[0, actual_tokens:] = np.arange(
            last_position + 1,
            last_position + 1 + seq_len - actual_tokens,
            dtype=np.int32,
        )

    targets = np.full((1, seq_len), -100, dtype=np.int32)
    targets[0, :actual_tokens] = orig_targets[0, :actual_tokens]
    loss_mask_flat = np.asarray(micro_batch["loss_mask"]).flatten()
    shifted_mask = np.zeros(actual_tokens, dtype=bool)
    if actual_tokens > 1:
        shifted_mask[:-1] = loss_mask_flat[1:actual_tokens].astype(bool)
    targets[0, :actual_tokens][~shifted_mask] = -100

    def padded_float(name: str, fill: float = 0.0) -> np.ndarray:
        result = np.full(seq_len, fill, dtype=np.float32)
        values = np.asarray(micro_batch[name]).flatten()
        result[:actual_tokens] = values[:actual_tokens]
        return result

    def padded_mask(name: str) -> np.ndarray:
        result = np.zeros(seq_len, dtype=np.uint8)
        values = np.asarray(micro_batch[name]).flatten()
        result[:actual_tokens] = values[:actual_tokens].astype(np.uint8)
        return result

    teacher = micro_batch["teacher_logprobs"]
    teacher_logprobs = None
    if teacher is not None:
        teacher_logprobs = np.zeros(seq_len, dtype=np.float32)
        teacher_values = np.asarray(teacher).flatten()
        teacher_logprobs[:actual_tokens] = teacher_values[:actual_tokens]

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "targets": targets,
        "temperatures": padded_float("temperatures", fill=1.0).reshape(1, seq_len),
        "inference_logprobs": padded_float("inference_logprobs"),
        "advantages": padded_float("advantages"),
        "loss_mask": padded_mask("loss_mask"),
        "teacher_logprobs": teacher_logprobs,
        "opd_reference_logprobs": padded_float("opd_reference_logprobs"),
        "hindsight_logprobs": padded_float("hindsight_logprobs"),
        "hindsight_mask": padded_mask("hindsight_mask"),
        "replay_mask": padded_mask("replay_mask"),
        "replay_weights": padded_float("replay_weights", fill=1.0),
        "sample_starts": np.asarray([start for start, _ in sample_ranges], dtype=np.int32),
        "sample_ends": np.asarray([end for _, end in sample_ranges], dtype=np.int32),
    }


def _stack_grpo_rank_rows(
    micro_batches: list[dict[str, np.ndarray | None]],
    *,
    rank_width: int,
    seq_len: int,
) -> dict[str, np.ndarray | int | None]:
    """Build one native call with distinct rows for every data-parallel GPU."""

    if not micro_batches or len(micro_batches) > rank_width:
        raise ValueError("rank row group must contain between one and rank_width micro-batches")
    rows = [_prepare_grpo_rank_row(micro_batch, seq_len) for micro_batch in micro_batches]
    valid_rows = len(rows)
    while len(rows) < rank_width:
        rows.append(
            {
                "input_ids": np.zeros((1, seq_len), dtype=np.int32),
                "position_ids": np.arange(seq_len, dtype=np.int32).reshape(1, seq_len),
                "targets": np.full((1, seq_len), -100, dtype=np.int32),
                "temperatures": np.ones((1, seq_len), dtype=np.float32),
                "inference_logprobs": np.zeros(seq_len, dtype=np.float32),
                "advantages": np.zeros(seq_len, dtype=np.float32),
                "loss_mask": np.zeros(seq_len, dtype=np.uint8),
                "teacher_logprobs": None,
                "opd_reference_logprobs": np.zeros(seq_len, dtype=np.float32),
                "hindsight_logprobs": np.zeros(seq_len, dtype=np.float32),
                "hindsight_mask": np.zeros(seq_len, dtype=np.uint8),
                "replay_mask": np.zeros(seq_len, dtype=np.uint8),
                "replay_weights": np.ones(seq_len, dtype=np.float32),
                "sample_starts": np.empty(0, dtype=np.int32),
                "sample_ends": np.empty(0, dtype=np.int32),
            }
        )

    samples_per_rank = max(max(len(np.asarray(row["sample_starts"])) for row in rows), 1)
    sample_starts = np.full((rank_width, samples_per_rank), -1, dtype=np.int32)
    sample_ends = np.full((rank_width, samples_per_rank), -1, dtype=np.int32)
    for rank, row in enumerate(rows):
        starts = np.asarray(row["sample_starts"])
        ends = np.asarray(row["sample_ends"])
        sample_starts[rank, : len(starts)] = starts
        sample_ends[rank, : len(ends)] = ends

    def concat(name: str) -> np.ndarray:
        return np.ascontiguousarray(np.concatenate([np.asarray(row[name]) for row in rows], axis=0))

    any_teacher = any(row["teacher_logprobs"] is not None for row in rows)
    teacher_logprobs = None
    if any_teacher:
        teacher_logprobs = np.ascontiguousarray(
            np.concatenate(
                [
                    np.asarray(row["teacher_logprobs"])
                    if row["teacher_logprobs"] is not None
                    else np.zeros(seq_len, dtype=np.float32)
                    for row in rows
                ]
            )
        )

    return {
        "input_ids": concat("input_ids"),
        "position_ids": concat("position_ids"),
        "targets": concat("targets"),
        "temperatures": concat("temperatures"),
        "inference_logprobs": concat("inference_logprobs"),
        "advantages": concat("advantages"),
        "loss_mask": concat("loss_mask"),
        "teacher_logprobs": teacher_logprobs,
        "opd_reference_logprobs": concat("opd_reference_logprobs"),
        "hindsight_logprobs": concat("hindsight_logprobs"),
        "hindsight_mask": concat("hindsight_mask"),
        "replay_mask": concat("replay_mask"),
        "replay_weights": concat("replay_weights"),
        "sample_starts": np.ascontiguousarray(sample_starts.reshape(-1)),
        "sample_ends": np.ascontiguousarray(sample_ends.reshape(-1)),
        "samples_per_rank": samples_per_rank,
        "valid_rows": valid_rows,
    }


class GRPOTrainer:
    """GRPO RL trainer using Surogate's C++ engine."""

    def __init__(self, config: GRPOTrainConfig, external_weights: list[list[dict]] | None = None):
        self.config = config
        self.start_step = _latest_checkpoint_step(config)

        # Build DSL IR for the model (same pattern as SurogateTrainerWrapper)
        from surogate.dsl.ir_builder import build_dsl_ir_for_model

        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json

        # Compile JIT kernels (e.g. gated delta rule Triton kernels)
        from surogate.kernels.jit_compile import compile_jit_kernels

        jit_manifests = compile_jit_kernels(ir_json)
        if jit_manifests:
            config.runtime_config.jit_kernel_manifests = jit_manifests

        # Create C++ trainer using inherited config objects
        logger.info(f"Creating GRPO trainer for {config.model} ({config.gpus} GPUs)")
        self.trainer = _surogate.SurogateTrainer(
            ngpu=config.gpus,
            config=_surogate.PretrainedConfig.from_pretrained(config.model_dir, to_surogate_dtype(config.torch_dtype)),
            options=config.runtime_config,
            batch_size=config.per_device_train_batch_size,
            seq_len=config.sequence_len,
            grad_accum=1,  # Set dynamically per step via set_grad_accumulation()
            memcpy_all_gather=config.memcpy_all_gather,
            memcpy_send_recv=config.memcpy_send_recv,
            lora_config=config.lora_config,
            qlora_config=config.qlora_config,
        )

        # uses shift_tensor_right with pad_value=log(1/vocab_size).
        self._pad_logprob = None
        try:
            from surogate.core.model.hf_config import HfConfigFactory

            vocab_size = HfConfigFactory.get_config_attr(config.model_info.config, "vocab_size")
        except Exception:
            vocab_size = None
        if vocab_size:
            self._pad_logprob = float(np.log(1.0 / float(vocab_size)))

        # Import pretrained weights
        model_weights_path = get_model_weights_path(config.model_dir)
        weights_path = _weights_path_for_start_step(config, model_weights_path, self.start_step)
        initial_adapter = _set_initial_adapter(config, self.trainer, self.start_step)
        if initial_adapter is not None:
            logger.info(f"Initializing GRPO LoRA from {initial_adapter}")
        if external_weights is not None:
            # Zero-copy import from external GPU pointers (colocate mode with vLLM)
            logger.info(f"Importing weights from external GPU pointers (non-quantized from {weights_path})")
            self.trainer.import_weights_from_external(weights_path, external_weights)
        else:
            logger.info(f"Importing weights from {weights_path}")
            self.trainer.import_weights(weights_path)
        _load_initial_trainable_adapter(config, self.trainer, initial_adapter)
        if self.start_step > 0:
            logger.info(f"Loading GRPO trainer checkpoint from step {self.start_step}")
            self.trainer.load_checkpoint(str(config.checkpoint_dir), self.start_step)
            logger.info("GRPO trainer checkpoint loaded successfully")

        # loss_scale is computed dynamically per pack — see train() loop

        # LR schedule — max_steps here is "orchestrator steps" (one per pack() cycle).
        # The internal micro-step counter is separate.
        lr_max_steps = config.max_steps if config.max_steps > 0 else 1_000_000
        warmup_steps = config.warmup_steps
        if warmup_steps == 0 and config.warmup_ratio > 0:
            warmup_steps = int(lr_max_steps * config.warmup_ratio)

        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=lr_max_steps,
            warmup_steps=warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type,
            wsd_decay_steps_fraction=config.wsd_decay_steps_fraction,
        )

        # Weight broadcast (with optional QeRL noise injection)
        if config.weight_broadcast_type == "colocate":
            from surogate.grpo.weight_broadcast import ColocateWeightBroadcast

            self.broadcast = ColocateWeightBroadcast(
                output_dir=config.output_dir,
                max_async_level=config.max_async_level,
                noise_config=config.noise_scheduler,
                base_model_dir=config.model_dir,
                max_steps=config.max_steps,
            )
        else:
            self.broadcast = SurogateWeightBroadcast(
                output_dir=config.output_dir,
                adapter_only=config.lora,
                max_async_level=config.max_async_level,
                noise_config=config.noise_scheduler,
                base_model_dir=config.model_dir,
                max_steps=config.max_steps,
            )

        # Data loader setup is deferred to train() since packer must run first
        self.data_loader: GRPODataLoader | None = None
        self.packer = None

        # Optional metrics writers (driven by config.report_to)
        self.metrics_writer: MetricsWriter | None = None
        backends = config.report_to or []
        if isinstance(backends, str):
            backends = [backends]
        if "surogate" in backends:
            self.metrics_writer = MetricsWriter(output_path=config.surogate_metrics_path)
            self.metrics_writer.log_config(**_filtered_config_for_logging(config))

    def _copy_tokenizer_files(self, src_dir: str, dst_dir: str):
        """Copy tokenizer, vocab, and config files from source model to output directory."""
        tokenizer_files = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "chat_template.jinja",
            "generation_config.json",
        ]
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for filename in tokenizer_files:
            src = src_path / filename
            if src.exists():
                shutil.copy(src, dst_path / filename)

    def _setup_data(self, start_step: int = 0):
        """Set up data loader and optionally packer (master only)."""
        config = self.config

        # Build transport config
        if config.transport_type == "zmq":
            from surogate.core.config.grpo_orch_config import ZMQTransportConfig

            transport_config = ZMQTransportConfig({})
        else:
            from surogate.core.config.grpo_orch_config import FileSystemTransportConfig

            transport_config = FileSystemTransportConfig({})

        # dp_rank = 0 for single-node (all GPUs on same node share data)
        dp_rank = 0
        dp_world_size = 1  # Surogate handles multi-GPU internally

        # Initialize MultiRunManager singleton (required before packer)
        from surogate.grpo.packer import init_multi_run_manager, setup_grpo_packer

        init_multi_run_manager(output_dir=config.output_dir)

        # Setup packer on master (packs TrainingBatch -> MicroBatch)
        tokenizer = config.tokenizer
        self.packer = setup_grpo_packer(
            dp_world_size=dp_world_size,
            seq_len=config.sequence_len,
            pad_to_multiple_of=config.pad_to_multiple_of,
            tokenizer=tokenizer,
            transport_config=transport_config,
            start_step=start_step,
            sample_packing=config.sample_packing,
        )

        # Setup data loader (receives packed MicroBatches)
        self.data_loader = GRPODataLoader(
            output_dir=config.output_dir,
            dp_rank=dp_rank,
            start_step=start_step,
            transport_config=transport_config,
        )

    def train(self):
        """Main GRPO training loop."""
        config = self.config
        max_steps = config.max_steps

        self._setup_data(start_step=self.start_step)

        # Get MultiRunManager — packer auto-increments progress[0].step after
        # each pack() call.
        mrm = get_multi_run_manager()

        logger.info("Starting GRPO training loop")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  GPUs: {config.gpus}")
        logger.info(f"  Sequence length: {config.sequence_len}")
        logger.info("  Gradient accumulation: dynamic (from packer, no fixed cap)")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  LoRA: enabled={config.lora}, rank={config.lora_rank}, alpha={config.lora_alpha}")
        logger.info(f"  Recipe: {config.recipe}")
        logger.info(f"  Optimizer: {config.optimizer}")
        logger.info(
            f"  Loss: kl_tau={config.loss.kl_tau}, adv_tau={config.loss.adv_tau}, "
            f"ipo_mask_low={config.loss.ipo_mask_low}, ipo_mask_high={config.loss.ipo_mask_high}"
        )
        logger.info(f"  Doc masking: {config.doc_masking}")
        if config.noise_scheduler and config.noise_scheduler.enabled:
            ns = config.noise_scheduler
            logger.info(
                f"  QeRL noise: sigma_start={ns.sigma_start}, sigma_end={ns.sigma_end}, num_stages={ns.num_stages}"
            )
        if max_steps > 0:
            logger.info(f"  Max steps (orchestrator): {max_steps}")
        else:
            logger.info("  Running indefinitely (waiting for orchestrator)")

        step = self.start_step  # Internal trainer step (one per grad_accum chunk, for LR schedule + logging)
        while True:
            orch_step = mrm.progress[0].step if 0 in mrm.progress else self.start_step

            # 1. Broadcast weights (after first orchestrator step)
            if orch_step > 0:
                self.broadcast.broadcast(self.trainer, orch_step)
                self.broadcast.cleanup(orch_step)

            # Check if we've reached max steps
            if 0 < max_steps <= orch_step:
                logger.info(f"Reached max steps ({max_steps}), stopping")
                break

            # 2. Pack and wait for batch — packer increments progress[0].step
            self.packer.pack()
            self.data_loader.wait_for_batch()

            # 3. Get micro-batches
            micro_batches = self.data_loader.get_batch()
            if not micro_batches:
                logger.warning("No micro-batches received, retrying...")
                continue

            # Accumulate ALL micro-batches from one pack into a single optimizer step.
            # No fixed gradient_accumulation_steps — the count is determined dynamically by the packer each step.
            seq_len = config.sequence_len
            n_mb = len(micro_batches)

            # Divide total loss by the sum of loss_mask
            # across all micro-batches in this optimizer step.
            loss_scale = int(sum(int(mb["loss_mask"].sum()) for mb in micro_batches))
            loss_scale = max(loss_scale, 1)

            step_start = time.time()

            # Note: loss_scale normalization is applied explicitly to per-token grads
            # (loss = total_loss / loss_scale).
            # Each native trainer rank holds a model replica. Feed it a distinct
            # packed row instead of tiling the same row across every GPU. LoRA
            # gradients are averaged across ranks, so divide the global token
            # denominator by rank_width to recover the exact global-sum/global-token
            # gradient after that average. The final partial group is padded with
            # zero-loss rows and follows the same normalization.
            if getattr(config, "ep_size", 1) != 1:
                raise RuntimeError("rank-batched native GRPO currently requires ep_size=1")
            rank_width = int(config.gpus)
            native_micro_steps = (n_mb + rank_width - 1) // rank_width
            self.trainer.set_grad_accumulation(native_micro_steps)
            native_loss_scale = float(loss_scale) / float(rank_width)
            progress_every = max(1, native_micro_steps // 8)

            # 4. Process up to one distinct micro-batch per GPU per native call.
            for native_idx, start in enumerate(range(0, n_mb, rank_width)):
                rank_batch = _stack_grpo_rank_rows(
                    micro_batches[start : start + rank_width],
                    rank_width=rank_width,
                    seq_len=seq_len,
                )
                self.trainer.step_grpo_native(
                    rank_batch["input_ids"],
                    rank_batch["targets"],
                    rank_batch["inference_logprobs"],
                    rank_batch["advantages"],
                    rank_batch["loss_mask"],
                    rank_batch["sample_starts"],
                    rank_batch["sample_ends"],
                    position_ids=rank_batch["position_ids"],
                    temperatures=rank_batch["temperatures"],
                    teacher_logprobs=rank_batch["teacher_logprobs"],
                    opd_reference_logprobs=rank_batch["opd_reference_logprobs"],
                    hindsight_logprobs=rank_batch["hindsight_logprobs"],
                    hindsight_mask=rank_batch["hindsight_mask"],
                    replay_mask=rank_batch["replay_mask"],
                    replay_weights=rank_batch["replay_weights"],
                    loss_scale=native_loss_scale,
                    ipo_mask_low=float(config.loss.ipo_mask_low),
                    ipo_mask_high=float(config.loss.ipo_mask_high),
                    adv_tau=float(config.loss.adv_tau),
                    teacher_tau=float(config.loss.teacher_tau),
                    opd_tau=float(config.loss.opd_tau),
                    opd_beta=float(config.loss.opd_beta),
                    replay_tau=float(config.loss.replay_tau),
                    kl_tau=float(config.loss.kl_tau),
                    samples_per_rank=int(rank_batch["samples_per_rank"]),
                )
                completed = native_idx + 1
                if completed == native_micro_steps or completed % progress_every == 0:
                    logger.info(
                        "GRPO native progress: %d/%d rank-batched calls (%d/%d micro-batches)",
                        completed,
                        native_micro_steps,
                        min(start + rank_width, n_mb),
                        n_mb,
                    )

            # 5. Optimizer step — one per orchestrator step
            lr = self.lr_schedule.get_lr(orch_step)
            opt_config = _surogate.OptimizerConfig(
                optimizer=config.optimizer,
                learning_rate=lr,
                weight_decay=config.weight_decay,
                grad_clip=config.max_grad_norm,
                adamw_beta1=config.adamw_beta1,
                adamw_beta2=config.adamw_beta2,
                adamw_epsilon=config.adamw_epsilon,
            )
            # VTC is the last native row's diagnostic count, not the global
            # denominator. The global exact-token check uses native
            # metrics["total_tokens"] below.
            vtc = self.trainer.get_valid_token_count(0)
            expected_loss_scale = loss_scale

            # Replay-bearing updates are the ALE product path. Native metrics
            # and every replica's LoRA gradient norm are readable after
            # backward and before update_with_config() mutates optimizer state
            # or weights. Any rejection propagates out of train(), so neither
            # the candidate broadcast nor final export can run.
            if float(config.loss.replay_tau) > 0.0:
                step_metrics = dict(self.trainer.get_grpo_native_metrics())
                preflight_grad_norms = list(
                    self.trainer.preflight_grpo_native_lora_gradient_norms(float(config.max_grad_norm))
                )
                _validate_replay_anchored_native_update(
                    metrics=step_metrics,
                    grad_norms=preflight_grad_norms,
                    valid_token_count=vtc,
                    expected_loss_scale=expected_loss_scale,
                )
                result = self.trainer.update_with_config(opt_config, step)
            else:
                result = self.trainer.update_with_config(opt_config, step)
                step_metrics = dict(self.trainer.get_grpo_native_metrics())

            step_time = time.time() - step_start

            # 6. Logging
            if step % config.logging_steps == 0:
                logger.info(
                    f"step={step} loss={step_metrics.get('policy_loss', 0):.4f} "
                    f"grad_norm={result['norm']:.4f} lr={lr:.2e} "
                    f"kl={step_metrics.get('mismatch_kl', 0):.4f} "
                    f"masked={step_metrics.get('is_masked', 0):.2%} "
                    f"tokens={step_metrics.get('total_tokens', 0)} "
                    f"micro_batches={n_mb} native_micro_steps={native_micro_steps} "
                    f"vtc={vtc} expected={expected_loss_scale} "
                    f"time={step_time:.2f}s"
                )

                if self.metrics_writer is not None:
                    duration_ms = step_time * 1000.0
                    total_tokens = int(step_metrics.get("total_tokens", 0))
                    tps = total_tokens / step_time if step_time > 0 else 0.0
                    self.metrics_writer.track(
                        step,
                        **{
                            "train/policy_loss": float(step_metrics.get("policy_loss", 0.0)),
                            "train/mismatch_kl": float(step_metrics.get("mismatch_kl", 0.0)),
                            "train/masked_mismatch_kl": float(step_metrics.get("masked_mismatch_kl", 0.0)),
                            "train/unmasked_mismatch_kl": float(step_metrics.get("unmasked_mismatch_kl", 0.0)),
                            "train/is_masked": float(step_metrics.get("is_masked", 0.0)),
                            "train/is_masked_low": float(step_metrics.get("is_masked_low", 0.0)),
                            "train/is_masked_high": float(step_metrics.get("is_masked_high", 0.0)),
                            "train/teacher_kl": float(step_metrics.get("teacher_kl", 0.0)),
                            "train/opd_loss": float(step_metrics.get("opd_loss", 0.0)),
                            "train/opd_gate": float(step_metrics.get("opd_gate", 0.0)),
                            "train/opd_shift": float(step_metrics.get("opd_shift", 0.0)),
                            "train/opd_tokens": float(step_metrics.get("opd_tokens", 0.0)),
                            "train/replay_loss": float(step_metrics.get("replay_loss", 0.0)),
                            "train/replay_tokens": float(step_metrics.get("replay_tokens", 0.0)),
                            "train/replay_weight_sum": float(step_metrics.get("replay_weight_sum", 0.0)),
                            "train/keep_tokens": float(step_metrics.get("keep_tokens", 0.0)),
                            "train/total_tokens": float(total_tokens),
                            "train/grad_norm": float(result["norm"]),
                            "train/lr": float(lr),
                            "train/micro_batches": float(n_mb),
                            "train/native_micro_steps": float(native_micro_steps),
                            "train/sample_count": float(step_metrics.get("sample_count", 0.0)),
                            "train/vtc": float(vtc),
                            "train/expected_loss_scale": float(expected_loss_scale),
                            "train/duration_ms": duration_ms,
                            "train/tokens_per_second": tps,
                            "train/orch_step": float(orch_step),
                        },
                    )

            # 7. Checkpointing
            if config.save_steps > 0 and step > 0 and step % config.save_steps == 0 and config.checkpoint_dir:
                logger.info(f"Saving checkpoint at step {step}...")
                self.trainer.save_checkpoint(config.checkpoint_dir, step)

            step += 1

        # Final weight broadcast
        self.broadcast.broadcast(self.trainer, mrm.progress[0].step)

        # Save final adapter/model
        output_path = Path(config.output_dir)
        if config.lora:
            adapter_dir = output_path / "final_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_adapter(str(adapter_dir))
            logger.info(f"Final LoRA adapter saved to {adapter_dir}")

            if config.merge_adapter:
                from surogate.utils.adapter_merge import merge_adapter

                merged_dir = output_path / "final_merged"
                merged_dir.mkdir(parents=True, exist_ok=True)
                try:
                    merge_adapter(
                        base_model_path=config.model_dir,
                        adapter_path=str(adapter_dir),
                        output_path=str(merged_dir),
                        max_shard_size="5GB",
                        cpu_offload=True,
                    )
                    self._copy_tokenizer_files(config.model_dir, str(merged_dir))
                    logger.info(f"Merged model saved to {merged_dir}")
                except Exception as e:
                    logger.error(f"Failed to merge adapter: {e}")
                    import traceback

                    logger.error(f"Traceback:\n{traceback.format_exc()}")
        else:
            model_dir = output_path / "final_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_model(str(model_dir))
            self._copy_tokenizer_files(config.model_dir, str(model_dir))
            logger.info(f"Final model saved to {model_dir}")

        logger.info(f"GRPO training complete after {step} steps")

    def close(self):
        if self.metrics_writer is not None:
            self.metrics_writer.close()
            self.metrics_writer = None


def grpo_train(config: GRPOTrainConfig):
    """Entry point for GRPO training."""
    trainer = GRPOTrainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()
