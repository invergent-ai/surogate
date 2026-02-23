"""Main GRPO trainer: replaces prime-rl's trainer with Surogate's C++ engine.

Training loop:
    1. Wait for batch from orchestrator (packer on master, transport to all ranks)
    2. For each micro-batch (packed sequence with multiple samples):
       a. compute_logprobs() on packed batch -> current policy log-probs
       b. compute_grpo_per_token_grads() -> per-token gradient multipliers
       c. step_with_custom_loss() on packed batch -> forward + backward
    3. update_with_config() -> optimizer step
    4. Broadcast updated weights to inference engine

Document-level attention masking (Flash Attention varlen) can be enabled to
prevent cross-sample attention in packed sequences. Prime-RL enables this for
flash attention via position_id resets, so we default it on for parity.
"""

import shutil
import time
from pathlib import Path

import numpy as np

from surogate import _surogate
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.data import GRPODataLoader
from surogate.grpo.loss import compute_grpo_per_token_grads
from surogate.grpo.weight_broadcast import SurogateWeightBroadcast
from surogate.train.lr_schedule import LRSchedule
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype
from surogate.grpo.runs import get_multi_run_manager

logger = get_logger()


def _find_sample_boundaries(position_ids_flat: np.ndarray) -> list[tuple[int, int]]:
    """Find sample boundaries in packed position_ids.

    Packed sequences reset position_ids at each sample boundary (e.g.
    [0,1,2,0,1,0,1,2,3]).  Returns (start, end) tuples for each sample.
    """
    boundaries = [0]
    for i in range(1, len(position_ids_flat)):
        if position_ids_flat[i] == 0 and position_ids_flat[i - 1] != 0:
            # Only treat as a new sample if the next position is 1 (prime-rl behavior).
            if i + 1 < len(position_ids_flat) and position_ids_flat[i + 1] == 1:
                boundaries.append(i)
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(position_ids_flat)
        ranges.append((start, end))
    return ranges


class GRPOTrainer:
    """GRPO RL trainer using Surogate's C++ engine."""

    def __init__(self, config: GRPOTrainConfig, external_weights: list[list[dict]] | None = None):
        self.config = config

        # Build DSL IR for the model (same pattern as SurogateTrainerWrapper)
        from surogate.dsl.ir_builder import build_dsl_ir_for_model
        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json

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

        # Prime-RL uses shift_tensor_right with pad_value=log(1/vocab_size).
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
        if external_weights is not None:
            # Zero-copy import from external GPU pointers (colocate mode with vLLM)
            logger.info(f"Importing weights from external GPU pointers "
                        f"(non-quantized from {model_weights_path})")
            self.trainer.import_weights_from_external(model_weights_path, external_weights)
        else:
            logger.info(f"Importing weights from {model_weights_path}")
            self.trainer.import_weights(model_weights_path)

        # loss_scale is computed dynamically per pack — see train() loop

        # LR schedule — max_steps here is "orchestrator steps" (one per pack() cycle),
        # matching prime-rl's convention.  The internal micro-step counter is separate.
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

    def _copy_tokenizer_files(self, src_dir: str, dst_dir: str):
        """Copy tokenizer, vocab, and config files from source model to output directory."""
        tokenizer_files = [
            "config.json",
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt",
            "added_tokens.json", "chat_template.jinja", "generation_config.json"
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

        self._setup_data(start_step=0)

        # Get MultiRunManager — packer auto-increments progress[0].step after
        # each pack() call, matching prime-rl's convention.
        mrm = get_multi_run_manager()

        logger.info("Starting GRPO training loop")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  GPUs: {config.gpus}")
        logger.info(f"  Sequence length: {config.sequence_len}")
        logger.info(f"  Gradient accumulation: dynamic (from packer, no fixed cap)")
        logger.info(f"  Learning rate: {config.learning_rate}")
        logger.info(f"  LoRA: enabled={config.lora}, rank={config.lora_rank}, alpha={config.lora_alpha}")
        logger.info(f"  Recipe: {config.recipe}")
        logger.info(f"  Optimizer: {config.optimizer}")
        logger.info(f"  Loss: ratio_type={config.loss.ratio_type}, "
                     f"kl_tau={config.loss.kl_tau}, adv_tau={config.loss.adv_tau}")
        logger.info(f"  Doc masking: {config.doc_masking}")
        if config.noise_scheduler and config.noise_scheduler.enabled:
            ns = config.noise_scheduler
            logger.info(f"  QeRL noise: sigma_start={ns.sigma_start}, "
                         f"sigma_end={ns.sigma_end}, num_stages={ns.num_stages}")
        if max_steps > 0:
            logger.info(f"  Max steps (orchestrator): {max_steps}")
        else:
            logger.info("  Running indefinitely (waiting for orchestrator)")

        step = 0  # Internal trainer step (one per grad_accum chunk, for LR schedule + logging)
        while True:
            orch_step = mrm.progress[0].step if 0 in mrm.progress else 0

            # 1. Broadcast weights (after first orchestrator step)
            if orch_step > 0:
                self.broadcast.broadcast(self.trainer, orch_step)
                self.broadcast.cleanup(orch_step)

            # Check if we've reached max steps (same check as prime-rl's trainer)
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

            # Match prime-rl: accumulate ALL micro-batches from one pack into
            # a single optimizer step.  No fixed gradient_accumulation_steps —
            # the count is determined dynamically by the packer each step.
            seq_len = config.sequence_len
            n_mb = len(micro_batches)

            # Tell the C++ engine how many micro-steps this optimizer step has.
            self.trainer.set_grad_accumulation(n_mb)

            # Prime-RL loss scaling: divide total loss by the sum of loss_mask
            # across all micro-batches in this optimizer step.
            loss_scale = int(sum(int(mb["loss_mask"].sum()) for mb in micro_batches))
            loss_scale = max(loss_scale, 1)

            step_start = time.time()

            # Note: loss_scale normalization is applied explicitly to per-token grads
            # (matching prime-rl's loss = total_loss / loss_scale).

            # 4. Process each micro-batch (gradients accumulate in C++)
            step_metrics = {
                "policy_loss": 0.0,
                "mismatch_kl": 0.0,
                "is_masked_frac": 0.0,
                "keep_tokens": 0,
                "total_tokens": 0,
            }

            for mb_idx, mb in enumerate(micro_batches):
                mb_start = time.time()

                # Original micro-batch data
                orig_input_ids = mb["input_ids"]          # [1, T_mb]
                orig_position_ids = mb["position_ids"]    # [1, T_mb]
                orig_targets = mb["targets"]              # [1, T_mb]
                advantages_flat = mb["advantages"].flatten()
                inference_lp = mb["inference_logprobs"].flatten()
                loss_mask_flat = mb["loss_mask"].flatten()
                temps_flat = mb["temperatures"].flatten()

                teacher_lp = None
                if mb["teacher_logprobs"] is not None:
                    teacher_lp = mb["teacher_logprobs"].flatten()

                T_actual = orig_input_ids.shape[1]
                ngpu = config.gpus

                # Find sample boundaries for per-sample logprob/gradient shifting
                pos_flat = orig_position_ids.flatten()
                sample_ranges = _find_sample_boundaries(pos_flat)

                # --- Build logprob targets (global next-token shift across packed sequence) ---
                # Matches prime-rl's shift_tensor_left on the packed input.
                logprob_targets = np.full((1, seq_len), -100, dtype=np.int32)
                logprob_targets[0, :T_actual] = orig_targets[0, :T_actual]

                # Pad inputs to seq_len. Always pad position_ids sequentially
                # from the last real position so RoPE resets are preserved.
                input_padded = np.zeros((1, seq_len), dtype=np.int32)
                input_padded[0, :T_actual] = orig_input_ids[0, :T_actual]
                pos_padded = np.zeros((1, seq_len), dtype=np.int32)
                pos_padded[0, :T_actual] = orig_position_ids[0, :T_actual]
                if T_actual < seq_len:
                    last_pos = int(pos_padded[0, T_actual - 1])
                    pos_padded[0, T_actual:] = np.arange(
                        last_pos + 1, last_pos + 1 + seq_len - T_actual, dtype=np.int32
                    )
                temp_padded = np.ones((1, seq_len), dtype=np.float32)
                temp_padded[0, :T_actual] = temps_flat[:T_actual]

                # --- Compute logprobs on packed batch ---
                # C++ engine detects document boundaries from position_ids and
                # uses Flash Attention varlen for document-level masking.
                logprob_start = time.time()
                raw_lp_full = self.trainer.compute_logprobs(
                    input_padded, logprob_targets, use_lora=config.lora,
                    position_ids=pos_padded,
                    temperatures=temp_padded,
                )
                raw_lp = np.asarray(raw_lp_full[0, :T_actual], dtype=np.float32)
                logprob_time = time.time() - logprob_start

                # Right-shift globally (packed sequence), matching prime-rl's
                # shift_tensor_right after a packed shift_tensor_left.
                all_trainer_logprobs = np.zeros(T_actual, dtype=np.float32)
                if T_actual > 1:
                    all_trainer_logprobs[1:T_actual] = raw_lp[:T_actual - 1]
                if self._pad_logprob is not None and T_actual > 0:
                    all_trainer_logprobs[0] = self._pad_logprob

                # --- GRPO gradient computation (per-sample, matching prime-rl) ---
                grpo_start = time.time()
                per_token_grads_flat, mb_metrics = compute_grpo_per_token_grads(
                    trainer_logprobs=all_trainer_logprobs,
                    inference_logprobs=inference_lp[:T_actual],
                    advantages=advantages_flat[:T_actual],
                    loss_mask=loss_mask_flat[:T_actual],
                    loss_config=config.loss,
                    sample_ranges=sample_ranges,
                    teacher_logprobs=teacher_lp[:T_actual] if teacher_lp is not None else None,
                )
                if loss_scale > 1:
                    per_token_grads_flat = per_token_grads_flat / float(loss_scale)

                # --- Diagnostic: per-micro-batch logprob + masking summary ---
                if step <= 1:
                    mask = loss_mask_flat[:T_actual].astype(bool)
                    t_lp = all_trainer_logprobs[:T_actual]
                    n_nan = int(np.isnan(t_lp[mask]).sum())
                    n_fin = int(np.isfinite(t_lp[mask]).sum())
                    logger.info(
                        f"[DIAG] step={step} mb={mb_idx}: "
                        f"T={T_actual} mask_count={mask.sum()} "
                        f"nan_trainer={n_nan} finite_trainer={n_fin} "
                        f"mb_is_masked_frac={mb_metrics.get('is_masked_frac', -1):.4f} "
                        f"mb_policy_loss={mb_metrics.get('policy_loss', -1):.4f} "
                        f"n_samples={len(sample_ranges)}"
                    )

                # --- Global left-shift of gradients (packed sequence) ---
                # Align with global next-token shift used for labels.
                surogate_grads = np.zeros(T_actual, dtype=np.float32)
                if T_actual > 1:
                    surogate_grads[:T_actual - 1] = per_token_grads_flat[1:T_actual]
               
                # Packed backward: all samples in one call via flash varlen.
                # Mask out prompt tokens using a left-shifted loss mask so that
                # targets align with per-token grads (grads are shifted left to labels).
                # This keeps ValidTokenCount aligned with the GRPO loss mask while
                # avoiding zeroing the first response token's gradient per sample.
                # Use the same packed next-token targets as compute_logprobs
                # to keep gradients aligned with the logprobs used for GRPO.
                targets_padded = logprob_targets.copy()
                tmask = loss_mask_flat[:T_actual].astype(bool)
                tmask_shifted = np.zeros_like(tmask)
                if T_actual > 1:
                    tmask_shifted[:-1] = tmask[1:]
                targets_padded[0, :T_actual][~tmask_shifted] = -100
                grads_padded = np.zeros((1, seq_len), dtype=np.float32)
                grads_padded[0, :T_actual] = surogate_grads

                input_step = input_padded
                pos_step = pos_padded
                temp_step = temp_padded
                if ngpu > 1:
                    input_step = np.tile(input_padded, (ngpu, 1))
                    pos_step = np.tile(pos_padded, (ngpu, 1))
                    targets_padded = np.tile(targets_padded, (ngpu, 1))
                    grads_padded = np.tile(grads_padded, (ngpu, 1))
                    temp_step = np.tile(temp_padded, (ngpu, 1))

                self.trainer.step_with_custom_loss(
                    input_step, targets_padded, grads_padded,
                    position_ids=pos_step,
                    temperatures=temp_step,
                )

                # Accumulate metrics
                for key in step_metrics:
                    if key in mb_metrics:
                        step_metrics[key] += mb_metrics[key]

            # 5. Optimizer step — one per orchestrator step (matching prime-rl)
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
            # Read VTC before optimizer step for diagnostics
            vtc = self.trainer.get_valid_token_count(0)
            expected_loss_scale = loss_scale

            result = self.trainer.update_with_config(opt_config, step + 1)

            step_time = time.time() - step_start

            # 6. Logging
            if step % config.logging_steps == 0:
                avg_metrics = {k: v / max(n_mb, 1) for k, v in step_metrics.items()
                               if isinstance(v, float)}
                logger.info(
                    f"step={step} loss={avg_metrics.get('policy_loss', 0):.4f} "
                    f"grad_norm={result['norm']:.4f} lr={lr:.2e} "
                    f"kl={avg_metrics.get('mismatch_kl', 0):.4f} "
                    f"masked={avg_metrics.get('is_masked_frac', 0):.2%} "
                    f"tokens={step_metrics.get('total_tokens', 0)} "
                    f"micro_batches={n_mb} "
                    f"vtc={vtc} expected={expected_loss_scale} "
                    f"time={step_time:.2f}s"
                )

            # 7. Checkpointing
            if (config.save_steps > 0 and step > 0
                    and step % config.save_steps == 0
                    and config.checkpoint_dir):
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
        else:
            model_dir = output_path / "final_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.export_model(str(model_dir))
            self._copy_tokenizer_files(config.model_dir, str(model_dir))
            logger.info(f"Final model saved to {model_dir}")

        logger.info(f"GRPO training complete after {step} steps")


def grpo_train(config: GRPOTrainConfig):
    """Entry point for GRPO training."""
    trainer = GRPOTrainer(config)
    trainer.train()
