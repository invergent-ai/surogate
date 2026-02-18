import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from surogate import _surogate
from surogate.core.config.sft_config import SFTConfig
from surogate.train.early_stopping import EarlyStopping
from surogate.train.gradient_tracker import GradientTracker
from surogate.train.loss_guard import LossGuard
from surogate.train.moe_monitor import MoEMonitor
from surogate.train.lr_schedule import LRSchedule
from surogate.train.training_advisor import TrainingAdvisor
from surogate.train.metrics import MoEMetrics, StepMetrics
from surogate.train.phase_detector import PhaseDetector
from surogate.train.plateau_detector import PlateauDetector
from surogate.train.reporter import training_logger_context
from surogate.train.training_plot import generate_training_plot
from surogate.train.on_the_fly_mm import OnTheFlyMultimodalBatcher, init_mm_helpers, load_multimodal_datasets
from surogate.utils.adapter_merge import merge_adapter
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_surogate_dtype
from surogate.utils.model import estimate_model_parameters

logger = get_logger()


class SurogateTrainerWrapper():
    def __init__(
            self,
            config: SFTConfig,
            train_files: List[str],
            eval_files: Optional[List[str]] = None
    ):
        self.config = config
        self._block_types = None
        self._mm_on_the_fly = bool(config.multimodal_on_the_fly and config.model_template.is_multimodal)

        model_weights_path = get_model_weights_path(config.model_dir)

        from surogate.dsl.ir_builder import build_dsl_ir_for_model
        # Pass training-time config overrides that affect graph compilation
        dsl_extra = {}
        if getattr(config, "ep_size", 1) > 1:
            dsl_extra["ep_size"] = config.ep_size
        ir_json = build_dsl_ir_for_model(config.model_dir, extra_config=dsl_extra or None)
        config.runtime_config.dsl_ir_json = ir_json
        
        # Setup data loaders / on-the-fly batcher
        self.total_batch_size = config.per_device_train_batch_size * config.sequence_len * config.gpus * config.gradient_accumulation_steps
        self.chunk_size = config.per_device_train_batch_size * config.sequence_len * config.gpus

        if self._mm_on_the_fly:
            if config.sample_packing:
                logger.warning("multimodal_on_the_fly disables sample_packing; forcing sample_packing=False.")
                config.sample_packing = False
            if config.padding_free:
                logger.warning("multimodal_on_the_fly disables padding_free; forcing padding_free=False.")
                config.padding_free = False

            self.train_loader = None
            self.eval_loader = None
            self._mm_hf_model, self._mm_processor, self._mm_template_processor, self._mm_vision_device, self._mm_rope_fn = init_mm_helpers(self.config)
            self.mm_train_dataset, self.mm_eval_dataset = load_multimodal_datasets(self.config)
            pad_token_id = config.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = config.tokenizer.eos_token_id if config.tokenizer.eos_token_id is not None else 0
            global_batch = config.per_device_train_batch_size * config.gpus
            self.mm_batcher = OnTheFlyMultimodalBatcher(
                dataset=self.mm_train_dataset,
                template_processor=self._mm_template_processor,
                hf_model=self._mm_hf_model,
                vision_device=self._mm_vision_device,
                rope_fn=self._mm_rope_fn,
                batch_size=global_batch,
                seq_len=config.sequence_len,
                pad_token_id=pad_token_id,
                seed=config.train_seed,
                shuffle=True,
                repeat=True,
            )
            self.steps_per_epoch = self.mm_batcher.steps_per_epoch
        else:
            self.train_loader = _surogate.DataLoader(train_files, self.chunk_size, seed=config.train_seed)
            self.eval_loader = _surogate.DataLoader(eval_files, self.chunk_size,
                                                    seed=config.eval_seed) if eval_files else None

            # Calculate steps
            self.steps_per_epoch = self.train_loader.num_tokens // self.total_batch_size
            
        # Create trainer
        self.start_step = 0
        if config.resume_from_checkpoint:
            self.start_step = _surogate.find_latest_checkpoint(config.checkpoint_dir)
            if self.start_step >= 0:
                self.trainer = _surogate.SurogateTrainer(
                    ngpu=config.gpus,
                    config=_surogate.PretrainedConfig.from_pretrained(config.model_dir, to_surogate_dtype(config.torch_dtype)),
                    options=config.runtime_config,
                    batch_size=config.per_device_train_batch_size,
                    seq_len=config.sequence_len,
                    grad_accum=config.gradient_accumulation_steps,
                    memcpy_all_gather=config.memcpy_all_gather,
                    memcpy_send_recv=config.memcpy_send_recv,
                    lora_config=config.lora_config,
                    qlora_config=config.qlora_config
                )
                # Base model weights must be imported first to initialize the weight structure.
                # For LoRA: checkpoint only contains adapter weights + optimizer state, so we
                #           need the original base model weights.
                # For FFT/upcycle: checkpoint contains trained weights. We import from the
                #                  checkpoint's model.safetensors to handle upcycled models
                #                  where config.model_dir points to a different architecture.
                if config.lora:
                    # LoRA: use base model weights
                    base_weights_path = str(Path(config.model_dir) / "model.safetensors")
                    if not Path(base_weights_path).exists():
                        base_weights_path = config.model_dir
                else:
                    # FFT/upcycle: use checkpoint's saved weights (handles architecture changes)
                    checkpoint_dir = Path(config.checkpoint_dir) / f"step_{self.start_step:08d}"
                    checkpoint_weights = checkpoint_dir / "model.safetensors"
                    if checkpoint_weights.exists():
                        base_weights_path = str(checkpoint_weights)
                    else:
                        # Fallback to base model if checkpoint doesn't have model.safetensors
                        base_weights_path = str(Path(config.model_dir) / "model.safetensors")
                        if not Path(base_weights_path).exists():
                            base_weights_path = config.model_dir
                logger.info(f"Importing base model weights from {base_weights_path}...")
                if config.adapter_path:
                    logger.info(f"Merging adapter from {config.adapter_path} into base weights...")
                    self.trainer.set_adapter_path(config.adapter_path)
                self.trainer.import_weights(base_weights_path)
                logger.info(f"Loading checkpoint from step {self.start_step}...")
                self.trainer.load_checkpoint(str(config.checkpoint_dir), self.start_step)
            else:
                logger.warning("No checkpoint found to resume from. Starting training from beginning.")
                self.start_step = 0

        if not hasattr(self, 'trainer'):
            if config.lora and config.lora_rank and config.lora_alpha and config.lora_target_modules:
                self.trainer = _surogate.SurogateTrainer(
                    ngpu=config.gpus,
                    config=_surogate.PretrainedConfig.from_pretrained(config.model_dir, to_surogate_dtype(config.torch_dtype)),
                    options=config.runtime_config,
                    batch_size=config.per_device_train_batch_size,
                    seq_len=config.sequence_len,
                    grad_accum=config.gradient_accumulation_steps,
                    memcpy_all_gather=config.memcpy_all_gather,
                    memcpy_send_recv=config.memcpy_send_recv,
                    lora_config=config.lora_config,
                    qlora_config=config.qlora_config
                )
                if config.adapter_path:
                    logger.info(f"Merging adapter from {config.adapter_path} into base weights...")
                    self.trainer.set_adapter_path(config.adapter_path)
                self.trainer.import_weights(model_weights_path)

            elif config.from_scratch:
                self.trainer = _surogate.SurogateTrainer(
                    ngpu=config.gpus,
                    config=_surogate.PretrainedConfig.from_name(config.model_info.model_name, to_surogate_dtype(config.torch_dtype)),
                    options=config.runtime_config,
                    batch_size=config.per_device_train_batch_size,
                    seq_len=config.sequence_len,
                    grad_accum=config.gradient_accumulation_steps,
                    memcpy_all_gather=config.memcpy_all_gather,
                    memcpy_send_recv=config.memcpy_send_recv
                )
                self.trainer.init_weights()
            else:
                self.trainer = _surogate.SurogateTrainer.from_pretrained(
                    name=config.model_dir,
                    ngpu=config.gpus,
                    dtype=to_surogate_dtype(config.torch_dtype),
                    options=config.runtime_config,
                    batch_size=config.per_device_train_batch_size,
                    seq_len=config.sequence_len,
                    grad_accum=config.gradient_accumulation_steps,
                    memcpy_all_gather=config.memcpy_all_gather,
                    memcpy_send_recv=config.memcpy_send_recv
                )
                
        if self.config.from_scratch:
            # Chinchilla token budget (optimal tokens ≈ 20 × params)
            self.num_params = estimate_model_parameters(config.model_info.config)
            self.chinchilla_tokens = 20 * self.num_params
            self.tokens_per_step = self.total_batch_size

        # Determine max_steps
        if config.max_steps > 0:
            self.max_steps = config.max_steps
        elif self._mm_on_the_fly:
            if self.steps_per_epoch == 0:
                raise ValueError("multimodal_on_the_fly requires max_steps when dataset length is unknown.")
            self.max_steps = self.steps_per_epoch * self.config.num_epochs
            logger.info(f"Derived {self.max_steps} steps from {self.config.num_epochs} epoch(s)")
        elif config.epoch_adjustment and self.config.from_scratch:
            # Adjust epochs to reach Chinchilla-optimal token budget
            chinchilla_epochs = max(1, int(np.ceil(self.chinchilla_tokens / max(self.train_loader.num_tokens, 1))))
            if chinchilla_epochs != self.config.num_epochs:
                logger.info(
                    f"Epoch adjustment: {self.config.num_epochs} -> {chinchilla_epochs} epochs "
                    f"(Chinchilla budget {self.chinchilla_tokens / 1e9:.1f}B tokens, "
                    f"dataset {self.train_loader.num_tokens / 1e9:.1f}B tokens)"
                )
                self.config.num_epochs = chinchilla_epochs
            self.max_steps = self.steps_per_epoch * self.config.num_epochs
            logger.info(f"Derived {self.max_steps} steps from {self.config.num_epochs} epoch(s) (epoch_adjustment)")
        else:
            self.max_steps = self.steps_per_epoch * self.config.num_epochs
            logger.info(f"Derived {self.max_steps} steps from {self.config.num_epochs} epoch(s)")

        # Apply warmup_ratio if warmup_steps is 0
        self.warmup_steps = config.warmup_steps
        if self.warmup_steps == 0 and config.warmup_ratio > 0:
            self.warmup_steps = int(self.max_steps * config.warmup_ratio)
            logger.info(f"Derived {self.warmup_steps} warmup steps from warmup_ratio={config.warmup_ratio}")

        # Setup learning rate schedule
        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type,
            wsd_decay_steps_fraction=config.wsd_decay_steps_fraction
        )

    def _load_block_types(self) -> Optional[list]:
        if self._block_types is not None:
            return self._block_types

        try:
            from surogate.dsl.ir_builder import load_hf_config
        except Exception:
            self._block_types = None
            return None

        try:
            cfg = load_hf_config(self.config.model_dir)
        except Exception:
            self._block_types = None
            return None

        block_types = None
        if isinstance(cfg.get("layers_block_type"), list):
            block_types = [str(x).lower() for x in cfg["layers_block_type"]]
        else:
            pattern = cfg.get("hybrid_override_pattern")
            if isinstance(pattern, str) and pattern:
                mapping = {
                    "M": "mamba",
                    "*": "attention",
                    "A": "attention",
                    "-": "mlp",
                    "P": "mlp",
                    "E": "moe",
                }
                try:
                    block_types = [mapping[c] for c in pattern]
                except KeyError:
                    block_types = None

        self._block_types = block_types
        return self._block_types

    def _copy_tokenizer_files(self, src_dir: str, dst_dir: str):
        """Copy tokenizer and vocab files from source model to output directory."""
        tokenizer_files = [
            "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "vocab.json", "merges.txt"
        ]
        src_path = Path(src_dir)
        dst_path = Path(dst_dir)
        for filename in tokenizer_files:
            src = src_path / filename
            if src.exists():
                shutil.copy(src, dst_path / filename)
                logger.info(f"Copied {filename}")

    def train(self):
        with training_logger_context(self.config) as train_logger:
            # Log dataset information
            if not self._mm_on_the_fly and self.eval_loader:
                train_logger.log_dataset(self.train_loader, self.eval_loader)
            elif self._mm_on_the_fly:
                try:
                    train_len = len(self.mm_train_dataset)
                except Exception:
                    train_len = None
                try:
                    eval_len = len(self.mm_eval_dataset) if self.mm_eval_dataset is not None else None
                except Exception:
                    eval_len = None
                if train_len is not None:
                    logger.info(f"Multimodal train dataset: {train_len} samples")
                if eval_len is not None:
                    logger.info(f"Multimodal eval dataset: {eval_len} samples")

            # Log allocator stats
            for idx in range(self.config.gpus):
                alloc_info = self.trainer.get_allocator_info(idx)
                train_logger.log_allocator(alloc_info)

            # Calculate expected time per token for speed-of-light estimation
            train_logger.set_expected_time_per_token(self.trainer)

            # Print training info
            logger.info(f"Starting training from step {self.start_step}...")
            logger.info(f"Recipe: {self.config.recipe}")
            logger.info(f"Optimizer: {self.config.optimizer}")
            logger.info(f"Total batch size: {self.total_batch_size} tokens")
            logger.info(f"Steps per epoch: {self.steps_per_epoch}")
            logger.info(f"Max steps: {self.max_steps}")
            logger.info(
                f"LR schedule: {self.config.lr_scheduler_type} (warmup={self.warmup_steps}, cooldown={self.config.cooldown_steps})")
            
            if self.config.from_scratch:
                # Chinchilla token budget
                planned_tokens = self.max_steps * self.tokens_per_step
                ratio = planned_tokens / max(self.chinchilla_tokens, 1)
                def _fmt(n):
                    if n >= 1e12: return f"{n/1e12:.1f}T"
                    if n >= 1e9: return f"{n/1e9:.1f}B"
                    if n >= 1e6: return f"{n/1e6:.1f}M"
                    return f"{n/1e3:.1f}K"
                logger.info(
                    f"Chinchilla budget: {_fmt(self.chinchilla_tokens)} tokens (20 × {_fmt(self.num_params)} params) | "
                    f"Planned: {_fmt(planned_tokens)} tokens ({ratio:.1%} of budget)"
                )

            # Print LoRA info if enabled
            if self.config.lora and self.config.lora_config:
                logger.info(f"LoRA enabled:")
                logger.info(f"  Rank: {self.config.lora_config.rank}")
                logger.info(f"  Alpha: {self.config.lora_config.alpha}")
                logger.info(f"  Scaling: {self.config.lora_config.scaling:.4f}")
                logger.info(f"  DType: {self.config.lora_dtype}")
                logger.info(f"  Target modules: {self.config.lora_config.target_modules}")
                if self.config.qlora_fp8:
                    logger.info(f"  QLoRA-FP8 enabled: block_size={self.config.qlora_block_size}")
                elif self.config.qlora_fp4:
                    logger.info("  QLoRA-FP4 enabled: NVFP4 (E2M1)")
                logger.info("Note: Base model weights are frozen, only LoRA adapters will be trained")

            if self._mm_on_the_fly:
                self.run_training_loop_mm(train_logger)
            else:
                self.run_training_loop(train_logger)

            # Save final model
            if self.config.lora:
                # Export LoRA adapter in PEFT-compatible format
                adapter_dir = Path(self.config.output_dir)
                logger.info(f"Saving LoRA adapter to {adapter_dir}...")
                adapter_dir.mkdir(parents=True, exist_ok=True)
                self.trainer.export_adapter(str(adapter_dir))
                logger.info("done")
                logger.info(f"LoRA adapter saved to {adapter_dir}")

                # Merge adapter into base model if requested
                if self.config.merge_adapter:
                    merged_dir = Path(self.config.output_dir)
                    try:
                        merge_adapter(
                            base_model_path=self.config.model_dir,
                            adapter_path=str(adapter_dir),
                            output_path=str(merged_dir),
                            max_shard_size="5GB",
                            cpu_offload=True
                        )
                        # Generate training plot in merged directory
                        generate_training_plot(self.config.log_file, merged_dir / "training_plot.png")
                    except Exception as e:
                        logger.error(f"Failed to merge adapter: {e}")
                        import traceback
                        logger.error(f"Traceback:\n{traceback.format_exc()}")
                        logger.warning("Adapter merge failed, but adapter was saved successfully")
                else:
                    logger.info("To use with HuggingFace PEFT, load the base model and apply this adapter.")

                # Generate training plot in adapter directory
                generate_training_plot(self.config.log_file, adapter_dir / "training_plot.png")
            else:
                logger.info(f"Saving model to {self.config.output_dir}...")
                self.trainer.export_model(str(self.config.output_dir))
                # Copy tokenizer files from source model
                self._copy_tokenizer_files(self.config.model_dir, self.config.output_dir)
                logger.info("done")
                # Generate training plot in output directory
                generate_training_plot(self.config.log_file, Path(self.config.output_dir) / "training_plot.png")

            logger.info(f"\nTraining complete! Logs saved to {self.config.log_file}")

    def run_training_loop_mm(self, train_logger: _surogate.TrainingRunLogger):
        use_full_step_graphs = False
        if self.config.use_cuda_graphs:
            logger.info("CUDA graphs disabled for multimodal on-the-fly training.")

        # Auto LR reduction guard
        loss_guard = LossGuard(self.lr_schedule, logger) if self.config.auto_lr_reduction else None
        plateau_detector = PlateauDetector(logger)
        phase_detector = PhaseDetector(logger)
        gradient_tracker = GradientTracker(logger)
        moe_monitor = MoEMonitor(
            logger,
            num_experts=self.config.moe_num_experts,
            num_experts_per_tok=self.config.moe_num_experts_per_tok,
        )
        advisor = TrainingAdvisor(
            logger, phase_detector, gradient_tracker, plateau_detector,
            loss_guard, moe_monitor, self.lr_schedule, self.max_steps,
            warmup_steps=self.warmup_steps,
        )

        # Early stopping
        if self.config.early_stop:
            num_params = estimate_model_parameters(self.config.model_info.config)
            tokens_per_step = self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gradient_accumulation_steps * self.config.gpus
            early_stopping = EarlyStopping(logger, num_params, tokens_per_step)
        else:
            early_stopping = None

        # Training loop
        logger.info(f"Starting training loop: steps {self.start_step} to {self.max_steps - 1}")
        for step in range(self.start_step, self.max_steps):
            # Periodic evaluation (before training step)
            if self.mm_eval_dataset is not None and self.config.eval_steps > 0 and step % self.config.eval_steps == 0 and step > self.start_step:
                val_loss, elapsed_ms, batches_processed = self.run_evaluation_mm(max_steps=100)
                epoch = self.mm_batcher.epoch() + 0.01 * self.mm_batcher.progress()
                eval_tokens = batches_processed * self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gpus
                train_logger.log_eval(step, epoch, eval_tokens, elapsed_ms, val_loss)
                if early_stopping is not None and early_stopping.check_eval(val_loss, step):
                    break

            # Periodic checkpointing (before training step)
            if self.config.save_steps > 0 and step % self.config.save_steps == 0 and step > self.start_step:
                logger.info(f"Saving checkpoint to {self.config.checkpoint_dir}...")
                try:
                    self.trainer.save_checkpoint(self.config.checkpoint_dir, step)
                    logger.info(f"Checkpoint saved successfully at step {step}")

                    checkpoint_plot_path = Path(self.config.checkpoint_dir) / f"step_{step:08d}" / "training_plot.png"
                    generate_training_plot(self.config.log_file, checkpoint_plot_path)

                    if self.config.save_total_limit > 0:
                        removed = _surogate.clean_old_checkpoints(self.config.checkpoint_dir, self.config.save_total_limit,
                                                                  -1)
                        if removed:
                            logger.info(
                                f"Removed {removed} old checkpoints, keeping the most recent {self.config.save_total_limit}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step}: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.warning("Training will continue without saving this checkpoint")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Training step
            step_start = time.time()
            for micro_step in range(self.config.gradient_accumulation_steps):
                batch = self.mm_batcher.next_batch()
                self.trainer.set_visual_inputs(
                    batch["visual_pos_masks"],
                    batch["visual_embeds"],
                    batch["deepstack_visual_embeds"],
                )
                self.trainer.step(batch["inputs"], batch["targets"], batch["position_ids"])

            # Log GPU utilization
            if self.config.log_gpu_util > 0 and step % self.config.log_gpu_util == 0:
                infos = self.trainer.get_gpu_info()
                for i, info in enumerate(infos):
                    train_logger.log_gpu_state(step, i, info)

            lr = self.lr_schedule.get_lr(step)
            opt_config = _surogate.OptimizerConfig(
                optimizer=self.config.optimizer,
                learning_rate=lr,
                weight_decay=self.config.weight_decay,
                grad_clip=self.config.max_grad_norm,
                adamw_beta1=self.config.adamw_beta1,
                adamw_beta2=self.config.adamw_beta2,
                adamw_epsilon=self.config.adamw_epsilon,
                normuon_momentum=self.config.normuon_momentum,
                normuon_beta2=self.config.normuon_beta2,
                normuon_lr=lr,
                normuon_cautious_wd=self.config.normuon_cautious_wd
            )

            self._maybe_log_lora_grad_stats(step)
            result = self.trainer.update_with_config(opt_config, step + 1)

            step_time = time.time() - step_start
            tokens_processed = self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gradient_accumulation_steps * self.config.gpus

            if loss_guard is not None:
                loss_guard.step(result['loss'], result['norm'], step)
            plateau_detector.step(result['loss'], step)
            phase = phase_detector.step(result['loss'], step)
            gradient_tracker.step(result['norm'], step)
            train_logger.set_phase(phase.value)

            metrics = StepMetrics(
                step=step,
                epoch=self.mm_batcher.epoch() + 0.01 * self.mm_batcher.progress(),
                loss=result['loss'],
                grad_norm=result['norm'],
                grad_norm_mean=gradient_tracker.mean,
                grad_norm_max=gradient_tracker.max,
                grad_norm_trend=gradient_tracker.trend,
                lr=lr,
                tokens=tokens_processed,
                elapsed_ms=int(step_time * 1000),
                phase=phase.value,
                lr_overridden=self.lr_schedule.has_override,
                moe=MoEMetrics.from_dict(self.trainer.get_moe_stats()),
            )
            moe_monitor.step(metrics.moe, step)
            advisor.step(metrics, step)

            if early_stopping is not None and early_stopping.check_step(metrics.loss, phase, step):
                break

            if step % self.config.logging_steps == 0:
                if metrics.moe is not None:
                    train_logger.log_step_moe(metrics.step, metrics.epoch, metrics.tokens, metrics.elapsed_ms,
                                              metrics.grad_norm, metrics.loss, metrics.lr,
                                              metrics.moe.aux_loss,
                                              metrics.moe.z_loss,
                                              metrics.moe.load_imbalance,
                                              metrics.moe.expert_utilization)
                else:
                    train_logger.log_step(metrics.step, metrics.epoch, metrics.tokens, metrics.elapsed_ms,
                                          metrics.grad_norm, metrics.loss, metrics.lr)

        logger.info(f"Training loop completed successfully after step {self.max_steps - 1}")

    def run_evaluation_mm(self, max_steps: int) -> Tuple[float, int, int]:
        if max_steps == 0:
            return 0.0, 0, 0
        if self.mm_eval_dataset is None:
            return 0.0, 0, 0

        pad_token_id = self.config.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.config.tokenizer.eos_token_id if self.config.tokenizer.eos_token_id is not None else 0

        global_batch = self.config.per_device_train_batch_size * self.config.gpus
        eval_batcher = OnTheFlyMultimodalBatcher(
            dataset=self.mm_eval_dataset,
            template_processor=self._mm_template_processor,
            hf_model=self._mm_hf_model,
            vision_device=self._mm_vision_device,
            rope_fn=self._mm_rope_fn,
            batch_size=global_batch,
            seq_len=self.config.sequence_len,
            pad_token_id=pad_token_id,
            seed=self.config.eval_seed,
            shuffle=False,
            repeat=False,
        )

        start_time = time.time()
        total_loss = 0.0
        batches = 0

        while max_steps < 0 or batches < max_steps:
            try:
                batch = eval_batcher.next_batch()
            except StopIteration:
                break
            self.trainer.set_visual_inputs(
                batch["visual_pos_masks"],
                batch["visual_embeds"],
                batch["deepstack_visual_embeds"],
            )
            loss = self.trainer.validate(batch["inputs"], batch["targets"], batch["position_ids"])
            total_loss += loss
            batches += 1

        if batches == 0:
            logger.warning("Insufficient validation data")
            return 0.0, 0, 0

        return total_loss / batches, int((time.time() - start_time) * 1000), batches

    def run_training_loop(self, train_logger: _surogate.TrainingRunLogger):
        use_full_step_graphs = True
        if use_full_step_graphs and self.config.optimizer not in ("adamw_8bit", "normuon"):
            raise RuntimeError("DSL training requires optimizer 'adamw_8bit' or 'normuon' for full-step execution.")
        if use_full_step_graphs and not self.config.use_cuda_graphs:
            logger.info("CUDA graphs disabled")

        # Allocate token buffers
        micro_steps = self.config.gradient_accumulation_steps if use_full_step_graphs else 1
        total_rows = self.config.gpus * self.config.per_device_train_batch_size * micro_steps
        in_tokens = np.empty((total_rows, self.config.sequence_len), dtype=np.int32)
        out_tokens = np.empty((total_rows, self.config.sequence_len), dtype=np.int32)
        pos_ids = np.empty((total_rows, self.config.sequence_len), dtype=np.int32)

        # Preload first batch (eager path only)
        if not use_full_step_graphs:
            self.train_loader.load_batch(in_tokens, out_tokens, pos_ids)

        # Auto LR reduction guard
        loss_guard = LossGuard(self.lr_schedule, logger) if self.config.auto_lr_reduction else None
        plateau_detector = PlateauDetector(logger)
        phase_detector = PhaseDetector(logger)
        gradient_tracker = GradientTracker(logger)
        moe_monitor = MoEMonitor(
            logger,
            num_experts=self.config.moe_num_experts,
            num_experts_per_tok=self.config.moe_num_experts_per_tok,
        )
        advisor = TrainingAdvisor(
            logger, phase_detector, gradient_tracker, plateau_detector,
            loss_guard, moe_monitor, self.lr_schedule, self.max_steps,
            warmup_steps=self.warmup_steps,
        )

        # Early stopping
        if self.config.early_stop:
            from surogate.utils.model import estimate_model_parameters
            num_params = estimate_model_parameters(self.config.model_info.config)
            tokens_per_step = self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gradient_accumulation_steps * self.config.gpus
            early_stopping = EarlyStopping(logger, num_params, tokens_per_step)
        else:
            early_stopping = None

        # Training loop
        logger.info(f"Starting training loop: steps {self.start_step} to {self.max_steps - 1}")
        for step in range(self.start_step, self.max_steps):
            # Check if we need to advance epoch
            if not self.train_loader.has_next(self.config.gradient_accumulation_steps):
                self.train_loader.advance_epoch()
                if not use_full_step_graphs:
                    self.train_loader.load_batch(in_tokens, out_tokens, pos_ids)

            # Periodic evaluation (before training step)
            if self.eval_loader and self.config.eval_steps > 0 and step % self.config.eval_steps == 0 and step > self.start_step:
                # Limit periodic eval to 100 batches for speed; full eval runs at end of training
                if use_full_step_graphs:
                    chunk = self.config.gpus * self.config.per_device_train_batch_size
                    val_loss, elapsed_ms, batches_processed = self.run_evaluation(
                        in_tokens[:chunk], out_tokens[:chunk], pos_ids[:chunk], max_steps=100
                    )
                else:
                    val_loss, elapsed_ms, batches_processed = self.run_evaluation(in_tokens, out_tokens, pos_ids, max_steps=100)
                epoch = self.train_loader.epoch() + 0.01 * self.train_loader.progress()
                # Calculate actual tokens processed based on batches run
                # Note: eval uses same batch size as training (per_device_train_batch_size) since buffers are shared
                eval_tokens = batches_processed * self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gpus
                train_logger.log_eval(step, epoch, eval_tokens, elapsed_ms, val_loss)
                if early_stopping is not None and early_stopping.check_eval(val_loss, step):
                    break
                # Reload training batch after evaluation (eval leaves its last batch in the buffers)
                if use_full_step_graphs:
                    chunk = self.config.gpus * self.config.per_device_train_batch_size
                    self.train_loader.load_batch(in_tokens[:chunk], out_tokens[:chunk], pos_ids[:chunk])
                else:
                    self.train_loader.load_batch(in_tokens, out_tokens, pos_ids)

            # Periodic checkpointing (before training step)
            if self.config.save_steps > 0 and step % self.config.save_steps == 0 and step > self.start_step:
                logger.info(f"Saving checkpoint to {self.config.checkpoint_dir}...")
                try:
                    self.trainer.save_checkpoint(self.config.checkpoint_dir, step)
                    logger.info(f"Checkpoint saved successfully at step {step}")

                    # Generate training plot in checkpoint directory
                    checkpoint_plot_path = Path(self.config.checkpoint_dir) / f"step_{step:08d}" / "training_plot.png"
                    generate_training_plot(self.config.log_file, checkpoint_plot_path)

                    # Clean old checkpoints
                    if self.config.save_total_limit > 0:
                        removed = _surogate.clean_old_checkpoints(self.config.checkpoint_dir, self.config.save_total_limit,
                                                                  -1)
                        if removed:
                            logger.info(
                                f"Removed {removed} old checkpoints, keeping the most recent {self.config.save_total_limit}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint at step {step}: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.warning("Training will continue without saving this checkpoint")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")

            # Training step
            step_start = time.time()

            if use_full_step_graphs:
                chunk = self.config.gpus * self.config.per_device_train_batch_size
                for micro_step in range(self.config.gradient_accumulation_steps):
                    if not self.train_loader.has_next():
                        self.train_loader.advance_epoch()
                    start = micro_step * chunk
                    end = start + chunk
                    self.train_loader.load_batch(in_tokens[start:end], out_tokens[start:end], pos_ids[start:end])
            else:
                for micro_step in range(self.config.gradient_accumulation_steps):
                    self.trainer.step(in_tokens, out_tokens, pos_ids)
                    if self.train_loader.has_next():
                        self.train_loader.load_batch(in_tokens, out_tokens, pos_ids)

            # Log GPU utilization
            if self.config.log_gpu_util > 0 and step % self.config.log_gpu_util == 0:
                infos = self.trainer.get_gpu_info()
                for i, info in enumerate(infos):
                    train_logger.log_gpu_state(step, i, info)

            # Optimizer update
            lr = self.lr_schedule.get_lr(step)

            # Create optimizer config based on selected optimizer
            opt_config = _surogate.OptimizerConfig(
                optimizer=self.config.optimizer,
                learning_rate=lr,
                weight_decay=self.config.weight_decay,
                grad_clip=self.config.max_grad_norm,
                adamw_beta1=self.config.adamw_beta1,
                adamw_beta2=self.config.adamw_beta2,
                adamw_epsilon=self.config.adamw_epsilon,
                normuon_momentum=self.config.normuon_momentum,
                normuon_beta2=self.config.normuon_beta2,
                normuon_lr=lr,  # Use same LR for NorMuon
                normuon_cautious_wd=self.config.normuon_cautious_wd
            )
            if use_full_step_graphs:
                result = self.trainer.train_step_graphed(in_tokens, out_tokens, pos_ids, opt_config, step + 1)
                # Optional LoRA grad debug runs after graph replay (post-update).
                self._maybe_log_lora_grad_stats(step)
            else:
                # Optional LoRA gradient debug (before optimizer update)
                self._maybe_log_lora_grad_stats(step)
                result = self.trainer.update_with_config(opt_config, step + 1)

            # Build structured metrics for this step
            step_time = time.time() - step_start
            tokens_processed = self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gradient_accumulation_steps * self.config.gpus

            # Check for loss spikes / gradient explosions
            if loss_guard is not None:
                loss_guard.step(result['loss'], result['norm'], step)
            plateau_detector.step(result['loss'], step)
            phase = phase_detector.step(result['loss'], step)
            gradient_tracker.step(result['norm'], step)
            train_logger.set_phase(phase.value)

            metrics = StepMetrics(
                step=step,
                epoch=self.train_loader.epoch() + 0.01 * self.train_loader.progress(),
                loss=result['loss'],
                grad_norm=result['norm'],
                grad_norm_mean=gradient_tracker.mean,
                grad_norm_max=gradient_tracker.max,
                grad_norm_trend=gradient_tracker.trend,
                lr=lr,
                tokens=tokens_processed,
                elapsed_ms=int(step_time * 1000),
                phase=phase.value,
                lr_overridden=self.lr_schedule.has_override,
                moe=MoEMetrics.from_dict(self.trainer.get_moe_stats()),
            )
            moe_monitor.step(metrics.moe, step)
            advisor.step(metrics, step)

            if early_stopping is not None and early_stopping.check_step(metrics.loss, phase, step):
                break

            # Log training step
            if step % self.config.logging_steps == 0:
                if metrics.moe is not None:
                    train_logger.log_step_moe(metrics.step, metrics.epoch, metrics.tokens, metrics.elapsed_ms,
                                              metrics.grad_norm, metrics.loss, metrics.lr,
                                              metrics.moe.aux_loss,
                                              metrics.moe.z_loss,
                                              metrics.moe.load_imbalance,
                                              metrics.moe.expert_utilization)
                else:
                    train_logger.log_step(metrics.step, metrics.epoch, metrics.tokens, metrics.elapsed_ms,
                                          metrics.grad_norm, metrics.loss, metrics.lr)

        logger.info(f"Training loop completed successfully after step {self.max_steps - 1}")

    def _maybe_log_lora_grad_stats(self, step: int) -> None:
        if not self.config.lora:
            return
        if os.environ.get("SUROGATE_DEBUG_LORA_GRADS", "0") not in {"1", "true", "True", "yes", "YES"}:
            return
        try:
            every = int(os.environ.get("SUROGATE_DEBUG_LORA_GRADS_EVERY", "1"))
        except ValueError:
            every = 1
        if every <= 0 or step % every != 0:
            return

        try:
            import torch
        except Exception as exc:  # pragma: no cover - debug-only
            logger.warning(f"LoRA grad debug requested but torch unavailable: {exc}")
            return

        try:
            grads = self.trainer.get_lora_gradients(0)
        except Exception as exc:
            logger.warning(f"Failed to fetch LoRA gradients for debug: {exc}")
            return

        total_sq = None
        max_abs = None
        any_nan = False
        tensor_count = 0
        module_stats = {}
        layer_stats = {}
        proj_stats = {}
        try:
            topk = int(os.environ.get("SUROGATE_DEBUG_LORA_GRADS_TOPK", "10"))
        except ValueError:
            topk = 10

        for name, arr in grads.items():
            try:
                t = torch.utils.dlpack.from_dlpack(arr)
            except Exception:
                # Fallback: try as_tensor (may copy to CPU)
                t = torch.as_tensor(arr)
            if t.numel() == 0:
                continue
            tensor_count += 1
            t_f = t.float()
            has_nan = torch.isnan(t_f).any().item()
            has_inf = torch.isinf(t_f).any().item()
            if has_nan or has_inf:
                any_nan = True
                nan_count = torch.isnan(t_f).sum().item()
                inf_count = torch.isinf(t_f).sum().item()
                logger.info(
                    "  NaN/Inf in LoRA grad '%s' shape=%s nan=%d inf=%d abs_max=%.6g",
                    name, list(t.shape), nan_count, inf_count,
                    t_f[~torch.isnan(t_f)].abs().max().item() if nan_count < t_f.numel() else float('nan')
                )
            sq = (t_f * t_f).sum()
            total_sq = sq if total_sq is None else total_sq + sq
            t_max = t_f.abs().max()
            max_abs = t_max if max_abs is None else torch.maximum(max_abs, t_max)

            module_name = name.rsplit(".lora_", 1)[0] if ".lora_" in name else name
            mstat = module_stats.get(module_name)
            if mstat is None:
                module_stats[module_name] = {
                    "sq": sq.item(),
                    "max_abs": t_max.item(),
                    "tensors": 1,
                    "nan": bool(has_nan or has_inf),
                }
            else:
                mstat["sq"] += sq.item()
                mstat["max_abs"] = max(mstat["max_abs"], t_max.item())
                mstat["tensors"] += 1
                if has_nan or has_inf:
                    mstat["nan"] = True

            proj_name = module_name.rsplit(".", 1)[-1]
            if proj_name in {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"}:
                pstat = proj_stats.get(proj_name)
                if pstat is None:
                    proj_stats[proj_name] = {
                        "sq": sq.item(),
                        "max_abs": t_max.item(),
                        "tensors": 1,
                        "nan": bool(has_nan or has_inf),
                    }
                else:
                    pstat["sq"] += sq.item()
                    pstat["max_abs"] = max(pstat["max_abs"], t_max.item())
                    pstat["tensors"] += 1
                    if has_nan or has_inf:
                        pstat["nan"] = True

            layer_match = re.search(r"\.layers\.(\d+)\.", module_name)
            if layer_match:
                layer_idx = int(layer_match.group(1))
                lstat = layer_stats.get(layer_idx)
                if lstat is None:
                    layer_stats[layer_idx] = {
                        "sq": sq.item(),
                        "max_abs": t_max.item(),
                        "modules": {module_name},
                        "nan": bool(has_nan or has_inf),
                    }
                else:
                    lstat["sq"] += sq.item()
                    lstat["max_abs"] = max(lstat["max_abs"], t_max.item())
                    lstat["modules"].add(module_name)
                    if has_nan or has_inf:
                        lstat["nan"] = True

        if total_sq is None:
            logger.info("LoRA grad debug: no gradients found")
            return

        total_norm = total_sq.sqrt().item()
        max_abs_val = max_abs.item() if max_abs is not None else float("nan")
        logger.info(
            "LoRA grad debug: step=%d tensors=%d norm=%.6g max_abs=%.6g nan=%s",
            step, tensor_count, total_norm, max_abs_val, any_nan
        )

        if module_stats:
            sorted_modules = sorted(module_stats.items(), key=lambda kv: kv[1]["sq"], reverse=True)
            for module_name, stat in sorted_modules[:topk]:
                logger.info(
                    "  LoRA grad module: %s norm=%.6g max_abs=%.6g tensors=%d nan=%s",
                    module_name,
                    math.sqrt(stat["sq"]),
                    stat["max_abs"],
                    stat["tensors"],
                    stat["nan"],
                )

        if layer_stats:
            block_types = self._load_block_types()
            sorted_layers = sorted(layer_stats.items(), key=lambda kv: kv[1]["sq"], reverse=True)
            for layer_idx, stat in sorted_layers[:topk]:
                logger.info(
                    "  LoRA grad layer: %d norm=%.6g max_abs=%.6g modules=%d nan=%s%s",
                    layer_idx,
                    math.sqrt(stat["sq"]),
                    stat["max_abs"],
                    len(stat["modules"]),
                    stat["nan"],
                    f" type={block_types[layer_idx]}" if block_types and layer_idx < len(block_types) else "",
                )

            if block_types:
                block_stats = {}
                for layer_idx, stat in layer_stats.items():
                    if layer_idx < len(block_types):
                        block = block_types[layer_idx]
                    else:
                        block = "unknown"
                    entry = block_stats.get(block)
                    if entry is None:
                        entry = {
                            "sq": 0.0,
                            "max_layer_sq": 0.0,
                            "max_abs": 0.0,
                            "layers": 0,
                            "nan": False,
                        }
                        block_stats[block] = entry
                    entry["sq"] += stat["sq"]
                    entry["max_layer_sq"] = max(entry["max_layer_sq"], stat["sq"])
                    entry["max_abs"] = max(entry["max_abs"], stat["max_abs"])
                    entry["layers"] += 1
                    if stat["nan"]:
                        entry["nan"] = True

                sorted_blocks = sorted(block_stats.items(), key=lambda kv: kv[1]["sq"], reverse=True)
                for block, stat in sorted_blocks:
                    logger.info(
                        "  LoRA grad block: %s norm=%.6g max_layer_norm=%.6g max_abs=%.6g layers=%d nan=%s",
                        block,
                        math.sqrt(stat["sq"]),
                        math.sqrt(stat["max_layer_sq"]),
                        stat["max_abs"],
                        stat["layers"],
                        stat["nan"],
                    )

        if proj_stats:
            sorted_projs = sorted(proj_stats.items(), key=lambda kv: kv[1]["sq"], reverse=True)
            for proj_name, stat in sorted_projs:
                logger.info(
                    "  LoRA grad proj: %s norm=%.6g max_abs=%.6g tensors=%d nan=%s",
                    proj_name,
                    math.sqrt(stat["sq"]),
                    stat["max_abs"],
                    stat["tensors"],
                    stat["nan"],
                )

    def run_evaluation(self, in_tokens: np.ndarray, out_tokens: np.ndarray, pos_ids: np.ndarray, max_steps: int) -> Tuple[float, int, int]:
        """
        Run evaluation on test set.
        Args:
            in_tokens (np.ndarray): Input token buffer.
            out_tokens (np.ndarray): Output token buffer.
            pos_ids (np.ndarray): Position id buffer.
            max_steps (int): Maximum number of eval batches to process. Pass -1 to process all available batches.
        Returns:
            Tuple of (mean_loss, elapsed_ms, batches_processed)
        """
        if max_steps == 0:
            return 0.0, 0, 0

        start_time = time.time()
        self.eval_loader.set_state(self.eval_loader.seed, 0, 0, 0)
        total_loss = 0.0
        batches = 0

        # Use has_next() to check data availability (matches C++ implementation)
        # max_steps < 0 means process all available batches
        while self.eval_loader.has_next() and (max_steps < 0 or batches < max_steps):
            self.eval_loader.load_batch(in_tokens, out_tokens, pos_ids)
            loss = self.trainer.validate(in_tokens, out_tokens, pos_ids)
            total_loss += loss
            batches += 1

        if batches == 0:
            logger.warning("Insufficient validation data")
            return 0.0, 0, 0

        return total_loss / batches, int((time.time() - start_time) * 1000), batches
