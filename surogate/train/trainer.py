import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from surogate import _surogate
from surogate.core.config.sft_config import SFTConfig
from surogate.train.lr_schedule import LRSchedule
from surogate.train.reporter import training_logger_context
from surogate.train.training_plot import generate_training_plot
from surogate.utils.hf import get_model_weights_path
from surogate.utils.logger import get_logger
from surogate.utils.system_info import get_system_info, print_system_diagnostics
from surogate.utils.tensor import to_surogate_dtype

logger = get_logger()


class SurogateTrainerWrapper():
    def __init__(
            self,
            config: SFTConfig,
            train_files: List[str],
            eval_files: Optional[List[str]] = None
    ):
        self.config = config

        model_weights_path = get_model_weights_path(config.model_dir)
        
        # Setup data loaders
        self.total_batch_size = config.per_device_train_batch_size * config.sequence_len * config.gpus * config.gradient_accumulation_steps
        self.chunk_size = config.per_device_train_batch_size * config.sequence_len * config.gpus

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
                logger.info(f"Loading checkpoint from step {self.start_step}...")
                self.trainer.load_checkpoint(str(config.checkpoint_dir), self.start_step)
            else:
                logger.error("No checkpoint found to resume from.")
                sys.exit(1)
        elif config.lora and config.lora_rank and config.lora_alpha and config.lora_target_modules:
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

        # Determine max_steps
        if config.max_steps > 0:
            self.max_steps = config.max_steps
        else:
            self.max_steps = self.steps_per_epoch * self.config.num_epochs
            logger.info(f"Derived {self.max_steps} steps from {self.config.num_epochs} epoch(s)")

        # Setup learning rate schedule
        self.lr_schedule = LRSchedule(
            base_lr=config.learning_rate,
            max_steps=self.max_steps,
            warmup_steps=config.warmup_steps,
            cooldown_steps=config.cooldown_steps,
            final_lr=config.learning_rate * config.final_lr_fraction,
            schedule_type=config.lr_scheduler_type
        )

    def train(self):
        with training_logger_context(self.config) as train_logger:
            # Log dataset information
            if self.eval_loader:
                train_logger.log_dataset(self.train_loader, self.eval_loader)

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
                f"LR schedule: {self.config.lr_scheduler_type} (warmup={self.config.warmup_steps}, cooldown={self.config.cooldown_steps})")

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

            self.run_training_loop(train_logger)

            # Save final model
            if self.config.lora:
                # Export LoRA adapter in PEFT-compatible format
                adapter_dir = Path(self.config.output_dir) / "adapter"
                logger.info(f"Saving LoRA adapter to {adapter_dir}...")
                adapter_dir.mkdir(parents=True, exist_ok=True)
                self.trainer.export_adapter(str(adapter_dir))
                logger.info("done")
                logger.info(f"LoRA adapter saved to {adapter_dir}")
                logger.info("To use with HuggingFace PEFT, load the base model and apply this adapter.")
                # Generate training plot in adapter directory
                generate_training_plot(self.config.log_file, adapter_dir / "training_plot.png")
            else:
                logger.info(f"Saving model to {self.config.output_dir}...")
                self.trainer.export_model(str(self.config.output_dir))
                logger.info("done")
                # Generate training plot in output directory
                generate_training_plot(self.config.log_file, Path(self.config.output_dir) / "training_plot.png")

            logger.info(f"\nTraining complete! Logs saved to {self.config.log_file}")

    def run_training_loop(self, train_logger: _surogate.TrainingRunLogger):
        # Allocate token buffers
        in_tokens = np.empty((self.config.gpus * self.config.per_device_train_batch_size, self.config.sequence_len),
                             dtype=np.int32)
        out_tokens = np.empty((self.config.gpus * self.config.per_device_train_batch_size, self.config.sequence_len),
                              dtype=np.int32)

        # Preload first batch
        self.train_loader.load_batch(in_tokens, out_tokens)

        # Training loop
        for step in range(self.start_step, self.max_steps):
            # Check if we need to advance epoch
            if not self.train_loader.has_next(self.config.gradient_accumulation_steps):
                self.train_loader.advance_epoch()
                self.train_loader.load_batch(in_tokens, out_tokens)

            # Periodic evaluation (before training step)
            if self.eval_loader and self.config.eval_steps > 0 and step % self.config.eval_steps == 0 and step > self.start_step:
                # Limit periodic eval to 100 batches for speed; full eval runs at end of training
                val_loss, elapsed_ms, batches_processed = self.run_evaluation(in_tokens, out_tokens, max_steps=100)
                epoch = self.train_loader.epoch() + 0.01 * self.train_loader.progress()
                # Calculate actual tokens processed based on batches run
                # Note: eval uses same batch size as training (per_device_train_batch_size) since buffers are shared
                eval_tokens = batches_processed * self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gpus
                train_logger.log_eval(step, epoch, eval_tokens, elapsed_ms, val_loss)
                # Reload training batch after evaluation (eval leaves its last batch in the buffers)
                self.train_loader.load_batch(in_tokens, out_tokens)

            # Periodic checkpointing (before training step)
            if self.config.save_steps > 0 and step % self.config.save_steps == 0 and step > self.start_step:
                logger.info(f"Saving checkpoint to {self.config.checkpoint_dir}...")
                self.trainer.save_checkpoint(self.config.checkpoint_dir, step)

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

            # Training step
            step_start = time.time()

            for micro_step in range(self.config.gradient_accumulation_steps):
                self.trainer.step(in_tokens, out_tokens)
                if self.train_loader.has_next():
                    self.train_loader.load_batch(in_tokens, out_tokens)

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
            result = self.trainer.update_with_config(opt_config, step + 1)

            step_time = time.time() - step_start
            elapsed_ms = int(step_time * 1000)

            # Log training step
            tokens_processed = self.config.per_device_train_batch_size * self.config.sequence_len * self.config.gradient_accumulation_steps * self.config.gpus
            epoch = self.train_loader.epoch() + 0.01 * self.train_loader.progress()
            train_logger.log_step(step, epoch, tokens_processed, elapsed_ms,
                                  result['norm'], result['loss'], lr)

    def run_evaluation(self, in_tokens: np.ndarray, out_tokens: np.ndarray, max_steps: int) -> Tuple[float, int, int]:
        """
        Run evaluation on test set.
        Args:
            in_tokens (np.ndarray): Input token buffer.
            out_tokens (np.ndarray): Output token buffer.
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
            self.eval_loader.load_batch(in_tokens, out_tokens)
            loss = self.trainer.validate(in_tokens, out_tokens)
            total_loss += loss
            batches += 1

        if batches == 0:
            logger.warning("Insufficient validation data")
            return 0.0, 0, 0

        return total_loss / batches, int((time.time() - start_time) * 1000), batches
