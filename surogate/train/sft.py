from pathlib import Path

import datasets

from surogate.core.config.sft_config import SFTConfig
from surogate.train.tokenize import TokenizeDatasets
from surogate.train.trainer import SurogateTrainerWrapper
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

datasets.logging.set_verbosity_warning()

logger = get_logger()


class SurogateSFT(TokenizeDatasets):
    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args)

    def run(self):
        # Check if distributed training is configured
        # In distributed mode, each worker node tokenizes its own shard of the data
        if self.config.distributed and self.config.distributed.num_nodes >= 1:
            logger.info("Distributed mode: each node will process its own data shard")
            logger.info(f"Starting training run '{self.config.run_name}'...")
            return self._train_distributed()

        if self.config.train_vision and self.config.is_multimodal:
            logger.info("Vision training enabled; skipping tokenization.")
            logger.info(f"Starting training run '{self.config.run_name}'...")
            return self.train_with_oom_recovery([], [])

        # Single-node mode: tokenize datasets on the driver
        super().run()

        # Setup data loaders
        output_path = Path(self.config.output_dir)
        train_files = sorted([str(p) for p in output_path.glob("train-*.bin")])
        eval_files = sorted([str(p) for p in output_path.glob("eval*.bin")])

        if not train_files:
            logger.error(f"No training files found matching '{self.config.output_dir}/train-*.bin'")
            return
        if not eval_files:
            logger.warning(f"No eval files found matching '{self.config.output_dir}/eval*.bin'")

        logger.info(f"Starting training run '{self.config.run_name}'...")

        return self.train_with_oom_recovery(train_files, eval_files)

    def _train_distributed(self):
        """Run multi-node distributed training via Ray.

        Each worker node will tokenize its own 1/num_nodes shard of the dataset,
        enabling parallel tokenization and reducing driver memory pressure.
        """
        from surogate.train.distributed import RayDistributedTrainer

        dist_cfg = self.config.distributed
        logger.info(f"Starting distributed training with {dist_cfg.num_nodes} nodes...")
        logger.metric("Ray address", dist_cfg.ray_address)
        logger.metric("Nodes", dist_cfg.num_nodes)
        logger.metric("GPUs per node", dist_cfg.gpus_per_node or self.config.gpus)
        tokenize_on_node = not (self.config.train_vision and self.config.is_multimodal)
        logger.metric("Per-node tokenization", "enabled" if tokenize_on_node else "disabled (vision training)")

        trainer = RayDistributedTrainer(
            config=self.config,
            train_files=[],  # Workers will tokenize their own data
            eval_files=None,
            ray_address=dist_cfg.ray_address,
            num_nodes=dist_cfg.num_nodes,
            gpus_per_node=dist_cfg.gpus_per_node or self.config.gpus,
            tokenize_on_node=tokenize_on_node,  # Tokenize per node unless on-the-fly mode
        )

        try:
            trainer.train()
        finally:
            trainer.shutdown()

    def _cleanup_failed_trainer(self) -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def train_with_oom_recovery(self, train_files, eval_files):
        original_batch_size = self.config.per_device_train_batch_size
        original_grad_accum = self.config.gradient_accumulation_steps
        min_batch_size = 1
        attempt = 0
        max_attempts = 10
        res = None

        trainer = None

        while self.config.per_device_train_batch_size >= min_batch_size and attempt < max_attempts:
            attempt += 1
            retry_after_oom = False

            try:
                trainer = SurogateTrainerWrapper(config=self.config, train_files=train_files, eval_files=eval_files)
                res = trainer.train()
                logger.info("Training completed successfully.")
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_oom = any(
                    x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"]
                )
                if is_oom:
                    logger.warning(f"Out of memory error encountered during training attempt {attempt}.")
                    trainer = None

                    current_batch = self.config.per_device_train_batch_size
                    current_grad_accum = self.config.gradient_accumulation_steps

                    if current_grad_accum < 16 and current_batch > 1:
                        new_batch_size = max(1, current_batch // 2)
                        new_grad_accum = min(32, current_grad_accum * 2)
                    elif current_batch > 1:
                        new_batch_size = max(1, current_batch // 2)
                        new_grad_accum = current_grad_accum
                    else:
                        logger.error("Cannot reduce batch size further to recover from OOM.")
                        raise

                    self.config.per_device_train_batch_size = new_batch_size
                    self.config.gradient_accumulation_steps = new_grad_accum

                    logger.info("Adjusting training configuration to recover from OOM:")
                    logger.metric("New batch size", f"{current_batch} → {new_batch_size}")
                    logger.metric("New gradient accumulation", f"{current_grad_accum} → {new_grad_accum}")
                    logger.metric(
                        "New effective batch size",
                        f"{current_batch * current_grad_accum} → {new_batch_size * new_grad_accum}",
                    )
                    retry_after_oom = True
                else:
                    raise

            if retry_after_oom:
                self._cleanup_failed_trainer()
                continue

        if attempt >= max_attempts:
            logger.error(f"Training failed after {max_attempts} attempts")
            raise RuntimeError(f"Could not complete training after {max_attempts} OOM recovery attempts")

        final_batch = self.config.per_device_train_batch_size
        final_grad_accum = self.config.gradient_accumulation_steps

        if final_batch != original_batch_size or final_grad_accum != original_grad_accum:
            logger.info("Training completed with adjusted batch size and/or gradient accumulation steps:")
            logger.metric("Batch size", f"{original_batch_size} → {final_batch}")
            logger.metric("Gradient accumulation", f"{original_grad_accum} → {final_grad_accum}")
            logger.metric(
                "Effective batch size",
                f"{original_batch_size * original_grad_accum} → {final_batch * final_grad_accum}",
            )

        return res


def sft_main(config: SFTConfig, args: DictDefault):
    SurogateSFT(config, args).run()
