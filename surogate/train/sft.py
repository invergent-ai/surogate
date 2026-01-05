import os
from functools import partial
from pathlib import Path
from typing import List, Union
from datasets import Dataset as HfDataset
from transformers import PreTrainedTokenizerBase

from surogate.core.config.sft_config import SFTConfig
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.core.model.loader import get_model_tokenizer
from surogate.train.tokenize import TokenizeDatasets
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed
from surogate.train.trainer import SurogateTrainerWrapper
import datasets

datasets.logging.set_verbosity_warning()

logger = get_logger()

class SurogateSFT(TokenizeDatasets):
    template_processor: ChatTemplateProcessor
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args)

    def run(self):
        # Tokenize datasets
        super().run()

        # Setup data loaders
        output_path = Path(self.config.output_dir)
        train_files = sorted([str(p) for p in output_path.glob("train-*.bin")])
        eval_files = sorted([str(p) for p in output_path.glob("eval.bin")])

        if not train_files:
            logger.error(f"No training files found matching '{self.config.output_dir}/train-*.bin'")
            return
        if not eval_files:
            logger.warning(f"No eval files found matching '{self.config.output_dir}/eval.bin'")

        logger.info(f"Starting training run '{self.config.run_name}'...")
        return self.train_with_oom_recovery(train_files, eval_files)
    
    def train_with_oom_recovery(self, train_files, eval_files):
        original_batch_size = self.config.per_device_train_batch_size
        original_grad_accum = self.config.gradient_accumulation_steps
        min_batch_size = 1
        attempt = 0
        max_attempts = 10
        res = None

        trainer = SurogateTrainerWrapper(
            config=self.config,
            train_files=train_files,
            eval_files=eval_files
        )

        while self.config.per_device_train_batch_size >= min_batch_size and attempt < max_attempts:
            attempt += 1

            try:
                res = trainer.train()
                logger.info("Training completed successfully.")
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_oom = any(
                    x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"])
                if is_oom:
                    logger.warning(f"Out of memory error encountered during training attempt {attempt}.")

                    import gc
                    gc.collect()

                    current_batch = self.config.per_device_train_batch_size
                    current_grad_accum = self.config.gradient_accumulation_steps

                    if current_grad_accum < 16 and current_batch > 1:
                        # If gradient accumulation is reasonable, increase it and reduce batch size
                        new_batch_size = max(1, current_batch // 2)
                        new_grad_accum = min(32, current_grad_accum * 2)
                    elif current_batch > 1:
                        # Just reduce batch size
                        new_batch_size = max(1, current_batch // 2)
                        new_grad_accum = current_grad_accum
                    else:
                        # Can't reduce further
                        logger.error("Cannot reduce batch size further to recover from OOM.")
                        raise

                    self.config.per_device_train_batch_size = new_batch_size
                    self.config.gradient_accumulation_steps = new_grad_accum

                    logger.info(f"Adjusting training configuration to recover from OOM:")
                    logger.metric("New batch size", f"{current_batch} → {new_batch_size}")
                    logger.metric("New gradient accumulation", f"{current_grad_accum} → {new_grad_accum}")
                    logger.metric("New effective batch size",
                                    f"{current_batch * current_grad_accum} → {new_batch_size * new_grad_accum}")
                    
                    trainer = SurogateTrainerWrapper(
                            config=self.config,
                            train_files=train_files,
                            eval_files=eval_files,
                    )
                else:
                    raise

        if attempt >= max_attempts:
            logger.error(f"Training failed after {max_attempts} attempts")
            raise RuntimeError(f"Could not complete training after {max_attempts} OOM recovery attempts")

        final_batch = self.config.per_device_train_batch_size
        final_grad_accum = self.config.gradient_accumulation_steps

        if final_batch != original_batch_size or final_grad_accum != original_grad_accum:
            logger.info("Training completed with adjusted batch size and/or gradient accumulation steps:")
            logger.metric("Batch size", f"{original_batch_size} → {final_batch}")
            logger.metric("Gradient accumulation", f"{original_grad_accum} → {final_grad_accum}")
            logger.metric("Effective batch size",
                          f"{original_batch_size * original_grad_accum} → {final_batch * final_grad_accum}")

        return res
                


def sft_main(config: SFTConfig, args: DictDefault):
    SurogateSFT(config, args).run()