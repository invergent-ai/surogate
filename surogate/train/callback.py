import time

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, trainer

from surogate.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger

logger = get_logger()

class SurogateTrainCallback(TrainerCallback):
    def __init__(self,  config: SFTConfig):
        self.config = config

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = time.time()

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Starting training step {state.global_step + 1}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Completed training step {state.global_step}")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Starting epoch {state.epoch + 1}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Completed epoch {state.epoch}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Evaluation at step {state.global_step}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f"### Logging at step {state.global_step}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.debug(f"### Model checkpoint saved at step {state.global_step} to {args.output_dir}")


class NoopCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

trainer.DEFAULT_PROGRESS_CALLBACK = NoopCallback