import numpy as np

from surogate.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

logger = get_logger()


class EarlyStopCallback(TrainerCallback):
    def __init__(self, config: SFTConfig, total_interval=3):
        self.best_metric = None
        self.interval = 0
        self.total_interval = total_interval
        self.config = config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.best_metric is None or np.less(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
            self.interval = 0
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f'Training stop because of eval metric is stable at step {state.global_step}')
            control.should_training_stop = True
