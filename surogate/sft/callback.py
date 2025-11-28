from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from surogate.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger

logger = get_logger()

class SurogateSftCallback(TrainerCallback):
    def __init__(self,  config: SFTConfig):
        self.config = config

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f"Model checkpoint saved at step {state.global_step} to {args.output_dir}")
