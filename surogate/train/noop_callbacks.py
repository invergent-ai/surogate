from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, trainer, PrinterCallback

from surogate.utils.logger import get_logger

logger = get_logger()

class NoopTrainerCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass


class NoopPrinterCallback(PrinterCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

