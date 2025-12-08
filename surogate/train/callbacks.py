import json
import time

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PrinterCallback, trainer

from surogate.utils.dist import is_master, get_torch_device, get_device_count, is_mp
from surogate.utils.logger import get_logger
from surogate.utils.time import format_time

logger = get_logger()

class NoopTrainerCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

class NoopPrinterCallback(PrinterCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

def get_max_reserved_memory() -> float:
    devices = list(range(get_device_count())) if is_mp() else [None]
    mems = [get_torch_device().max_memory_reserved(device=device) for device in devices]
    return sum(mems) / 1024**3

class SFTTrainerCallback(trainer.TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return super().on_train_begin(args, state, control, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if is_master():
            step_log = state.log_history[-1]
            step_log['global_step/max_steps'] = f'{state.global_step}/{state.max_steps}'
            train_percentage = state.global_step / state.max_steps if state.max_steps else 0.
            step_log['percentage'] = f'{train_percentage * 100:.2f}%'
            elapsed = time.time() - self.start_time
            step_log['elapsed_time'] = format_time(elapsed)
            if train_percentage != 0:
                step_log['remaining_time'] = format_time(elapsed / train_percentage - elapsed)
            for k, v in step_log.items():
                if isinstance(v, float):
                    step_log[k] = round(step_log[k], 8)
            state.max_memory = max(getattr(state, 'max_memory', 0), get_max_reserved_memory())
            step_log['memory(GiB)'] = round(state.max_memory, 2)
            step_log['train_speed(iter/s)'] = round(state.global_step / elapsed, 6)
            step_log['train_speed(sec/iter)'] = round(elapsed / state.global_step, 6) if state.global_step > 0 else 0
            logger.info(json.dumps(step_log))