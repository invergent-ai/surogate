import inspect
from contextlib import contextmanager
from types import MethodType

from peft import PeftModel
from transformers import Trainer
from  transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import unwrap_model

from surogate.utils.logger import get_logger

logger = get_logger()

class PatchDeepspeedLoadCheckpoint(Trainer):
    @contextmanager
    def _patch_deepspeed_load_checkpoint(self):
        from transformers import trainer
        if not self.args.resume_from_checkpoint or not self.args.resume_only_model or not hasattr(
                trainer, 'deepspeed_load_checkpoint'):
            yield
            return
        origin_deepspeed_load_checkpoint = trainer.deepspeed_load_checkpoint

        def deepspeed_load_checkpoint(*args, **kwargs):
            try:
                return origin_deepspeed_load_checkpoint(*args, **kwargs)
            except Exception as e:
                logger.warning('Failed to call deepspeed_load_checkpoint function. '
                               f'If `--resume_only_model true` is set, this warning can be ignored. {e}.')

        trainer.deepspeed_load_checkpoint = deepspeed_load_checkpoint

        try:
            yield
        finally:
            trainer.deepspeed_load_checkpoint = origin_deepspeed_load_checkpoint


    def _fix_zero3_gather_all_parameters(self) -> None:
        if is_deepspeed_zero3_enabled() and not hasattr(self.deepspeed, '_zero3_consolidated_16bit_state_dict_origin'):
            parameters = inspect.signature(self.deepspeed._zero3_consolidated_16bit_state_dict).parameters
            if 'exclude_frozen_parameters' in parameters:
                self.deepspeed._zero3_consolidated_16bit_state_dict_origin = (
                    self.deepspeed._zero3_consolidated_16bit_state_dict)

                def _zero3_consolidated_16bit_state_dict(model, exclude_frozen_parameters=False):
                    unwrapped = unwrap_model(model)
                    exclude_frozen_parameters = False
                    if isinstance(unwrapped, PeftModel):
                        exclude_frozen_parameters = True
                    return model._zero3_consolidated_16bit_state_dict_origin(exclude_frozen_parameters)

                self.deepspeed._zero3_consolidated_16bit_state_dict = MethodType(
                    _zero3_consolidated_16bit_state_dict, self.deepspeed)