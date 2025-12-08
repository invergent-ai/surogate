import inspect
import os
import shutil
import time
from contextlib import contextmanager
from functools import partial
from types import MethodType, FunctionType
from typing import Union, Optional, Dict, Callable, List, Tuple

import torch
from datasets import Dataset as HfDataset
from peft import PeftModel

from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.core.model.utils import update_generation_config_eos_token
from surogate.train.trainer_mixins.patch_deepspeed import PatchDeepspeedLoadCheckpoint
from torch import nn
from torch.nn import Module
from transformers import PreTrainedModel, DataCollator, TrainerCallback, TrainingArguments, Trainer, IntervalStrategy

from surogate.core.config.sft_config import SFTConfig
from surogate.core.model.registry import ModelTemplate
from surogate.train.trainer_mixins.dataloader import DataLoaderMixin
from surogate.train.trainer_mixins.fix_gradnorm import FixGradnormNan
from surogate.train.trainer_mixins.gradient_checkpointing import GradientCheckpointingMixin
from surogate.utils.fs import copy_files_by_pattern
from surogate.utils.logger import get_logger

logger = get_logger()


class SurogateTrainer(
    FixGradnormNan,
    PatchDeepspeedLoadCheckpoint,
    GradientCheckpointingMixin,
    DataLoaderMixin,
    Trainer
):
    config: SFTConfig
    model_template: ModelTemplate
    template_processor: ChatTemplateProcessor

    def __init__(
            self,
            config: SFTConfig,
            args: TrainingArguments = None,
            model: Union[PreTrainedModel, Module] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[HfDataset] = None,
            eval_dataset: Optional[Union[HfDataset, Dict[str, HfDataset]]] = None,
            template_processor: Optional[ChatTemplateProcessor] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            **kwargs
    ):
        self.config = config
        self.model_accepts_loss_kwargs = True  # fix transformers>=4.46.2
        self.template_processor = template_processor
        self.model_template = config.model_template

        if not hasattr(train_dataset, '__len__') and config.dataloader_num_workers > 1:
            config.dataloader_num_workers = 1
            logger.warning('Using IterableDataset, setting args.dataloader_num_workers to 1.')

        if eval_dataset is None and args:
            if getattr(args, 'eval_dataset', None):
                # Avoid trainer throwing errors.
                eval_dataset = []
            else:
                args.evaluation_strategy = IntervalStrategy.NO
                args.eval_strategy = IntervalStrategy.NO

        trainer_parameters = inspect.signature(Trainer.__init__).parameters
        tokenizer_key = 'processing_class' if 'processing_class' in trainer_parameters else 'tokenizer'
        kwargs[tokenizer_key] = template_processor.tokenizer

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs)

        if get_function(model.__class__.forward) is not get_function(model.forward):
            self.label_names = find_labels(model)
            self.can_return_loss = can_return_loss(model)

        self.label_names = self.label_names or ['labels']
        self.start_time = time.time()
        self._fix_gradient_checkpointing()
        update_generation_config_eos_token(self.model.generation_config, self.template_processor)
        if getattr(self.model, 'origin_generation_config', None):
            self.model.origin_generation_config.eos_token_id = self.model.generation_config.eos_token_id
        if self.config.resume_only_model and self.args.ignore_data_skip:
            # The weights have already been loaded outside the trainer,
            # so reading train_state is skipped here.
            self.args.resume_from_checkpoint = None


    def train(self, *args, **kwargs):
        if self.model_template.is_multimodal:
            models = []
            for model_name in ['model', 'ref_model', 'value_model', 'teacher_model']:
                model = getattr(self, model_name, None)
                if isinstance(model, nn.Module):
                    models.append(model)

            models = list(set(self.accelerator.unwrap_model(model) for model in models))  # Deduplicate
            self.template_processor.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}.')

        # gradient_checkpointing
        gradient_checkpointing = self.args.gradient_checkpointing
        self._prepare_gradient_checkpointing(self.accelerator.unwrap_model(self.model))

        with self._fix_grad_norm_nan(), self._patch_skip_first_batches(), self._patch_deepspeed_load_checkpoint():
            res = super().train(*args, **kwargs)

        self.template_processor.remove_post_encode_hook()
        self.args.gradient_checkpointing = gradient_checkpointing  # recover
        return res

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template_processor.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the master process, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._save_model(output_dir, state_dict)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))

        # args.json
        args_path = os.path.join(os.path.dirname(output_dir), 'args.json')
        if os.path.exists(args_path):
            shutil.copy(args_path, os.path.join(output_dir, 'args.json'))

    def _save_model(self, output_dir: Optional[str] = None, state_dict=None):
        if self.model.__class__.__name__ != 'SentenceTransformer':
            self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
        else:
            @contextmanager
            def save_context():
                save_pretrained = self.model[0].auto_model.save_pretrained
                _state_dict = {
                    key[len('0.auto_model.'):] if 'auto_model' in key else key: value
                    for key, value in state_dict.items()
                }
                self.model[0].auto_model.save_pretrained = partial(
                    self.model[0].auto_model.save_pretrained, state_dict=_state_dict)
                yield
                self.model[0].auto_model.save_pretrained = save_pretrained

            with save_context():
                self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)

            copy_files_by_pattern(
                self.model.model_dir, output_dir, '*.py', exclude_patterns=['model.safetensors.index.json'])
            copy_files_by_pattern(
                self.model.model_dir, output_dir, '*.json', exclude_patterns=['model.safetensors.index.json'])

    def _save_checkpoint(self, *args, **kwargs):
        self.state.last_model_checkpoint = os.path.join(self.args.output_dir, f'checkpoint-{self.state.global_step}')
        self._fix_zero3_gather_all_parameters()
        result = super()._save_checkpoint(*args, **kwargs)
        logger.info(f'Saving model checkpoint to {self.state.last_model_checkpoint}')
        return result


def get_function(method_or_function: Union[MethodType, FunctionType]) -> FunctionType:
    if isinstance(method_or_function, MethodType):
        method_or_function = method_or_function.__func__
    return method_or_function

def find_labels(model: Module) -> List[str]:
    """Find the labels used by a given model."""
    model_name = model.__class__.__name__
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    if 'QuestionAnswering' in model_name:
        return [p for p in signature.parameters if 'label' in p or p in ('start_positions', 'end_positions')]
    else:
        return [p for p in signature.parameters if 'label' in p]

def can_return_loss(model: Module) -> bool:
    """Check if a given model can return loss."""
    if isinstance(model, PeftModel):
        signature = inspect.signature(model.model.forward)
    else:
        signature = inspect.signature(model.forward)
    for p in signature.parameters:
        if p == 'return_loss' and signature.parameters[p].default is True:
            return True
    return False