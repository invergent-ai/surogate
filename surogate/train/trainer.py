import functools
import inspect
import os
import shutil
import time
from contextlib import contextmanager
from functools import partial
from types import MethodType, FunctionType
from typing import Union, Optional, Dict, Callable, List, Tuple, Any

import torch
from accelerate.utils import AORecipeKwargs
from datasets import Dataset as HfDataset
from peft import PeftModel
from torchao.float8 import Float8LinearConfig

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

        # for logging tokens per second
        self._last_log_time = 0
        self._token_count = 0

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

        model.model.is_quantized = False

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

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if "input_ids" in inputs:
            num_tokens = inputs["input_ids"].numel()
            self._token_count += num_tokens

        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        current_time = time.time()
        time_diff = current_time - self._last_log_time
        if time_diff > 0:
            tokens_per_sec = self._token_count / time_diff
            logs["tokens_per_sec"] = round(tokens_per_sec, 2)
            self._token_count = 0
            self._last_log_time = current_time
        super().log(logs)

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


    def create_accelerator_and_postprocess(self):
        from accelerate import DataLoaderConfiguration, Accelerator
        from accelerate.utils import PrecisionType

        accelerator_config = self.args.accelerator_config.to_dict()
        dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params}
        )
        dataloader_config.data_seed = self.args.data_seed
        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "dataloader_config": dataloader_config,
        }

        if self.config.model_info.quant_method == 'fp8':
            args.update({
                "mixed_precision": PrecisionType.FP8,
                "kwargs_handlers": [AORecipeKwargs(config=Float8LinearConfig(
                    enable_fsdp_float8_all_gather=False
                ))]
            })


        self.accelerator = Accelerator(**args)
        self.gather_function = self.accelerator.gather_for_metrics
        if "use_gather_object" in inspect.signature(self.gather_function).parameters:
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()


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