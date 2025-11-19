import collections
import inspect
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional, Union, Callable, List, Tuple, Any, Dict

import torch
from datasets import IterableDataset as HFIterableDataset, Dataset as HFDataset
from swift.llm import Template, DataLoaderShard, DataLoaderDispatcher
from swift.llm.utils import update_generation_config_eos_token
from swift.utils.env import is_dist
from swift.plugin.metric import compute_acc, MeanMetric
from torch import nn
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, PreTrainedModel, DataCollator, TrainerCallback

from surogate.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger

logger = get_logger()


class SurogateSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            config: Optional[SFTConfig] = None,
            model: Optional[Union[PreTrainedModel, nn.Module]] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, HFIterableDataset, HFDataset]] = None,
            eval_dataset: Optional[Union[Dataset, HFIterableDataset, HFDataset]] = None,
            template: Optional[Template] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        self.config = config
        self.template = template
        self.model_meta = model.model_meta

        training_args = self.config.to_hf_training_args()
        training_args.max_epochs = None

        kwargs = {
            "processing_class": template.tokenizer,
            "model": model,
            "args": training_args,
            "data_collator": data_collator,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "model_init": model_init,
            "callbacks": callbacks,
            "optimizers": optimizers,
        }
        super().__init__(**kwargs)

        self.start_time = time.time()
        logger.info(f'output_dir: {training_args.output_dir}')

        self._fix_gradient_checkpointing()

        update_generation_config_eos_token(self.model.generation_config, self.template)
        if getattr(self.model, 'origin_generation_config', None):
            self.model.origin_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        self.model_accepts_loss_kwargs = True

        def _get_mean_metric():
            return MeanMetric(nan_value=None, device=training_args.device)

        self.custom_metrics = {
            'train': collections.defaultdict(_get_mean_metric),
        }

    def train(self, *args, **kwargs):
        if self.model_meta.is_multimodal:
            models = []
            for model_name in ['model', 'ref_model', 'value_model', 'teacher_model']:
                model = getattr(self, model_name, None)
                if isinstance(model, nn.Module):
                    models.append(model)

            models = list(set(self.accelerator.unwrap_model(model) for model in models))  # Deduplicate
            self.template.register_post_encode_hook(models)
            logger.info(f'Successfully registered post_encode hook: {[model.__class__.__name__ for model in models]}.')

        gradient_checkpointing = self.config.gradient_checkpointing
        self._prepare_gradient_checkpointing(self.accelerator.unwrap_model(self.model))
        with self._fix_grad_norm_nan(), self._patch_skip_first_batches(), self._patch_deepspeed_load_checkpoint():
            res = super().train(*args, **kwargs)

        self.template.remove_post_encode_hook()
        self.config.gradient_checkpointing = gradient_checkpointing
        return res

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        return super().log(logs, *args, **kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps=num_training_steps)

    def training_step(self, model, inputs, *args, **kwargs):
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        labels = inputs['labels']
        outputs.loss = outputs.loss.to(labels.device)

        # fix https://github.com/huggingface/transformers/issues/34263
        if num_items_in_batch is not None:
            outputs.loss = outputs.loss * ((labels[:, 1:] != -100).sum() / num_items_in_batch)

        if isinstance(outputs, dict) and 'loss' not in outputs:
            raise ValueError(
                'The model did not return a loss from the inputs, only the following keys: '
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}.")
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        if getattr(self.args, 'average_tokens_across_devices',
                   False) and self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            loss *= self.accelerator.num_processes

        if outputs.logits is not None and labels is not None:
            self._compute_acc(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def _compute_acc(self, outputs, labels, cu_seqlens=None) -> None:
        logits = outputs.logits
        preds = logits.argmax(dim=-1)
        metrics = compute_acc(
            preds,
            labels,
            acc_strategy='token',
            is_encoder_decoder=self.template.is_encoder_decoder,
            cu_seqlens=cu_seqlens)

        if metrics:
            for k, v in metrics.items():
                self.custom_metrics['train'][k].update(v)

    def _prepare_gradient_checkpointing(self, model) -> None:
        from swift.llm import HfConfigFactory, deep_getattr, dynamic_gradient_checkpointing
        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)
        if self.config.gradient_checkpointing:
            dynamic_gradient_checkpointing(model, self.config.gradient_checkpointing)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        gc_kwargs = {}
        parameters = inspect.signature(model.gradient_checkpointing_enable).parameters
        if 'gradient_checkpointing_kwargs' in parameters:
            gc_kwargs['gradient_checkpointing_kwargs'] = {}

        model_meta = model.model_meta
        model_arch = model_meta.model_arch
        if model_meta.is_multimodal and model_arch:
            for vision_tower_name in model_arch.vision_tower:
                vision_tower = deep_getattr(model, vision_tower_name)
                if hasattr(vision_tower, 'enable_input_require_grads'):
                    try:
                        if self.config.gradient_checkpointing:
                            vision_tower.gradient_checkpointing_enable(**gc_kwargs)
                            vision_tower.enable_input_require_grads()
                        else:
                            vision_tower.gradient_checkpointing_disable()
                            vision_tower.disable_input_require_grads()
                    except (NotImplementedError, AttributeError) as e:
                        logger.warning(f'prepare gradient_checkpointing failed: {e}')

        self.config.gradient_checkpointing = False

    def _fix_gradient_checkpointing(self):
        if hasattr(torch.utils.checkpoint, '_old_checkpoint'):  # avoid double patching
            return

        if is_dist() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
            use_reentrant_ = False
        else:
            use_reentrant_ = True

        _old_checkpoint = torch.utils.checkpoint.checkpoint

        @wraps(_old_checkpoint)
        def _new_checkpoint(*args, use_reentrant=None, **kwargs):
            return _old_checkpoint(*args, use_reentrant=use_reentrant_, **kwargs)

        torch.utils.checkpoint._old_checkpoint = _old_checkpoint
        torch.utils.checkpoint.checkpoint = _new_checkpoint

        try:
            # Fix the old version of transformers.
            import transformers.modeling_utils
            transformers.modeling_utils.checkpoint = _new_checkpoint
        except (ImportError, AttributeError):
            pass

    @staticmethod
    @contextmanager
    def _fix_grad_norm_nan():
        from accelerate import Accelerator
        origin_clip_grad_norm_ = Accelerator.clip_grad_norm_

        def clip_grad_norm_(self, parameters, *args, **kwargs):
            # If NaN occurs, ignore weight updates.
            parameters = list(parameters)
            grad_norm = origin_clip_grad_norm_(self, parameters, *args, **kwargs)
            if isinstance(grad_norm, torch.Tensor) and grad_norm.isnan().item():
                for p in parameters:
                    p.grad = None
            return grad_norm

        Accelerator.clip_grad_norm_ = clip_grad_norm_
        try:
            yield
        finally:
            Accelerator.clip_grad_norm_ = origin_clip_grad_norm_

    @contextmanager
    def _patch_skip_first_batches(self):
        from transformers import trainer
        origin_skip_first_batches = trainer.skip_first_batches

        def skip_first_batches(dataloader, num_batches=0):
            if isinstance(dataloader, (DataLoaderShard, DataLoaderDispatcher)):
                # DataLoaderMixin
                return self.get_train_dataloader(skip_batches=num_batches)
            else:
                return origin_skip_first_batches(dataloader, num_batches)

        trainer.skip_first_batches = skip_first_batches
        try:
            yield
        finally:
            trainer.skip_first_batches = origin_skip_first_batches

    @contextmanager
    def _patch_deepspeed_load_checkpoint(self):
        from transformers import trainer
        if not self.config.resume_from_checkpoint or not hasattr(
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
