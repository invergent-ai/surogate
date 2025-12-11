import os
from functools import partial
from typing import List

import datasets
import numpy as np
import torch
from datasets import Dataset as HfDataset
from transformers import trainer, TrainingArguments

from surogate.core.config.dataset_config import DatasetConfig
from surogate.core.config.sft_config import SFTConfig
from surogate.core.datasets.datasets import get_default_process_count
from surogate.core.datasets.loader import load_dataset_with_config, \
    post_process, pre_process, concat_datasets, shuffle_dataset, _load_from_local_path
from surogate.core.datasets.packing import IterablePackingDataset, PackingDataset
from surogate.core.datasets.preprocessor.encode import EncodePreprocessor
from surogate.core.datasets.utils import DATASET_TYPE, LazyLLMDataset
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.core.model.saver import save_checkpoint
from surogate.core.model.utils import check_tie_word_embeddings
from surogate.ray import RayHelper
from surogate.train.callbacks import NoopPrinterCallback, NoopTrainerCallback, SFTTrainerCallback
from surogate.train.train_utils import TrainUtils
from surogate.train.trainer import SurogateTrainer
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.dist import is_master, get_dist_setting
from surogate.utils.jsonl import append_to_jsonl
from surogate.utils.logger import get_logger
from surogate.utils.model import estimate_model_parameters, recommend_training_params
from surogate.utils.np_utils import get_seed

datasets.logging.set_verbosity_warning()

logger = get_logger()

trainer.DEFAULT_PROGRESS_CALLBACK = NoopTrainerCallback
trainer.PrinterCallback = NoopPrinterCallback

@RayHelper.worker(group=['default'])
class SurogateSFT(SurogateCommand):
    training_args: TrainingArguments
    template_processor: ChatTemplateProcessor

    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args)
        config.__post_init__()

        self._prepare_model_tokenizer()
        self._prepare_template()
        self._prepare_callbacks()

    @RayHelper.function(group='default')
    def _prepare_model_tokenizer(self):
        self.model, self.processor = self.config.get_model_processor()
        if self.config.sequence_parallel_size > 1:
            from sequence_parallel import sequence_parallel
            sequence_parallel.prepare(
                self.config.sequence_parallel_size, model=self.model, tokenizer=self.processor,
                padding_free=self.config.padding_free)
        if self.model is None:
            return

    @RayHelper.function(group='default')
    def _prepare_template(self) -> None:
        template_processor = self.config.get_template_processor(self.processor)
        template_processor.set_mode('train')
        if template_processor.use_model:
            template_processor.model = self.model
        if self.config.model_template.is_multimodal and (
                self.config.padding_free or self.config.sample_packing) and not template_processor.support_padding_free:
            raise ValueError(f'Template `{self.config.template}` does not support padding free or packing.')
        self.template_processor = template_processor

    @RayHelper.function(group='default')
    def _prepare_callbacks(self):
        self.callbacks = [SFTTrainerCallback()]

    @RayHelper.function(group='default')
    def _prepare_datasets(self):
        train_datasets, val_datasets = [], []
        train_dataset, val_dataset = None, None
        if not self.config.stream_datasets:
            for cached_ds in ['train_dataset', 'val_dataset']:
                if os.path.exists(os.path.join(self.config.save_path, cached_ds)):
                    logger.info(f"Loading cached {cached_ds}...")
                    load_dataset_kwargs = {
                        'streaming': False,
                    }
                    ds = _load_from_local_path(
                        DatasetConfig(DictDefault({'path': os.path.join(self.config.save_path, cached_ds)})), load_dataset_kwargs)
                    if cached_ds == 'train_dataset':
                        train_dataset = ds
                    else:
                        val_dataset = ds
        else:
            seed = np.random.RandomState(self.config.seed)
            for ds_config in self.config.datasets:
                dataset = load_dataset_with_config(ds_config, streaming=self.config.stream_datasets)
                dataset = pre_process(dataset, ds_config, num_proc=get_default_process_count())
                train_dataset, val_dataset = post_process(
                    dataset,
                    dataset_sample=ds_config.samples,
                    split_dataset_ratio=self.config.validation_split_ratio,
                    streaming=self.config.stream_datasets,
                    random_state=seed,
                )
                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)

            for ds_config in self.config.validation_datasets:
                dataset = load_dataset_with_config(ds_config, streaming=self.config.stream_datasets)
                dataset = pre_process(dataset, ds_config, num_proc=get_default_process_count())
                _, val_dataset = post_process(
                    dataset,
                    dataset_sample=ds_config.samples,
                    split_dataset_ratio=1.0,
                    streaming=self.config.stream_datasets,
                    random_state=seed,
                )
                val_datasets.append(val_dataset)

            train_dataset = concat_datasets(train_datasets)
            train_dataset = shuffle_dataset(
                train_dataset, seed=get_seed(seed), buffer_size=1000)

            val_dataset = concat_datasets(val_datasets)
            val_dataset = shuffle_dataset(
                val_dataset, seed=get_seed(seed), buffer_size=1000)

            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=True)

            if not self.config.stream_datasets:
                train_dataset.save_to_disk(os.path.join(self.config.save_path, 'train_dataset'))
                if val_dataset is not None:
                    val_dataset.save_to_disk(os.path.join(self.config.save_path, 'val_dataset'))

        _datasets = [train_dataset, val_dataset]
        _datasets = self._post_process_datasets(_datasets)
        return _datasets

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        template_processor = self.template_processor

        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets

        origin_template_model = template_processor.model
        template_processor.model = None  # Avoid serializing the model.
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            preprocessor = EncodePreprocessor(template=template_processor,
                                              pre_tokenize=self.config.model_template.is_multimodal)
            batch_size = 100 if self.config.model_template.is_multimodal else 1000
            dataset = preprocessor(
                dataset,
                num_proc=get_default_process_count(),
                load_from_cache_file=False,
                strict=False,
                batch_size=batch_size)
            datasets[i] = dataset
        template_processor.model = origin_template_model

        return datasets

    def _post_process_datasets(self, datasets: List) -> List:
        template = self.template_processor
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            if self.config.model_template.is_multimodal and not self.config.stream_datasets:
                dataset = LazyLLMDataset(dataset, template.encode, strict=False, random_state=self.config.seed)
            if self.config.sample_packing:
                packing_dataset_cls = IterablePackingDataset if self.config.stream_datasets else PackingDataset
                dataset = packing_dataset_cls(
                    template,
                    dataset,
                    num_proc=get_default_process_count(),
                    packing_length=self.config.sequence_len,
                    strict=False,
                    load_from_cache_file=True)
            elif self.config.stream_datasets:
                preprocessor = EncodePreprocessor(template=template)
                dataset = preprocessor(
                    dataset,
                    num_proc=get_default_process_count(),
                    load_from_cache_file=True,
                    strict=False)
            datasets[i] = dataset
        return datasets

    @RayHelper.function(group='default')
    def run(self):
        train_dataset, val_dataset = self._prepare_datasets()
        data_collator = self._get_data_collator()

        self.model = TrainUtils.prepare_model(self.config, self.model, task_type="causal_lm")

        if self.config.apply_recommended_values:
            optim_config = self._determine_recommended_config(train_dataset)

            if self.config.per_device_train_batch_size != optim_config['per_device_train_batch_size']:
                logger.warning(f'The configured per_device_train_batch_size '
                               f'({self.config.per_device_train_batch_size}) differs from '
                               f'the recommended per_device_train_batch_size is {optim_config["per_device_train_batch_size"]}. '
                               f'Using the recommended value.')
                self.config.trainer_args.per_device_train_batch_size = optim_config['per_device_train_batch_size']

            if self.config.gradient_accumulation_steps is None:
                self.config.trainer_args.gradient_accumulation_steps = optim_config['gradient_accumulation_steps']

            if self.config.deepspeed is None:
                self.config.trainer_args.deepspeed = optim_config['deepspeed']

        logger.info(f"Starting training run '{self.config.run_name}'...")
        return self.train_with_oom_recovery(data_collator, train_dataset, val_dataset)

    def _get_data_collator(self):
        return partial(self.template_processor.data_collator, padding_to=None)

    def train_with_oom_recovery(self, data_collator, train_dataset, val_dataset):
        original_batch_size = self.config.trainer_args.per_device_train_batch_size
        original_grad_accum = self.config.trainer_args.gradient_accumulation_steps
        min_batch_size = 1
        attempt = 0
        max_attempts = 10
        res = None

        trainer = SurogateTrainer(
            config=self.config,
            args=self.config.trainer_args,
            model=self.model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            template_processor=self.template_processor,
            callbacks=self.callbacks,
        )

        while self.config.trainer_args.per_device_train_batch_size >= min_batch_size and attempt < max_attempts:
            attempt += 1

            try:
                res = trainer.train(self.config.resume_from_checkpoint)
                logger.info("Training completed successfully.")
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_oom = any(
                    x in error_msg for x in ["out of memory", "oom", "cuda out of memory", "mps out of memory"])
                if is_oom:
                    logger.warning(f"Out of memory error encountered during training attempt {attempt}.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                    import gc
                    gc.collect()

                    current_batch = self.config.trainer_args.per_device_train_batch_size
                    current_grad_accum = self.config.trainer_args.gradient_accumulation_steps

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

                    self.config.trainer_args.per_device_train_batch_size = new_batch_size
                    self.config.trainer_args.gradient_accumulation_steps = new_grad_accum

                    logger.info(f"Adjusting training configuration to recover from OOM:")
                    logger.metric("New batch size", f"{current_batch} → {new_batch_size}")
                    logger.metric("New gradient accumulation", f"{current_grad_accum} → {new_grad_accum}")
                    logger.metric("New effective batch size",
                                  f"{current_batch * current_grad_accum} → {new_batch_size * new_grad_accum}")

                    trainer = SurogateTrainer(
                        config=self.config,
                        model=self.model,
                        args=self.config.trainer_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        callbacks=self.callbacks,
                        template_processor=self.template_processor,
                    )
                else:
                    raise

            finally:
                res = self._save_trainer_state(trainer)

        if attempt >= max_attempts:
            logger.error(f"Training failed after {max_attempts} attempts")
            raise RuntimeError(f"Could not complete training after {max_attempts} OOM recovery attempts")

        final_batch = self.config.trainer_args.per_device_train_batch_size
        final_grad_accum = self.config.trainer_args.gradient_accumulation_steps

        if final_batch != original_batch_size or final_grad_accum != original_grad_accum:
            logger.info("Training completed with adjusted batch size and/or gradient accumulation steps:")
            logger.metric("Batch size", f"{original_batch_size} → {final_batch}")
            logger.metric("Gradient accumulation", f"{original_grad_accum} → {final_grad_accum}")
            logger.metric("Effective batch size",
                          f"{original_batch_size * original_grad_accum} → {final_batch * final_grad_accum}")

        return res

    def _determine_recommended_config(self, train_dataset: DATASET_TYPE):
        params = estimate_model_parameters(self.model.config)
        dataset_size = len(train_dataset)
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        optimal_config = recommend_training_params(
            model_size_b=params / 1e9,
            dataset_size=dataset_size,
            quantization="4bit" if self.config.qlora else "bf16",
            num_gpus=world_size,
            vram_per_gpu_gb=int(self.system_info['gpu_memory_gb']),
            seq_len=self.config.sequence_len,
        )

        if hasattr(self.model, 'tie_weights') and optimal_config['deepspeed_stage'] == 3:
            # Check if weights are tied
            if self.model.config.tie_word_embeddings and is_master():
                logger.warning(
                    "The model has weight tying enabled - this can cause ZeRO-3 memory imbalance across GPUs.")

        mem = optimal_config['memory_breakdown']
        logger.header("Memory Breakdown per GPU:")
        logger.metric("Weights", f"{mem['weight_mem_gb']:6.2f}", "GB")
        logger.metric("Gradients", f"{mem['grad_mem_gb']:6.2f}", "GB")
        logger.metric("Optimizer States", f"{mem['optimizer_mem_gb']:6.2f}", "GB")
        logger.metric("Total (before ZeRO)", f"{mem['optimizer_mem_gb']:6.2f}", "GB")
        logger.metric("Per GPU (after ZeRO)", f"{mem['per_gpu_mem_gb']:6.2f}", "GB")
        if mem.get('zero3_allgather_peak_gb', 0) > 0:
            logger.metric("ZeRO-3 all-gather Peak", f"{mem['zero3_allgather_peak_gb']:6.2f}", "GB")
        logger.metric("Activation/sample", f"{mem['activation_per_sample_gb']:6.2f}", "GB")
        logger.metric("Available for batch", f"{mem['available_for_batch_gb']:6.2f}", "GB")

        if optimal_config['warnings']:
            for w in optimal_config['warnings']:
                logger.warning(f"{w}")

        if is_master():
            logger.header("Recommended Training Configuration")
            for key in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'deepspeed_stage', 'use_offload',
                        'offload_device']:
                logger.metric(key, optimal_config[key])

        return optimal_config

    def _save_trainer_state(self, trainer):
        state = trainer.state

        if self.config.merge_adapter:
            check_tie_word_embeddings(self.model)
            self.model.merge_and_unload()
            model = self.model.model
            logger.info('Saving merged weights...')
            save_checkpoint(
                model,
                self.template_processor.processor,
                f"{self.config.save_path}/merged",
                safe_serialization=True,
                model_dirs=[state.last_model_checkpoint],
                max_shard_size='5GB',
                additional_saved_files=self.config.model_template.additional_saved_files)
        elif hasattr(state, 'last_model_checkpoint'):
            if trainer.args.push_to_hub:
                trainer.push_to_hub()

def sft_main(config: SFTConfig, args: DictDefault):
    SurogateSFT(config, args).run()
