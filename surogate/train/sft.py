import json
from functools import partial
from pathlib import Path
from typing import List

from transformers import SchedulerType

from surogate.config.dataset_config import DatasetConfig
from surogate.config.enums import SurogateDatasetType
from surogate.config.sft_config import SFTConfig
from surogate.datasets.conversation import ConversationPreprocessor
from surogate.datasets.datasets import get_default_process_count
from surogate.datasets.instruction import InstructionPreprocessor
from surogate.datasets.loader import _check_if_hub_dataset, swift_load_dataset
from surogate.datasets.text import TextPreprocessor
from surogate.train.callback import SurogateTrainCallback
from surogate.train.deepspeed import DEEPSPEED_CONFIGS
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.model import estimate_model_parameters, recommend_training_params
from swift.llm import BaseArguments, TrainArguments, DATASET_MAPPING, DatasetMeta, AutoPreprocessor, DATASET_TYPE
from swift.llm.dataset.loader import load_dataset
from swift.llm.train import SwiftSft
from swift.ray import RayHelper
from swift.trainers import TrainerFactory
from swift.utils import get_model_parameter_info, get_dist_setting, is_master

import datasets
datasets.logging.set_verbosity_warning()

logger = get_logger()


@RayHelper.worker(group=['default'])
class SurogateSFT(SurogateCommand, SwiftSft):
    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args, swift=True)
        config.__post_init__()
        if self.sg_config.sequence_len is None and self.model.model_info is not None:
            self.sg_config.sequence_len = self.model.model_info.max_model_len

        self._pre_prepare_datasets()

    def _pre_prepare_datasets(self):
        DATASET_MAPPING.clear()
        train_dataset_ids = [ds.path for ds in self.sg_config.datasets]
        val_dataset_ids = [ds.path for ds in self.sg_config.validation_datasets]

        def to_swift_dataset_id(datasets: List[DatasetConfig], dataset_id: str) -> str:
            ds_config: DatasetConfig = next(filter(lambda ds: ds.path == dataset_id, datasets), None)
            load_function = partial(swift_load_dataset, dataset_config=ds_config, sg_args=self.sg_args)
            if ds_config.type == SurogateDatasetType.instruction:
                preprocess_func = InstructionPreprocessor(ds_config)
            elif ds_config.type == SurogateDatasetType.conversation:
                preprocess_func = ConversationPreprocessor(ds_config)
            elif ds_config.type == SurogateDatasetType.text:
                preprocess_func = TextPreprocessor(ds_config)
            else:
                preprocess_func = AutoPreprocessor()

            if Path(dataset_id).exists():
                DATASET_MAPPING[dataset_id] = DatasetMeta(
                    dataset_path=dataset_id,
                    load_function=load_function,
                    preprocess_func=preprocess_func
                )
                return dataset_id
            elif _check_if_hub_dataset(dataset_id, self.sg_args):
                DATASET_MAPPING[dataset_id] = DatasetMeta(
                    hf_dataset_id=dataset_id,
                    load_function=load_function,
                    preprocess_func=preprocess_func
                )
                dataset_id = f"hf::{dataset_id}"
                if ds_config.samples:
                    dataset_id += f"#{ds_config.samples}"
                return dataset_id
            else:
                DATASET_MAPPING[dataset_id] = DatasetMeta(
                    dataset_path=dataset_id,
                    load_function=load_function,
                    preprocess_func=preprocess_func
                )
                return dataset_id

        self.args.dataset = list(map(partial(to_swift_dataset_id, self.sg_config.datasets), train_dataset_ids))
        self.args.val_dataset = list(
            map(partial(to_swift_dataset_id, self.sg_config.validation_datasets), val_dataset_ids))

    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        train_dataset, val_dataset = load_dataset(
            self.args.dataset, num_proc=get_default_process_count(),
            split_dataset_ratio=self.args.split_dataset_ratio, shuffle=self.args.dataset_shuffle,
            use_hf=True, hub_token=self.sg_args['hub_token'])

        if len(self.args.val_dataset) > 0:
            _, val_dataset = load_dataset(
                self.args.val_dataset, num_proc=get_default_process_count(),
                split_dataset_ratio=1.0, shuffle=self.args.val_dataset_shuffle,
                use_hf=True, hub_token=self.sg_args['hub_token'])
            assert self.args.split_dataset_ratio == 0.

        return train_dataset, val_dataset

    @RayHelper.function(group='default')
    def _prepare_callbacks(self):
        super()._prepare_callbacks()
        self.callbacks.append(SurogateTrainCallback(self.sg_config))

    @RayHelper.function(group='default')
    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info

        optim_config = self._determine_optimal_config(train_dataset)

        if self.sg_config.per_device_train_batch_size != optim_config['per_device_train_batch_size']:
            logger.warning(f'The configured per_device_train_batch_size '
                           f'({self.sg_config.per_device_train_batch_size}) differs from '
                           f'the recommended per_device_train_batch_size is {optim_config["per_device_train_batch_size"]}')
            self.args.training_args.per_device_train_batch_size = optim_config['per_device_train_batch_size']

        if self.sg_config.gradient_accumulation_steps is None:
            self.args.training_args.gradient_accumulation_steps = optim_config['gradient_accumulation_steps']

        if self.sg_config.deepspeed is None:
            self.args.training_args.deepspeed = optim_config['deepspeed']

        trainer_cls = TrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

    def _determine_optimal_config(self, train_dataset: DATASET_TYPE):
        params = estimate_model_parameters(self.model.config)
        dataset_size = len(train_dataset)
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        optimal_config = recommend_training_params(
            model_size_b=params / 1e9,
            dataset_size=dataset_size,
            quantization="bf16",
            num_gpus=world_size,
            vram_per_gpu_gb=int(self.system_info['gpu_memory_gb']),
            seq_len=self.sg_config.sequence_len,
        )

        if hasattr(self.model, 'tie_weights') and optimal_config['deepspeed_stage'] == 3:
            # Check if weights are tied
            if self.model.config.tie_word_embeddings and is_master():
                logger.warning("Weight tying detected - this can cause ZeRO-3 imbalance")

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
            for key in ['per_device_train_batch_size', 'gradient_accumulation_steps', 'deepspeed_stage', 'use_offload', 'offload_device']:
                logger.metric(key, optimal_config[key])

        return optimal_config

    def to_swift_args(self) -> BaseArguments:
        dataset_paths = [dataset.path for dataset in self.sg_config.datasets]
        val_dataset_paths = [dataset.path for dataset in self.sg_config.validation_datasets]
        swift_args = TrainArguments(
            model=self.sg_config.model,
            model_type=self.sg_config.model_type,
            num_train_epochs=self.sg_config.num_train_epochs,
            output_dir=self.sg_config.save_path,
            add_version=False,
            create_checkpoint_symlink=True,
            resume_from_checkpoint=self.sg_config.resume_from_checkpoint,
            seed=self.sg_config.seed,
            dataset=dataset_paths,
            dataset_shuffle=True,
            val_dataset=val_dataset_paths,
            val_dataset_shuffle=True,
            split_dataset_ratio=self.sg_config.validation_split_ratio,
            streaming=self.sg_config.stream_datasets,
            max_length=self.sg_config.sequence_len,
            gradient_checkpointing=self.sg_config.gradient_checkpointing,
            learning_rate=self.sg_config.learning_rate,
            lr_scheduler_type=SchedulerType.COSINE,
            save_steps=self.sg_config.checkpoint_steps,
            eval_steps=self.sg_config.eval_steps,
            save_total_limit=self.sg_config.max_checkpoints_to_keep,
            report_to=self.sg_config.report_to,
            max_steps=self.sg_config.max_steps,
            warmup_ratio=self.sg_config.warmup_ratio,
            weight_decay=self.sg_config.weight_decay,
            max_grad_norm=self.sg_config.gradient_clip_norm,
            per_device_train_batch_size=self.sg_config.per_device_train_batch_size,
            deepspeed=DEEPSPEED_CONFIGS[self.sg_config.deepspeed] if self.sg_config.deepspeed else None,
            packing=self.sg_config.sample_packing,
            use_liger_kernel=self.sg_config.should_apply_liger_kernel(),
            gradient_accumulation_steps=self.sg_config.gradient_accumulation_steps,

            lora_rank=self.sg_config.lora_rank,
            lora_alpha=self.sg_config.lora_alpha,
            lora_dropout=self.sg_config.lora_dropout,
            target_modules=self.sg_config.lora_target_modules,

            attn_impl='flash_attention_2',

            use_hf=True,

            use_ray=self.sg_config.use_ray,
            device_groups=json.dumps({
                'nproc_per_node': self.sg_config.ray_nproc_per_node,
                **{
                    name: {
                        'device': group.device,
                        'ranks': group.ranks,
                        'workers': group.workers
                    } for name, group in self.sg_config.ray_groups.items()
                }
            }) if self.sg_config.use_ray else None,
        )

        return swift_args


def sft_main(config: SFTConfig, args: DictDefault):
    SurogateSFT(config, args).main()
