import json
from functools import partial
from pathlib import Path
from typing import List

from swift.llm import BaseArguments, TrainArguments, DATASET_MAPPING, DatasetMeta, AutoPreprocessor
from swift.llm.dataset.loader import load_dataset
from swift.llm.train import SwiftSft
from swift.plugin import extra_callbacks
from swift.ray import RayHelper
from transformers import SchedulerType

from surogate.config.dataset_config import DatasetConfig
from surogate.config.enums import SurogateDatasetType
from surogate.config.sft_config import SFTConfig
from surogate.datasets.conversation import ConversationPreprocessor
from surogate.datasets.datasets import get_default_process_count
from surogate.datasets.instruction import InstructionPreprocessor
from surogate.datasets.loader import _check_if_hub_dataset, swift_load_dataset
from surogate.datasets.text import TextPreprocessor
from surogate.sft.callback import SurogateSftCallback
from surogate.sft.deepspeed import DEEPSPEED_CONFIGS
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()

@RayHelper.worker(group=['default'])
class SurogateSFT(SurogateCommand, SwiftSft):
    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args, swift=True)
        config.__post_init__()
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
        self.args.val_dataset = list(map(partial(to_swift_dataset_id, self.sg_config.validation_datasets), val_dataset_ids))


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
        self.callbacks.append(SurogateSftCallback(self.sg_config))


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
