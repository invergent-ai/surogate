import json
import os
from typing import List
import numpy as np
import torch
from swift.llm import get_template
from surogate.config.sft_config import SFTConfig
from surogate.datasets.datasets import load_datasets, get_default_process_count
from surogate.eval.datasets import DatasetLoader
from surogate.loaders.loader import get_model_and_tokenizer
from surogate.sft.early_stop_callback import EarlyStopCallback
from surogate.sft.train_callback import TrainAdapterCallback
from surogate.sft.trainer import SurogateSeq2SeqTrainer
from surogate.utils.command import SurogateCommand
from surogate.utils.distributed import is_main_process
from surogate.utils.logger import get_logger
from swift.llm.train.tuner import get_multimodal_target_regex
from swift.llm.dataset.loader import DatasetLoader
from swift.llm.dataset.utils import EncodePreprocessor, LazyLLMDataset, IterablePackingDataset, PackingDataset
from swift.utils import seed_everything, check_json_format, find_all_linears, find_embedding, get_model_parameter_info
from swift.tuners import LoraConfig, get_peft_model
from swift.utils.env import is_master
from swift.utils.io_utils import append_to_jsonl

logger = get_logger()


class SurogateSFT(SurogateCommand):
    config: SFTConfig

    def __init__(self, **kwargs):
        super().__init__(SFTConfig, **kwargs)

        if self.config.seed:
            seed_everything(self.config.seed)

        self.model, self.tokenizer = get_model_and_tokenizer(self.config.model)

        self.template = get_template(
            self.config.model_meta.template,
            padding_free=self.config.padding_free,
            processor=self.tokenizer)
        self.template.set_mode('train')

        if self.config.model_meta.is_multimodal and self.config.sample_packing and not self.template.support_padding_free:
            logger.warning('Disabling sample packing as the template does not support padding free.')
            self.config.sample_packing = False

        self.callbacks = [TrainAdapterCallback(self.config), EarlyStopCallback(self.config)]
        self.train_msg = {}

    def run(self):
        train_dataset, val_dataset = self._prepare_datasets()
        self._save_training_config()
        data_collator = self.template.data_collator
        self.model = self.prepare_model()

        logger.debug(f'Model Architecture: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = SurogateSeq2SeqTrainer(
            config=self.config,
            model=self.model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
        )

        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')
        logger.info(f'The logging file will be saved in: {logging_path}')

        try:
            trainer.train(trainer.args.resume_from_checkpoint)
        finally:
            res = self._save_trainer_state(trainer)

        return res

    def prepare_model(self):
        self.model.requires_grad_(False)
        model = self.prepare_adapter()

        # fix bug: Attempting to unscale FP16 gradients.
        #   peft: https://github.com/huggingface/peft/issues/1249
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float16:
                logger.info_once('Convert trainable parameters from fp16 to fp32.')
                p.data = p.data.to(dtype=torch.float32)

        return model

    def prepare_adapter(self):
        target_modules = self.config.lora_target_modules.copy()
        if 'all-linear' in target_modules:
            if self.config.model_meta.is_multimodal:
                return get_multimodal_target_regex(
                    self.model,
                    include_embedding='all-embedding' in target_modules)
            else:
                target_modules.remove('all-linear')
                target_modules += find_all_linears(self.model)

        if 'all-embedding' in target_modules:
            target_modules.remove('all-embedding')
            target_modules += find_embedding(self.model)

        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=self.config.lora_rank,
            target_modules=target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias='none',
            init_lora_weights=True
        )

        return get_peft_model(self.model, lora_config)

    def _prepare_datasets(self):
        train_dataset = load_datasets(self.config.datasets, self.args, self.config.save_path, self.config.seed)
        if self.config.validation_datasets:
            val_dataset = load_datasets(
                self.config.validation_datasets,
                self.args,
                self.config.save_path,
                self.config.seed)
        else:
            train_dataset, val_dataset = DatasetLoader.post_process(
                train_dataset,
                split_dataset_ratio=self.config.validation_split_ratio,
                shuffle=False,
                random_state=np.random.RandomState(self.config.seed))

        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        return self._post_process_datasets([train_dataset, val_dataset])

    def _encode_dataset(self, train_dataset, val_dataset):
        template = self.template
        datasets = [train_dataset, val_dataset]

        origin_template_model = template.model
        template.model = None  # Avoid serializing the model.

        for i, dataset in enumerate(datasets):
            if not self.config.stream_datasets:
                preprocessor = EncodePreprocessor(template=template, pre_tokenize=self.config.model_meta.is_multimodal)
                dataset = preprocessor(
                    dataset,
                    num_proc=get_default_process_count(),
                    load_from_cache_file=False,
                    strict=False,
                    batch_size=100 if self.config.model_meta.is_multimodal else 1000)
            datasets[i] = dataset

        template.model = origin_template_model
        return datasets

    def _post_process_datasets(self, datasets: List) -> List:
        template = self.template
        for i, dataset in enumerate(datasets):
            if self.config.model_meta.is_multimodal and not self.config.stream_datasets:
                dataset = LazyLLMDataset(dataset, template.encode, strict=False, random_state=self.config.seed)

            if self.config.sample_packing:
                packing_dataset_cls = IterablePackingDataset if self.config.stream_datasets else PackingDataset
                dataset = packing_dataset_cls(
                    template,
                    dataset,
                    num_proc=get_default_process_count(),
                    packing_length=self.config.sequence_len or self.config.model_info.max_model_len,
                    load_from_cache_file=False)
            elif self.config.stream_datasets:
                preprocessor = EncodePreprocessor(template=template)
                dataset = preprocessor(
                    dataset,
                    num_proc=get_default_process_count(),
                    load_from_cache_file=False)

            datasets[i] = dataset

        return datasets

    def _save_training_config(self):
        if is_main_process():
            os.makedirs(self.config.save_path, exist_ok=True)
            fpath = os.path.join(self.config.save_path, 'args.json')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(self.config), f, ensure_ascii=False, indent=2)

    def _save_trainer_state(self, trainer):
        training_args = trainer.args
        state = trainer.state

        if hasattr(state, 'last_model_checkpoint'):
            if self.args.create_checkpoint_symlink:
                last_checkpoint = os.path.join(self.args.output_dir, 'last')
                best_checkpoint = os.path.join(self.args.output_dir, 'best')
                if is_master():
                    os.symlink(state.last_model_checkpoint, last_checkpoint)
                    os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint
                state.best_model_checkpoint = best_checkpoint
        else:
            state.last_model_checkpoint = None

        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')

        self.train_msg.update({
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': getattr(state, 'max_memory', None),
        })

        if is_master():
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')
            append_to_jsonl(jsonl_path, self.train_msg, strict=False)

        return self.train_msg
