from contextlib import contextmanager
from functools import partial

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.train.data_loader import DataLoaderShard, DataLoaderDispatcher, BatchSamplerShard
from surogate.utils.dist import seed_worker


class DataLoaderMixin(Trainer):
    template_processor: ChatTemplateProcessor

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

    def get_train_dataloader(self, skip_batches=0):
        dataloader = None
        if self.template_processor.sequence_parallel_size > 1:
            dataloader = self.get_sequence_parallel_dataloader(self.train_dataset, self._train_batch_size,
                                                               skip_batches=skip_batches)

        if dataloader is None:
            # Higher efficiency
            if self.train_dataset is None:
                raise ValueError('Trainer: training requires a train_dataset.')
            args = self.args
            train_dataset = self.train_dataset

            dataloader_params = {
                'collate_fn': self.data_collator,
                'num_workers': args.dataloader_num_workers,
                'pin_memory': args.dataloader_pin_memory,
                'persistent_workers': args.dataloader_persistent_workers,
                'prefetch_factor': args.dataloader_prefetch_factor
            }

            batch_sampler_params = {
                'drop_last': args.dataloader_drop_last,
                'shuffle': True,
                'data_seed': args.data_seed,
                'tp_size':
                    args.deepspeed['tensor_parallel']['autotp_size']
                    if args.deepspeed and 'tensor_parallel' in args.deepspeed
                    else 1,
            }

            if hasattr(train_dataset, '__len__'):
                batch_sampler = BatchSamplerShard(
                    len(train_dataset), batch_size=self._train_batch_size, **batch_sampler_params)
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)
                dataloader_params['batch_sampler'] = batch_sampler
                dataloader = DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)
            else:
                # IterableDataset
                if torch.distributed.is_initialized() and dataloader_params['prefetch_factor']:
                    dataloader_params['prefetch_factor'] = dataloader_params[
                                                               'prefetch_factor'] * torch.distributed.get_world_size()
                dataloader = DataLoader(train_dataset, batch_size=self._train_batch_size, **dataloader_params)
                dataloader = DataLoaderDispatcher(dataloader, self.accelerator.device, skip_batches=skip_batches)

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = None
        if self.template_processor.sequence_parallel_size > 1:
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError('Trainer: evaluation requires an eval_dataset.')
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            dataloader = self.get_sequence_parallel_dataloader(eval_dataset, self.args.eval_batch_size)
        if dataloader is None:
            return super().get_eval_dataloader(eval_dataset=eval_dataset)
        return dataloader

    def get_sequence_parallel_dataloader(self, dataset, batch_size, skip_batches=0):
        from surogate.train.sequence_parallel import sequence_parallel
        from surogate.train.sequence_parallel.utils import SequenceParallelSampler, SequenceParallelDispatcher

        data_collator = self.data_collator
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        if hasattr(dataset, '__len__'):
            sampler = SequenceParallelSampler(sequence_parallel, dataset, seed=42)
            dataloader_params = {
                'batch_size': batch_size,
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
            }

            if not isinstance(dataset, torch.utils.data.IterableDataset):
                if skip_batches > 0:
                    from accelerate.data_loader import SkipBatchSampler
                    sampler = SkipBatchSampler(sampler, skip_batches=skip_batches * batch_size)

                dataloader_params['sampler'] = sampler
                dataloader_params['drop_last'] = self.args.dataloader_drop_last
                dataloader_params['worker_init_fn'] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index)

            return DataLoaderShard(dataset, device=self.accelerator.device, **dataloader_params)
        else:
            dataloader_params = {
                'collate_fn': data_collator,
                'num_workers': self.args.dataloader_num_workers,
                'pin_memory': self.args.dataloader_pin_memory,
                'persistent_workers': self.args.dataloader_persistent_workers,
                'prefetch_factor': self.args.dataloader_prefetch_factor
            }
            if torch.distributed.is_initialized() and dataloader_params['prefetch_factor']:
                dataloader_params['prefetch_factor'] = dataloader_params[
                                                           'prefetch_factor'] * torch.distributed.get_world_size()

            dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_params)
            dataloader = SequenceParallelDispatcher(
                dataloader, sequence_parallel, self.accelerator.device, skip_batches=skip_batches)

            return dataloader
