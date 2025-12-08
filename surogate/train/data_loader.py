from typing import Optional

import torch
from torch.utils.data import DataLoader

from surogate.core.datasets.progress import create_hfhub_tqdm
from surogate.utils.tensor import to_device


class BatchSamplerShard:
    """
    Distributed batch sampler that shards data across processes, with special handling for tensor parallelism (TP).
    In distributed training, you want each GPU/process to work on a different portion of the data.
    This sampler divides the dataset and ensures each rank gets unique, non-overlapping indices.
    """
    def __init__(self,
                 total_samples: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 data_seed: Optional[int],
                 tp_size: int = 1):
        self.tp_size = tp_size
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed

    @property
    def rank(self):
        return (torch.distributed.get_rank() // self.tp_size) if torch.distributed.is_initialized() else 0

    @property
    def world_size(self):
        return (torch.distributed.get_world_size() // self.tp_size) if torch.distributed.is_initialized() else 1

    def __iter__(self):
        # Calculates which slice of indices this rank owns
        start_idx = self.rank * self.total_samples
        if self.shuffle:
            # If shuffling, generates a global permutation using the same seed on all ranks, then slices out its portion.
            # This ensures no overlap while maintaining reproducibility.
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[start_idx:start_idx + self.total_samples]
        else:
            total_idx = list(range(start_idx, start_idx + self.total_samples))

        # Batch the indices and yield them
        batch = []
        # Last batch if not complete will be dropped.
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        return

    def set_epoch(self, epoch: int):
        """
        Changes the random seed each epoch so shuffling differs across epochs
        """
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class DataLoaderShard(DataLoader):
    """
    Automatic device placement and epoch synchronization for distributed samplers.
    Moves tensors to the right GPU and keeps distributed samplers properly synchronized.
        ┌──────┐ ┌──────┐
        │Rank 0│ │Rank 1│
        │reads │ │reads │
        │shard0│ │shard1│
        └──────┘ └──────┘
    """
    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        """
        Must be called before each epoch to ensure different shuffling across epochs.
        Without this, every epoch would use the same shuffle order, hurting training.
        """
        if self.batch_sampler is not None and hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        """
        Intercepts each batch from the parent iterator and moves it to the specified device before yielding.
        """
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item


class DataLoaderDispatcher:
    """
    Distributed data loading pattern: centralized dispatch rather than distributed sharding.
    Instead of each rank reading its own data slice, rank 0 reads everything and scatters batches to other ranks.
         ┌──────────┐
         │  Rank 0  │ ← reads ALL data
         │ (master) │
         └────┬─────┘
              │ scatter
         ┌────┴────┐
         ▼         ▼
       Rank 0    Rank 1
    """
    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return torch.distributed.get_rank(self.group) if torch.distributed.is_initialized() else 0

    @property
    def world_size(self):
        return torch.distributed.get_world_size(self.group) if torch.distributed.is_initialized() else 1

    @property
    def group(self):
        return torch.distributed.group.WORLD if torch.distributed.is_initialized() else 1

    def _scatter_object_list(self, inputs):
        if not torch.distributed.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = torch.distributed.get_global_rank(self.group, 0)
        torch.distributed.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            _tqdm = create_hfhub_tqdm("DataLoaderDispatcher: ")
            for _ in _tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                [next(base_iter) for _ in range(self.world_size)]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                # Only rank 0 pulls from the actual dataloader, collecting `world_size` batches at once.
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    # When exhausted, send None to all ranks as a termination signal
                    data = [None] * self.world_size

                # sends batch i to rank i
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data