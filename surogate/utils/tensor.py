from collections.abc import Mapping
from typing import Any, Union, List, Dict

import numpy as np
import torch

from surogate.utils.dist import get_device_count, get_dist_setting


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data

def _get_max_memory(device_ids: List[int]) -> Dict[Union[int, str], int]:
    """add feat in accelerate to support MP + DDP"""
    import psutil
    # Make sure CUDA is initialized on each GPU to have the right memory info.
    for i in device_ids:
        _ = torch.tensor([0], device=i)

    device_ids_set = set(device_ids)
    max_memory = {}
    for i in range(get_device_count()):
        max_memory[i] = 0
        if i in device_ids_set:
            max_memory[i] = torch.cuda.mem_get_info(i)[0]
    max_memory['cpu'] = psutil.virtual_memory().available
    return max_memory

def _sync_max_memory(max_memory: Dict[Union[int, str], int]) -> Dict[Union[int, str], int]:
    """Make sure that the model structure of MP(device_map) is the same, when using DDP."""
    max_memory_list = [v for k, v in max_memory.items() if (v > 0 and k != 'cpu')]
    _, local_rank, world_size, _ = get_dist_setting()
    src_tensor = torch.tensor(max_memory_list).to(local_rank)
    tgt_tensor_list = [torch.zeros_like(src_tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tgt_tensor_list, src_tensor)
    tgt_tensor = torch.stack(tgt_tensor_list, dim=0)
    new_max_memory_iter = iter(tgt_tensor.min(dim=0)[0].tolist())
    new_max_memory = {}
    for k, v in max_memory.items():
        new_max_memory[k] = v
        if v > 0 and k != 'cpu':
            new_max_memory[k] = next(new_max_memory_iter)
    return new_max_memory

def get_dataset_lengths(dataset, from_arrow=False):
    if "length" in dataset.column_names:
        lengths = np.array(dataset["length"])
    elif "position_ids" in dataset.column_names:
        position_ids = dataset["position_ids"]
        lengths = np.array([x[-1] + 1 for x in position_ids])
    else:
        if from_arrow:
            input_ids = dataset.data.column("input_ids")
            lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        else:
            input_ids = dataset["input_ids"]
            lengths = np.array([len(seq) for seq in input_ids])
    return lengths