import random
from typing import Optional, Any, Union, Mapping
from typing_extensions import Literal

import numpy as np
import torch
from transformers import enable_full_determinism, set_seed
import surogate._surogate as _surogate
import surogate

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

def seed_everything(seed: Optional[int] = None, full_determinism: bool = False) -> int:
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    if full_determinism:
        enable_full_determinism(seed)
    else:
        set_seed(seed)
    return seed


def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
    """Compute cu_seqlens from packed position_ids.

    A new sequence starts whenever position_ids are not strictly consecutive
    (i.e., diff != 1). This mirrors HF packed-sequence detection.
    """
    position_ids = position_ids[0]
    if position_ids.numel() == 0:
        return torch.tensor([0], device=position_ids.device, dtype=torch.int32)

    diffs = position_ids[1:] - position_ids[:-1]
    boundaries = torch.where(diffs != 1)[0] + 1  # indices where a new seq starts
    starts = torch.cat(
        [torch.tensor([0], device=position_ids.device, dtype=boundaries.dtype), boundaries]
    )
    ends = torch.cat(
        [boundaries, torch.tensor([len(position_ids)], device=position_ids.device, dtype=boundaries.dtype)]
    )
    seq_lengths = ends - starts
    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], device=position_ids.device, dtype=seq_lengths.dtype), seq_lengths]),
        dim=0,
    )
    return cu_seqlens


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
    
def to_surogate_dtype(torch_dtype: torch.dtype) -> Literal['bf16', 'fp32']:
    if torch_dtype == torch.bfloat16:
        return "bf16"
    elif torch_dtype == torch.float32:
        return "fp32"
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
