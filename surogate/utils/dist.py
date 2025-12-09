import hashlib
import os
from contextlib import contextmanager
from typing import Tuple, Optional, Union

import torch
from datasets.utils._filelock import FileLock
from transformers import set_seed
from transformers.utils import strtobool, is_torch_cuda_available, is_torch_mps_available

from surogate.utils.fs import get_cache_dir
from surogate.utils.logger import get_logger

logger = get_logger()


def is_deepspeed_enabled():
    return strtobool(os.environ.get('ACCELERATE_USE_DEEPSPEED', '0'))

def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}

def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0

def is_local_master():
    local_rank = get_dist_setting()[1]
    return local_rank in {-1, 0}

def is_last_rank():
    rank, _, world_size, _ = get_dist_setting()
    return rank in {-1, world_size - 1}

def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size

def is_mp() -> bool:
    n_gpu = get_device_count()
    local_world_size = get_dist_setting()[3]
    assert n_gpu % local_world_size == 0, f'n_gpu: {n_gpu}, local_world_size: {local_world_size}'
    if n_gpu // local_world_size >= 2:
        return True
    return False

def is_mp_ddp() -> bool:
    _, _, world_size, _ = get_dist_setting()
    if is_dist() and is_mp() and world_size > 1:
        logger.info_once('Using MP(device_map) + DDP')
        return True
    return False

def print_distributed_config():
    rank, local_rank, world_size, local_world_size = get_dist_setting()
    is_distributed = world_size > 1
    logger.info(f"Distributed Environment: {'Yes' if is_distributed else 'No'}")
    if is_distributed:
        logger.info(f"World Size: {world_size}, Local Rank: {local_rank}")

def get_device_count() -> int:
    if is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0

def set_device(local_rank: Optional[Union[str, int]] = None):
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    if is_torch_cuda_available():
        torch.cuda.set_device(local_rank)

def get_torch_device():
    if is_torch_cuda_available():
        return torch.cuda
    else:
        return torch.cpu

def get_device_type() -> torch.device:
    device = torch.device("cpu")
    if is_torch_cuda_available():
        device = torch.device("cuda")
    return device

def get_current_device():
    if is_torch_cuda_available():
        current_device = torch.cuda.current_device()
    elif is_torch_mps_available():
        current_device = 'mps'
    else:
        current_device = 'cpu'
    return current_device

def get_device(local_rank: Optional[Union[str, int]] = None) -> str:
    if local_rank is None:
        local_rank = max(0, get_dist_setting()[1])
    local_rank = str(local_rank)
    if is_torch_mps_available():
        device = 'mps:{}'.format(local_rank)
    elif is_torch_cuda_available():
        device = 'cuda:{}'.format(local_rank)
    else:
        device = 'cpu'

    return device

_DISABLE_USE_BARRIER = False

@contextmanager
def disable_safe_ddp_context_use_barrier():
    global _DISABLE_USE_BARRIER
    _DISABLE_USE_BARRIER = True
    try:
        yield
    finally:
        _DISABLE_USE_BARRIER = False

@contextmanager
def safe_ddp_context(hash_id: Optional[str], use_barrier: bool = True):
    if _DISABLE_USE_BARRIER:
        use_barrier = False
    if use_barrier and torch.distributed.is_initialized():
        if is_dist():
            if not is_master():
                torch.distributed.barrier()
            if not is_local_master():
                # Compatible with multi-machine scenarios,
                # where each machine uses different storage hardware.
                torch.distributed.barrier()
        yield
        if is_dist():
            if is_master():
                torch.distributed.barrier()
            if is_local_master():
                torch.distributed.barrier()
    elif hash_id is not None:
        lock_dir = os.path.join(get_cache_dir(), 'lockers')
        os.makedirs(lock_dir, exist_ok=True)
        file_path = hashlib.sha256(hash_id.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(lock_dir, file_path)
        with FileLock(file_path):
            yield
    else:
        yield


def compute_and_broadcast(fn):
    """
    Compute a value using the function 'fn' only on the specified rank (default is 0).
    The value is then broadcasted to all other ranks.

    Args:
    - fn (callable): A function that computes the value. This should not have any side effects.
    - rank (int, optional): The rank that computes the value. Default is 0.

    Returns:
    - The computed value (int or float).
    """
    cur_device = f"{get_device_type()}:{get_current_device()}"
    if is_master():
        value_scalar = fn()
        value_tensor = torch.tensor(
            value_scalar, device=cur_device, dtype=torch.float32
        )
    else:
        value_tensor = torch.tensor(
            0.0, device=cur_device, dtype=torch.float32
        )  # Placeholder tensor

    # Broadcast the tensor to all processes.
    with safe_ddp_context(hash_id=None):
        torch.distributed.broadcast(value_tensor, src=0)

    # Convert the tensor back to its original type (int or float)
    if value_tensor == value_tensor.int():
        return int(value_tensor.item())
    return float(value_tensor.item())


def gather_from_all_ranks(fn, world_size=1):
    """
    Run a callable 'fn' on all ranks and gather the results on the specified rank.

    Args:
    - fn (callable): A function that computes the value. This should not have any side effects.
    - rank (int, optional): The rank that gathers the values. Default is 0.
    - world_size (int, optional): Total number of processes in the current distributed setup.

    Returns:
    - A list of computed values from all ranks if on the gathering rank, otherwise None.
    """
    value_scalar = fn()
    value_tensor = torch.tensor(
        value_scalar, device=f"{get_device_type()}:{get_current_device()}"
    ).float()

    # Placeholder tensor for gathering results
    if is_master():
        gathered_tensors = [torch.zeros_like(value_tensor) for _ in range(world_size)]
    else:
        gathered_tensors = None

    torch.distributed.gather(value_tensor, gather_list=gathered_tensors, dst=0)

    if is_master():
        # Convert tensors back to their original type (int or float)
        gathered_values = []
        for tensor in gathered_tensors:
            if tensor == tensor.int():
                gathered_values.append(int(tensor.item()))
            else:
                gathered_values.append(float(tensor.item()))
        return gathered_values
    return None

def reduce_and_broadcast(fn1, fn2):
    """
    Run a callable 'fn1' on all ranks, gather the results, reduce them using 'fn2',
    and then broadcast the reduced result to all ranks.

    Args:
    - fn1 (callable): A function that computes the value on each rank.
    - fn2 (callable): A reduction function that takes a list of values and returns a single value.
    - world_size (int, optional): Total number of processes in the current distributed setup.

    Returns:
    - The reduced and broadcasted value.
    """

    # Gather values from all ranks using fn1
    if not is_dist():
        return fn2([fn1()])

    gathered_values = gather_from_all_ranks(fn1, world_size=torch.distributed.get_world_size())

    # Use compute_and_broadcast to compute the reduced value on the main process
    # and then broadcast it to all ranks
    return compute_and_broadcast(lambda: fn2(gathered_values))


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)