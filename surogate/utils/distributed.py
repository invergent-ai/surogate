import os
from datetime import timedelta

import torch.distributed as dist

from accelerate import PartialState

distributed_state: PartialState | None = None


def init_distributed_state():
    global distributed_state
    if distributed_state is None:
        timeout = int(os.environ.get("SUROGATE_NCCL_TIMEOUT", 1800))
        try:
            distributed_state = PartialState(timeout=timedelta(seconds=timeout))
        except ValueError:
            pass


def get_distributed_state() -> PartialState | None:
    return distributed_state


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    init_distributed_state()

    if distributed_state is None:
        return False

    return distributed_state.use_distributed and distributed_state.initialized


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def is_local_main_process() -> bool:
    if get_distributed_state() is None:
        return os.environ.get("LOCAL_RANK", "0") == "0"
    return PartialState().is_local_main_process


def is_main_process() -> bool:
    """
    Check if the current process is the main process. If not in distributed mode,
    always return `True`.

    We use a simpler logic when the distributed state is not initialized: we just log
    on the 0-th local rank.

    Returns:
        `True` if the current process is the main process, `False` otherwise.
    """
    if get_distributed_state() is None:
        return os.environ.get("LOCAL_RANK", "0") == "0"
    if not is_distributed():
        return True
    return dist.get_rank() == 0
