import torch
import triton
from triton.language.extra import libdevice

from surogate.utils.dist import get_device_count

DEVICE_COUNT: int = get_device_count()

if DEVICE_COUNT > 1:
    torch_gpu_device = torch.cuda.device
else:
    from contextlib import nullcontext


    def torch_gpu_device(device):
        return nullcontext()

triton_tanh = libdevice.tanh

MAX_FUSED_SIZE: int = 65536


def calculate_settings(n: int) -> (int, int):
    BLOCK_SIZE: int = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    dtype = None
    for dtype_name in check_order:
        if dtype is None:
            dtype = getattr(config, dtype_name, None)
    return dtype


def _get_dtype(dtype):
    __DTYPE_MAP = {
        "float32": torch.float32,
        torch.float32: torch.float32,
        "float16": torch.float16,
        torch.float16: torch.float16,
        "bfloat16": torch.bfloat16,
        torch.bfloat16: torch.bfloat16,
    }

    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            dtype = dtype.lower()
            return getattr(torch, dtype, None)
        elif isinstance(dtype, torch.dtype):
            return dtype
    return None


def move_to_device(target_device, *tensors):
    """
    Move multiple tensors to target device if they're not already there.

    Args:
        target_device: The target device to move tensors to
        *tensors: Variable number of tensors to potentially move

    Returns:
        tuple: The tensors on the target device (same objects if already on device, new if moved)
    """
    if isinstance(target_device, int):
        target_device = torch.device(target_device)
    elif isinstance(target_device, str):
        # if string we expect it to be a device name like "cuda:0"
        target_device = torch.device(target_device)
    elif isinstance(target_device, torch.device):
        pass
    else:
        raise ValueError(f"Invalid target device: {target_device}")
    moved_tensors = []
    for tensor in tensors:
        if tensor.device != target_device:
            moved_tensors.append(tensor.to(target_device))
        else:
            moved_tensors.append(tensor)
    return tuple(moved_tensors) if len(moved_tensors) > 1 else moved_tensors[0]

