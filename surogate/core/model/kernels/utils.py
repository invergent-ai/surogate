import functools

import torch
import triton
import triton.language as tl
from transformers import PretrainedConfig

from surogate.utils.dist import get_device_count, get_device_type

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2
DEVICE_COUNT: int = get_device_count()
DEVICE_TYPE_TORCH = "cuda"
HAS_TORCH_DTYPE = "torch_dtype" in PretrainedConfig.__doc__

if DEVICE_COUNT > 1:
    torch_gpu_device = torch.cuda.device
else:
    from contextlib import nullcontext
    def torch_gpu_device(device):
        return nullcontext()

from triton.language.extra import libdevice
triton_tanh = libdevice.tanh
triton_cast = tl.cast

def is_cdna():
    return False

def calculate_settings(
        n: int,
) -> (
        int,
        int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
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

def fast_linear_forward(proj, X, temp_lora = None, out = None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch.matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant, bias)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)
        out = torch.matmul(X, W, out = out)

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(
                X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora
            )
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        out = out.view(bsz, 1, out_dim)

    if bias is not None:
        out += bias

    return out

def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if isinstance(W, Float8Tensor):
        assert W.ndim == 2
        if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
            # In the backward pass, rowwise scaled becomes colwise scaled after we
            # transpose the weight tensor. Use this case to detect backward.
            # TODO: would be simpler if we simply don't call `matmul_lora` in backward
            W = W.dequantize()
        else:
            W = W.contiguous()
        out = torch_matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer = True)
        out = torch_matmul(X, W.t(), out = out)

    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch.matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out

# to move multiple tensors to the same device
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

def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    if HAS_TORCH_DTYPE:
        check_order = ['torch_dtype', 'dtype']
    dtype = None
    for dtype_name in check_order:
        if dtype is None:
            dtype = getattr(config, dtype_name, None)
    return dtype

__DTYPE_MAP = {
    "float32": torch.float32,
    torch.float32: torch.float32,
    "float16": torch.float16,
    torch.float16: torch.float16,
    "bfloat16": torch.bfloat16,
    torch.bfloat16: torch.bfloat16,
}
def _get_dtype(dtype):
    try:
        return __DTYPE_MAP[dtype]
    except:
        if type(dtype) is str:
            dtype = dtype.lower()
            return getattr(torch, dtype, None)
        elif isinstance(dtype, torch.dtype):
            return dtype
    return None
