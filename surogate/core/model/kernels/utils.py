import ctypes
import functools
import inspect
import os

import torch
import triton
import triton.language as tl
from transformers import PretrainedConfig
import bitsandbytes as bnb
from torchao.quantization import Float8Tensor

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

_CUDA_STREAMS = {
    (index := torch.cuda.device(i).idx): ctypes.c_void_p(
        torch._C._cuda_getCurrentRawStream(index)
    )
    for i in range(DEVICE_COUNT)
}
CUDA_STREAMS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
WEIGHT_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
ABSMAX_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
for k, v in _CUDA_STREAMS.items():
    CUDA_STREAMS[k] = v
CUDA_STREAMS = tuple(CUDA_STREAMS)
del _CUDA_STREAMS

def is_cdna():
    return False


@functools.lru_cache(1)
def determine_compile_threads():
    # See https://github.com/pytorch/pytorch/blob/ab2294d8289a7757a2fc321cdefac88e2b378edf/torch/_inductor/config.py#L771
    cpu_count = os.cpu_count()
    return min(32, max(4, cpu_count))


def get_torch_compile_options(
        epilogue_fusion=True,
        max_autotune=False,
        shape_padding=True,
        debug=False,
        cudagraphs=False,
        coordinate_descent_tuning=False,
        logging=False,
        combo_kernels=False,
        group_fusion=True,
        memory_planning=True,
        multi_kernel=False,
        use_block_ptr=False,
):
    # https://github.com/pytorch/pytorch/blob/c665594c1edca9a507b0ec8b18ab74a0ecb65bc3/torch/_inductor/config.py#L1283
    # Needs integer
    multi_kernel = 1 if multi_kernel else 0

    # Instead of Inductor Compilation:
    try:
        import torch._inductor.async_compile
        from torch.hub import tqdm
        def replaced_tqdm(*args, **kwargs):
            kwargs["desc"] = "Compiling kernels"
            return tqdm(*args, **kwargs)

        torch._inductor.async_compile.tqdm = replaced_tqdm
    except:
        print("Failed editing tqdm to replace Inductor Compilation:")

    torch_compile_options = {
        "epilogue_fusion": epilogue_fusion,
        "max_autotune": max_autotune,
        "shape_padding": shape_padding,
        "trace.enabled": debug,
        "triton.cudagraphs": cudagraphs,
        "debug": debug,
        "dce": True,
        "memory_planning": memory_planning,
        "coordinate_descent_tuning": coordinate_descent_tuning,
        "trace.graph_diagram": debug,
        "compile_threads": determine_compile_threads(),
        # Auto detects via https://github.com/unslothai/unsloth-zoo/pull/187
        "group_fusion": group_fusion,  # [DEPRECATED]
        "disable_progress": not logging,
        "verbose_progress": logging,

        "triton.multi_kernel": multi_kernel,  # RuntimeError: name 'multi_kernel_0' is not defined
        "triton.use_block_ptr": use_block_ptr,
        "triton.enable_persistent_tma_matmul": True,
        "triton.autotune_at_compile_time": False,
        "triton.cooperative_reductions": False,
        # "reorder_for_compute_comm_overlap"  : True, # Fails for single GPU
        "cuda.compile_opt_level": "-O2",
        "cuda.enable_cuda_lto": True,
        # "cuda.use_fast_math"                : True, # Disable fast math
        # Causes incompatible gradient sizes on 2.6
        # And TypeError: bad operand type for unary -: 'SymbolicCallArg'
        "combo_kernels": combo_kernels,
        "benchmark_combo_kernel": True,
        "combo_kernel_foreach_dynamic_shapes": True,
    }
    final_torch_compile_options = {}
    inductor_config_source = inspect.getsource(torch._inductor.config)
    for key, value in torch_compile_options.items():
        splits = key.split(".")
        if all(k in inductor_config_source for k in splits):
            final_torch_compile_options[key] = value
    return final_torch_compile_options


torch_compile_options = get_torch_compile_options(
    epilogue_fusion=True,
    max_autotune=False,
    shape_padding=True,
    debug=False,
    cudagraphs=False,
    coordinate_descent_tuning=False,
    logging=False,
    combo_kernels=False,
    memory_planning=False,
    multi_kernel=False,
    use_block_ptr=False,
)

torch_compile = functools.partial(
    torch.compile,
    options=torch_compile_options,
)


def calculate_settings(n: int) -> (int, int):
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


def dtype_from_config(config):
    check_order = ['dtype', 'torch_dtype']
    if HAS_TORCH_DTYPE:
        check_order = ['torch_dtype', 'dtype']
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


@torch.inference_mode
def fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    if isinstance(W, Float8Tensor):
        return W.dequantize()
    if quant_state is None:
        return W
    if W.dtype == torch.float8_e4m3fn:
        return weight_dequant(W, quant_state)
    if type(quant_state) is not list:
        # New quant_state as a class
        # https://github.com/TimDettmers/bitsandbytes/pull/763/files
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        # Old quant_state as a list of lists
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass
    global CUDA_STREAMS
    device = W.device
    device_index = device.index
    CUDA_STREAM = CUDA_STREAMS[device_index]

    n_elements_absmax = absmax.numel()

    # Create weight matrix
    if use_global_buffer:
        # Use same buffers for faster inference
        size = shape[0] * shape[1]
        global WEIGHT_BUFFERS
        global ABSMAX_BUFFERS
        WEIGHT_BUFFER = WEIGHT_BUFFERS[device_index]
        ABSMAX_BUFFER = ABSMAX_BUFFERS[device_index]
        if WEIGHT_BUFFER is None:
            WEIGHT_BUFFERS[device_index] = WEIGHT_BUFFER = torch.empty(
                size, dtype=dtype, device=device, requires_grad=False
            )
            ABSMAX_BUFFERS[device_index] = ABSMAX_BUFFER = torch.empty(
                n_elements_absmax,
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )

        if size > WEIGHT_BUFFER.numel():
            WEIGHT_BUFFER.resize_(size)
        if n_elements_absmax > ABSMAX_BUFFER.numel():
            ABSMAX_BUFFER.resize_(n_elements_absmax)

        out = WEIGHT_BUFFER[:size].view(shape)
        out_absmax = ABSMAX_BUFFER[:n_elements_absmax]
    else:
        if out is None:
            out = torch.empty(
                shape, dtype=dtype, device=device, requires_grad=False
            )
        else:
            assert out.shape == shape
            assert out.dtype == dtype
        out_absmax = torch.empty(
            n_elements_absmax,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
    pass

    # NF4 dequantization of statistics
    ptr_out_absmax = bnb.functional.get_ptr(out_absmax)
    with torch_gpu_device(device):
        bnb.functional.lib.cdequantize_blockwise_fp32(
            bnb.functional.get_ptr(code2),
            bnb.functional.get_ptr(absmax),
            bnb.functional.get_ptr(absmax2),
            ptr_out_absmax,
            ctypes.c_int(blocksize2),
            ctypes.c_int(n_elements_absmax),
            CUDA_STREAM,
        )
        out_absmax += offset

        # Dequantize W
        fx = (
            bnb.functional.lib.cdequantize_blockwise_fp16_nf4
            if dtype == torch.float16
            else bnb.functional.lib.cdequantize_blockwise_bf16_nf4
        )
        fx(
            bnb.functional.get_ptr(None),
            bnb.functional.get_ptr(W),
            ptr_out_absmax,
            bnb.functional.get_ptr(out),
            ctypes.c_int(blocksize),
            ctypes.c_int(out.numel()),
            CUDA_STREAM,
        )
    pass
    # Careful returning transposed data
    is_transposed = True if W.shape[0] == 1 else False
    return out.t() if is_transposed else out


def _maybe_fake_quantize_activations(
        X: torch.Tensor, proj: torch.nn.Module
) -> torch.Tensor:
    """
    If QAT is enabled, fake quantize the input activations.
    Otherwise, just return the input activations as is.
    Weights are fake quantized separately in `get_lora_parameters`.
    """
    base_layer = getattr(proj, "base_layer", proj)
    activation_fake_quantizer = getattr(base_layer, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        X = activation_fake_quantizer(X)
    return X


def get_lora_parameters(proj):
    """
    Return a 5-tuple of (weight, weight quant_state, lora A, lora B, and lora scale).
    If QAT is enabled, additionally fake quantize the base layer and lora weights.
    """
    # For DPO or disabled adapters
    base_layer = getattr(
        proj, "base_layer", proj
    )  # (proj.base_layer if hasattr(proj, "base_layer") else proj)

    W = base_layer.weight

    # Get quant state for 4bit or FP8
    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(base_layer, "quant_method", None) == "fp8":
        # we need to somehow store and pass this information :)
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    # Optionally apply fake quantization to lora weights for QAT
    lora_A_linear = proj.lora_A[adapter]
    lora_B_linear = proj.lora_B[adapter]
    A = lora_A_linear.weight
    B = lora_B_linear.weight
    if hasattr(lora_A_linear, "weight_fake_quantizer"):
        lora_A_fake_quantizer = getattr(lora_A_linear, "weight_fake_quantizer", None)
        if lora_A_fake_quantizer is not None:
            A = lora_A_fake_quantizer(A)
    if hasattr(lora_B_linear, "weight_fake_quantizer"):
        lora_B_fake_quantizer = getattr(lora_B_linear, "weight_fake_quantizer", None)
        if lora_B_fake_quantizer is not None:
            B = lora_B_fake_quantizer(B)

    return (
        W,
        W_quant,
        A,
        B,
        proj.scaling[adapter],
    )


def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16):
    if s.shape[1] == 1:
        # this is row quantized weight, just simple multiplication suffices
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            # sometimes, this is called with the transpose of the weight. Adjust for that.
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(f"Incompatible shapes {x.shape = }, {s.shape = }")
        return y
    else:
        # this is block quantized weight
        return weight_dequant_block(x, s, dtype=dtype)


def weight_dequant_block(
        x: torch.Tensor, s: torch.Tensor, block_size: int = 128, dtype=torch.bfloat16
) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)
