"""Extract quantized weight GPU pointers from a running vLLM server via CUDA IPC.

In co-locate mode, vLLM's engine runs in a child process (AsyncMPClient).
We use collective_rpc to call extract_weight_ipc_handles() on the
ColocateWeightUpdateWorker, which returns CUDA IPC memory handles.
We then open those handles in the parent process using PyTorch's
rebuild_cuda_tensor() to get valid GPU pointers — achieving zero-copy sharing.

Supported quantization formats:
  - BnB NF4 (bitsandbytes 4-bit NormalFloat)
  - FP8 E4M3 (block quantized)
  - NVFP4 (FP4 E2M1 with 2D block scaling)
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from surogate.utils.logger import get_logger

logger = get_logger()

# Keep references to IPC tensors so their GPU memory stays mapped
_ipc_tensor_refs: list[torch.Tensor] = []

# Map from PyTorch dtype string to torch.dtype
_TORCH_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int32": torch.int32,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
}


def _rebuild_tensor_from_ipc(ipc_info: dict[str, Any]) -> torch.Tensor:
    """Reconstruct a CUDA tensor from IPC info using PyTorch's rebuild_cuda_tensor.

    Args:
        ipc_info: Dict from _get_ipc_info() in the worker process, containing
                  all storage IPC data and tensor metadata.

    Returns:
        A torch.Tensor mapped to the same GPU memory as in the worker process.
    """
    torch_dtype = _TORCH_DTYPE_MAP.get(ipc_info["torch_dtype"])
    if torch_dtype is None:
        raise ValueError(f"Unknown dtype: {ipc_info['torch_dtype']}")

    tensor = rebuild_cuda_tensor(
        torch.Tensor,                          # tensor_cls
        tuple(ipc_info["tensor_size"]),         # tensor_size
        tuple(ipc_info["tensor_stride"]),       # tensor_stride
        ipc_info["tensor_offset"],              # tensor_offset
        torch.storage.TypedStorage,             # storage_cls
        torch_dtype,                            # dtype
        ipc_info["storage_device"],             # storage_device
        ipc_info["storage_handle"],             # storage_handle
        ipc_info["storage_size_bytes"],         # storage_size_bytes
        ipc_info["storage_offset_bytes"],       # storage_offset_bytes
        ipc_info["requires_grad"],              # requires_grad
        ipc_info["ref_counter_handle"],         # ref_counter_handle
        ipc_info["ref_counter_offset"],         # ref_counter_offset
        ipc_info["event_handle"],               # event_handle
        ipc_info["event_sync_required"],        # event_sync_required
    )

    # Keep reference to prevent GC (which would unmap the IPC memory)
    _ipc_tensor_refs.append(tensor)
    return tensor


def _ipc_entry_to_external_weight(entry: dict[str, Any]) -> dict:
    """Convert an IPC entry (from collective_rpc) to an ExternalWeight dict.

    Reconstructs CUDA tensors from IPC handles and extracts raw GPU pointers
    for the C++ binding.
    """
    # Rebuild data tensor
    data_tensor = _rebuild_tensor_from_ipc(entry["data"])

    # Rebuild scales tensor
    scales_tensor = None
    if entry.get("scales"):
        scales_tensor = _rebuild_tensor_from_ipc(entry["scales"])

    # Rebuild meta tensor (double quant absmax)
    meta_tensor = None
    if entry.get("meta"):
        meta_tensor = _rebuild_tensor_from_ipc(entry["meta"])

    # Rebuild meta2 tensor (double quant offset or NVFP4 scale2)
    meta2_tensor = None
    if entry.get("meta2"):
        meta2_tensor = _rebuild_tensor_from_ipc(entry["meta2"])

    return {
        "name": entry["name"],
        "format": entry["format"],
        "M": entry["M"],
        "K": entry["K"],
        "block_size": entry["block_size"],
        "double_quant": entry["double_quant"],
        "double_quant_group_size": entry["double_quant_group_size"],
        "global_scale": entry["global_scale"],
        "device": entry["data"]["storage_device"],
        # Data
        "data_ptr": data_tensor.data_ptr(),
        "data_shape": list(data_tensor.shape),
        "data_dtype": entry["data"]["dtype"],
        # Scales
        "scales_ptr": scales_tensor.data_ptr() if scales_tensor is not None else 0,
        "scales_shape": list(scales_tensor.shape) if scales_tensor is not None else [],
        "scales_dtype": entry["scales"]["dtype"] if entry.get("scales") else "fp32",
        # Meta
        "meta_ptr": meta_tensor.data_ptr() if meta_tensor is not None else 0,
        "meta_shape": list(meta_tensor.shape) if meta_tensor is not None else [],
        "meta_dtype": entry["meta"]["dtype"] if entry.get("meta") else "fp32",
        # Meta2
        "meta2_ptr": meta2_tensor.data_ptr() if meta2_tensor is not None else 0,
        "meta2_shape": list(meta2_tensor.shape) if meta2_tensor is not None else [],
        "meta2_dtype": entry["meta2"]["dtype"] if entry.get("meta2") else "fp32",
        # Fuse swap: swap equal halves after dequant (e.g., gate/up reorder)
        "fuse_swap": entry.get("fuse_swap", False),
    }


def extract_vllm_weights_via_ipc(
    engine_client: Any,
    event_loop: asyncio.AbstractEventLoop,
) -> list[dict]:
    """Extract quantized weight GPU pointers from vLLM via CUDA IPC.

    Uses collective_rpc to call extract_weight_ipc_handles() on the
    ColocateWeightUpdateWorker inside vLLM's engine process, then
    reconstructs CUDA tensors from IPC handles in this process.

    Args:
        engine_client: The AsyncLLM engine client (captured from init_app_state)
        event_loop: The event loop running in the vLLM background thread

    Returns:
        List of ExternalWeight dicts with GPU pointers valid in this process.
    """
    # Call collective_rpc on the vLLM event loop from this (main) thread
    coro = engine_client.collective_rpc("extract_weight_ipc_handles")
    future = asyncio.run_coroutine_threadsafe(coro, event_loop)

    # collective_rpc returns list[list[result]] (one list per worker)
    # For single-GPU (dp=1), there's one worker
    try:
        results = future.result(timeout=60)
    except Exception as e:
        raise RuntimeError(f"collective_rpc extract_weight_ipc_handles failed: {e}") from e

    if not results or not results[0]:
        logger.warning("No IPC entries returned from vLLM worker")
        return []

    # results[0] is the list of IPC entries from the first (only) worker
    ipc_entries = results[0]

    # Reconstruct tensors from IPC handles and build ExternalWeight dicts
    external_weights = []
    for i, entry in enumerate(ipc_entries):
        try:
            ew = _ipc_entry_to_external_weight(entry)
            external_weights.append(ew)
        except Exception as e:
            logger.warning(f"Failed to reconstruct IPC tensor for {entry.get('name', '?')}: {e}")
            continue

    logger.info(f"Reconstructed {len(external_weights)} CUDA IPC weight tensors "
                f"({len(ipc_entries)} from worker)")
    return external_weights
