"""vLLM worker extension for colocate mode: zero-copy weight sharing with surogate.

In colocate mode, surogate's C++ engine and vLLM share GPU memory.
This worker extension:
  1. Extracts CUDA IPC handles for quantized weights (for cross-process sharing)
  2. Receives LoRA weight tensor updates from surogate

CUDA IPC allows the parent process (surogate trainer) to map the same GPU
memory that vLLM's worker process allocated, achieving zero-copy sharing.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_colocate")

# Keep references to reordered tensors so their GPU memory stays valid for IPC
_reordered_refs: list[torch.Tensor] = []

# QuantFormat enum values must match C++ qlora::QuantFormat
QUANT_FORMAT_NONE = 0
QUANT_FORMAT_BNB_NF4 = 1
QUANT_FORMAT_FP8_PER_BLOCK = 2
QUANT_FORMAT_FP4_BLOCK_2D = 3

# ETensorDType string names (must match C++ dtype_from_str)
DTYPE_MAP = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.int8: "int8",
    torch.uint8: "byte",
    torch.int32: "int32",
}


def _dtype_str(t: torch.Tensor) -> str:
    if t.dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    return DTYPE_MAP.get(t.dtype, "byte")


def _get_ipc_info(tensor: torch.Tensor) -> dict[str, Any]:
    """Get CUDA IPC handle and metadata for a GPU tensor.

    Uses PyTorch's _share_cuda_() which returns all data needed for
    cross-process reconstruction via rebuild_cuda_tensor().

    PyTorch 2.10+ _share_cuda_() returns 8 values:
      (device, storage_handle, storage_size_bytes, storage_offset_bytes,
       ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
    """
    storage = tensor.untyped_storage()
    share_data = storage._share_cuda_()
    # Unpack all values (PyTorch 2.10+: 8 values)
    (device_idx, storage_handle, storage_size_bytes, storage_offset_bytes,
     ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = share_data

    return {
        # Storage IPC info (for _new_shared_cuda / rebuild_cuda_tensor)
        "storage_device": device_idx,
        "storage_handle": bytes(storage_handle),
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "ref_counter_handle": bytes(ref_counter_handle),
        "ref_counter_offset": ref_counter_offset,
        "event_handle": bytes(event_handle),
        "event_sync_required": event_sync_required,
        # Tensor metadata (for reconstructing the tensor view)
        "tensor_size": list(tensor.shape),
        "tensor_stride": list(tensor.stride()),
        "tensor_offset": tensor.storage_offset(),
        "requires_grad": tensor.requires_grad,
        # Surogate dtype string
        "dtype": _dtype_str(tensor),
        # PyTorch dtype string (for rebuild_cuda_tensor)
        "torch_dtype": str(tensor.dtype),
    }


class ColocateWeightUpdateWorker(Worker):
    """vLLM worker extension for zero-copy weight updates from surogate's GPU memory."""

    # vLLM fuses projections (qkv_proj, gate_up_proj) but Surogate's mapping
    # table uses individual HF names (q_proj, k_proj, v_proj, gate_proj, up_proj).
    # Map fused name → first individual name so the C++ Fuse lookup finds it.
    _VLLM_FUSED_TO_HF = {
        "qkv_proj": "q_proj",
        "gate_up_proj": "gate_proj",
    }

    # Fused projections where vLLM and surogate use different row orderings.
    # vLLM gate_up_proj stores [gate; up], but surogate's SwiGLUMLP expects
    # fuse(up_proj, gate_proj) = [up; gate]. We must swap the halves.
    # QKV doesn't need swapping: both use [q; k; v] order.
    _FUSE_SWAP_NAMES = {"gate_up_proj"}

    def init_broadcaster(self) -> None:
        """No-op: colocate mode doesn't need a broadcaster."""
        ...

    @staticmethod
    def _normalize_vllm_name(name: str) -> str:
        """Convert vLLM parameter name to HuggingFace-compatible name.

        Handles two naming differences:
          1. LoRA wrappers add '.base_layer.' (e.g., o_proj.base_layer.weight)
          2. vLLM fuses projections (qkv_proj → q_proj, gate_up_proj → gate_proj)
        """
        # Strip .base_layer (LoRA wrapper)
        name = name.replace(".base_layer.", ".")
        # Map fused projections to first individual name
        for fused, first_hf in ColocateWeightUpdateWorker._VLLM_FUSED_TO_HF.items():
            name = name.replace(f".{fused}.", f".{first_hf}.")
        return name

    @staticmethod
    def _swap_rows(
        tensor: torch.Tensor,
        first_rows: int,
        second_rows: int,
    ) -> torch.Tensor:
        """Swap two row-partitions of a tensor, preserving any trailing padding.

        Input layout:  [first(first_rows) ; second(second_rows) ; padding(...)]
        Output layout: [second(second_rows) ; first(first_rows) ; padding(...)]
        """
        first = tensor[:first_rows]
        second = tensor[first_rows:first_rows + second_rows]
        rest = tensor[first_rows + second_rows:]  # trailing padding (may be empty)
        swapped = torch.cat([second, first, rest], dim=0).contiguous()
        _reordered_refs.append(swapped)
        return swapped

    def extract_weight_ipc_handles(self) -> list[dict[str, Any]]:
        """Extract CUDA IPC handles for all quantized weights in the model.

        Called via collective_rpc from the parent process. Returns a list of
        dicts, each describing one quantized weight with IPC handles for the
        data tensor and any associated scale/meta tensors.

        The IPC handles (bytes) can be opened in another process on the same
        machine to map the same GPU memory — achieving zero-copy sharing.
        """
        model = self.model_runner.model
        if hasattr(model, "runnable"):
            model = model.runnable

        device = next(model.parameters()).device
        device_idx = device.index if device.index is not None else 0

        results: list[dict[str, Any]] = []

        for param_name, param in model.named_parameters():
            if not param_name.endswith(".weight"):
                continue

            # Get the parent module for this parameter (use original vLLM name)
            parts = param_name.rsplit(".", 1)
            module_name = parts[0] if len(parts) > 1 else ""
            module = model
            if module_name:
                for attr in module_name.split("."):
                    module = getattr(module, attr, module)

            entry = self._extract_quant_ipc(param_name, param, module, device_idx)
            if entry is not None:
                # Normalize to HF-compatible name for C++ lookup
                entry["name"] = self._normalize_vllm_name(entry["name"])
                results.append(entry)

        logger.info("Extracted %d quantized weight IPC handles on device %d",
                     len(results), device_idx)
        return results

    def _get_fuse_swap_sizes(self, name: str, module: Any) -> tuple[int, int] | None:
        """Return (first_rows, second_rows) for fused projections that need swapping.

        vLLM's gate_up_proj stores [gate; up] but surogate expects [up; gate].
        Returns (gate_size, up_size) so the caller can swap them.
        """
        for fused_name in self._FUSE_SWAP_NAMES:
            if f".{fused_name}." in name:
                sizes = getattr(module, "output_partition_sizes", None)
                if sizes and len(sizes) == 2:
                    return (sizes[0], sizes[1])
        return None

    def _extract_quant_ipc(
        self, name: str, param: torch.Tensor, module: Any, device: int
    ) -> dict[str, Any] | None:
        """Try to extract IPC info for a quantized weight. Returns None if not quantized."""
        swap_sizes = self._get_fuse_swap_sizes(name, module)

        # Try BnB NF4
        # TODO: apply gate/up swap for BnB NF4 (flat 1D data needs byte-level split)
        entry = self._extract_bnb_nf4_ipc(name, param, module, device)
        if entry is not None:
            return entry

        # Try FP8
        entry = self._extract_fp8_ipc(name, param, module, device, swap_sizes)
        if entry is not None:
            return entry

        # Try NVFP4 (ModelOpt)
        entry = self._extract_nvfp4_ipc(name, param, module, device, swap_sizes)
        if entry is not None:
            return entry

        return None

    def _extract_bnb_nf4_ipc(
        self, name: str, param: torch.Tensor, module: Any, device: int
    ) -> dict[str, Any] | None:
        quant_states = getattr(param, "bnb_quant_state", None)
        if quant_states is None:
            return None

        qs = quant_states[0] if isinstance(quant_states, (list, tuple)) else quant_states
        absmax = getattr(qs, "absmax", None)
        if absmax is None:
            return None

        block_size = getattr(qs, "blocksize", 64)
        M = qs.shape[0] if hasattr(qs, "shape") else 0
        K = qs.shape[1] if hasattr(qs, "shape") and len(qs.shape) > 1 else 0

        result: dict[str, Any] = {
            "name": name,
            "format": QUANT_FORMAT_BNB_NF4,
            "M": M, "K": K,
            "block_size": block_size,
            "double_quant": hasattr(qs, "nested") and qs.nested,
            "double_quant_group_size": getattr(qs, "nested_blocksize", 256),
            "global_scale": 1.0,
            "device": device,
            "data": _get_ipc_info(param),
            "scales": _get_ipc_info(absmax),
            "meta": None,
            "meta2": None,
        }

        if result["double_quant"] and hasattr(qs, "state2"):
            state2 = qs.state2
            if hasattr(state2, "absmax") and state2.absmax is not None:
                result["meta"] = _get_ipc_info(state2.absmax)
            if hasattr(qs, "offset") and qs.offset is not None:
                result["meta2"] = _get_ipc_info(qs.offset)

        return result

    def _extract_fp8_ipc(
        self, name: str, param: torch.Tensor, module: Any, device: int,
        swap_sizes: tuple[int, int] | None = None,
    ) -> dict[str, Any] | None:
        if param.dtype != torch.float8_e4m3fn:
            return None

        weight_scale = getattr(module, "weight_scale", None)
        if weight_scale is None:
            weight_scale = getattr(module, "weight_scale_inv", None)
        if weight_scale is None:
            return None

        M, K = param.shape[0], param.shape[1] if param.ndim > 1 else 1
        block_size = 128
        if hasattr(module, "weight_block_size"):
            block_size = module.weight_block_size[0]

        # Swap fused gate/up halves if needed
        if swap_sizes is not None:
            gate_rows, up_rows = swap_sizes
            # FP8 scales have ceil(M/bs) rows, not M rows
            s_gate = (gate_rows + block_size - 1) // block_size
            s_up = (up_rows + block_size - 1) // block_size
            param = self._swap_rows(param, gate_rows, up_rows)
            weight_scale = self._swap_rows(weight_scale, s_gate, s_up)
            logger.info("Swapped gate/up halves for %s (gate=%d, up=%d)",
                        name, gate_rows, up_rows)

        return {
            "name": name,
            "format": QUANT_FORMAT_FP8_PER_BLOCK,
            "M": M, "K": K,
            "block_size": block_size,
            "double_quant": False,
            "double_quant_group_size": 256,
            "global_scale": 1.0,
            "device": device,
            "data": _get_ipc_info(param),
            "scales": _get_ipc_info(weight_scale),
            "meta": None,
            "meta2": None,
        }

    def _extract_nvfp4_ipc(
        self, name: str, param: torch.Tensor, module: Any, device: int,
        swap_sizes: tuple[int, int] | None = None,
    ) -> dict[str, Any] | None:
        """Try to extract IPC info for NVFP4 (ModelOpt) weight.

        After vLLM's process_weights_after_loading, the layer has:
          - weight: uint8 packed FP4 [M, K//2] (possibly padded for CUTLASS)
          - weight_scale: fp8_e4m3fn, already F8_128x4 swizzled (matching surogate's dequant layout)
          - weight_global_scale: fp32 scalar (the global decode scale)
          - output_size_per_partition / input_size_per_partition: original M, K

        The F8_128x4 scale swizzle applied by vLLM (swizzle_blockscale) is identical
        to surogate's swizzle_fp8_scales_rowmajor_to_f8_128x4, so the scales can be
        shared zero-copy without re-swizzling.
        """
        if param.dtype != torch.uint8:
            return None

        # NVFP4 weights have weight_scale (fp8 block scales) and weight_global_scale (fp32)
        weight_scale = getattr(module, "weight_scale", None)
        if weight_scale is None or weight_scale.dtype != torch.float8_e4m3fn:
            return None

        weight_global_scale = getattr(module, "weight_global_scale", None)
        if weight_global_scale is None:
            return None

        # Swap fused gate/up halves if needed.
        # NVFP4 scales are F8_128x4 swizzled but each 128-row block is self-contained,
        # so swapping at a 128-aligned boundary works correctly.
        if swap_sizes is not None:
            gate_rows, up_rows = swap_sizes
            assert gate_rows % 128 == 0 and up_rows % 128 == 0, (
                f"gate/up swap requires 128-row alignment, got gate={gate_rows}, up={up_rows}"
            )
            param = self._swap_rows(param, gate_rows, up_rows)
            weight_scale = self._swap_rows(weight_scale, gate_rows, up_rows)
            logger.info("Swapped gate/up halves for %s (gate=%d, up=%d)",
                        name, gate_rows, up_rows)

        # Use original dimensions (pre-padding) from the module
        M = getattr(module, "output_size_per_partition", param.shape[0])
        K = getattr(module, "input_size_per_partition", None)
        if K is None:
            # Fallback: K = packed_cols * 2 (2 FP4 values per byte)
            K = param.shape[1] * 2

        block_size = 16  # NVFP4 group size

        global_scale_val = float(weight_global_scale.item())

        return {
            "name": name,
            "format": QUANT_FORMAT_FP4_BLOCK_2D,
            "M": M, "K": K,
            "block_size": block_size,
            "double_quant": False,
            "double_quant_group_size": 256,
            "global_scale": global_scale_val,
            "device": device,
            "data": _get_ipc_info(param),
            "scales": _get_ipc_info(weight_scale),
            "meta": None,
            "meta2": None,
        }

    def update_weights_from_tensors(self, weight_dict: dict[str, torch.Tensor]) -> None:
        """Update vLLM model weights from surogate's GPU tensors.

        Args:
            weight_dict: Dict mapping PEFT parameter names to GPU tensors.
                         Tensors are live views into surogate's GPU memory.
                         vLLM's load_weights() will copy them into its own buffers.
        """
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        # vLLM's load_weights accepts an iterator of (name, tensor) pairs
        weights_iter = iter(weight_dict.items())
        model.load_weights(weights_iter)  # type: ignore

        # Process weights after loading (important for some models)
        device = next(model.parameters()).device
        process_weights_after_loading(model, self.model_runner.model_config, device)

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Colocate mode weight update via shared state.

        In colocate mode, the trainer calls update_weights_from_tensors() directly.
        This method is called via the /update_weights endpoint which passes a string.
        We use it as a signal to pull from the shared state.
        """
        from surogate.grpo.inference.vllm.colocate_state import get_pending_weights

        weight_dict = get_pending_weights()
        if weight_dict is None:
            logger.warning("update_weights_from_path called but no pending weights found")
            return

        self.update_weights_from_tensors(weight_dict)
