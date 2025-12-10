from typing import Any, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._prims_common import suggest_memory_format
from torchao.float8 import LinearMMConfig, GemmInputRole, Float8TrainingTensor
from torchao.float8.float8_training_tensor import hp_tensor_and_scale_to_float8

from surogate.core.model.quant.float8_scaling_utils import _maybe_initialize_amaxes_scales_for_float8_cast, \
    hp_tensor_to_float8_delayed

# FSDP pads its local tensor on dim-0. The subclass should be preserved such
# that the padded local tensor (and any transformations like copying to GPU)
# is of the subclass as well.
_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}

class WeightWithDelayedFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
            cls,
            tensor: torch.Tensor,
            amax_buffer: torch.Tensor,
            amax_history_buffer: torch.Tensor,
            scale_buffer: torch.Tensor,
            linear_mm_config: LinearMMConfig,
            dtype: torch.dtype,
            is_amax_initialized: bool,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
            self,
            tensor: torch.Tensor,
            amax_buffer: torch.Tensor,
            amax_history_buffer: torch.Tensor,
            scale_buffer: torch.Tensor,
            linear_mm_config: LinearMMConfig,
            dtype: torch.dtype,
            is_amax_initialized: bool,
    ):
        self._tensor = tensor
        self._amax_buffer = amax_buffer
        self._amax_history_buffer = amax_history_buffer
        self._scale_buffer = scale_buffer
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = is_amax_initialized

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithDelayedFloat8CastTensor(
                args[0]._tensor,
                args[0]._amax_buffer,
                args[0]._amax_history_buffer,
                args[0]._scale_buffer,
                args[0]._linear_mm_config,
                args[0]._dtype,
                args[0].is_amax_initialized,
            )
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None
        amax_buffer: Optional[torch.Tensor] = None
        amax_history_buffer: Optional[torch.Tensor] = None
        scale_buffer: Optional[torch.Tensor] = None
        is_amax_initialized: Optional[bool] = None

        def unwrap(t):
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            nonlocal amax_buffer
            if amax_buffer is None:
                amax_buffer = t._amax_buffer
            nonlocal amax_history_buffer
            if amax_history_buffer is None:
                amax_history_buffer = t._amax_history_buffer
            nonlocal scale_buffer
            if scale_buffer is None:
                scale_buffer = t._scale_buffer
            nonlocal is_amax_initialized
            if is_amax_initialized is None:
                is_amax_initialized = t.is_amax_initialized
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithDelayedFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithDelayedFloat8CastTensor(
                x,
                amax_buffer,
                amax_history_buffer,
                scale_buffer,
                mm_config,
                dtype,
                is_amax_initialized,
            ),
            out,
        )

    def __tensor_flatten__(self):
        return (
            [
                "_tensor",
                "_amax_buffer",
                "_amax_history_buffer",
                "_scale_buffer",
            ],
            {
                "mm_config": self._linear_mm_config,
                "dtype": self._dtype,
                "is_amax_initialized": self.is_amax_initialized,
            },
        )

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return WeightWithDelayedFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_amax_buffer"],
            inner_tensors["_amax_history_buffer"],
            inner_tensors["_scale_buffer"],
            metadata["mm_config"],
            metadata["dtype"],
            metadata["is_amax_initialized"],
        )

    def __repr__(self):
        return f"WeightWithDelayedFloat8CastTensor(tensor={self._tensor}, amax_buffer={self._amax_buffer}, scale_buffer={self._scale_buffer}, mm_config={self._linear_mm_config}, dtype={self._dtype})"

    def fsdp_pre_all_gather(self, mesh):
        # initialize if needed
        # TODO(before land): ensure settings are consistent between Float8Linear and here
        if not self.is_amax_initialized:
            _maybe_initialize_amaxes_scales_for_float8_cast(
                self._tensor,
                self._amax_buffer,
                self._amax_history_buffer,
                self._scale_buffer,
                "max",  # TODO(before land): read this from parent
                self._dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.is_amax_initialized = True

        float8_tensor = hp_tensor_to_float8_delayed(
            self._tensor,
            self._scale_buffer,
            self._dtype,
            self._amax_buffer,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
            self,
            all_gather_outputs: Tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            assert isinstance(out, Float8TrainingTensor), f"{type(out)}"
            out._scale = scale
            return
        return Float8TrainingTensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)


class WeightWithStaticFloat8CastTensor(torch.Tensor):
    @staticmethod
    def __new__(
            cls,
            tensor: torch.Tensor,
            static_scale: torch.Tensor,
            linear_mm_config: LinearMMConfig,
            dtype: torch.dtype,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(
            self,
            tensor: torch.Tensor,
            static_scale: torch.Tensor,
            linear_mm_config: LinearMMConfig,
            dtype: torch.dtype,
    ):
        self._tensor = tensor
        self._static_scale = static_scale
        self._linear_mm_config = linear_mm_config
        self._dtype = dtype

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return WeightWithStaticFloat8CastTensor(
                args[0]._tensor,
                args[0]._static_scale,
                args[0]._linear_mm_config,
                args[0]._dtype,
            )
        static_scale: Optional[torch.Tensor] = None
        mm_config: Optional[LinearMMConfig] = None
        dtype: Optional[torch.dtype] = None

        def unwrap(t):
            nonlocal static_scale
            if static_scale is None:
                static_scale = t._static_scale
            nonlocal mm_config
            if mm_config is None:
                mm_config = t._linear_mm_config
            else:
                assert t._linear_mm_config == mm_config
            nonlocal dtype
            if dtype is None:
                dtype = t._dtype
            else:
                assert t._dtype == dtype
            return t._tensor

        args, kwargs = pytree.tree_map_only(
            WeightWithStaticFloat8CastTensor, unwrap, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _ops_to_preserve_subclass:
            return out
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: WeightWithStaticFloat8CastTensor(
                x, static_scale, mm_config, dtype
            ),
            out,
        )

    def __tensor_flatten__(self):
        return ["_tensor", "_static_scale"], {
            "mm_config": self._linear_mm_config,
            "dtype": self._dtype,
        }

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        return WeightWithStaticFloat8CastTensor(
            inner_tensors["_tensor"],
            inner_tensors["_static_scale"],
            flatten_spec["mm_config"],
            flatten_spec["dtype"],
        )

    def __repr__(self):
        return f"WeightWithStaticFloat8CastTensor(tensor={self._tensor}, static_scale={self._static_scale}, linear_mm_config={self._linear_mm_config}, dtype={self.dtype})"

    def fsdp_pre_all_gather(self, mesh):
        float8_tensor = hp_tensor_and_scale_to_float8(
            self._tensor,
            self._static_scale,
            self._dtype,
            self._linear_mm_config,
            GemmInputRole.WEIGHT,
        )
        return (float8_tensor._data,), (float8_tensor._scale,)

    def fsdp_post_all_gather(
            self,
            all_gather_outputs: Tuple[torch.Tensor, ...],
            metadata: Any,
            param_dtype: torch.dtype,
            *,
            out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs
        (scale,) = metadata
        if out is not None:
            from torch.distributed._tensor import DTensor

            if isinstance(out, Float8TrainingTensor):
                out._scale = scale
            elif isinstance(out, DTensor) and isinstance(
                    out._local_tensor, Float8TrainingTensor
            ):
                out._local_tensor._scale = scale
            else:
                raise RuntimeError(
                    f"out must be a Float8TrainingTensor or DTensor(_local_tensor=Float8TrainingTensor), but got {out}"
                )
            return
        return Float8TrainingTensor(
            data,
            scale,
            param_dtype,
            self._linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        ), (data,)

