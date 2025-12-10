from typing import Optional

import torch
from torchao.float8 import (
    Float8LinearConfig,
    ScalingGranularity,
    ScalingType,
    Float8TrainingTensor, LinearMMConfig, GemmInputRole, ScaledMMConfig,
)
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic, hp_tensor_and_scale_to_float8
from torchao.float8.float8_utils import (
    tensor_to_scale,
)


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    CPU_FLAG = False
    if weight.device == torch.device('cpu'):
        weight = weight.to("cuda")
        CPU_FLAG = True

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    if CPU_FLAG:
        L = L.to("cpu")
        R = R.to("cpu")
        U = U.to("cpu")
        S = S.to("cpu")
        Vh = Vh.to("cpu")
        weight.to("cpu")
        # Empty CUDA cache to free up memory
        torch.cuda.empty_cache()

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


def get_maybe_axiswise_dim(logical_dim: int, scaling_granularity: ScalingGranularity) -> Optional[int]:
    """Get the axiswise dimension if using axiswise scaling, else None."""
    if scaling_granularity is ScalingGranularity.AXISWISE:
        return logical_dim
    return None


@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_float8(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in float8.
    Supports tensorwise scaling granularity only.
    """

    @staticmethod
    def forward(
            ctx,
            input_fp8,
            weight_fp8_t,
            B,
            linear_mm_config: LinearMMConfig,
            config: Float8LinearConfig,
    ):
        orig_shape = input_fp8.shape
        input_fp8_reshaped = input_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_fp8_reshaped, weight_fp8_t)
        out_backbone, out_A = torch.split(res_bits, [res_bits.shape[-1] - B.shape[0], B.shape[0]], dim=-1)
        out_backbone = out_backbone.reshape(*orig_shape[:-1], out_backbone.shape[-1])
        ctx.save_for_backward(weight_fp8_t[:, :(res_bits.shape[-1] - B.shape[0])], out_A)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        return out_backbone

    @staticmethod
    def backward(ctx, grad_output_fp8):
        (weight_fp8_t, out_A) = ctx.saved_tensors

        grad_output_fp8_orig_shape = grad_output_fp8.shape
        grad_output_fp8_reshaped = grad_output_fp8.reshape(-1, grad_output_fp8_orig_shape[-1])

        grad_input = torch.mm(grad_output_fp8_reshaped, weight_fp8_t.t())
        grad_input = grad_input.reshape(*grad_output_fp8_orig_shape[:-1], grad_input.shape[-1])

        out_A_scale = tensor_to_scale(out_A, torch.float8_e4m3fn)
        out_A_fp8 = hp_tensor_and_scale_to_float8(
            out_A, out_A_scale, torch.float8_e4m3fn,
            linear_mm_config=ctx.linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        )

        grad_B_t = grad_output_fp8_reshaped.t() @ out_A_fp8

        return grad_input, None, grad_B_t.T, None, None


@torch._dynamo.allow_in_graph
class manual_float8_matmul_with_args_in_hp(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in high precision and cast to float8 inside.
    Supports dynamic scaling type and axiswise granularity.
    """

    @staticmethod
    def forward(
            ctx,
            input_hp: torch.Tensor,
            weight_hp_t: torch.Tensor,
            linear_mm_config: LinearMMConfig,
            config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input_hp
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_hp,
                c.cast_config_input.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_input.scaling_granularity),
            )

        if c.cast_config_weight.scaling_type is ScalingType.DISABLED:
            weight_maybe_fp8_t = weight_hp_t
        else:
            weight_maybe_fp8_t = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_weight.scaling_granularity),
            )

        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors
        c = ctx.config

        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(-1, grad_output_orig_shape[-1])

        # grad_input
        if c.cast_config_grad_output.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim0 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_grad_output.scaling_granularity),
            )

        if c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8_dim0 = weight_hp_t
        else:
            weight_t_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight_for_grad_input.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight_for_grad_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(-1, c.cast_config_weight_for_grad_input.scaling_granularity),
            )

        grad_input = torch.mm(grad_output_reshaped_maybe_fp8_dim0, weight_t_maybe_fp8_dim0.t())
        grad_input = grad_input.reshape(*grad_output_orig_shape[:-1], grad_input.shape[-1])

        input_hp_orig_shape = input_hp.shape
        input_hp_reshaped = input_hp.reshape(-1, input_hp_orig_shape[-1])

        # grad_weight
        if c.cast_config_grad_output_for_grad_weight.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_grad_output_for_grad_weight.scaling_granularity),
            )

        if c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_reshaped_maybe_fp8_dim1 = input_hp_reshaped
        else:
            input_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                input_hp_reshaped,
                c.cast_config_input_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(0, c.cast_config_input_for_grad_weight.scaling_granularity),
            )

        grad_weight = torch.mm(grad_output_reshaped_maybe_fp8_dim1.t(), input_reshaped_maybe_fp8_dim1)

        return grad_input, grad_weight.t(), None, None


class Float8Linear_falqon(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        config = kwargs.pop("config")
        emulate = config.emulate
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type

        self.config = config
        self.always_float32_buffers = set()

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

        self.create_buffers(device=self.weight.device)
        self.warmup = True

    def create_buffers(self, device):
        if self.config.cast_config_input.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_input",
                self.config.cast_config_input.static_scale.to(device),
            )
        if self.config.cast_config_weight.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_weight",
                self.config.cast_config_weight.static_scale.to(device),
            )
        if self.config.cast_config_grad_output.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_grad_output",
                self.config.cast_config_grad_output.static_scale.to(device),
            )

    def register_always_float32_buffer(
            self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True
    ) -> None:
        self.register_buffer(name=name, tensor=tensor, persistent=persistent)
        self.always_float32_buffers.add(name)

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        self.convert_amax_buffer_to_float32()
        return ret

    def convert_amax_buffer_to_float32(self):
        for key in self.always_float32_buffers:
            if self._buffers[key] is not None:
                self._buffers[key] = self._buffers[key].to(torch.float32)

    def cast_input_to_float8(
            self, input: torch.Tensor, is_amax_initialized: bool
    ) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        if self.scaling_type_input is ScalingType.DYNAMIC:
            return hp_tensor_to_float8_dynamic(
                input,
                self.config.cast_config_input.target_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )
        elif self.scaling_type_input is ScalingType.STATIC:
            scale = self.fp8_static_scale_input
            return hp_tensor_and_scale_to_float8(
                input, scale, self.config.cast_config_input.target_dtype,
                self.linear_mm_config, gemm_input_role=GemmInputRole.INPUT,
            )
        else:
            return input

    def get_weight_scale(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if self.scaling_type_weight is ScalingType.DYNAMIC:
            return tensor_to_scale(weight, self.config.cast_config_weight.target_dtype)
        elif self.scaling_type_weight is ScalingType.STATIC:
            return self.fp8_static_scale_weight
        return None

    def cast_weight_to_float8_t(
            self,
            weight: torch.Tensor,
            weight_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(weight, Float8TrainingTensor):
            return weight.t()
        weight_fp8 = hp_tensor_and_scale_to_float8(
            weight,
            weight_scale,
            self.config.cast_config_weight.target_dtype,
            self.linear_mm_config,
            gemm_input_role=GemmInputRole.WEIGHT,
        )
        return weight_fp8.t()

    def cast_weight_to_original_t(self, weight: torch.Tensor):
        if isinstance(weight, Float8TrainingTensor):
            return weight.to_original_precision().t()
        else:
            return weight.t()

    def cast_output_to_float8_in_bw(self, output: torch.Tensor) -> torch.Tensor:
        # With dynamic/static scaling, the backward cast is handled inside autograd functions
        return output

    def forward_fp8_matmul(self, input: torch.Tensor) -> torch.Tensor:
        has_any_axiswise_scaling = any(
            cc.scaling_granularity is ScalingGranularity.AXISWISE
            for cc in [
                self.config.cast_config_input,
                self.config.cast_config_weight,
                self.config.cast_config_grad_output,
                self.config.cast_config_input_for_grad_weight,
                self.config.cast_config_weight_for_grad_input,
                self.config.cast_config_grad_output_for_grad_weight,
            ]
        )

        if not has_any_axiswise_scaling:
            input_fp8 = self.cast_input_to_float8(input)
            output = manual_float8_matmul_with_args_in_float8.apply(
                input_fp8, self.weight, self.B.T, self.linear_mm_config, self.config
            )
        else:
            # for now, axiswise path is separate
            # TODO(future PR): unify to support mix and match
            output = manual_float8_matmul_with_args_in_hp.apply(
                input,
                self.weight,
                self.linear_mm_config,
                self.config,
            )
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.forward_fp8_matmul(input)

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(
                f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}"
            )
        cast_config_str = ",".join(parts)
        s = f'{super().extra_repr()}, cast_configs={cast_config_str}"'
        return s

    def get_A_and_B(self):
        A = self.weight[:, self.out_features:].to_original_precision()
        B = self.B
        return A, B

    def set_warmup(self, warmup: bool):
        self.warmup = warmup

    @torch.no_grad()
    def apply_delta_weight(self):
        orig_datatype = self.weight._data.dtype
        B_dtype = self.B.dtype
        fp8_W, fp8_A = torch.split(self.weight._data, [self.out_features, self.rank], dim=1)

        # Get top-k rows with highest magnitude
        B_magnitudes = torch.abs(self.B).sum(dim=-1)
        if self.warmup:
            topk_indices = torch.topk(B_magnitudes, k=self.rank).indices
        else:
            topk_indices = torch.topk(B_magnitudes, k=self.num_topk).indices

        fp8_W_orig_t = fp8_W.to(B_dtype).T
        delta_W_t = self.B[topk_indices] @ fp8_A.to(B_dtype).T
        fp8_W_orig_t[topk_indices] = fp8_W_orig_t[topk_indices] + delta_W_t

        merged_weight = torch.cat([fp8_W_orig_t, fp8_A.to(B_dtype).t()], dim=0).clamp(
            min=torch.finfo(orig_datatype).min, max=torch.finfo(orig_datatype).max).to(orig_datatype).contiguous().t()

        self.B.zero_()
        self.weight._data = merged_weight

    @classmethod
    def from_float(
            cls,
            mod,
            config: Optional[Float8LinearConfig] = None,
            rank: Optional[int] = 32,
            lora_alpha: Optional[float] = 16,
            lora_init: Optional[str] = "svd",
            num_topk: Optional[int] = 10,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False, config=config)

        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.rank = rank
        new_mod.lora_alpha = lora_alpha
        new_mod.scaling = lora_alpha / rank
        new_mod.lora_init = lora_init
        new_mod.num_topk = num_topk

        with torch.no_grad():
            weight = mod.weight

            if lora_init == "svd":
                fp8_weight_scale = new_mod.get_weight_scale(weight)
                fp8_weight = new_mod.cast_weight_to_float8_t(weight, new_mod.is_amax_initialized, fp8_weight_scale)
                fp8_weight_dequant = new_mod.cast_weight_to_original_t(fp8_weight)
                fp8_error = mod.weight - fp8_weight_dequant
                svd_output = _low_rank_decomposition(fp8_error.float(), reduced_rank=rank)
                A = svd_output["R"]
                B = torch.zeros_like(svd_output["L"])
            elif lora_init == "random":
                A_random_dist = torch.normal(
                    mean=weight.mean(),
                    std=weight.std(),
                    size=(rank, mod.weight.shape[1]),
                    device=mod.weight.device,
                    dtype=mod.weight.dtype
                )
                weight_min = weight.min()
                weight_max = weight.max()
                A = torch.clip(A_random_dist, weight_min, weight_max)
                B = torch.zeros(mod.weight.shape[0], rank, device=mod.weight.device, dtype=mod.weight.dtype)
            else:
                raise ValueError(f"Invalid lora_init: {lora_init}")

            merged_weight = torch.cat([weight, A], dim=0)
            fp8_merged_weight_scale = new_mod.get_weight_scale(merged_weight)
            fp8_merged_weight = new_mod.cast_weight_to_float8_t(merged_weight, fp8_merged_weight_scale)

        new_mod.weight = torch.nn.Parameter(fp8_merged_weight)
        new_mod.weight.requires_grad = False
        new_mod.B = torch.nn.Parameter(B)
        new_mod.B.requires_grad = True

        # need to create buffers again when moving from meta device to
        # real device
        new_mod.create_buffers(device=mod.weight.device)

        return new_mod
