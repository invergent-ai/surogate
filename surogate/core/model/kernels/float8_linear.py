from typing import Optional, Callable

import torch
from torch import nn
from torchao.float8 import Float8LinearConfig, ScalingType
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor

from surogate.core.model.kernels.fp8 import fp8_torch_block_quant_forward
from surogate.utils.logger import get_logger

logger = get_logger()

torch.autograd.set_detect_anomaly(True)

class CustomFloat8Linear(Float8Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_method = "fp8"

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8LinearConfig] = None,
    ):
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = CustomFloat8Linear(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )

        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        if hasattr(mod, "weight_scale_inv"):
            new_mod.weight_scale_inv = getattr(mod, "weight_scale_inv", None)

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        if config.enable_fsdp_float8_all_gather:
            assert config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            new_mod.weight = torch.nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    new_mod.weight,
                    new_mod.linear_mm_config,
                    new_mod.config.cast_config_weight.target_dtype,
                ),
                requires_grad=new_mod.weight.requires_grad,
            )

        return new_mod

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return fp8_torch_block_quant_forward(X, self.weight, self.weight_scale_inv)


def custom_convert_to_float8_training(
        model: nn.Module,
        *,
        module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
        config: Optional['Float8LinearConfig'] = None,
) -> nn.Module:
    from torchao.float8.float8_linear_utils import swap_linear_layers
    torch._C._log_api_usage_once("torchao.float8.convert_to_float8_training")
    from_float = lambda m: CustomFloat8Linear.from_float(
        m,
        config=config,
    )

    def filter_lora_layers(module, fqn: str) -> bool:
        layers_to_filter = ["lora_A", "lora_B"]
        for layer in layers_to_filter:
            if layer in fqn:
                return False
        return module_filter_fn(module, fqn) if module_filter_fn is not None else True

    return swap_linear_layers(
        model,
        from_float,
        module_filter_fn=filter_lora_layers,
    )
