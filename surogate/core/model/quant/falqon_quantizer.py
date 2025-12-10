from typing import Optional

import torch
from torchao.float8 import Float8LinearConfig
from transformers import PreTrainedModel
from transformers.modeling_utils import _load_parameter_into_model
from transformers.quantizers import HfQuantizer, get_module_from_name

from surogate.core.model.quant.float8_linear_falqon import Float8Linear_falqon
from surogate.utils.logger import get_logger

logger = get_logger()

class FalqonQuantizer(HfQuantizer):
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def _process_model_before_weight_loading(
            self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[list[str]] = None, **kwargs
    ):
        pass

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, torch.nn.Linear):
            if module.in_features % 16 != 0 or module.out_features % 16 != 0:
                return False
        return True

    def create_quantized_param(
            self,
            model: "PreTrainedModel",
            param_value: "torch.Tensor",
            param_name: str,
            target_device: "torch.device",
            **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, torch.nn.Linear):
            import pydevd_pycharm
            pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)

            quantized_param = Float8Linear_falqon.from_float(
                module,
                config=Float8LinearConfig(),
                rank=16,
                lora_alpha=32,
                lora_init="svd",
                num_topk=10,
            )
            _load_parameter_into_model(model, param_name, quantized_param)

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass

    @property
    def is_compileable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None) -> bool:
        return True
