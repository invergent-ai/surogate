from typing import Any

from transformers.utils.quantization_config import QuantizationConfigMixin


class FalqonQuantizationConfig(QuantizationConfigMixin):
    def __init__(self):
        self.quant_method = "falqon"

    def to_dict(self) -> dict[str, Any]:
        return {"quant_method": self.quant_method}

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        return FalqonQuantizationConfig()
