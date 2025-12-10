import transformers.utils.quantization_config

from surogate.core.model.quant.falqon_quantizer import FalqonQuantizer

transformers.utils.quantization_config.QuantizationMethod.FALQON = "falqon"
transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING['falqon'] = FalqonQuantizer