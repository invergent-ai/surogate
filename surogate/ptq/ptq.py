import sys

import torch
from llmcompressor import oneshot
from llmcompressor.metrics import PythonLogger
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from loguru import logger as loguru_logger
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from surogate.config.ptq_config import PTQConfig
from surogate.datasets.datasets import load_datasets
from surogate.datasets.tokenization import tokenize_dataset, PromptTokenizingStrategy
from surogate.loaders.loader import load_model_and_tokenizer
from surogate.utils.command import SurogateCommand
from surogate.utils.logger import get_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="ERROR")

PythonLogger._global_file_sink_id = "null"

logger = get_logger()


SUPPORTED_SCHEMES = {
    'fp8': 'Quantizes the weights and activations to FP8',
    'awq': 'Quantize weights to INT4 (INT4 W4A16) with AWQ',
    'gptq_int4': 'Quantize weights to INT4 (INT4 W4A16) with GPTQ',
    'gptq_int8': 'Quantize weights and activations to INT8 (INT8 W8A8) with GPTQ',
    'nvfp4': 'Quantizes the weights and activations to FP4',
}

class SurogatePTQ(SurogateCommand):
    config: PTQConfig
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, **kwargs):
        super().__init__(PTQConfig, **kwargs)

        self.ignore_layers = ["lm_head", "re:.*lm_head"]
        self.ignore_layers.extend(self.config.ignore_layers)

    def run(self):
        if not torch.cuda.is_available():
            raise OSError("GPU is required for PTQ.")

        self.model, self.tokenizer = load_model_and_tokenizer(self.config.model, self.config.model_type, self.args, True)

        calibration_dataset = None
        if self.config.scheme in ['gptq_int4', 'gptq_int8', 'awq', 'nvfp4']:
            calibration_dataset = load_datasets(self.config.datasets, self.args, self.config.save_path, self.config.seed)
            calibration_dataset = tokenize_dataset(
                PromptTokenizingStrategy(
                    tokenizer=self.tokenizer,
                    mode='pt',
                ),
                calibration_dataset,
            )

        recipe = None
        if self.config.scheme == 'fp8':
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=self.ignore_layers,
            )
        elif self.config.scheme == 'gptq_int4' or self.config.scheme == 'gptq_int8':
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16" if self.config.scheme == 'gptq_int4' else "W8A8",
                ignore=self.ignore_layers
            ),
        elif self.config.scheme == 'awq':
            recipe = AWQModifier(
                targets="Linear",
                scheme="W4A16_ASYM",
                ignore=self.ignore_layers
            )
        elif self.config.scheme == 'nvfp4':
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="NVFP4",
                ignore=self.ignore_layers,
            )

        quantized_model = oneshot(
            model=self.model,
            recipe=recipe,
            dataset=calibration_dataset,
            log_dir=None,
            max_seq_length=self.config.sequence_len
        )

        self.save_model(quantized_model)

    def save_model(self, quantized_model):
        quantized_model.save_pretrained(self.config.save_path, save_compressed=True)
        self.tokenizer.save_pretrained(self.config.save_path)
