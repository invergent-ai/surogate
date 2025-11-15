import random
import sys
from typing import Literal

import numpy as np
import torch
from llmcompressor import oneshot
from llmcompressor.metrics import PythonLogger
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from loguru import logger as loguru_logger
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from surogate.datasets.datasets import load_datasets
from surogate.datasets.tokenization import tokenize_dataset, PromptTokenizingStrategy
from surogate.loaders.loader import load_model_and_tokenizer
from surogate.utils.command import SurogateCommand
from surogate.utils.logger import get_logger

loguru_logger.remove()
loguru_logger.add(sys.stderr, level="ERROR")

PythonLogger._global_file_sink_id = "null"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

logger = get_logger()

RAND_SEED = 1234
DEFAULT_MAX_SEQ_LEN = 2048

SUPPORTED_SCHEMES = {
    'fp8': 'Quantizes the weights and activations to FP8',
    'awq': 'Quantize weights to INT4 (INT4 W4A16) with AWQ',
    'gptq_int4': 'Quantize weights to INT4 (INT4 W4A16) with GPTQ',
    'gptq_int8': 'Quantize weights and activations to INT8 (INT8 W8A8) with GPTQ',
    'nvfp4': 'Quantizes the weights and activations to FP4',
}


class SurogatePtq(SurogateCommand):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scheme: Literal['fp8', 'gptq_int4', 'gptq_int8', 'awq', 'nvfp4'] = self.config['scheme']
        if self.scheme not in SUPPORTED_SCHEMES.keys():
            raise ValueError(f"Unsupported quantization scheme: {self.config['scheme']}. "
                             f"Supported schemes are: {list(SUPPORTED_SCHEMES.keys())}")

        if self.scheme in ['gptq_int4', 'gptq_int8', 'awq', 'nvfp4']:
            self.dataset_config = self.config.get('datasets', [])
            if len(self.dataset_config) == 0:
                raise ValueError(
                    f"At least one calibration dataset must be provided for {self.scheme.upper()} quantization.")

        self.additional_ignore_layers = self.config.get('ignore_layers', [])
        self.ignore_layers = ["lm_head", "re:.*lm_head"]
        self.ignore_layers.extend(self.additional_ignore_layers)

        self.seed = self.config.get('seed', RAND_SEED)
        self.save_dir = self.config.get('save_path', "./output")

    def run(self):
        if not torch.cuda.is_available():
            raise OSError("GPU is required for PTQ.")

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.model, self.tokenizer = load_model_and_tokenizer(self.config, self.args)

        calibration_dataset = None
        if self.scheme in ['gptq_int4', 'gptq_int8', 'awq', 'nvfp4']:
            calibration_dataset = load_datasets(self.config, self.args)
            calibration_dataset = tokenize_dataset(
                PromptTokenizingStrategy(
                    tokenizer=self.tokenizer,
                    mode='pt',
                ),
                calibration_dataset,
            )

        recipe = None
        if self.scheme == 'fp8':
            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=self.ignore_layers,
            )
        elif self.scheme == 'gptq_int4' or self.scheme == 'gptq_int8':
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16" if self.scheme == 'gptq_int4' else "W8A8",
                ignore=self.ignore_layers
            ),
        elif self.scheme == 'awq':
            recipe = AWQModifier(
                targets="Linear",
                scheme="W4A16_ASYM",
                ignore=self.ignore_layers
            )
        elif self.scheme == 'nvfp4':
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
            max_seq_length=self.config.get('sequence_len', DEFAULT_MAX_SEQ_LEN)
        )

        self.save_model(quantized_model)

    def save_model(self, quantized_model):
        quantized_model.save_pretrained(self.save_dir, save_compressed=True)
        self.tokenizer.save_pretrained(self.save_dir)
