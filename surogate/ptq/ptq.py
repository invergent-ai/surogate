import random

# import modelopt.torch.opt as mto
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from surogate.datasets.datasets import load_datasets
from surogate.datasets.tokenization import tokenize_dataset, PromptTokenizingStrategy
from surogate.loaders.loader import load_model_and_tokenizer
from surogate.utils.command import SurogateCommand
from surogate.utils.logger import get_logger

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

logger = get_logger()

# mto.enable_huggingface_checkpointing()

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = self.config.get('save_path', "./output")
        self.additional_ignore_layers = self.config.get('ignore_layers', [])
        self.model_id = self.config['model']

    def run(self):
        if not torch.cuda.is_available():
            raise OSError("GPU is required for PTQ.")

        random.seed(RAND_SEED)
        np.random.seed(RAND_SEED)

        if self.config['scheme'] not in SUPPORTED_SCHEMES.keys():
            raise ValueError(f"Unsupported quantization scheme: {self.config['scheme']}. "
                             f"Supported schemes are: {list(SUPPORTED_SCHEMES.keys())}")

        if self.config['scheme'] == 'fp8':
            self.do_fp8()
        elif self.config['scheme'] == 'gptq_int4':
            self.do_gptq('int4')
        elif self.config['scheme'] == 'gptq_int8':
            pass
        elif self.config['scheme'] == 'awq':
            pass
        elif self.config['scheme'] == 'nvfp4':
            pass

    def do_fp8(self):
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor import oneshot
        from llmcompressor.metrics import PythonLogger
        PythonLogger._global_file_sink_id = "null"

        ignore_layers = ["lm_head", "re:.*lm_head"]
        ignore_layers.extend(self.additional_ignore_layers)

        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8_DYNAMIC",
            ignore=ignore_layers,
        )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model = oneshot(model=model, recipe=recipe, log_dir=None)
        model.save_pretrained(self.save_dir, save_compressed=True)
        tokenizer.save_pretrained(self.save_dir)

    def do_gptq(self, precision: str):
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor import oneshot
        from llmcompressor.metrics import PythonLogger
        PythonLogger._global_file_sink_id = "null"

        dataset_config = self.config.get('datasets', [])
        if len(dataset_config) == 0:
            raise ValueError("At least one calibration dataset must be provided for GPTQ quantization.")

        ignore_layers = ["lm_head", "re:.*lm_head"]
        ignore_layers.extend(self.additional_ignore_layers)

        model, tokenizer = load_model_and_tokenizer(self.config)
        calibration_dataset = load_datasets(self.config)
        calibration_dataset = tokenize_dataset(
            PromptTokenizingStrategy(
                tokenizer=tokenizer,
                mode='pt',
            ),
            calibration_dataset,
        )

        if precision == "int8":
            recipe = [
                GPTQModifier(
                    targets="Linear",
                    scheme="W8A8",
                    ignore=ignore_layers
                ),
            ]
        elif precision == "int4":
            recipe = GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=ignore_layers
            )

        oneshot(
            model=model,
            dataset=calibration_dataset,
            recipe=recipe,
            max_seq_length=self.config.get('sequence_len', DEFAULT_MAX_SEQ_LEN),
        )
