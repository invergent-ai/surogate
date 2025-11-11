import random

import modelopt.torch.opt as mto
import numpy as np
import torch
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from swift import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from surogate.utils.config import load_config

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

logger = get_logger()

mto.enable_huggingface_checkpointing()

RAND_SEED = 1234

SUPPORTED_SCHEMES = {
    'fp8': 'Quantizes the weights and activations to FP8',
    'awq': 'Quantize weights to INT4 (INT4 W4A16) with AWQ',
    'gptq_int4': 'Quantize weights to INT4 (INT4 W4A16) with GPTQ',
    'gptq_int8': 'Quantize weights and activations to INT8 (INT8 W8A8) with GPTQ',
    'nvfp4': 'Quantizes the weights and activations to FP4',
}

class SurogatePtq:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])

    def run(self):
        if not torch.cuda.is_available():
            raise OSError("GPU is required for PTQ.")

        random.seed(RAND_SEED)
        np.random.seed(RAND_SEED)

        # launch a memory monitor to read the currently used GPU memory.
        monitor = launch_memory_monitor()

        if self.config['scheme'] not in SUPPORTED_SCHEMES.keys():
            monitor.stop()
            raise ValueError(f"Unsupported quantization scheme: {self.config['scheme']}. "
                             f"Supported schemes are: {list(SUPPORTED_SCHEMES.keys())}")

        model_id = self.config['model']
        save_dir = self.config.get('save_path', "./output")
        additional_ignore_layers = self.config.get('ignore_layers', [])

        if self.config['scheme'] == 'fp8':
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from llmcompressor import oneshot
            from llmcompressor.metrics import PythonLogger

            PythonLogger._global_file_sink_id = "null"
            ignore_layers = ["lm_head", "re:.*lm_head"]
            ignore_layers.extend(additional_ignore_layers)

            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=ignore_layers,
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            model = oneshot(model=model, recipe=recipe, log_dir=None)
            model.save_pretrained(save_dir, save_compressed=True)
            tokenizer.save_pretrained(save_dir)
