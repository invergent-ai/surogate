from typing import Optional, Union, List

import torch
from transformers import PreTrainedTokenizerBase, GenerationConfig, BaseImageProcessor, \
    FeatureExtractionMixin
from transformers import ProcessorMixin as HfProcessorMixin
from transformers.utils import is_torch_bf16_gpu_available, is_torch_cuda_available

from surogate.utils.logger import get_logger

logger = get_logger()

Processor = Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, HfProcessorMixin]
Prompt = List[Union[str, List[int], List[str]]]
Word = Union[str, List[int]]
Context = Word


class ContextType:
    RESPONSE = 'response'
    SUFFIX = 'suffix'
    OTHER = 'other'

def get_default_torch_dtype(torch_dtype: Optional[torch.dtype]):
    # torch_dtype: torch_dtype in config.json
    if torch_dtype is not None:
        return torch_dtype

    try:
        is_bf16_available = is_torch_bf16_gpu_available()
    except:  # noqa
        is_bf16_available = False

    if is_torch_cuda_available():
        if is_bf16_available:
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # cpu
        return torch.float32

def fix_do_sample_warning(generation_config: GenerationConfig) -> None:
    # Use the default values of temperature/top_p/top_k in generation_config.
    if generation_config.temperature == 0:
        generation_config.do_sample = False
    if generation_config.do_sample is False:
        generation_config.temperature = 1.
        generation_config.top_p = 1.
        generation_config.top_k = 50
