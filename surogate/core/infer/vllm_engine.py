import os

import torch

from surogate.core.infer.infer_engine import InferEngine
from surogate.utils.logger import get_logger

logger = get_logger()
try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, EngineArgs, LLMEngine
    from vllm.pooling_params import PoolingParams
except Exception:
    raise

try:
    from vllm.reasoning import ReasoningParserManager
except ImportError:
    ReasoningParserManager = None

dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}

class VllmEngine(InferEngine):
    pass
