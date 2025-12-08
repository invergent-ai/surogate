from dataclasses import dataclass
from typing import Optional, Literal, List

from surogate.core.config.model_config import ModelConfig
from surogate.utils.dict import DictDefault
from surogate.utils.net import find_free_port


@dataclass
class AdapterConfig:
    """
    ServeConfig class is a dataclass that holds configuration parameters for serving a model using SurogateServe.

    Args:
        name (Optional[str]): The name of the adapter. Default is None.
        path (Optional[str]): The id/path to the adapter. Default is None.
    """
    name: Optional[str] = None
    path: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.name = cfg['name']
        self.path = cfg['path']



@dataclass
class ServeCacheConfig:
    enabled: Optional[bool] = None
    chunk_size: Optional[int] = None
    max_memory_cache_gb: Optional[float] = None
    max_disk_cache_gb: Optional[float] = None
    disk_cache_path: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.enabled = cfg['enabled'] or False
        self.chunk_size = cfg['chunk_size'] or 256
        self.max_memory_cache_gb = cfg['max_memory_cache_gb'] or 0
        self.max_disk_cache_gb = cfg['max_disk_cache_gb'] or 0
        self.disk_cache_path = cfg['disk_cache_path'] or '/tmp'


@dataclass
class ServeConfig(ModelConfig, AdapterConfig):
    """
    ServeConfig class is a dataclass that holds configuration parameters for serving a model using SurogateServe.

    Args:
        infer_backend (Literal['vllm', 'pytorch', 'sglang']): Backend to use for inference. Default is 'pytorch'.
        host (str): The host address to bind the server to. Default is '0.0.0.0'.
        port (int): The port number to bind the server to. Default is 8000.
        api_key (Optional[str]): The API key for authentication. Default is None.
        served_name (Optional[str]): The name of the model being served. Default is None.
        max_logprobs(int): Max number of logprobs to return. Default is 20.
        seed (int): Random seed for reproducibility. Default is 1234.
        adapters (Optional[List[AdapterConfig]]): List of adapter configurations. Default is None.
        deterministic (bool): Whether to use deterministic inference. Default is False.
        max_context (Optional[int]): Maximum context length for the model. Default is None.
        tensor_parallel (int): Tensor parallelism size. Default is 1.
        max_memory (float): Maximum GPU memory utilization. Default is 0.9 (90%).
        use_chat_template (bool): Whether to use model's chat template. Default is True.
    """
    infer_backend: Optional[Literal['vllm', 'pytorch', 'sglang']] = None

    host: Optional[str] = None
    port: Optional[int] = None
    max_logprobs: Optional[int] = None
    api_key: Optional[str] = None
    served_name: Optional[str] = None
    seed: Optional[int] = None
    adapters: Optional[List[AdapterConfig]] = None
    deterministic: Optional[bool] = None
    max_context: Optional[int] = None
    tensor_parallel: Optional[int] = None
    max_memory: Optional[float] = None
    use_chat_template: Optional[bool] = None
    cache: Optional[ServeCacheConfig] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)

        self.infer_backend = cfg['infer_backend'] or 'pytorch'
        self.host = cfg['host'] or '0.0.0.0'
        self.port = cfg['port'] or 8000
        self.max_logprobs = cfg['max_logprobs'] or 20
        self.api_key = cfg['api_key']
        self.served_name = cfg['served_name']
        self.seed = cfg['seed']
        self.adapters = [AdapterConfig(cfg) for cfg in cfg['adapters']] if cfg['adapters'] else []
        self.deterministic = cfg['deterministic'] or False
        self.max_context = cfg['max_context']
        self.tensor_parallel = cfg['tensor_parallel'] or 1
        self.max_memory = cfg['max_memory'] or 0.9
        self.use_chat_template = cfg['use_chat_template'] or True
        self.cache = ServeCacheConfig(cfg['cache']) if cfg['cache'] else None
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        if self.infer_backend not in ['vllm', 'pytorch', 'sglang']:
            raise ValueError(f'Unsupported infer_backend: {self.infer_backend}. Supported backends: '
                             f'["vllm", "pytorch", "sglang"]')

        self.port = find_free_port(self.port)

