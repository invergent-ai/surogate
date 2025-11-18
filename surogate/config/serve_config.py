from dataclasses import dataclass
from typing import Optional, Literal, List

from surogate.config.model_config import ModelConfig
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
class ServeConfig(ModelConfig, AdapterConfig):
    """
    ServeConfig class is a dataclass that holds configuration parameters for serving a model using SurogateServe.

    Args:
        infer_backend (Literal['vllm', 'pytorch', 'sglang']): Backend to use for inference. Default is 'pytorch'.
        host (str): The host address to bind the server to. Default is '0.0.0.0'.
        port (int): The port number to bind the server to. Default is 8000.
        api_key (Optional[str]): The API key for authentication. Default is None.
        served_model_name (Optional[str]): The name of the model being served. Default is None.
        max_logprobs(int): Max number of logprobs to return. Default is 20.
        seed (int): Random seed for reproducibility. Default is 1234.
        adapters (Optional[List[AdapterConfig]]): List of adapter configurations. Default is None.
        deterministic (bool): Whether to use deterministic inference. Default is False.
    """
    infer_backend: Optional[Literal['vllm', 'pytorch', 'sglang']] = None

    host: Optional[str] = None
    port: Optional[int] = None
    max_logprobs: Optional[int] = None
    api_key: Optional[str] = None
    served_model_name: Optional[str] = None
    seed: Optional[int] = None
    adapters: Optional[List[AdapterConfig]] = None
    deterministic: Optional[bool] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)

        self.infer_backend = cfg['infer_backend'] or 'pytorch'
        self.host = cfg['host'] or '0.0.0.0'
        self.port = cfg['port'] or 8000
        self.max_logprobs = cfg['max_logprobs'] or 20
        self.api_key = cfg['api_key']
        self.served_model_name = cfg['served_model_name']
        self.seed = cfg['seed']
        self.adapters = [AdapterConfig(cfg) for cfg in cfg['adapters']] if cfg['adapters'] else []
        self.deterministic = cfg['deterministic'] or False
        self.__post_init__()

    def __post_init__(self):
        if not self.infer_backend in ['vllm', 'pytorch', 'sglang']:
            raise ValueError(f'Unsupported infer_backend: {self.infer_backend}. Supported backends: '
                             f'["vllm", "pytorch", "sglang"]')

        self.port = find_free_port(self.port)
