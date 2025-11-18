from dataclasses import dataclass
from typing import Optional, Literal

from surogate.utils.dict import DictDefault
from surogate.utils.net import find_free_port


@dataclass
class ServeConfig:
    """
    ServeConfig class is a dataclass that holds configuration parameters for serving a model using SurogateServe.

    Args:
        infer_backend (Literal['vllm', 'pytorch', 'sglang']): Backend to use for inference. Default is 'pytorch'.
        host (str): The host address to bind the server to. Default is '0.0.0.0'.
        port (int): The port number to bind the server to. Default is 8000.
        api_key (Optional[str]): The API key for authentication. Default is None.
        served_model_name (Optional[str]): The name of the model being served. Default is None.
        max_logprobs(int): Max number of logprobs to return. Default is 20.
    """
    infer_backend: Literal['vllm', 'pytorch', 'sglang']

    host: str
    port: int
    max_logprobs: int
    api_key: Optional[str] = None
    served_model_name: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.infer_backend = cfg['infer_backend'] or 'pytorch'
        self.host = cfg['host'] or '0.0.0.0'
        self.port = cfg['port'] or 8000
        self.max_logprobs = cfg['max_logprobs'] or 20
        self.api_key = cfg['api_key']
        self.served_model_name = cfg['served_model_name']
        self.__post_init__()

    def __post_init__(self):
        if not self.infer_backend in ['vllm', 'pytorch', 'sglang']:
            raise ValueError(f'Unsupported infer_backend: {self.infer_backend}. Supported backends: '
                             f'["vllm", "pytorch", "sglang"]')

        self.port = find_free_port(self.port)
