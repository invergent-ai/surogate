from dataclasses import dataclass
from typing import Optional
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()

@dataclass
class ServerConfig:
    host: str = '0.0.0.0'
    port: int = 8888
    workers: int = 1
    log_level: str = 'info'
    database_url: str = 'postgresql+asyncpg://surogate:surogate@127.0.0.1:5432/surogate'
    dstack_database_url: str = 'postgresql+asyncpg://dstack:dstack@127.0.0.1:5432/dstack'
    lakefs_endpoint: Optional[str] = None
    lakefs_s3_endpoint: Optional[str] = None
    lakefs_k8s_s3_endpoint: Optional[str] = 'http://lakefs-s3.lakefs.svc'
    lakefs_access_key: Optional[str] = None
    prometheus_endpoint: Optional[str] = 'https://metrics.k8s.localhost'
    
    lakefs_secret_key: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get('host', self.host)
        self.port = cfg.get('port', self.port)
        self.workers = cfg.get('workers', self.workers)
        self.log_level = cfg.get('log_level', self.log_level)
        self.database_url = cfg.get('database_url', self.database_url)
        self.dstack_database_url = cfg.get('dstack_database_url', self.dstack_database_url)
        self.lakefs_endpoint = cfg.get('lakefs_endpoint', self.lakefs_endpoint)
        self.lakefs_s3_endpoint = cfg.get('lakefs_s3_endpoint', self.lakefs_s3_endpoint)
        self.lakefs_access_key = cfg.get('lakefs_access_key', self.lakefs_access_key)
        self.lakefs_secret_key = cfg.get('lakefs_secret_key', self.lakefs_secret_key)

    def __post_init__(self):
        pass