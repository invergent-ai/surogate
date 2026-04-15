from dataclasses import dataclass, fields
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
    database_url: str = 'postgresql+asyncpg://surogate:surogate@127.0.0.1:32432/surogate'
    dstack_database_url: str = 'postgresql+asyncpg://dstack:dstack@127.0.0.1:32432/dstack'
    surogates_database_url: str = 'postgresql+asyncpg://surogates:surogates@127.0.0.1:32432/surogates'
    lakefs_endpoint: Optional[str] = None
    lakefs_s3_endpoint: Optional[str] = None
    lakefs_k8s_s3_endpoint: Optional[str] = 'http://lakefs-s3.lakefs.svc'
    lakefs_access_key: Optional[str] = None
    lakefs_secret_key: Optional[str] = None
    prometheus_endpoint: Optional[str] = 'https://metrics.k8s.localhost'

    agent_s3_endpoint: Optional[str] = 'https://garage.k8s.localhost'
    agent_s3_region: Optional[str] = 'garage'
    agent_s3_bucket: Optional[str] = 'surogates'
    agent_s3_access_key: Optional[str] = None
    agent_s3_secret_key: Optional[str] = None

    def __init__(self, cfg: DictDefault):
        for f in fields(self):
            setattr(self, f.name, cfg.get(f.name, f.default))
