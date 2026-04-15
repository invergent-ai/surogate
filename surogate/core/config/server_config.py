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

    # S3 endpoint used by agent pods — must be an in-cluster DNS name
    # since ``garage.k8s.localhost`` only resolves on the host.
    agent_s3_endpoint: Optional[str] = 'http://surogates-s3-garage.default.svc:3900'
    agent_s3_region: Optional[str] = 'garage'
    agent_s3_bucket: Optional[str] = 'surogates'
    agent_s3_access_key: Optional[str] = None
    agent_s3_secret_key: Optional[str] = None

    surogates_helm_chart: str = '/work/surogates/helm/surogates'
    agent_base_domain: str = 'k8s.localhost'
    # Database + Redis URLs handed to agent pods.  The surogate server
    # itself reaches these via host-exposed NodePorts, but pods must
    # use in-cluster DNS.
    agent_surogates_database_url: str = (
        'postgresql+asyncpg://surogates:surogates@surogate-db-postgresql.default.svc:5432/surogates'
    )
    # Worker pods run inside k3d; the surogate server runs on the host.
    # k3d auto-injects ``host.k3d.internal`` into CoreDNS pointing at the
    # host's default bridge IP, so pods can reach host services there.
    platform_api_url: str = 'http://host.k3d.internal:8888'

    # Built-in ``web_search`` / ``web_extract`` / ``web_crawl`` tools
    # pick a backend from env vars; Tavily is the default.  Blank
    # disables web search in the worker.
    tavily_api_key: Optional[str] = None
    agent_redis_url: str = 'redis://surogates-redis-master.default.svc:6379/0'
    kubeconfig_path: str = '~/.surogate/kubeconfig'
    helm_binary: str = '~/.surogate/bin/helm'

    def __init__(self, cfg: DictDefault):
        for f in fields(self):
            setattr(self, f.name, cfg.get(f.name, f.default))
