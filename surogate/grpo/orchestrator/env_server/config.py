from pathlib import Path
from typing import Annotated

from pydantic import Field

from surogate.core.config.grpo_orch_config import EnvConfig
from surogate.grpo.utils.config import LogConfig
from surogate.grpo.utils.pydantic_config import BaseSettings


class EnvServerConfig(BaseSettings):
    """Configures an environment server."""

    env: EnvConfig = EnvConfig()
    log: LogConfig = LogConfig()

    output_dir: Annotated[
        Path,
        Field(
            description="Directory to write outputs to. Will be populated with checkpoints, weights, rollouts and logs as subdirectories. Should be set to a persistent directory with enough disk space. This value should be distinct across experiments running on a single node. See the README for more details."
        ),
    ] = Path("outputs/run_default")
