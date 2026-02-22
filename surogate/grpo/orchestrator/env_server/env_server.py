import asyncio

from verifiers.workers import ZMQEnvServer

from surogate.grpo.orchestrator.env_server.config import EnvServerConfig
from surogate.grpo.utils.logger import setup_logger
from surogate.grpo.utils.pathing import get_log_dir
from surogate.grpo.utils.pydantic_config import parse_argv
from surogate.grpo.utils.utils import clean_exit, get_env_ids_to_install, install_env, strip_env_version


@clean_exit
def run_server(config: EnvServerConfig):
    setup_logger(config.log.level, json_logging=config.log.json_logging)

    # install environment if not already installed
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install([config.env]))
    for env_id in env_ids_to_install:
        install_env(env_id)

    env_name = config.env.name or config.env.id
    log_file = (get_log_dir(Path(config.output_dir)) / "train" / f"{env_name}.log").as_posix()

    server = ZMQEnvServer(
        env_id=strip_env_version(config.env.id),
        env_args=config.env.args,
        extra_env_kwargs=config.env.extra_env_kwargs,
        log_level=config.log.level,
        log_file_level=config.log.vf_level,
        log_file=log_file,
        json_logging=config.log.json_logging,
        **{"address": config.env.address} if config.env.address is not None else {},
    )
    asyncio.run(server.run())


def main():
    """Main entry-point for env-server. Run using `uv run env-server`"""
    run_server(parse_argv(EnvServerConfig))


if __name__ == "__main__":
    main()
