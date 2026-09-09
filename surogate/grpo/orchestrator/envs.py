from typing import TYPE_CHECKING, Any

from surogate.grpo.utils.envs import _ENV_PARSERS as _BASE_ENV_PARSERS, get_env_value, get_dir, set_defaults

if TYPE_CHECKING:
    # Enable type checking for shared envs
    # ruff: noqa
    from surogate.grpo.utils.envs import *

    # tqdm
    TQDM_DISABLE: int


_ORCHESTRATOR_ENV_PARSERS = {
    "TQDM_DISABLE": int,
    **_BASE_ENV_PARSERS,
}

_ORCHESTRATOR_ENV_DEFAULTS: dict[str, str] = {}

set_defaults(_ORCHESTRATOR_ENV_DEFAULTS)


def __getattr__(name: str) -> Any:
    return get_env_value(_ORCHESTRATOR_ENV_PARSERS, name)


def __dir__() -> list[str]:
    return get_dir(_ORCHESTRATOR_ENV_PARSERS)
