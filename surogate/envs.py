import os
from typing import TYPE_CHECKING, Any, Literal, Callable

if TYPE_CHECKING:
    SUROGATE_TARGET_DEVICE: str = "cuda"
    SUROGATE_MAIN_CUDA_VERSION: str = "12.9"
    MAX_JOBS: str | None = None
    NVCC_THREADS: str | None = None
    CMAKE_BUILD_TYPE: Literal["Debug", "Release", "RelWithDebInfo"] | None = None

def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(
                f"Invalid value '{value}' for {env_name}. "
                f"Valid options: {actual_choices}."
            )

        return value

    return _get_validated_env

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    "SUROGATE_TARGET_DEVICE": lambda: os.getenv("SUROGATE_TARGET_DEVICE", "cuda").lower(),
    "SUROGATE_MAIN_CUDA_VERSION": lambda: os.getenv("SUROGATE_MAIN_CUDA_VERSION", "").lower() or "12.9",
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    "CMAKE_BUILD_TYPE": env_with_choices(
        "CMAKE_BUILD_TYPE", None, ["Debug", "Release", "RelWithDebInfo"]
    ),
}

def __getattr__(name: str):
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(environment_variables.keys())