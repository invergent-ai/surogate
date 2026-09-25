from __future__ import annotations

import asyncio
import os
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import verifiers as vf
from httpx import AsyncClient
from openai import NotFoundError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from surogate.core.config.grpo_orch_config import GRPOClientConfig
from surogate.grpo.utils.capacity import admin_client_timeout_s
from surogate.utils.logger import get_logger

logger = get_logger()


@runtime_checkable
class InferencePool(Protocol):
    """Protocol for inference pools."""

    @property
    def clients(self) -> list[vf.ClientConfig]:
        """Get inference clients."""
        ...

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_next_client(self) -> vf.ClientConfig:
        """Get next client in round-robin fashion."""
        ...

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        """Wait for inference pool to be ready."""
        ...

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        """Update weights on all inference servers."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Get pool metrics."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class StaticInferencePool:
    """Static inference pool with fixed client list."""

    def __init__(
        self, clients: list[vf.ClientConfig], admin_clients: list[AsyncClient], skip_model_check: bool = False
    ):
        self._clients = clients
        self._admin_clients = admin_clients
        self._skip_model_check = skip_model_check
        self._idx_to_client = {client.client_idx: client for client in clients}
        self._client_cycle = cycle(clients)
        self.model_name = None  # unused

    @property
    def clients(self) -> list[vf.ClientConfig]:
        return self._clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    async def get_next_client(self) -> vf.ClientConfig:
        return next(self._client_cycle)

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        await check_health(self._admin_clients, timeout=timeout)
        await maybe_check_has_model(self._admin_clients, model_name, skip_model_check=self._skip_model_check)

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        await update_weights(self._admin_clients, weight_dir, lora_name=lora_name, step=step)

    def get_metrics(self) -> dict[str, float]:
        return {}

    async def stop(self) -> None:
        pass


async def setup_inference_pool(
    client_config: GRPOClientConfig, model_name: str, client_type: str = "openai_chat_completions"
) -> InferencePool:
    """Create an inference pool from config."""

    logger.info(
        f"Initializing static inference pool (base_url={', '.join(client_config.base_url)}, "
        f"api_key_var={client_config.api_key_var}, headers={client_config.headers})"
    )
    return StaticInferencePool(
        clients=setup_clients(client_config, client_type=client_type),
        admin_clients=setup_admin_clients(client_config),
        skip_model_check=client_config.skip_model_check,
    )


def setup_clients(
    client_config: GRPOClientConfig, client_type: str = "openai_chat_completions"
) -> list[vf.ClientConfig]:
    def setup_client(client_idx: int, base_url: str) -> vf.ClientConfig:
        return vf.ClientConfig(
            client_idx=client_idx,
            client_type=client_type,
            api_base_url=base_url,
            api_key_var=client_config.api_key_var,
            timeout=client_config.timeout,
            # verifiers defaults connect_timeout to 5.0s. Under a completion
            # wave (hundreds of rollouts firing conductor calls as their harbor
            # steps finish together) the orchestrator's own event loop delays
            # the TCP handshake past 5s CLIENT-side, and the call dies as
            # ConnectTimeout while the conductor sits idle — measured
            # 2026-08-14: abort bursts of ~70/min against a server showing
            # 0-9 running requests, 0 waiting, 0.3ms accept latency. Aborted
            # rollouts empty their group and zero its GRPO advantage, so the
            # 5s default was silently deleting training signal.
            # 60 -> 1200 (2026-08-15, second bite of the same class): the swe
            # lane runs episodes in asyncio.to_thread INSIDE the env process,
            # and thread/GIL pressure during busy phases stalls the event loop
            # long enough that pending conductor connects miss a 60s window —
            # plan calls die as ConnectTimeout while the conductor idles at
            # ~12ms (swebench group 7: 42 reschedules at 0/64, zero artifacts,
            # terminal lane unaffected because harbor episodes are
            # SUBPROCESSES). On localhost a separate connect timer only
            # false-kills; equal to the total budget it can never fire first.
            connect_timeout=1200.0,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers=client_config.headers,
        )

    return [setup_client(client_idx, base_url) for client_idx, base_url in enumerate(client_config.base_url)]


def setup_admin_clients(client_config: GRPOClientConfig) -> list[AsyncClient]:
    """Create a dedicated admin client for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    """

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        headers = client_config.headers.copy()  # avoid mutating config
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            # A finite deadline, not Timeout(None): the weight-update POST runs
            # inside the scheduler's _apply_policy_update, which gates all new
            # rollout scheduling. With no deadline, a server that accepts the
            # request but never finishes the reload wedges the orchestrator
            # silently and indefinitely (observed 2026-08-14: conductor reload
            # at 20:47 -> zero accepted rollouts for ~6h, no error anywhere).
            #
            # Derived rather than fixed, and deliberately LONGER than the
            # engine's own `adapter_update_timeout_ms`. A weight update waits for
            # every admitted request to release its adapter claim, so the wait
            # scales with the in-flight backlog, and a flat 600s aborted runs
            # that were draining normally. Whoever gives up first decides what
            # the run reports: the engine's 503 names the adapter it was waiting
            # on, a timeout here only says the socket went quiet. Expiry is NOT
            # retried and is not recovered from -- `_apply_policy_update`'s
            # exception is stored and re-raised by
            # `raise_if_policy_update_failed`, which aborts the run.
            timeout=httpx.Timeout(admin_client_timeout_s(client_config.timeout), connect=60.0),
        )

    return [_setup_admin_client(base_url) for base_url in client_config.base_url]


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin_client.get("/v1/models") for admin_client in admin_clients])
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
) -> None:
    """Update weights on static inference servers.

    With a LoRA adapter the server hot-loads it through /load_lora_adapter. A
    full-model update posts /update_weights, which the engine does not serve
    yet, so that path fails the run rather than continuing without it.
    """
    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
    else:

        async def _update_weights(admin_client: AsyncClient, weight_dir: str | None) -> None:
            try:
                response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir})
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Skipping this leaves the policy frozen at its initial weights
                    # while the trainer keeps stepping and broadcasting: the run
                    # completes, reports a loss curve, and every rollout after the
                    # first came from a model that never changed.
                    raise RuntimeError(
                        "The rollout engine cannot reload full weights, so this "
                        "run would train against a policy that never updates. "
                        "Use lora: true."
                    ) from e
                raise

        await asyncio.gather(*[_update_weights(admin_client, weight_dir_posix) for admin_client in admin_clients])


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    # A failure with no status at all: the connection was refused, dropped, or
    # never established. This used to fall through unretried, so one dead socket
    # ended a run that had already trained and already saved its adapter. Loading
    # an adapter by name and path is idempotent, so a second attempt is safe.
    #
    # Not a read timeout, though it is also a TransportError. That one means the
    # server took the request and has not answered within the deadline, so it is
    # very likely still reloading, and retrying stacks ten more 600s waits on top
    # of a reload that may yet succeed. `_apply_policy_update` gates all new
    # rollout scheduling while this runs and leaves `checkpoint_ready` cleared,
    # so those waits are a silently stalled run -- which is the failure the
    # deadline was added to end, not to multiply.
    if isinstance(exception, httpx.ReadTimeout):
        return False
    return isinstance(exception, httpx.TransportError)


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Ask the inference server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    lora_path_posix = lora_path.as_posix()

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        response = await admin_client.post(
            "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path_posix},
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])
