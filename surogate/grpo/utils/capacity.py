"""How much of a run each engine should agree to hold at once.

Kept out of `split.py`, which is where this is called from but which cannot be
imported without the CUDA extension, so the arithmetic would only be testable on
a GPU. There is nothing device-shaped about it: it reconciles numbers that two
different configs know separately.
"""

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.utils.logger import get_logger

logger = get_logger()

# The engine's own default (serve_options.h). It is a serving default, not a
# training one: a server answering ad-hoc callers should shed load, a training
# run should not shed its own step.
ENGINE_DEFAULT_PENDING = 16

# `http_pool_sizes` gives every held request a worker thread, eagerly created.
# A thread is cheap in memory (~8 kB resident) but not free in process count:
# `std::thread` throws from the pool's constructor once the cgroup's `pids.max`
# is reached, and container limits of 1024 are ordinary. That failure lands at
# engine startup, before a single request is served, which is worse than the
# 429s this exists to avoid. Kept well under 1024 total threads, since the
# worker count is this plus the concurrency plus one. A run that genuinely wants
# more can say so explicitly.
MAX_DERIVED_PENDING = 512

# How long the engine holds a request, as a fraction of the caller's own
# deadline. Below 1, deliberately: past the caller's timeout nobody is waiting
# for the answer, and a held request keeps its admission slot and its worker for
# as long as the hold lasts, so the client's own retry queues behind a request
# it has already abandoned. Expiring just before the caller gives up frees the
# slot for that retry.
PENDING_TIMEOUT_FRACTION = 0.9
DEFAULT_CLIENT_TIMEOUT_S = 1200

# How long a weight update may wait for the adapter to stop being read.
#
# A request claims the adapter when it is admitted, not when it starts
# generating, and `load_lora_adapter` waits for every claim to clear. So this is
# really "how long may the admitted backlog take to drain", and the longest a
# single rollout can legitimately hold its claim is its caller's own timeout:
# past that nobody is waiting for it. Hence 1.0 rather than a guess.
#
# It is deliberately NOT `pending_timeout_ms`, which the engine used for both
# jobs. Sizing the pending hold to the run then silently sized this too, and
# pushed it past the admin client's patience, so the update failed as an opaque
# client-side timeout rather than as the engine's own 503. Two jobs, two numbers.
ADAPTER_UPDATE_FRACTION = 1.0

# The admin client has to outlast the deadline above, because whoever gives up
# first decides what the run reports. The engine's 503 names the adapter it was
# waiting on; a client timeout can only say the socket went quiet. This margin
# is what keeps the engine ahead in that race.
ADMIN_TIMEOUT_MARGIN_S = 300


def admin_client_timeout_s(client_timeout: int | None) -> float:
    """How long the weight-update POST should wait for an answer.

    Derived from the same input as the engine's own deadline rather than passed
    across the process boundary: `split.py` sizes the engine before the
    orchestrator is spawned, and two numbers that must stay ordered should not
    depend on that order holding.

    That shared derivation only keeps this above the engine's deadline while the
    engine's deadline is itself derived. An explicit
    `inference.adapter_update_timeout_ms` above this can invert the pair, and
    `_size` warns when it does, because this function cannot see that config: in
    3-process mode the orchestrator never holds it.
    """
    timeout_s = client_timeout or DEFAULT_CLIENT_TIMEOUT_S
    return timeout_s * ADAPTER_UPDATE_FRACTION + ADMIN_TIMEOUT_MARGIN_S


def size_pending_capacity(infer_config: GRPOInferenceConfig, orch_config: GRPOOrchestratorConfig) -> None:
    """Let the rollout server hold this run's rollouts instead of refusing them.

    `max_num_seqs + max_pending_requests` is the engine's admission bound, and a
    request past it is answered 429. The rollout client is the OpenAI SDK with
    ten retries and roughly 48s of backoff, so a request that stays over the
    bound for that long is not a slow rollout, it is a lost one, and enough lost
    ones trip the rollout-failure guard. How many arrive at once is the
    orchestrator's decision, not the engine's, and this is the only process that
    holds both numbers.

    Measured, 128 in flight against a bound of 24 and clients that give up the
    way the SDK does: 259 rollouts lost with the bound left at its default, 0
    with it raised past the in-flight count, and the same throughput either way,
    because the engine was the bottleneck in both. What changes is whether the
    queue is inside the engine, where it is FIFO and visible, or in the client's
    retry loop, where it is neither.
    """
    _size(infer_config, _rollout_demand(orch_config), orch_config)


def size_judge_pending_capacity(judge_config: GRPOInferenceConfig, orch_config: GRPOOrchestratorConfig) -> None:
    """The same, for a judge server spawned on its own port.

    It receives judge calls rather than rollouts, so it is sized to those. Left
    at the engine default it gets a bound of 17 against a default
    `max_concurrent_judges` of 32, and RULER swallows the resulting failures.
    """
    _size(judge_config, _judge_demand(orch_config), orch_config)


def _size(config: GRPOInferenceConfig, demand: int, orch_config: GRPOOrchestratorConfig) -> None:
    """Hold `demand` requests at once, within what the process can carry.

    An explicit value in the config always wins: someone who has set it has a
    reason, and the point of a default is to be replaceable.
    """
    concurrency = config.max_num_seqs or 1
    if config.max_pending_requests is None:
        wanted = max(ENGINE_DEFAULT_PENDING, demand - concurrency)
        config.max_pending_requests = min(MAX_DERIVED_PENDING, wanted)
        if wanted > MAX_DERIVED_PENDING:
            # Past this the bound stops covering the run, which is the failure
            # this module exists to prevent, so it cannot pass in silence.
            logger.warning(
                f"This run can have {demand} rollouts in flight, more than the "
                f"{MAX_DERIVED_PENDING + concurrency} the engine will hold without running its "
                f"thread pool past a container's pid limit. Requests over that bound are refused "
                f"and enough refusals become lost rollouts. Set inference.max_pending_requests "
                f"explicitly to override, or lower batch_size / oversampling_factor."
            )

    # Read the caller's deadline rather than assuming it: `client.timeout` is
    # configurable, so a hardcoded span drifts from it silently.
    client = orch_config.client
    timeout_s = (client.timeout if client else None) or DEFAULT_CLIENT_TIMEOUT_S
    # The two must stay ordered, and only one of them may be explicit, so the
    # ordering cannot be left to the two fractions alone. The config-level check
    # compares them only when BOTH are set, and it runs at construction, before
    # this. So an explicit value on one side used to reach argv inverted against a
    # derived value on the other, and the engine refuses that pair at startup: a
    # server that never turns healthy, which is what those checks exist to stop.
    #
    # The gap is the one the two fractions already imply rather than a fresh
    # constant, so tuning either fraction cannot silently change it.
    gap_ms = max(1, int(timeout_s * (ADAPTER_UPDATE_FRACTION - PENDING_TIMEOUT_FRACTION) * 1000))
    explicit_hold = config.pending_timeout_ms is not None

    if config.pending_timeout_ms is None:
        config.pending_timeout_ms = max(1, int(timeout_s * PENDING_TIMEOUT_FRACTION * 1000))

    if config.adapter_update_timeout_ms is None:
        # Clear the hold even when the hold was set by hand and is longer than the
        # caller's own deadline.
        config.adapter_update_timeout_ms = max(
            int(timeout_s * ADAPTER_UPDATE_FRACTION * 1000),
            config.pending_timeout_ms + gap_ms,
        )
    elif config.adapter_update_timeout_ms <= 1:
        # A hold has to be at least 1ms and the update deadline has to exceed it,
        # so nothing below 2ms can ever be satisfied. Say so here rather than
        # letting the engine refuse the pair after execv.
        raise ValueError(
            f"inference.adapter_update_timeout_ms is {config.adapter_update_timeout_ms}ms, which "
            f"leaves no room for a pending hold above zero; it must exceed "
            f"inference.pending_timeout_ms and so must be at least 2"
        )
    elif config.adapter_update_timeout_ms <= config.pending_timeout_ms and not explicit_hold:
        # The update deadline was chosen by hand and the hold was not, so the
        # derived one gives way. Loud, because a hold this short sheds requests the
        # caller would still have waited for.
        lowered = max(1, config.adapter_update_timeout_ms - gap_ms)
        logger.warning(
            f"inference.adapter_update_timeout_ms is {config.adapter_update_timeout_ms}ms, which is "
            f"not above the {config.pending_timeout_ms}ms hold derived from the rollout client's "
            f"{timeout_s}s timeout. Lowering the hold to {lowered}ms to keep the two ordered, since "
            f"the engine refuses the inverted pair at startup. Requests will be shed sooner than "
            f"the caller would wait; set inference.pending_timeout_ms explicitly to choose both."
        )
        config.pending_timeout_ms = lowered

    # An explicit pair that is inverted is refused at construction, so reaching
    # here inverted would mean this function did it.
    assert config.adapter_update_timeout_ms > config.pending_timeout_ms

    # The admin client derives its own deadline from the same T, so it only
    # outlasts the engine while the engine's deadline is the derived one. An
    # explicit value above that puts the opaque client-side timeout back in front
    # of the engine's 503, which is the swap this whole arrangement exists to
    # prevent. Warned rather than refused: it is a legitimate thing to want, and
    # `client.py` cannot see this config in 3-process mode, so this is the only
    # place the inversion is visible at all.
    admin_ms = int(admin_client_timeout_s(client.timeout if client else None) * 1000)
    if config.adapter_update_timeout_ms > admin_ms:
        logger.warning(
            f"inference.adapter_update_timeout_ms is {config.adapter_update_timeout_ms}ms but the "
            f"weight-update client gives up after {admin_ms}ms, so a slow adapter reload will fail "
            f"as a client-side timeout with nothing in the engine's log, rather than as the "
            f"engine's 503 naming the adapter. Raise client.timeout to move both."
        )


def _rollout_demand(orch_config: GRPOOrchestratorConfig) -> int:
    """How many requests the rollout server can have arriving at once."""
    demand = orch_config.max_inflight_rollouts

    # Online eval fans out on the same server, and by default it does not stop
    # the training rollouts first (`cancel_inflight_rollouts_on_eval` is False),
    # so the two bursts overlap. This is the burst that lost the weight update.
    evaluation = orch_config.eval
    if evaluation is not None and not evaluation.cancel_inflight_rollouts_on_eval:
        examples = evaluation.num_examples or 0
        # A negative count means "the whole set", whose size is not knowable
        # from config alone; the clamp below is what covers that case.
        if examples > 0:
            demand += examples * (evaluation.rollouts_per_example or 1)

    # A judge only competes for this server when it dials this server. Split
    # mode spawns one on its own port and GPUs (`_validate_judge_args` refuses
    # sharing either), and those calls land there instead.
    ruler = orch_config.ruler
    if ruler is not None and ruler.enabled and _judge_shares_rollout_server(orch_config):
        demand += _judge_demand(orch_config)
    return demand


def _judge_demand(orch_config: GRPOOrchestratorConfig) -> int:
    """How many judge calls can be in flight at once.

    `max_concurrent_judges` documents `None` as unbounded, so it cannot be read
    as zero; the judge's own connection pool is the real ceiling in that case.
    """
    ruler = orch_config.ruler
    if ruler is None or not ruler.enabled:
        return 0
    if ruler.max_concurrent_judges is not None:
        return ruler.max_concurrent_judges
    return ruler.judge.max_connections or 0


def _judge_shares_rollout_server(orch_config: GRPOOrchestratorConfig) -> bool:
    """Does the RULER judge send its calls to the rollout server?

    A colocated judge is configured with the rollout server's own base URL, so
    its calls join the same queue. A judge given its own URL, or spawned on its
    own port by split mode, does not.
    """
    client = orch_config.client
    judge_urls = orch_config.ruler.judge.base_url or []
    return bool(set(judge_urls) & set((client.base_url if client else None) or []))
