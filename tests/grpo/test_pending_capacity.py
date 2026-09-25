"""The engine is told to hold this run's rollouts rather than refuse them.

The reasoning, and the measurements behind the numbers, are in
`surogate/grpo/utils/capacity.py`.
"""

from __future__ import annotations

from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
from surogate.grpo.utils.capacity import (
    ENGINE_DEFAULT_PENDING,
    MAX_DERIVED_PENDING,
    admin_client_timeout_s,
    size_judge_pending_capacity,
    size_pending_capacity,
)
from surogate.utils.dict import DictDefault


def _infer(**kwargs) -> GRPOInferenceConfig:
    return GRPOInferenceConfig(DictDefault({"model": "m", **kwargs}))


# An enabled ruler refuses to validate without both of these.
_RULER = {
    "enabled": True,
    "judge_model": "j",
    "judge": {"base_url": ["http://judge:9000/v1"]},
}


def _orch(**kwargs) -> GRPOOrchestratorConfig:
    cfg = {
        "model": {"name": "m"},
        "env": [{"id": "e", "path": "."}],
        "batch_size": 8,
        "rollouts_per_example": 8,
        "max_steps": 1,
        **kwargs,
    }
    return GRPOOrchestratorConfig(DictDefault(cfg))


def test_the_bound_covers_the_rollouts_the_orchestrator_keeps_in_flight():
    infer = _infer(max_num_seqs=8)
    orch = _orch(batch_size=128, rollouts_per_example=8)
    size_pending_capacity(infer, orch)
    assert infer.max_num_seqs + infer.max_pending_requests >= orch.max_inflight_rollouts


def test_a_small_run_keeps_the_engines_own_default():
    """Nothing is gained by holding more than a small run can produce."""
    infer = _infer(max_num_seqs=8)
    size_pending_capacity(infer, _orch(batch_size=8))
    assert infer.max_pending_requests == ENGINE_DEFAULT_PENDING


def test_an_explicit_value_is_never_overridden():
    """Someone who set this has a reason; derivation is only for the default."""
    infer = _infer(max_num_seqs=8, max_pending_requests=3)
    size_pending_capacity(infer, _orch(batch_size=128))
    assert infer.max_pending_requests == 3


def test_online_eval_counts_because_it_lands_on_the_same_server():
    """Eval does not stop the training rollouts first, so the bursts overlap.

    That overlap is the one that lost a weight update in the first place.
    """
    plain = _infer(max_num_seqs=8)
    size_pending_capacity(plain, _orch(batch_size=128))
    with_eval = _infer(max_num_seqs=8)
    size_pending_capacity(with_eval, _orch(batch_size=128, eval={"num_examples": 75, "rollouts_per_example": 4}))
    assert with_eval.max_pending_requests > plain.max_pending_requests


def test_eval_that_stops_the_rollouts_first_does_not_count():
    infer = _infer(max_num_seqs=8)
    size_pending_capacity(
        infer,
        _orch(
            batch_size=128,
            eval={
                "num_examples": 75,
                "rollouts_per_example": 4,
                "cancel_inflight_rollouts_on_eval": True,
            },
        ),
    )
    assert infer.max_pending_requests == 128 - 8


def test_the_derived_bound_stays_under_the_process_thread_ceiling():
    """`std::thread` throws from the pool constructor at the cgroup's pids.max."""
    infer = _infer(max_num_seqs=8)
    size_pending_capacity(infer, _orch(batch_size=8192))
    assert infer.max_pending_requests == MAX_DERIVED_PENDING
    assert infer.max_num_seqs + infer.max_pending_requests + 1 < 1024


def test_the_hold_expires_before_the_caller_gives_up():
    """Past the caller's deadline the slot is held for an answer nobody wants."""
    infer = _infer(max_num_seqs=8)
    orch = _orch(batch_size=128, client={"timeout": 1200})
    size_pending_capacity(infer, orch)
    assert infer.pending_timeout_ms < 1200 * 1000


def test_a_spawned_judge_server_is_sized_for_judge_calls():
    """Left at the engine default it gets 17 against 32 concurrent judges."""
    judge = _infer(max_num_seqs=1)
    size_judge_pending_capacity(judge, _orch(batch_size=128, ruler=_RULER))
    assert judge.max_num_seqs + judge.max_pending_requests >= 32


def test_an_eval_over_the_whole_set_does_not_shrink_the_bound():
    """`num_examples` defaults to -1, meaning "all of them", a size config
    cannot know. Multiplied out it would subtract from the demand instead of
    adding to it; the clamp below is what covers the unknown."""
    infer = _infer(max_num_seqs=8)
    size_pending_capacity(infer, _orch(batch_size=128, eval={"rollouts_per_example": 4}))
    assert infer.max_pending_requests == 128 - 8


def test_a_colocated_judge_counts_against_the_rollout_server():
    """A judge given the rollout server's own URL joins the same queue."""
    shared = {**_RULER, "judge": {"base_url": ["http://rollout:8000/v1"]}}
    colocated = _infer(max_num_seqs=8)
    size_pending_capacity(
        colocated,
        _orch(batch_size=128, client={"base_url": ["http://rollout:8000/v1"]}, ruler=shared),
    )
    elsewhere = _infer(max_num_seqs=8)
    size_pending_capacity(
        elsewhere,
        _orch(batch_size=128, client={"base_url": ["http://rollout:8000/v1"]}, ruler=_RULER),
    )
    assert colocated.max_pending_requests == elsewhere.max_pending_requests + 32


def test_unbounded_judges_fall_back_to_the_judges_own_connection_pool():
    """`max_concurrent_judges: null` documents itself as unbounded, so it
    cannot be read as zero. The pool is the real ceiling."""
    judge = _infer(max_num_seqs=1)
    unbounded = {**_RULER, "max_concurrent_judges": None}
    size_judge_pending_capacity(judge, _orch(batch_size=128, ruler=unbounded))
    assert judge.max_num_seqs + judge.max_pending_requests >= 256


def test_an_unset_concurrency_is_not_read_as_no_capacity():
    """`max_num_seqs` is None until bug 69 lands in ops, and the engine then
    runs one lane. Subtracting None would raise rather than derive."""
    infer = _infer()
    size_pending_capacity(infer, _orch(batch_size=128))
    assert infer.max_pending_requests == 128 - 1


def test_colocate_sizes_against_the_concurrency_it_will_actually_run():
    """Colocate derives its own concurrency when the config leaves it unset, and
    the C++ used to hardcode the bound at `max(16, that)`. Sizing has to see the
    same number the server runs, not the None the config still held."""
    infer = _infer()
    orch = _orch(batch_size=128)
    infer.max_num_seqs = 16  # what native_colocate.py computes and writes back
    size_pending_capacity(infer, orch)
    assert infer.max_pending_requests == 128 - 16


# --- the three deadlines that must never cross -------------------------------
#
# A pending request holds a claim on the adapter it has not started using, and
# loading new weights waits for every claim to clear. Before this the engine
# used ONE number for "how long may a request wait" and "how long may a weight
# update wait", so sizing the first to the run silently sized the second too,
# and pushed it past the admin client's own patience. These assert the order.


def test_the_update_deadline_outlasts_the_longest_legitimate_hold():
    """A request may sit pending for most of the caller's timeout, so a weight
    update that gives up sooner would abort a run that was merely busy."""
    infer = _infer(max_num_seqs=8)
    size_pending_capacity(infer, _orch(batch_size=128, client={"timeout": 1200}))
    assert infer.adapter_update_timeout_ms > infer.pending_timeout_ms


def test_the_admin_client_outlasts_the_update_deadline():
    """Whoever gives up first decides the error the run reports. The engine has
    to win that race: it names the adapter it was waiting on, where the client
    can only say the socket went quiet."""
    for client_timeout in (600, 1200, 3600):
        infer = _infer(max_num_seqs=8)
        orch = _orch(batch_size=128, client={"timeout": client_timeout})
        size_pending_capacity(infer, orch)
        assert admin_client_timeout_s(client_timeout) * 1000 > infer.adapter_update_timeout_ms


def test_the_whole_chain_is_ordered_for_any_caller_timeout():
    for client_timeout in (30, 600, 1200, 7200):
        infer = _infer(max_num_seqs=8)
        size_pending_capacity(infer, _orch(batch_size=128, client={"timeout": client_timeout}))
        assert (
            infer.pending_timeout_ms < infer.adapter_update_timeout_ms < admin_client_timeout_s(client_timeout) * 1000
        )


def test_an_explicit_update_deadline_is_never_overridden():
    infer = _infer(max_num_seqs=8, adapter_update_timeout_ms=5000)
    size_pending_capacity(infer, _orch(batch_size=128))
    assert infer.adapter_update_timeout_ms == 5000


def test_the_clamp_says_so_instead_of_silently_refusing_rollouts(caplog):
    """Past the thread ceiling the bound stops covering the run, which is the
    failure this module exists to prevent. It has to be audible."""
    import logging

    infer = _infer(max_num_seqs=8)
    with caplog.at_level(logging.WARNING):
        size_pending_capacity(infer, _orch(batch_size=8192))
    assert "max_pending_requests" in caplog.text


# --- the ordering must survive a mix of explicit and derived values ----------
#
# The config-level check compares the two fields only when BOTH are set, and it
# runs at construction, before any sizing. So an explicit value on one side and a
# derived value on the other reached argv inverted, and the engine refuses that
# pair at startup: a server that never turns healthy, which is the failure those
# checks exist to prevent. Both directions were broken.


def test_an_explicit_hold_pushes_the_derived_update_deadline_above_it():
    """Explicit pending, derived update. Was: pending 1500000 > update 1200000."""
    infer = _infer(max_num_seqs=8, pending_timeout_ms=1_500_000)
    size_pending_capacity(infer, _orch(batch_size=128, client={"timeout": 1200}))
    assert infer.adapter_update_timeout_ms > infer.pending_timeout_ms
    assert infer.pending_timeout_ms == 1_500_000, "an explicit value still wins"


def test_an_explicit_update_deadline_pulls_the_derived_hold_below_it():
    """Explicit update, derived pending. Was: pending 1080000 > update 60000."""
    infer = _infer(max_num_seqs=8, adapter_update_timeout_ms=60_000)
    size_pending_capacity(infer, _orch(batch_size=128, client={"timeout": 1200}))
    assert infer.adapter_update_timeout_ms > infer.pending_timeout_ms
    assert infer.adapter_update_timeout_ms == 60_000, "an explicit value still wins"
    assert infer.pending_timeout_ms >= 1


def test_the_ordering_holds_for_every_mix_of_explicit_and_derived():
    """The property, rather than the two cases that happened to be found."""
    for pending in (None, 1, 60_000, 1_080_000, 1_500_000, 9_000_000):
        # 2 is the floor: the hold must be >= 1 and the update must exceed it.
        for update in (None, 2, 60_000, 1_200_000, 9_000_000):
            if pending is not None and update is not None and update <= pending:
                continue  # refused at construction, tested separately
            kwargs = {}
            if pending is not None:
                kwargs["pending_timeout_ms"] = pending
            if update is not None:
                kwargs["adapter_update_timeout_ms"] = update
            infer = _infer(max_num_seqs=8, **kwargs)
            size_pending_capacity(infer, _orch(batch_size=128, client={"timeout": 1200}))
            assert infer.adapter_update_timeout_ms > infer.pending_timeout_ms, (
                f"inverted for pending={pending} update={update}: "
                f"got {infer.pending_timeout_ms} / {infer.adapter_update_timeout_ms}"
            )
            assert infer.pending_timeout_ms >= 1


def test_an_update_deadline_with_no_room_for_a_hold_is_refused_here():
    """1ms cannot be above a hold of at least 1ms, so no sizing satisfies it. The
    engine would refuse the pair after execv; this says so before the run starts."""
    import pytest

    infer = _infer(max_num_seqs=8, adapter_update_timeout_ms=1)
    with pytest.raises(ValueError, match="no room for a pending hold"):
        size_pending_capacity(infer, _orch(batch_size=128))
