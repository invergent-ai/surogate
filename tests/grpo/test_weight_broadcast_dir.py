"""The trainer must publish weights where the orchestrator looks for them.

2026-09-23: it did not. The directory was inferred by globbing the *train*
output_dir for `run_*`, because the train config does not carry the
orchestrator's output_dir and nobody passed it. See `guess_broadcast_dir` for
why that can only ever land on `run_default`, and the TimeoutError in
`wait_for_path` for what it looked like from the outside.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from surogate.utils.logger import get_logger
from surogate.grpo.utils.pathing import (
    get_broadcast_dir,
    guess_broadcast_dir,
    sync_wait_for_path,
    wait_for_path,
)


def test_the_orchestrators_directory_is_used_verbatim(tmp_path):
    """What split mode passes, and the only thing that makes the two agree."""
    orch_output = tmp_path / "outputs" / "grpo" / "run_tools"
    assert get_broadcast_dir(orch_output) == orch_output / "broadcasts"


def test_the_guess_misses_any_run_dir_that_is_not_run_default(tmp_path, caplog):
    """The bug itself, both ways it resolves wrongly.

    Not hypothetical: this is `tools-orch.yaml` (run_tools) against
    `train.yaml`. The guess also has to announce itself -- a silent one is what
    made this cost a day.
    """
    train_out = tmp_path / "outputs" / "grpo"
    orch_out = train_out / "run_tools"
    get_broadcast_dir(orch_out).mkdir(parents=True)

    # Nothing to glob yet -> fallback.
    empty = tmp_path / "empty"
    empty.mkdir()
    # Zero candidates: uncertain, so it warns. Exactly one candidate does not,
    # because that is the case colocate guarantees is correct.
    # Not caplog: get_logger sets `propagate = False` (logger.py), so records
    # never reach the root handler caplog installs. Attach to the logger the
    # code actually uses.
    said: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record):
            said.append(record.getMessage())

    handler = _Catch()
    real = get_logger()._logger
    real.addHandler(handler)
    try:
        assert guess_broadcast_dir(empty) == empty / "run_default" / "broadcasts"
    finally:
        real.removeHandler(handler)
    assert any("broadcast" in m.lower() for m in said), "the guess must warn"

    # Both present -> alphabetical first, which is still not run_tools.
    (train_out / "run_default").mkdir(parents=True)
    guessed = guess_broadcast_dir(train_out)
    assert guessed == train_out / "run_default" / "broadcasts"
    assert guessed != get_broadcast_dir(orch_out)


def test_an_unambiguous_guess_stays_quiet(tmp_path):
    """One `run_*` is what colocate guarantees; warning there is a false alarm.

    The warning fired on every colocate run before this was scoped, telling
    operators their weights were going somewhere nothing reads when they were
    not.
    """
    train_out = tmp_path / "outputs" / "grpo"
    (train_out / "run_only").mkdir(parents=True)

    said: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record):
            said.append(record.getMessage())

    handler = _Catch()
    real = get_logger()._logger
    real.addHandler(handler)
    try:
        assert guess_broadcast_dir(train_out) == train_out / "run_only" / "broadcasts"
    finally:
        real.removeHandler(handler)
    assert not [m for m in said if "broadcast" in m.lower()], "one candidate is not ambiguous"


def test_waiting_for_weights_gives_up_instead_of_hanging(tmp_path):
    """The orchestrator's side: a mismatch becomes an error, not a forever-wait."""
    missing = tmp_path / "step_1" / "STABLE"

    with pytest.raises(TimeoutError) as excinfo:
        # `0` disables the timeout, here as in batch_stall_timeout, so a
        # small positive bound is what exercises it.
        asyncio.run(wait_for_path(missing, interval=0.01, timeout=0.05))

    assert str(missing) in str(excinfo.value)
    assert "broadcast" in str(excinfo.value)


def test_the_trainers_side_gives_up_too(tmp_path):
    """The same pipe, the other direction.

    `sync_wait_for_path` is how the trainer waits for a rollout micro-batch.
    Bounding only the orchestrator's wait would have fixed half a symmetric
    hang.
    """
    missing = tmp_path / "step_1" / "rank_0.bin"

    with pytest.raises(TimeoutError) as excinfo:
        sync_wait_for_path(missing, interval=0.01, timeout=0.05)

    assert str(missing) in str(excinfo.value)


@pytest.mark.parametrize("waiter", ["async", "sync"])
def test_waiting_still_succeeds_when_the_path_appears(tmp_path, waiter):
    """The timeouts must not break the path they exist to bound."""
    present = tmp_path / "STABLE"
    present.touch()
    if waiter == "async":
        asyncio.run(wait_for_path(present, interval=0.01, timeout=5))
    else:
        sync_wait_for_path(present, interval=0.01, timeout=5)
