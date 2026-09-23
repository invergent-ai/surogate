"""The trainer must publish weights where the orchestrator looks for them.

2026-09-23: it did not. `SurogateWeightBroadcast` inferred the directory by
globbing the train output_dir for `run_*`, because the train config does not
carry the orchestrator's output_dir. The guess resolves to `run_default` two
different ways -- by the no-match fallback, and by `sorted(...)[0]` when
several exist, since "run_default" sorts before almost anything else -- so any
orchestrator configured elsewhere waited forever for weights written somewhere
else. The shipped tool example (`tools-orch.yaml`, output_dir
`outputs/grpo/run_tools`) could never train.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from surogate.grpo.utils.pathing import (
    get_broadcast_dir,
    resolve_broadcast_dir,
    wait_for_path,
)


# The shipped function, not a copy of it. It lives in pathing.py precisely so
# that it can be reached without the CUDA extension the broadcaster's module
# pulls in -- an earlier version of this file reimplemented the resolution and
# would have kept passing after the real code changed.
_resolve = resolve_broadcast_dir


def test_an_explicit_dir_is_used_verbatim(tmp_path):
    """What split mode now passes, and the only thing that makes them agree."""
    orch_output = tmp_path / "outputs" / "grpo" / "run_tools"
    resolved = _resolve(tmp_path / "outputs" / "grpo", get_broadcast_dir(orch_output))
    assert resolved == orch_output / "broadcasts"


def test_the_guess_misses_any_run_dir_that_is_not_run_default(tmp_path):
    """The bug itself, both ways it resolves wrongly.

    Not a hypothetical: this is `tools-orch.yaml` against `train.yaml`.
    """
    train_out = tmp_path / "outputs" / "grpo"
    orch_out = train_out / "run_tools"
    get_broadcast_dir(orch_out).mkdir(parents=True)

    # Nothing to glob yet -> fallback.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _resolve(empty) == empty / "run_default" / "broadcasts"

    # Both present -> alphabetical first, which is still not run_tools.
    (train_out / "run_default").mkdir(parents=True)
    guessed = _resolve(train_out)
    assert guessed == train_out / "run_default" / "broadcasts"
    assert guessed != get_broadcast_dir(orch_out)


def test_the_guess_announces_itself(tmp_path, caplog):
    """A silent guess is what made this cost a day; an explicit one says nothing."""
    import logging

    with caplog.at_level(logging.WARNING):
        _resolve(tmp_path)
    assert any("guessing" in r.message for r in caplog.records), "the fallback must warn"

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _resolve(tmp_path, tmp_path / "given")
    assert not [r for r in caplog.records if "guessing" in r.message], (
        "an explicit directory has nothing to warn about"
    )


def test_waiting_for_weights_gives_up_instead_of_hanging(tmp_path):
    """The second half: a mismatch must become an error, not a forever-wait."""
    missing = tmp_path / "step_1" / "STABLE"

    with pytest.raises(TimeoutError) as excinfo:
        asyncio.run(wait_for_path(missing, interval=1, timeout=1))

    # The message has to name the cause, because the symptom (a run that looks
    # healthy) points nowhere.
    assert str(missing) in str(excinfo.value)
    assert "broadcast" in str(excinfo.value)


def test_waiting_still_succeeds_when_the_path_appears(tmp_path):
    """The timeout must not break the path it exists to bound."""
    present = tmp_path / "STABLE"
    present.touch()
    asyncio.run(wait_for_path(present, interval=1, timeout=5))
