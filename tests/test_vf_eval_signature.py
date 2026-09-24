"""`vf-eval`'s calls into `verifiers` still bind against the pinned version.

`surogate vf-eval` calls two functions in `verifiers.utils.eval_utils` and does
nothing else interesting at startup, so a renamed parameter takes the whole
command out:

    TypeError: run_evaluations_tui() got an unexpected keyword argument 'tui_mode'

That is exactly what happened across `verifiers` 0.1.11 -> 0.3.0, which renamed
`tui_mode` to `fullscreen`. Nothing caught it, because nothing in the suite
invokes the CLI, and it surfaced instead as a command that had been dead in the
published image for as long as the pin had been current. The cost was not the
crash; it was that `vf-eval` is the tool you reach for when a reward is
unexpectedly zero, so its being broken pushed that work into hand-written
probes -- which then gave a wrong answer of their own.

These bind the real call shapes rather than assert a parameter name, so a
reorder or a removal fails here too. They check the contract, not the
behaviour: `verifiers` is free to change what the display does, and this stays
green as long as the call still fits.
"""

from __future__ import annotations

import inspect

import pytest

verifiers = pytest.importorskip("verifiers", reason="verifiers is an optional import for the CLI")


def _sig(name: str) -> inspect.Signature:
    from verifiers.utils import eval_utils

    fn = getattr(eval_utils, name, None)
    assert fn is not None, f"verifiers.utils.eval_utils.{name} is gone; vf_eval.py imports it"
    return inspect.signature(fn)


def test_the_display_call_still_binds() -> None:
    """Mirrors `run_evaluations_tui(eval_run_config, fullscreen=args.tui)`."""
    _sig("run_evaluations_tui").bind(object(), fullscreen=True)


def test_the_debug_call_still_binds() -> None:
    """Mirrors the `--debug` branch, `run_evaluations(eval_run_config)`."""
    _sig("run_evaluations").bind(object())
