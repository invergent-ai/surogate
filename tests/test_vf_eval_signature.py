"""`vf-eval`'s calls into `verifiers` still bind against the pinned version.

`surogate vf-eval` calls two functions in `verifiers.utils.eval_utils` and does
nothing else interesting at startup, so a renamed parameter takes the whole
command out:

    TypeError: run_evaluations_tui() got an unexpected keyword argument 'tui_mode'

That is what happened across `verifiers` 0.1.11 -> 0.3.0, which renamed
`tui_mode` to `fullscreen`. Nothing caught it, because nothing in the suite
invokes the CLI, so the command was dead in the published image for as long as
the pin had been current. The cost was not the crash; `vf-eval` is the tool you
reach for when a reward is unexpectedly zero, so its being broken pushed that
work into hand-written probes, which then gave a wrong answer of their own.

These read the **real call sites** out of `vf_eval.py` with `ast` and bind those
shapes against the installed signatures. An earlier version of this file bound a
hand-copied `(config, fullscreen=True)` instead, which was vacuous in the
direction that matters: editing the call site alone left it green, and its
mutation check had really only mutated the test. Parsing the source means the
test follows the code both ways -- a reorder, an added keyword, or a revert to
`tui_mode=` all fail here.

The module is located through the installed package rather than a path relative
to the repo, so this file can also be mounted into the built image and run
against the pinned `verifiers` there. `.github/workflows/docker.yml` does that
on every image build, which is the only place it runs unattended.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

pytest.importorskip("verifiers", reason="verifiers is imported only by the vf-eval CLI")


def _call_shapes(func_name: str) -> list[tuple[int, list[str]]]:
    """Every `func_name(...)` call in vf_eval.py, as (positional count, keyword names)."""
    import surogate.cli.vf_eval as vf_eval

    tree = ast.parse(Path(vf_eval.__file__).read_text(encoding="utf-8"))
    shapes = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != func_name:
            continue
        # `*args` / `**kwargs` would make the shape unknowable here. No call
        # uses them today, and one appearing should fail loudly rather than be
        # waved through as a bind that proves nothing.
        assert not any(isinstance(a, ast.Starred) for a in node.args), f"{func_name}: *args"
        names = [kw.arg for kw in node.keywords]
        assert all(names), f"{func_name}: **kwargs"
        shapes.append((len(node.args), [n for n in names if n]))
    return shapes


def _assert_calls_bind(func_name: str) -> None:
    from verifiers.utils import eval_utils

    func = getattr(eval_utils, func_name, None)
    assert func is not None, f"verifiers.utils.eval_utils.{func_name} is gone; vf_eval.py imports it"

    shapes = _call_shapes(func_name)
    assert shapes, f"no call to {func_name} found in vf_eval.py; this test no longer guards anything"

    signature = inspect.signature(func)
    for positional, keywords in shapes:
        signature.bind(*[object()] * positional, **{name: object() for name in keywords})


def test_the_display_call_still_binds() -> None:
    _assert_calls_bind("run_evaluations_tui")


def test_the_debug_call_still_binds() -> None:
    _assert_calls_bind("run_evaluations")
