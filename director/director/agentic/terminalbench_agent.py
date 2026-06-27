"""Terminal-Bench integration: the Director router as a terminal-bench Agent.

terminal-bench owns the task container + native grading; we plug in as a ``BaseAgent``
(via ``agent_import_path``) whose ``perform_task`` runs our per-step routing loop against
the tmux session. Each turn the router selects a worker (over the raw transcript), the
worker emits one shell command, we run it in the session, capture output, and repeat —
the same decision-only, per-step routing as the SWE-Bench path. Resolution is computed
by terminal-bench's own tests/parsers.

Run via ``director terminal-bench-eval`` (see agentic.run), which points the harness at
``director.agentic.terminalbench_agent.DirectorAgent``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from ..fugu.inference import select_worker
from ..shared.transcript import Transcript
from ..shared.types import Sampling
from .actions import parse_action
from .prompts import AGENT_SYSTEM, wrap_observation

# The terminal-bench Harness builds a fresh DirectorAgent per trial. Loading a new 0.6B router
# each time blows up GPU memory under parallelism (N trials -> N backbones). Cache ONE router per
# (config, ckpt) and reuse it across trials; serialize its GPU forward with _GPU_LOCK since trials
# run in concurrent threads (a single torch model isn't safe under concurrent forwards).
_ROUTER_CACHE: dict = {}
_ROUTER_BUILD_LOCK = threading.Lock()
_GPU_LOCK = threading.Lock()
# DirectorAgent's spend lands in its own pool; the harness doesn't surface it. Each agent writes
# its accumulated $ cost here keyed by a per-run id so the caller (run_terminal) can read it back.
_COST_BY_RUN: dict = {}


def _shared_router(director_config: str | None, ckpt: str | None):
    key = (director_config, ckpt)
    with _ROUTER_BUILD_LOCK:
        if key not in _ROUTER_CACHE:
            from ..fugu.run import build_router, load_config
            _ROUTER_CACHE[key] = build_router(load_config(director_config), ckpt=ckpt)
        return _ROUTER_CACHE[key]


class DirectorAgent(BaseAgent):
    """Per-step-routed agent. Built by the harness with ``agent_kwargs``."""

    def __init__(
        self,
        director_config: str | None = None,
        ckpt: str | None = None,
        allowed: "list[str] | set[str] | None" = None,
        cost_key: str | None = None,        # caller reads back this run's $ cost via _COST_BY_RUN
        max_turns: int = 30,
        max_tokens: int = 32768,            # agentic: high generation length (Fugu setting)
        temperature: float = 0.2,
        reasoning_effort: str | None = "high",  # agentic: max reasoning effort
        command_timeout_sec: float = 300.0,  # per-command cap; slow builds/installs need >60s
        agent_budget_sec: float = 560.0,     # self-terminate before terminal-bench's global agent
        #                                      timeout (600s) so we never race its container teardown
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Imported lazily so importing this module doesn't require torch/config.
        from ..fugu.run import load_config
        from ..shared.providers import build_pool

        cfg = load_config(director_config)
        self._pool = build_pool(cfg.pool, cfg.workers)   # pool is CPU/small; per-instance is fine
        self._router = _shared_router(director_config, ckpt)  # ONE backbone, shared across trials
        self._allowed = set(allowed) if allowed else None   # force a solo worker for baselines/gen
        self._cost_key = cost_key
        self._cost = 0.0
        self._max_turns = max_turns
        self._sampling = Sampling(temperature=temperature, max_tokens=max_tokens,
                                  reasoning_effort=reasoning_effort)
        self._command_timeout = command_timeout_sec
        self._agent_budget = agent_budget_sec

    @staticmethod
    def name() -> str:
        return "director"

    def perform_task(
        self, instruction: str, session: TmuxSession, logging_dir: Path | None = None
    ) -> AgentResult:
        tx = Transcript()
        tx.add("system", AGENT_SYSTEM)
        tx.add("user", instruction)
        in_tok = out_tok = 0
        failure = FailureMode.NONE

        # ONE event loop for ALL turns. The pool's async HTTP client binds to the first loop it
        # runs on; the old per-turn asyncio.run() created a fresh loop each turn, so turn 2+ hit
        # "Event loop is closed" -> every multi-turn rollout failed as unknown_agent_error (no
        # terminal data). Reuse a single loop here, then close the client on it before teardown.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        t_start = time.monotonic()
        try:
            for _ in range(self._max_turns):
                if time.monotonic() - t_start > self._agent_budget:
                    # self-terminate before terminal-bench's global timeout tears down the container
                    # under us (which would 404 the next exec from this un-killable thread). The
                    # harness still grades final container state -> a real verdict, not a crash.
                    break
                with _GPU_LOCK:  # shared router across concurrent trials -> serialize the forward
                    worker_id = select_worker(self._router, tx.render(), allowed=self._allowed)
                comp = loop.run_until_complete(
                    self._pool.call(worker_id, tx.as_messages(), self._sampling))
                in_tok += comp.prompt_tokens
                out_tok += comp.completion_tokens
                self._cost += float(comp.cost_usd or 0.0)
                tx.add("assistant", comp.text)

                action = parse_action(comp.text)
                if action.submit:
                    break
                if action.command is None:
                    tx.add("user", wrap_observation("No bash block found. Emit one ```bash command."))
                    continue
                try:
                    session.send_keys([action.command, "Enter"], block=True,
                                      max_timeout_sec=self._command_timeout)
                    obs = session.get_incremental_output()
                except TimeoutError:
                    # A single long-running command must NOT fail the whole rollout (this was the
                    # unknown_agent_error flood -> 0 terminal cells). Report the timeout to the agent
                    # and continue; the task is still graded on final container state at the end.
                    obs = (session.get_incremental_output()
                           + f"\n[command exceeded {self._command_timeout:.0f}s and was left running]")
                tx.add("user", wrap_observation(obs))
        except Exception as e:  # noqa: BLE001 - report as agent failure, harness still grades state
            failure = FailureMode.UNKNOWN_AGENT_ERROR
            import traceback as _tb
            print(f"[DirectorAgent ERROR] {type(e).__name__}: {str(e)[:200]}\n"
                  f"{_tb.format_exc()}", flush=True)
        finally:
            try:  # close the async HTTP client on its own loop, then tear the loop down cleanly
                loop.run_until_complete(self._pool._provider._client.close())
            except Exception:  # noqa: BLE001
                pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            asyncio.set_event_loop(None)
            loop.close()
        if self._cost_key is not None:  # publish $ spend so run_terminal can bank a real cost
            _COST_BY_RUN[self._cost_key] = self._cost

        return AgentResult(
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            failure_mode=failure,
            timestamped_markers=[],
        )
