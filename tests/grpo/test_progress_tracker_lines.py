"""Plain-log progress lines from ProgressTracker (nohup visibility, 2026-07-29)."""

from surogate.grpo.utils import logger as logger_mod
from surogate.grpo.utils.logger import ProgressTracker


class _Capture:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    def bind(self, **kw):
        return self


def test_plain_mode_emits_rate_and_eta(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(logger_mod, "get_logger", lambda: cap)
    t = ProgressTracker(total=100, desc="Generating rollouts (train)",
                        json_logging=False, step=10)
    t._pbar = None  # force the plain path without a TTY bar
    t.update = ProgressTracker.update.__get__(t)  # rebind after pbar strip

    # emulate the tqdm branch calling _maybe_log_line
    for _ in range(25):
        t.current += 1
        t._maybe_log_line()
    assert any("[progress] step 10" in ln and "eta" in ln for ln in cap.lines)
    first = cap.lines[0]
    assert "25/100" in cap.lines[-1] or "(2" in cap.lines[-1]
    assert "/min" in first


def test_percent_cadence_not_every_update(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(logger_mod, "get_logger", lambda: cap)
    t = ProgressTracker(total=1000, desc="d", json_logging=False)
    t._pbar = None
    for _ in range(99):   # 9.9% — below the 10% cadence
        t.current += 1
        t._maybe_log_line()
    assert len(cap.lines) <= 1  # at most the first crossing at 0%+


def test_stall_tick_emits_and_rate_limits(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(logger_mod, "get_logger", lambda: cap)
    t = ProgressTracker(total=256, desc="Generating rollouts (train)",
                        json_logging=False, step=11)
    t._pbar = None
    t.current = 128
    t._last_line_time = 0.0
    t.stall_tick(inflight=64, scoring=3)
    assert len(cap.lines) == 1
    assert "NO completions" in cap.lines[0] and "128/256" in cap.lines[0]
    t.stall_tick(inflight=64, scoring=3)   # within the same window
    assert len(cap.lines) == 1             # rate-limited
