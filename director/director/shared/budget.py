"""USD spend tracking with a hard cap.

The cap is checked before each *uncached* call; cached calls are free and never
counted. Raising before the call (rather than after) keeps spend bounded.
"""

from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, cap_usd: float | None = None):
        self._cap = cap_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return self._spent

    @property
    def cap_usd(self) -> float | None:
        return self._cap

    def check(self) -> None:
        """Raise if we are already at/over the cap (call before an uncached request)."""
        if self._cap is not None and self._spent >= self._cap:
            raise BudgetExceeded(
                f"budget cap ${self._cap:.4f} reached (spent ${self._spent:.4f})"
            )

    def add(self, usd: float) -> None:
        with self._lock:
            self._spent += usd
