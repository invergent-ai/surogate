"""Shared in-process state for colocate mode (surogate + vLLM in same process).

The trainer deposits weight tensors here, and the vLLM worker picks them up.
This avoids disk I/O entirely — weights go directly from surogate's GPU memory
to vLLM's model via DLPack zero-copy views.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

_lock = threading.Lock()
_pending_weights: Optional[dict[str, torch.Tensor]] = None


def set_pending_weights(weights: dict[str, torch.Tensor]) -> None:
    """Deposit weight tensors for vLLM to pick up."""
    global _pending_weights
    with _lock:
        _pending_weights = weights


def get_pending_weights() -> Optional[dict[str, torch.Tensor]]:
    """Retrieve and clear pending weight tensors."""
    global _pending_weights
    with _lock:
        w = _pending_weights
        _pending_weights = None
        return w
