from pathlib import Path

from transformers.tokenization_utils import PreTrainedTokenizer

from surogate.core.config.grpo_orch_config import GRPOReportingConfig
from surogate.grpo.utils.monitor.base import Monitor, NoOpMonitor
from surogate.grpo.utils.monitor.multi import MultiMonitor
from surogate.grpo.utils.monitor.wandb import WandbMonitor

__all__ = [
    "Monitor",
    "WandbMonitor",
    "MultiMonitor",
    "NoOpMonitor",
    "setup_monitor",
    "get_monitor",
]

_MONITOR: Monitor | None = None


def get_monitor() -> Monitor:
    """Returns the global monitor."""
    global _MONITOR
    if _MONITOR is None:
        raise RuntimeError("Monitor not initialized. Please call `setup_monitor` first.")
    return _MONITOR


def setup_monitor(
    wandb_config: GRPOReportingConfig | None = None,
    output_dir: Path | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    run_config = None,
) -> Monitor:
    """
    Sets up monitors to log metrics.
    """
    global _MONITOR
    if _MONITOR is not None:
        raise RuntimeError("Monitor already initialized. Please call `setup_monitor` only once.")

    monitors: list[Monitor] = []

    if wandb_config is not None:
        monitors.append(
            WandbMonitor(
                config=wandb_config,
                output_dir=output_dir,
                tokenizer=tokenizer,
                run_config=run_config,
            )
        )

    if len(monitors) == 0:
        _MONITOR = NoOpMonitor()
    elif len(monitors) == 1:
        _MONITOR = monitors[0]
    else:
        _MONITOR = MultiMonitor(monitors)

    return _MONITOR
