import asyncio
import time
from pathlib import Path

from surogate.utils.logger import get_logger

logger = get_logger()


def get_log_dir(output_dir: Path) -> Path:
    return output_dir / "logs"


def get_ckpt_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints"


def get_weights_dir(output_dir: Path) -> Path:
    return output_dir / "weights"


def get_rollout_dir(output_dir: Path) -> Path:
    return output_dir / "rollouts"


def get_eval_dir(output_dir: Path) -> Path:
    return output_dir / "evals"


def get_broadcast_dir(output_dir: Path) -> Path:
    return output_dir / "broadcasts"


def guess_broadcast_dir(train_output_dir) -> Path:
    """Guess where to publish weights when no orchestrator directory was given.

    Only for callers with no orchestrator config in scope. Anyone holding one
    should pass `get_broadcast_dir(orch_config.output_dir)` instead: this can
    only ever land on `run_default`, and the orchestrator polls wherever its
    own `output_dir` says.
    """
    parent = Path(train_output_dir)
    run_dirs = sorted(parent.glob("run_*"))
    run_dir = run_dirs[0] if run_dirs else parent / "run_default"
    guessed = run_dir / "broadcasts"
    logger.warning_once(
        f"No broadcast directory given; publishing to {guessed}. Nothing reads this unless an "
        f"orchestrator is polling exactly there. If you are in split mode, weights are going "
        f"somewhere it is not looking."
        # No hash_id: LoggerWrapper.warning_once reads it from kwargs but does
        # not remove it, so it reaches the stdlib logger and raises. The
        # message is the key by default, which is what we want anyway -- one
        # warning per distinct directory.
    )
    return guessed


def get_step_path(path: Path, step: int) -> Path:
    return path / f"step_{step}"


def get_all_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets all checkpoint steps from the checkpoint directory, sorted in ascending order."""
    step_dirs = list(ckpt_dir.glob("step_*"))
    return sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])


def get_stable_ckpt_steps(ckpt_dir: Path) -> list[int]:
    """Gets checkpoint steps that have STABLE file, sorted in ascending order."""
    steps = get_all_ckpt_steps(ckpt_dir)
    return [s for s in steps if (ckpt_dir / f"step_{s}" / "STABLE").exists()]


def resolve_latest_ckpt_step(ckpt_dir: Path) -> int | None:
    """Gets the latest checkpoint step from the checkpoint directory. Returns None if no checkpoints are found."""
    steps = get_all_ckpt_steps(ckpt_dir)
    if len(steps) == 0:
        logger.warning(f"No checkpoints found in {ckpt_dir}. Starting from scratch.")
        return None
    latest_step = steps[-1]
    logger.info(f"Found latest checkpoint in {ckpt_dir}: {latest_step}")
    return latest_step


def sync_wait_for_path(
    path: Path, interval: int = 1, log_interval: int = 10, timeout: int | None = 1800
) -> None:
    """Blocking twin of `wait_for_path`, bounded for the same reason.

    Its caller is the trainer waiting for a rollout micro-batch the
    orchestrator writes (`transport/filesystem.py`). That is the same pipe as
    the weight broadcast, in the other direction, with the same way to fail: if
    the two disagree about the directory, or the orchestrator dies, an
    unbounded wait blocks the trainer forever holding its GPUs.
    """
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if timeout is not None and wait_time >= timeout:
            raise TimeoutError(
                f"waited {wait_time}s for `{path}` and it never appeared. The process that "
                f"writes it has stopped, or is writing somewhere else."
            )
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


async def wait_for_path(
    path: Path, interval: int = 1, log_interval: int = 10, timeout: int | None = 1800
) -> None:
    """Wait for `path` to appear, giving up after `timeout` seconds.

    Unbounded until 2026-09-23, which is how a directory disagreement between
    trainer and orchestrator became a run that held its GPUs forever. `None`
    restores that.
    """
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if timeout is not None and wait_time >= timeout:
            raise TimeoutError(
                f"waited {wait_time}s for `{path}` and it never appeared. If this is a GRPO "
                f"weight broadcast, the trainer is writing somewhere else: the orchestrator "
                f"polls get_broadcast_dir(orch_config.output_dir) and the trainer must be "
                f"given the same path."
            )
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        await asyncio.sleep(interval)
        wait_time += interval
