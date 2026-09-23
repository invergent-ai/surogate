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


def resolve_broadcast_dir(train_output_dir, broadcast_dir=None) -> Path:
    """Where the trainer publishes weights for the orchestrator to collect.

    `broadcast_dir` is the answer when the caller knows it -- split mode holds
    both configs and passes `get_broadcast_dir(orch_config.output_dir)`. It has
    to match exactly, or the orchestrator waits for a checkpoint being written
    somewhere else, indefinitely, while its log says training is progressing
    normally.

    Without it there is nothing to go on but the train output_dir, which does
    not contain the answer, so this guesses and says so. The guess is only ever
    right for an orchestrator whose output_dir is `run_default`: it lands there
    by the fallback when nothing matches, and by `sorted(...)[0]` when several
    do, because "run_default" sorts before most names -- "run_tools" among
    them, which is why the shipped tool example could never train.
    """
    if broadcast_dir is not None:
        return Path(broadcast_dir)
    parent = Path(train_output_dir)
    run_dirs = sorted(parent.glob("run_*"))
    run_dir = run_dirs[0] if run_dirs else parent / "run_default"
    guessed = run_dir / "broadcasts"
    logger.warning(
        f"No broadcast_dir given; guessing {guessed}. If the orchestrator's output_dir is "
        f"not {run_dir}, it will wait there for weights that are never written. Pass "
        f"broadcast_dir=get_broadcast_dir(orch_config.output_dir)."
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


def sync_wait_for_path(path: Path, interval: int = 1, log_interval: int = 10) -> None:
    wait_time = 0
    logger.debug(f"Waiting for path `{path}`")
    while True:
        if path.exists():
            logger.debug(f"Found path `{path}`")
            break
        if wait_time % log_interval == 0 and wait_time > 0:  # Every log_interval seconds
            logger.debug(f"Waiting for path `{path}` for {wait_time} seconds")
        time.sleep(interval)
        wait_time += interval


async def wait_for_path(
    path: Path, interval: int = 1, log_interval: int = 10, timeout: int | None = 1800
) -> None:
    """Wait for `path` to appear, giving up after `timeout` seconds.

    Unbounded by default until 2026-09-23. The orchestrator waits here for a
    checkpoint the trainer publishes, and if the two disagree about where that
    is, the wait never ends: the run holds its GPUs indefinitely while the log
    says "Training is progressing normally". That disagreement was real -- the
    trainer used to guess the directory -- and a silent forever-wait is what
    made it take a day to find. `None` restores the old behaviour.
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
