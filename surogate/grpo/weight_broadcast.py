"""Weight broadcast: saves updated weights to filesystem for vLLM to pick up."""

import shutil
from pathlib import Path

from surogate.utils.logger import get_logger

logger = get_logger()


class SurogateWeightBroadcast:
    """Broadcasts weights to the inference engine via shared filesystem.

    After each optimizer step, saves the LoRA adapter (or full model) to a
    step-specific directory and writes a STABLE marker file. The vLLM inference
    engine polls for STABLE files to hot-reload weights.

    Directory structure: {output_dir}/broadcasts/step_{step}/STABLE
    """

    def __init__(self, output_dir: str, adapter_only: bool = True, max_async_level: int = 1):
        # The orchestrator's scheduler polls {orch_output_dir}/broadcasts/step_{step}/STABLE
        # (via get_broadcast_dir() which returns output_dir / "broadcasts").
        # The orchestrator's output_dir defaults to "outputs/run_default", so we must
        # write broadcasts inside the run_* subdirectory to match.
        parent = Path(output_dir)
        run_dirs = sorted(parent.glob("run_*"))
        if run_dirs:
            run_dir = run_dirs[0]
        else:
            # Fallback: create run_default if no run dir exists yet
            run_dir = parent / "run_default"
        self.broadcast_dir = run_dir / "broadcasts"
        self.broadcast_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_only = adapter_only
        self.max_async_level = max_async_level
        logger.info(f"Weight broadcast dir: {self.broadcast_dir}")

    def broadcast(self, trainer, step: int) -> None:
        """Save weights and notify the inference engine."""
        save_dir = self.broadcast_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.adapter_only:
            trainer.export_adapter(str(save_dir))
        else:
            trainer.export_model(str(save_dir))

        # Write STABLE marker to signal readiness
        (save_dir / "STABLE").touch()

        # Reset ready_to_update so the packer's TrainingBatchReceiver will
        # accept the next batch.  In prime-rl's normal flow this is done by
        # FileSystemWeightBroadcast.broadcast_weights().
        from surogate.grpo.runs import get_multi_run_manager
        mrm = get_multi_run_manager()
        for idx in mrm.used_idxs:
            mrm.ready_to_update[idx] = False

    def cleanup(self, current_step: int) -> None:
        """Remove old broadcast directories, keeping only recent ones."""
        if not self.broadcast_dir.exists():
            return

        # Sort numerically by step number (step_10 > step_9, not lexicographic)
        def _step_num(p: Path) -> int:
            try:
                return int(p.name.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        step_dirs = sorted(self.broadcast_dir.iterdir(), key=_step_num)
        # Keep max_async_level + 1 most recent directories
        keep = self.max_async_level + 1
        to_remove = step_dirs[:-keep] if len(step_dirs) > keep else []

        for d in to_remove:
            if d.is_dir():
                shutil.rmtree(d)
                logger.debug(f"Cleaned up broadcast dir: {d}")
