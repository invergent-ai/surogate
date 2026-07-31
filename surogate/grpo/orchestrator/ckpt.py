import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from surogate.core.config.grpo_orch_config import GRPOCheckpointConfig
from surogate.grpo.orchestrator.buffer import Buffer
from surogate.grpo.utils.logger import get_logger
from surogate.grpo.utils.pathing import get_all_ckpt_steps
from surogate.grpo.utils.utils import get_ckpt_dir, get_step_path


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


logger = get_logger()


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, output_dir: Path, config: GRPOCheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def _save_to_path(
        self,
        ckpt_path: Path,
        progress: Progress,
        buffer: Buffer,
    ):
        logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.perf_counter()

        # Save progress
        with open(ckpt_path / "progress.pt", "wb") as f:
            torch.save({"progress": progress}, f)

        # Save buffer
        buffer.save(ckpt_path / "buffer")

        logger.debug(f"Orchestrator checkpoint saved in {time.perf_counter() - start_time:.2f} seconds")

    def _load_from_path(self, ckpt_path: Path, progress: Progress, buffer: Buffer) -> None:
        """Loads a checkpoint from a given path in-place."""
        logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.perf_counter()

        # Load progress
        if self.config.skip_progress:
            logger.info("Skipping progress loading from checkpoint")
        else:
            with open(ckpt_path / "progress.pt", "rb") as f:
                state = torch.load(f, weights_only=False)

            # Set progress in-place
            for key, value in asdict(state["progress"]).items():
                setattr(progress, key, value)

        # Load buffer
        if self.config.skip_buffer:
            logger.info("Skipping buffer loading from checkpoint")
        else:
            buffer.load(ckpt_path / "buffer")

        logger.debug(f"Orchestrator checkpoint loaded in {time.perf_counter() - start_time:.2f} seconds")

    def load(self, progress: Progress, buffer: Buffer, step: int) -> None:
        """Loads a checkpoint from a given path."""
        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, progress, buffer)

    def save(
        self,
        progress: Progress,
        buffer: Buffer,
        step: int,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        self._save_to_path(ckpt_path, progress, buffer)
        self._cleanup_old_ckpts()

    def _cleanup_old_ckpts(self) -> None:
        """Enforces keep_last/keep_interval: removes old step_N checkpoint dirs, keeping the
        newest keep_last and any step that is a keep_interval multiple. No-op when keep_last
        is unset. Only step_N directories are touched (other files in the dir are left alone)."""
        keep_last = self.config.keep_last
        if keep_last is None:
            return
        steps = get_all_ckpt_steps(self.ckpt_dir)
        protected = set(steps[-keep_last:]) if keep_last > 0 else set()
        keep_interval = self.config.keep_interval
        for s in steps:
            if s in protected or (keep_interval and s % keep_interval == 0):
                continue
            step_dir = get_step_path(self.ckpt_dir, s)
            if not (step_dir.is_dir() and step_dir.name.startswith("step_")):
                continue
            try:
                shutil.rmtree(step_dir)
            except Exception as e:
                logger.warning(f"Failed to clean old orchestrator checkpoint {step_dir}: {e}")


def setup_ckpt_manager(output_dir: Path, config: GRPOCheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
