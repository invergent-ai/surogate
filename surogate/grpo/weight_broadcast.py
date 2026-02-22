"""Weight broadcast: saves updated weights to filesystem for vLLM to pick up.

Also provides ColocateWeightBroadcast for zero-copy GPU weight sharing when
surogate and vLLM run in the same process.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import torch

from surogate.grpo.config import NoiseSchedulerConfig
from surogate.utils.logger import get_logger

logger = get_logger()


class SurogateWeightBroadcast:
    """Broadcasts weights to the inference engine via shared filesystem.

    After each optimizer step, saves the LoRA adapter (or full model) to a
    step-specific directory and writes a STABLE marker file. The vLLM inference
    engine polls for STABLE files to hot-reload weights.

    Optionally injects QeRL Adaptive Quantization Noise (AQN) into RMSNorm
    weights before writing the STABLE marker.

    Directory structure: {output_dir}/broadcasts/step_{step}/STABLE
    """

    def __init__(
        self,
        output_dir: str,
        adapter_only: bool = True,
        max_async_level: int = 1,
        noise_config: Optional[NoiseSchedulerConfig] = None,
        base_model_dir: Optional[str] = None,
        max_steps: int = 0,
    ):
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
        self.noise_config = noise_config
        self.base_model_dir = Path(base_model_dir) if base_model_dir else None
        self.max_steps = max_steps
        logger.info(f"Weight broadcast dir: {self.broadcast_dir}")
        if noise_config and noise_config.enabled:
            logger.info(
                f"QeRL noise scheduler enabled: sigma_start={noise_config.sigma_start}, "
                f"sigma_end={noise_config.sigma_end}, num_stages={noise_config.num_stages}"
            )

    def broadcast(self, trainer, step: int) -> None:
        """Save weights and notify the inference engine."""
        save_dir = self.broadcast_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.adapter_only:
            trainer.export_adapter(str(save_dir))
        else:
            trainer.export_model(str(save_dir))

        # QeRL: inject noise into RMSNorm weights before signaling readiness
        self._maybe_inject_noise(save_dir, step)

        # Write STABLE marker to signal readiness
        (save_dir / "STABLE").touch()

        # Reset ready_to_update so the packer's TrainingBatchReceiver will
        # accept the next batch.  In prime-rl's normal flow this is done by
        # FileSystemWeightBroadcast.broadcast_weights().
        from surogate.grpo.runs import get_multi_run_manager
        mrm = get_multi_run_manager()
        for idx in mrm.used_idxs:
            mrm.ready_to_update[idx] = False

    def _maybe_inject_noise(self, save_dir: Path, step: int) -> None:
        """Inject QeRL AQN noise into exported weights if enabled."""
        if not self.noise_config or not self.noise_config.enabled:
            return
        if self.max_steps <= 0:
            return

        from surogate.grpo.noise_scheduler import (
            compute_sigma,
            inject_noise_adapter,
            inject_noise_model,
        )

        sigma = compute_sigma(step, self.max_steps, self.noise_config)
        if sigma <= 0.0:
            return

        if self.adapter_only:
            if not self.base_model_dir:
                logger.warning("QeRL noise: base_model_dir not set, skipping noise injection")
                return
            n = inject_noise_adapter(save_dir, self.base_model_dir, sigma)
        else:
            n = inject_noise_model(save_dir, sigma)

        if n > 0:
            logger.info(f"QeRL noise: injected sigma={sigma:.6f} into {n} norm tensors (step {step})")

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


class ColocateWeightBroadcast:
    """Zero-copy weight broadcast for colocate mode (surogate + vLLM in same process).

    Instead of exporting weights to disk and having vLLM reload them, this class:
    1. Gets LoRA weight tensors directly from surogate's GPU memory via DLPack
    2. Deposits them in shared in-process state
    3. Writes a STABLE marker so the orchestrator's scheduler continues
    4. The orchestrator calls /update_weights on vLLM, which picks up from shared state

    QeRL noise is applied in-memory to the GPU tensors (no disk I/O needed).
    """

    def __init__(
        self,
        output_dir: str,
        max_async_level: int = 1,
        noise_config: Optional[NoiseSchedulerConfig] = None,
        base_model_dir: Optional[str] = None,
        max_steps: int = 0,
    ):
        parent = Path(output_dir)
        run_dirs = sorted(parent.glob("run_*"))
        if run_dirs:
            run_dir = run_dirs[0]
        else:
            run_dir = parent / "run_default"
        self.broadcast_dir = run_dir / "broadcasts"
        self.broadcast_dir.mkdir(parents=True, exist_ok=True)
        self.max_async_level = max_async_level
        self.noise_config = noise_config
        self.base_model_dir = Path(base_model_dir) if base_model_dir else None
        self.max_steps = max_steps
        logger.info(f"Colocate weight broadcast dir: {self.broadcast_dir}")
        if noise_config and noise_config.enabled:
            logger.info(
                f"QeRL noise scheduler enabled (colocate): sigma_start={noise_config.sigma_start}, "
                f"sigma_end={noise_config.sigma_end}, num_stages={noise_config.num_stages}"
            )

    def broadcast(self, trainer, step: int) -> None:
        """Export LoRA adapter to disk for vLLM to pick up.

        In colocate mode, base weights are zero-copy (shared via CUDA IPC).
        LoRA adapter updates are small (~10MB) and go through disk using
        the standard vLLM load_lora_adapter path. The vLLM engine runs in
        a child process, so in-process shared state is not an option.
        """
        save_dir = self.broadcast_dir / f"step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Export PEFT adapter files (adapter_config.json + safetensors)
        trainer.export_adapter(str(save_dir))

        # QeRL: inject noise into exported adapter files
        self._maybe_inject_noise_on_disk(save_dir, step)

        # Write STABLE marker so orchestrator's scheduler continues
        (save_dir / "STABLE").touch()

        # Reset ready_to_update
        from surogate.grpo.runs import get_multi_run_manager
        mrm = get_multi_run_manager()
        for idx in mrm.used_idxs:
            mrm.ready_to_update[idx] = False

    def _maybe_inject_noise_on_disk(self, save_dir: Path, step: int) -> None:
        """Inject QeRL AQN noise into exported adapter files (same as SurogateWeightBroadcast)."""
        if not self.noise_config or not self.noise_config.enabled:
            return
        if self.max_steps <= 0:
            return

        from surogate.grpo.noise_scheduler import compute_sigma, inject_noise_adapter

        sigma = compute_sigma(step, self.max_steps, self.noise_config)
        if sigma <= 0.0:
            return

        if not self.base_model_dir:
            logger.warning("QeRL noise: base_model_dir not set, skipping noise injection")
            return

        n = inject_noise_adapter(save_dir, self.base_model_dir, sigma)
        if n > 0:
            logger.info(f"QeRL noise (colocate): injected sigma={sigma:.6f} into {n} norm tensors (step {step})")

    def cleanup(self, current_step: int) -> None:
        """Remove old broadcast directories, keeping only recent ones."""
        if not self.broadcast_dir.exists():
            return

        def _step_num(p: Path) -> int:
            try:
                return int(p.name.split("_", 1)[1])
            except (IndexError, ValueError):
                return -1
        step_dirs = sorted(self.broadcast_dir.iterdir(), key=_step_num)
        keep = self.max_async_level + 1
        to_remove = step_dirs[:-keep] if len(step_dirs) > keep else []

        for d in to_remove:
            if d.is_dir():
                shutil.rmtree(d)
                logger.debug(f"Cleaned up broadcast dir: {d}")
