"""Atomic checkpoints joining one native policy update with rollout progress."""

import hashlib
import json
import random
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import torch

from surogate.core.config.grpo_orch_config import GRPOCheckpointConfig
from surogate.grpo.orchestrator.ckpt import CheckpointManager
from surogate.utils.logger import get_logger

logger = get_logger()


class NativeCheckpointCoordinator:
    def __init__(self, train, orch):
        self.train, self.orch = train, orch
        self.root = Path(train.checkpoint_dir or train.output_dir) / "native_colocate"
        self.run = Path(orch.output_dir)
        self.interval = int(train.save_steps or 0)
        self.max_steps = train.max_steps
        self.identifier = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.manager = CheckpointManager(self.run, GRPOCheckpointConfig({}))
        model_config = Path(train.model_dir) / "config.json"
        self.identity = dict(model=train.model, config_sha256=hashlib.sha256(model_config.read_bytes()).hexdigest(),
                             rank=train.lora_rank, alpha=train.lora_alpha, dtype=train.lora_dtype,
                             targets=sorted(train.lora_target_modules or []), sequence_len=train.sequence_len,
                             optimizer=getattr(train, "optimizer", "adamw_8bit"),
                             use_rs_lora=bool(getattr(train, "use_rs_lora", False)))
        self.resume_step = 0
        self.resume_path = None
        requested = orch.ckpt.resume_step if orch.ckpt else None
        resume = bool(train.resume_from_checkpoint)
        if requested is not None and not resume:
            raise ValueError("native checkpoint resume requires resume_from_checkpoint: true")
        if requested is not None and requested < -1:
            raise ValueError("resume_step must be -1 or a completed checkpoint step")
        candidates = sorted((p for p in self.root.glob("step_*") if p.name[5:].isdigit()),
                            key=lambda p: int(p.name[5:]), reverse=True)
        if resume:
            for path in candidates:
                step = int(path.name[5:])
                if requested is not None and requested >= 0 and step != requested:
                    continue
                manifest = self._read_complete(path)
                if manifest is None:
                    continue
                if manifest["identity"] != self.identity:
                    raise ValueError("native checkpoint model, adapter, optimizer, or sequence configuration does not match this run")
                self.resume_step, self.resume_path = step, path
                break
        if self.resume_path is None:
            if requested is not None and requested >= 0:
                raise FileNotFoundError(f"No complete native checkpoint at step {requested}")
            if candidates or (Path(train.output_dir) / "rollouts").exists() or any(
                (self.run / name).exists() for name in ("broadcasts", "rollouts", "checkpoints", "control", "live_spool")
            ):
                raise ValueError("No matching complete native checkpoint; use a fresh output directory for a new run")
        if self.resume_path and orch.ckpt and (orch.ckpt.skip_progress or orch.ckpt.skip_buffer):
            raise ValueError("native resume restores progress and buffer together; skip_progress/skip_buffer are unsupported")

    def _read_complete(self, path):
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            step = int(path.name[5:])
            if manifest["version"] != 1 or manifest["next_step"] != step or manifest["trainer_step"] != step - 1:
                return None
            for name, size in manifest["files"].items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or (path / relative).stat().st_size != size:
                    return None
            required = {f"trainer/step_{step - 1:08d}/checkpoint.json", "orchestrator/progress.pt", "runtime.pt"}
            if not required.issubset(manifest["files"]):
                return None
            return manifest
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @property
    def trainer_resume(self):
        return (str(self.resume_path / "trainer"), self.resume_step - 1) if self.resume_path else ("", -1)

    def prepare_resume(self):
        if self.resume_path is None:
            return
        # Preserve later/unconsumed work without letting stale readiness markers
        # or transport files race regenerated batches from the restored policy.
        paths = [self.run / name for name in ("broadcasts", "rollouts", "control", "live_spool")]
        paths.append(Path(self.train.output_dir) / "rollouts")
        paths.extend(p for p in self.root.glob("step_*") if p.name[5:].isdigit() and int(p.name[5:]) > self.resume_step)
        for path in paths:
            if path.exists():
                backup = path.parent / "recovery" / self.identifier / path.name
                backup.parent.mkdir(parents=True, exist_ok=True)
                path.rename(backup)
        if self.orch.ckpt is None:
            self.orch.ckpt = GRPOCheckpointConfig({})
        self.orch.ckpt.resume_step = self.resume_step
        logger.info(f"Resuming native GRPO after {self.resume_step} completed updates")

    def should_save(self, next_step):
        return self.interval > 0 and next_step > 0 and (next_step % self.interval == 0 or next_step == self.max_steps)

    def _pending(self, step):
        return self.root / f".pending_{step}_{self.identifier}"

    def save_training(self, trainer, next_step):
        if not self.should_save(next_step):
            return
        path = self._pending(next_step)
        path.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path / "trainer"), next_step - 1)
        (path / "trainer.ready").touch()
        self._publish(next_step)

    def save_orchestrator(self, progress, buffer, *, next_group_id, depth_state):
        if not self.should_save(progress.step):
            return
        path = self._pending(progress.step)
        target = path / "orchestrator"
        target.mkdir(parents=True, exist_ok=True)
        runtime = dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                       next_group_id=next_group_id, depth_state=depth_state)
        self.manager._save_to_path(target, progress, buffer)
        torch.save(runtime, path / "runtime.pt")
        (path / "orchestrator.ready").touch()
        self._publish(progress.step)

    def _publish(self, next_step):
        with self.lock:
            path = self._pending(next_step)
            if not all((path / name).exists() for name in ("trainer.ready", "orchestrator.ready")):
                return
            metadata = json.loads((path / "trainer" / f"step_{next_step - 1:08d}" / "checkpoint.json").read_text())
            if metadata["run"]["step"] != next_step - 1:
                raise ValueError("native checkpoint training step does not match rollout progress")
            files = {str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()}
            manifest = dict(version=1, next_step=next_step, trainer_step=next_step - 1, identity=self.identity, files=files)
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
            path.rename(self.root / f"step_{next_step}")
            logger.info(f"Saved complete native checkpoint at step {next_step}")
            self._cleanup(next_step)

    def _cleanup(self, latest):
        config = self.orch.ckpt
        if config is None or config.keep_last is None:
            return
        steps = sorted(int(p.name[5:]) for p in self.root.glob("step_*") if p.name[5:].isdigit())
        keep = set(steps[-config.keep_last:]) if config.keep_last > 0 else set()
        keep.add(latest)
        for step in steps:
            if step not in keep and not (config.keep_interval and step % config.keep_interval == 0):
                shutil.rmtree(self.root / f"step_{step}")

    def load_orchestrator(self, progress, buffer):
        if self.resume_path is None:
            return {}
        self.manager._load_from_path(self.resume_path / "orchestrator", progress, buffer)
        if progress.step != self.resume_step:
            raise ValueError("native checkpoint rollout progress does not match policy version")
        runtime = torch.load(self.resume_path / "runtime.pt", weights_only=False)
        random.setstate(runtime["python_rng"])
        np.random.set_state(runtime["numpy_rng"])
        torch.set_rng_state(runtime["torch_rng"])
        return runtime
