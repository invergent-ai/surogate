"""Periodic merged-weight serving reload for the r5 GRPO loop (fix #2).

vLLM LoRA is a silent no-op on the 27B hybrid stack (adapter audit
2026-07-26), so the behavior policy on :8011 can only be refreshed by
serving MERGED weights. This controller is the campaign's `on_checkpoint`
hook: every `checkpoint_every` sent batches it

    1. finds the trainer's newest adapter checkpoint,
    2. merges it onto the base (surogate.utils.adapter_merge — the same
       path the trainer's final save uses),
    3. restarts the vLLM on OUR port (matched by `--port {port}`; other
       servers on the host are never touched),
    4. blocks until /v1/models answers, and
    5. prunes old merged snapshots (each is ~55 GB; `keep_merged` bounds
       disk).

The campaign blocks inside the hook, so collection never races a reload,
and a reload failure raises — halting paid spend rather than silently
collecting from a stale (or dead) policy. Loss-side staleness between
reloads is importance-corrected (loss.py ratio + mismatch_kl monitoring).

All process/network effects are injected for offline testing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def _default_merge(base_model_path: str, adapter_path: str,
                   output_path: str) -> None:
    from surogate.utils.adapter_merge import merge_adapter

    merge_adapter(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        output_path=output_path,
        max_shard_size="5GB",
        cpu_offload=True,
    )


def _copy_serving_files(base: Path, merged: Path) -> None:
    """Tokenizer/config files the merge step doesn't emit."""
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "special_tokens_map.json",
                 "generation_config.json", "chat_template.jinja",
                 "configuration.json"):
        src = base / name
        if src.exists() and not (merged / name).exists():
            shutil.copy2(src, merged / name)


@dataclass
class MergedReloadController:
    """Reloads the conductor server from the trainer's latest checkpoint."""

    base_model_dir: Path                 # the LoRA's base (r4 merged bf16)
    checkpoints_dir: Path                # trainer output/step_XXXXXXXX dirs
    merged_root: Path                    # merged snapshots live here
    python_bin: str = ".venv/bin/python"
    port: int = 8011
    served_model_name: str = "fugu-27b-r4-merged"
    # The orch requests by model PATH (its model.name doubles as tokenizer
    # path); serve that path as a second alias so both ids route.
    served_model_alias: str | None = \
        "/home/densemax/work/flavius/surogate/output/fugu_27b_r4_merged_bf16"
    # 8-GPU plan (2026-07-27): serving is TP2 on GPUs 6-7 so the trainer
    # keeps 0-5. 27GB bf16 weights per 32GB card is tight: high memory
    # utilization + eager mode (no CUDA-graph reservation) make it fit.
    tensor_parallel: int = 2
    gpu_ids: str | None = "6,7"          # CUDA_VISIBLE_DEVICES for serving
    gpu_memory_utilization: float = 0.95
    enforce_eager: bool = True
    # 32768 (user 2026-07-28: "serve the conductor with unlimited context"):
    # removes the 400-crash mode on long task payloads. The conductor's
    # TRAINED view stays 12k chars (sequence_len 6144 is the hardware fit);
    # the window is headroom, not a training-regime change. If KV pressure
    # bites on the 32GB cards, add kv_cache_dtype="fp8" (user-approved).
    max_model_len: int = 32768
    keep_merged: int = 2
    health_timeout_s: float = 1800.0
    health_poll_s: float = 5.0
    server_log: Path | None = None
    # injected effects (tests replace these)
    merge_fn: Callable[[str, str, str], None] = _default_merge
    run_cmd: Callable[..., object] = subprocess.run
    spawn: Callable[[list[str]], object] | None = None
    health_check: Callable[[], bool] | None = None
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = print

    reloads: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------- pieces

    def latest_checkpoint(self) -> Path | None:
        # Gate on checkpoint.json, which the trainer writes LAST after the
        # NCCL barrier — adapter_model.safetensors alone can be a mid-save
        # torn state (scoping review 2026-07-27, checkpoint.cpp:130-163).
        candidates = sorted(
            d for d in self.checkpoints_dir.glob("step_*")
            if (d / "adapter_model.safetensors").exists()
            and (d / "checkpoint.json").exists()
        )
        return candidates[-1] if candidates else None

    def _merged_dir_for(self, checkpoint: Path) -> Path:
        return self.merged_root / f"merged_{checkpoint.name}"

    def _healthy(self) -> bool:
        if self.health_check is not None:
            return self.health_check()
        try:
            with urllib.request.urlopen(
                f"http://localhost:{self.port}/v1/models", timeout=10
            ) as response:
                payload = json.load(response)
            served = {m.get("id") for m in payload.get("data", [])}
            wanted = {self.served_model_name}
            if self.served_model_alias:
                wanted.add(self.served_model_alias)
            return bool(wanted & served)
        except Exception:  # noqa: BLE001 — any failure means "not healthy"
            return False

    def _stop_server(self) -> None:
        # Match OUR port explicitly; the host runs other vLLMs (e.g. :8012)
        # that must never be touched.
        pattern = f"vllm.entrypoints.openai.api_server.*--port {self.port}"
        self.run_cmd(["pkill", "-f", pattern], capture_output=True)
        # WAIT until the old server is actually gone: a dying server keeps
        # answering /v1/models for seconds, so starting the new one and
        # health-checking immediately can pass against the OLD process and
        # resume collection on a stale policy (observed live 2026-07-27
        # during the TP4->TP2 switch).
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            probe = self.run_cmd(["pgrep", "-f", pattern],
                                 capture_output=True)
            if getattr(probe, "returncode", 1) != 0:
                return  # no matching process remains
            self.sleep(2.0)
        self.run_cmd(["pkill", "-9", "-f", pattern], capture_output=True)
        self.sleep(5.0)

    def _start_server(self, model_dir: Path) -> None:
        cmd = []
        if self.gpu_ids:
            # `env` prefix keeps the pinning visible in the command line
            # (auditable via ps) and works under plain Popen.
            cmd += ["env", f"CUDA_VISIBLE_DEVICES={self.gpu_ids}"]
        served = [self.served_model_name]
        if self.served_model_alias:
            served.append(self.served_model_alias)
        cmd += [
            self.python_bin, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_dir),
            "--served-model-name", *served,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.spawn is not None:
            self.spawn(cmd)
            return
        log_handle = (self.server_log.open("a")
                      if self.server_log else subprocess.DEVNULL)
        subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                         start_new_session=True)

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self._healthy():
                return
            self.sleep(self.health_poll_s)
        raise RuntimeError(
            f"served model {self.served_model_name!r} not healthy on "
            f":{self.port} within {self.health_timeout_s}s after reload — "
            f"halting the campaign (spend must not continue against a dead "
            f"or stale policy)")

    def _prune(self) -> None:
        merged = sorted(self.merged_root.glob("merged_step_*"))
        for old in merged[: max(0, len(merged) - self.keep_merged)]:
            shutil.rmtree(old, ignore_errors=True)
            self.log(f"[reload] pruned {old.name}")

    # ------------------------------------------------------------- hook

    def on_checkpoint(self, sent_batches: int) -> None:
        """Reload hook: merge newest checkpoint and re-serve it."""
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            self.log(f"[reload] at {sent_batches} sent batches: no trainer "
                     f"checkpoint yet — serving stays as-is")
            return
        if self.reloads and self.reloads[-1]["checkpoint"] == checkpoint.name:
            self.log(f"[reload] {checkpoint.name} already serving — skip "
                     f"(trainer lags the reload cadence)")
            return
        merged_dir = self._merged_dir_for(checkpoint)
        if not (merged_dir / "config.json").exists():
            self.log(f"[reload] merging {checkpoint.name} "
                     f"(~minutes, CPU-offloaded)")
            merged_dir.mkdir(parents=True, exist_ok=True)
            self.merge_fn(str(self.base_model_dir), str(checkpoint),
                          str(merged_dir))
            _copy_serving_files(self.base_model_dir, merged_dir)
        self.log(f"[reload] restarting :{self.port} on {merged_dir.name}")
        self._stop_server()
        self._start_server(merged_dir)
        self._wait_healthy()
        self._prune()
        self.reloads.append(
            {"sent_batches": sent_batches, "checkpoint": checkpoint.name})
        self.log(f"[reload] serving {checkpoint.name} "
                 f"(reload #{len(self.reloads)})")
