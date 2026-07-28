"""Merged-reload sidecar for the orch-driven r5 campaign.

The stage2 loop refreshed the served policy via adapter broadcasts, which
the 27B hybrid cannot consume (vLLM LoRA no-op). This sidecar replaces the
consumer side: it watches the TRAINER's checkpoint dir (checkpoint_dir ==
output_dir; a checkpoint is complete only once `checkpoint.json` exists —
written last, after the NCCL barrier) and drives
`MergedReloadController.on_checkpoint` for every NEW complete checkpoint.

The trainer's own broadcast stays ON — the orch's pacing blocks on the
STABLE markers broadcasts write; with `model.lora_adapter` unset the orch's
consumer path is a 404-tolerant no-op, so broadcasts are pacing-only.

Downtime tolerance during a reload: the merge runs BEFORE the server stops,
so the outage is restart+load only; the orch's scheduler re-schedules
rollouts that error against :8011 within their group, and groups tolerate
`max_off_policy_steps` policy updates. A reload failure raises out of the
controller (unhealthy server ≠ silent stale collection) and the sidecar
exits nonzero — treat that as a campaign halt.

Run:  python -m ultra.serving_reload_sidecar [--poll 30] [--once]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

from ultra.serving_reload import MergedReloadController

ROOT = Path(__file__).resolve().parents[2]


def watch(
    controller: MergedReloadController,
    poll_s: float = 30.0,
    once: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> int:
    """Reload on every new complete checkpoint; returns reload count.

    `once` performs a single poll cycle (dry-test / unit use). The
    controller itself skips when the newest checkpoint is already serving,
    so the loop is safely re-entrant and resumable.
    """
    reloads_done = 0
    log(f"[sidecar] watching {controller.checkpoints_dir} "
        f"(poll {poll_s:.0f}s, keep_merged {controller.keep_merged})")
    while True:
        before = len(controller.reloads)
        controller.on_checkpoint(reloads_done)
        if len(controller.reloads) > before:
            reloads_done += 1
            log(f"[sidecar] reload #{reloads_done}: "
                f"{controller.reloads[-1]['checkpoint']}")
        if once:
            return reloads_done
        sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-output",
                    default=str(ROOT / "output/fugu_ultra_r5"))
    ap.add_argument("--merged-root",
                    default=str(ROOT / "output/fugu_ultra_r5/merged"))
    ap.add_argument("--base-model",
                    default=str(ROOT / "output/fugu_27b_r4_merged_bf16"))
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    controller = MergedReloadController(
        base_model_dir=Path(args.base_model),
        checkpoints_dir=Path(args.trainer_output),
        merged_root=Path(args.merged_root),
        python_bin=str(ROOT / ".venv/bin/python"),
        server_log=Path(args.trainer_output) / "vllm_reloads.log",
    )
    controller.merged_root.mkdir(parents=True, exist_ok=True)
    watch(controller, poll_s=args.poll, once=args.once)


if __name__ == "__main__":
    main()
