#!/usr/bin/env python3
"""Phase 3 benchmark gate runner for buffer-runtime-v4.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_buffer_runtime_v4.py \
        --config examples/sft/qwen3/qwen3-lora-bf16-bench.yaml \
        --mode stream

Writes a single JSON result line to stdout so a driver can aggregate runs.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _compute_tokens_per_step(config_path: str) -> int:
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return int(
        cfg.get("per_device_train_batch_size", 1)
        * cfg.get("sequence_len", 0)
        * cfg.get("gradient_accumulation_steps", 1)
        * cfg.get("gpus", 1)
    )


STEP_LINE_RE = re.compile(
    r":: step\s+(?P<step>\d+)\s+\[[^\]]+\][^|]*loss\s+(?P<loss>[-\d.]+)[^|]*\|"
    r"[^|]*norm[^|]*\|[^|]*tps\s*\|\s*(?P<ms>\d+)\s*ms"
)


def poll_peak_memory(device_ids: list, stop_event: threading.Event, out: dict, interval_s: float = 0.25):
    peak_per_dev = {d: 0 for d in device_ids}
    smi = [
        "nvidia-smi",
        f"--id={','.join(str(d) for d in device_ids)}",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        try:
            r = subprocess.run(smi, capture_output=True, text=True, timeout=5)
            for line in r.stdout.strip().splitlines():
                idx_str, used_str = [s.strip() for s in line.split(",")]
                idx = int(idx_str)
                used = int(used_str)
                if used > peak_per_dev.get(idx, 0):
                    peak_per_dev[idx] = used
        except Exception:
            pass
        stop_event.wait(interval_s)
    out["peak_per_device_mib"] = peak_per_dev
    out["peak_mib"] = max(peak_per_dev.values()) if peak_per_dev else 0


def run_benchmark(config: str, mode: str, warmup: int = 3) -> dict:
    env = os.environ.copy()
    if mode == "legacy":
        env["SUROGATE_LEGACY_EXECUTOR"] = "1"
    elif mode == "stream":
        env.pop("SUROGATE_LEGACY_EXECUTOR", None)
    else:
        raise ValueError(f"unknown mode: {mode}")

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "0")
    device_ids = [int(x) for x in cuda_visible.split(",") if x.strip()]

    mem_out = {}
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_peak_memory, args=(device_ids, stop_event, mem_out), daemon=True)
    poller.start()

    cmd = [".venv/bin/surogate", "sft", config]
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    step_records = []
    output_lines = []
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    try:
        for line in proc.stdout:
            output_lines.append(line)
            clean = ansi_re.sub("", line)
            m = STEP_LINE_RE.search(clean)
            if m:
                step_records.append(
                    {
                        "step": int(m.group("step")),
                        "loss": float(m.group("loss")),
                        "duration_ms": int(m.group("ms")),
                    }
                )
        proc.wait()
    finally:
        stop_event.set()
        poller.join(timeout=2)

    wall_s = time.time() - start

    measured = [r for r in step_records if int(r["step"]) >= warmup]
    n = len(measured)
    if n == 0:
        return {
            "config": config,
            "mode": mode,
            "ok": False,
            "error": "no post-warmup steps captured",
            "returncode": proc.returncode,
            "wall_s": wall_s,
            "tail": "".join(output_lines[-40:]),
        }

    tokens_per_step = _compute_tokens_per_step(config)
    durations = [int(r["duration_ms"]) for r in measured]
    avg_ms = sum(durations) / n
    tps = (1000.0 * tokens_per_step / avg_ms) if avg_ms > 0 and tokens_per_step else 0.0

    return {
        "config": config,
        "mode": mode,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "warmup_dropped": warmup,
        "measured_steps": n,
        "avg_step_ms": avg_ms,
        "min_step_ms": min(durations),
        "max_step_ms": max(durations),
        "tokens_per_step": tokens_per_step,
        "tokens_per_s": tps,
        "peak_mib": mem_out.get("peak_mib", 0),
        "peak_per_device_mib": mem_out.get("peak_per_device_mib", {}),
        "durations_ms": durations,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["legacy", "stream"], required=True)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    result = run_benchmark(args.config, args.mode, warmup=args.warmup)
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
