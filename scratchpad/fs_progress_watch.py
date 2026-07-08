"""Progress watcher for the full-strength verdict eval.

Every INTERVAL seconds: reads scratchpad/fs_verdict_rows.jsonl, computes per-arm
progress + running rates, and rewrites the marked block in MISSION.md
(<!-- FS_VERDICT_PROGRESS_START --> ... END -->). Exits when the eval process is gone
and the row count has been stable for 3 consecutive checks (writes a final block).

Run:  .venv/bin/python scratchpad/fs_progress_watch.py [--interval 120]
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / os.environ.get("FS_ROWS", "scratchpad/fs_verdict_rows.jsonl")
MISSION = ROOT / "MISSION.md"
START = "<!-- FS_VERDICT_PROGRESS_START -->"
END = "<!-- FS_VERDICT_PROGRESS_END -->"
N_TASKS = int(os.environ.get("FS_N_TASKS", "60"))
EXPECTED = {"solo": 4 * N_TASKS, "solo2": 4 * N_TASKS, "fu1": N_TASKS, "fu2": N_TASKS}


def eval_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "eval_fullstrength_verdict.py"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def render() -> str:
    rows = []
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    by_arm: dict[str, list] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    groups = {"solo": [a for a in by_arm if a.startswith("solo__")],
              "solo2": [a for a in by_arm if a.startswith("solo2__")],
              "fu1": [a for a in by_arm if a == "fu1"],
              "fu2": [a for a in by_arm if a == "fu2"]}
    done = {g: sum(len(by_arm[a]) for a in arms) for g, arms in groups.items()}
    total_done = sum(done.values())
    total_expected = sum(EXPECTED.values())
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"**{now}** — rows {total_done}/{total_expected} "
             f"({', '.join(f'{g} {done[g]}/{EXPECTED[g]}' for g in EXPECTED)})", ""]
    lines.append("| arm | n | rate | errors |")
    lines.append("|---|--:|--:|--:|")
    for a in sorted(by_arm):
        rs = by_arm[a]
        rate = sum(r["score"] for r in rs) / len(rs)
        errs = sum(1 for r in rs if str(r.get("status", "")).startswith(("error", "exec_error", "parse_fail", "empty_gen")))
        lines.append(f"| {a} | {len(rs)} | {rate:.3f} | {errs} |")
    running = eval_running()
    lines.append("")
    lines.append(f"eval process: {'RUNNING' if running else 'not running'}")
    return "\n".join(lines)


def write_block(body: str) -> None:
    text = MISSION.read_text()
    pre, _, rest = text.partition(START)
    _, _, post = rest.partition(END)
    MISSION.write_text(pre + START + "\n" + body + "\n" + END + post)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    stable = 0
    last_count = -1
    while True:
        body = render()
        write_block(body)
        count = body.split("rows ", 1)[1].split("/", 1)[0] if "rows " in body else "0"
        if not eval_running():
            stable = stable + 1 if count == str(last_count) or last_count == -1 else 0
            if stable >= 3:
                write_block(body + "\n\n**WATCHER EXITED (run finished or stopped).**")
                print("watcher: run finished/stopped, exiting", flush=True)
                return
        else:
            stable = 0
        last_count = count
        print(f"watcher: {count} rows, running={eval_running()}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
