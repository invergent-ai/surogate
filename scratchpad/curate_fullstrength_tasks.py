"""Curate the 50-task manifest for the full-strength cross-fitted headroom test.

25 = the derisk-study tasks (fetched from SWE-bench/SWE-smith BY instance_id — the
stream shuffle is not reproducible across datasets versions); 25 fresh = repo-balanced
sample (seed 20260708, max 3/repo) from study-proven repos with local Docker images.
Frozen to scratchpad/fullstrength_tasks.jsonl BEFORE any worker call.

Run with director's venv (datasets dep):
  cd /home/densemax/work/flavius/surogate && \
    director/.venv/bin/python scratchpad/curate_fullstrength_tasks.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERISK = ROOT / "director/manifests/fugu_clean_v1/derisk_swesmith_routability.jsonl"
OUT = ROOT / "scratchpad/fullstrength_tasks.jsonl"
N_FRESH = 25
MAX_PER_REPO = 3
SEED = 20260708


def local_images() -> set[str]:
    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}"],
        capture_output=True, text=True, check=True,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def main() -> None:
    study_ids = {json.loads(l)["task_id"] for l in DERISK.read_text().splitlines() if l.strip()}
    study_repos = {tid.split(".")[0] for tid in study_ids}
    images = local_images()
    print(f"study tasks: {len(study_ids)}, repos: {len(study_repos)}")

    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)

    study_rows: dict[str, dict] = {}
    candidates: dict[str, list[dict]] = defaultdict(list)
    seen = 0
    for r in ds:
        seen += 1
        row = dict(r)
        iid = row["instance_id"]
        repo = iid.split(".")[0]
        if iid in study_ids:
            study_rows[iid] = row
        elif (
            repo in study_repos
            and str(row.get("image_name", "")).split(":")[0] in images
            and str(row.get("problem_statement", "")).strip()
        ):
            candidates[repo].append(row)
    print(f"scanned {seen} rows; study found {len(study_rows)}/{len(study_ids)}; "
          f"fresh candidates: { {k: len(v) for k, v in sorted(candidates.items())} }")

    missing = study_ids - set(study_rows)
    if missing:
        raise SystemExit(f"FATAL: study tasks missing from dataset: {sorted(missing)}")

    # Repo-balanced fresh sample: round-robin over repos, max 3 per repo, fixed seed.
    rng = random.Random(SEED)
    for repo in candidates:
        rng.shuffle(candidates[repo])
    fresh: list[dict] = []
    per_repo: Counter = Counter()
    order = sorted(candidates)
    round_i = 0
    while len(fresh) < N_FRESH and round_i < MAX_PER_REPO:
        for repo in order:
            if len(fresh) >= N_FRESH:
                break
            if round_i < len(candidates[repo]) and per_repo[repo] < MAX_PER_REPO:
                fresh.append(candidates[repo][round_i])
                per_repo[repo] += 1
        round_i += 1

    out_rows = [
        {"task_id": iid, "cohort": "study25", "payload": row}
        for iid, row in sorted(study_rows.items())
    ] + [
        {"task_id": row["instance_id"], "cohort": "fresh25", "payload": row}
        for row in fresh
    ]
    ids = [r["task_id"] for r in out_rows]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    with OUT.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")

    repos = Counter(r["task_id"].split(".")[0] for r in out_rows)
    print(f"wrote {len(out_rows)} tasks to {OUT}")
    print("repo mix:", dict(repos))
    print("cohorts:", dict(Counter(r["cohort"] for r in out_rows)))


if __name__ == "__main__":
    sys.exit(main())
