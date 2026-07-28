"""DABstep training manifests (office lane 4): data analysis over files.

DABstep ships its 450-task set with answers WITHHELD (live leaderboard), but
the leaderboard also publishes every submission's per-task answer together
with a correctness bit (``data/task_scores/*.jsonl``). Every task has correct
public submissions, so the gold is reconstructed as the SET of accepted
answer variants (all distinct correct answers above a small support
threshold — the official scorer accepts multiple formats, and the set
preserves that).

Lane shape: single-turn code execution. The worker writes ONE python script;
``dabstep_exec`` runs it with CWD = the public context directory (23MB
payments.csv is read by the script, never inlined) and matches the script's
final stdout line against the accepted set. Single-turn + registry grader
means the lane probe-curates like every other single-turn lane.

REPORTING CONSEQUENCE (recorded in MISSION.md): training on reconstructed
golds burns DABstep as a reportable benchmark; the 10-answer dev split stays
an internal smoke eval only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CAPABILITY = "office_data_analysis"
GRADER = "dabstep_exec"
MIN_SUPPORT = 2          # accepted variants need >=2 correct votes (scorer flukes)
SCRIPT_TIMEOUT_S = 120.0
PROBE_SAMPLE_NAME = "dabstep_probe_candidates_taskspecs.jsonl"

# Question FIRST: the conductor's view of this prompt truncates at
# max_task_chars (12k), and the manual alone is 22k chars — a trailing
# question would be invisible to the router.
PROMPT_TEMPLATE = """\
You are answering a data-analysis question over the DABstep payments dataset.

Question: {question}

Guidelines: {guidelines}

Write ONE python script that computes the answer. Reply with a single
```python fenced script; it must print ONLY the final answer (formatted per
the guidelines) as its last line of output. The script runs with its working
directory set to the dataset directory, which contains:

{inventory}

payments-readme.md:
{readme}

The dataset manual (domain rules — fee logic, definitions; consult it for
anything fee- or rule-related):
{manual}"""


def reconstruct_golds(scores_dir: Path, *, min_support: int = MIN_SUPPORT
                      ) -> dict[str, dict[str, Any]]:
    """task_id -> {"accepted": [variants], "votes": total_correct}."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for path in sorted(scores_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("score") is True:
                answer = str(row.get("agent_answer") or "").strip()
                if answer:
                    votes[str(row.get("task_id"))][answer] += 1
    golds: dict[str, dict[str, Any]] = {}
    for task_id, counter in votes.items():
        accepted = [a for a, n in counter.most_common() if n >= min_support]
        if not accepted:  # solved only once ever — keep that single variant
            accepted = [counter.most_common(1)[0][0]]
        golds[task_id] = {"accepted": accepted, "votes": sum(counter.values())}
    return golds


def _inventory(context_dir: Path) -> str:
    lines = []
    for path in sorted(context_dir.iterdir()):
        if path.suffix == ".py" or path.name.startswith("."):
            continue
        lines.append(f"- {path.name} ({path.stat().st_size:,} bytes)")
    return "\n".join(lines)


def build_taskspec(task: dict[str, Any], gold: dict[str, Any],
                   context_dir: Path, *, readme: str, manual: str,
                   inventory: str) -> dict[str, Any]:
    task_id = f"dabstep__{int(task['task_id']):04d}"
    prompt = PROMPT_TEMPLATE.format(
        inventory=inventory, readme=readme.strip(), manual=manual.strip(),
        question=str(task["question"]).strip(),
        guidelines=str(task.get("guidelines") or "").strip(),
    )
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "capability": CAPABILITY,
        "source": {
            "name": "dabstep", "url_or_ref": "https://huggingface.co/datasets/adyen/DABstep",
            "version": "dabstep-v1", "license": "CC BY 4.0",
            "policy": "train_allowed", "source_commit": None,
        },
        "input": {"messages": [{"role": "user", "content": prompt}],
                  "context_documents": [], "assets": [], "repo": None, "tools": []},
        "environment": {"harness": "code_exec", "image": None, "cpu_limit": None,
                        "memory_mb": None, "disk_mb": None,
                        "network_policy": "model-relay-only", "wall_time_seconds": 900},
        "grader": {"type": GRADER, "command": None,
                   "expected_answer": {"accepted": list(gold["accepted"]),
                                       "context_dir": str(context_dir),
                                       "timeout": SCRIPT_TIMEOUT_S},
                   "score_range": [0.0, 1.0], "success_threshold": 1.0,
                   "deterministic": True},
        "splitting": {"split": "grpo_train", "group_id": "dabstep_main",
                      "contamination_group": f"dabstep/{task['task_id']}"},
        "metadata": {"domain": CAPABILITY, "subdomain": "dabstep",
                     "difficulty_estimate": {"easy": 0.3, "hard": 0.8}.get(
                         str(task.get("level")), None),
                     "estimated_worker_calls": 1, "requires_tools": False,
                     "requires_long_context": True,
                     "tags": ["dabstep", "data_analysis",
                              f"level:{task.get('level')}",
                              "reconstructed_gold"]},
    }


def export(tasks_file: Path, scores_dir: Path, context_dir: Path,
           out_dir: Path) -> dict[str, Any]:
    golds = reconstruct_golds(scores_dir)
    tasks = [json.loads(l) for l in tasks_file.read_text().splitlines() if l.strip()]
    context_dir = context_dir.resolve()
    readme = (context_dir / "payments-readme.md").read_text()
    manual = (context_dir / "manual.md").read_text()
    inventory = _inventory(context_dir)

    rows: list[str] = []
    unsolved = 0
    for task in tasks:
        gold = golds.get(str(task["task_id"]))
        if gold is None:
            unsolved += 1
            continue
        rows.append(json.dumps(build_taskspec(
            task, gold, context_dir, readme=readme, manual=manual,
            inventory=inventory), sort_keys=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "dabstep_taskspecs.jsonl"
    manifest.write_text("".join(r + "\n" for r in rows))
    # every task doubles as a probe candidate (450 is already probe-sized)
    probe = out_dir / PROBE_SAMPLE_NAME
    probe.write_text("".join(r + "\n" for r in rows))
    return {"tasks": len(rows), "unsolved_excluded": unsolved,
            "manifest": str(manifest), "probe_candidates": str(probe)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.tasks, args.scores_dir, args.context_dir,
                            args.out_dir), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
