"""Offline pool-selection analysis for the Ultra worker mix.

This module uses the evidence we already paid for:

* ``probe.jsonl``: six open-weight workers over the large direct bank.
* ``agentic_bank.jsonl``: six open-weight workers over tau-style agentic tasks.
* ``pool_matrix_frontier.jsonl``: Opus/Gemini/GPT/GLM repeated samples on an
  80-task joined slice.

It deliberately does not call any provider. Its job is to make the current model-mix
claim auditable and to specify the smallest paid workflow tournament needed next.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from .pool_tournament import MODEL_PRICING, MODEL_SLUGS


OPEN_TO_CANONICAL = {
    "deepseek": "deepseek-pro",
    "kimi": "kimi-code",
    "glm": "glm",
    "mimo": "mimo",
    "minimax": "minimax",
    "deepseek_flash": "flash",
}

CANONICAL_TO_OPEN = {v: k for k, v in OPEN_TO_CANONICAL.items()}

OPEN_MODELS = ["flash", "deepseek-pro", "glm", "kimi-code", "mimo", "minimax"]
COMMERCIAL_MODELS = ["opus", "gemini", "gpt"]
ALL_MODELS = ["flash", "deepseek-pro", "glm", "kimi-code", "mimo", "minimax", "opus", "gemini", "gpt"]

# The current hypothesis, not the conclusion.
PROPOSED_POOL = ["opus", "gemini", "gpt", "glm", "flash", "mimo"]

# Quality-first Ultra recommendation. This is a worker pool, not a directive to
# route every task to every worker; role assignment is part of the Conductor.
QUALITY_FIRST_CORE = ["opus", "gemini", "gpt", "kimi-code", "mimo", "glm", "flash"]
QUALITY_FIRST_EXPANDED = QUALITY_FIRST_CORE + ["minimax", "deepseek-pro"]

SCAFFOLD_AWARE_CORE = [
    "codex:gpt-5-codex",
    "claude-code:opus-4.8",
    "opencode:kimi-code",
    "opencode:mimo",
    "opencode:glm",
    "direct:gemini-3.1-pro",
    "direct:gpt-5.5",
    "direct:opus-4.8",
    "opencode:flash",
]

TRACE_SOURCE_MIX = [
    ["OpenCode traces", "25-35%", "Open and controllable baseline harness"],
    ["Claude Code traces", "25-35%", "Opus-style debugging and long-horizon repo work"],
    ["Codex traces", "25-35%", "GPT/Codex-style building, repair, and skills"],
    ["Fresh benchmark tasks", "10-20%", "Avoid overfitting to existing agent traces"],
]

SCAFFOLD_BASELINES = [
    "Claude Code solo",
    "Codex solo",
    "OpenCode/Kimi solo",
    "best direct frontier model solo",
    "best-of-N single scaffold",
    "single-scaffold self-reflection",
    "fixed multi-agent workflow",
    "trained Fugu-Ultra Conductor",
]

# Diagnostic only: useful if a deployment constraint forces a six-worker pool.
AGENTIC_WEIGHTED_SIX = ["opus", "glm", "flash", "mimo", "kimi-code", "minimax"]

GENERAL_AGENTIC_WEIGHTS = {
    "direct80": 0.10,
    "hist_tau80_open": 0.20,
    "live_tau4": 0.35,
    "coding3": 0.35,
}

# Provisional coding-primary pool from the only live agentic-coding shard.
# Flash is a cheap fixed-size filler, not a positive coding signal on the 3-task shard.
CODING_PRIMARY_OPEN_SIX = ["kimi-code", "mimo", "glm", "deepseek-pro", "minimax", "flash"]
CODING_PRIMARY_POSITIVE_FIVE = ["kimi-code", "mimo", "glm", "deepseek-pro", "minimax"]

CANDIDATE_POOLS = {
    "original-six": PROPOSED_POOL,
    "quality-first/core-seven": QUALITY_FIRST_CORE,
    "quality-first/expanded-nine": QUALITY_FIRST_EXPANDED,
    "deployment-constrained/agentic-six": AGENTIC_WEIGHTED_SIX,
    "coding-ablation/open-six": CODING_PRIMARY_OPEN_SIX,
    "coding-ablation/positive-five": CODING_PRIMARY_POSITIVE_FIVE,
    "drop-opus-add-kimi": ["gemini", "gpt", "glm", "flash", "mimo", "kimi-code"],
    "drop-gemini-add-kimi": ["opus", "gpt", "glm", "flash", "mimo", "kimi-code"],
    "drop-flash-add-kimi": ["opus", "gemini", "gpt", "glm", "mimo", "kimi-code"],
    "drop-mimo-add-kimi": ["opus", "gemini", "gpt", "glm", "flash", "kimi-code"],
    "drop-glm-add-kimi": ["opus", "gemini", "gpt", "flash", "mimo", "kimi-code"],
}


@dataclass(frozen=True)
class JoinedTask:
    task_id: str
    domain: str
    scores: dict[str, float]


@dataclass(frozen=True)
class AgenticTask:
    item_id: str
    domain: str
    scores: dict[str, float]
    costs: dict[str, float]


def default_manifest_dir() -> Path:
    # ultra/ultra/pool_selection.py -> repo root
    return Path(__file__).resolve().parents[2] / "director" / "manifests" / "fugu_clean_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_optional_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _read_jsonl(path)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def _subset_score(rows: list[JoinedTask] | list[AgenticTask], subset: Iterable[str]) -> float:
    members = list(subset)
    if not rows or not members:
        return 0.0
    return mean(max((row.scores.get(m, 0.0) for m in members), default=0.0) for row in rows)


def _task_values(rows: list[JoinedTask] | list[AgenticTask], subset: Iterable[str]) -> list[float]:
    members = list(subset)
    return [max((row.scores.get(m, 0.0) for m in members), default=0.0) for row in rows]


def _paired_delta_ci(
    rows: list[JoinedTask] | list[AgenticTask],
    a: Iterable[str],
    b: Iterable[str],
    *,
    seed: int = 0,
    iters: int = 4000,
) -> tuple[float, tuple[float, float]]:
    av = _task_values(rows, a)
    bv = _task_values(rows, b)
    diffs = [x - y for x, y in zip(av, bv, strict=True)]
    if not diffs:
        return 0.0, (0.0, 0.0)
    point = mean(diffs)
    rnd = random.Random(seed)
    boots = []
    n = len(diffs)
    for _ in range(iters):
        boots.append(mean(diffs[rnd.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * (iters - 1))]
    hi = boots[int(0.975 * (iters - 1))]
    return point, (lo, hi)


def load_joined_tasks(manifest_dir: Path, *, frontier_threshold: float = 0.5) -> list[JoinedTask]:
    """Join the 80-task frontier slice against the six-open-worker probe bank.

    Frontier workers have eight samples per task. To match the user's derived table, we
    turn that into a binary task success when ``r_bar >= frontier_threshold``.
    """

    probe_by_id = {row["task_id"]: row for row in _read_jsonl(manifest_dir / "probe.jsonl")}
    meta = json.loads((manifest_dir / "meta.json").read_text())
    open_workers = meta["worker_ids"]

    out: list[JoinedTask] = []
    for frontier in _read_jsonl(manifest_dir / "pool_matrix_frontier.jsonl"):
        task_id = frontier["task_id"]
        probe = probe_by_id[task_id]
        scores: dict[str, float] = {}
        for idx, worker in enumerate(open_workers):
            scores[OPEN_TO_CANONICAL[worker]] = float(probe["rewards"][idx])
        for idx, worker in enumerate(frontier["worker_ids"]):
            # The frontier matrix includes a repeated-sample GLM control, but the
            # nine-model comparison requested here uses the six open-worker probe
            # bank for open models. Do not overwrite that GLM value.
            if worker in OPEN_MODELS:
                continue
            r_bar = mean(float(x) for x in frontier["rewards"][idx])
            scores[worker] = 1.0 if r_bar >= frontier_threshold else 0.0
        out.append(JoinedTask(task_id=task_id, domain=frontier["domain"], scores=scores))
    return out


def load_open_probe(manifest_dir: Path) -> list[JoinedTask]:
    meta = json.loads((manifest_dir / "meta.json").read_text())
    open_workers = meta["worker_ids"]
    out = []
    for row in _read_jsonl(manifest_dir / "probe.jsonl"):
        scores = {
            OPEN_TO_CANONICAL[worker]: float(row["rewards"][idx])
            for idx, worker in enumerate(open_workers)
        }
        out.append(JoinedTask(task_id=row["task_id"], domain=row["domain"], scores=scores))
    return out


def load_agentic_tasks(manifest_dir: Path) -> list[AgenticTask]:
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"scores": {}, "costs": {}}
    )
    for row in _read_jsonl(manifest_dir / "agentic_bank.jsonl"):
        key = (row["domain"], row["item_id"])
        model = OPEN_TO_CANONICAL[row["worker"]]
        grouped[key]["scores"][model] = float(row["reward"])
        grouped[key]["costs"][model] = float(row["cost"])
    return [
        AgenticTask(item_id=item_id, domain=domain, scores=v["scores"], costs=v["costs"])
        for (domain, item_id), v in sorted(grouped.items())
    ]


def load_live_tau_tasks(
    manifest_dir: Path,
    *,
    filename: str = "agentic_frontier_tau4.jsonl",
) -> list[AgenticTask]:
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"scores": {}, "costs": {}}
    )
    for row in _read_optional_jsonl(manifest_dir / filename):
        key = (row["domain"], row["item_id"])
        worker = row["worker"]
        valid = bool(row.get("valid", True))
        grouped[key]["scores"][worker] = float(row.get("reward", 0.0)) if valid else 0.0
        grouped[key]["costs"][worker] = float(row.get("cost", 0.0))
    return [
        AgenticTask(item_id=item_id, domain=domain, scores=v["scores"], costs=v["costs"])
        for (domain, item_id), v in sorted(grouped.items())
    ]


def load_coding_tasks(
    manifest_dir: Path,
    *,
    filename: str = "agentic_coding_frontier_direct3.jsonl",
) -> list[AgenticTask]:
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: {"scores": {}, "costs": {}})
    for row in _read_optional_jsonl(manifest_dir / filename):
        if row.get("stage") != "direct":
            continue
        workers = row.get("workers") or []
        if len(workers) != 1:
            continue
        worker = workers[0]
        valid = bool(row.get("valid", True))
        grouped[row["task_id"]]["scores"][worker] = float(row.get("reward", 0.0)) if valid else 0.0
        grouped[row["task_id"]]["costs"][worker] = float(row.get("cost", 0.0))
    return [
        AgenticTask(item_id=task_id, domain="coding", scores=v["scores"], costs=v["costs"])
        for task_id, v in sorted(grouped.items())
    ]


def model_accuracy(rows: list[JoinedTask] | list[AgenticTask], models: list[str]) -> dict[str, float]:
    return {
        model: mean(row.scores[model] for row in rows if model in row.scores)
        for model in models
        if any(model in row.scores for row in rows)
    }


def model_accuracy_by_domain(
    rows: list[JoinedTask] | list[AgenticTask], models: list[str]
) -> dict[str, dict[str, float]]:
    domains = sorted({row.domain for row in rows})
    out: dict[str, dict[str, float]] = {}
    for model in models:
        out[model] = {}
        for domain in domains:
            vals = [row.scores[model] for row in rows if row.domain == domain and model in row.scores]
            out[model][domain] = mean(vals) if vals else 0.0
    return out


def best_subsets(
    rows: list[JoinedTask] | list[AgenticTask],
    models: list[str],
    *,
    max_size: int,
) -> dict[int, tuple[float, list[tuple[str, ...]]]]:
    out = {}
    for size in range(1, max_size + 1):
        best = -1.0
        winners: list[tuple[str, ...]] = []
        for subset in itertools.combinations(models, size):
            score = _subset_score(rows, subset)
            if score > best + 1e-12:
                best = score
                winners = [subset]
            elif abs(score - best) <= 1e-12:
                winners.append(subset)
        out[size] = (best, winners)
    return out


def _domain_counts(rows: list[JoinedTask] | list[AgenticTask]) -> Counter:
    return Counter(row.domain for row in rows)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def _coverage_rows(rows: list[JoinedTask], subsets: dict[str, list[str]]) -> list[list[str]]:
    out = []
    for name, subset in subsets.items():
        out.append([name, ", ".join(subset), _fmt_pct(_subset_score(rows, subset))])
    return out


def _catalog_rows() -> list[list[str]]:
    rows = []
    for model in ALL_MODELS:
        pricing = MODEL_PRICING[model]
        rows.append(
            [
                model,
                MODEL_SLUGS[model],
                f"${pricing['prompt'] * 1_000_000:.3f}",
                f"${pricing['completion'] * 1_000_000:.3f}",
            ]
        )
    return rows


def _worker_success_rows(rows: list[AgenticTask], models: list[str]) -> list[list[str]]:
    out = []
    for model in models:
        wins = sum(row.scores.get(model, 0.0) >= 1.0 for row in rows)
        total_cost = sum(row.costs.get(model, 0.0) for row in rows)
        cost_per_task = total_cost / len(rows) if rows else 0.0
        out.append(
            [
                model,
                f"{wins}/{len(rows)}",
                _fmt_pct(wins / len(rows) if rows else 0.0),
                f"${cost_per_task:.6f}",
                f"${total_cost:.6f}",
            ]
        )
    return out


def _coding_primary_rows(rows: list[AgenticTask]) -> list[list[str]]:
    scored = []
    for model in ALL_MODELS:
        wins = sum(row.scores.get(model, 0.0) >= 1.0 for row in rows)
        total_cost = sum(row.costs.get(model, 0.0) for row in rows)
        cost_per_task = total_cost / len(rows) if rows else 0.0
        scored.append((wins, -cost_per_task, model, total_cost))
    scored.sort(reverse=True)
    out = []
    for wins, neg_cost_per_task, model, total_cost in scored:
        cost_per_task = -neg_cost_per_task
        out.append(
            [
                model,
                f"{wins}/{len(rows)}",
                f"${cost_per_task:.6f}",
                f"${total_cost:.6f}",
            ]
        )
    return out


def _short_task_name(task_id: str) -> str:
    return task_id.split("__", 1)[0]


def _coding_attempt_audit_rows(
    manifest_dir: Path,
    *,
    filename: str = "agentic_coding_frontier_direct3.jsonl",
    models: tuple[str, ...] = ("opus", "gemini", "kimi-code"),
) -> list[list[str]]:
    rows = []
    selected = set(models)
    for row in _read_optional_jsonl(manifest_dir / filename):
        if row.get("stage") != "direct":
            continue
        workers = row.get("workers") or []
        if len(workers) != 1 or workers[0] not in selected:
            continue
        step = (row.get("steps") or [{}])[0]
        rows.append(
            [
                workers[0],
                _short_task_name(row["task_id"]),
                str(int(float(row.get("reward", 0.0)))),
                str(step.get("status")),
                str(step.get("diff_len")),
                f"{float(row.get('elapsed_s', 0.0)):.0f}s",
                f"${float(row.get('cost', 0.0)):.2f}",
            ]
        )
    return rows


def _coding_diff_len_notes(
    manifest_dir: Path,
    *,
    filename: str = "agentic_coding_frontier_direct3.jsonl",
) -> list[str]:
    notes = []
    for task_id in sorted({row["task_id"] for row in _read_optional_jsonl(manifest_dir / filename)}):
        grouped: dict[int, list[str]] = defaultdict(list)
        for row in _read_optional_jsonl(manifest_dir / filename):
            if row.get("stage") != "direct" or row.get("task_id") != task_id:
                continue
            workers = row.get("workers") or []
            step = (row.get("steps") or [{}])[0]
            diff_len = step.get("diff_len")
            if workers and isinstance(diff_len, int) and diff_len > 0:
                grouped[diff_len].append(workers[0])
        for diff_len, workers in sorted(grouped.items()):
            if len(workers) >= 3:
                notes.append(f"{_short_task_name(task_id)} has diff_len={diff_len} for {', '.join(sorted(workers))}.")
    return notes


def _solver_rows(rows: list[AgenticTask]) -> list[list[str]]:
    out = []
    for row in rows:
        solvers = [model for model in ALL_MODELS if row.scores.get(model, 0.0) >= 1.0]
        out.append([row.domain, row.item_id, ", ".join(solvers) if solvers else "(none)"])
    return out


def _combined_candidate_rows(
    strata: list[tuple[str, list[JoinedTask] | list[AgenticTask]]],
) -> list[list[str]]:
    out = []
    for name, pool in CANDIDATE_POOLS.items():
        scores = [_subset_score(rows, pool) for _, rows in strata]
        out.append(
            [name, str(len(pool)), ", ".join(pool)]
            + [f"{score:.4f}" for score in scores]
            + [f"{mean(scores):.4f}" if scores else "0.0000"]
        )
    return out


def _weighted_candidate_rows(
    strata: list[tuple[str, list[JoinedTask] | list[AgenticTask]]],
    weights: dict[str, float],
) -> list[list[str]]:
    out = []
    for name, pool in CANDIDATE_POOLS.items():
        weighted = 0.0
        displayed = []
        for stratum_name, rows in strata:
            score = _subset_score(rows, pool)
            weighted += weights.get(stratum_name, 0.0) * score
            displayed.append(f"{score:.4f}")
        out.append([name, str(len(pool)), ", ".join(pool), f"{weighted:.4f}"] + displayed)
    out.sort(key=lambda row: float(row[3]), reverse=True)
    return out


def _weighted_subset_rows(
    strata: list[tuple[str, list[JoinedTask] | list[AgenticTask]]],
    weights: dict[str, float],
    *,
    size: int,
    limit: int = 10,
) -> list[list[str]]:
    scored = []
    for subset in itertools.combinations(ALL_MODELS, size):
        scores = {name: _subset_score(rows, subset) for name, rows in strata}
        weighted = sum(weights.get(name, 0.0) * score for name, score in scores.items())
        scored.append((weighted, scores, subset))
    scored.sort(key=lambda item: (item[0], *item[1].values(), item[2]), reverse=True)
    return [
        ["+".join(subset), f"{weighted:.4f}"] + [f"{scores[name]:.4f}" for name, _ in strata]
        for weighted, scores, subset in scored[:limit]
    ]


def _strict_subset_rows(
    strata: list[tuple[str, list[JoinedTask] | list[AgenticTask]]],
    *,
    size: int,
    limit: int = 10,
) -> list[list[str]]:
    scored = []
    for subset in itertools.combinations(ALL_MODELS, size):
        scores = [_subset_score(rows, subset) for _, rows in strata]
        scored.append((mean(scores), scores, subset))
    scored.sort(key=lambda item: (item[0], *item[1], item[2]), reverse=True)
    return [
        ["+".join(subset), f"{avg:.4f}"] + [f"{score:.4f}" for score in scores]
        for avg, scores, subset in scored[:limit]
    ]


def render_report(manifest_dir: Path, *, budget_usd: float = 200.0, seed: int = 0) -> str:
    joined = load_joined_tasks(manifest_dir)
    open_probe = load_open_probe(manifest_dir)
    agentic = load_agentic_tasks(manifest_dir)
    live_tau = load_live_tau_tasks(manifest_dir)
    coding = load_coding_tasks(manifest_dir)
    combined_strata: list[tuple[str, list[JoinedTask] | list[AgenticTask]]] = [
        ("direct80", joined),
        ("hist_tau80_open", agentic),
    ]
    if live_tau:
        combined_strata.append(("live_tau4", live_tau))
    if coding:
        combined_strata.append(("coding3", coding))

    joined_acc = model_accuracy(joined, ALL_MODELS)
    joined_by_domain = model_accuracy_by_domain(joined, ALL_MODELS)
    open_acc = model_accuracy(open_probe, OPEN_MODELS)
    agentic_acc = model_accuracy(agentic, OPEN_MODELS)

    proposed = PROPOSED_POOL
    joined_best = best_subsets(joined, ALL_MODELS, max_size=6)
    open_best = best_subsets(open_probe, OPEN_MODELS, max_size=6)
    agentic_best = best_subsets(agentic, OPEN_MODELS, max_size=6)

    proposed_joined = _subset_score(joined, proposed)
    proposed_joined_ci = {
        m: _paired_delta_ci(joined, proposed, [x for x in proposed if x != m], seed=seed)
        for m in proposed
    }
    challenger_swap_rows = []
    for challenger in ["deepseek-pro", "kimi-code", "minimax"]:
        add_pool = proposed + [challenger]
        add_delta, add_ci = _paired_delta_ci(joined, add_pool, proposed, seed=seed)
        challenger_swap_rows.append(
            [
                f"add {challenger}",
                _fmt_pct(_subset_score(joined, add_pool)),
                _fmt_pct(add_delta),
                f"[{_fmt_pct(add_ci[0])}, {_fmt_pct(add_ci[1])}]",
            ]
        )
        for incumbent in proposed:
            swapped = [x for x in proposed if x != incumbent] + [challenger]
            delta, ci = _paired_delta_ci(joined, swapped, proposed, seed=seed)
            challenger_swap_rows.append(
                [
                    f"{challenger} for {incumbent}",
                    _fmt_pct(_subset_score(joined, swapped)),
                    _fmt_pct(delta),
                    f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]",
                ]
            )

    agentic_core = ["glm", "flash", "mimo"]
    agentic_core_score = _subset_score(agentic, agentic_core)
    agentic_marginal = {
        m: _paired_delta_ci(agentic, agentic_core, [x for x in agentic_core if x != m], seed=seed)
        for m in agentic_core
    }

    joined_rows = []
    domains = ["math", "code", "science", "general"]
    for model in ALL_MODELS:
        joined_rows.append(
            [model, _fmt_pct(joined_acc[model])]
            + [_fmt_pct(joined_by_domain[model].get(domain, 0.0)) for domain in domains]
        )

    open_rows = [[model, _fmt_pct(open_acc[model])] for model in OPEN_MODELS]
    agentic_rows = [[model, _fmt_pct(agentic_acc[model])] for model in OPEN_MODELS]

    loo_rows = []
    for model in proposed:
        delta, ci = proposed_joined_ci[model]
        without = [x for x in proposed if x != model]
        loo_rows.append(
            [
                model,
                _fmt_pct(_subset_score(joined, without)),
                _fmt_pct(delta),
                f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]",
            ]
        )

    agentic_loo_rows = []
    for model in agentic_core:
        delta, ci = agentic_marginal[model]
        without = [x for x in agentic_core if x != model]
        agentic_loo_rows.append(
            [
                model,
                _fmt_pct(_subset_score(agentic, without)),
                _fmt_pct(delta),
                f"[{_fmt_pct(ci[0])}, {_fmt_pct(ci[1])}]",
            ]
        )

    best_rows = []
    for size, (score, winners) in joined_best.items():
        shown = ["+".join(x) for x in winners[:4]]
        suffix = "" if len(winners) <= 4 else f" (+{len(winners) - 4} ties)"
        best_rows.append([str(size), _fmt_pct(score), "; ".join(shown) + suffix])

    live_tau_lines = [
        "## Live Tau Frontier Shard",
        "",
    ]
    if live_tau:
        live_tau_lines.extend(
            [
                f"Rows: {len(live_tau) * len(ALL_MODELS)} expected worker-task cells across {len(live_tau)} tasks.",
                "",
                _table(
                    ["model", "successes", "success rate", "reported cost/task", "reported total"],
                    _worker_success_rows(live_tau, ALL_MODELS),
                ),
                "",
                "Task-level solvers:",
                "",
                _table(["domain", "task", "solvers"], _solver_rows(live_tau)),
                "",
            ]
        )
    else:
        live_tau_lines.extend(["No live tau shard found.", ""])

    coding_lines = [
        "## Live Coding-Agent Shard",
        "",
    ]
    if coding:
        coding_lines.extend(
            [
                f"Rows: {len(coding) * len(ALL_MODELS)} expected worker-task cells across {len(coding)} SWE-smith tasks.",
                "",
                _table(
                    ["model", "successes", "success rate", "reported cost/task", "reported total"],
                    _worker_success_rows(coding, ALL_MODELS),
                ),
                "",
                "Task-level solvers:",
                "",
                _table(["domain", "task", "solvers"], _solver_rows(coding)),
                "",
                "Coding-primary ranking:",
                "",
                _table(["model", "solved", "reported cost/task", "reported total"], _coding_primary_rows(coding)),
                "",
                "Saved-rollout audit for commercial failures versus Kimi-Code:",
                "",
                _table(
                    ["model", "task", "reward", "status", "diff_len", "elapsed", "reported cost"],
                    _coding_attempt_audit_rows(manifest_dir),
                ),
                "",
                "Diff-length audit notes:",
                "",
                *[
                    f"- {note}"
                    for note in (
                        _coding_diff_len_notes(manifest_dir)
                        or ["No repeated nonzero diff lengths across three or more workers."]
                    )
                ],
                "",
            ]
        )
    else:
        coding_lines.extend(["No live coding shard found.", ""])

    combined_headers = ["candidate", "size", "members"] + [name for name, _ in combined_strata] + [
        "equal avg"
    ]
    strict_headers = ["subset", "equal avg"] + [name for name, _ in combined_strata]
    weighted_headers = ["candidate", "size", "members", "weighted"] + [
        name for name, _ in combined_strata
    ]
    weighted_subset_headers = ["subset", "weighted"] + [name for name, _ in combined_strata]
    weight_text = ", ".join(
        f"{name}={GENERAL_AGENTIC_WEIGHTS[name]:.2f}"
        for name, _ in combined_strata
        if name in GENERAL_AGENTIC_WEIGHTS
    )

    lines = [
        "# Ultra Pool Selection Report",
        "",
        "## Evidence Sources",
        "",
        f"- Joined 9-model direct slice: {len(joined)} tasks; domains {dict(_domain_counts(joined))}.",
        f"- Full open direct bank: {len(open_probe)} tasks; domains {dict(_domain_counts(open_probe))}.",
        f"- Open agentic bank: {len(agentic)} tau tasks; domains {dict(_domain_counts(agentic))}.",
        f"- Live frontier tau shard: {len(live_tau)} tasks; domains {dict(_domain_counts(live_tau))}.",
        f"- Live coding-agent shard: {len(coding)} tasks; domains {dict(_domain_counts(coding))}.",
        "- Frontier slice binarization: commercial/frontier task success is `r_bar >= 0.5`, matching the derived table.",
        "- Pool score is task-level coverage: a task is covered when any selected worker solves it.",
        "",
        "## OpenRouter Catalog",
        "",
        "Legacy OpenRouter prices are USD per million tokens. Current Yunwu runs do not report cost here; external cost monitoring is authoritative.",
        "",
        _table(["model", "slug", "input / MTok", "output / MTok"], _catalog_rows()),
        "",
        "## Joined 9-Model Direct Accuracy",
        "",
        _table(["model", "overall", "math", "code", "science", "general"], joined_rows),
        "",
        "## Direct Coverage On Joined Slice",
        "",
        _table(
            ["pool", "members", "coverage"],
            _coverage_rows(
                joined,
                {
                    "commercial-only": COMMERCIAL_MODELS,
                    "open-only": OPEN_MODELS,
                    "proposed-six": proposed,
                    "all-nine": ALL_MODELS,
                    "minimal-best-four": ["flash", "glm", "opus", "gemini"],
                },
            ),
        ),
        "",
        "Best subsets by direct coverage on the joined slice:",
        "",
        _table(["size", "coverage", "best subset(s)"], best_rows),
        "",
        "Leave-one-out direct coverage for the proposed six:",
        "",
        _table(["removed", "coverage without", "delta kept", "bootstrap 95% CI"], loo_rows),
        "",
        "Challenger direct swaps/additions against the proposed six:",
        "",
        _table(["comparison", "coverage", "delta vs proposed", "bootstrap 95% CI"], challenger_swap_rows),
        "",
        "## Full Open Direct Bank",
        "",
        _table(["model", "overall"], open_rows),
        "",
        f"Best open-only direct subset of size 1: {'+'.join(open_best[1][1][0])} at {_fmt_pct(open_best[1][0])}.",
        f"Best open-only direct subset of size 4: {'+'.join(open_best[4][1][0])} at {_fmt_pct(open_best[4][0])}.",
        "",
        "## Open Agentic Bank",
        "",
        _table(["model", "overall"], agentic_rows),
        "",
        f"Agentic core `{'+'.join(agentic_core)}` coverage: {_fmt_pct(agentic_core_score)}.",
        "",
        _table(["removed", "coverage without", "delta kept", "bootstrap 95% CI"], agentic_loo_rows),
        "",
        f"Best open-only agentic subset of size 1: {'+'.join(agentic_best[1][1][0])} at {_fmt_pct(agentic_best[1][0])}.",
        f"Best open-only agentic subset of size 3: {'+'.join(agentic_best[3][1][0])} at {_fmt_pct(agentic_best[3][0])}.",
        "",
        *live_tau_lines,
        *coding_lines,
        "## Quality-First Ultra Decision",
        "",
        "The product target is a general agentic model, not a coding-only model.",
        "The main pool should be frontier triad plus empirically useful open/specialist workers.",
        "The objective is to train workflows that beat each individual frontier/specialist baseline, not to select the cheapest single-axis worker set.",
        "",
        f"- Quality-first core: `{'+'.join(QUALITY_FIRST_CORE)}`.",
        f"- Optional expanded pool: `{'+'.join(QUALITY_FIRST_EXPANDED)}`.",
        "- Use the seven-worker core for the first serious GRPO pilot unless rollout budget supports all nine.",
        "- Treat the open-only coding pool as a systems-integration / low-cost ablation, not the main Ultra implementation.",
        "",
        "Role intent:",
        "",
        "- Opus: debugger, verifier, security/code reviewer, hard agentic escalation, final repair.",
        "- Gemini: science/factual specialist, long-context reasoner, knowledge-heavy aggregator, final synthesizer.",
        "- GPT: planner, math/physics specialist, algorithm designer, alternate-perspective reviewer.",
        "- Kimi-Code: primary OpenCode builder, implementation specialist, repair worker.",
        "- MiMo: tool-dialogue worker, agentic executor, cheap independent attempt, procedural repair.",
        "- GLM: open generalist, structured coding/debugging worker, secondary builder.",
        "- Flash: strong fast open direct worker for easy subtasks, first-pass answers, and low-risk branches.",
        "- MiniMax and DeepSeek-Pro: optional expanded-pool challengers until held-out workflow evidence proves core value.",
        "",
        "## Scaffold-Aware Coding Layer",
        "",
        "OpenCode is one coding harness adapter, not the full coding data distribution.",
        "Claude Code and Codex traces should be first-class trace sources, and Claude Code/Codex should be first-class scaffold backends where available.",
        "A worker identity is `model + scaffold + settings`, not only a model name.",
        "",
        "Trace source mix for the coding portion:",
        "",
        _table(["source", "share", "purpose"], TRACE_SOURCE_MIX),
        "",
        "Scaffold-aware coding worker candidates:",
        "",
        "\n".join(f"- `{worker}`" for worker in SCAFFOLD_AWARE_CORE),
        "",
        "Fair scaffold-aware baselines:",
        "",
        "\n".join(f"- {baseline}" for baseline in SCAFFOLD_BASELINES),
        "",
        "Implementation milestones:",
        "",
        "- Trace ingestion: AgentTrace schema plus OpenCode, Claude Code, and Codex adapters.",
        "- Harness parity canary: run the same toy repo tasks through OpenCode/Kimi, OpenCode/MiMo, Claude Code/Opus, and Codex/GPT.",
        "- Scaffold-aware pool tournament: test Codex builder, Claude Code debugger, OpenCode/Kimi builder, OpenCode/MiMo repair, and OpenCode/GLM alternate builder roles.",
        "- GRPO training: train over scaffold-aware worker IDs; the workflow JSON stays the same while worker_id resolution changes.",
        "",
        "## Diagnostic Role-Weighted Table",
        "",
        "The table below is diagnostic only. It helps expose tradeoffs, but it is not allowed to drop the frontier triad from the quality-first candidate pool by optimizing a tiny shard.",
        f"Current diagnostic weights: {weight_text}.",
        "",
        _table(weighted_headers, _weighted_candidate_rows(combined_strata, GENERAL_AGENTIC_WEIGHTS)),
        "",
        "Top strict-six subsets by diagnostic weighted score:",
        "",
        _table(
            weighted_subset_headers,
            _weighted_subset_rows(combined_strata, GENERAL_AGENTIC_WEIGHTS, size=6, limit=10),
        ),
        "",
        f"- If a deployment constraint forces six workers today, the diagnostic six is `{'+'.join(AGENTIC_WEIGHTED_SIX)}`.",
        "- That six-worker compression is not the scientific default for Fugu-Ultra.",
        "",
        "## Diagnostic Equal-Stratum Table",
        "",
        "This table reports each stratum separately and gives an equal-weight average for audit only.",
        "It is not the decision rule for the general-agentic pool; it is kept to expose dilution and sensitivity.",
        "Historical tau only contains open-worker measurements, so commercial workers receive no credit in that stratum.",
        "",
        _table(combined_headers, _combined_candidate_rows(combined_strata)),
        "",
        "Top strict-six subsets by equal-stratum average:",
        "",
        _table(strict_headers, _strict_subset_rows(combined_strata, size=6, limit=10)),
        "",
        "## Coding-Focused Ablation",
        "",
        "If the product target were coding-primary, the live coding shard would be decisive until contradicted by a larger coding rerun.",
        "That is not the final Ultra decision here, but it defines how the Conductor should route coding-heavy work.",
        "Reported costs below are legacy provider telemetry where available; do not treat zero/missing Yunwu cost as free inference.",
        "",
        f"- Provisional low-cost coding pool, fixed six: `{'+'.join(CODING_PRIMARY_OPEN_SIX)}`.",
        f"- Coding-positive five, no filler: `{'+'.join(CODING_PRIMARY_POSITIVE_FIVE)}`.",
        "- `kimi-code` is the anchor: 3/3 solved on the live SWE-smith/OpenCode shard.",
        "- `mimo` is the second coding worker: 2/3 solved with the lowest successful legacy reported cost per task.",
        "- `glm`, `deepseek-pro`, and `minimax` each solved 1/3 and are cheap enough to keep as challengers/coverage workers.",
        "- `flash` solved 0/3; include it only as a cheap fixed-six filler or direct-QA worker, not as a coding-positive result.",
        "- `gpt` solved 1/3 but had much higher legacy reported cost than the open workers and added no task coverage beyond Kimi/MiMo/GLM on this shard.",
        "- `opus` and `gemini` solved 0/3 while having much higher legacy reported cost than MiMo; they are not justified as coding-core workers from current evidence.",
        "- Saved rollouts show Opus/Gemini were not empty-diff or tool-calling failures: status was `ok`, errors were null, and nonzero source diffs were produced.",
        "- Caveat: n=3 is too small. The remaining uncertainty is agent behavior/prompt sensitivity, especially patch-once-and-stop behavior versus Kimi-Code's longer iteration.",
        "- Nikola has repeated diff_len values across several independent agents, so inspect actual patches before overinterpreting diff length on that task.",
        "",
        "## Current Scientific Conclusion",
        "",
        f"- Direct-only evidence was insufficient: `{'+'.join(proposed)}` matches all-nine direct coverage on the joined slice at {_fmt_pct(proposed_joined)}, but it misses coding coverage.",
        f"- Final proposed quality-first core: `{'+'.join(QUALITY_FIRST_CORE)}`.",
        f"- Optional expanded candidate universe: `{'+'.join(QUALITY_FIRST_EXPANDED)}`.",
        "- Coding implementation should be scaffold-aware: include OpenCode, Claude Code, and Codex as trace sources and harness backends.",
        "- Opus, Gemini, and GPT stay because Fugu-Ultra is quality-first and must beat those same frontier models as individual baselines.",
        "- Kimi-Code and MiMo are mandatory because they carry the coding-agent signal and also contribute live tau coverage.",
        "- GLM remains as the strongest open generalist; Flash remains as the strong fast open direct worker.",
        "- MiniMax and DeepSeek-Pro are optional expanded-pool workers; include them in the fixed-workflow tournament if rollout budget can support the larger action space.",
        "- The next scientific spend should run a performance-first role tournament over the quality-first pool, then prune by held-out workflow contribution.",
        "",
        "## Preregistered Low-Spend Paid Test",
        "",
        f"Budget cap: ${budget_usd:.2f}. With Yunwu, enforce this via the external spend monitor because provider-reported cost may be absent.",
        "",
        "Stage 1: saved-rollout and prompt-behavior audit before more spend.",
        "",
        "- Inspect saved OpenCode transcripts and actual patches for Opus/Gemini/Kimi on the three coding tasks.",
        "- Confirm whether Opus/Gemini stopped after one plausible patch while Kimi-Code persisted through longer test/repair loops.",
        "- Inspect Nikola actual patches because repeated diff_len=1711 appears across multiple agents and may be a diff-capture or task-specific quirk.",
        "- Do not route Opus/Gemini as default first-pass OpenCode builders from direct/tau performance alone.",
        "",
        "Stage 2: broaden coding-agent evidence.",
        "",
        "- Expand from 3 SWE-smith tasks to 12-20 tasks before spending on more direct QA.",
        "- Prioritize the quality-first core plus MiniMax/DeepSeek-Pro challengers if budget permits.",
        "- Score by task coverage and marginal contribution, not standalone average accuracy.",
        "",
        "Stage 3: mixed workflow-role tournament.",
        "",
        "- Test GPT specifically as planner/math/alternate-reasoning worker, not just another direct worker.",
        "- Test Opus on debugging, verification, security/review, and hard tool-use/airline-style tasks where live tau found unique coverage.",
        "- Test Gemini on science/factual/long-context synthesis and aggregator roles.",
        "- Test Kimi/MiMo/GLM fixed workflows for coding repair and synthesis.",
        "",
        "Stage 4: tau/tool-dialogue expansion.",
        "",
        "- Add hard tau airline tasks because Opus and Kimi/MiMo separated there.",
        "- Keep task selection discriminative: avoid all-solved retail tasks and all-failed dead zones.",
        "",
        "Decision rule:",
        "",
        "- Keep a worker if leave-one-out removal lowers paired held-out workflow success or moves the cost-quality frontier.",
        "- Reject an excluded challenger if no swap improves the proposed pool by at least 1 point and its paired CI is not positive.",
        "- If a worker only helps one capability family, keep it only when that family has declared product weight.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline Ultra pool-selection report")
    parser.add_argument("--manifest-dir", type=Path, default=default_manifest_dir())
    parser.add_argument("--budget", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = render_report(args.manifest_dir, budget_usd=args.budget, seed=args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
