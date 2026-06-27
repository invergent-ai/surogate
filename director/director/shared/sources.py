"""Multi-domain source loaders for the curation pipeline.

Each loader yields verifiable single-step ``Task``s tagged with ``domain`` and
``source`` (in ``Task.metadata``). The curation pipeline (``shared.curate``) probes a
large candidate pool drawn from these, keeps the worker-discriminative items, balances
across domains, and splits into train/held-out.

Loaders are streaming-friendly (``limit`` / ``shuffle`` / ``seed``) so the candidate
pool can scale to 10^5 without holding everything in memory at once.
"""

from __future__ import annotations

import hashlib
import re
import string
from collections.abc import Iterator

from .tasks import Dataset, Task

_LETTERS = string.ascii_uppercase  # A, B, C, ...

_MATH_SYS = "Solve the problem. Put your final answer in \\boxed{}."
_MC_SYS = "Answer the question. Put the final answer letter in \\boxed{}."
_CODE_SYS = "Complete the function in Python. Return only the full function in a code block."


def _stream(path: str, split: str, name: str | None = None):
    from datasets import load_dataset

    ds = load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
    return ds


def _take(rows, limit: int | None, shuffle: bool, seed: int) -> list:
    rows = list(rows)
    if shuffle:
        import random

        random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def _stream_shuffled(path: str, split: str, name: str | None = None,
                     shuffle: bool = False, seed: int = 0, buffer: int = 10000):
    """Streaming iterator for LARGE datasets (taco/code_contests/numina) so we never download the
    whole multi-GB split — we pull rows lazily and stop at the caller's limit. Optional buffered
    shuffle for variety across the stream."""
    from datasets import load_dataset

    ds = (load_dataset(path, name, split=split, streaming=True) if name
          else load_dataset(path, split=split, streaming=True))
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=buffer)
    return ds


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def load_math500(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    for i, r in enumerate(_take(_stream("HuggingFaceH4/MATH-500", "test"), limit, shuffle, seed)):
        yield Task(
            task_id=f"math500-{r.get('unique_id', i)}",
            prompt=r["problem"],
            solution=r["answer"],
            grader="math_equal",
            system=_MATH_SYS,
            metadata={"domain": "math", "source": "math500", "level": r.get("level")},
        )


def load_numina_math(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    """NuminaMath-1.5 (~1M competition problems, difficulty range) — the large math TRAIN source.
    Streamed (the set is huge); we keep only rows with a usable final answer (skip proofs) so
    math_equal can grade them. Distinct from the held-out MATH-500 / AIME eval sets."""
    from datasets import load_dataset

    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=seed, buffer_size=10000)
    n = 0
    for r in ds:
        if limit and n >= limit:
            break
        ans = (r.get("answer") or "").strip()
        if not ans or ans.lower() in {"proof", "notfound", "none"}:
            continue
        if str(r.get("problem_is_valid", "Yes")).lower() in {"no", "false"}:
            continue
        yield Task(
            task_id=f"numina-{n}",
            prompt=r["problem"],
            solution=ans,
            grader="math_equal",
            system=_MATH_SYS,
            metadata={"domain": "math", "source": "numina_math", "type": r.get("problem_type")},
        )
        n += 1


def load_aime(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    for r in _take(_stream("Maxwell-Jia/AIME_2024", "train"), limit, shuffle, seed):
        yield Task(
            task_id=f"aime2024-{r['ID']}",
            prompt=r["Problem"],
            solution=str(r["Answer"]),
            grader="math_equal",
            system="Solve the problem. Put the final integer answer in \\boxed{}.",
            metadata={"domain": "math", "source": "aime2024"},
        )


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


def load_humaneval(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    for r in _take(_stream("openai/openai_humaneval", "test"), limit, shuffle, seed):
        yield Task(
            task_id=r["task_id"],
            prompt=r["prompt"],
            solution={"test": r["test"], "entry_point": r["entry_point"]},
            grader="code_exec",
            system=_CODE_SYS,
            metadata={"domain": "code", "source": "humaneval"},
        )


def load_code_contests(limit=None, shuffle=False, seed=0, min_difficulty=6, max_difficulty=10, max_tests=8) -> Iterator[Task]:
    """Competitive-programming problems (stdin->stdout), banded to the RATED MID range
    [min_difficulty, max_difficulty]. difficulty 0 is UNKNOWN/unrated and 11+ is too hard for our
    open pool (the full set measured ~48% all-fail) — the mid band is where some workers solve and
    some don't, which is the routing signal.
    """
    n = 0
    for r in _stream_shuffled("deepmind/code_contests", "train", shuffle=shuffle, seed=seed):
        if limit and n >= limit:
            break
        d = r.get("difficulty") or 0
        if d < min_difficulty or d > max_difficulty:
            continue
        pt = r["public_tests"]
        tests = [
            {"input": i, "output": o}
            for i, o in zip(pt["input"][:max_tests], pt["output"][:max_tests])
        ]
        if not tests:
            continue
        yield Task(
            task_id=f"codecontests-{r['name'][:40]}",
            prompt=r["description"],
            solution={"tests": tests, "timeout": 10},
            grader="code_exec_stdio",
            system="Write a complete Python program that reads from stdin and writes to stdout. Return only the program in a code block.",
            metadata={"domain": "code", "source": "code_contests", "difficulty": r.get("difficulty")},
        )
        n += 1


def load_mbpp(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    for r in _take(_stream("google-research-datasets/mbpp", "test", name="full"), limit, shuffle, seed):
        tests = "\n".join([r.get("test_setup_code", ""), *r["test_list"]]).strip()
        prompt = (
            f"{r['text']}\n\nYour function must pass these tests:\n"
            + "\n".join(r["test_list"])
        )
        yield Task(
            task_id=f"mbpp-{r['task_id']}",
            prompt=prompt,
            # entry_point="" => code_exec runs candidate + raw assert tests directly.
            solution={"test": tests, "entry_point": ""},
            grader="code_exec",
            system=_CODE_SYS,
            metadata={"domain": "code", "source": "mbpp"},
        )


# ---------------------------------------------------------------------------
# science MC
# ---------------------------------------------------------------------------


def load_gpqa(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    from .verifiers import extract_boxed

    for i, r in enumerate(_take(_stream("hendrydong/gpqa_diamond_mc", "test"), limit, shuffle, seed)):
        letter = (extract_boxed(r["solution"]) or str(r["solution"])).strip()[:1].upper()
        yield Task(
            task_id=f"gpqa-{i}",
            prompt=r["problem"],
            solution=letter,
            grader="mc_letter",
            system=_MC_SYS,
            metadata={"domain": "science", "source": "gpqa_diamond", "subdomain": r.get("domain")},
        )


# ---------------------------------------------------------------------------
# general MC
# ---------------------------------------------------------------------------


# MMLU-Pro category -> Director domain. STEM sciences feed the (thin) science pool,
# the math category feeds math, everything else feeds general.
_MMLU_DOMAIN = {
    "physics": "science", "chemistry": "science", "biology": "science", "health": "science",
    "math": "math",
}


def _mmlu_domain(category: str) -> str:
    return _MMLU_DOMAIN.get(category, "general")


def load_mmlu_pro(limit=None, shuffle=False, seed=0, domains: set[str] | None = None) -> Iterator[Task]:
    """MMLU-Pro, with each row routed to a domain by its category. ``domains`` filters
    to a subset (e.g. {"science"}) so the curation pool can target thin domains."""
    n = 0
    for r in _take(_stream("TIGER-Lab/MMLU-Pro", "test"), None, shuffle, seed):
        if limit and n >= limit:
            break
        domain = _mmlu_domain(r["category"])
        if domains and domain not in domains:
            continue
        opts = r["options"]
        lines = [f"{_LETTERS[j]}) {opt}" for j, opt in enumerate(opts)]
        prompt = r["question"] + "\n\nOptions:\n" + "\n".join(lines)
        gold = str(r["answer"]).strip()[:1].upper()
        yield Task(
            task_id=f"mmlupro-{r['question_id']}",
            prompt=prompt,
            solution=gold,
            grader="mc_letter",
            system=_MC_SYS,
            metadata={"domain": domain, "source": "mmlu_pro", "subdomain": r.get("category")},
        )
        n += 1


# ---------------------------------------------------------------------------
# SuperGPQA (graduate MC across 285 disciplines) + Omni-MATH + TACO (harder code)
# ---------------------------------------------------------------------------

_SUPERGPQA_SCI = {"Science", "Medicine", "Engineering", "Agronomy"}


def load_supergpqa(limit=None, shuffle=False, seed=0, domains: set[str] | None = None) -> Iterator[Task]:
    """SuperGPQA: ~26k graduate MC across 285 disciplines (A-J). STEM disciplines map
    to science, the rest to general; ``domains`` filters to a subset."""
    n = 0
    for r in _take(_stream("m-a-p/SuperGPQA", "train"), None, shuffle, seed):
        if limit and n >= limit:
            break
        domain = "science" if r["discipline"] in _SUPERGPQA_SCI else "general"
        if domains and domain not in domains:
            continue
        opts = r["options"]
        lines = [f"{_LETTERS[j]}) {opt}" for j, opt in enumerate(opts)]
        prompt = r["question"] + "\n\nOptions:\n" + "\n".join(lines)
        yield Task(
            task_id=f"supergpqa-{r['uuid']}",
            prompt=prompt,
            solution=str(r["answer_letter"]).strip()[:1].upper(),
            grader="mc_letter",
            system=_MC_SYS,
            metadata={"domain": domain, "source": "supergpqa", "subdomain": r.get("field")},
        )
        n += 1


def load_omni_math(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    """Omni-MATH: ~4.4k olympiad problems with verifiable final answers."""
    for i, r in enumerate(_take(_stream("KbsdJames/Omni-MATH", "test"), limit, shuffle, seed)):
        yield Task(
            task_id=f"omnimath-{i}",
            prompt=r["problem"],
            solution=r["answer"],
            grader="math_equal",
            system=_MATH_SYS,
            metadata={"domain": "math", "source": "omni_math", "difficulty": r.get("difficulty")},
        )


def load_taco(limit=None, shuffle=False, seed=0, difficulties=("EASY", "MEDIUM", "MEDIUM_HARD"), max_tests=8) -> Iterator[Task]:
    """TACO-verified competitive programming (stdin->stdout).

    Restricted to the MID difficulty band (EASY..MEDIUM_HARD): for our open-weight pool the
    HARD/VERY_HARD slices are mostly all-fail ("dead", no routing signal) — disagreement lives
    where some workers solve it and some don't. stdin/stdout only (fn_name problems are skipped).
    """
    import json

    diffs = set(difficulties)
    n = 0
    for r in _stream_shuffled("likaixin/TACO-verified", "train", shuffle=shuffle, seed=seed):
        if limit and n >= limit:
            break
        if r.get("difficulty") not in diffs:
            continue
        try:
            io = json.loads(r["input_output"]) if isinstance(r["input_output"], str) else r["input_output"]
        except (json.JSONDecodeError, TypeError):
            continue
        if not io or "fn_name" in io or not io.get("inputs"):
            continue
        tests = [
            {"input": str(i), "output": str(o)}
            for i, o in zip(io["inputs"][:max_tests], io["outputs"][:max_tests])
        ]
        if not tests:
            continue
        yield Task(
            task_id=f"taco-{r['id']}",
            prompt=r["question"],
            solution={"tests": tests, "timeout": 10},
            grader="code_exec_stdio",
            system="Write a complete Python program that reads from stdin and writes to stdout. Return only the program in a code block.",
            metadata={"domain": "code", "source": "taco", "difficulty": r.get("difficulty")},
        )
        n += 1


# ---------------------------------------------------------------------------
# abstract reasoning (ARC-AGI-2)
# ---------------------------------------------------------------------------


def _grid_str(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def load_arc_agi2(limit=None, shuffle=False, seed=0) -> Iterator[Task]:
    """ARC-AGI-2 abstract-reasoning tasks. Each row has ``fewshots`` (demonstration
    input/output grids) and ``question`` (the test pair, whose output is the gold)."""
    for i, r in enumerate(_take(_stream("arc-agi-community/arc-agi-2", "test"), limit, shuffle, seed)):
        demos = r["fewshots"]
        test = r["question"][0]
        parts = ["Infer the transformation from the examples, then produce the output grid."]
        for k, d in enumerate(demos):
            parts.append(f"\nExample {k + 1} input:\n{_grid_str(d['input'])}\nExample {k + 1} output:\n{_grid_str(d['output'])}")
        parts.append(f"\nTest input:\n{_grid_str(test['input'])}\nTest output:")
        yield Task(
            task_id=f"arcagi2-{i}",
            prompt="\n".join(parts),
            solution=test["output"],
            grader="grid_exact",
            system="You solve ARC puzzles. Output ONLY the test output grid as rows of space-separated integers.",
            metadata={"domain": "reasoning", "source": "arc_agi2"},
        )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

import functools

# source name -> (loader, primary-domain). The primary-domain is for organization /
# default selection; the authoritative per-task domain is in Task.metadata (so the
# category-routed MMLU-Pro sources can span their mapped domains).
SOURCES = {
    "math500": (load_math500, "math"),
    "numina_math": (load_numina_math, "math"),
    "aime2024": (load_aime, "math"),
    "humaneval": (load_humaneval, "code"),
    "mbpp": (load_mbpp, "code"),
    "code_contests": (load_code_contests, "code"),
    "gpqa": (load_gpqa, "science"),
    "mmlu_sci": (functools.partial(load_mmlu_pro, domains={"science"}), "science"),
    "mmlu_math": (functools.partial(load_mmlu_pro, domains={"math"}), "math"),
    "mmlu_gen": (functools.partial(load_mmlu_pro, domains={"general"}), "general"),
    "arc_agi2": (load_arc_agi2, "reasoning"),
    "omni_math": (load_omni_math, "math"),
    "taco": (load_taco, "code"),
    "supergpqa_sci": (functools.partial(load_supergpqa, domains={"science"}), "science"),
    "supergpqa_gen": (functools.partial(load_supergpqa, domains={"general"}), "general"),
}

# source key -> underlying HuggingFace dataset id (recorded on every task for provenance)
SOURCE_DATASET = {
    "math500": "HuggingFaceH4/MATH-500",
    "numina_math": "AI-MO/NuminaMath-1.5",
    "aime2024": "Maxwell-Jia/AIME_2024",
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
    "code_contests": "deepmind/code_contests",
    "gpqa": "hendrydong/gpqa_diamond_mc",
    "mmlu_sci": "TIGER-Lab/MMLU-Pro",
    "mmlu_math": "TIGER-Lab/MMLU-Pro",
    "mmlu_gen": "TIGER-Lab/MMLU-Pro",
    "arc_agi2": "arc-agi-community/arc-agi-2",
    "omni_math": "KbsdJames/Omni-MATH",
    "taco": "likaixin/TACO-verified",
    "supergpqa_sci": "m-a-p/SuperGPQA",
    "supergpqa_gen": "m-a-p/SuperGPQA",
}


# Held-out evaluation benchmarks: excluded from the training/curation pool so their numbers stay
# clean generalization measurements (matching Fugu, which evaluates on benchmarks UNSEEN during
# training). These are exactly the Fugu-reported eval sets we have loaders for:
#   gpqa = GPQA-Diamond, humaneval = HumanEval, math500 = MATH-500, aime2024 = AIME.
# Training instead comes from the large NON-benchmark sources (numina_math, code_contests, taco,
# omni_math, mmlu, supergpqa). Agentic eval benchmarks (SWE-bench Verified, Terminal-Bench, tau³
# banking) are held out at the loop level, not here. The loaders remain available for eval.
EVAL_ONLY = {"gpqa", "humaneval", "math500", "aime2024"}


def normalize_prompt(p: str) -> str:
    """Canonical form for contamination matching: lowercase, keep only [a-z0-9], collapse whitespace.
    Strips formatting/markup differences so a mirrored or renamed eval row still matches the original."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (p or "").lower()).split())


def prompt_hash(p: str) -> str:
    return hashlib.sha256(normalize_prompt(p).encode()).hexdigest()


def eval_prompt_hashes(limit_per: int | None = None) -> set[str]:
    """Normalized prompt hashes of every held-out EVAL_ONLY item. A training candidate whose prompt
    hashes into this set is a mirrored/renamed eval row and must be dropped (the Non-Negotiable Rule:
    never train on anything we evaluate on, even under a different dataset name)."""
    hashes: set[str] = set()
    for src in EVAL_ONLY:
        fn = SOURCES.get(src, (None,))[0]
        if fn is None:
            continue
        try:
            for t in fn(limit=limit_per, shuffle=False, seed=0):
                hashes.add(prompt_hash(t.prompt))
        except Exception as e:  # a broken eval loader must not silently disable the denylist
            print(f"[eval-denylist] WARNING: could not load {src} for hashing: {type(e).__name__}: {e}")
    return hashes


def train_sources() -> list[str]:
    """Sources eligible for the training/curation pool (everything except EVAL_ONLY)."""
    return [s for s in SOURCES if s not in EVAL_ONLY]


# domain -> source names (full catalog, incl. eval-only)
DOMAINS: dict[str, list[str]] = {}
for _src, (_fn, _dom) in SOURCES.items():
    DOMAINS.setdefault(_dom, []).append(_src)


def build_candidates(
    sources: list[str] | None = None,
    *,
    per_source_limit: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Dataset:
    """Assemble a raw candidate pool. Default = all training sources (EVAL_ONLY held out)."""
    names = sources or train_sources()
    tasks: list[Task] = []
    for name in names:
        if name not in SOURCES:
            raise KeyError(f"unknown source {name!r}; have {sorted(SOURCES)}")
        fn, _domain = SOURCES[name]
        dataset_id = SOURCE_DATASET.get(name)
        for t in fn(limit=per_source_limit, shuffle=shuffle, seed=seed):
            t.metadata.setdefault("dataset", dataset_id)
            tasks.append(t)
    return Dataset(tasks, name="candidates")
