"""OOD few-shot assembly from mined high-advantage conductor plans.

The stage2 mine holds parseable positive-advantage completions (task, parsed
steps, GRPO advantage). This module turns them into a few-shot block for the
conductor's `guidelines` slot, in the TYPED contract the 27B speaks.

Selection is strictly OUTCOME- and STRUCTURE-based (hard rule: no text,
keyword, or language classification anywhere):
  * rank by measured advantage (the outcome signal), and
  * diversify by topology SHAPE (step count, independent leaves, aggregation
    arity) so the block teaches a repertoire, not one pattern.

Rendering is model-agnostic by construction: examples carry neutral
profile placeholders (profile_a, profile_b, ...) — the runtime binding
supplies real capability refs, and no worker/model name ever appears.

Adoption is a MEASURED decision, not a default: the prompt A/B stage-2
verdict decides whether few-shot content enters the r5 serving prompt at
all (arm C vs B). This module only makes the mined variant buildable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_EXAMPLES = 4
DEFAULT_MAX_CHARS = 4000


@dataclass(frozen=True)
class Exemplar:
    task: str
    steps: tuple[dict, ...]
    advantage: float

    @property
    def shape(self) -> tuple[int, int, int]:
        """(steps, independent leaves, max aggregation arity) — the
        structural signature used for diversity."""
        leaves = sum(1 for s in self.steps if not s.get("access"))
        max_arity = max((len(s.get("access") or ()) for s in self.steps),
                        default=0)
        return (len(self.steps), leaves, max_arity)


def load_candidates(path: Path) -> list[Exemplar]:
    out = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        steps = row.get("steps") or []
        if not steps:
            continue
        out.append(Exemplar(
            task=row.get("task", ""),
            steps=tuple(steps),
            advantage=float(row.get("advantage", 0.0)),
        ))
    return out


def select_exemplars(
    candidates: list[Exemplar],
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> list[Exemplar]:
    """Highest-advantage exemplar per distinct topology shape, best first.

    One exemplar per shape: a block of near-identical plans teaches less
    than one plan each of {2-step verify, 3-leaf tree, deep pipeline, ...}.
    Deterministic: ties break by task text so reruns freeze byte-identically.
    """
    ranked = sorted(candidates,
                    key=lambda e: (-e.advantage, e.task))
    chosen: list[Exemplar] = []
    seen_shapes: set[tuple[int, int, int]] = set()
    for exemplar in ranked:
        if exemplar.shape in seen_shapes:
            continue
        seen_shapes.add(exemplar.shape)
        chosen.append(exemplar)
        if len(chosen) >= max_examples:
            break
    return chosen


_PROFILE_NAMES = "abcdefghijklmnopqrstuvwxyz"


def render_typed_example(exemplar: Exemplar) -> str:
    """One exemplar as a typed-contract JSON action, model-anonymous."""
    worker_ids = []
    for step in exemplar.steps:
        wid = step.get("worker_id", 0)
        if wid not in worker_ids:
            worker_ids.append(wid)
    profile_of = {wid: f"profile_{_PROFILE_NAMES[i % len(_PROFILE_NAMES)]}"
                  for i, wid in enumerate(worker_ids)}
    steps = []
    for step in exemplar.steps:
        access = step.get("access") or []
        positions = (list(range(len(exemplar.steps) - 1))
                     if access == ["all"] or access == ("all",)
                     else [a for a in access if isinstance(a, int)])
        steps.append({
            "subtask": step.get("subtask", ""),
            "profile_ref": profile_of[step.get("worker_id", 0)],
            "access_positions": positions,
        })
    return json.dumps({"action": "replan",
                       "reason": "",
                       "steps": steps})


def render_fewshot_block(
    exemplars: list[Exemplar],
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """The guidelines-slot appendix: bounded, deterministic, contract-pure.

    Examples that would push the block past `max_chars` are dropped from the
    END (the lowest-advantage shapes), never truncated mid-JSON.
    """
    parts: list[str] = []
    total = 0
    for exemplar in exemplars:
        rendered = f"\nExample plan (measured high-advantage):\n" \
                   f"{render_typed_example(exemplar)}"
        if total + len(rendered) > max_chars:
            break
        parts.append(rendered)
        total += len(rendered)
    return "".join(parts)


def assemble(
    candidates_path: Path,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    return render_fewshot_block(
        select_exemplars(load_candidates(candidates_path), max_examples),
        max_chars)
